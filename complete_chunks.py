# Sample usage:
#   python complete_chunks.py \
#       data=vfront_houses \
#       checkpoint=<path/to/gpt.ckpt> \
#       vqvae_checkpoint=<path/to/vqvae.ckpt> \
#       num_samples=100000 \
#       completion.prompt_mode=spatial_quarter_x \
#       output_dir=<output_dir>
#
#   To restrict to one or more scenes (note: Hydra requires + prefix for keys not in the config struct):
#       +scene_id=67438c51-6887-4963-8004-f7c0e620e21b
#       +scene_id=[67438c51-6887-4963-8004-f7c0e620e21b,24cbb52c-4daf-417f-a324-0e9a055272a9]

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import imageio.v3 as iio
import lightning
import torch
from data.vfront_dataset import VFrontPreprocessedDataModule
from model.gaussian_gpt import GaussianGPT
from model.gaussian_vqvae import GaussianVQVAE
from omegaconf import DictConfig, OmegaConf
from utils.completion_resample import (
    DEFAULT_MAX_RETRIES,
    complete_with_empty_column_retry,
)
from utils.pos_tokens import pos_tokens_to_centered_coords
from utils.render import (
    GaussianScene,
    get_view_matrices_looking_at_origin,
    render,
    render_and_save_trajectory_strip,
)

# pylint: disable=E1120,W0212


def _path_matches(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return False
    left_norm = str(Path(left).expanduser().resolve(strict=False))
    right_norm = str(Path(right).expanduser().resolve(strict=False))
    return left_norm == right_norm


def _resolve_completion_data_cfg(
    data_cfg: Dict[str, Any],
    *,
    dataset: str,
    vqvae_checkpoint: Optional[str],
) -> Dict[str, Any]:
    if data_cfg.get("dataset_name") == dataset and data_cfg.get("vqvae_path"):
        return data_cfg

    conf_data_dir = Path(__file__).resolve().parent / "conf" / "data"
    if conf_data_dir.exists():
        for config_path in sorted(conf_data_dir.glob("tokenized_*.yaml")):
            candidate = OmegaConf.to_container(
                OmegaConf.load(config_path),
                resolve=True,
            )
            if not isinstance(candidate, dict):
                continue
            if candidate.get("dataset_name") != dataset:
                continue
            if _path_matches(candidate.get("vqvae_path"), vqvae_checkpoint):
                print(
                    f"Completion data config auto-resolved to {config_path.name}.",
                    flush=True,
                )
                return candidate

    raise ValueError(
        "Completion requires tokenized data. Either pass a tokenized data config "
        "(e.g. `data=tokenized_vfront`) or ensure one tokenized config in "
        "`conf/data` has matching `dataset_name` and `vqvae_path`."
    )


def _center_scene_reference(
    scene: GaussianScene, reference: GaussianScene
) -> GaussianScene:
    coords = reference.means
    if coords.numel() == 0:
        return scene
    min_vals, max_vals = torch.aminmax(coords, dim=0)
    center = 0.5 * (min_vals + max_vals)
    return GaussianScene(
        means=scene.means - center,
        sh0=scene.sh0,
        sh=scene.sh,
        opacities=scene.opacities,
        scales=scene.scales,
        quats=scene.quats,
    )


def _decode_tokens_to_scene(
    gpt: GaussianGPT,
    tokens: torch.Tensor,
    *,
    base_side_length: int,
    pad_id: int,
) -> GaussianScene:
    if tokens.numel() == 0:
        return GaussianScene.from_dict(gpt._empty_scene_dict(tokens.device))

    if gpt.dense_chunks:
        decoded = gpt._decode_dense_tokens(tokens)
        return GaussianScene.from_dict(decoded)

    tokens_per_latent = gpt.tokens_per_latent
    usable_len = (tokens.numel() // tokens_per_latent) * tokens_per_latent
    if usable_len == 0:
        return GaussianScene.from_dict(gpt._empty_scene_dict(tokens.device))
    if usable_len != tokens.numel():
        tokens = tokens[:usable_len]

    token_rows = tokens.view(-1, tokens_per_latent)
    num_pos_tokens = gpt.num_position_tokens
    position_vocab_size = gpt.position_vocab_size
    feature_vocab_size = gpt.feature_vocab_size
    feature_offset = gpt.feature_token_offset
    pos_tokens = token_rows[:, :num_pos_tokens]
    feature_ids = token_rows[:, num_pos_tokens:]
    all_pad = (feature_ids == pad_id).all(dim=1)
    pos_invalid = (pos_tokens < 0) | (pos_tokens >= position_vocab_size)
    feature_out_of_range = (feature_ids < feature_offset) | (
        feature_ids >= feature_offset + feature_vocab_size
    )
    feature_invalid = (~all_pad).unsqueeze(1) & feature_out_of_range
    invalid = (~all_pad) & (pos_invalid.any(dim=1) | feature_invalid.any(dim=1))
    if invalid.any():
        print(
            "WARNING: Sparse tokens contain invalid ids; replacing with safe defaults.",
            flush=True,
        )
        token_rows = token_rows.clone()
        pos_tokens = token_rows[:, :num_pos_tokens]
        feature_ids = token_rows[:, num_pos_tokens:]
        pos_tokens = pos_tokens.masked_fill(pos_invalid, 0)
        feature_ids = feature_ids.masked_fill(feature_invalid, feature_offset)
        token_rows[:, :num_pos_tokens] = pos_tokens
        token_rows[:, num_pos_tokens:] = feature_ids
        pos_tokens = token_rows[:, :num_pos_tokens]
        feature_ids = token_rows[:, num_pos_tokens:]

    valid_mask = ~all_pad
    if not valid_mask.any():
        return GaussianScene.from_dict(gpt._empty_scene_dict(tokens.device))

    coords = pos_tokens_to_centered_coords(
        pos_tokens[valid_mask],
        num_pos_tokens,
        base_side_length,
    ).to(dtype=torch.long)
    raw_feature_ids = (feature_ids[valid_mask] - feature_offset).to(dtype=torch.long)
    decoded = gpt.vqvae.decode(coords, raw_feature_ids)
    return GaussianScene.from_dict(decoded)


def _build_completion_dataset(
    data_cfg: Dict[str, Any],
    gpt: GaussianGPT,
    *,
    split: str,
) -> torch.utils.data.Dataset:
    dataset_name = str(data_cfg.get("dataset_name", "")).lower()
    if dataset_name not in {"vfront", "ase", "photoshape", "spp", "spp_v2"}:
        raise ValueError(
            "Completion currently supports 'vfront', 'ase', 'photoshape', 'spp', and 'spp_v2', "
            f"got '{dataset_name}'."
        )
    if data_cfg.get("sort_latents") is not None:
        raise ValueError(
            "`data.sort_latents` is deprecated outside tokenization; use `model.chunk_order`."
        )
    chunk_shape = getattr(gpt, "chunk_shape", None)
    if chunk_shape is None:
        raise ValueError(
            "Completion requires chunked GPT checkpoints; `model.chunk_shape` is missing."
        )
    if isinstance(chunk_shape, int):
        chunk_shape = [chunk_shape] * 3
    if len(chunk_shape) != 3 or any(int(v) <= 0 for v in chunk_shape):
        raise ValueError(
            "Completion requires `model.chunk_shape` to be 3 positive integers."
        )

    preprocessed_kwargs = dict(
        data_path=data_cfg.get("data_path"),
        train_list_path=data_cfg.get("train_split"),
        val_list_path=data_cfg.get("val_split"),
        dataloader_kwargs={"batch_size": 1},
        overfit_scenes=0,
        overfit_epoch_size=1000,
        verbose=False,
        background_color=data_cfg.get("background_color", "white"),
        num_position_tokens=gpt.num_position_tokens,
        position_vocab_size=getattr(gpt, "position_vocab_size", None),
        codebook_size=gpt.vqvae.autoencoder.vq.codebook_size,
        chunk_shape=chunk_shape,
        dense_chunks=getattr(gpt, "dense_chunks", False),
        chunk_order=getattr(gpt, "chunk_order", "xyz"),
        min_chunk_occupancy=data_cfg.get("min_chunk_occupancy", 0.0),
        max_chunk_attempts=data_cfg.get("max_chunk_attempts", 1),
        chunk_origin=data_cfg.get("chunk_origin"),
        load_augmented_tokens=bool(data_cfg.get("load_augmented_tokens", False)),
    )
    preprocessed_subpath = data_cfg.get("preprocessed_subpath")
    if preprocessed_subpath is not None:
        preprocessed_kwargs["preprocessed_subpath"] = preprocessed_subpath

    data_module = VFrontPreprocessedDataModule(**preprocessed_kwargs)
    data_module.setup()
    dataset = data_module.train_dataset if split == "train" else data_module.val_dataset
    if dataset is None:
        raise RuntimeError(f"Failed to build {split} dataset for completion.")
    return dataset


def _shard_range(total: int, shard_id: int, num_shards: int) -> Tuple[int, int, int]:
    base = total // num_shards
    extra = total % num_shards
    local_n = base + int(shard_id < extra)
    start = shard_id * base + min(shard_id, extra)
    end = start + local_n
    return local_n, start, end


def _resolve_sharding(cfg: DictConfig) -> Tuple[int, int]:
    shard_id_cfg = cfg.get("shard_id")
    num_shards_cfg = cfg.get("num_shards")

    if shard_id_cfg is None:
        env_shard_id = os.getenv("GAUSS_SHARD_ID")
        if env_shard_id is not None:
            shard_id_cfg = int(env_shard_id)
    if num_shards_cfg is None:
        env_num_shards = os.getenv("GAUSS_NUM_SHARDS")
        if env_num_shards is not None:
            num_shards_cfg = int(env_num_shards)

    shard_id = 0 if shard_id_cfg is None else int(shard_id_cfg)
    num_shards = 1 if num_shards_cfg is None else int(num_shards_cfg)

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1.")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards - 1}], got {shard_id}.")
    return shard_id, num_shards


def _completion_seed(
    base_seed: int,
    shard_id: int,
    global_scene_idx: int,
    completion_idx: int,
) -> int:
    mod = 2**31 - 1
    seed = (
        int(base_seed)
        + int(shard_id) * 1_000_003
        + int(global_scene_idx) * 97_003
        + int(completion_idx) * 8_191
    ) % mod
    return int(seed)


def _scene_pt_payload(scene: GaussianScene) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in scene.to_dict().items()
    }


def _render_and_save_xview_strip(
    scenes: List[GaussianScene],
    path: Path,
    *,
    shared_camera_pos_xyz: Optional[Tuple[float, float, float]] = None,
    per_scene_camera_positions_xyz: Optional[List[Tuple[float, float, float]]] = None,
    background_color: str,
) -> None:
    if not scenes:
        raise ValueError("Expected at least one scene for x-view rendering.")
    if shared_camera_pos_xyz is None and per_scene_camera_positions_xyz is None:
        raise ValueError(
            "Either shared_camera_pos_xyz or per_scene_camera_positions_xyz must be provided."
        )
    if shared_camera_pos_xyz is not None and per_scene_camera_positions_xyz is not None:
        raise ValueError(
            "shared_camera_pos_xyz and per_scene_camera_positions_xyz are mutually exclusive."
        )
    if per_scene_camera_positions_xyz is not None and len(
        per_scene_camera_positions_xyz
    ) != len(scenes):
        raise ValueError(
            "per_scene_camera_positions_xyz length must match number of scenes in strip."
        )

    device = scenes[0].means.device
    render_scenes: List[GaussianScene] = []
    for scene in scenes:
        if scene.means.device != device:
            scene = scene.to(device)
        render_scenes.append(scene)

    strips = []
    for scene_idx, scene in enumerate(render_scenes):
        if per_scene_camera_positions_xyz is not None:
            cam_pos = per_scene_camera_positions_xyz[scene_idx]
        else:
            assert shared_camera_pos_xyz is not None
            cam_pos = shared_camera_pos_xyz
        view_mats = get_view_matrices_looking_at_origin(
            torch.tensor([list(cam_pos)], device=device, dtype=torch.float32),
            world_up=torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32),
            device=device,
        )
        frame, _ = render(
            scene,
            view_mats,
            background_color=background_color,
        )
        strips.append(frame.squeeze(0))

    strip = torch.cat(strips, dim=2)
    image = strip.permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
    iio.imwrite(path, image)


def _render_and_save_fixed_view(
    scene: GaussianScene,
    path: Path,
    *,
    camera_pos_xyz: Tuple[float, float, float],
    background_color: str,
) -> None:
    device = scene.means.device
    view_mats = get_view_matrices_looking_at_origin(
        torch.tensor([list(camera_pos_xyz)], device=device, dtype=torch.float32),
        world_up=torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32),
        device=device,
    )
    frame, _ = render(
        scene,
        view_mats,
        background_color=background_color,
    )
    image = (
        frame.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
    )
    iio.imwrite(path, image)


def _split_prompt_spatial_x_fraction(
    gpt: GaussianGPT,
    tokens: torch.Tensor,
    *,
    fraction: float,
    pad_id: int,
    base_side_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    if fraction <= 0.0 or fraction >= 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}.")

    full_tokens = tokens
    full_len = int(full_tokens.numel())
    if full_len == 0:
        empty = torch.empty(0, device=full_tokens.device, dtype=torch.long)
        return full_tokens, empty, empty, 0, 0

    if gpt.dense_chunks:
        if gpt.chunk_shape is None or len(gpt.chunk_shape) != 3:
            raise ValueError("Dense checkpoints require a 3D chunk_shape.")
        chunk_x, chunk_y, chunk_z = [int(v) for v in gpt.chunk_shape]
        num_features = int(gpt.vqvae.autoencoder.vq.num_tokens)
        prefix_x = int(chunk_x * fraction)
        prefix_latents = int(prefix_x * chunk_y * chunk_z)
        prefix_len = max(0, min(prefix_latents * num_features, full_len))
        prefix_tokens = full_tokens[:prefix_len]
        partial_tokens = full_tokens.clone()
        partial_tokens[prefix_len:] = pad_id
        remaining_len = full_len - prefix_len
        return full_tokens, prefix_tokens, partial_tokens, prefix_len, remaining_len

    tokens_per_latent = int(gpt.tokens_per_latent)
    usable_len = (full_len // tokens_per_latent) * tokens_per_latent
    if usable_len <= 0:
        empty = torch.empty(0, device=full_tokens.device, dtype=torch.long)
        return empty, empty, empty, 0, 0
    if usable_len != full_len:
        full_tokens = full_tokens[:usable_len]
        full_len = usable_len

    token_rows = full_tokens.view(-1, tokens_per_latent)
    num_rows = int(token_rows.shape[0])
    num_pos_tokens = int(gpt.num_position_tokens)
    pos_tokens = token_rows[:, :num_pos_tokens]
    feature_ids = token_rows[:, num_pos_tokens:]
    non_pad_rows = ~(feature_ids == pad_id).all(dim=1)
    valid_pos_rows = (
        (pos_tokens >= 0) & (pos_tokens < int(gpt.position_vocab_size))
    ).all(dim=1)
    rows_for_split = non_pad_rows & valid_pos_rows

    prefix_rows = 0
    if bool(rows_for_split.any().item()):
        coords = pos_tokens_to_centered_coords(
            pos_tokens[rows_for_split],
            num_pos_tokens,
            base_side_length,
        )
        x_vals = coords[:, 0]
        split_x = x_vals.min() + float(fraction) * (x_vals.max() - x_vals.min())
        keep_rows = torch.zeros(num_rows, device=full_tokens.device, dtype=torch.bool)
        keep_rows[rows_for_split] = x_vals <= split_x
        for keep in keep_rows.tolist():
            if keep:
                prefix_rows += 1
            else:
                break

    if prefix_rows == 0 and num_rows > 0:
        prefix_rows = int(num_rows * fraction)

    prefix_len = int(prefix_rows * tokens_per_latent)
    prefix_tokens = full_tokens[:prefix_len]
    partial_tokens = prefix_tokens
    remaining_len = full_len - prefix_len
    return full_tokens, prefix_tokens, partial_tokens, prefix_len, remaining_len


@torch.no_grad()
def run_completion_inference(
    *,
    cfg: DictConfig,
    data_cfg: Dict[str, Any],
    dataset_name: str,
    completion_cfg: Dict[str, Any],
    output_dir: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(str(cfg.checkpoint))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist.")

    background_color = str(data_cfg.get("background_color", "white"))

    vqvae_checkpoint = cfg.get("vqvae_checkpoint")
    if not vqvae_checkpoint:
        model_cfg = (
            OmegaConf.to_container(cfg.get("model"), resolve=True)
            if cfg.get("model")
            else {}
        )
        if isinstance(model_cfg, dict):
            vqvae_config = model_cfg.get("vqvae")
            if isinstance(vqvae_config, dict):
                vqvae_checkpoint = vqvae_config.get("checkpoint_path")
    if not vqvae_checkpoint:
        vqvae_checkpoint = data_cfg.get("vqvae_path")
    if vqvae_checkpoint is None:
        raise ValueError(
            "A VQ-VAE checkpoint must be provided via `model.vqvae.checkpoint_path`, "
            "`data.vqvae_path`, or an explicit `vqvae_checkpoint` override."
        )

    print(f"Loading VQ-VAE from checkpoint: {vqvae_checkpoint}", flush=True)
    vqvae = GaussianVQVAE.load_from_checkpoint(str(vqvae_checkpoint)).eval()
    vqvae.set_background_color(background_color)

    print(f"Loading GaussianGPT from checkpoint: {checkpoint_path}", flush=True)
    gpt = GaussianGPT.load_from_checkpoint(str(checkpoint_path), vqvae=vqvae)
    gpt = gpt.eval().to(device)
    if getattr(gpt, "vqvae", None) is not None:
        gpt.vqvae.set_background_color(background_color)

    completion_split = str(completion_cfg.get("split", "val"))
    num_completions = int(completion_cfg.get("num_completions", 1))
    if num_completions < 1:
        raise ValueError("completion.num_completions must be >= 1.")
    prompt_mode = str(completion_cfg.get("prompt_mode", "spatial_half_x"))
    if prompt_mode == "spatial_half_x":
        prompt_fraction = 0.5
    elif prompt_mode == "spatial_quarter_x":
        prompt_fraction = 0.25
    else:
        raise ValueError(
            "completion.prompt_mode must be one of "
            "{'spatial_half_x', 'spatial_quarter_x'}."
        )

    completion_data_cfg = _resolve_completion_data_cfg(
        data_cfg,
        dataset=dataset_name,
        vqvae_checkpoint=str(vqvae_checkpoint),
    )
    data_max_tokens = completion_data_cfg.get("max_tokens")
    if data_max_tokens is not None:
        data_max_tokens = int(data_max_tokens)
    completion_dataset = _build_completion_dataset(
        completion_data_cfg,
        gpt,
        split=completion_split,
    )

    total_scenes = min(int(cfg.num_samples), len(completion_dataset))
    if total_scenes <= 0:
        raise ValueError("No samples available for completion.")

    shard_id, num_shards = _resolve_sharding(cfg)
    local_n, global_start, global_end = _shard_range(total_scenes, shard_id, num_shards)
    local_scene_indices = list(range(global_start, global_end))

    scene_id_filter = cfg.get("scene_id")
    if scene_id_filter is not None:
        if isinstance(scene_id_filter, (list, tuple)):
            scene_ids = [str(s) for s in scene_id_filter]
        else:
            scene_ids = [str(scene_id_filter)]
        matched = [
            i
            for i in range(len(completion_dataset))
            if any(sid in str(completion_dataset.file_list[i]) for sid in scene_ids)
        ]
        missing = [
            sid
            for sid in scene_ids
            if not any(sid in str(completion_dataset.file_list[i]) for i in matched)
        ]
        if missing:
            raise ValueError(
                f"scene_id(s) not found in the dataset file list: {missing}"
            )
        local_scene_indices = matched
        local_n = len(local_scene_indices)
        print(
            f"scene_id filter {scene_ids}: found {local_n} matching entries.",
            flush=True,
        )

    print(
        f"Completion setup: split={completion_split}, total_scenes={total_scenes}, "
        f"num_completions={num_completions}, shard={shard_id}/{num_shards} "
        f"prompt={prompt_mode}, local_scenes={local_n} [{global_start}, {global_end}).",
        flush=True,
    )

    render_gifs = bool(cfg.get("render_gifs", True))
    store_tokens = bool(cfg.get("store_tokens", True))
    needs_scene_decode = True

    gif_dir = output_dir / "gif" / f"shard_{shard_id:04d}"
    xview_dir = output_dir / "xview" / f"shard_{shard_id:04d}"
    xview_single_dir = output_dir / "xview_single" / f"shard_{shard_id:04d}"
    samples_dir = output_dir / "samples" / f"shard_{shard_id:04d}"
    tokens_dir = output_dir / "tokens" / f"shard_{shard_id:04d}"
    shards_dir = output_dir / "shards"

    shards_dir.mkdir(parents=True, exist_ok=True)
    if render_gifs:
        gif_dir.mkdir(parents=True, exist_ok=True)
    xview_dir.mkdir(parents=True, exist_ok=True)
    xview_single_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    if store_tokens:
        tokens_dir.mkdir(parents=True, exist_ok=True)

    pad_id = int(gpt.gpt.pad_token_id)
    base_side_length = int(gpt.position_side_length)
    batch_size = int(cfg.batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    completion_records: List[Dict[str, Any]] = []
    total_completion_jobs = local_n * num_completions
    completed_jobs = 0
    start_time = time.perf_counter()
    avg_batch_time = None
    num_batches = math.ceil(local_n / batch_size) if local_n > 0 else 0

    for batch_idx in range(num_batches):
        batch_scene_indices = local_scene_indices[
            batch_idx * batch_size : min((batch_idx + 1) * batch_size, local_n)
        ]
        batch_start = time.perf_counter()

        for global_scene_idx in batch_scene_indices:
            data_idx = global_scene_idx
            tokens = completion_dataset[data_idx]
            if not torch.is_tensor(tokens):
                tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.to(device=device, dtype=torch.long)

            (
                full_tokens,
                prefix_tokens,
                partial_tokens,
                prefix_len,
                remaining_len,
            ) = _split_prompt_spatial_x_fraction(
                gpt,
                tokens,
                fraction=prompt_fraction,
                pad_id=pad_id,
                base_side_length=base_side_length,
            )

            full_len = int(full_tokens.numel())
            if full_len == 0:
                prefix_len = 0
                remaining_len = 0
                gt_scene = None
                partial_scene = None
            else:
                gt_scene = None
                partial_scene = None
                if needs_scene_decode:
                    gt_scene = _decode_tokens_to_scene(
                        gpt,
                        full_tokens,
                        base_side_length=base_side_length,
                        pad_id=pad_id,
                    )
                    partial_scene = _decode_tokens_to_scene(
                        gpt,
                        partial_tokens,
                        base_side_length=base_side_length,
                        pad_id=pad_id,
                    )

            render_gt = None
            render_partial = None
            if needs_scene_decode:
                if gt_scene is None:
                    gt_scene = GaussianScene.from_dict(
                        gpt._empty_scene_dict(tokens.device)
                    )
                if partial_scene is None:
                    partial_scene = gt_scene
                render_gt = _center_scene_reference(gt_scene, gt_scene)
                render_partial = _center_scene_reference(partial_scene, gt_scene)

            partial_path = samples_dir / f"sample_{global_scene_idx:07d}_partial.pt"
            gt_path = samples_dir / f"sample_{global_scene_idx:07d}_gt.pt"
            if not partial_path.exists():
                torch.save(_scene_pt_payload(render_partial), partial_path)
            if not gt_path.exists():
                torch.save(_scene_pt_payload(render_gt), gt_path)

            partial_completed_view_path = (
                xview_single_dir
                / f"sample_{global_scene_idx:07d}_partial_completed_side.png"
            )
            partial_condition_view_path = (
                xview_single_dir
                / f"sample_{global_scene_idx:07d}_partial_condition_side.png"
            )
            gt_opposite_partial_view_path = (
                xview_single_dir
                / f"sample_{global_scene_idx:07d}_gt_opposite_partial_side.png"
            )
            if not partial_completed_view_path.exists():
                _render_and_save_fixed_view(
                    render_partial,
                    partial_completed_view_path,
                    camera_pos_xyz=(1.0, 0.0, 0.0),
                    background_color=background_color,
                )
            if not partial_condition_view_path.exists():
                _render_and_save_fixed_view(
                    render_partial,
                    partial_condition_view_path,
                    camera_pos_xyz=(-1.0, 0.0, 0.0),
                    background_color=background_color,
                )
            if not gt_opposite_partial_view_path.exists():
                _render_and_save_fixed_view(
                    render_gt,
                    gt_opposite_partial_view_path,
                    camera_pos_xyz=(-1.0, 0.0, 0.0),
                    background_color=background_color,
                )

            for completion_idx in range(num_completions):
                completion_seed = _completion_seed(
                    int(cfg.seed),
                    shard_id,
                    global_scene_idx,
                    completion_idx,
                )

                if remaining_len > 0:
                    if gpt.dense_chunks:
                        effective_max_retries = (
                            int(cfg.get("resample_max_retries", DEFAULT_MAX_RETRIES))
                            if bool(cfg.get("resample_empty_columns", True))
                            else 0
                        )
                        completed_tokens, _retry_stats = (
                            complete_with_empty_column_retry(
                                gpt,
                                prefix_tokens,
                                full_len=full_len,
                                pad_id=pad_id,
                                temperature=float(cfg.temperature),
                                top_k=cfg.get("top_k"),
                                top_p=cfg.get("top_p"),
                                seed=completion_seed,
                                max_retries=effective_max_retries,
                            )
                        )
                    else:
                        sparse_budget = max(0, int(gpt.model_config.n_ctx) - prefix_len)
                        completed_tokens = gpt.gpt.sample_sequence_with_prompt(
                            prefix_tokens,
                            max_new_tokens=sparse_budget,
                            num_samples=1,
                            temperature=float(cfg.temperature),
                            top_k=cfg.get("top_k"),
                            top_p=cfg.get("top_p"),
                            stop_on_eos=True,
                            seed=completion_seed,
                        )[0]
                else:
                    completed_tokens = prefix_tokens

                if gpt.dense_chunks:
                    # complete_with_empty_column_retry already guarantees full_len,
                    # but we keep the safety pad/truncate for the remaining_len==0
                    # branch above and for sparse fallbacks.
                    if completed_tokens.numel() > full_len:
                        completed_tokens = completed_tokens[:full_len]
                    elif completed_tokens.numel() < full_len:
                        pad = torch.full(
                            (full_len - completed_tokens.numel(),),
                            pad_id,
                            device=device,
                            dtype=torch.long,
                        )
                        completed_tokens = torch.cat([completed_tokens, pad], dim=0)

                token_path = None
                if store_tokens:
                    token_path = (
                        tokens_dir
                        / f"sample_{global_scene_idx:07d}_completion_{completion_idx:03d}.pt"
                    )
                    token_payload = {
                        "global_scene_idx": int(global_scene_idx),
                        "dataset_index": int(data_idx),
                        "completion_index": int(completion_idx),
                        "seed": int(completion_seed),
                        "full_length": int(full_len),
                        "prefix_length": int(prefix_len),
                        "completed_tokens": completed_tokens.detach()
                        .cpu()
                        .to(dtype=torch.long),
                    }
                    torch.save(token_payload, token_path)

                sample_path = None
                gif_path = None
                xview_path = (
                    xview_dir
                    / f"sample_{global_scene_idx:07d}_completion_{completion_idx:03d}.png"
                )
                render_completed = None
                if needs_scene_decode:
                    completed_scene = _decode_tokens_to_scene(
                        gpt,
                        completed_tokens,
                        base_side_length=base_side_length,
                        pad_id=pad_id,
                    )
                    assert gt_scene is not None
                    render_completed = _center_scene_reference(
                        completed_scene, gt_scene
                    )

                assert render_completed is not None
                sample_path = (
                    samples_dir
                    / f"sample_{global_scene_idx:07d}_completion_{completion_idx:03d}.pt"
                )
                torch.save(_scene_pt_payload(render_completed), sample_path)
                assert render_partial is not None
                assert render_gt is not None
                _render_and_save_xview_strip(
                    [render_partial, render_completed, render_gt],
                    xview_path,
                    per_scene_camera_positions_xyz=[
                        (1.0, 0.0, 0.0),  # partial (condition-facing side)
                        (-1.0, 0.0, 0.0),  # completion (hidden side)
                        (-1.0, 0.0, 0.0),  # GT (hidden side)
                    ],
                    background_color=background_color,
                )
                completion_xview_path = xview_single_dir / (
                    f"sample_{global_scene_idx:07d}_completion_"
                    f"{completion_idx:03d}_opposite_partial_side.png"
                )
                _render_and_save_fixed_view(
                    render_completed,
                    completion_xview_path,
                    camera_pos_xyz=(-1.0, 0.0, 0.0),
                    background_color=background_color,
                )

                if render_gifs:
                    assert render_partial is not None
                    assert render_completed is not None
                    assert render_gt is not None
                    gif_path = (
                        gif_dir
                        / f"sample_{global_scene_idx:07d}_completion_{completion_idx:03d}.gif"
                    )
                    render_and_save_trajectory_strip(
                        [render_partial, render_completed, render_gt],
                        str(gif_path),
                        num_frames=int(getattr(cfg, "gif_frames", 120)),
                        fps=int(getattr(cfg, "gif_fps", 24)),
                        background_color=background_color,
                    )

                completion_records.append(
                    {
                        "global_scene_idx": int(global_scene_idx),
                        "dataset_index": int(data_idx),
                        "completion_index": int(completion_idx),
                        "seed": int(completion_seed),
                        "full_length": int(full_len),
                        "prefix_length": int(prefix_len),
                        "token_path": (
                            str(token_path) if token_path is not None else None
                        ),
                        "sample_path": (
                            str(sample_path) if sample_path is not None else None
                        ),
                        "gif_path": str(gif_path) if gif_path is not None else None,
                        "xview_path": str(xview_path),
                        "xview_single_completion_path": str(completion_xview_path),
                    }
                )

                completed_jobs += 1

        batch_time = time.perf_counter() - batch_start
        avg_batch_time = (
            batch_time
            if avg_batch_time is None
            else 0.9 * avg_batch_time + 0.1 * batch_time
        )
        elapsed = time.perf_counter() - start_time
        remaining_batches = num_batches - (batch_idx + 1)
        eta = remaining_batches * avg_batch_time
        print(
            f"Batch {batch_idx + 1}/{num_batches} done in {batch_time:.2f}s "
            f"(scenes={len(batch_scene_indices)}, completions={completed_jobs}/{total_completion_jobs}). "
            f"Elapsed {elapsed:.1f}s, ETA ~{eta:.1f}s.",
            flush=True,
        )

    shard_payload = {
        "metadata": {
            "shard_id": int(shard_id),
            "num_shards": int(num_shards),
            "global_start_scene_idx": int(global_start),
            "global_end_scene_idx_exclusive": int(global_end),
            "num_local_scenes": int(local_n),
            "num_completions": int(num_completions),
            "num_local_completions": int(len(completion_records)),
            "checkpoint": str(checkpoint_path),
            "vqvae_checkpoint": str(vqvae_checkpoint),
            "dataset": str(dataset_name),
            "completion_split": completion_split,
            "prompt_mode": prompt_mode,
            "temperature": float(cfg.temperature),
            "top_k": cfg.get("top_k"),
            "top_p": cfg.get("top_p"),
            "seed": int(cfg.seed),
            "batch_size": int(batch_size),
            "tokens_per_latent": int(gpt.tokens_per_latent),
            "dense_chunks": bool(getattr(gpt, "dense_chunks", False)),
            "chunk_shape": (
                [int(v) for v in gpt.chunk_shape]
                if getattr(gpt, "chunk_shape", None) is not None
                else None
            ),
            "chunk_order": str(getattr(gpt, "chunk_order", "xyz")),
            "store_tokens": bool(store_tokens),
            "store_samples": True,
            "render_gifs": bool(render_gifs),
            "render_xview": True,
            "render_xview_single": True,
            "background_color": str(background_color),
            "max_tokens": data_max_tokens,
        },
        "completion_records": completion_records,
    }
    shard_path = shards_dir / f"shard_{shard_id:04d}.pt"
    torch.save(shard_payload, shard_path)

    total_time = time.perf_counter() - start_time
    print(
        f"Wrote shard {shard_path} in {total_time:.2f}s "
        f"(local_scenes={local_n}, local_completions={len(completion_records)}).",
        flush=True,
    )


@hydra.main(config_path="conf", config_name="complete_chunks", version_base=None)
def main(cfg: DictConfig) -> None:
    completion_cfg = (
        OmegaConf.to_container(cfg.get("completion"), resolve=True)
        if cfg.get("completion")
        else {}
    )
    if not isinstance(completion_cfg, dict):
        raise TypeError("Expected `cfg.completion` to convert to a dictionary.")
    if not bool(completion_cfg.get("enabled", True)):
        raise ValueError("complete_chunks requires completion.enabled=true.")

    data_cfg: Dict[str, Any] = {}
    if cfg.get("data") is not None:
        data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
        if not isinstance(data_cfg, dict):
            raise TypeError("Expected `cfg.data` to convert to a dictionary.")
        if data_cfg.get("sort_latents") is not None:
            raise ValueError(
                "`data.sort_latents` is deprecated outside tokenization; use `model.chunk_order`."
            )
    else:
        raise ValueError(
            "Completion requires a data config. Pass a tokenized data config via `data=...`."
        )

    dataset_name = cfg.get("dataset")
    if dataset_name is None:
        dataset_name = data_cfg.get("dataset_name")
    if dataset_name is None:
        raise ValueError(
            "Completion requires dataset name via `dataset` or `data.dataset_name`."
        )

    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    lightning.seed_everything(int(cfg.seed), workers=True)

    run_completion_inference(
        cfg=cfg,
        data_cfg=data_cfg,
        dataset_name=str(dataset_name),
        completion_cfg=completion_cfg,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
