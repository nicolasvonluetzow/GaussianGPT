from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from utils.render import GaussianScene, rotate_scene_y_up_to_z_up
from utils.render import render as render_fn

REQUIRED_SCENE_KEYS = {"coords", "sh0", "opacities", "scales", "quats"}
NUM_FRAMES = 180
A_X_DEG = -30.0
MOVE_BACK = 0.2
GIF_FPS = 24
EPS = 1e-6


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render fixed circular-view suites from decoded Gaussian scene .pt files."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a decoded scene .pt file or directory containing .pt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output root directory. Writes frames to <output-dir>/views and gifs to <output-dir>/gif.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help="When --input is a directory, process only the first N .pt files after sorting.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Square render resolution in pixels.",
    )
    parser.add_argument(
        "--background-color",
        choices=["white", "black"],
        default="white",
        help="Background color used for rendering.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for rendering.",
    )
    parser.add_argument(
        "--up",
        choices=["z", "y"],
        default="z",
        help="Up axis of the input payload. 'y' rotates the scene to z-up before rendering.",
    )
    parser.add_argument(
        "--move-back",
        type=float,
        default=MOVE_BACK,
        help=(
            "Camera pull-back distance as a multiple of the smallest AABB extent. "
            "Default orbits inside rooms; use ~2-3 to orbit outside an object."
        ),
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def _discover_input_files(
    input_path: Path, max_files: int | None
) -> tuple[list[Path], int]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pt":
            raise ValueError(f"--input file must end with .pt, got {input_path}.")
        return [input_path], 1

    if input_path.is_dir():
        all_files = sorted(input_path.rglob("*.pt"), key=lambda p: str(p.resolve()))
        discovered = len(all_files)
        if discovered == 0:
            raise ValueError(f"No .pt files found recursively under {input_path}.")
        if max_files is not None:
            all_files = all_files[:max_files]
        if len(all_files) == 0:
            raise ValueError("No files selected for processing after --max-files.")
        return all_files, discovered

    raise ValueError(
        f"--input must be an existing file or directory, got {input_path}."
    )


def _load_scene(path: Path, device: torch.device) -> GaussianScene:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a dict payload in {path}, got {type(payload).__name__}."
        )

    missing = REQUIRED_SCENE_KEYS.difference(payload.keys())
    if missing:
        if "tokens_grid" in payload or "tokens_sequence" in payload:
            raise ValueError(
                f"{path} looks like a token file. Use a decoded scene .pt file instead."
            )
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            f"{path} does not contain a renderable scene dict. Missing keys: {missing_str}."
        )
    return GaussianScene.from_dict(payload).to(device)


def _rotation_x(angle_deg: float, *, device: torch.device) -> torch.Tensor:
    angle = torch.deg2rad(
        torch.tensor(float(angle_deg), device=device, dtype=torch.float32)
    )
    c = torch.cos(angle)
    s = torch.sin(angle)
    rot = torch.eye(4, device=device, dtype=torch.float32)
    rot[1, 1] = c
    rot[1, 2] = -s
    rot[2, 1] = s
    rot[2, 2] = c
    return rot


def _rotation_z(angle_deg: float, *, device: torch.device) -> torch.Tensor:
    angle = torch.deg2rad(
        torch.tensor(float(angle_deg), device=device, dtype=torch.float32)
    )
    c = torch.cos(angle)
    s = torch.sin(angle)
    rot = torch.eye(4, device=device, dtype=torch.float32)
    rot[0, 0] = c
    rot[0, 1] = -s
    rot[1, 0] = s
    rot[1, 1] = c
    return rot


def _build_fixed_views(scene: GaussianScene, move_back: float) -> torch.Tensor:
    coords = scene.means
    if coords.numel() == 0:
        raise ValueError("Input scene is empty.")

    min_vals = coords.amin(dim=0)
    max_vals = coords.amax(dim=0)
    center = 0.5 * (max_vals + min_vals)
    extent = torch.clamp((max_vals - min_vals).min(), min=EPS)

    device = coords.device
    c2w = torch.eye(4, device=device, dtype=torch.float32)
    c2w[:3, 3] = center.to(dtype=torch.float32)

    rot_x = _rotation_x(-90.0 + A_X_DEG, device=device)
    translate_back = torch.eye(4, device=device, dtype=torch.float32)
    translate_back[2, 3] = -move_back * extent

    view_mats = []
    rot_z_step = 360.0 / float(NUM_FRAMES)
    for frame_idx in range(NUM_FRAMES):
        rot_z = _rotation_z(rot_z_step * frame_idx, device=device)
        curr_c2w = c2w @ rot_z @ rot_x @ translate_back
        curr_w2c = torch.linalg.inv(curr_c2w)
        view_mats.append(curr_w2c)
    return torch.stack(view_mats, dim=0)


def _render_views(
    scene: GaussianScene,
    view_mats: torch.Tensor,
    resolution: int,
    background_color: str,
) -> np.ndarray:
    rendered, _ = render_fn(
        scene,
        view_mats,
        render_size=(int(resolution), int(resolution)),
        background_color=background_color,
    )
    images = rendered.permute(0, 2, 3, 1).mul(255.0).clamp(0, 255).byte().cpu().numpy()
    return images


def _render_scene_suite(
    scene_path: Path,
    scene_key: str,
    *,
    views_root: Path,
    gif_root: Path,
    resolution: int,
    background_color: str,
    device: torch.device,
    up: str,
    move_back: float,
) -> None:
    scene = _load_scene(scene_path, device=device)
    if up == "y":
        scene = rotate_scene_y_up_to_z_up(scene)
    view_mats = _build_fixed_views(scene, move_back)
    images = _render_views(
        scene,
        view_mats,
        resolution=resolution,
        background_color=background_color,
    )

    scene_views_dir = views_root / scene_key
    scene_views_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, image in enumerate(images):
        frame_path = scene_views_dir / f"view_{frame_idx:03d}.png"
        iio.imwrite(frame_path, image)

    gif_path = gif_root / f"{scene_key}.gif"
    iio.imwrite(gif_path, images, fps=GIF_FPS, loop=0, palette_size=256)


def main() -> None:
    args = _build_args()
    if args.resolution <= 0:
        raise ValueError(f"--resolution must be > 0, got {args.resolution}.")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError(f"--max-files must be > 0, got {args.max_files}.")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"--input does not exist: {input_path}")
    if input_path.is_file() and args.max_files is not None:
        raise ValueError("--max-files can only be used when --input is a directory.")

    output_root = Path(args.output_dir).expanduser().resolve()
    views_root = output_root / "views"
    gif_root = output_root / "gif"
    views_root.mkdir(parents=True, exist_ok=True)
    gif_root.mkdir(parents=True, exist_ok=True)

    files, discovered_count = _discover_input_files(input_path, args.max_files)
    device = _resolve_device(args.device)

    print(f"Input: {input_path}", flush=True)
    if input_path.is_dir():
        print(
            f"Discovered {discovered_count} .pt files, selected {len(files)} for processing.",
            flush=True,
        )
    print(f"Output views root: {views_root}", flush=True)
    print(f"Output gif root:   {gif_root}", flush=True)
    print(f"Device: {device}", flush=True)

    processed = 0
    for idx, scene_path in enumerate(files):
        scene_key = f"{scene_path.stem}__idx{idx:04d}"
        print(
            f"[{idx + 1}/{len(files)}] Rendering {scene_path} -> {scene_key}",
            flush=True,
        )
        _render_scene_suite(
            scene_path,
            scene_key,
            views_root=views_root,
            gif_root=gif_root,
            resolution=int(args.resolution),
            background_color=args.background_color,
            device=device,
            up=args.up,
            move_back=float(args.move_back),
        )
        processed += 1

    print(
        f"Done. Rendered {processed} scene(s) with {NUM_FRAMES} views each.",
        flush=True,
    )


if __name__ == "__main__":
    main()
