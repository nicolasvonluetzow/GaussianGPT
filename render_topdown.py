from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import torch

from utils.render import GaussianScene, render, rotate_scene_y_up_to_z_up

REQUIRED_SCENE_KEYS = {"coords", "sh0", "opacities", "scales", "quats"}
FIT_MARGIN = 1.05


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one top-down PNG from a decoded Gaussian scene .pt file."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Decoded scene .pt file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path. Default: <input_stem>_topdown_q{quantile}.png",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=75.0,
        help="Remove points with z above this quantile (0-100).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Square output resolution in pixels.",
    )
    parser.add_argument(
        "--background-color",
        choices=["white", "black"],
        default="white",
        help="Background color used for rendering.",
    )
    parser.add_argument(
        "--up",
        choices=["z", "y"],
        default="z",
        help="Up axis of the input payload. 'y' rotates the scene to z-up before rendering.",
    )
    return parser.parse_args()


def _load_scene_payload(path: Path, device: torch.device) -> dict:
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
    return payload


def _filter_by_z_quantile(
    payload: dict, quantile: float
) -> tuple[dict, float, int, int]:
    coords = payload["coords"]
    if coords.numel() == 0:
        raise ValueError("Input scene is empty.")

    threshold = float(torch.quantile(coords[:, 2], float(quantile) / 100.0).item())
    keep_mask = coords[:, 2] <= threshold
    total_points = int(coords.shape[0])
    kept_points = int(keep_mask.sum().item())
    if kept_points == 0:
        raise ValueError(
            "No points left after quantile filtering. Lower `--quantile` or use a different scene."
        )

    filtered = {}
    for key, value in payload.items():
        if (
            torch.is_tensor(value)
            and value.dim() > 0
            and value.shape[0] == total_points
        ):
            filtered[key] = value[keep_mask]
        else:
            filtered[key] = value
    return filtered, threshold, kept_points, total_points


def _center_xy(payload: dict) -> dict:
    centered = dict(payload)
    coords = centered["coords"].clone()
    min_xy = coords[:, :2].amin(dim=0)
    max_xy = coords[:, :2].amax(dim=0)
    xy_center = 0.5 * (min_xy + max_xy)
    coords[:, 0] -= xy_center[0]
    coords[:, 1] -= xy_center[1]
    centered["coords"] = coords
    return centered


def _build_topdown_view_matrix(
    camera_pos: torch.Tensor,
    look_at: torch.Tensor,
    *,
    up_world: torch.Tensor,
) -> torch.Tensor:
    # Match the world->camera convention used in utils.render.
    fwd = torch.nn.functional.normalize(look_at - camera_pos, dim=0)
    right = torch.nn.functional.normalize(torch.cross(fwd, up_world, dim=0), dim=0)
    up_cam = torch.cross(right, fwd, dim=0)
    rot = torch.stack([right, up_cam, fwd], dim=1)
    trans = -(rot.transpose(0, 1) @ camera_pos)

    view = torch.eye(4, device=camera_pos.device, dtype=torch.float32)
    view[:3, :3] = rot.transpose(0, 1)
    view[:3, 3] = trans
    return view


def _camera_for_topdown(
    payload: dict, resolution: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    coords = payload["coords"]
    device = coords.device
    half_x = float(coords[:, 0].abs().amax().item())
    half_y = float(coords[:, 1].abs().amax().item())
    half_extent = max(half_x, half_y, 1e-6)

    z_min = float(coords[:, 2].amin().item())
    z_max = float(coords[:, 2].amax().item())
    z_look = 0.5 * (z_min + z_max)
    top_clearance = z_max - z_look

    focal = 0.9 * float(resolution)
    principal = 0.5 * float(resolution)
    distance_xy = half_extent * focal / principal
    distance_fit = max(distance_xy * FIT_MARGIN, 1e-4)
    camera_z = z_look + top_clearance + distance_fit

    camera_pos = torch.tensor([0.0, 0.0, camera_z], device=device, dtype=torch.float32)
    look_at = torch.tensor([0.0, 0.0, z_look], device=device, dtype=torch.float32)
    up_world = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)

    view = _build_topdown_view_matrix(camera_pos, look_at, up_world=up_world)
    intrinsics = torch.zeros((1, 3, 3), device=device, dtype=torch.float32)
    intrinsics[0, 0, 0] = focal
    intrinsics[0, 1, 1] = focal
    intrinsics[0, 0, 2] = principal
    intrinsics[0, 1, 2] = principal
    intrinsics[0, 2, 2] = 1.0
    return view.unsqueeze(0), intrinsics, float(camera_z)


def _quantile_suffix(quantile: float) -> str:
    if float(quantile).is_integer():
        return str(int(quantile))
    return str(quantile).replace(".", "p")


def main() -> None:
    args = _build_args()
    if args.quantile < 0.0 or args.quantile > 100.0:
        raise ValueError(f"--quantile must be in [0, 100], got {args.quantile}.")
    if args.resolution <= 0:
        raise ValueError(f"--resolution must be > 0, got {args.resolution}.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if args.output is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_topdown_q{_quantile_suffix(args.quantile)}.png"
        )
    else:
        output_path = Path(args.output)
        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = _load_scene_payload(input_path, device=device)
    if args.up == "y":
        payload = rotate_scene_y_up_to_z_up(GaussianScene.from_dict(payload)).to_dict()
    filtered, threshold, kept_points, total_points = _filter_by_z_quantile(
        payload,
        quantile=float(args.quantile),
    )
    centered = _center_xy(filtered)
    scene = GaussianScene.from_dict(centered)

    view_mats, intrinsics, camera_z = _camera_for_topdown(
        centered, int(args.resolution)
    )
    rendered, _ = render(
        scene,
        view_mats,
        intrinsics=intrinsics,
        render_size=(int(args.resolution), int(args.resolution)),
        background_color=args.background_color,
    )
    image = rendered[0].permute(1, 2, 0).mul(255.0).clamp(0, 255).byte().cpu().numpy()
    iio.imwrite(output_path, image)

    print(f"Saved top-down render to {output_path}", flush=True)
    print(
        f"Kept {kept_points}/{total_points} points (z <= {threshold:.6f}, q={args.quantile:.2f}).",
        flush=True,
    )
    print(f"Camera z position: {camera_z:.6f}", flush=True)


if __name__ == "__main__":
    main()
