import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization


def get_default_intrinsics(
    width: int, height: int, device: torch.device = torch.device("cpu")
):
    """
    Returns a pinhole camera matrix with focal points at the center of the image.
    """
    return torch.tensor(
        [[0.5 * width, 0, 0.5 * width], [0, 0.5 * height, 0.5 * height], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )


def get_default_view_matrix(
    distance: float = 1.0, device: torch.device = torch.device("cpu")
):
    """
    Returns a view matrix looking at the origin from a distance of distance.
    """
    view_matrices = get_view_matrices_looking_at_origin(
        torch.tensor([0.0, 0.0, -distance], device=device).unsqueeze(0), device=device
    )
    return view_matrices.squeeze(0)


def _resolve_background_color(
    background_color: str | Sequence[float] | torch.Tensor,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert a background color specification into a tensor of shape (count, 3).
    Accepts string identifiers ('white' or 'black'), RGB sequences, or tensors.
    """

    if isinstance(background_color, torch.Tensor):
        bg = background_color.to(device=device, dtype=torch.float32)
        if bg.dim() == 1:
            bg = bg.unsqueeze(0)
        if bg.shape[-1] != 3:
            raise ValueError(
                f"Background tensor must have shape (*, 3), got {bg.shape}."
            )
        if bg.shape[0] == 1:
            bg = bg.expand(count, -1)
        elif bg.shape[0] != count:
            bg = bg.repeat(int(math.ceil(count / bg.shape[0])), 1)[:count]
        return bg.contiguous()

    if isinstance(background_color, (tuple, list)):
        if len(background_color) != 3:
            raise ValueError(
                f"Background color sequence must have length 3, got {len(background_color)}."
            )
        return (
            torch.tensor(background_color, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .expand(count, -1)
            .contiguous()
        )

    if isinstance(background_color, str):
        if background_color == "white":
            value = 1.0
        elif background_color == "black":
            value = 0.0
        else:
            raise ValueError(
                f"Unsupported background color '{background_color}'. Expected 'white' or 'black'."
            )
        return torch.full((count, 3), value, device=device, dtype=torch.float32)

    raise TypeError(
        "background_color must be a string identifier, RGB sequence, or tensor."
    )


class GaussianScene:
    """
    A class to represent Gaussian splats in a scene and provide some QoL functionality.
    """

    def __init__(
        self,
        means,
        sh0,
        opacities,
        scales,
        quats,
        sh: Optional[torch.Tensor] = None,
    ):
        self.means = means
        self.sh0 = sh0
        self.sh = sh
        self.opacities = opacities
        self.scales = scales
        self.quats = quats

        self.device = self.means.device

    def to(self, *args, **kwargs):
        self.means = self.means.to(*args, **kwargs)
        self.sh0 = self.sh0.to(*args, **kwargs)
        if self.sh is not None:
            self.sh = self.sh.to(*args, **kwargs)
        self.opacities = self.opacities.to(*args, **kwargs)
        self.scales = self.scales.to(*args, **kwargs)
        self.quats = self.quats.to(*args, **kwargs)

        self.device = self.means.device

        return self

    @classmethod
    def from_dict(cls, d: dict):
        is_batched = d["coords"].ndim == 3
        assert (
            not is_batched or d["coords"].shape[0] == 1
        ), f"Class only describes a single scene, so batch dimension must be 1, got {d['coords'].shape} for coords."

        if is_batched:
            sh = d["sh"].squeeze(0) if "sh" in d else None
            return cls(
                means=d["coords"].squeeze(0),
                opacities=d["opacities"].squeeze(0),
                sh0=d["sh0"].squeeze(0),
                scales=d["scales"].squeeze(0),
                quats=d["quats"].squeeze(0),
                sh=sh,
            )
        else:
            sh = d["sh"] if "sh" in d else None
            return cls(
                means=d["coords"],
                opacities=d["opacities"],
                sh0=d["sh0"],
                scales=d["scales"],
                quats=d["quats"],
                sh=sh,
            )

    def to_dict(self, batched: bool = False):
        if batched:
            out = {
                "coords": self.means.unsqueeze(0),
                "opacities": self.opacities.unsqueeze(0),
                "sh0": self.sh0.unsqueeze(0),
                "scales": self.scales.unsqueeze(0),
                "quats": self.quats.unsqueeze(0),
            }
            if self.sh is not None:
                out["sh"] = self.sh.unsqueeze(0)
            return out
        else:
            out = {
                "coords": self.means,
                "opacities": self.opacities,
                "sh0": self.sh0,
                "scales": self.scales,
                "quats": self.quats,
            }
            if self.sh is not None:
                out["sh"] = self.sh
            return out

    def render(self, background_color: str | Sequence[float] | torch.Tensor = "white"):
        """
        QoL that returns a rendered image of the GaussianScene rendered from a default view matrix.
        """
        rendered_frame, _ = render(
            self,
            get_default_view_matrix(device=self.means.device).unsqueeze(0),
            background_color=background_color,
        )
        return rendered_frame

    def render_and_save_trajectory(
        self,
        path: str,
        num_frames: int = 100,
        fps: int = 30,
        background_color: str | Sequence[float] | torch.Tensor = "white",
    ):
        return render_and_save_trajectory(
            self,
            path,
            num_frames,
            fps,
            background_color=background_color,
        )


def center_scene_aabb(scene: GaussianScene) -> GaussianScene:
    coords = scene.means
    if coords.numel() == 0:
        return scene
    min_vals, max_vals = torch.aminmax(coords, dim=0)
    center = 0.5 * (min_vals + max_vals)
    return GaussianScene(
        means=coords - center,
        sh0=scene.sh0,
        sh=scene.sh,
        opacities=scene.opacities,
        scales=scene.scales,
        quats=scene.quats,
    )


def render_core(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    sh: Optional[torch.Tensor],
    view_matrices: torch.Tensor,
    intrinsics: torch.Tensor = None,
    render_size: Tuple[int, int] = (512, 512),
    scales_activation_fn: Callable = torch.exp,
    opacities_activation_fn: Callable = torch.sigmoid,
    background_color: str | Sequence[float] | torch.Tensor = "white",
    render_mode: str = "RGB",
):

    assert view_matrices.dim() == 3 and view_matrices.shape[1:] == (
        4,
        4,
    ), f"view_matrices must be a 3D tensor of shape (C, 4, 4), got {view_matrices.shape}."
    assert intrinsics is None or (
        intrinsics.dim() == 3 and intrinsics.shape[1:] == (3, 3)
    ), f"intrinsics must be a 3D tensor of shape (C, 3, 3), got {intrinsics.shape}."

    # parse splats
    device = means.device

    means = means.to(torch.float32)
    quats = quats.to(torch.float32)
    scales = scales_activation_fn(scales.to(torch.float32))
    opacities = opacities_activation_fn(opacities.to(torch.float32).flatten())
    sh0 = sh0.to(torch.float32)
    if sh0.dim() != 2 or sh0.shape[1] != 3:
        raise ValueError(f"Expected sh0 with shape (N, 3), got {sh0.shape}.")

    colors = sh0[:, None]  # N, 1, 3
    if sh is not None:
        sh = sh.to(torch.float32)
        if sh.dim() != 2 or sh.shape[0] != sh0.shape[0] or sh.shape[1] % 3 != 0:
            raise ValueError(
                "Expected sh with shape (N, D) where D is divisible by 3 and N matches sh0, "
                f"got {sh.shape} with sh0 {sh0.shape}."
            )
        sh = sh.reshape(sh.shape[0], sh.shape[1] // 3, 3)
        colors = torch.cat([colors, sh], dim=1)

    sh_degree = int(math.sqrt(colors.shape[1]) - 1)
    if (sh_degree + 1) ** 2 != colors.shape[1]:
        raise ValueError(
            f"Invalid SH coefficient count: got {colors.shape[1]} coefficients per point."
        )
    C = view_matrices.shape[0]  # number of cameras
    backgrounds = _resolve_background_color(background_color, C, device)

    if intrinsics is None:
        intrinsics = get_default_intrinsics(
            render_size[0], render_size[1], device=device
        )
        intrinsics = intrinsics.unsqueeze(0).repeat(C, 1, 1)

    render_results = dict()

    rasterization_kwargs = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=view_matrices,
        Ks=intrinsics,
        width=render_size[0],
        height=render_size[1],
        sh_degree=sh_degree,
        backgrounds=backgrounds,
        render_mode=render_mode,
    )
    try:
        render_colors, render_alphas, _ = rasterization(**rasterization_kwargs)
    except TypeError as exc:
        if "render_mode" not in str(exc):
            raise
        if render_mode != "RGB":
            raise ValueError(
                "Current gsplat rasterization backend does not support "
                f"render_mode='{render_mode}'."
            ) from exc
        rasterization_kwargs.pop("render_mode", None)
        render_colors, render_alphas, _ = rasterization(**rasterization_kwargs)
    render_results["alphas"] = render_alphas

    # render_colors shape is (C, H, W, channels) - move channels first for training code.
    render_colors = render_colors.permute(0, 3, 1, 2).to(device)

    # auxilliary shape is originally (C, H, W, 1 or 3) - we just adjust this so its consistent with the colors
    for key in render_results:
        render_results[key] = render_results[key].permute(0, 3, 1, 2).to(device)

    return render_colors, render_results


def render(
    gaussian_splats: GaussianScene,
    view_matrices: torch.Tensor,
    intrinsics: torch.Tensor = None,
    render_size: Tuple[int, int] = (512, 512),
    scales_activation_fn: Callable = torch.exp,
    opacities_activation_fn: Callable = torch.sigmoid,
    background_color: str | Sequence[float] | torch.Tensor = "white",
    render_mode: str = "RGB",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Render a GaussianScene with gradients. Wrapper for render_core that accepts GaussianScenes.
    """

    return render_core(
        gaussian_splats.means,
        gaussian_splats.quats,
        gaussian_splats.scales,
        gaussian_splats.opacities,
        gaussian_splats.sh0,
        gaussian_splats.sh,
        view_matrices,
        intrinsics,
        render_size,
        scales_activation_fn,
        opacities_activation_fn,
        background_color=background_color,
        render_mode=render_mode,
    )


def get_view_matrices_looking_at_origin(
    positions: torch.Tensor,
    world_up: torch.Tensor = torch.tensor([0.0, -1.0, 0.0]),
    device: torch.device = torch.device("cpu"),
):
    # Given positions (N, 3) return the world-to-cam transforms (N, 4, 4) looking at the origin
    assert (
        positions.dim() == 2 and positions.shape[1] == 3
    ), f"positions must be a 2D tensor of shape (N, 3), got {positions.shape}"

    # get the world to cam transformations
    pos = positions.to(device, dtype=torch.float32)
    up_world = world_up.to(device, dtype=torch.float32)
    up_world = F.normalize(up_world, dim=0)

    # forward vector camera to origin
    fwd = F.normalize(-pos, dim=1)

    # Right is fwd x up
    right = torch.cross(fwd, up_world.expand_as(fwd), dim=1)

    # we have an issue if fwd and up are parallel (or really close to it)
    # pylint: disable-next=not-callable
    unstable = torch.linalg.norm(right, dim=1) < 1e-6
    if unstable.any():
        alt_up = torch.tensor([0.0, 0.0, 1.0], device=device)
        right[unstable] = torch.cross(
            fwd[unstable], alt_up.expand(unstable.sum(), 3), dim=1
        )
    right = F.normalize(right, dim=1)

    # camera space up is right x fwd
    up_cam = torch.cross(right, fwd, dim=1)

    # rotation matrix is fully defined by the above vectors, we just stack them
    rot = torch.stack([right, up_cam, fwd], dim=2)

    # world to cam translation is t = -R^T * pos
    trans = -(rot.transpose(1, 2) @ pos.unsqueeze(-1))

    # assemble the homogeneous matrices
    view_matrices = torch.eye(4, device=device).repeat(positions.shape[0], 1, 1)
    view_matrices[:, :3, :3] = rot.transpose(1, 2)
    view_matrices[:, :3, 3] = trans.squeeze(-1)

    return view_matrices


def render_and_save_trajectory(
    gaussian_splats: GaussianScene,
    path: str,
    num_frames: int = 100,
    fps: int = 30,
    background_color: str | Sequence[float] | torch.Tensor = "white",
):
    # trajectory is spiralling around the origin, looking at it,
    # basically just a circle around the world Z axis, where we change the Z coordinate and derive the rotation from
    # the position afterwards. We use a sine scaled so the start/end Z coordinate matches for a smooth loop.

    z_min, z_max = -0.5, 0.5
    radius = 1

    device = gaussian_splats.means.device
    ts = torch.linspace(0.0, 1.0, num_frames)  # (num_frames,)

    positions = torch.stack(
        [
            radius * torch.cos(ts * 2 * math.pi),
            radius * torch.sin(ts * 2 * math.pi),
            torch.sin(ts * 2 * math.pi + math.pi) * (z_max - z_min) / 2.0
            + (z_max + z_min) / 2.0,
        ],
        dim=1,
    )
    view_matrices = get_view_matrices_looking_at_origin(
        positions,
        world_up=torch.tensor([0.0, 0.0, -1.0], device=device),
        device=device,
    )

    # could run this in batches but this is very fast already and this way we avoid some possible memory issues
    frames = []
    for view_matrix in view_matrices:
        render_pred, _ = render(
            gaussian_splats,
            view_matrix.unsqueeze(0),
            background_color=background_color,
        )
        frames.append(
            render_pred.squeeze(0)
            .permute(1, 2, 0)
            .mul(255)
            .clamp(0, 255)
            .byte()
            .cpu()
            .numpy()
        )

    # combine into a GIF
    iio.imwrite(path, np.stack(frames), fps=fps, loop=0, palette_size=256)


def render_and_save_trajectory_strip(
    gaussian_splats_list: Sequence[GaussianScene],
    path: str,
    num_frames: int = 100,
    fps: int = 30,
    background_color: str | Sequence[float] | torch.Tensor = "white",
):
    if not gaussian_splats_list:
        raise ValueError("Expected at least one scene for strip rendering.")

    device = gaussian_splats_list[0].means.device
    scenes = []
    for scene in gaussian_splats_list:
        if scene.means.device != device:
            scene = scene.to(device)
        scenes.append(scene)

    z_min, z_max = -0.5, 0.5
    radius = 1

    ts = torch.linspace(0.0, 1.0, num_frames, device=device)  # (num_frames,)
    positions = torch.stack(
        [
            radius * torch.cos(ts * 2 * math.pi),
            radius * torch.sin(ts * 2 * math.pi),
            torch.sin(ts * 2 * math.pi + math.pi) * (z_max - z_min) / 2.0
            + (z_max + z_min) / 2.0,
        ],
        dim=1,
    )
    view_matrices = get_view_matrices_looking_at_origin(
        positions,
        world_up=torch.tensor([0.0, 0.0, -1.0], device=device),
        device=device,
    )

    frames = []
    for view_matrix in view_matrices:
        strips = []
        for scene in scenes:
            render_pred, _ = render(
                scene,
                view_matrix.unsqueeze(0),
                background_color=background_color,
            )
            strips.append(render_pred.squeeze(0))
        strip = torch.cat(strips, dim=2)
        frames.append(
            strip.permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
        )

    iio.imwrite(path, np.stack(frames), fps=fps, loop=0, palette_size=256)


def flip_gaussian_scene(scene: GaussianScene, axis: str | int) -> GaussianScene:
    if isinstance(axis, str):
        axis_to_idx = {"x": 0, "y": 1, "z": 2}
        if axis not in axis_to_idx:
            raise ValueError(f"axis must be one of 'x', 'y', 'z', got {axis!r}")
        axis = axis_to_idx[axis]
    elif axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, 2 or 'x', 'y', 'z', got {axis!r}")

    means = scene.means.clone()
    means[:, axis].neg_()

    quats = scene.quats.clone()
    if axis == 0:
        quats[:, 2:].neg_()
    elif axis == 1:
        quats[:, 1].neg_()
        quats[:, 3].neg_()
    else:
        quats[:, 1:3].neg_()

    return GaussianScene(
        means=means,
        sh0=scene.sh0,
        sh=scene.sh,
        opacities=scene.opacities,
        scales=scene.scales,
        quats=quats,
    )


def rotate_scene_y_up_to_z_up(scene: GaussianScene) -> GaussianScene:
    # Rx(+90 deg): (x, y, z) -> (x, -z, y). Converts Inria-style y-up payloads
    # (e.g. PhotoShape) to the z-up convention the renderers assume.
    means = scene.means.clone()
    y = means[:, 1].clone()
    means[:, 1] = -means[:, 2]
    means[:, 2] = y

    quats = scene.quats.clone()
    w = quats[:, 0].clone()
    qx = quats[:, 1].clone()
    qy = quats[:, 2].clone()
    qz = quats[:, 3].clone()
    c = math.sqrt(0.5)
    quats[:, 0] = c * (w - qx)
    quats[:, 1] = c * (qx + w)
    quats[:, 2] = c * (qy - qz)
    quats[:, 3] = c * (qz + qy)

    sh = scene.sh
    if sh is not None:
        if sh.shape[1] != 9:
            raise NotImplementedError(
                f"SH rotation only implemented for degree 1 (9 coeffs), got {sh.shape[1]}."
            )
        # Degree-1 coeffs (m=-1,0,1 blocks of RGB) rotate with the direction:
        # (c_1m1, c_10, c_11) -> (c_10, -c_1m1, c_11) under Rx(+90).
        sh = torch.cat([sh[:, 3:6], -sh[:, 0:3], sh[:, 6:9]], dim=1)

    return GaussianScene(
        means=means,
        sh0=scene.sh0,
        sh=sh,
        opacities=scene.opacities,
        scales=scene.scales,
        quats=quats,
    )


def rotate_gaussian_scene(scene: GaussianScene, degrees: int) -> GaussianScene:
    degrees = int(degrees) % 360
    if degrees not in (90, 180, 270):
        raise ValueError(f"degrees must be one of 90, 180, 270, got {degrees!r}")

    means = scene.means.clone()
    x = means[:, 0].clone()
    y = means[:, 1].clone()
    if degrees == 90:
        means[:, 0] = -y
        means[:, 1] = x
    elif degrees == 180:
        means[:, 0] = -x
        means[:, 1] = -y
    else:
        means[:, 0] = y
        means[:, 1] = -x

    quats = scene.quats.clone()
    w = quats[:, 0].clone()
    qx = quats[:, 1].clone()
    qy = quats[:, 2].clone()
    qz = quats[:, 3].clone()

    if degrees == 180:
        quats[:, 0] = -qz
        quats[:, 1] = -qy
        quats[:, 2] = qx
        quats[:, 3] = w
    else:
        c = math.sqrt(0.5)
        if degrees == 90:
            quats[:, 0] = c * (w - qz)
            quats[:, 1] = c * (qx - qy)
            quats[:, 2] = c * (qy + qx)
            quats[:, 3] = c * (qz + w)
        else:
            quats[:, 0] = c * (w + qz)
            quats[:, 1] = c * (qx + qy)
            quats[:, 2] = c * (qy - qx)
            quats[:, 3] = c * (qz - w)

    return GaussianScene(
        means=means,
        sh0=scene.sh0,
        sh=scene.sh,
        opacities=scene.opacities,
        scales=scene.scales,
        quats=quats,
    )
