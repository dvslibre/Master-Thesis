"""SPECT ray sampling utility."""

from typing import Tuple

import torch


def sample_spect_rays(
    volume_shape: Tuple[int, int, int],
    voxel_size: float | Tuple[float, float, float],
    view: str,
    num_samples: int,
    batch_size: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Uniform SPECT ray sampling along the AP axis.

    Args:
        volume_shape: (D, H, W) = (SI, AP, LR).
        voxel_size: float (isotropic) or tuple (vz, vy, vx) in world units.
        view: 'AP' (rays along +AP) or 'PA' (rays along -AP).
        num_samples: number of points per ray along AP.
        batch_size: optional batch dimension to replicate rays for B elements.
        device/dtype: optional torch device/dtype for outputs.

    Returns:
        xyz_points: (N, 3) or (B, N, 3) sampled world coordinates.
        ray_dirs:   (N, 3) or (B, N, 3) normalized ray directions; (0, +1, 0) for AP, (0, -1, 0) for PA.

    Note:
        N = D * W * num_samples (one ray per (d, w), sampled uniformly along AP), replicated across batch if provided.
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    D, H, W = volume_shape
    if isinstance(voxel_size, (float, int)):
        vz = vy = vx = float(voxel_size)
    else:
        vz, vy, vx = voxel_size

    # World extents centered at origin
    z_coords = (torch.arange(D, device=device, dtype=dtype) - (D - 1) / 2.0) * vz  # SI
    x_coords = (torch.arange(W, device=device, dtype=dtype) - (W - 1) / 2.0) * vx  # LR

    y_min = -(H - 1) / 2.0 * vy
    y_max = (H - 1) / 2.0 * vy
    ap_extent = (H - 1) * vy

    # Grid over (D, W)
    zz, xx = torch.meshgrid(z_coords, x_coords, indexing="ij")
    zz = zz.reshape(-1, 1)  # (D*W, 1)
    xx = xx.reshape(-1, 1)  # (D*W, 1)

    t_vals = torch.linspace(0.0, ap_extent, steps=num_samples, device=device, dtype=dtype)  # (num_samples,)

    if view.upper() == "AP":
        ys = y_min + t_vals  # from negative AP towards positive
        ray_dir = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    elif view.upper() == "PA":
        ys = y_max - t_vals  # from positive AP towards negative
        ray_dir = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=dtype)
    else:
        raise ValueError(f"view must be 'AP' or 'PA', got {view}")

    # Broadcast to all rays
    ys = ys.view(1, -1).expand(zz.shape[0], -1)  # (D*W, num_samples)
    zz_exp = zz.expand(-1, num_samples)          # (D*W, num_samples)
    xx_exp = xx.expand(-1, num_samples)          # (D*W, num_samples)

    # Stack and reshape to (N, 3)
    xyz_points = torch.stack(
        [
            xx_exp.reshape(-1),  # x (LR)
            ys.reshape(-1),      # y (AP)
            zz_exp.reshape(-1),  # z (SI)
        ],
        dim=-1,
    )

    ray_dirs = ray_dir.view(1, 3).expand(xyz_points.shape[0], -1)  # (N, 3)
    if batch_size is not None:
        xyz_points = xyz_points.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        ray_dirs = ray_dirs.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    return xyz_points, ray_dirs
