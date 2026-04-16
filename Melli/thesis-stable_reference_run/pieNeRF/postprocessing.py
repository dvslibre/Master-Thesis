#!/usr/bin/env python3
"""
Postprocessing for NeRF volumetric activity outputs.

This script is strictly post-hoc: it does NOT re-fit projections or influence training.
It targets z-axis artifacts while preserving XY layout. The recommended mode is
`holefill`, which closes short zero-drops between active regions without reducing
peak heights or merging separated plateaus.


Aufruf: 
python3 postprocessing.py \
  --in results_spect/postprocessing/phantom_01/pred_act_step02000.npy \
  --method holefill \
  --z-axis 0 \
  --export-projections \
  --ct data/phantom_01/spect_att.npy \
  --target-ap data/phantom_01/ap.npy \
  --target-pa data/phantom_01/pa.npy \
  --outdir results_spect/postprocessing/phantom_01/proj_withCT


"""

from __future__ import annotations

import argparse
import torch
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np


def load_volume(path: Path, npz_key: Optional[str] = None) -> Tuple[np.ndarray, Optional[str]]:
    """
    Load a volume from .npy, .npz, or .pt (if torch is available).
    Returns (volume, resolved_npz_key).
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path), None
    if suffix == ".npz":
        data = np.load(path)
        if npz_key is None:
            keys = list(data.keys())
            if len(keys) != 1:
                raise ValueError(
                    f"{path} contains multiple arrays {keys}; pass --npz-key to select one."
                )
            npz_key = keys[0]
        if npz_key not in data:
            raise ValueError(f"--npz-key {npz_key!r} not found in {path}.")
        return data[npz_key], npz_key
    if suffix == ".pt":
        try:
            import torch  # pylint: disable=import-error
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("Loading .pt requires torch.") from exc
        tensor = torch.load(path, map_location="cpu")
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        return np.asarray(tensor), None
    raise ValueError(f"Unsupported input format: {path.suffix}")


def save_volume(path: Path, volume: np.ndarray, npz_key: Optional[str] = None) -> None:
    """
    Save a volume as .npy, .npz (with key), or .pt (if torch is available).
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, volume)
        return
    if suffix == ".npz":
        key = npz_key or "volume"
        np.savez(path, **{key: volume})
        return
    if suffix == ".pt":
        try:
            import torch  # pylint: disable=import-error
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("Saving .pt requires torch.") from exc
        torch.save(torch.as_tensor(volume), path)
        return
    raise ValueError(f"Unsupported output format: {path.suffix}")


def _gaussian_kernel1d(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be > 0 for Gaussian smoothing.")
    if radius is None:
        radius = int(3 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def _convolve1d_numpy(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Fallback 1D convolution along axis=0 using NumPy."""
    z_len, n_cols = arr.shape
    pad = len(kernel) // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(arr)
    for i in range(z_len):
        window = padded[i : i + len(kernel), :]
        out[i, :] = np.sum(window * kernel[:, None], axis=0)
    return out


def smooth_along_z(
    volume: np.ndarray,
    sigma_z: float,
    z_axis: int = 0,
) -> np.ndarray:
    """
    Apply 1D Gaussian smoothing along z only (no XY blur).
    This reduces z-lamellae while preserving XY structure.
    """
    arr = np.asarray(volume, dtype=np.float32)
    arr = np.moveaxis(arr, z_axis, 0)
    try:
        from scipy.ndimage import gaussian_filter1d  # type: ignore

        smoothed = gaussian_filter1d(arr, sigma=sigma_z, axis=0, mode="nearest")
    except Exception:
        kernel = _gaussian_kernel1d(sigma_z)
        smoothed = _convolve1d_numpy(arr, kernel)
    smoothed = np.moveaxis(smoothed, 0, z_axis)
    return smoothed


def _suppress_short_active_runs(active: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1:
        return active
    out = active.copy()
    z_len = len(active)
    idx = 0
    while idx < z_len:
        if not active[idx]:
            idx += 1
            continue
        start = idx
        while idx < z_len and active[idx]:
            idx += 1
        if idx - start < min_len:
            out[start:idx] = False
    return out


def _holefill_profile(
    profile: np.ndarray,
    tau: float,
    max_gap: int,
    fill_mode: str,
    min_seg_len: int,
) -> np.ndarray:
    max_val = float(profile.max())
    if max_val <= 0 or max_gap < 1:
        return profile
    threshold = tau * max_val
    active = profile > threshold
    if min_seg_len > 1:
        active = _suppress_short_active_runs(active, min_seg_len)
    if not active.any():
        return profile

    filled = profile.copy()
    z_len = len(profile)
    prev_end = None
    idx = 0
    while idx < z_len:
        if not active[idx]:
            idx += 1
            continue
        start = idx
        while idx < z_len and active[idx]:
            idx += 1
        end = idx - 1
        if prev_end is not None:
            gap_start = prev_end + 1
            gap_end = start - 1
            gap_len = gap_end - gap_start + 1
            if 0 < gap_len <= max_gap:
                left_val = profile[prev_end]
                right_val = profile[start]
                if fill_mode == "minedge":
                    filled[gap_start:start] = min(left_val, right_val)
                else:
                    ramp = np.linspace(
                        left_val, right_val, gap_len + 2, dtype=filled.dtype
                    )[1:-1]
                    filled[gap_start:start] = ramp
        prev_end = end
    return filled


def holefill_along_z(
    volume: np.ndarray,
    z_axis: int = 0,
    tau: float = 0.10,
    max_gap: int = 4,
    fill_mode: str = "minedge",
    min_seg_len: int = 2,
) -> np.ndarray:
    """
    Fill short inactive gaps between active segments along z without smoothing peaks.

    This targets lamella-like zero-drops: only gaps <= max_gap between active segments
    are filled, allowing multiple separated plateaus to remain distinct.
    """
    if tau <= 0:
        raise ValueError("hf_tau must be > 0.")
    if fill_mode not in {"minedge", "linear"}:
        raise ValueError("hf_fill_mode must be 'minedge' or 'linear'.")
    if max_gap < 0:
        raise ValueError("hf_max_gap must be >= 0.")
    if min_seg_len < 1:
        raise ValueError("hf_min_seg_len must be >= 1.")

    arr = np.asarray(volume, dtype=np.float32)
    arr = np.moveaxis(arr, z_axis, 0)
    z_len = arr.shape[0]
    flat = arr.reshape(z_len, -1)
    out = flat.copy()
    for col in range(flat.shape[1]):
        out[:, col] = _holefill_profile(
            flat[:, col],
            tau=tau,
            max_gap=max_gap,
            fill_mode=fill_mode,
            min_seg_len=min_seg_len,
        )
    out = out.reshape(arr.shape)
    return np.moveaxis(out, 0, z_axis)


def _stats(name: str, vol: np.ndarray) -> str:
    return (
        f"{name}: min={vol.min():.4g} max={vol.max():.4g} "
        f"mean={vol.mean():.4g} std={vol.std():.4g}"
    )


def _apply_mask(original: np.ndarray, processed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if mask.shape != original.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match volume {original.shape}.")
    return np.where(mask, processed, original)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocessing: z-only filtering for NeRF volumes (no refit). "
            "Recommended: holefill for lamella-like zero-drops without peak loss."
        )
    )
    parser.add_argument("--in", dest="in_path", required=True, help="Input volume (.npy/.npz/.pt)")
    parser.add_argument("--out", dest="out_path", default=None, help="Output path (.npy/.npz/.pt)")
    parser.add_argument(
        "--method",
        choices=["gauss1d", "holefill"],
        default="gauss1d",
        help="Filtering method (default: gauss1d).",
    )
    parser.add_argument("--sigma-z", type=float, default=1.5, help="Gaussian sigma along z.")
    parser.add_argument(
        "--hf-tau",
        type=float,
        default=0.10,
        help="Holefill threshold as a fraction of per-profile max (default 0.10).",
    )
    parser.add_argument(
        "--hf-max-gap",
        type=int,
        default=4,
        help="Max inactive gap length to fill between active segments.",
    )
    parser.add_argument(
        "--hf-fill-mode",
        choices=["minedge", "linear"],
        default="minedge",
        help="Holefill fill mode: minedge (conservative) or linear ramp.",
    )
    parser.add_argument(
        "--hf-min-seg-len",
        type=int,
        default=2,
        help="Minimum active segment length; shorter runs are ignored.",
    )
    parser.add_argument("--z-axis", type=int, default=0, help="Axis index for z/depth.")
    parser.add_argument("--npz-key", type=str, default=None, help="Key for input .npz.")
    parser.add_argument("--npz-key-out", type=str, default=None, help="Key for output .npz.")
    parser.add_argument(
        "--save-diff",
        action="store_true",
        help="Also save diff volume and print summary stats.",
    )
    parser.add_argument(
        "--use-ct-mask",
        type=str,
        default=None,
        help="Optional CT volume path for a mask (default off).",
    )
    parser.add_argument(
        "--ct-threshold",
        type=float,
        default=0.0,
        help="CT threshold to define mask (ct > threshold).",
    )
    parser.add_argument(
        "--export-profiles",
        action="store_true",
        help="Export depth-profile PNGs along selected rays (optional).",
    )
    parser.add_argument(
        "--export-projections",
        action="store_true",
        help="Render and export AP/PA projections from the processed volume (optional).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for profile/projection PNGs (default: alongside output volume).",
    )
    parser.add_argument(
        "--profile-xy",
        nargs="+",
        default=None,
        help="Optional list of x,y pairs for depth profiles: x1,y1 x2,y2 ...",
    )
    parser.add_argument(
        "--compare-raw",
        type=str,
        default=None,
        help="Optional raw volume path to overlay in profile plots and render raw projections.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML for geometry/data (default: pieNeRF/configs/spect.yaml).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional results_spect run directory for autodiscovery.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Override manifest.csv for targets/CT discovery.",
    )
    parser.add_argument("--ct", type=str, default=None, help="Explicit CT volume path.")
    parser.add_argument("--target-ap", type=str, default=None, help="Explicit AP target path.")
    parser.add_argument("--target-pa", type=str, default=None, help="Explicit PA target path.")
    parser.add_argument(
        "--ray-fg-thr",
        type=str,
        default="0.0",
        help="Foreground threshold for profile selection (number or 'quantile').",
    )
    parser.add_argument(
        "--ray-fg-quantile",
        type=float,
        default=0.2,
        help="Quantile q if --ray-fg-thr=quantile.",
    )
    parser.add_argument(
        "--ray-split-tile",
        type=int,
        default=32,
        help="Tile size for ray split when choosing FG/BG profiles.",
    )
    parser.add_argument(
        "--ray-split-seed",
        type=int,
        default=123,
        help="Seed for ray split when choosing FG/BG profiles.",
    )
    parser.add_argument(
        "--pa-xflip",
        action="store_true",
        help="Flip PA in x-direction to map pixels to AP coordinates (repo default: False).",
    )

    args = parser.parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path) if args.out_path else None

    vol, npz_key = load_volume(in_path, npz_key=args.npz_key)
    vol = np.asarray(vol, dtype=np.float32)

    if out_path is None:
        stem = in_path.stem
        out_path = in_path.with_name(f"{stem}_post_{args.method}.npy")

    if args.method == "gauss1d":
        processed = smooth_along_z(vol, sigma_z=args.sigma_z, z_axis=args.z_axis)
    else:
        processed = holefill_along_z(
            vol,
            z_axis=args.z_axis,
            tau=args.hf_tau,
            max_gap=args.hf_max_gap,
            fill_mode=args.hf_fill_mode,
            min_seg_len=args.hf_min_seg_len,
        )

    if args.use_ct_mask:
        ct_path = Path(args.use_ct_mask)
        ct_vol, _ = load_volume(ct_path, npz_key=None)
        ct_vol = np.asarray(ct_vol, dtype=np.float32)
        ct_mask = ct_vol > float(args.ct_threshold)
        processed = _apply_mask(vol, processed, ct_mask)

    save_volume(out_path, processed, npz_key=args.npz_key_out or npz_key)

    if args.save_diff:
        diff = processed - vol
        diff_path = out_path.with_name(f"{out_path.stem}_diff.npy")
        np.save(diff_path, diff)
        print(_stats("before", vol))
        print(_stats("after", processed))
        print(f"mean(abs(diff))={np.mean(np.abs(diff)):.4g}")

    if args.export_profiles or args.export_projections:
        _export_artifacts(
            processed_volume=processed,
            out_path=out_path,
            args=args,
        )


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r") as f:
        return yaml.safe_load(f)


def _resolve_config_path(config_path: Optional[str]) -> Path:
    if config_path:
        return Path(config_path).expanduser().resolve()
    return (Path(__file__).resolve().parent / "configs" / "spect.yaml").resolve()


def _resolve_manifest_path(
    config: Optional[Dict[str, Any]],
    manifest_override: Optional[str],
    config_path: Path,
) -> Optional[Path]:
    if manifest_override:
        return Path(manifest_override).expanduser().resolve()
    if config and "data" in config and config["data"].get("manifest"):
        manifest_path = Path(config["data"]["manifest"])
        if not manifest_path.is_absolute():
            manifest_path = (config_path.parent / manifest_path).resolve()
        return manifest_path
    return None


def _load_manifest_entries(manifest_path: Path) -> List[Dict[str, Path]]:
    import csv

    entries: List[Dict[str, Path]] = []
    base_dir = manifest_path.parent
    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def _resolve(p: Optional[str]) -> Optional[Path]:
                if not p:
                    return None
                path = Path(p)
                if not path.is_absolute():
                    path = (base_dir / path).resolve()
                return path

            entries.append(
                {
                    "patient_id": row.get("patient_id", ""),
                    "ap_path": _resolve(row.get("ap_path")),
                    "pa_path": _resolve(row.get("pa_path")),
                    "ct_path": _resolve(row.get("ct_path")),
                    "act_path": _resolve(row.get("act_path")) if row.get("act_path") else None,
                }
            )
    return entries


def _match_entry_for_volume(in_path: Path, entries: List[Dict[str, Path]]) -> Optional[Dict[str, Path]]:
    if not entries:
        return None
    in_resolved = in_path.expanduser().resolve()
    in_stem = in_resolved.stem
    exact_match = None
    stem_match = None
    for entry in entries:
        act_path = entry.get("act_path")
        if act_path is None:
            continue
        act_resolved = act_path.expanduser().resolve()
        if in_resolved == act_resolved:
            exact_match = entry
            break
        if act_resolved.stem and act_resolved.stem in in_stem:
            stem_match = entry
    return exact_match or stem_match


def _parse_profile_xy(values: Optional[List[str]]) -> Optional[List[Tuple[int, int]]]:
    if not values:
        return None
    coords = []
    for item in values:
        if "," not in item:
            raise ValueError(f"Invalid --profile-xy entry {item!r}, expected x,y.")
        x_str, y_str = item.split(",", 1)
        coords.append((int(y_str), int(x_str)))
    return coords


def _parse_fg_threshold(raw_thr: str, quantile: float) -> float:
    val = str(raw_thr).strip().lower()
    if val == "quantile":
        return -abs(float(quantile))
    return float(val)


def _resolve_aux_inputs(
    in_path: Path,
    args,
) -> Dict[str, Optional[Path]]:
    config_path = _resolve_config_path(args.config)
    config = None
    if config_path.exists():
        config = _load_yaml(config_path)
    elif args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        for candidate in ("config.yaml", "config.yml"):
            cand_path = run_dir / candidate
            if cand_path.exists():
                config_path = cand_path
                config = _load_yaml(cand_path)
                break
    elif not args.config:
        for parent in [in_path] + list(in_path.parents):
            if parent.name == "results_spect":
                run_dir = in_path.parent
                for candidate in ("config.yaml", "config.yml"):
                    cand_path = run_dir / candidate
                    if cand_path.exists():
                        config_path = cand_path
                        config = _load_yaml(cand_path)
                        break
            if config is not None:
                break

    manifest_path = _resolve_manifest_path(config, args.manifest, config_path) if (args.export_profiles or args.export_projections) else None
    entries = _load_manifest_entries(manifest_path) if manifest_path and manifest_path.exists() else []
    match = _match_entry_for_volume(in_path, entries) if entries else None

    ct_path = Path(args.ct).expanduser().resolve() if args.ct else None
    target_ap = Path(args.target_ap).expanduser().resolve() if args.target_ap else None
    target_pa = Path(args.target_pa).expanduser().resolve() if args.target_pa else None

    if match:
        ct_path = ct_path or match.get("ct_path")
        target_ap = target_ap or match.get("ap_path")
        target_pa = target_pa or match.get("pa_path")

    return {
        "config": config,
        "config_path": config_path,
        "manifest_path": manifest_path,
        "ct_path": ct_path,
        "target_ap": target_ap,
        "target_pa": target_pa,
    }


def _save_png(arr: np.ndarray, path: Path, title: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt

    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        a = a[None, :]
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    # robust contrast: percentile scaling (prevents "only contour" look)
    p1, p99 = np.percentile(a, [1, 99])
    if np.isclose(p1, p99):
        vmin, vmax = float(a.min()), float(a.max())
    else:
        vmin, vmax = float(p1), float(p99)

    if np.isclose(vmin, vmax):
        img = np.zeros_like(a, dtype=np.float32)
    else:
        img = np.clip((a - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)

    plt.figure(figsize=(6, 4))
    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

 


def _build_grid_context(volume: np.ndarray, radius: float) -> Dict[str, Any]:
    import torch

    if radius <= 0:
        raise ValueError("grid_radius must be > 0 for volume sampling.")
    vol = torch.from_numpy(volume).float()
    if vol.ndim != 3:
        raise ValueError(f"Expected volume with D,H,W axes, got {vol.shape}.")
    vol = vol.unsqueeze(0).unsqueeze(0)
    ctx = {
        "volume": vol,
        "grid_radius": float(radius),
    }
    if vol.numel() > 0:
        ctx["value_range"] = (float(vol.min().item()), float(vol.max().item()))
    return ctx


def _render_projection_from_volume(
    volume: np.ndarray,
    ct_volume: Optional[np.ndarray],
    config: Dict[str, Any],
    pose: "torch.Tensor",
    H: int,
    W: int,
) -> np.ndarray:
    import torch
    from nerf.run_nerf_mod import sample_ct_volume
    from nerf.run_nerf_helpers_mod import get_rays, get_rays_ortho

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nerf_cfg = config.get("nerf", {})
    data_cfg = config.get("data", {})
    N_samples = int(nerf_cfg.get("N_samples", 96))
    near = float(data_cfg.get("near", 0.0))
    far = float(data_cfg.get("far", 1.0))
    use_attenuation = bool(nerf_cfg.get("use_attenuation", False))
    orthographic = bool(data_cfg.get("orthographic", True))
    fov = float(data_cfg.get("fov", 60.0))
    radius = data_cfg.get("radius", 0.5)
    if isinstance(radius, str):
        radius_val = max(float(r) for r in radius.split(","))
    elif isinstance(radius, (tuple, list)):
        radius_val = float(max(radius))
    else:
        radius_val = float(radius)

    if orthographic:
        size = 2.0 * radius_val
        rays_o, rays_d = get_rays_ortho(H, W, pose, size, size)
    else:
        focal = W / 2 * 1.0 / np.tan(0.5 * fov * np.pi / 180.0)
        rays_o, rays_d = get_rays(H, W, focal, pose)

    rays_o = rays_o.reshape(-1, 3).to(device)
    rays_d = rays_d.reshape(-1, 3).to(device)

    t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand(rays_o.shape[0], N_samples)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    act_ctx = _build_grid_context(volume, radius_val)
    act_ctx["volume"] = act_ctx["volume"].to(device)
    lambda_vals = sample_ct_volume(pts, act_ctx)
    if lambda_vals is None:
        raise RuntimeError("Failed to sample activity volume along rays.")
    if lambda_vals.ndim == 3 and lambda_vals.shape[-1] == 1:
        lambda_vals = lambda_vals[..., 0]
    lambda_vals = torch.clamp(lambda_vals, min=0.0)

    mu_vals = None
    if use_attenuation and ct_volume is not None:
        ct_ctx = _build_grid_context(ct_volume, radius_val)
        ct_ctx["volume"] = ct_ctx["volume"].to(device)
        mu_vals = sample_ct_volume(pts, ct_ctx)
        if mu_vals is not None and mu_vals.ndim == 3 and mu_vals.shape[-1] == 1:
            mu_vals = mu_vals[..., 0]

    with torch.no_grad():
        # step size must be positive
        ds = (z_vals[:, 1] - z_vals[:, 0]).abs()

        if mu_vals is not None and use_attenuation:
            mu_vals = torch.clamp(mu_vals, min=0.0)
            tau = torch.cumsum(mu_vals * ds[:, None], dim=1)
            tau_prev = torch.cat([torch.zeros_like(tau[:, :1]), tau[:, :-1]], dim=1)
            weights = torch.exp(-tau_prev)
        else:
            weights = torch.ones_like(lambda_vals)

        proj_map = torch.sum(lambda_vals * weights * ds[:, None], dim=1)
        if proj_map.ndim != 1 or proj_map.numel() != H * W:
            raise RuntimeError(
                f"proj_map shape {tuple(proj_map.shape)} (numel={proj_map.numel()}) "
                f"does not match H*W={H*W}. H={H}, W={W}"
            )

    proj = proj_map.reshape(H, W).detach().cpu().numpy()
    return proj


def _select_profile_coords(
    score_ap: Optional[np.ndarray],
    score_pa: Optional[np.ndarray],
    score_fallback: np.ndarray,
    args,
) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(int(args.ray_split_seed))
    H, W = score_fallback.shape
    if score_ap is not None and score_pa is not None:
        try:
            from pieNeRF.utils.ray_split import make_pixel_split_from_ap_pa
        except ModuleNotFoundError:
            from utils.ray_split import make_pixel_split_from_ap_pa

        thr_val = _parse_fg_threshold(args.ray_fg_thr, args.ray_fg_quantile)
        split = make_pixel_split_from_ap_pa(
            score_ap,
            score_pa,
            train_frac=0.8,
            tile=int(args.ray_split_tile),
            thr=thr_val,
            seed=int(args.ray_split_seed),
            pa_xflip=bool(args.pa_xflip),
        )
        fg_pool = split.train_idx_fg if split.train_idx_fg.size > 0 else split.test_idx_fg
        bg_pool = split.train_idx_bg if split.train_idx_bg.size > 0 else split.test_idx_bg
        coords: List[Tuple[int, int]] = []
        for pool, label in ((fg_pool, "fg"), (bg_pool, "bg")):
            if pool.size == 0:
                continue
            picks = rng.choice(pool, size=min(2, pool.size), replace=False)
            for idx in picks:
                y = int(idx // W)
                x = int(idx % W)
                coords.append((y, x))
        while len(coords) < 4:
            idx = int(rng.integers(0, H * W))
            coords.append((idx // W, idx % W))
        return coords[:4]

    flat = score_fallback.reshape(-1)
    top_idx = np.argsort(-flat)[:2]
    bot_idx = np.argsort(flat)[:2]
    coords = [(int(i // W), int(i % W)) for i in np.concatenate([top_idx, bot_idx])]
    return coords[:4]


def _export_profiles(
    processed: np.ndarray,
    raw: Optional[np.ndarray],
    coords: List[Tuple[int, int]],
    outdir: Path,
    target_ap: Optional[np.ndarray],
    target_pa: Optional[np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    depth = np.arange(processed.shape[0])
    for y, x in coords:
        if not (0 <= y < processed.shape[1] and 0 <= x < processed.shape[2]):
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        post_curve = processed[:, y, x]
        ax.plot(depth, post_curve, label="post")
        if raw is not None:
            raw_curve = raw[:, y, x]
            ax.plot(depth, raw_curve, label="raw")
        title_bits = [f"x={x}, y={y}"]
        if target_ap is not None and 0 <= y < target_ap.shape[0] and 0 <= x < target_ap.shape[1]:
            title_bits.append(f"I_AP={target_ap[y, x]:.2e}")
        if target_pa is not None and 0 <= y < target_pa.shape[0] and 0 <= x < target_pa.shape[1]:
            title_bits.append(f"I_PA={target_pa[y, x]:.2e}")
        ax.set_title(" | ".join(title_bits))
        ax.set_xlabel("z index")
        ax.set_ylabel("activity")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)
        fname = outdir / f"profiles_xy_{x}_{y}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)


def _export_artifacts(processed_volume: np.ndarray, out_path: Path, args) -> None:
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else out_path.parent.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    in_path = Path(args.in_path).expanduser().resolve()

    aux = _resolve_aux_inputs(in_path, args)
    config = aux["config"] or {}
    ct_path = aux["ct_path"]
    target_ap_path = aux["target_ap"]
    target_pa_path = aux["target_pa"]

    def _as_depth_first(vol: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if vol is None:
            return None
        if args.z_axis == 0:
            return vol
        return np.moveaxis(vol, args.z_axis, 0)

    ct_vol = None
    if ct_path is not None and ct_path.exists():
        ct_vol_np, _ = load_volume(ct_path, npz_key=None)
        ct_vol = np.asarray(ct_vol_np, dtype=np.float32)
        ct_vol = _as_depth_first(ct_vol)
    elif args.export_projections and config.get("nerf", {}).get("use_attenuation", False):
        print(
            "Warning: CT volume not found for attenuation; provide --ct or --manifest to enable attenuation."
        )

    target_ap = None
    target_pa = None
    if target_ap_path is not None and target_ap_path.exists():
        target_ap_np, _ = load_volume(target_ap_path, npz_key=None)
        target_ap = np.asarray(target_ap_np, dtype=np.float32)
    if target_pa_path is not None and target_pa_path.exists():
        target_pa_np, _ = load_volume(target_pa_path, npz_key=None)
        target_pa = np.asarray(target_pa_np, dtype=np.float32)
    if (args.export_profiles or args.export_projections) and target_ap is None and target_pa is None:
        print(
            "Warning: target projections not found; provide --target-ap/--target-pa or --manifest to enable target outputs."
        )

    compare_raw = None
    if args.compare_raw:
        raw_path = Path(args.compare_raw).expanduser().resolve()
        compare_raw, _ = load_volume(raw_path, npz_key=None)
        compare_raw = np.asarray(compare_raw, dtype=np.float32)

    processed_for_render = _as_depth_first(processed_volume)
    compare_raw_for_render = _as_depth_first(compare_raw)

    profile_coords = _parse_profile_xy(args.profile_xy)
    if profile_coords is None:
        fallback_score = processed_for_render.max(axis=0)
        if target_ap is not None and target_ap.shape != fallback_score.shape:
            target_ap = None
        if target_pa is not None and target_pa.shape != fallback_score.shape:
            target_pa = None
        profile_coords = _select_profile_coords(target_ap, target_pa, fallback_score, args)

    if args.export_profiles:
        _export_profiles(
            processed=processed_for_render,
            raw=compare_raw_for_render,
            coords=profile_coords,
            outdir=outdir,
            target_ap=target_ap,
            target_pa=target_pa,
        )

    if args.export_projections:
        if not config:
            raise ValueError(
                "Missing config for rendering. Provide --config or ensure pieNeRF/configs/spect.yaml exists."
            )
        from graf.generator import _pose_from_loc
        import torch

        data_cfg = config.get("data", {})
        radius = data_cfg.get("radius", 0.5)
        if isinstance(radius, str):
            radius_val = max(float(r) for r in radius.split(","))
        elif isinstance(radius, (tuple, list)):
            radius_val = float(max(radius))
        else:
            radius_val = float(radius)
        loc_ap = np.array([0.0, 0.0, +radius_val], dtype=np.float32)
        loc_pa = np.array([0.0, 0.0, -radius_val], dtype=np.float32)
        pose_ap = _pose_from_loc(loc_ap)
        pose_pa = _pose_from_loc(loc_pa)
        # Optional PA x-flip to match alternate handedness conventions.
        if args.pa_xflip:
            pose_pa[:, 0] *= -1.0

        if target_ap is not None:
            H, W = target_ap.shape
        elif target_pa is not None:
            H, W = target_pa.shape
        else:
            H, W = processed_volume.shape[1], processed_volume.shape[2]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pose_ap = pose_ap.to(device)
        pose_pa = pose_pa.to(device)

        proj_post_ap = _render_projection_from_volume(processed_for_render, ct_vol, config, pose_ap, H, W)
        proj_post_pa = _render_projection_from_volume(processed_for_render, ct_vol, config, pose_pa, H, W)
        _save_png(proj_post_ap, outdir / "proj_post_AP.png", title="post AP")
        _save_png(proj_post_pa, outdir / "proj_post_PA.png", title="post PA")

        if compare_raw_for_render is not None:
            proj_raw_ap = _render_projection_from_volume(compare_raw_for_render, ct_vol, config, pose_ap, H, W)
            proj_raw_pa = _render_projection_from_volume(compare_raw_for_render, ct_vol, config, pose_pa, H, W)
            _save_png(proj_raw_ap, outdir / "proj_raw_AP.png", title="raw AP")
            _save_png(proj_raw_pa, outdir / "proj_raw_PA.png", title="raw PA")

        if target_ap is not None:
            _save_png(target_ap, outdir / "proj_target_AP.png", title="target AP")
        if target_pa is not None:
            _save_png(target_pa, outdir / "proj_target_PA.png", title="target PA")

        if target_ap is not None:
            _save_png(proj_post_ap - target_ap, outdir / "proj_diff_AP.png", title="post - target AP")
        if target_pa is not None:
            _save_png(proj_post_pa - target_pa, outdir / "proj_diff_PA.png", title="post - target PA")


if __name__ == "__main__":
    main()
