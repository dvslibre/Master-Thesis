#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qplanar.py (Full-Forward variant)

Model-based planar quantification (QPlanar-like) on synthetic phantom data.
- Uses meta_simple.json for ground-truth organ activities (kBq/mL) and organ IDs.
- Builds system matrix A by forward-projecting unit-concentration organ masks (1.0 kBq/mL).
- Builds measurement vector b via Full-Forward projection of the full GT activity volume.
- Solves Ax ≈ b with NNLS (nonnegative least squares).

Key physics:
- Scatter: 2D Gaussian per slice (xy)
- Attenuation: exp(-cumsum(mu * Δz)) with explicit step length in cm
- Collimator blur: depth-dependent 2D convolution with kernel_mat[:,:,depth]
  (kernel is normalized to sum=1 per depth slice)
- Projection: z-sum
- AP/PA orientation matches Stratos conventions.

Example:
python3 qplanar.py \
  --base /home/mnguest12/projects/thesis/PhantomGenerator/phantom_01 \
  --spect_bin phantom_01_spect208keV.par_atn_1.bin \
  --mask_bin phantom_01_mask.par_act_1.bin \
  --meta_json /home/mnguest12/projects/thesis/PhantomGenerator/phantom_01/meta_simple.json \
  --kernel_mat /home/mnguest12/projects/thesis/PhantomGenerator/LEAP_Kernel.mat \
  --shape 256,256,651 \
  --pixel_size_mm 1.5 \
  --save_pngs \
  --out_dir qplanar_results \
  --poisson --counts_per_pixel 20000
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Any, List

import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.signal import fftconvolve
from scipy.optimize import nnls

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------------------

def load_bin_xyz(path: Path, shape_str: str, dtype: str = "float32", order: str = "F") -> np.ndarray:
    """Load a raw .bin volume as (x,y,z) in Fortran order by default."""
    x, y, z = [int(s) for s in shape_str.split(",")]
    arr = np.fromfile(path, dtype=np.dtype(dtype))
    expected = x * y * z
    if arr.size != expected:
        raise ValueError(f"{path}: size {arr.size} != {expected} for shape {(x,y,z)}")
    vol = arr.reshape((x, y, z), order=order)
    return vol.astype(np.float32, copy=False)


def convert_mu_units(mu_xyz: np.ndarray, src_unit: str, tgt_unit: str) -> np.ndarray:
    """Convert attenuation coefficients between 1/mm and 1/cm."""
    if src_unit == tgt_unit:
        return mu_xyz
    if src_unit == "per_mm" and tgt_unit == "per_cm":
        return mu_xyz * 10.0
    if src_unit == "per_cm" and tgt_unit == "per_mm":
        return mu_xyz / 10.0
    raise ValueError(f"Unknown conversion: {src_unit} -> {tgt_unit}")


def resolve_path(base: Path, src_dir: Path, p: Path) -> Path:
    """Resolve relative paths against a few likely roots."""
    if p.is_absolute():
        return p
    if p.exists():
        return p
    if (src_dir / p).exists():
        return src_dir / p
    if (base / p).exists():
        return base / p
    script_dir = Path(__file__).resolve().parent
    if (script_dir / p).exists():
        return script_dir / p
    if (script_dir / p.name).exists():
        return script_dir / p.name
    return p


# ------------------------------------------------------------------------------
# Kernel loading + normalization
# ------------------------------------------------------------------------------

def load_kernel_mat(mat_path: Path, key: str = None) -> np.ndarray:
    """
    Load kernel_mat from .mat file.
    If key is None, auto-detect the first non-__ variable.
    """
    data = sio.loadmat(mat_path)
    keys = [k for k in data.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError(f"No variables found in MAT file: {mat_path}")
    use_key = key if key is not None else keys[0]
    K = data[use_key].astype(np.float32)
    if K.ndim != 3:
        raise ValueError(f"kernel_mat must be 3D, got shape {K.shape} (key={use_key})")
    return K


def normalize_kernel_slices(kernel: np.ndarray) -> np.ndarray:
    """
    Normalize each depth slice kernel[:,:,d] so that sum = 1.
    This makes the kernel a pure PSF (redistribution), not an efficiency factor.
    """
    K = kernel.astype(np.float32, copy=True)
    sums = K.sum(axis=(0, 1), keepdims=True)  # shape (1,1,D)
    sums[sums == 0] = 1.0
    K /= sums
    return K


# ------------------------------------------------------------------------------
# Physics model (Stratos-like)
# ------------------------------------------------------------------------------

def _process_view_phys(
    act_data: np.ndarray,
    mu_data: np.ndarray,
    kernel_mat: np.ndarray,
    sigma: float,
    z0_slices: int,
    step_len_cm: float,
) -> np.ndarray:
    """
    One view physics:
    1) Scatter (2D Gaussian per z-slice)
    2) Attenuation along z: exp(-cumsum(mu * step_len))
    3) Collimator blur: depth-dependent conv2 (from z0+1)
    4) Projection: sum over z
    """
    # 1) Scatter per slice (linear)
    if sigma > 0:
        act_sc = np.empty_like(act_data, dtype=np.float32)
        for z in range(act_data.shape[2]):
            act_sc[:, :, z] = gaussian_filter(act_data[:, :, z], sigma=sigma, mode="nearest")
    else:
        act_sc = act_data.astype(np.float32, copy=False)

    # 2) Attenuation along projection direction (z)
    # mu_data should be in 1/cm, step_len_cm in cm
    mu_cum = np.cumsum(mu_data * step_len_cm, axis=2)
    vol_atn = act_sc * np.exp(-mu_cum).astype(np.float32)

    # 3) Collimator blur (depth PSF)
    Z = vol_atn.shape[2]
    vol_coll = np.zeros_like(vol_atn, dtype=np.float32)

    for z in range(Z):
        if z <= z0_slices:
            vol_coll[:, :, z] = vol_atn[:, :, z]
        else:
            zz = min(z - z0_slices, kernel_mat.shape[2] - 1)
            K = kernel_mat[:, :, zz]  # already normalized slice-wise
            vol_coll[:, :, z] = fftconvolve(vol_atn[:, :, z], K, mode="same").astype(np.float32)

    # 4) z-summation
    return np.sum(vol_coll, axis=2)


def gamma_camera_core(
    act_xyz: np.ndarray,
    mu_xyz: np.ndarray,
    kernel_mat: np.ndarray,
    sigma: float,
    z0_slices: int,
    step_len_cm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce AP and PA projections with the same orientation convention as Stratos.
    Input volumes are assumed to be (x,y,z).
    """
    # Stratos/MATLAB-style pre-rotations
    act_ap = np.transpose(act_xyz, (0, 2, 1))
    mu_ap  = np.transpose(mu_xyz,  (0, 2, 1))

    act_ap = np.rot90(act_ap)
    mu_ap  = np.rot90(mu_ap)

    act_ap = np.transpose(act_ap, (1, 0, 2))
    mu_ap  = np.transpose(mu_ap,  (1, 0, 2))

    act_pa = np.flip(act_ap, axis=2)
    mu_pa  = np.flip(mu_ap,  axis=2)

    proj_ap_raw = _process_view_phys(act_ap, mu_ap, kernel_mat, sigma, z0_slices, step_len_cm)
    proj_pa_raw = _process_view_phys(act_pa, mu_pa, kernel_mat, sigma, z0_slices, step_len_cm)

    # Orientation patch: rot90 -> flipud -> transpose
    def orient_patch(P: np.ndarray) -> np.ndarray:
        return np.flipud(np.rot90(P)).T

    return orient_patch(proj_ap_raw), orient_patch(proj_pa_raw)


# ------------------------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------------------------

def save_png(arr: np.ndarray, path: Path, title: str = None) -> None:
    """Render the projection with a 90° rotation (counter-clockwise) while keeping axes/labels unchanged."""
    plt.figure(figsize=(6, 6))
    rotated = np.rot90(arr, k=1)
    plt.imshow(rotated, cmap="inferno", origin="lower")
    plt.colorbar(label="Intensity")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ------------------------------------------------------------------------------
# Mask perturbations
# ------------------------------------------------------------------------------

def dilate_labels_radius1_xy(mask_ids: np.ndarray, label_ids: List[int]) -> np.ndarray:
    """
    Cheap 2D dilation (radius 1 voxel, 3x3 neighborhood) per organ label and z-slice.
    No cross-slice growth in z-direction.
    Background stays 0.
    Overlap resolution is deterministic: higher label ID wins.
    """
    out = np.zeros_like(mask_ids, dtype=np.int64)
    structure_xy = np.ones((3, 3), dtype=bool)

    # Higher organ ID wins in overlaps: process ascending and let later writes overwrite.
    for lid in sorted(label_ids):
        organ_mask = (mask_ids == lid)
        if not np.any(organ_mask):
            continue
        for z in range(mask_ids.shape[2]):
            if not np.any(organ_mask[:, :, z]):
                continue
            dil_xy = binary_dilation(organ_mask[:, :, z], structure=structure_xy, iterations=1)
            out[:, :, z][dil_xy] = lid
    return out


def shift_mask_zero_pad(mask_ids: np.ndarray, shift_xyz: Tuple[int, int, int]) -> np.ndarray:
    """
    Shift complete label volume by integer voxels (dx,dy,dz) with zero-padding.
    No wrap-around.
    """
    dx, dy, dz = shift_xyz
    out = np.zeros_like(mask_ids, dtype=np.int64)

    def _src_dst(n: int, d: int) -> Tuple[slice, slice]:
        if d >= 0:
            src = slice(0, n - d)
            dst = slice(d, n)
        else:
            src = slice(-d, n)
            dst = slice(0, n + d)
        return src, dst

    sx, dxs = _src_dst(mask_ids.shape[0], dx)
    sy, dys = _src_dst(mask_ids.shape[1], dy)
    sz, dzs = _src_dst(mask_ids.shape[2], dz)

    if (sx.stop - sx.start) <= 0 or (sy.stop - sy.start) <= 0 or (sz.stop - sz.start) <= 0:
        return out

    out[dxs, dys, dzs] = mask_ids[sx, sy, sz]
    return out


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QPlanar-like quantification (Full-Forward b)")
    parser.add_argument("--base", type=Path, required=True, help="Phantom base folder")
    parser.add_argument("--spect_bin", type=Path, required=True, help="Attenuation BIN (SPECT energy)")
    parser.add_argument("--mask_bin", type=Path, required=True, help="Mask BIN (organ IDs)")
    parser.add_argument("--meta_json", type=Path, required=True, help="meta_simple.json path")
    parser.add_argument("--kernel_mat", type=Path, required=True, help="LEAP_Kernel.mat path")
    parser.add_argument("--kernel_key", type=str, default=None, help="Optional: variable name in .mat (default: auto-detect)")

    parser.add_argument("--shape", type=str, required=True, help="Shape x,y,z, e.g. 256,256,651")
    parser.add_argument("--pixel_size_mm", type=float, default=1.5, help="Voxel size / z-spacing in mm")

    parser.add_argument("--psf_sigma", type=float, default=2.0, help="Scatter sigma (pixels)")
    parser.add_argument("--z0_slices", type=int, default=29, help="Number of near-detector slices without collimator blur")
    parser.add_argument("--mu_unit_in", type=str, default="per_mm", choices=["per_mm", "per_cm"], help="Unit of mu in BIN")

    parser.add_argument("--poisson", action="store_true", help="Apply Poisson noise to b")
    parser.add_argument("--counts_per_pixel", type=float, default=2e4, help="Typical count level (scales noise)")

    parser.add_argument("--save_pngs", action="store_true", help="Save projections as PNG")
    parser.add_argument("--out_dir", type=str, default="qplanar_results", help="Output directory")
    parser.add_argument(
        "--exclude_organs",
        type=str,
        default="small_intest",
        help="Comma-separated organ names to exclude completely from modeling/evaluation (default: small_intest)",
    )
    parser.add_argument(
        "--mask-robustness-suite",
        action="store_true",
        help="Run baseline + 2D-slice dilation(r=1) + shifts (x=1cm and x=2cm) sequentially",
    )

    # optional debug checks
    parser.add_argument("--debug_consistency", action="store_true", help="Print ||b - A@x_gt|| / ||b||")

    args = parser.parse_args()

    base = args.base
    src_dir = base / "src"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spect_path = resolve_path(base, src_dir, args.spect_bin)
    mask_path  = resolve_path(base, src_dir, args.mask_bin)
    meta_path  = resolve_path(base, src_dir, args.meta_json)
    kern_path  = resolve_path(base, src_dir, args.kernel_mat)

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta JSON not found: {meta_path}")
    if not spect_path.exists():
        raise FileNotFoundError(f"SPECT mu BIN not found: {spect_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask BIN not found: {mask_path}")
    if not kern_path.exists():
        raise FileNotFoundError(f"Kernel MAT not found: {kern_path}")

    print(f"[INFO] Phantom: {base.name}")
    print(f"[INFO] mask:   {mask_path}")
    print(f"[INFO] mu:     {spect_path}")
    print(f"[INFO] meta:   {meta_path}")
    print(f"[INFO] kernel: {kern_path}")

    # 1) Load meta
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    organ_info: Dict[str, Any] = meta.get("organ_activity_info", {})
    if not organ_info:
        raise ValueError("meta_simple.json has no 'organ_activity_info'")

    # 2) Load volumes
    mask_vol = load_bin_xyz(mask_path, args.shape, dtype="float32")
    mu_vol   = load_bin_xyz(spect_path, args.shape, dtype="float32")
    mu_vol   = convert_mu_units(mu_vol, args.mu_unit_in, "per_cm")

    # mask IDs should be integer-like
    mask_ids = np.rint(mask_vol).astype(np.int64)

    # Optional organ exclusion applied globally (system matrix, b, reporting, perturbations).
    excluded_organs = {s.strip() for s in args.exclude_organs.split(",") if s.strip()}
    excluded_label_ids: List[int] = []
    for name in sorted(excluded_organs):
        if name in organ_info:
            excluded_label_ids.append(int(organ_info[name]["organ_id"]))
    if excluded_label_ids:
        for oid in excluded_label_ids:
            mask_ids[mask_ids == oid] = 0
        print(f"[INFO] Excluding organs: {sorted(excluded_organs)}")
    else:
        print("[INFO] Excluding organs: []")

    # 3) Load and normalize kernel (sum=1 per depth)
    kernel_raw = load_kernel_mat(kern_path, key=args.kernel_key)
    kernel_mat = normalize_kernel_slices(kernel_raw)
    print(f"[INFO] Kernel normalized slice-wise to sum=1. Shape={kernel_mat.shape}")

    # 4) Geometry / units
    vox_cm = args.pixel_size_mm / 10.0
    voxel_ml = vox_cm ** 3
    step_len_cm = vox_cm

    # 5) Organ list
    organ_names: List[str] = sorted([k for k in organ_info.keys() if k not in excluded_organs])
    n_org = len(organ_names)
    if n_org == 0:
        raise ValueError("No organs left for quantification after exclusion.")
    all_nonzero_label_ids = np.unique(mask_ids[mask_ids != 0]).astype(np.int64).tolist()
    print(f"[INFO] #Organs in meta: {n_org}")

    # 6) Determine projection size via dummy run
    dummy = np.zeros_like(mu_vol, dtype=np.float32)
    d_ap, d_pa = gamma_camera_core(dummy, mu_vol, kernel_mat, args.psf_sigma, args.z0_slices, step_len_cm)
    H, W = d_ap.shape
    n_pix = H * W
    print(f"[INFO] Projection size: {H} x {W}  (n_pix={n_pix})")

    def run_quantification(
        run_name: str,
        run_mask_ids: np.ndarray,
        run_out_dir: Path,
        measurement_mask_ids: np.ndarray = None,
    ) -> Dict[str, Any]:
        run_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[RUN] {run_name}")
        if measurement_mask_ids is None:
            measurement_mask_ids = run_mask_ids

        # 7) Build A and x_gt (unit concentration per organ)
        A = np.zeros((2 * n_pix, n_org), dtype=np.float64)
        x_gt = np.zeros((n_org,), dtype=np.float64)

        # Each column corresponds to 1.0 kBq/mL concentration in that organ,
        # converted to MBq/voxel: 1.0 kBq/mL * 1e-3 (MBq/kBq) * voxel_ml (mL/voxel)
        unit_mbq_per_voxel = 1.0 * 1e-3 * voxel_ml

        print("[INFO] Building system matrix A (unit organ projections)...")
        for i, organ in enumerate(organ_names):
            oid = int(organ_info[organ]["organ_id"])
            gt_conc = float(organ_info[organ]["assigned_value_kBq_per_ml"])  # kBq/mL
            x_gt[i] = gt_conc

            act_unit = np.zeros_like(mu_vol, dtype=np.float32)
            act_unit[run_mask_ids == oid] = unit_mbq_per_voxel

            ap, pa = gamma_camera_core(act_unit, mu_vol, kernel_mat, args.psf_sigma, args.z0_slices, step_len_cm)

            A[:n_pix, i] = ap.ravel()
            A[n_pix:, i] = pa.ravel()

        # 8) Build b via Full-Forward projection of full GT activity volume
        print("[INFO] Building measurement b via Full-Forward projection of GT activity volume...")
        act_full = np.zeros_like(mu_vol, dtype=np.float32)
        for i, organ in enumerate(organ_names):
            oid = int(organ_info[organ]["organ_id"])
            gt_conc = x_gt[i]  # kBq/mL
            act_full[measurement_mask_ids == oid] = (gt_conc * 1e-3 * voxel_ml)  # MBq/voxel

        b_ap_img, b_pa_img = gamma_camera_core(act_full, mu_vol, kernel_mat, args.psf_sigma, args.z0_slices, step_len_cm)
        b = np.concatenate([b_ap_img.ravel(), b_pa_img.ravel()]).astype(np.float64)

        # Optional debug: check linear consistency
        if args.debug_consistency:
            rel = np.linalg.norm(b - (A @ x_gt)) / (np.linalg.norm(b) + 1e-12)
            print(f"[DEBUG] Consistency ||b - A@x_gt|| / ||b|| = {rel:.6e}")

        # 9) Optional Poisson noise on b
        if args.poisson:
            # Interpret b as "activity-equivalent" and map to a count regime.
            # Simple and stable: scale so that a reference level corresponds to counts_per_pixel.
            # Here we use max(b) as reference; you can switch to percentile if desired.
            b_max = float(np.max(b))
            if b_max <= 0:
                print("[WARN] b.max() <= 0, skipping Poisson.")
            else:
                scale = float(args.counts_per_pixel) / (b_max + 1e-12)
                b_counts = np.clip(b * scale, 0, None)
                b_noisy = np.random.poisson(b_counts).astype(np.float64) / scale
                b = b_noisy
                print(f"[INFO] Poisson noise applied (counts_per_pixel≈{args.counts_per_pixel:.0f})")

        # 10) Solve NNLS
        scaling_factor = 1e5
        print(f"[INFO] Solving NNLS: min ||A x - b|| s.t. x>=0 (scaling={scaling_factor:.0e})")
        x_rec, resid = nnls(A * scaling_factor, b * scaling_factor)

        # 11) Reprojection for output visuals
        b_rec = A @ x_rec
        b_rec_ap = b_rec[:n_pix].reshape(H, W)
        b_rec_pa = b_rec[n_pix:].reshape(H, W)

        # 12) Print results
        print("\n=== ERGEBNISSE ===")
        print(f"{'Organ':<15} | {'GT (kBq/mL)':<15} | {'Rec (kBq/mL)':<15} | {'Rel. Error (%)':<15}")
        print("-" * 75)
        results = {}
        for i, organ in enumerate(organ_names):
            gt = float(x_gt[i])
            rc = float(x_rec[i])
            rel_err = abs(rc - gt) / (abs(gt) + 1e-12) * 100.0 if gt != 0 else abs(rc) * 100.0
            print(f"{organ:<15} | {gt:<15.4f} | {rc:<15.4f} | {rel_err:<15.4f}")
            results[organ] = {
                "organ_id": int(organ_info[organ]["organ_id"]),
                "gt_kBq_per_ml": gt,
                "rec_kBq_per_ml": rc,
                "rel_error_percent": rel_err,
            }

        # 13) Save JSON summary
        res_json = run_out_dir / "quantification_results.json"
        with open(res_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OUT] Saved results JSON: {res_json}")

        # 14) Optional PNGs
        if args.save_pngs:
            print("[INFO] Saving PNGs...")
            save_png(b_ap_img, run_out_dir / "proj_GT_AP.png", f"{run_name}: GT Projection AP (Full-Forward)")
            save_png(b_pa_img, run_out_dir / "proj_GT_PA.png", f"{run_name}: GT Projection PA (Full-Forward)")
            save_png(b_rec_ap, run_out_dir / "proj_Rec_AP.png", f"{run_name}: Reconstructed Projection AP (A@x_rec)")
            save_png(b_rec_pa, run_out_dir / "proj_Rec_PA.png", f"{run_name}: Reconstructed Projection PA (A@x_rec)")

        return results

    if args.mask_robustness_suite:
        print("[INFO] Running mask robustness suite: baseline + 2D-dilation + shift(x=1cm,2cm)")
        print("[INFO] 2D-dilation overlap policy: higher label ID wins")
        print("[INFO] Robustness convention: b from baseline mask, A from perturbed mask")
        suite_summary: Dict[str, Any] = {}
        baseline_mask_ids = mask_ids

        suite_summary["baseline"] = run_quantification(
            "baseline",
            baseline_mask_ids,
            out_dir / "baseline",
            measurement_mask_ids=baseline_mask_ids,
        )

        mask_dil_xy = dilate_labels_radius1_xy(baseline_mask_ids, all_nonzero_label_ids)
        suite_summary["dilation2d_r1_xy"] = run_quantification(
            "dilation2d_r1_xy",
            mask_dil_xy,
            out_dir / "dilation2d_r1_xy",
            measurement_mask_ids=baseline_mask_ids,
        )

        for shift_cm in (1.0, 2.0):
            shift_vox = max(1, int(round((shift_cm * 10.0) / args.pixel_size_mm)))
            achieved_cm = (shift_vox * args.pixel_size_mm) / 10.0
            print(
                f"[INFO] Shift target x={shift_cm:.1f} cm -> dx={shift_vox} vox "
                f"(achieved ~{achieved_cm:.3f} cm at {args.pixel_size_mm:.3f} mm/voxel)"
            )
            run_name = f"shift_x{int(shift_cm)}cm_dx{shift_vox}"
            mask_shift = shift_mask_zero_pad(baseline_mask_ids, shift_xyz=(shift_vox, 0, 0))
            suite_summary[run_name] = run_quantification(
                run_name,
                mask_shift,
                out_dir / run_name,
                measurement_mask_ids=baseline_mask_ids,
            )

        suite_json = out_dir / "mask_robustness_suite_summary.json"
        with open(suite_json, "w", encoding="utf-8") as f:
            json.dump(suite_summary, f, indent=2)
        print(f"[OUT] Saved suite summary JSON: {suite_json}")
    else:
        run_quantification("baseline", mask_ids, out_dir)

    print("[DONE]")


if __name__ == "__main__":
    main()
