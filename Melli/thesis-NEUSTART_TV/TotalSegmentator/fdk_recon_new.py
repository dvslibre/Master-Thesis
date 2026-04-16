#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDK Reconstruction (Cone-Beam CT) for xCAT/ct_projector data using ITK-RTK.

Pipeline:
  1) Read inputs (.mat with mu_stack, angles_out/angles_deg optional; geometry .txt)
  2) Build RTK 3D circular projection geometry (per projection)
  3) FDK reconstruction (with Ramp/Hann filter; optional CUDA if available)
  4) Write 3D volume as NIfTI (.nii) in RAS (x=Right, y=Anterior, z=Superior)
  5) Optional preview PNGs (coronal/axial/sagittal) with mm-extents and windowing
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import itk
import nibabel as nib
import matplotlib.pyplot as plt

# ----------------------------
# I/O: Projections (.mat / .npy)
# ----------------------------

def read_mat_mu_stack(mat_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Reads mu_stack and optional angles (angles_out / angles_deg) from a .mat file.
    Returns:
        mu_stack: np.ndarray with shape (nProj, nr, nc)
        angles_deg: np.ndarray with shape (nProj,) in degrees or None
    """
    mu_stack = None
    angles_deg = None

    def _pick_angle_key(d):
        for k in ["angles_out", "angles_deg", "angles", "ang_deg"]:
            if k in d:
                return k
        return None

    # Try SciPy
    try:
        import scipy.io as sio  # type: ignore
        md = sio.loadmat(mat_path)
        if "mu_stack" not in md:
            raise KeyError("mu_stack not found in MAT file")
        mu_stack = md["mu_stack"]
        if mu_stack.ndim != 3:
            raise ValueError(f"mu_stack has ndim={mu_stack.ndim}, expected 3")
        # Common .mat layout: (nr, nc, nProj) -> (nProj, nr, nc)
        if mu_stack.shape[2] >= max(mu_stack.shape[0], mu_stack.shape[1]):
            mu_stack = np.transpose(mu_stack, (2, 0, 1))

        k = _pick_angle_key(md)
        if k is not None:
            angles = md[k].squeeze()
            angles = np.array(angles, dtype=float)
            angles_deg = angles
        return mu_stack.astype(np.float32), angles_deg
    except Exception as e1:
        # Try h5py (MAT v7.3)
        try:
            import h5py  # type: ignore
            with h5py.File(mat_path, "r") as f:
                def _read(name):
                    d = f[name][()]
                    return np.array(d)

                if "mu_stack" in f:
                    mu = _read("mu_stack")
                else:
                    cand = [k for k in f.keys() if "mu" in k and "stack" in k]
                    if not cand:
                        raise KeyError("mu_stack dataset not found in MAT file")
                    mu = _read(cand[0])

                mu = np.array(mu, order="C")
                if mu.ndim != 3:
                    raise ValueError(f"mu_stack has ndim={mu.ndim}, expected 3")
                if mu.shape[2] >= max(mu.shape[0], mu.shape[1]):
                    mu = np.transpose(mu, (2, 0, 1))
                mu_stack = mu.astype(np.float32)

                angle_key = None
                for k in ["angles_out", "angles_deg", "angles", "ang_deg"]:
                    if k in f:
                        angle_key = k
                        break
                if angle_key is not None:
                    a = _read(angle_key).squeeze()
                    angles_deg = np.array(a, dtype=float)

            return mu_stack, angles_deg
        except Exception as e2:
            print(
                f"[ERROR] Could not read MAT file '{mat_path}'. "
                f"Tried scipy.io.loadmat ({e1}) and h5py ({e2}).\n"
                "Please install SciPy (recommended) or h5py, or export your mu_stack to .npy.\n"
                "Example: np.save('mu_stack.npy', mu_stack)",
                file=sys.stderr,
            )
            sys.exit(2)


def maybe_read_npy_fallback(npy_path: Optional[str]) -> Optional[np.ndarray]:
    if npy_path and os.path.isfile(npy_path):
        arr = np.load(npy_path)
        if arr.ndim != 3:
            raise ValueError("Expected npy mu_stack with shape (nProj, nr, nc)")
        return arr.astype(np.float32)
    return None


# ----------------------------
# Geometry parser (robust, supports "value : key")
# ----------------------------

def read_geometry_txt(txt_path: str) -> dict:
    """
    Robust parser for proj_*.txt.
    Supports:
      - "key = value"
      - "key : value"
      - "value : key" (e.g., "512 :num_rows")
    Ignores comments after '#' or '//'. Extracts first numeric token.
    """
    import re

    def normalize_key(k: str) -> str:
        k = k.strip().lower()
        k = k.split('#', 1)[0].strip()
        k = k.split('//', 1)[0].strip()
        k = re.sub(r"\(.*?\)", "", k)          # remove parentheses content
        k = re.sub(r"[ \-\_:]", "", k)         # remove spaces, - _ :
        return k

    def parse_number(s: str) -> float:
        s = s.replace("−", "-")                # unicode minus
        m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
        if not m:
            raise ValueError(f"no numeric token in '{s}'")
        return float(m.group(0))

    kv = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith(("#", "//")):
                continue
            if "=" in raw:
                left, right = raw.split("=", 1)
            elif ":" in raw:
                left, right = raw.split(":", 1)
            else:
                continue
            left, right = left.strip(), right.strip()
            left_has_letters  = any(c.isalpha() for c in left)
            right_has_letters = any(c.isalpha() for c in right)

            if left_has_letters and not right_has_letters:
                key_norm, val_str = normalize_key(left), right
            elif right_has_letters and not left_has_letters:
                key_norm, val_str = normalize_key(right), left  # value : key
            else:
                # Heuristic: side with more letters is the key
                letters_left  = sum(ch.isalpha() for ch in left)
                letters_right = sum(ch.isalpha() for ch in right)
                if letters_left >= letters_right:
                    key_norm, val_str = normalize_key(left), right
                else:
                    key_norm, val_str = normalize_key(right), left

            try:
                kv[key_norm] = parse_number(val_str)
            except Exception:
                pass

    def pick(*cands, default=None):
        for c in cands:
            if c in kv:
                return kv[c]
        return default

    DSO   = pick("distancetosource", "distancetosourcemms", "dso", "sourceisocenterdistance", "s2iso")
    DTD   = pick("distancetodetector", "distancetodetectormms", "dtd", "isocentertodetector", "is2det")
    height= pick("height", "heightmms", "detectorheight", "vheight")
    width = pick("width",  "widthmms",  "detectorwidth",  "uwidth")
    nr    = pick("numrows", "rows", "detectorrows", "vrows")
    nc    = pick("numchannels", "columns", "detectorcols", "detectorcolumns", "ucols")
    half  = pick("halffanangle", "halffan", "fananglehalf")
    shift = pick("detectorshift", "detectoroffset", "uoffset", "detectoruoffset", default=0.0)
    outtp = pick("outputtype", default=0.0)

    missing = [name for name, val in [
        ("distance_to_source (DSO)", DSO),
        ("distance_to_detector (DTD)", DTD),
        ("height", height),
        ("width", width),
        ("num_rows (nr)", nr),
        ("num_channels (nc)", nc),
        ("Half_fan_angle (deg)", half),
    ] if val is None]
    if missing:
        found_keys = ", ".join(sorted(kv.keys()))
        raise ValueError("Missing required geometry keys: " + ", ".join(missing) +
                         f"\nFound keys: {found_keys}")

    geom = {
        "DSO": float(DSO),
        "DTD": float(DTD),
        "height": float(height),
        "width": float(width),
        "nr": int(round(nr)),
        "nc": int(round(nc)),
        "half_fan_angle_deg": float(half),
        "detector_shift": float(shift),
        "output_type": float(outtp),
    }
    geom["SDD"] = geom["DSO"] + geom["DTD"]
    geom["du"]  = geom["width"] / geom["nc"]
    geom["dv"]  = geom["height"] / geom["nr"]
    return geom


# ----------------------------
# Angles, transforms, auto-orientation
# ----------------------------

def build_default_angles(n_proj: int, start_deg: float = 90.0) -> np.ndarray:
    """Uniform 360° coverage starting at start_deg."""
    return (start_deg + np.arange(n_proj) * (360.0 / n_proj)).astype(np.float64)

def apply_angle_transforms(
    angles_deg: np.ndarray,
    angle_offset_deg: float = 0.0,
    reverse_angles: bool = False,
) -> np.ndarray:
    a = angles_deg.astype(np.float64).copy()
    if reverse_angles:
        a = a[::-1]
    if angle_offset_deg != 0.0:
        a = a + angle_offset_deg
    return a

def auto_detect_orientation(mu_stack: np.ndarray) -> Tuple[bool, bool]:
    """
    Heuristic detection of (reverse_angles, flip_u).
    """
    def _norm(x):
        x = x.astype(np.float32)
        x = (x - np.mean(x)) / (np.std(x) + 1e-6)
        return x

    n, r, c = mu_stack.shape
    rows = slice(r//4, 3*r//4)

    def _avg_adj_corr(arr):
        cc = []
        for i in range(1, min(n, 50)):
            a = _norm(arr[i-1, rows, :]).ravel()
            b = _norm(arr[i,   rows, :]).ravel()
            cc.append(np.dot(a, b) / (len(a)))
        return float(np.mean(cc)) if cc else 0.0

    corr_forward = _avg_adj_corr(mu_stack)
    corr_reverse = _avg_adj_corr(mu_stack[::-1, :, :])
    suggest_reverse = corr_reverse > corr_forward * 1.02

    a0 = _norm(mu_stack[0, rows, :]).ravel()
    a1 = _norm(mu_stack[1, rows, :]).ravel() if n > 1 else a0
    a1f = _norm(mu_stack[1, rows, ::-1]).ravel() if n > 1 else a1
    corr_no_flip = float(np.dot(a0, a1) / len(a0))
    corr_flip    = float(np.dot(a0, a1f) / len(a0))
    suggest_flip = corr_flip > corr_no_flip * 1.02

    return suggest_reverse, suggest_flip


# ----------------------------
# Shape fix for mu_stack
# ----------------------------

def fix_mu_stack_shape(mu_stack: np.ndarray, expected_nr: int, expected_nc: int) -> np.ndarray:
    """
    Ensure shape is (nProj, nr, nc).
    Handles common permutations:
      - (nr, nc, nProj)  -> transpose to (nProj, nr, nc)
      - (nc, nr, nProj)  -> transpose to (nProj, nr, nc)
      - (nProj, nr, nc)  -> keep
    Uses expected_nr/nc from geometry to decide.
    """
    s = mu_stack.shape
    if mu_stack.ndim != 3:
        raise ValueError(f"mu_stack must be 3D, got shape {s}")

    # Already correct?
    if s[1] == expected_nr and s[2] == expected_nc:
        return mu_stack

    # Case A: (nr, nc, nProj)
    if s[0] == expected_nr and s[1] == expected_nc:
        t = np.transpose(mu_stack, (2, 0, 1))
        print(f"[INFO] Transposed mu_stack (nr,nc,nProj)->(nProj,nr,nc): {s} -> {t.shape}")
        return t

    # Case B: (nc, nr, nProj)
    if s[0] == expected_nc and s[1] == expected_nr:
        t = np.transpose(mu_stack, (2, 1, 0))
        print(f"[INFO] Transposed mu_stack (nc,nr,nProj)->(nProj,nr,nc): {s} -> {t.shape}")
        return t

    # Try all perms
    perms = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    for p in perms:
        t = np.transpose(mu_stack, p)
        if t.shape[1] == expected_nr and t.shape[2] == expected_nc:
            print(f"[INFO] Transposed mu_stack by {p}: {s} -> {t.shape}")
            return t

    # Fallback: assume last axis is nProj after moving the largest dim to front
    idx_nproj = int(np.argmax(s))
    order = [idx_nproj] + [i for i in range(3) if i != idx_nproj]
    t = np.transpose(mu_stack, order)
    print(f"[WARN] Heuristic perm {order}: {s} -> {t.shape}")
    return t


# ----------------------------
# ITK conversion & RTK geometry
# ----------------------------

def numpy_to_itk_projections(
    mu_stack: np.ndarray, du: float, dv: float, detector_shift_mm: float
) -> itk.Image:
    """
    Convert numpy (nProj, nr, nc) to RTK projections image (u, v, projection)
    Shape in ITK: [nc, nr, nProj] with spacing [du, dv, 1.0]
    Origin (u,v,w): center the detector, apply lateral shift on u (in mm).
    """
    nProj, nr, nc = mu_stack.shape
    proj = np.transpose(mu_stack, (2, 1, 0)).copy()  # (nc, nr, nProj)

    PixelType = itk.F
    ImageType = itk.Image[PixelType, 3]
    itk_img = itk.image_from_array(proj, is_vector=False)  # zyx; here z=nProj
    itk_img = itk.cast_image_filter(itk_img, ttype=(type(itk_img), ImageType))

    itk_img.SetSpacing((float(du), float(dv), 1.0))

    width = du * nc
    height = dv * nr
    origin_u = -0.5 * width + 0.5 * du + float(detector_shift_mm)
    origin_v = -0.5 * height + 0.5 * dv
    origin_w = 0.0
    itk_img.SetOrigin((origin_u, origin_v, origin_w))

    dir_mat = itk.matrix_from_array(np.eye(3, dtype=float))
    itk_img.SetDirection(dir_mat)

    return itk_img


def build_rtk_geometry(
    angles_deg: np.ndarray,
    DSO: float,
    SDD: float,
    det_shift_mm: float,
) -> "itk.RTK.ThreeDCircularProjectionGeometry":
    """
    Construct RTK 3D circular projection geometry; one AddProjection per angle.
    RTK expects angles in radians, detector offsets (projOffsetX, projOffsetY) in mm.
    """
    GeometryType = itk.RTK.ThreeDCircularProjectionGeometry
    geometry = GeometryType.New()
    for a_deg in angles_deg:
        angle_rad = float(np.deg2rad(a_deg))
        geometry.AddProjection(float(DSO), float(SDD), angle_rad, float(det_shift_mm), 0.0)
    return geometry


def estimate_default_fov_mm(width_det_mm: float, height_det_mm: float, DSO: float, SDD: float) -> Tuple[float, float, float]:
    """
    Estimate a reasonable FOV at isocenter by backprojecting detector size:
      FOV_xy ≈ detector_width * (DSO / SDD)
      FOV_z  ≈ detector_height * (DSO / SDD)
    """
    M = float(DSO) / float(SDD)
    fov_x = width_det_mm * M
    fov_y = width_det_mm * M
    fov_z = height_det_mm * M
    return fov_x, fov_y, fov_z


# ----------------------------
# FDK run + NIfTI export + preview
# ----------------------------

def run_fdk_recon(
    proj_itk: itk.Image,
    geometry,
    voxel_size_mm: Tuple[float, float, float],
    fov_mm: Tuple[float, float, float],
    hann_cut: float,
    use_cuda: bool,
) -> itk.Image:
    """
    Configure and run RTK FDK reconstruction.
    Output spacing/origin/size defined by voxel_size_mm and fov_mm centered at isocenter (0,0,0 in LPS).
    """
    sx, sy, sz = [float(s) for s in voxel_size_mm]
    fovx, fovy, fovz = [float(f) for f in fov_mm]

    size_x = max(1, int(np.round(fovx / sx)))
    size_y = max(1, int(np.round(fovy / sy)))
    size_z = max(1, int(np.round(fovz / sz)))

    origin_x = -0.5 * size_x * sx + 0.5 * sx
    origin_y = -0.5 * size_y * sy + 0.5 * sy
    origin_z = -0.5 * size_z * sz + 0.5 * sz

    OutputImageType = itk.Image[itk.F, 3]
    out_img = OutputImageType.New()
    region = itk.ImageRegion[3]()
    region.SetSize([size_x, size_y, size_z])
    out_img.SetRegions(region)
    out_img.SetSpacing((sx, sy, sz))
    out_img.SetOrigin((origin_x, origin_y, origin_z))
    out_img.SetDirection(itk.matrix_from_array(np.eye(3, dtype=float)))
    out_img.Allocate()
    out_img.FillBuffer(0)

    FDKType = itk.RTK.FDKConeBeamReconstructionFilter[OutputImageType]
    fdk = FDKType.New()
    fdk.SetInput(0, out_img)
    fdk.SetInput(1, proj_itk)
    fdk.SetGeometry(geometry)

    if hasattr(fdk, "SetHannCutFrequency"):
        fdk.SetHannCutFrequency(float(hann_cut))
    if hasattr(fdk, "SetTruncationCorrection"):
        fdk.SetTruncationCorrection(False)

    if use_cuda:
        for attr in ["SetUseCuda", "SetHardwareAcceleration", "SetCudaOn"]:
            if hasattr(fdk, attr):
                try:
                    getattr(fdk, attr)(True)
                except Exception:
                    pass

    fdk.Update()
    recon = fdk.GetOutput()
    recon.DisconnectPipeline()
    return recon


def itk_to_nifti_ras(itk_img: itk.Image, out_path: str):
    """
    Convert ITK (LPS world) to NIfTI (RAS world) by converting the affine:
        A_ras = A_lps @ diag([-1, -1, 1, 1])
    """
    arr = itk.array_from_image(itk_img).astype(np.float32)  # (z,y,x)

    spacing = np.array(list(itk_img.GetSpacing()), dtype=float)
    origin  = np.array(list(itk_img.GetOrigin()), dtype=float)
    direction = np.array(itk_img.GetDirection()).reshape(3,3).astype(float)

    A_lps = np.eye(4, dtype=float)
    A_lps[:3, :3] = direction @ np.diag(spacing)
    A_lps[:3,  3] = origin

    LPS2RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
    A_ras = A_lps @ LPS2RAS

    nii = nib.Nifti1Image(arr, affine=A_ras)
    nii.set_sform(A_ras, code=1)
    nii.set_qform(A_ras, code=1)
    nib.save(nii, out_path)
    return out_path


def save_preview_pngs(
    nifti_path: str,
    view: str,
    coronal_orient: str,
    out_png: str,
):
    """Create a single PNG preview with mm axes."""
    img = nib.load(nifti_path)
    vol = img.get_fdata().astype(np.float32)
    A = img.affine

    vx = float(np.linalg.norm(A[:3, 0]))
    vy = float(np.linalg.norm(A[:3, 1]))
    vz = float(np.linalg.norm(A[:3, 2]))

    nx, ny, nz = vol.shape[2], vol.shape[1], vol.shape[0]

    def _window(im):
        lo, hi = np.percentile(im, [2, 98])
        if hi <= lo:
            lo, hi = np.min(im), np.max(im)
        return np.clip((im - lo) / (hi - lo + 1e-6), 0, 1)

    view = view.lower()
    coronal_orient = coronal_orient.upper()

    if view == "coronal":
        ix = nx // 2
        sl = vol[:, :, ix]       # (z, y)
        img2d = _window(sl).T    # (y, z)
        extent = [0, ny*vy, 0, nz*vz]
        xlabel, ylabel = "Anterior–Posterior (mm)", "Superior–Inferior (mm)"
        if coronal_orient == "PA":
            img2d = img2d[::-1, :]
    elif view == "axial":
        iy = ny // 2
        sl = vol[:, iy, :]       # (z, x)
        img2d = _window(sl)
        extent = [0, nx*vx, 0, nz*vz]
        xlabel, ylabel = "Right–Left (mm)", "Superior–Inferior (mm)"
    elif view == "sagittal":
        iz = nz // 2
        sl = vol[iz, :, :]       # (y, x)
        img2d = _window(sl)
        extent = [0, nx*vx, 0, ny*vy]
        xlabel, ylabel = "Right–Left (mm)", "Anterior–Posterior (mm)"
    else:
        raise ValueError(f"Unknown preview view '{view}'")

    plt.figure(figsize=(6, 6))
    plt.imshow(img2d, extent=extent, origin="lower", aspect="equal")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{view.capitalize()} view")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="FDK Reconstruction for xCAT/ct_projector Cone-Beam CT (ITK-RTK).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--mat", type=str, required=False, default="mu_stack_400.mat",
                   help="Path to .mat containing mu_stack (nProj,nr,nc) and optional angles (degrees).")
    p.add_argument("--npy", type=str, default=None,
                   help="Optional: .npy file with mu_stack (nProj,nr,nc) if you cannot read .mat.")
    p.add_argument("--geom_txt", type=str, required=False, default="proj_400.txt",
                   help="Path to proj_*.txt with geometry parameters.")
    p.add_argument("--out_nifti", type=str, default="recon_fdk.nii",
                   help="Output NIfTI path (RAS).")

    # Orientation / angle control
    p.add_argument("--angle_offset", type=float, default=0.0,
                   help="Angle offset in degrees (added to all angles).")
    p.add_argument("--reverse_angles", action="store_true",
                   help="Reverse the order of projection angles.")
    p.add_argument("--flip_u", action="store_true",
                   help="Flip detector u-direction (horizontal flip of projections).")
    p.add_argument("--detector_shift_unit", type=str, choices=["mm", "pixel"], default="mm",
                   help="Unit of detector_shift in geometry .txt.")
    p.add_argument("--auto_orient", action="store_true",
                   help="Try to auto-detect reverse_angles/flip_u suggestion and apply it.")

    # Reconstruction grid / FOV
    p.add_argument("--voxel", type=float, nargs=3, default=[1.5, 1.5, 1.5],
                   metavar=("SX", "SY", "SZ"),
                   help="Output voxel size in mm (x,y,z).")
    p.add_argument("--fov", type=float, nargs=3, default=None,
                   metavar=("FX", "FY", "FZ"),
                   help="Field-of-View (mm) for x,y,z. If omitted, estimated from detector size.")
    p.add_argument("--hann", type=float, default=0.7,
                   help="Hann filter cut frequency (0..1).")
    p.add_argument("--cuda", action="store_true",
                   help="Try to enable CUDA acceleration (if RTK was built with CUDA).")

    # Preview
    p.add_argument("--preview_view", type=str, choices=["coronal", "axial", "sagittal"], default=None,
                   help="Create a single PNG preview in the chosen view.")
    p.add_argument("--coronal_orient", type=str, choices=["AP", "PA"], default="AP",
                   help="Coronal orientation (AP or PA) for preview only.")
    p.add_argument("--preview_png", type=str, default="preview.png",
                   help="Output PNG path for preview.")

    # Angles handling if absent in .mat
    p.add_argument("--start_angle", type=float, default=90.0,
                   help="Start angle in degrees if angles are not found in .mat")
    return p.parse_args()


# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_args()

    # 1) Read geometry
    geom = read_geometry_txt(args.geom_txt)
    DSO = float(geom["DSO"])
    DTD = float(geom["DTD"])
    SDD = float(geom["SDD"])
    nr  = int(geom["nr"])
    nc  = int(geom["nc"])
    du  = float(geom["du"])
    dv  = float(geom["dv"])
    det_shift_in = float(geom["detector_shift"])
    det_shift_mm = det_shift_in * du if args.detector_shift_unit == "pixel" else det_shift_in

    # 2) Read projections + angles
    mu_stack = maybe_read_npy_fallback(args.npy)
    if mu_stack is None:
        mu_stack, angles_deg = read_mat_mu_stack(args.mat)
    else:
        angles_deg = None

    # Fix shape to (nProj, nr, nc) using geometry hints
    mu_stack = fix_mu_stack_shape(mu_stack, expected_nr=nr, expected_nc=nc)
    nProj, nr_in, nc_in = mu_stack.shape
    if (nr_in != nr) or (nc_in != nc):
        print(f"[WARN] Geometry (nr,nc)=({nr},{nc}) differs from data shape ({nr_in},{nc_in}); continuing with data shape.", file=sys.stderr)
        nr, nc = nr_in, nc_in

    # Angles: from .mat or default uniform 360°
    if angles_deg is None:
        angles_deg = build_default_angles(nProj, start_deg=args.start_angle)
        print(f"[INFO] No angles in .mat; using uniform 360° from start={args.start_angle}°", file=sys.stderr)
    else:
        angles_deg = np.asarray(angles_deg, dtype=float).ravel()
        if angles_deg.size != nProj:
            print(f"[WARN] angles array has length {angles_deg.size}, but nProj={nProj}. Using uniform angles.", file=sys.stderr)
            angles_deg = build_default_angles(nProj, start_deg=args.start_angle)
        else:
            print(f"[INFO] Using {angles_deg.size} angles from .mat (e.g., 'angles_out'/'angles_deg').", file=sys.stderr)

    # Optional auto-detection (heuristic)
    if args.auto_orient:
        suggest_reverse, suggest_flip = auto_detect_orientation(mu_stack)
        if suggest_reverse:
            print("[INFO] Auto-detect suggests: --reverse_angles", file=sys.stderr)
            args.reverse_angles = True
        if suggest_flip:
            print("[INFO] Auto-detect suggests: --flip_u", file=sys.stderr)
            args.flip_u = True

    # Apply transforms
    angles_deg = apply_angle_transforms(
        angles_deg,
        angle_offset_deg=args.angle_offset,
        reverse_angles=args.reverse_angles
    )
    if args.flip_u:
        mu_stack = mu_stack[:, :, ::-1].copy()
        det_shift_mm = -det_shift_mm  # flipping u => negate lateral offset

    # 3) ITK projections image
    proj_itk = numpy_to_itk_projections(mu_stack, du=du, dv=dv, detector_shift_mm=det_shift_mm)

    # 4) RTK geometry
    geometry = build_rtk_geometry(angles_deg=angles_deg, DSO=DSO, SDD=SDD, det_shift_mm=det_shift_mm)

    # 5) FOV defaults
    if args.fov is None:
        fovx, fovy, fovz = estimate_default_fov_mm(width_det_mm=geom["width"], height_det_mm=geom["height"], DSO=DSO, SDD=SDD)
        fov_mm = (fovx, fovy, fovz)
        print(f"[INFO] Estimated FOV (mm): x={fovx:.1f}, y={fovy:.1f}, z={fovz:.1f}", file=sys.stderr)
    else:
        fov_mm = tuple(float(x) for x in args.fov)

    # 6) Run FDK
    recon = run_fdk_recon(
        proj_itk=proj_itk,
        geometry=geometry,
        voxel_size_mm=tuple(args.voxel),
        fov_mm=fov_mm,
        hann_cut=float(args.hann),
        use_cuda=bool(args.cuda),
    )

    # 7) Save NIfTI (RAS)
    out_path = args.out_nifti
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    itk_to_nifti_ras(recon, out_path)
    print(f"[OK] Wrote NIfTI (RAS): {out_path}")

    # 8) Optional preview
    if args.preview_view:
        save_preview_pngs(
            nifti_path=out_path,
            view=args.preview_view,
            coronal_orient=args.coronal_orient,
            out_png=args.preview_png,
        )
        print(f"[OK] Wrote preview PNG: {args.preview_png}")


if __name__ == "__main__":
    main()