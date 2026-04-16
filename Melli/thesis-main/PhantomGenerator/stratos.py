# stratos.py — activity-ID based (organ_ids.txt), dynamic 'others', optimized projector
#
# What this file contains:
# - Fortran-order raw volume loader (matches MATLAB fread+reshape)
# - Gamma camera forward model (scatter -> attenuation -> depth PSF -> z-sum)
# - Core path without per-call file I/O (faster A-matrix build)
# - Mapping strictly via organ_ids.txt (Activity-ID namespace for mask)
# - 'others' group auto-filled with all mask activity-IDs not used in main groups
# - End-to-end pipeline run_lgs_and_nnls(...)
# - Forward-only helper run_forward_projection(...)

import os
import numpy as np
from typing import Dict, List, Tuple
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d, fftconvolve
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
USE_FFT_CONV = True   # faster for larger kernels; physics unchanged

# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------
def load_raw_volume(path: str, dims: Tuple[int,int,int], dtype=np.float32, endian="native"):
    """Load raw binary 3D volume with Fortran order to match MATLAB reshape."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Not found: {path}")
    if endian.lower().startswith('l'):
        dtype = np.dtype(dtype).newbyteorder('<')
    elif endian.lower().startswith('b'):
        dtype = np.dtype(dtype).newbyteorder('>')
    data = np.fromfile(path, dtype=dtype, count=np.prod(dims))
    if data.size != np.prod(dims):
        raise ValueError(f"Size mismatch for {path}: expected {np.prod(dims)}, got {data.size}")
    return data.reshape(dims, order='F')

def save_npy(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

# -----------------------------------------------------------------------------
# organ_ids.txt mapping (Activity-ID namespace, matches mask values)
# -----------------------------------------------------------------------------
def read_organ_ids_table(path_txt: str) -> Dict[str, int]:
    """Reads lines like 'liver_activity = 134' -> {name: id}. Ignores comments/blank lines."""
    mapping: Dict[str,int] = {}
    if not os.path.isfile(path_txt):
        return mapping
    with open(path_txt, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#') or '=' not in s:
                continue
            name, val = s.split('=', 1)
            name = name.strip()
            try:
                vid = int(float(val.strip()))
                mapping[name] = vid
            except ValueError:
                continue
    return mapping

# -----------------------------------------------------------------------------
# Gamma camera forward model
# -----------------------------------------------------------------------------
def _process_view_phys(act_data: np.ndarray, atn_data: np.ndarray,
                       comp_scatter: bool, atn_on: bool, coll_on: bool,
                       kernel_mat: np.ndarray, sigma: float,
                       z0_slices: int) -> np.ndarray:
    """Pipeline: (1) 2D Gaussian scatter per slice (xy); (2) attenuation exp(-cumsum mu along z);
    (3) depth-dependent collimator conv2 from z0+1; (4) projection: sum over z."""
    # (1) Scatter per z-slice
    if comp_scatter:
        act_sc = np.empty_like(act_data, dtype=np.float32)
        for z in range(act_data.shape[2]):
            act_sc[:,:,z] = gaussian_filter(act_data[:,:,z], sigma=sigma, mode='nearest')
    else:
        act_sc = act_data.astype(np.float32, copy=False)

    # (2) Attenuation along z
    if atn_on:
        mu_cum = np.cumsum(atn_data, axis=2)
        vol_atn = act_sc * np.exp(-mu_cum)
    else:
        vol_atn = act_sc

    # (3) Collimator convolution from z0+1
    if coll_on:
        Z = vol_atn.shape[2]
        vol_coll = np.zeros_like(vol_atn, dtype=np.float32)
        conv2 = fftconvolve if USE_FFT_CONV else convolve2d
        for z in range(Z):
            if z <= z0_slices:
                vol_coll[:,:,z] = vol_atn[:,:,z]
            else:
                zz = min(z - z0_slices, kernel_mat.shape[2]-1)
                K = kernel_mat[:,:,zz]
                vol_coll[:,:,z] = conv2(vol_atn[:,:,z], K, mode='same').astype(np.float32)
    else:
        vol_coll = vol_atn

    # (4) z-summation
    return np.sum(vol_coll, axis=2)


def gamma_camera_core(act_data: np.ndarray, atn_data: np.ndarray,
                      view: str,
                      comp_scatter: bool, atn_on: bool, coll_on: bool,
                      kernel_mat: np.ndarray,
                      nx: int, ny: int, nz: int,
                      sigma: float,
                      z0_slices: int) -> Tuple[np.ndarray, np.ndarray]:
    """Same physics/orientation as MATLAB path, but without per-call file I/O."""
    assert act_data.shape == (nx,ny,nz) and atn_data.shape == (nx,ny,nz)
    if view != 'frontal':
        raise ValueError("Only 'frontal' is implemented (AP/PA).")

    # MATLAB pre-rotations
    act_ap = np.transpose(act_data, (0,2,1)); atn_ap = np.transpose(atn_data, (0,2,1))
    act_ap = np.rot90(act_ap);               atn_ap = np.rot90(atn_ap)
    act_ap = np.transpose(act_ap, (1,0,2));  atn_ap = np.transpose(atn_ap, (1,0,2))
    act_pa = np.flip(act_ap, axis=2);        atn_pa = np.flip(atn_ap, axis=2)

    proj_AP = _process_view_phys(act_ap, atn_ap, comp_scatter, atn_on, coll_on, kernel_mat, sigma, z0_slices)
    proj_PA = _process_view_phys(act_pa, atn_pa, comp_scatter, atn_on, coll_on, kernel_mat, sigma, z0_slices)

    # Orientation patch: rot90 -> flipud -> transpose
    def orient_patch(P):
        P1 = np.rot90(P); P2 = np.flipud(P1); return P2.T
    return orient_patch(proj_AP), orient_patch(proj_PA)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def rrmse(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a.ravel() - b.ravel()) / max(np.linalg.norm(a.ravel()), 1e-12)


def rrmse_alpha(gt: np.ndarray, rec: np.ndarray, roi_border_px: int = 0) -> Tuple[float,float]:
    if roi_border_px > 0 and gt.shape[0] > 2*roi_border_px and gt.shape[1] > 2*roi_border_px:
        gt  = gt[roi_border_px:-roi_border_px, roi_border_px:-roi_border_px]
        rec = rec[roi_border_px:-roi_border_px, roi_border_px:-roi_border_px]
    gtv  = gt.astype(np.float64).ravel()
    recv = rec.astype(np.float64).ravel()
    denom = max(np.dot(recv, recv), 1e-12)
    alpha = float(np.dot(recv, gtv) / denom)
    return float(np.linalg.norm(alpha*recv - gtv) / max(np.linalg.norm(gtv), 1e-12)), alpha

# -----------------------------------------------------------------------------
# Plot helper (eine 2×2-Übersicht, Heatmap, 90° CW Rotation)
# -----------------------------------------------------------------------------
def _save_quad_figure(ap_gt: np.ndarray, pa_gt: np.ndarray,
                      ap_rec: np.ndarray, pa_rec: np.ndarray,
                      path: str):
    """
    Speichert eine 2×2-Abbildung:
      [ GT–AP | GT–PA ]
      [ REC–AP| REC–PA ]
    Darstellung: Heatmap ('inferno'), alle Bilder 90° clockwise gedreht.
    Ein gemeinsamer Farbbereich (vmin/vmax) für faire Helligkeitsvergleiche.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 90° CW drehen (k=-1)
    ap_gt_r  = np.rot90(ap_gt,  k=-1)
    pa_gt_r  = np.rot90(pa_gt,  k=-1)
    ap_rec_r = np.rot90(ap_rec, k=-1)
    pa_rec_r = np.rot90(pa_rec, k=-1)

    vmin = min(ap_gt_r.min(), pa_gt_r.min(), ap_rec_r.min(), pa_rec_r.min())
    vmax = max(ap_gt_r.max(), pa_gt_r.max(), ap_rec_r.max(), pa_rec_r.max())

    plt.figure(figsize=(10, 9))
    imgs = [ap_gt_r, pa_gt_r, ap_rec_r, pa_rec_r]
    titles = ["GT – AP", "GT – PA", "Rekonstruktion – AP", "Rekonstruktion – PA"]

    # vier Subplots
    for i, (img, title) in enumerate(zip(imgs, titles), start=1):
        ax = plt.subplot(2, 2, i)
        im = ax.imshow(img, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        # kleine Farbleiste pro Plot
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# A-matrix (core path; no file I/O inside loop)
# -----------------------------------------------------------------------------
def build_system_matrix_columns(mask_ids: np.ndarray,
                                group_idlist: List[Tuple[str, List[int]]],
                                atn: np.ndarray,
                                nx: int, ny: int, nz: int,
                                kernel_mat: np.ndarray,
                                sigma: float,
                                z0_slices: int,
                                comp_scatter=True, atn_on=True, coll_on=True) -> Tuple[np.ndarray, List[str], List[np.ndarray], Tuple[Tuple[int,int],Tuple[int,int]]]:
    """Build A by projecting a binary mask (1 per group voxel) per group."""
    dummy = np.zeros((nx,ny,nz), dtype=np.float32)
    proj_AP_dummy, proj_PA_dummy = gamma_camera_core(
        dummy, atn, 'frontal', comp_scatter, atn_on, coll_on,
        kernel_mat, nx, ny, nz, sigma, z0_slices
    )
    b_len = proj_AP_dummy.size + proj_PA_dummy.size
    N = len(group_idlist)
    A = np.zeros((b_len, N), dtype=np.float64)
    organs: List[str] = []
    sizes = (proj_AP_dummy.shape, proj_PA_dummy.shape)

    for i, (gname, ids) in enumerate(group_idlist):
        organs.append(gname)
        if len(ids) == 0:
            continue
        mask_vol = np.isin(mask_ids, np.array(ids, dtype=mask_ids.dtype)).astype(np.float32)
        proj_AP_i, proj_PA_i = gamma_camera_core(
            mask_vol, atn, 'frontal', comp_scatter, atn_on, coll_on,
            kernel_mat, nx, ny, nz, sigma, z0_slices
        )
        A[:, i] = np.concatenate([proj_AP_i.ravel(), proj_PA_i.ravel()], axis=0)
    return A, organs, [proj_AP_dummy, proj_PA_dummy], sizes

# -----------------------------------------------------------------------------
# Full workflow
# -----------------------------------------------------------------------------

def run_lgs_and_nnls(
    base_dir="/home/mnguest12/projects/thesis/PhantomGenerator",
    phantom_name="phantom_01",
    dims=(256,256,650),
    sigma=2.0,
    z0_slices=29,
    roi_border_px=16,
    activity_ranges: Dict[str, Tuple[float,float]] = None,
    organ_groups: List[Tuple[str, List[str]]] = None,
    lambda_reg: float = 0.0,
    scalingFactor: float = 1e5,
    results_subdir: str = "results",
    realistic_poisson: bool = False,
    counts_per_pixel: float = 2e4,  # typische Zählstatistik (20000 Counts)
):
    """End-to-end: GT -> b, A -> NNLS, reprojection, metrics, save (optimized)."""
    nx, ny, nz = dims
    phantom_dir = os.path.join(base_dir, phantom_name)
    out_dir = os.path.join(phantom_dir, results_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Paths (with _1 fallback)
    fnameATN  = os.path.join(phantom_dir, f"{phantom_name}_ct.par_atn_1.bin")
    fnameMASK = os.path.join(phantom_dir, f"{phantom_name}_mask.par_act_1.bin")
    if not os.path.isfile(fnameATN):
        alt = fnameATN.replace("_1.bin", ".bin")
        if os.path.isfile(alt): fnameATN = alt
    if not os.path.isfile(fnameMASK):
        alt = fnameMASK.replace("_1.bin", ".bin")
        if os.path.isfile(alt): fnameMASK = alt

    # Load volumes once
    atn  = load_raw_volume(fnameATN,  (nx,ny,nz), dtype=np.float32)
    mask_ids = load_raw_volume(fnameMASK, (nx,ny,nz), dtype=np.float32).astype(np.uint32)
    kernel_mat = loadmat(os.path.join(base_dir, "LEAP_Kernel.mat"))["kernel_mat"].astype(np.float32)

    # Default groups (tokens to match names in organ_ids.txt like '*_activity')
    if organ_groups is None:
        organ_groups = [
            ("liver",        ["liver"]),
            ("kidney",       ["lkidney","rkidney"]),
            ("prostate",     ["prostate"]),
            ("small_intest", ["small_intest"]),
            ("spleen",       ["spleen"]),
            ("others",       []),  # gets filled automatically
        ]
    if activity_ranges is None:
        activity_ranges = {
            "liver":        (3,7),
            "kidney":       (10,35),
            "prostate":     (35,50),
            "small_intest": (3,5),
            "spleen":       (3,5),
            "others":       (0,0),
        }

    # --- Use Activity-ID namespace from organ_ids.txt (matches mask IDs) ---
    name2id_act = read_organ_ids_table(os.path.join(base_dir, "organ_ids.txt"))
    if not name2id_act:
        raise RuntimeError("organ_ids.txt nicht gefunden oder leer – ohne Activity-IDs kann die Maske nicht gruppiert werden.")
    print("[Info] Mapping-Quelle: organ_ids (Activity-IDs)")

    # IDs per group via substring match against organ_ids.txt keys
    group_idlist: List[Tuple[str, List[int]]] = []
    used_ids = set()
    for gname, tokens in organ_groups:
        ids = []
        for nm, vid in name2id_act.items():
            nml = nm.lower()
            if any(tok.lower() in nml for tok in tokens if tok):
                ids.append(int(vid))
        ids = sorted(list(set(ids)))
        group_idlist.append((gname, ids))
        used_ids.update(ids)

    # Fill 'others' with all present activity-IDs not used above
    ids_present = np.unique(mask_ids.astype(np.int64))
    all_known_ids = set(int(v) for v in name2id_act.values())
    present_known = set(int(v) for v in ids_present if int(v) in all_known_ids)
    other_ids = sorted(list(present_known - used_ids))
    for i, (gname, ids) in enumerate(group_idlist):
        if gname.lower() == "others":
            group_idlist[i] = (gname, other_ids)
            break

    # Diagnose
    print(f"[Diag] {phantom_name}: {ids_present.size} einzigartige Maskenwerte.")
    for gname, ids in group_idlist:
        inter = np.intersect1d(ids_present, np.array(ids, dtype=ids_present.dtype))
        if inter.size == 0:
            print(f"[Warn] Gruppe '{gname}': 0 IDs im Volumen (Tokens: {organ_groups[[g for g,_ in organ_groups].index(gname)][1]})")
        else:
            vox = int(np.isin(mask_ids, inter).sum())
            print(f"[OK]   Gruppe '{gname}': {inter.size} IDs, Voxel={vox}")

    # Sample GT activities
    rng = np.random.default_rng(1234)
    true_x = []
    for gname, _ in group_idlist:
        lo, hi = activity_ranges.get(gname, (0,0))
        val = float(rng.uniform(lo, hi)) if hi > lo else 0.0
        true_x.append(val)
    true_x = np.array(true_x, dtype=np.float64)

    # Build GT ACT volume
    act_gt = np.zeros_like(atn, dtype=np.float32)
    for (gname, ids), v in zip(group_idlist, true_x):
        if v == 0.0 or len(ids) == 0:
            continue
        act_gt[np.isin(mask_ids, np.array(ids, dtype=mask_ids.dtype))] = float(v)

    # (1) GT projections -> b (no file I/O)
    proj_AP_gt, proj_PA_gt = gamma_camera_core(
        act_gt, atn, 'frontal', True, True, True,
        kernel_mat, nx, ny, nz, sigma, z0_slices
    )
    b = np.concatenate([proj_AP_gt.ravel(), proj_PA_gt.ravel()], axis=0)

    # --- Poisson-Rauschen auf b (Simulierte Zählstatistik) ---
    if realistic_poisson and counts_per_pixel > 0:
        rng = np.random.default_rng(1234)
        b_scaled = np.maximum(b, 0) * counts_per_pixel
        b_noisy = rng.poisson(b_scaled).astype(np.float64) / counts_per_pixel
        b = b_noisy
        print(f"[Info] Poisson noise applied (counts_per_pixel={counts_per_pixel:.0f})")

    # (2) A columns
    A, organs, _, _ = build_system_matrix_columns(
        mask_ids, group_idlist, atn, nx, ny, nz, kernel_mat, sigma, z0_slices,
        comp_scatter=True, atn_on=True, coll_on=True
    )

    # (3) NNLS with scaling and optional Tikhonov
    if lambda_reg > 0.0:
        I = np.sqrt(lambda_reg) * np.eye(A.shape[1], dtype=np.float64)
        A_reg = np.vstack([A * scalingFactor, I])
        b_reg = np.concatenate([b * scalingFactor, np.zeros(A.shape[1], dtype=np.float64)], axis=0)
    else:
        A_reg = A * scalingFactor
        b_reg = b * scalingFactor
    x_est, nnls_resid = nnls(A_reg, b_reg, maxiter=None)

    # (4) LS lower bound (diagnostic)
    x_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
    b_hat_ls = A @ x_ls
    rrmse_lb = float(np.linalg.norm(b - b_hat_ls) / np.linalg.norm(b))

    # (5) Reprojection
    b_est = A @ x_est
    proj_AP_rec = b_est[:proj_AP_gt.size].reshape(proj_AP_gt.shape)
    proj_PA_rec = b_est[proj_AP_gt.size:].reshape(proj_PA_gt.shape)

    # (6) Metrics
    rmse_val  = float(np.sqrt(np.mean((b - b_est)**2)))
    rrmse_val = float(np.linalg.norm(b - b_est) / np.linalg.norm(b))
    r_ap, a_ap = rrmse_alpha(proj_AP_gt, proj_AP_rec, roi_border_px)
    r_pa, a_pa = rrmse_alpha(proj_PA_gt, proj_PA_rec, roi_border_px)

    # Save results into subfolder
    save_npy(os.path.join(out_dir, f"{phantom_name}_proj_AP_gt.npy"),  proj_AP_gt.astype(np.float32))
    save_npy(os.path.join(out_dir, f"{phantom_name}_proj_PA_gt.npy"),  proj_PA_gt.astype(np.float32))
    save_npy(os.path.join(out_dir, f"{phantom_name}_proj_AP_rec.npy"), proj_AP_rec.astype(np.float32))
    save_npy(os.path.join(out_dir, f"{phantom_name}_proj_PA_rec.npy"), proj_PA_rec.astype(np.float32))
    np.save(os.path.join(out_dir, f"{phantom_name}_x_est.npy"),  x_est.astype(np.float64))
    np.save(os.path.join(out_dir, f"{phantom_name}_x_true.npy"), true_x.astype(np.float64))
    with open(os.path.join(out_dir, f"{phantom_name}_organs.txt"), 'w') as f:
        for g in organs:
            f.write(g + "\n")

    # Nur 2×2-Übersicht speichern (Heatmap, 90° CW)
    _save_quad_figure(
        proj_AP_gt, proj_PA_gt, proj_AP_rec, proj_PA_rec,
        os.path.join(out_dir, f"{phantom_name}_GT_vs_REC_quadrant.png")
    )

    return {
        "phantom_dir": phantom_dir,
        "organs": organs,
        "true_x": true_x,
        "x_est": x_est,
        "rmse": rmse_val,
        "rrmse": rrmse_val,
        "rrmse_lb": rrmse_lb,
        "rrmse_ap": r_ap, "alpha_ap": a_ap,
        "rrmse_pa": r_pa, "alpha_pa": a_pa,
        "proj_AP_gt": proj_AP_gt, "proj_PA_gt": proj_PA_gt,
        "proj_AP_rec": proj_AP_rec, "proj_PA_rec": proj_PA_rec,
    }

# -----------------------------------------------------------------------------
# Forward-only helper (organ_ids + dynamic 'others')
# -----------------------------------------------------------------------------

def run_forward_projection(
    base_dir="/home/mnguest12/projects/thesis/PhantomGenerator",
    phantom_name="phantom_01",
    dims=(256,256,650),
    sigma=2.0,
    z0_slices=29,
    activity_ranges: Dict[str, Tuple[float,float]] = None,
    organ_groups: List[Tuple[str, List[str]]] = None,
    results_subdir: str = "results",
):
    """
    Nur Vorwärtsprojektion:
      - zieht Zufallsaktivitäten gemäß Ranges
      - baut ACT aus Maske + Gruppen
      - generiert AP/PA und speichert .npy
    """
    nx,ny,nz = dims
    phantom_dir = os.path.join(base_dir, phantom_name)
    out_dir = os.path.join(phantom_dir, results_subdir)
    os.makedirs(out_dir, exist_ok=True)

    fnameATN  = os.path.join(phantom_dir, f"{phantom_name}_ct.par_atn_1.bin")
    fnameMASK = os.path.join(phantom_dir, f"{phantom_name}_mask.par_act_1.bin")
    if not os.path.isfile(fnameATN):
        alt = fnameATN.replace("_1.bin", ".bin")
        if os.path.isfile(alt): fnameATN = alt
    if not os.path.isfile(fnameMASK):
        alt = fnameMASK.replace("_1.bin", ".bin")
        if os.path.isfile(alt): fnameMASK = alt

    if organ_groups is None:
        organ_groups = [
            ("liver",        ["liver"]),
            ("kidney",       ["lkidney","rkidney"]),
            ("prostate",     ["prostate"]),
            ("small_intest", ["small_intest"]),
            ("spleen",       ["spleen"]),
            ("others",       []),
        ]

    if activity_ranges is None:
        activity_ranges = {
            "liver":        (3,7),
            "kidney":       (10,35),
            "prostate":     (35,50),
            "small_intest": (3,5),
            "spleen":       (3,5),
            "others":       (0,0),
        }

    # Load once
    atn  = load_raw_volume(fnameATN,  (nx,ny,nz), dtype=np.float32)
    mask_ids = load_raw_volume(fnameMASK, (nx,ny,nz), dtype=np.float32).astype(np.uint32)
    kernel_mat = loadmat(os.path.join(base_dir, "LEAP_Kernel.mat"))["kernel_mat"].astype(np.float32)

    # Build IDs from organ_ids.txt (activity namespace)
    name2id_act = read_organ_ids_table(os.path.join(base_dir, "organ_ids.txt"))

    group_idlist = []
    used_ids = set()
    for gname, tokens in organ_groups:
        ids = []
        for nm, vid in name2id_act.items():
            nml = nm.lower()
            if any(tok.lower() in nml for tok in tokens if tok):
                ids.append(int(vid))
        ids = sorted(list(set(ids)))
        group_idlist.append((gname, ids))
        used_ids.update(ids)

    ids_present = np.unique(mask_ids.astype(np.int64))
    all_known_ids = set(int(v) for v in name2id_act.values())
    present_known = set(int(v) for v in ids_present if int(v) in all_known_ids)
    other_ids = sorted(list(present_known - used_ids))
    for i,(gname,ids) in enumerate(group_idlist):
        if gname.lower()=="others":
            group_idlist[i] = (gname, other_ids)
            break

    # Random ACT based on ranges
    rng = np.random.default_rng(1234)
    act = np.zeros((nx,ny,nz), dtype=np.float32)
    for (gname, ids) in group_idlist:
        lo, hi = activity_ranges.get(gname, (0,0))
        val = float(rng.uniform(lo, hi)) if hi > lo else 0.0
        if len(ids) and val > 0:
            act[np.isin(mask_ids, np.array(ids, dtype=mask_ids.dtype))] = val

    proj_AP, proj_PA = gamma_camera_core(
        act, atn, 'frontal', True, True, True,
        kernel_mat, nx, ny, nz, sigma, z0_slices
    )
    save_npy(os.path.join(out_dir, f"{phantom_name}_proj_AP.npy"), proj_AP.astype(np.float32))
    save_npy(os.path.join(out_dir, f"{phantom_name}_proj_PA.npy"), proj_PA.astype(np.float32))
    return {"proj_AP": proj_AP, "proj_PA": proj_PA}
