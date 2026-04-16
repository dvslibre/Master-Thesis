#!/usr/bin/env python3
"""
preprocessing_new.py — end-to-end converter + ray builder for RayNet
(Parallel geometry only; µ from .bin/.nii, Aktivität aus Organ-Maske)

Creates folder structure:
  <base>/
    src/   (raw files MOVED here)
    data/  (standardized .npy + meta.json)
    out/   (rays_train.npz)

NPZ payload for train.py:
  mu_seq   [M,N]    — µ sampled along each ray
  T_ap_seq [M,N]    — transmissions AP (source→detector)
  T_pa_seq [M,N]    — transmissions PA (opposite direction)
  I_pairs  [M,2]    — (I_AP, I_PA) at the same (y,x)
  a_seq    [M,N]    — activity sampled along each ray (synthetic, aus Maske)
  ds       float    — representative step length along ray
                     (either voxel units = 1.0, or physical mm/cm if requested)

Neu in dieser Version:
- Es gibt KEIN .act-Input mehr.
- Stattdessen:
    * µ-Volumen (.atn) wie bisher (µ per_mm oder per_cm)
    * Masken-Volumen (.bin) mit Organ-IDs (gleiche Geometrie wie µ)
    * organ_ids.txt mit Zeilen wie:  head = 1
- Aus der Maske wird ein Aktivitätsvolumen gebaut:
    * Default: Hintergrund = 0
    * Für bis zu 6 vordefinierte Organe (head, r_ear, l_ear, chest_surface,
      arm_upp_right, arm_low_right) wird je eine homogene Zufallsaktivität
      in einem organspezifischen Bereich gezogen.

Beispielaufruf:
cd RayNet/code

python preprocessing.py \
  --base /home/mnguest12/projects/thesis/RayNet/phantom_06 \
  --mu_bin /home/mnguest12/projects/thesis/RayNet/phantom_06/src/phantom_06_ct.par_atn_1.bin \
  --mu_shape 256,256,651 \
  --mu_dtype float32 \
  --mask_bin /home/mnguest12/projects/thesis/RayNet/phantom_06/src/phantom_06_mask.par_act_1.bin \
  --mask_shape 256,256,651 \
  --mask_dtype float32 \
  --kernel_mat /home/mnguest12/projects/thesis/RayNet/tools/LEAP_Kernel.mat \
  --kernel_var kernel_mat \
  --organ_ids_txt /home/mnguest12/projects/thesis/RayNet/tools/organ_ids.txt \
  --bin_order F \
  --mu_unit per_mm --mu_target_unit per_cm \
  --sd_mm 1.5 --sh_mm 1.5 --sw_mm 1.5 --use_physical_ds \
  --N 64 --subsample 1 \
  --no_clip_low --clip_to_one \
  --clamp_mu_min 0.0
"""

from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import itertools

# optionale Bibliotheken
try:
    import scipy.io as sio           # MATLAB <=v7.2
except Exception:
    sio = None
try:
    import h5py as h5                # MATLAB v7.3
except Exception:
    h5 = None
try:
    import nibabel as nib            # NIfTI (optional; falls --mu_nii benutzt wird)
except Exception:
    nib = None
try:
    from scipy.ndimage import zoom as nd_zoom, gaussian_filter
except Exception:
    nd_zoom = None
    gaussian_filter = None

try:
    from scipy.signal import convolve2d, fftconvolve
except Exception:
    convolve2d = None
    fftconvolve = None


# -----------------
# Dir helpers
# -----------------

def ensure_dirs(base: Path):
    src = base / "src"; data = base / "data"; out = base / "out"
    for d in (src, data, out): d.mkdir(parents=True, exist_ok=True)
    return src, data, out


def resolve_into_src(base: Path, src_dir: Path, p: Path) -> Path:
    """If file is in base root, MOVE it into src/ and return the new path."""
    if p.is_absolute():
        if not p.exists(): raise FileNotFoundError(p)
        return p
    cand = base / p
    if cand.exists():
        if src_dir in cand.parents: return cand
        dst = src_dir / cand.name
        if not dst.exists(): shutil.move(str(cand), str(dst))
        return dst
    cand2 = base / p.name
    if cand2.exists():
        dst = src_dir / cand2.name
        if not dst.exists(): shutil.move(str(cand2), str(dst))
        return dst
    raise FileNotFoundError(p)


# -----------------
# Loaders
# -----------------

def load_mat_2d(path: Path, var: Optional[str]) -> np.ndarray:
    """Load 2D MATLAB matrix (v7.2 via scipy, v7.3 via h5py)."""
    if sio is not None:
        try:
            md = sio.loadmat(path)
            if var is None:
                for k, v in md.items():
                    if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2:
                        return v.astype(np.float32)
                raise ValueError(f"No 2D array found in {path}")
            v = md[var]
            return v.astype(np.float32)
        except NotImplementedError:
            pass
        except OSError as e:
            if "Please use HDF reader" not in str(e): raise
    if h5 is None: raise RuntimeError("Install h5py for MATLAB v7.3 files.")
    with h5.File(path, "r") as f:
        if var and var in f:
            arr = np.array(f[var][()]).T
            return arr.astype(np.float32)
        for k in f.keys():
            arr = np.array(f[k][()])
            if arr.ndim == 2: return arr.T.astype(np.float32)
    raise ValueError(f"No 2D dataset found in {path}")


def load_act_bin(path: Path, shape_str: str, dtype: str = "float32", order: str = "C") -> np.ndarray:
    """
    Liest ein Volumen aus einer rohen .bin-Datei.
    shape_str ist 'x,y,z' (MATLAB-Export in Fortran-Order -> order='F').
    Gibt (D,H,W) = (y,z,x) zurück.
    """
    x, y, z = [int(s) for s in shape_str.split(",")]
    expected = x * y * z
    arr = np.fromfile(path, dtype=np.dtype(dtype), count=expected)
    if arr.size != expected:
        raise ValueError(f"{path.name}: size {arr.size} != {expected} from shape {(x,y,z)}")
    vol_xyz = arr.reshape(x, y, z, order=order)
    vol_dhw = np.transpose(vol_xyz, (1, 2, 0))  # (y,z,x)->(D,H,W)
    return vol_dhw


def load_mu_bin_and_align(path: Path, shape_str: str, dtype: str,
                          target_dhw: tuple[int,int,int],
                          sp_guess_dhw: tuple[float,float,float],
                          order: str = "C") -> tuple[np.ndarray, tuple[float,float,float], tuple[int,int,int]]:
    # µ-Volumen aus Binärdatei lesen und auf Zielgröße bringen.
    vol = load_act_bin(path, shape_str, dtype, order=order)
    D_src, H_src, W_src = vol.shape
    sp_dhw_src = sp_guess_dhw

    D_tgt, H_tgt, W_tgt = target_dhw
    if (D_src, H_src, W_src) == (D_tgt, H_tgt, W_tgt):
        return vol.astype(np.float32), sp_dhw_src, (D_src, H_src, W_src)

    if nd_zoom is None: raise RuntimeError("Install scipy for resampling.")
    zooms = (D_tgt / D_src, H_tgt / H_src, W_tgt / W_src)
    vol_res = nd_zoom(vol, zoom=zooms, order=1).astype(np.float32)
    sp_dhw_tgt = (sp_dhw_src[0]/zooms[0], sp_dhw_src[1]/zooms[1], sp_dhw_src[2]/zooms[2])
    return vol_res, sp_dhw_tgt, (D_src, H_src, W_src)


def load_mu_nii_and_align(path: Path,
                          target_HW: tuple[int,int]) -> tuple[np.ndarray, tuple[float,float,float], tuple[int,int,int]]:
    """
    CT/µ in NIfTI einlesen und auf Zielgröße bringen.
    D bleibt erhalten, nur H/W werden auf target_HW gebracht.
    """
    if nib is None: raise RuntimeError("Install nibabel to read NIfTI files.")
    img = nib.load(str(path))
    vol = img.get_fdata(dtype=np.float32)
    # NIfTI RAS (X,Y,Z) → (D,H,W)=(Y,Z,X)
    vol_dhw_src = np.transpose(vol, (1, 2, 0))
    D_src, H_src, W_src = vol_dhw_src.shape
    hdr = img.header
    sx, sy, sz = [float(hdr.get_zooms()[i]) for i in range(3)]
    sp_dhw_src = (sy, sz, sx)

    H_tgt, W_tgt = target_HW
    D_tgt = D_src  # D nicht anpassen
    if (H_src, W_src) == (H_tgt, W_tgt):
        return vol_dhw_src.astype(np.float32), sp_dhw_src, (D_src, H_src, W_src)

    if nd_zoom is None: raise RuntimeError("Install scipy for resampling.")
    zooms = (1.0, H_tgt / H_src, W_tgt / W_src)
    vol_res = nd_zoom(vol_dhw_src, zoom=zooms, order=1).astype(np.float32)
    sp_dhw_tgt = (sp_dhw_src[0]/zooms[0], sp_dhw_src[1]/zooms[1], sp_dhw_src[2]/zooms[2])
    return vol_res, sp_dhw_tgt, (D_src, H_src, W_src)


def load_bin_xyz(path: Path, shape_str: str, dtype: str = "float32", order: str = "F") -> np.ndarray:
    """Liest rohes BIN als (x,y,z) in gewünschter Speicherordnung (F=MATLAB, C=NumPy)."""
    x, y, z = [int(s) for s in shape_str.split(",")]
    arr = np.fromfile(path, dtype=np.dtype(dtype))
    expected = x * y * z
    if arr.size != expected:
        raise ValueError(f"{path.name}: size {arr.size} != {expected} from shape {(x,y,z)}")
    return arr.reshape(x, y, z, order=order)  # (x,y,z)


# -----------------
# Gamma-Kamera Forward Model (aus stratos.py übernommen)
# -----------------

USE_FFT_CONV = True

def _process_view_phys(act_data: np.ndarray, atn_data: np.ndarray,
                       comp_scatter: bool, atn_on: bool, coll_on: bool,
                       kernel_mat: np.ndarray, sigma: float,
                       z0_slices: int,
                       step_len: float = 1.0) -> np.ndarray:
    """
    Pipeline:
      (1) 2D-Gauss-Scatter pro Slice (xy)
      (2) Abschwächung exp(-cumsum(mu * step_len) entlang z)
      (3) z-tiefenabhängige Kollimator-Faltung ab z0+1
      (4) Projektion: Summe über z

    step_len: physikalische Schrittweite entlang der Projektionsrichtung
              (gleiche Längeneinheit wie 1/µ, z.B. cm wenn µ in 1/cm).
    """
    if gaussian_filter is None or convolve2d is None or fftconvolve is None:
        raise RuntimeError("scipy.ndimage.gaussian_filter und scipy.signal.{convolve2d,fftconvolve} werden benötigt.")

    # (1) Scatter
    if comp_scatter:
        act_sc = np.empty_like(act_data, dtype=np.float32)
        for z in range(act_data.shape[2]):
            act_sc[:, :, z] = gaussian_filter(act_data[:, :, z], sigma=sigma, mode='nearest')
    else:
        act_sc = act_data.astype(np.float32, copy=False)

    # (2) Attenuation mit physikalischer Schrittweite
    if atn_on:
        # atn_data: µ (z.B. 1/cm), step_len: z-Schritt in gleicher Einheit (z.B. cm)
        mu_cum = np.cumsum(atn_data * step_len, axis=2)
        vol_atn = act_sc * np.exp(-mu_cum)
    else:
        vol_atn = act_sc

    # (3) Kollimator-Faltung
    if coll_on:
        Z = vol_atn.shape[2]
        vol_coll = np.zeros_like(vol_atn, dtype=np.float32)
        conv2 = fftconvolve if USE_FFT_CONV else convolve2d
        for z in range(Z):
            if z <= z0_slices:
                vol_coll[:, :, z] = vol_atn[:, :, z]
            else:
                zz = min(z - z0_slices, kernel_mat.shape[2] - 1)
                K = kernel_mat[:, :, zz]
                vol_coll[:, :, z] = conv2(vol_atn[:, :, z], K, mode='same').astype(np.float32)
    else:
        vol_coll = vol_atn

    # (4) z-Summation
    return np.sum(vol_coll, axis=2)



def gamma_camera_core(act_data: np.ndarray, atn_data: np.ndarray,
                      view: str,
                      comp_scatter: bool, atn_on: bool, coll_on: bool,
                      kernel_mat: np.ndarray,
                      nx: int, ny: int, nz: int,
                      sigma: float,
                      z0_slices: int,
                      step_len: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gleiche Physik/Orientierung wie in stratos.py.
    Erwartet Volumina in Shape (nx,ny,nz) = (x,y,z).
    Gibt AP/PA-Projektionen als 2D-Arrays zurück.
    """
    assert act_data.shape == (nx, ny, nz) and atn_data.shape == (nx, ny, nz)
    if view != "frontal":
        raise ValueError("Nur 'frontal' ist implementiert (AP/PA).")

    # MATLAB-Pre-Rotationen
    act_ap = np.transpose(act_data, (0, 2, 1)); atn_ap = np.transpose(atn_data, (0, 2, 1))
    act_ap = np.rot90(act_ap);                   atn_ap = np.rot90(atn_ap)
    act_ap = np.transpose(act_ap, (1, 0, 2));    atn_ap = np.transpose(atn_ap, (1, 0, 2))
    act_pa = np.flip(act_ap, axis=2);            atn_pa = np.flip(atn_ap, axis=2)

    proj_AP = _process_view_phys(act_ap, atn_ap,
                             comp_scatter, atn_on, coll_on,
                             kernel_mat, sigma, z0_slices,
                             step_len=step_len)
    proj_PA = _process_view_phys(act_pa, atn_pa,
                             comp_scatter, atn_on, coll_on,
                             kernel_mat, sigma, z0_slices,
                             step_len=step_len)

    # Orientierungspatch wie in stratos.py
    def orient_patch(P):
        P1 = np.rot90(P); 
        P2 = np.flipud(P1)
        return P2.T

    return orient_patch(proj_AP), orient_patch(proj_PA)


# -----------------
# Projection normalization
# -----------------

def normalize_projections(ap: np.ndarray,
                          pa: np.ndarray,
                          percentile: float = 99.9,
                          clip_to_one: bool = True,
                          no_clip_low: bool = True):
    """
    Robust normalization by a high percentile.
    - If no_clip_low=True: DO NOT clamp small values to 0 → avoids hard zeros.
    - If clip_to_one=True: clamp maxima to 1.0 after scaling.
    """
    scale = np.percentile(np.concatenate([ap.ravel(), pa.ravel()]), percentile)
    ap_n = ap / (scale + 1e-12)
    pa_n = pa / (scale + 1e-12)
    if not no_clip_low:
        ap_n = np.clip(ap_n, 0, None)
        pa_n = np.clip(pa_n, 0, None)
    if clip_to_one:
        ap_n = np.minimum(ap_n, 1.0)
        pa_n = np.minimum(pa_n, 1.0)
    return ap_n.astype(np.float32), pa_n.astype(np.float32), float(scale)


# -----------------
# 1D-Resampling (parallel)
# -----------------

def _resample_1d(x: np.ndarray, N: int) -> np.ndarray:
    # streckt / staucht Sequenz auf feste Länge N
    L = x.shape[0]
    if L == N: return x.astype(np.float32, copy=True)
    xi = np.linspace(0, L - 1, num=N, dtype=np.float64)         # gleichm. verteilte Abtastpunkte im Originalsignal
    x0 = np.floor(xi).astype(np.int64)                          # linker Index d. Intervalls
    x1 = np.clip(x0 + 1, 0, L - 1)                              # rechter Index (geclipped, damit es nicht aus dem Array läuft)
    t = xi - x0                                                 # Interpolationsgewicht zw. x0 und x1
    return ((1.0 - t) * x[x0] + t * x[x1]).astype(np.float32)   # lin. Interpolation


def build_rays_parallel(ap,
                        pa,
                        mu_dhw,
                        act_dhw,
                        N,
                        subsample,
                        zscore,
                        use_physical_ds: bool,
                        sd_mm: float,
                        mu_unit: str):
    H, W = ap.shape                    # z.B. (256,651)
    D, Hm, Wm = mu_dhw.shape           # (256,256,651)
    assert (Hm, Wm) == (H, W), f"Mismatch: mu_dhw({mu_dhw.shape}) vs ap({ap.shape})"

    # alle y,x-Korrdinaten (ggf. ausgedünnt d. subsample)
    ys = range(0, H, subsample)
    xs = range(0, W, subsample)
    coords = [(y, x) for y in ys for x in xs]
    M = len(coords)                                 # Anz. Rays = Anz. betr. Pixel

    # Arrays für: µ-Sequenzen, Transmission vorwärts/rückwärts, Intensitätspaare, Aktivität
    mu_seq   = np.zeros((M, N), np.float32)
    T_ap_seq = np.zeros((M, N), np.float32)
    T_pa_seq = np.zeros((M, N), np.float32)
    I_pairs  = np.zeros((M, 2), np.float32)
    a_seq    = None if act_dhw is None else np.zeros((M, N), np.float32)

    # speichert die x,y-Position des Rays im Bild
    xy_pairs = np.zeros((M, 2), np.float32)

    # Schrittweite entlang der Tiefenachse
    if use_physical_ds:
        if mu_unit == "per_cm":
            ds = sd_mm / 10.0  # cm
        else:
            ds = sd_mm         # mm
    else:
        ds = 1.0  # voxel units

    # Schleife über alle ausgewählten Bildpixel --> pro Pixel ein Ray durchs Volumen
    for i, (y, x) in enumerate(coords):
        # 1D-Profil der Schwächung entlang der Tiefenachse D durch diesen Pixel
        mu_line = mu_dhw[:, y, x].astype(np.float64)
        # Auf feste Länge N resamplen (für batchbares NN-Input)
        mu_res  = _resample_1d(mu_line, N).astype(np.float64)
        # Integrierte Schwächung in AP-Richtung (von "oben nach unten") / PA
        tau_ap = np.cumsum(mu_res) * ds
        tau_pa = np.cumsum(mu_res[::-1]) * ds
        # µ-Sequenz speichern
        mu_seq[i]   = mu_res.astype(np.float32)
        # Transmission T = exp(-∫µ ds), einmal für AP, einmal für PA
        T_ap_seq[i] = np.exp(-tau_ap).astype(np.float32)
        T_pa_seq[i] = np.exp(-tau_pa[::-1]).astype(np.float32)
        # Gemessene Intensitäten für dieses Pixel aus den Projektionen (AP, PA)
        I_pairs[i, 0] = ap[y, x]
        I_pairs[i, 1] = pa[y, x]
        # Optional: Aktivitätsprofil entlang des Rays, resampled auf N und >= 0
        if a_seq is not None:
            a_seq[i] = np.maximum(_resample_1d(act_dhw[:, y, x], N), 0.0)

        # (x,y)-Koordinaten des Rays im Bild speichern
        xy_pairs[i, 0] = float(x)   # x-Koordinate
        xy_pairs[i, 1] = float(y)   # y-Koordinate

    # Optional: µ-Sequenzen global z-normalisieren (für stabileres Training)
    if zscore:
        mu_seq = ((mu_seq - mu_seq.mean()) / (mu_seq.std() + 1e-8)).astype(np.float32)

    # Rückgabe inkl. ds (Schrittweite) und xy_pairs (Ray-Positionen)
    return mu_seq, T_ap_seq, T_pa_seq, I_pairs, a_seq, float(ds), xy_pairs


# -----------------
# Unit conversion helpers (µ)
# -----------------

def convert_mu_units(mu_dhw: np.ndarray, src_unit: str, tgt_unit: str) -> np.ndarray:
    if src_unit == tgt_unit:
        return mu_dhw
    if src_unit == "per_mm" and tgt_unit == "per_cm":
        return (mu_dhw * 10.0).astype(np.float32)  # 1/mm → 1/cm
    if src_unit == "per_cm" and tgt_unit == "per_mm":
        return (mu_dhw / 10.0).astype(np.float32)  # 1/cm → 1/mm
    raise ValueError(f"Unsupported unit conversion {src_unit} → {tgt_unit}")


# -----------------
# Activity from mask
# -----------------

def build_activity_from_mask(mask_dhw: np.ndarray,
                             organ_ids_txt: Path,
                             rng_seed: int = 1234) -> tuple[np.ndarray, dict]:
    """
    Baut ein Aktivitätsvolumen aus einer Organ-Maske.

    """

    # 1) organ_ids.txt einlesen:  name = id
    name_to_id: dict[str, int] = {}
    if organ_ids_txt is not None and organ_ids_txt.exists():
        with open(organ_ids_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): 
                    continue
                if "=" not in line:
                    continue
                name, val = line.split("=")
                name = name.strip()
                val_i = int(val.strip())
                name_to_id[name] = val_i

    # 2) Default-Bereiche für Organe
    default_ranges = {
        "liver":            (3.0, 7.0),
        "lkidney":          (10.0, 35.0),
        "rkidney":          (10.0, 35.0),
        "prostate":         (35.0, 50.0),
        "small_intest":     (3.0, 5.0),
        "spleen":           (3.0, 5.0),
    }

    act = np.zeros(mask_dhw.shape, dtype=np.float32)
    rng = np.random.default_rng(rng_seed)
    organ_activity_info: dict[str, dict] = {}

    for organ_name, (low, high) in default_ranges.items():
        if organ_name not in name_to_id:
            continue
        organ_id = name_to_id[organ_name]
        # homogene Zufallsaktivität aus [low, high]
        val = float(rng.uniform(low, high))
        act[mask_dhw == organ_id] = val
        organ_activity_info[organ_name] = {
            "organ_id": int(organ_id),
            "range": [float(low), float(high)],
            "assigned_value": val,
        }

    return act, organ_activity_info


# -----------------
# CLI
# -----------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)

    # Maske (Organ-Labels)
    p.add_argument("--mask_bin", type=Path, required=True,
                   help="label volume (.bin) with organ IDs (x,y,z).")
    p.add_argument("--mask_shape", type=str, required=True,
                   help="shape for --mask_bin as 'x,y,z'")
    p.add_argument("--mask_dtype", type=str, default="int16",
                   help="dtype of mask volume (e.g. int16, uint8).")
    p.add_argument("--organ_ids_txt", type=Path, default=Path("src/organ_ids.txt"),
                   help="Textdatei mit Zeilen 'name = id'.")

    # Gamma-Kamera-Kernel (LEAP)
    p.add_argument("--kernel_mat", type=Path, default=Path("src/LEAP_Kernel.mat"),
                   help="MATLAB-Datei mit LEAP-Kernel (enthält 3D-Array).")
    p.add_argument("--kernel_var", type=str, default="kernel_mat",
                   help="Variablenname in der .mat-Datei (z.B. 'kernel_mat').")
    p.add_argument("--psf_sigma", type=float, default=2.0,
                   help="Sigma für den Scatter-Gauss (Pixel).")
    p.add_argument("--z0_slices", type=int, default=29,
                   help="Anzahl der 'ungefilterten' Schichten vor PSF-Faltung.")
    p.add_argument("--disable_scatter", action="store_true",
                   help="Scatter abschalten.")
    p.add_argument("--disable_attenuation", action="store_true",
                   help="Attenuation abschalten.")
    p.add_argument("--disable_collimator", action="store_true",
                   help="Kollimator-Faltung abschalten.")

    # µ-Quelle (genau EINE davon setzen):
    p.add_argument("--mu_bin", type=Path, default=None,
                   help="raw attenuation volume (.bin)")
    p.add_argument("--mu_shape", type=str, default=None,
                   help="shape for --mu_bin as x,y,z")
    p.add_argument("--mu_dtype", type=str, default="float32")
    p.add_argument("--mu_nii", type=Path, default=None,
                   help="NIfTI attenuation map (optional)")

    # µ-Einheiten & Spacings
    p.add_argument("--mu_unit", type=str, choices=["per_mm", "per_cm"], default="per_mm",
                   help="unit of input µ volume (1/mm or 1/cm)")
    p.add_argument("--mu_target_unit", type=str, choices=["per_mm", "per_cm"], default="per_mm",
                   help="convert µ to this unit before building rays")
    p.add_argument("--sd_mm", type=float, default=1.0, help="voxel spacing along D (mm)")
    p.add_argument("--sh_mm", type=float, default=1.0, help="voxel spacing along H (mm)")
    p.add_argument("--sw_mm", type=float, default=1.0, help="voxel spacing along W (mm)")
    p.add_argument("--use_physical_ds", action="store_true",
                   help="integrate with physical step (uses sd_mm and µ unit)")

    # BIN-Speicherordnung (MATLAB -> Fortran "F")
    p.add_argument("--bin_order", type=str, choices=["F", "C"], default="F",
                   help="Memory order for raw .bin volumes: 'F' (MATLAB) or 'C' (NumPy default).")

    # µ clamp
    p.add_argument("--clamp_mu_min", type=float, default=0.0,
                   help="Clamp µ to >= this value (e.g. 0.0).")

    # Ray sampling
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--subsample", type=int, default=4)
    p.add_argument("--zscore", action="store_true")

    # projection normalization
    p.add_argument("--proj_scale", type=float, default=1.0,
                   help="extra global factor after normalization (e.g., 1e5)")
    p.add_argument("--no_clip_low", action="store_true",
                   help="do NOT clamp small AP/PA values to 0 (avoid hard zeros)")
    p.add_argument("--clip_to_one", action="store_true",
                   help="clamp normalized AP/PA to <= 1.0 (upper cap)")
    p.add_argument(
        "--normalize_projections",
        action="store_true",
        help="If set, normalize AP/PA projections; otherwise keep raw values.")

    # Aktivitäts-RNG
    p.add_argument("--activity_seed", type=int, default=1234,
                   help="Random-Seed für die synthetische Aktivität.")

    # Optional kalibrierte Intensitäten
    p.add_argument("--calibrate", action="store_true",
                   help="If set, additionally write rays_train_calibrated.npz with normalized I_AP/I_PA.")

    # Geometrie-Flag (nur parallel)
    p.add_argument("--geom", type=str, choices=["parallel"], default="parallel",
                   help="ray model for transmissions (parallel only in this script)")

    return p.parse_args()


# -----------------
# Main
# -----------------

def main():
    args = parse_args()
    base = args.base
    src_dir, data_dir, out_dir = ensure_dirs(base)

    # Masken- und µ-Dateien in src/ auflösen
    mask_bin = resolve_into_src(base, src_dir, args.mask_bin)

    if args.mu_bin is None and args.mu_nii is None:
        raise ValueError("Provide either --mu_bin or --mu_nii for the attenuation map.")

    mu_bin_path = resolve_into_src(base, src_dir, args.mu_bin) if args.mu_bin else None
    mu_nii_path = resolve_into_src(base, src_dir, args.mu_nii) if args.mu_nii else None

    # 1) µ-Volumen laden – exakt wie in stratos.py: (nx,ny,nz) = (256,256,651)
    if args.mu_bin is None and args.mu_nii is None:
        raise ValueError("Provide --mu_bin for the attenuation map (NIfTI path not wired yet).")

    mu_bin_path = resolve_into_src(base, src_dir, args.mu_bin)
    mask_bin    = resolve_into_src(base, src_dir, args.mask_bin)

    # µ und Maske aus rohem BIN in Fortran-Order laden: (x,y,z) = (256,256,651)
    mu_xyz   = load_bin_xyz(mu_bin_path,   args.mu_shape,   args.mu_dtype,  order=args.bin_order).astype(np.float32)
    mask_xyz = load_bin_xyz(mask_bin,      args.mask_shape, args.mask_dtype, order=args.bin_order).astype(np.int32)

    if mu_xyz.shape != mask_xyz.shape:
        raise ValueError(f"Shape mismatch between µ {mu_xyz.shape} and mask {mask_xyz.shape}")

    nx, ny, nz = mu_xyz.shape  # sollte (256,256,651) sein
    mu_src_kind = "bin"
    sp_dhw = (args.sd_mm, args.sh_mm, args.sw_mm)
    dhw_src = (nx, ny, nz)

    # 2) Für RayNet: (D,H,W) := (nx,ny,nz)
    #    Interpretation der ersten Achse als "D" (Sampling-Richtung der Rays)
    mu_dhw   = mu_xyz.copy()       # (256,256,651)
    mask_dhw = mask_xyz.copy()     # (256,256,651)

    if args.clamp_mu_min is not None:
        mu_dhw = np.clip(mu_dhw.astype(np.float32), args.clamp_mu_min, None)

    # 3) Aktivität aus Maske bauen (synthetisch)
    act_dhw, organ_activity_info = build_activity_from_mask(
        mask_dhw=mask_dhw,
        organ_ids_txt=args.organ_ids_txt,
        rng_seed=args.activity_seed,
    )

    # 4) µ-Einheiten ggf. konvertieren (z.B. 1/mm → 1/cm)
    mu_dhw = convert_mu_units(mu_dhw, args.mu_unit, args.mu_target_unit)
    if args.clamp_mu_min is not None:
        mu_dhw = np.clip(mu_dhw.astype(np.float32), args.clamp_mu_min, None)

    # Physikalische Einheiten: z-Score von µ nicht sinnvoll
    if args.zscore and args.use_physical_ds:
        print("[WARN] Disabling --zscore for µ because --use_physical_ds is set (physical units).")
        zscore_mu = False
    else:
        zscore_mu = args.zscore

    # Zwischenspeichern
    np.save(data_dir / "act.npy",  act_dhw.astype(np.float32))
    np.save(data_dir / "mu.npy",   mu_dhw.astype(np.float32))
    np.save(data_dir / "mask.npy", mask_dhw.astype(np.int32))
   

    # 5) Gamma-Kamera: AP/PA-Projektionen aus act + µ simulieren (stratos-kompatibel)

    if sio is None:
        raise RuntimeError("scipy.io (sio) wird für das Laden des LEAP-Kernels benötigt.")
    kernel_mat_path = resolve_into_src(base, src_dir, args.kernel_mat)
    kernel_md = sio.loadmat(kernel_mat_path)
    if args.kernel_var not in kernel_md:
        raise KeyError(f"Variable '{args.kernel_var}' nicht in {kernel_mat_path} gefunden.")
    kernel_mat = kernel_md[args.kernel_var].astype(np.float32)

    # Volumina liegen bereits wie in stratos.py: (nx,ny,nz) = (256,256,651)
    act_xyz = act_dhw.astype(np.float32)
    mu_xyz  = mu_dhw.astype(np.float32)
    nx, ny, nz = act_xyz.shape

    comp_scatter = not args.disable_scatter
    atn_on       = not args.disable_attenuation
    coll_on      = not args.disable_collimator

    # Physikalische Schrittweite entlang der Projektionsrichtung (z-Achse)
    # hier sd_mm als Slice-Abstand angenommen
    if args.mu_target_unit == "per_cm":
        step_len = args.sd_mm / 10.0  # mm -> cm
    else:
        step_len = args.sd_mm        # in mm

    print(f"[INFO] Gamma-Projektor: mu_unit_out={args.mu_target_unit}, step_len={step_len:.4f}")

    proj_AP_xyz, proj_PA_xyz = gamma_camera_core(
        act_data=act_xyz,
        atn_data=mu_xyz,
        view="frontal",
        comp_scatter=comp_scatter,
        atn_on=atn_on,
        coll_on=coll_on,
        kernel_mat=kernel_mat,
        nx=nx, ny=ny, nz=nz,
        sigma=args.psf_sigma,
        z0_slices=args.z0_slices,
        step_len=step_len,
    )

    # Orientierung direkt wie in stratos: (H,W) = (256,651)
    ap_raw = proj_AP_xyz.astype(np.float32)
    pa_raw = proj_PA_xyz.astype(np.float32)

    H, W = ap_raw.shape   # sollte (256,651) sein

    # 6) Projektionen normalisieren (99.9-Perzentil) oder Rohwerte lassen
    if args.normalize_projections:
        ap_norm, pa_norm, scale_auto = normalize_projections(
            ap_raw, pa_raw, percentile=99.9,
            clip_to_one=args.clip_to_one,
            no_clip_low=args.no_clip_low
        )
    else:
        # Rohwerte übernehmen (nur auf float32 casten)
        ap_norm = ap_raw.astype(np.float32, copy=False)
        pa_norm = pa_raw.astype(np.float32, copy=False)
        scale_auto = 1.0  # keine automatische Skala

    # Optionaler zusätzlicher globaler Faktor (wie bisher)
    if args.proj_scale != 1.0:
        ap_norm *= args.proj_scale
        pa_norm *= args.proj_scale

    # Monitoring der Wertebereiche
    print("AP range:", ap_norm.min(), ap_norm.max(), "nonzero:", np.count_nonzero(ap_norm))
    print("PA range:", pa_norm.min(), pa_norm.max(), "nonzero:", np.count_nonzero(pa_norm))

    np.save(data_dir / "ap.npy", ap_norm)
    np.save(data_dir / "pa.npy", pa_norm)


    # 7) Meta-Infos für Nachvollziehbarkeit
    meta = {
        "projection_normalization": {
            "enabled": bool(args.normalize_projections),
            "percentile": 99.9 if args.normalize_projections else None,
            "auto_scale": float(scale_auto),
            "user_scale": float(args.proj_scale),
            "effective_scale": float(scale_auto * args.proj_scale),
            "ap_shape": [int(H), int(W)],
            "pa_shape": [int(H), int(W)],
            "ap_min": float(ap_norm.min()), "ap_max": float(ap_norm.max()),
            "pa_min": float(pa_norm.min()), "pa_max": float(pa_norm.max()),
            "no_clip_low": bool(args.no_clip_low),
            "clip_to_one": bool(args.clip_to_one),
        },
        "spacing_dhw_mm": {
            "sd": float(sp_dhw[0]),
            "sh": float(sp_dhw[1]),
            "sw": float(sp_dhw[2])
        },
        "mu_src_kind": mu_src_kind,
        "mu_src_shape_dhw": list(map(int, dhw_src)),
        "mu_tgt_shape_dhw": [int(mu_dhw.shape[0]), int(mu_dhw.shape[1]), int(mu_dhw.shape[2])],
        "mu_unit_in": args.mu_unit,
        "mu_unit_out": args.mu_target_unit,
        "use_physical_ds": bool(args.use_physical_ds),
        "activity_from_mask": {
            "mask_shape_dhw": [int(mask_dhw.shape[0]), int(mask_dhw.shape[1]), int(mask_dhw.shape[2])],
            "organ_activity_info": organ_activity_info,
        }
    }
    with open(data_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 7) Volumina orientieren, so dass (H,W) von µ/Aktivität zu den Projektionen passt
    # Ziel: mu_dhw.shape = (D, H, W) mit (H, W) = ap_norm.shape = (256, 651)
    H_ap, W_ap = ap_norm.shape
    D_mu, H_mu, W_mu = mu_dhw.shape

    # Fall, der aktuell auftritt: mu_dhw = (651, 256, 256), ap = (256, 651)
    # → wir wollen (256, 256, 651): Permutation (1, 2, 0)
    if (D_mu, H_mu, W_mu) == (651, 256, 256) and (H_ap, W_ap) == (256, 651):
        mu_dhw  = np.transpose(mu_dhw,  (1, 2, 0))   # (651,256,256) -> (256,256,651)
        act_dhw = np.transpose(act_dhw, (1, 2, 0))   # gleiche Drehung für Aktivität
        # falls du mask_dhw später noch brauchst:
        # mask_dhw = np.transpose(mask_dhw, (1, 2, 0))

        D_mu, H_mu, W_mu = mu_dhw.shape

    # Sicherheitscheck:
    assert (H_mu, W_mu) == (H_ap, W_ap), \
        f"Nach Re-Orientierung: mu_dhw({mu_dhw.shape}) passt nicht zu ap({ap_norm.shape})"

    # 8) Rays bauen (parallel)
    print("[SHAPES] mu_dhw", mu_dhw.shape, "act_dhw", act_dhw.shape, "ap", ap_norm.shape, "pa", pa_norm.shape)
    mu_seq, T_ap_seq, T_pa_seq, I_pairs, a_seq, ds, xy_pairs = build_rays_parallel(
        ap=ap_norm, pa=pa_norm,
        mu_dhw=mu_dhw, act_dhw=act_dhw,
        N=args.N, subsample=args.subsample, zscore=zscore_mu,
        use_physical_ds=args.use_physical_ds,
        sd_mm=sp_dhw[0],
        mu_unit=args.mu_target_unit,
    )

    # --- Rays ohne Aktivität filtern ---
    # nur die Rays behalten, bei denen max(a_true) > thr ist.
    if a_seq is not None:
        max_a = a_seq.max(axis=1)           # [M]
        thr = 1e-3                          # Schwellwert, ggf. anpassbar (z.B. 1e-4 oder 1e-2)
        keep = max_a > thr                  # Bool-Maske
        print(f"[FILTER] keep {keep.sum()}/{len(keep)} rays with max(a_true) > {thr}")

        mu_seq   = mu_seq[keep]
        T_ap_seq = T_ap_seq[keep]
        T_pa_seq = T_pa_seq[keep]
        I_pairs  = I_pairs[keep]
        a_seq    = a_seq[keep]
        xy_pairs = xy_pairs[keep]

    # 8) Speichern wie gehabt
    out_npz = out_dir / "rays_train.npz"
    np.savez_compressed(
        out_npz,
        mu_seq=mu_seq,
        T_ap_seq=T_ap_seq,
        T_pa_seq=T_pa_seq,
        I_pairs=I_pairs,
        a_seq=a_seq,
        ds=np.array(ds, np.float32),
        xy_pairs=xy_pairs,
    )
    print(f"[DATA] ap.npy {ap_norm.shape} pa.npy {pa_norm.shape}")
    print(f"[DATA] act.npy {act_dhw.shape} mu.npy {mu_dhw.shape} mask.npy {mask_dhw.shape} | spacing_dhw(mm)={sp_dhw}")
    print(f"[META] mu_unit_in={args.mu_unit} → mu_unit_out={args.mu_target_unit} | use_physical_ds={args.use_physical_ds}")
    print(f"[OUT] Saved rays → {out_npz} | geom=parallel | ds={ds:.4f} ({'phys' if args.use_physical_ds else 'vox'}) | "
          f"M={mu_seq.shape[0]} N={mu_seq.shape[1]}")

    # 8) Optional: Kalibrierte Variante der Intensitäten schreiben
    if args.calibrate:
        I_calib = I_pairs.astype(np.float32).copy()
        eps = 1e-8
        for c in range(I_calib.shape[1]):  # meist 2 Spalten: [I_AP, I_PA]
            max_val = float(I_calib[:, c].max())
            if max_val > 0:
                I_calib[:, c] /= (max_val + eps)

        out_npz_calib = out_dir / "rays_train_calibrated.npz"
        np.savez_compressed(
            out_npz_calib,
            mu_seq=mu_seq,
            T_ap_seq=T_ap_seq,
            T_pa_seq=T_pa_seq,
            I_pairs=I_calib,
            a_seq=a_seq,
            ds=np.array(ds, np.float32),
            xy_pairs=xy_pairs,
        )
        print(f"[OUT] Saved calibrated rays → {out_npz_calib}")


if __name__ == "__main__":
    main()
