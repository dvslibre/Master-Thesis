#!/usr/bin/env python3
"""
preprocessing.py

Preprocessing-Schritt für XCAT-Phantome mit
- physikalisch plausibler Intensitätsverteilung (Lu-177-PSMA, counts/voxel)
- Gamma-Kamera-Forwardmodell (Scatter + Kollimator)
- optionaler Poisson-Rauschsimulation in den Projektionen
- rein RELATIVE Projektionen (AP/PA), global normalisiert

Erwartete Ordnerstruktur:
  <base>/
    src/   -> enthält .bin-Dateien (SPECT-µ, CT-µ, Maske)
    out/   -> wird erstellt, enthält .npy-Ausgaben

Erzeugt im out/-Ordner:
  spect_att.npy   — SPECT-Attenuation-Volumen (z.B. bei 208 keV), (x,y,z)
  ct_att.npy      — CT-Attenuation-Volumen (z.B. bei 80 keV), (x,y,z)
  act.npy         — Intensitätsverteilung (counts/voxel), (x,y,z)
  ap.npy          — AP-Projektion, RELATIVE Intensität (0..1 oder ähnlich)
  pa.npy          — PA-Projektion, RELATIVE Intensität
  meta_simple.json — Meta-Infos inkl. Normalisierungsfaktor

Beispielaufruf:

python preprocessing.py \
  --base /home/mnguest12/projects/thesis/Data_Processing/phantom_01 \
  --spect_bin phantom_01_spect208keV.par_atn_1.bin \
  --ct_bin    phantom_01_ct80keV.par_atn_1.bin \
  --mask_bin  phantom_01_mask.par_act_1.bin \
  --shape 256,256,651 \
  --spect_dtype float32 \
  --ct_dtype    float32 \
  --mask_dtype  float32 \
  --mu_unit per_mm --mu_target_unit per_cm \
  --sd_mm 1.5 \
  --kernel_mat LEAP_Kernel.mat --kernel_var kernel_mat \
  --bin_order F \
  --percentile 99.9 --clip_to_one --activity_seed -1 \
  --poisson_max_counts 3000 --poisson_ref_percentile 99.5
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple
import json
import hashlib
import numpy as np

# optionale SciPy-Teile für Gamma-Kamera-Modell
try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

try:
    from scipy.signal import convolve2d, fftconvolve
except Exception:
    convolve2d = None
    fftconvolve = None

try:
    import scipy.io as sio
except Exception:
    sio = None


# -----------------
# Helper: Ordner
# -----------------

def ensure_dirs(base: Path) -> Tuple[Path, Path]:
    src = base / "src"
    out = base / "out"
    if not src.exists():
        raise FileNotFoundError(f"src-Ordner nicht gefunden: {src}")
    out.mkdir(parents=True, exist_ok=True)
    return src, out


def resolve_path(base: Path, src_dir: Path, p: Path) -> Path:
    """Datei auflösen:
    - falls absolut: direkt benutzen
    - falls relativ: zuerst in src/, dann in base/ suchen
    """
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    cand = src_dir / p
    if cand.exists():
        return cand
    cand2 = base / p
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"Datei nicht gefunden (weder in src noch in base): {p}")


# -----------------
# BIN-Loader
# -----------------

def load_bin_xyz(path: Path, shape_str: str, dtype: str = "float32", order: str = "F") -> np.ndarray:
    """Liest ein rohes BIN-Volumen als (x,y,z).

    shape_str: 'x,y,z', z.B. '256,256,651'
    dtype:     Datentyp im Binärfile (z.B. float32, int16)
    order:     'F' für MATLAB-(Fortran)-Order, 'C' für NumPy-Standard
    """
    x, y, z = [int(s) for s in shape_str.split(",")]
    arr = np.fromfile(path, dtype=np.dtype(dtype))
    expected = x * y * z
    if arr.size != expected:
        raise ValueError(f"{path.name}: size {arr.size} != {expected} aus shape {(x, y, z)}")
    vol = arr.reshape(x, y, z, order=order)  # (x,y,z)
    return vol.astype(np.float32)


# -----------------
# Aktivität aus Maske (Lu-177-PSMA)
# -----------------

def build_activity_from_mask(mask_xyz: np.ndarray,
                             organ_ids_txt: Path,
                             rng_seed: int = 1234) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Baut ein Aktivitätskonzentrationsvolumen aus einer Organ-Maske.

    mask_xyz: Volumen mit Organ-IDs (gleiche Geometrie wie SPECT/CT), (x,y,z)
    organ_ids_txt: Textdatei mit Zeilen der Form 'name = id'

    Rückgabe:
      act_xyz:   Intensitätswerte, gleiche Shape wie Maske
      mask_roi:  0/1-Maske aller 'relevanten' Organe (gleiche Shape)
      info:      Dictionary mit den zugewiesenen Werten pro Organ
    """
    # 1) organ_ids.txt einlesen: name = id
    name_to_id: Dict[str, int] = {}
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
    else:
        raise FileNotFoundError(f"organ_ids.txt nicht gefunden: {organ_ids_txt}")

    # 2) Plausible Intensitätsverteilung für Lu-177-PSMA (Counts/Voxel; Range von VOI-Auswertung d. vorliegenden klin. SPECT übernommen)
    #    Alle Organe, die hier drin stehen UND in organ_ids.txt vorkommen,
    #    gelten als "relevant" für die spätere 0/1-Maske.
    default_ranges_counts = {
        # Prostata-Hotspot
        "prostate":     (2000.0, 6000.0),
        # Nieren (kritisches Organ, typ. hohe Aufnahme)
        "lkidney":      (300.0, 3800.0),
        "rkidney":      (300.0, 3800.0),
        # Milz (relativ hoher Uptake, aber unter Niere)
        "spleen":       (250.0, 1800.0),
        # Leber (moderater Hintergrund)
        "liver":        (50.0, 600.0),
        # Dünndarm / intestinale Aufnahme
        "small_intest": (0.0, 400.0),
    }

    act = np.zeros(mask_xyz.shape, dtype=np.float32)
    # 0/1-Maske der "relevanten" Organe (ROI), gleiche Shape
    mask_roi = np.zeros(mask_xyz.shape, dtype=np.float32)

    rng = np.random.default_rng(rng_seed)
    organ_activity_info: Dict[str, Dict] = {}

    for organ_name, (low, high) in default_ranges_counts.items():
        if organ_name not in name_to_id:
            continue
        organ_id = name_to_id[organ_name]

        # homogene Zufallsaktivitätsverteilung aus [low, high]
        val_counts = float(rng.uniform(low, high))

        organ_voxels = (mask_xyz == organ_id)
        if not np.any(organ_voxels):
            continue

        act[organ_voxels] = val_counts
        mask_roi[organ_voxels] = 1.0  # alle diese Voxeln sind Teil der relevanten Organe

        organ_activity_info[organ_name] = {
            "organ_id": int(organ_id),
            "range_counts": [float(low), float(high)],
            "assigned_value_counts": val_counts,
        }

    return act, mask_roi, organ_activity_info



# -----------------
# µ-Einheiten
# -----------------

def convert_mu_units(mu_xyz: np.ndarray, src_unit: str, tgt_unit: str) -> np.ndarray:
    """µ-Einheiten umrechnen (1/mm <-> 1/cm)."""
    if src_unit == tgt_unit:
        return mu_xyz
    if src_unit == "per_mm" and tgt_unit == "per_cm":
        return (mu_xyz * 10.0).astype(np.float32)  # 1/mm -> 1/cm
    if src_unit == "per_cm" and tgt_unit == "per_mm":
        return (mu_xyz / 10.0).astype(np.float32)  # 1/cm -> 1/mm
    raise ValueError(f"Unsupported unit conversion {src_unit} -> {tgt_unit}")


# -----------------
# Gamma-Kamera-Modell
# -----------------

USE_FFT_CONV = True


def _process_view_phys(act_data: np.ndarray, atn_data: np.ndarray,
                       comp_scatter: bool, atn_on: bool, coll_on: bool,
                       kernel_mat: np.ndarray, sigma: float,
                       z0_slices: int,
                       step_len: float = 1.0) -> np.ndarray:
    """Einfache Gamma-Kamera-Physik für eine Sicht.

    Pipeline:
      (1) 2D-Gauss-Scatter pro Slice (xy)
      (2) Abschwächung exp(-cumsum(mu * step_len) entlang z)
      (3) z-tiefenabhängige Kollimator-Faltung ab z0+1
      (4) Projektion: Summe über z

    step_len: physikalische Schrittweite entlang der Projektionsrichtung
              (gleiche Längeneinheit wie 1/µ, z.B. cm wenn µ in 1/cm).

    WICHTIG:
      - globaler Counts-Faktor wird hier nicht eingebaut (relative Einheiten)
    """
    if gaussian_filter is None or convolve2d is None or fftconvolve is None:
        raise RuntimeError("Für das Gamma-Kamera-Modell werden scipy.ndimage.gaussian_filter "
                           "und scipy.signal.{convolve2d,fftconvolve} benötigt.")

    # (1) Scatter
    if comp_scatter:
        act_sc = np.empty_like(act_data, dtype=np.float32)
        for z in range(act_data.shape[2]):
            act_sc[:, :, z] = gaussian_filter(act_data[:, :, z],
                                              sigma=sigma, mode="nearest")
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
                vol_coll[:, :, z] = conv2(vol_atn[:, :, z], K,
                                          mode="same").astype(np.float32)
    else:
        vol_coll = vol_atn

    # (4) z-Summation
    return np.sum(vol_coll, axis=2)


def gamma_camera_core(act_data: np.ndarray, atn_data: np.ndarray,
                      kernel_mat: np.ndarray,
                      sigma: float,
                      z0_slices: int,
                      step_len: float = 1.0,
                      comp_scatter: bool = True,
                      atn_on: bool = True,
                      coll_on: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Gamma-Kamera-Modell für AP/PA (wie in stratos.py).

    Erwartet Volumina in Shape (nx,ny,nz) = (x,y,z).
    Gibt AP/PA-Projektionen als 2D-Arrays zurück (relative Einheiten).
    """
    nx, ny, nz = act_data.shape
    assert atn_data.shape == (nx, ny, nz)

    # MATLAB-Pre-Rotationen (wie im Originalcode)
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

    # Orientierungspatch (wie in stratos.py)
    def orient_patch(P: np.ndarray) -> np.ndarray:
        P1 = np.rot90(P)
        P2 = np.flipud(P1)
        return P2.T

    return orient_patch(proj_AP), orient_patch(proj_PA)


# -----------------
# Projektionen normieren (rein relativ)
# -----------------

def normalize_projections(ap_raw: np.ndarray,
                          pa_raw: np.ndarray,
                          percentile: float = 99.9,
                          clip_to_one: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """Skaliert AP/PA-Projektionen robust auf einen relativen Bereich.

    - scale = gemeinsames Perzentil von AP/PA (z.B. 99.9%)
    - Normierung: ap_n = ap_raw / scale
    - negatives auf 0 clampen
    - optional ap_n, pa_n auf <= 1.0 clippen

    Rückgabe:
      ap_n, pa_n, scale
    so dass gilt:
      ap_raw ≈ ap_n * scale
      pa_raw ≈ pa_n * scale
    """
    stacked = np.concatenate([ap_raw.ravel(), pa_raw.ravel()])
    scale = np.percentile(stacked, percentile)
    if scale <= 0:
        scale = 1.0

    ap_n = ap_raw / (scale + 1e-12)
    pa_n = pa_raw / (scale + 1e-12)

    # negative Werte auf 0 (numerische Artefakte)
    ap_n = np.clip(ap_n, 0.0, None)
    pa_n = np.clip(pa_n, 0.0, None)

    if clip_to_one:
        ap_n = np.minimum(ap_n, 1.0)
        pa_n = np.minimum(pa_n, 1.0)

    return ap_n.astype(np.float32), pa_n.astype(np.float32), float(scale)


# -----------------
# CLI
# -----------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Converter: BIN -> spect_att.npy, ct_att.npy, act.npy (counts), ap.npy, pa.npy (relative, normalisiert)"
    )
    p.add_argument("--base", type=Path, required=True,
                   help="Basisordner des Phantoms (z.B. /pfad/zu/phantom_01)")

    # Dateinamen (relativ zu base/src oder absolut)
    p.add_argument("--spect_bin", type=Path, required=True,
                   help="BIN-Datei für SPECT-Attenuation (z.B. atn208kev.bin)")
    p.add_argument("--ct_bin", type=Path, required=True,
                   help="BIN-Datei für CT-Attenuation (z.B. atn80kev.bin)")
    p.add_argument("--mask_bin", type=Path, required=True,
                   help="BIN-Datei für Maske (Organ-IDs)")

    # gemeinsame Geometrie
    p.add_argument("--shape", type=str, required=True,
                   help="Volumen-Shape als 'x,y,z' (z.B. 256,256,651)")

    # Datentypen
    p.add_argument("--spect_dtype", type=str, default="float32",
                   help="dtype der SPECT-Bin-Datei (default: float32)")
    p.add_argument("--ct_dtype", type=str, default="float32",
                   help="dtype der CT-Bin-Datei (default: float32)")
    p.add_argument("--mask_dtype", type=str, default="float32",
                   help="dtype der Masken-Bin-Datei (default: float32)")

    # µ-Einheiten & Voxelabstand (für SPECT-µ)
    p.add_argument("--mu_unit", type=str, choices=["per_mm", "per_cm"], default="per_mm",
                   help="Einheit des SPECT-µ-Volumens (1/mm oder 1/cm)")
    p.add_argument("--mu_target_unit", type=str, choices=["per_mm", "per_cm"], default="per_cm",
                   help="Ziel-Einheit für µ vor dem Gamma-Kamera-Modell")
    p.add_argument("--sd_mm", type=float, default=1.5,
                   help="Voxelspacing entlang der Projektionsrichtung (z-Achse) in mm")

    # organ_ids.txt (im Überverzeichnis von base)
    p.add_argument("--organ_ids_txt", type=Path, default=Path("organ_ids.txt"),
                   help="organ_ids.txt (wird im Überverzeichnis von base gesucht)")

    # Speicherordnung (MATLAB-Export -> 'F')
    p.add_argument("--bin_order", type=str, choices=["F", "C"], default="F",
                   help="Speicherordnung der .bin-Dateien: 'F' (MATLAB) oder 'C' (NumPy)")

    # RNG-Seed für Aktivität
    p.add_argument("--activity_seed", type=int, default=1234,
                   help="Random-Seed für die Aktivitätszuweisung aus der Maske. "
                        "<0: deterministisch aus Phantom-Namen abgeleitet")

    # Gamma-Kamera-Kernel
    p.add_argument("--kernel_mat", type=Path, required=True,
                   help="MATLAB-Datei mit LEAP-Kernel (enthält 3D-Array)")
    p.add_argument("--kernel_var", type=str, default="kernel_mat",
                   help="Variablenname in der .mat-Datei (z.B. 'kernel_mat')")
    p.add_argument("--psf_sigma", type=float, default=2.0,
                   help="Sigma für den Scatter-Gauss (Pixel)")
    p.add_argument("--z0_slices", type=int, default=29,
                   help="Anzahl der 'ungefilterten' Schichten vor PSF-Faltung")

    # Poisson-Rauschen
    p.add_argument("--poisson_max_counts", type=float, default=3000.0,
                   help="Ziel-Maximum (Perzentil-basiert) der simulierten Counts in den Projektionen. "
                        "Wenn <=0, kein Poisson-Rauschen.")
    p.add_argument("--poisson_ref_percentile", type=float, default=99.5,
                   help="Perzentil der rohen Projektionen, das auf poisson_max_counts gemappt wird.")

    # AP/PA-Normierung
    p.add_argument("--percentile", type=float, default=99.9,
                   help="Perzentil für die robuste Normierung (z.B. 99.9)")
    p.add_argument("--clip_to_one", action="store_true",
                   help="Wenn gesetzt, werden AP/PA nach Normierung auf <=1.0 gecappt.")

    return p.parse_args()


# -----------------
# Main
# -----------------

def main():
    args = parse_args()
    base = args.base.resolve()
    src_dir, out_dir = ensure_dirs(base)

    # Pfade auflösen
    spect_bin_path = resolve_path(base, src_dir, args.spect_bin)
    ct_bin_path    = resolve_path(base, src_dir, args.ct_bin)
    mask_bin_path  = resolve_path(base, src_dir, args.mask_bin)

    # organ_ids.txt + Kernel immer im Überverzeichnis von base
    organ_ids_path = base.parent / args.organ_ids_txt
    if not organ_ids_path.exists():
        raise FileNotFoundError(f"organ_ids.txt nicht im Überverzeichnis gefunden: {organ_ids_path}")

    kernel_mat_path = base.parent / args.kernel_mat
    if not kernel_mat_path.exists():
        raise FileNotFoundError(f"LEAP_Kernel.mat nicht im Überverzeichnis gefunden: {kernel_mat_path}")

    print(f"[INFO] base:      {base}")
    print(f"[INFO] src:       {src_dir}")
    print(f"[INFO] out:       {out_dir}")
    print(f"[INFO] spect_bin: {spect_bin_path}")
    print(f"[INFO] ct_bin:    {ct_bin_path}")
    print(f"[INFO] mask_bin:  {mask_bin_path}")
    print(f"[INFO] shape:     {args.shape}")
    print(f"[INFO] organ_ids: {organ_ids_path}")
    print(f"[INFO] kernel:    {kernel_mat_path} (var='{args.kernel_var}')")

    # Volumina laden (x,y,z)
    spect_xyz = load_bin_xyz(spect_bin_path, args.shape,
                             dtype=args.spect_dtype, order=args.bin_order)
    ct_xyz    = load_bin_xyz(ct_bin_path,    args.shape,
                             dtype=args.ct_dtype,    order=args.bin_order)
    mask_xyz  = load_bin_xyz(mask_bin_path,  args.shape,
                             dtype=args.mask_dtype,  order=args.bin_order)

    if not (spect_xyz.shape == ct_xyz.shape == mask_xyz.shape):
        raise ValueError(f"Shape-Mismatch: spect={spect_xyz.shape}, ct={ct_xyz.shape}, mask={mask_xyz.shape}")

    print(f"[SHAPE] Volumina: {spect_xyz.shape} (x,y,z)")

    # Seed wählen: fix oder aus Phantom-Namen abgeleitet
    if args.activity_seed < 0:
        h = hashlib.sha256(base.name.encode("utf-8")).hexdigest()
        auto_seed = int(h[:8], 16)  # 32-bit Seed
        print(f"[INFO] Auto-Seed aus Phantom-Namen: {auto_seed}")
        rng_seed = auto_seed
    else:
        rng_seed = args.activity_seed

    # Aktivität aus Maske (counts) + 0/1-Organmaske der "relevanten" Organe
    act_xyz, mask_roi_xyz, organ_info = build_activity_from_mask(
        mask_xyz=mask_xyz,
        organ_ids_txt=organ_ids_path,
        rng_seed=rng_seed,
    )

    print(f"[ACT] act_xyz: shape={act_xyz.shape}, "
          f"min={act_xyz.min():.1f} counts/voxel, max={act_xyz.max():.1f} counts/voxel, "
          f"mean={act_xyz.mean():.1f} counts/voxel")

    # SPECT-µ in Ziel-Einheit bringen
    mu_xyz = convert_mu_units(spect_xyz, args.mu_unit, args.mu_target_unit)

    # Schrittweite entlang Projektionsrichtung (z-Achse)
    if args.mu_target_unit == "per_cm":
        step_len = args.sd_mm / 10.0  # mm -> cm
    else:
        step_len = args.sd_mm        # in mm

    print(f"[INFO] Gamma-Projektor: mu_unit_in={args.mu_unit} "
          f"-> mu_unit_out={args.mu_target_unit}, step_len={step_len:.4f}")

    # Kernel laden
    if sio is None:
        raise RuntimeError("scipy.io (sio) wird für das Laden des LEAP-Kernels benötigt.")
    kernel_md = sio.loadmat(kernel_mat_path)
    if args.kernel_var not in kernel_md:
        raise KeyError(f"Variable '{args.kernel_var}' nicht in {kernel_mat_path} gefunden.")
    kernel_mat = kernel_md[args.kernel_var].astype(np.float32)

    # Gamma-Kamera-Projektionen simulieren (AP/PA), relative Einheiten
    ap_raw, pa_raw = gamma_camera_core(
        act_data=act_xyz.astype(np.float32),
        atn_data=mu_xyz.astype(np.float32),
        kernel_mat=kernel_mat,
        sigma=args.psf_sigma,
        z0_slices=args.z0_slices,
        step_len=step_len,
        comp_scatter=True,
        atn_on=True,
        coll_on=True,
    )

    print("[RANGE] AP raw:", ap_raw.min(), ap_raw.max(),
          "PA raw:", pa_raw.min(), pa_raw.max())

    # ---------------------------------------------------------
    # Poisson-Rauschen (szintigraphie-gemäß)
    # ---------------------------------------------------------
    if args.poisson_max_counts > 0:
        stacked_raw = np.concatenate([ap_raw.ravel(), pa_raw.ravel()])
        ref_int = np.percentile(stacked_raw, args.poisson_ref_percentile)
        if ref_int <= 0:
            ref_int = stacked_raw.max()
        if ref_int <= 0:
            ref_int = 1.0

        scale_to_counts = args.poisson_max_counts / ref_int

        lam_ap = np.clip(ap_raw * scale_to_counts, 0.0, None)
        lam_pa = np.clip(pa_raw * scale_to_counts, 0.0, None)

        rng_poiss = np.random.default_rng(rng_seed + 1)
        ap_counts = rng_poiss.poisson(lam_ap).astype(np.float32)
        pa_counts = rng_poiss.poisson(lam_pa).astype(np.float32)

        print("[POISSON] ref_int (Perzentil "
              f"{args.poisson_ref_percentile}) = {ref_int:.4e}")
        print(f"[POISSON] scale_to_counts = {scale_to_counts:.4e}")
        print("[POISSON] AP counts: min/max/mean = "
              f"{ap_counts.min():.1f} / {ap_counts.max():.1f} / {ap_counts.mean():.1f}")
        print("[POISSON] PA counts: min/max/mean = "
              f"{pa_counts.min():.1f} / {pa_counts.max():.1f} / {pa_counts.mean():.1f}")

        ap_for_norm = ap_counts
        pa_for_norm = pa_counts
    else:
        print("[POISSON] Kein Poisson-Rauschen (poisson_max_counts <= 0).")
        ap_for_norm = ap_raw
        pa_for_norm = pa_raw

    # AP/PA robust normalisieren (rein relative Intensität)
    ap_norm, pa_norm, scale_auto = normalize_projections(
        ap_raw=ap_for_norm,
        pa_raw=pa_for_norm,
        percentile=args.percentile,
        clip_to_one=args.clip_to_one,
    )

    print("[RANGE] AP norm:", ap_norm.min(), ap_norm.max(),
          "PA norm:", pa_norm.min(), pa_norm.max())
    print(f"[INFO] verwendeter Normalisierungsfaktor (Perzentil {args.percentile}): {scale_auto:.4e}")

    # Speichern als .npy im out-Ordner
    np.save(out_dir / "spect_att.npy", spect_xyz.astype(np.float32))
    np.save(out_dir / "ct_att.npy",    ct_xyz.astype(np.float32))
    np.save(out_dir / "act.npy",       act_xyz.astype(np.float32))
    np.save(out_dir / "ap.npy",        ap_norm.astype(np.float32))
    np.save(out_dir / "pa.npy",        pa_norm.astype(np.float32))
    np.save(out_dir / "mask.npy",      mask_roi_xyz.astype(np.float32))  # NEU: 0/1-Organmaske

    # Meta-Info
    meta = {
        "shape_xyz": list(map(int, spect_xyz.shape)),
        "spect_bin": str(spect_bin_path),
        "ct_bin":    str(ct_bin_path),
        "mask_bin":  str(mask_bin_path),
        "organ_ids": str(organ_ids_path),
        "activity_unit": "counts/voxel",
        "lu177_psma_like": True,
        "organ_activity_info": organ_info,
        "has_binary_roi_mask": True,
        "roi_mask_file": "mask.npy",
        "mu_unit_in": args.mu_unit,
        "mu_unit_out": args.mu_target_unit,
        "sd_mm": float(args.sd_mm),
        "step_len": float(step_len),
        "kernel_mat": str(kernel_mat_path),
        "kernel_var": args.kernel_var,
        "projection_normalization": {
            "percentile": float(args.percentile),
            "auto_scale": float(scale_auto),
            "clip_to_one": bool(args.clip_to_one),
            "ap_raw_min": float(ap_raw.min()),
            "ap_raw_max": float(ap_raw.max()),
            "pa_raw_min": float(pa_raw.min()),
            "pa_raw_max": float(pa_raw.max()),
            "ap_norm_min": float(ap_norm.min()),
            "ap_norm_max": float(ap_norm.max()),
            "pa_norm_min": float(pa_norm.min()),
            "pa_norm_max": float(pa_norm.max()),
            "units_after_norm": "relative (counts/scale_auto)",
            "poisson_max_counts": float(args.poisson_max_counts),
            "poisson_ref_percentile": float(args.poisson_ref_percentile),
        },
    }
    with open(out_dir / "meta_simple.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OUT] Gespeichert in {out_dir}: "
          f"spect_att.npy, ct_att.npy, act.npy, ap.npy, pa.npy, meta_simple.json")


if __name__ == "__main__":
    main()
