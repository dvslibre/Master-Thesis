#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocessing.py

Kurzueberblick:
Laedt SPECT/CT/Masken-BINs, baut Aktivitaetsvolumen, simuliert Gamma-Kamera-
Projektionen und speichert Volumina/Projektionen als .npy plus meta_simple.json.
Eingaben: µ-Volumina, Organmaske, LEAP-Kernel, Geometrie/Einheiten.
Ausgaben: Volumina roh/norm und Projektionen roh/counts/norm (QA + NN-Input).
Einheiten: act_xyz in kBq/mL, A_xyz_MBq in MBq/voxel, ap/pa in MBq-aequiv./Counts.

Preprocessing-Schritt für XCAT-Phantome mit
- physikalisch plausibler Aktivitätskonzentration (Lu-177-PSMA)
- Gamma-Kamera-Forwardmodell (Scatter + Kollimator)
- keine Rauschsimulation in den Projektionen
- MBq-aequivalente Projektionen (AP/PA), global normalisiert

Erwartete Ordnerstruktur:
  <base>/
    src/   -> enthält .bin-Dateien (SPECT-µ, CT-µ, Maske)
    out/   -> wird erstellt, enthält .npy-Ausgaben

Erzeugt im out/-Ordner:
  spect_att.npy   — SPECT-Attenuation-Volumen (z.B. bei 208 keV), (x,y,z)
  ct_att.npy      — CT-Attenuation-Volumen (z.B. bei 80 keV), (x,y,z)
  act.npy         — Aktivitätskonzentration (kBq/mL), (x,y,z)
  spect_att_norm.npy — SPECT-Attenuation-Volumen, robust normiert (p99.9)
  ct_att_norm.npy    — CT-Attenuation-Volumen, robust normiert (p99.9)
  act_norm.npy       — Aktivitätskonzentration, robust normiert (p99.9)
  ap_counts.npy   — AP-Projektion als Counts (nach Sensitivität)
  pa_counts.npy   — PA-Projektion als Counts (nach Sensitivität)
  ap.npy          — AP-Projektion, robust normiert (joint p99.9, NN-Input)
  pa.npy          — PA-Projektion, robust normiert (joint p99.9, NN-Input)
  meta_simple.json — Meta-Infos inkl. Normalisierungsfaktoren

Beispielaufruf:

python3 preprocessing.py \
  --base /home/mnguest12/projects/thesis/Data_Processing/phantom_28 \
  --spect_bin phantom_28_spect208keV.par_atn_1.bin \
  --ct_bin    phantom_28_ct80keV.par_atn_1.bin \
  --mask_bin  phantom_28_mask.par_act_1.bin \
  --shape 256,256,651 \
  --spect_dtype float32 \
  --ct_dtype    float32 \
  --mask_dtype  float32 \
  --mu_unit per_mm --mu_target_unit per_cm \
  --sd_mm 1.5 \
  --kernel_mat LEAP_Kernel.mat --kernel_var kernel_mat \
  --bin_order F \
  --activity_seed -1 \
  --sensitivity_cps_per_mbq 65 \
  --acq_time_s 300 \
  --manifest /home/mnguest12/projects/thesis/pieNeRF/data/manifest_abs.csv \
  --patient-id phantom_28 \
  --manifest-id-column patient_id


"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple
import json
import csv
import hashlib
import numpy as np
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None
try:
    from PIL import Image
except Exception:
    Image = None

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


def update_manifest_scale(manifest_path: Path, patient_id: str, scale: float, id_column: str = "patient_id"):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv nicht gefunden: {manifest_path}")
    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "proj_scale_joint_p99" not in fieldnames:
            fieldnames.append("proj_scale_joint_p99")
        for row in reader:
            rows.append(row)

    updated = False
    for row in rows:
        if row.get(id_column) == patient_id:
            row["proj_scale_joint_p99"] = f"{float(scale):.8e}"
            updated = True
            break

    if not updated:
        available = [row.get(id_column, "") for row in rows]
        raise ValueError(
            f"patient_id '{patient_id}' nicht im manifest gefunden: {manifest_path}. "
            f"Verfuegbare IDs: {available}"
        )

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_image_backend():
    """Stellt sicher, dass wir PNGs schreiben können."""
    if imageio is None and Image is None:
        raise RuntimeError("Weder imageio noch Pillow verfügbar – PNG-Export nicht möglich.")


def save_png(arr2d: np.ndarray, path: Path):
    """Speichert ein 2D-Array als PNG (0-255 skaliert)."""
    _ensure_image_backend()
    arr = np.asarray(arr2d)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        scaled = np.zeros_like(arr, dtype=np.uint8)
    else:
        valid = arr[finite_mask]
        vmin = valid.min()
        vmax = valid.max()
        if vmax > vmin:
            scaled = (np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            scaled = np.zeros_like(arr, dtype=np.uint8)
    if imageio is not None:
        imageio.imwrite(path, scaled)
    else:
        Image.fromarray(scaled).save(path)


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


def normalize_volume_p999(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Normiert ein Volumen via robustem p99.9-Scale über alle Voxels.

    Inputs:
      x: 3D-Volumen (µ oder Aktivitaet), beliebige Einheit.
    Output:
      x_norm: robust normiert, negativ geklippt.
      scale: p99.9-Scale (Fallback auf 1.0 falls ungueltig).
      p999: p99.9 von x_norm (QA-Check, typ. ~1).

    Physik/Signal:
      Getrennte Scales pro Modalitaet, damit µ- und Aktivitaetsbereiche
      nicht vermischt werden; normierte Varianten sind fuer NN-Inputs.
    """
    x_clip = np.clip(x, 0, None)
    scale = float(np.percentile(x_clip, 99.9))
    if (not np.isfinite(scale)) or scale <= 0.0:
        scale = 1.0
    x_norm = np.clip(x_clip / scale, 0, None).astype(np.float32)
    p999 = float(np.percentile(x_norm, 99.9))
    return x_norm, scale, p999


# -----------------
# BIN-Loader
# -----------------

def load_bin_xyz(path: Path, shape_str: str, dtype: str = "float32", order: str = "F") -> np.ndarray:
    """Liest ein rohes BIN-Volumen als (x,y,z).

    Inputs:
      path: Datei mit flachem Binär-Array.
      shape_str: 'x,y,z', z.B. '256,256,651' (Geometrie in Voxeln).
      dtype: Datentyp im Binärfile (z.B. float32, int16).
      order: 'F' für MATLAB-(Fortran)-Order, 'C' für NumPy-Standard.
    Output:
      vol: (x,y,z) als float32, Einheiten bleiben unveraendert.
    """
    x, y, z = [int(s) for s in shape_str.split(",")]
    arr = np.fromfile(path, dtype=np.dtype(dtype))
    expected = x * y * z
    if arr.size != expected:
        raise ValueError(f"{path.name}: size {arr.size} != {expected} aus shape {(x, y, z)}")
    vol = arr.reshape(x, y, z, order=order)  # (x,y,z)
    return vol.astype(np.float32)


# -----------------
# Aktivität aus Maske (Lu-177-PSMA, kBq/mL)
# -----------------

def build_activity_from_mask(mask_xyz: np.ndarray,
                             organ_ids_txt: Path,
                             rng_seed: int = 1234) -> Tuple[np.ndarray, Dict]:
    """Baut ein Aktivitätskonzentrationsvolumen aus einer Organ-Maske.

    Inputs:
      mask_xyz: Volumen mit Organ-IDs (x,y,z), gleiche Geometrie wie SPECT/CT.
      organ_ids_txt: Textdatei mit Zeilen der Form 'name = id'.
      rng_seed: Seed fuer die organspezifischen Aktivitaetswerte.

    Rückgabe:
      act_xyz: Aktivitaetskonzentration (kBq/mL), gleiche Shape wie Maske.
      info: Dictionary mit den zugewiesenen Werten pro Organ.

    Physik/Modell:
      Jedem Organ wird eine homogene Aktivitaet aus plausiblen
      Lu-177-PSMA-Bereichen zugewiesen.
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

    # 2) Plausible Aktivitätskonzentrationen für Lu-177-PSMA (kBq/mL)
    default_ranges_kBqml = {
        # Tumor/Prostata-Hotspot
        "prostate":     (2000.0, 6000.0),
        # Nieren (kritisches Organ, typ. hohe Aufnahme)
        "lkidney":      (1000.0, 3000.0),
        "rkidney":      (1000.0, 3000.0),
        # Milz (relativ hoher Uptake, aber unter Niere)
        "spleen":       (400.0, 1000.0),
        # Leber (moderater Hintergrund)
        "liver":        (200.0, 600.0),
    }

    act = np.zeros(mask_xyz.shape, dtype=np.float32)
    rng = np.random.default_rng(rng_seed)
    organ_activity_info: Dict[str, Dict] = {}

    for organ_name, (low, high) in default_ranges_kBqml.items():
        if organ_name not in name_to_id:
            continue
        organ_id = name_to_id[organ_name]
        # homogene Zufallsaktivitätskonzentration aus [low, high] kBq/mL
        val_kBqml = float(rng.uniform(low, high))
        act[mask_xyz == organ_id] = val_kBqml
        organ_activity_info[organ_name] = {
            "organ_id": int(organ_id),
            "range_kBq_per_ml": [float(low), float(high)],
            "assigned_value_kBq_per_ml": val_kBqml,
        }

    return act, organ_activity_info


# -----------------
# µ-Einheiten
# -----------------

def convert_mu_units(mu_xyz: np.ndarray, src_unit: str, tgt_unit: str) -> np.ndarray:
    """µ-Einheiten umrechnen (1/mm <-> 1/cm).

    Inputs:
      mu_xyz: Attenuationskoeffizienten-Volumen (x,y,z).
      src_unit/tgt_unit: "per_mm" oder "per_cm".
    Output:
      µ-Volumen in Ziel-Einheit (float32).
    """
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

    Inputs:
      act_data: Aktivitaet in MBq/voxel, Shape (x,y,z) der Sicht.
      atn_data: µ in 1/cm, gleiche Shape wie act_data.
      kernel_mat: PSF/Kernel (x,y,z) fuer Kollimatorfaltung.
      sigma: Scatter-Gauss (Pixel).
      z0_slices: Anzahl unverfilteter Slices vor PSF.
      step_len: Schrittweite entlang Projektionsrichtung (cm oder mm).

    WICHTIG:
      - globaler Counts-Faktor wird hier nicht eingebaut (MBq-aequivalent)
    """
    if gaussian_filter is None or convolve2d is None or fftconvolve is None:
        raise RuntimeError("Für das Gamma-Kamera-Modell werden scipy.ndimage.gaussian_filter "
                           "und scipy.signal.{convolve2d,fftconvolve} benötigt.")

    def _stats(arr: np.ndarray) -> str:
        return (f"min={float(arr.min()):.4e}, max={float(arr.max()):.4e}, "
                f"mean={float(arr.mean()):.4e}, sum={float(arr.sum()):.4e}")

    print("[DBG] vol_in stats:", _stats(act_data))

    # (1) Scatter
    if comp_scatter:
        act_sc = np.empty_like(act_data, dtype=np.float32)
        for z in range(act_data.shape[2]):
            act_sc[:, :, z] = gaussian_filter(act_data[:, :, z],
                                              sigma=sigma, mode="nearest")
        print("[DBG] after scatter (3D) stats:", _stats(act_sc))
    else:
        act_sc = act_data.astype(np.float32, copy=False)

    # (2) Attenuation: mu * step_len ist dimensionslos -> exponentielle Daempfung
    if atn_on:
        # atn_data: µ (1/cm), step_len: z-Schritt in gleicher Einheit (cm)
        mu_cum = np.cumsum(atn_data * step_len, axis=2)
        vol_atn = act_sc * np.exp(-mu_cum)
        mid_z = vol_atn.shape[2] // 2
        print("[DBG] after attenuation (mid slice) stats:", _stats(vol_atn[:, :, mid_z]))
        print("[DBG] after attenuation (3D) stats:", _stats(vol_atn))
    else:
        vol_atn = act_sc

    # (3) Kollimator-Faltung (PSF), Normierung auf Summe=1 erhaelt Gesamtenergie
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
                k_sum = float(K.sum())
                if k_sum > 0:
                    K = K / k_sum
                if z in (z0_slices, Z // 2):
                    print("[DBG] kernel normalized sum (z=%d):" % z, float(K.sum()))
                if z in (z0_slices, Z // 2, Z - 1):
                    print("[DBG] kernel pre stats (z=%d): %s" % (z, _stats(vol_atn[:, :, z])))
                    print("[DBG] kernel sum/min/max (z=%d): %s" %
                          (z, f"{float(K.sum()):.4e}, {float(K.min()):.4e}, {float(K.max()):.4e}"))
                vol_coll[:, :, z] = conv2(vol_atn[:, :, z], K,
                                          mode="same").astype(np.float32)
                if z in (z0_slices, Z // 2, Z - 1):
                    print("[DBG] kernel post stats (z=%d): %s" % (z, _stats(vol_coll[:, :, z])))
    else:
        vol_coll = vol_atn

    # (4) z-Summation
    proj2d = np.sum(vol_coll, axis=2)
    print("[DBG] proj2d stats:", _stats(proj2d))
    return proj2d


def gamma_camera_core(act_data: np.ndarray, atn_data: np.ndarray,
                      kernel_mat: np.ndarray,
                      sigma: float,
                      z0_slices: int,
                      step_len: float = 1.0,
                      comp_scatter: bool = True,
                      atn_on: bool = True,
                      coll_on: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Gamma-Kamera-Modell für AP/PA (wie in stratos.py).

    Inputs:
      act_data: Aktivitaet in MBq/voxel, (x,y,z).
      atn_data: µ in 1/cm (x,y,z).
      kernel_mat: LEAP-PSF fuer Kollimator, (x,y,z).
      sigma: Scatter-Gauss (Pixel).
      z0_slices: Slice-Index, ab dem PSF greift.
      step_len: physikalische Schrittweite entlang z.

    Output:
      proj_AP/proj_PA: 2D-Projektionen (MBq-aequivalent).

    Physik/Modell:
      Scatter (Gauss), Attenuation entlang z, Kollimator-PSF, Summe ueber z.
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
# Projektionen normieren (rein skaliert)
# -----------------

def normalize_projections(ap_raw: np.ndarray,
                          pa_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normiert AP/PA mit einem gemeinsamen p99.9-Skalierungsfaktor.

    Skala s = quantile_0.999(concat(AP, PA)) auf Arrays, die gespeichert werden sollen
    (post-noise, post-clipping falls angewandt).
    Guard: falls s <= 0 -> max(concat) -> 1.0.

    Inputs:
      ap_raw/pa_raw: Projektionen in Counts oder MBq-aequivalent (2D).
    Output:
      ap_n/pa_n: robust normierte Projektionen (typisch ~[0..1]).
      scale: gemeinsamer p99.9-Skalierungsfaktor.

    Hinweis:
      Joint-p99.9 erzwingt keine identischen p99.9 pro Sicht.
    """
    ap_raw = np.clip(ap_raw, 0.0, None)
    pa_raw = np.clip(pa_raw, 0.0, None)
    stacked = np.concatenate([ap_raw.ravel(), pa_raw.ravel()])
    scale = float(np.quantile(stacked, 0.999)) if stacked.size > 0 else 1.0
    if scale <= 0:
        scale = float(stacked.max()) if stacked.size > 0 else 1.0
    if scale <= 0:
        scale = 1.0

    ap_n = ap_raw / scale
    pa_n = pa_raw / scale

    # negative Werte auf 0 (numerische Artefakte)
    ap_n = np.clip(ap_n, 0.0, None)
    pa_n = np.clip(pa_n, 0.0, None)

    return ap_n.astype(np.float32), pa_n.astype(np.float32), float(scale)


# -----------------
# CLI
# -----------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Converter: BIN -> spect_att.npy, ct_att.npy, act.npy (kBq/mL), ap.npy, pa.npy (normalisiert)"
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
    p.add_argument("--sensitivity_cps_per_mbq", type=float, default=6.0,
                   help="System-Sensitivitaet (cps pro MBq) fuer absolute Counts")
    p.add_argument("--acq_time_s", type=float, default=600.0,
                   help="Akquisitionszeit in Sekunden fuer absolute Counts")

    # Manifest-Update (optional)
    p.add_argument("--manifest", type=Path, default=None,
                   help="Optional: Pfad zu manifest.csv, um proj_scale_joint_p99 zu speichern.")
    p.add_argument("--patient-id", type=str, default=None,
                   help="Patient-ID fuer das manifest (Default: Ordnername).")
    p.add_argument("--manifest-id-column", type=str, default="patient_id",
                   help="Spaltenname fuer die Patient-ID im manifest (default: patient_id).")
    p.add_argument("--apply_global_rot90", action="store_true",
                   help="Wenn gesetzt, werden alle Volumina (und AP/PA) global 90° CCW rotiert.")

    return p.parse_args()


# -----------------
# Main
# -----------------

def main():
    # I/O and path setup
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

    # Load volumes
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

    np.save(out_dir / "mask.npy", mask_xyz.astype(np.float32))

    # Build activity (kBq/mL) -> A_xyz_MBq (MBq/voxel)
    # Seed wählen: fix oder aus Phantom-Namen abgeleitet
    if args.activity_seed < 0:
        h = hashlib.sha256(base.name.encode("utf-8")).hexdigest()
        auto_seed = int(h[:8], 16)  # 32-bit Seed
        print(f"[INFO] Auto-Seed aus Phantom-Namen: {auto_seed}")
        rng_seed = auto_seed
    else:
        rng_seed = args.activity_seed

    # Aktivität aus Maske (kBq/mL)
    act_xyz, organ_info = build_activity_from_mask(
        mask_xyz=mask_xyz,
        organ_ids_txt=organ_ids_path,
        rng_seed=rng_seed,
    )

    print(f"[ACT] act_xyz: shape={act_xyz.shape}, "
          f"min={act_xyz.min():.1f} kBq/mL, max={act_xyz.max():.1f} kBq/mL, "
          f"mean={act_xyz.mean():.1f} kBq/mL")

    # Prepare attenuation (mu) and projector params
    # SPECT/CT-µ in Ziel-Einheit bringen (für Speicherung und Projektionen)
    spect_mu_xyz = convert_mu_units(spect_xyz, args.mu_unit, args.mu_target_unit)
    ct_mu_xyz = convert_mu_units(ct_xyz, args.mu_unit, args.mu_target_unit)

    # Schrittweite entlang Projektionsrichtung (z-Achse)
    if args.mu_target_unit == "per_cm":
        step_len = args.sd_mm / 10.0  # mm -> cm
    else:
        step_len = args.sd_mm        # in mm

    print(f"[INFO] Gamma-Projektor: mu_unit_in={args.mu_unit} "
          f"-> mu_unit_out={args.mu_target_unit}, step_len={step_len:.4f}")

    # Voxelvolumen: sd_mm -> cm -> mL (1 cm^3 = 1 mL)
    voxel_cm = args.sd_mm / 10.0
    V_voxel_ml = voxel_cm ** 3
    # kBq/mL -> MBq/voxel (Volumenkonversion + kBq->MBq)
    A_xyz_MBq = act_xyz * 1e-3 * V_voxel_ml
    print(f"Voxel: sd_mm={float(args.sd_mm):.4f}, V_voxel_ml={float(V_voxel_ml):.6e}")
    print("act_xyz kBq/ml: min/max/mean="
          f"{float(act_xyz.min()):.4f}/"
          f"{float(act_xyz.max()):.4f}/"
          f"{float(act_xyz.mean()):.4f}, "
          f"sum={float(act_xyz.sum()):.4e} (kBq/ml * voxels nur informativ)")
    sum_MBq = float(A_xyz_MBq.sum())
    print("A_xyz_MBq: min/max/mean="
          f"{float(A_xyz_MBq.min()):.4f}/"
          f"{float(A_xyz_MBq.max()):.4f}/"
          f"{float(A_xyz_MBq.mean()):.4f}, "
          f"sum_MBq={sum_MBq:.4e}, sum_Bq={sum_MBq * 1e6:.4e}")

    # Kernel laden
    if sio is None:
        raise RuntimeError("scipy.io (sio) wird für das Laden des LEAP-Kernels benötigt.")
    kernel_md = sio.loadmat(kernel_mat_path)
    if args.kernel_var not in kernel_md:
        raise KeyError(f"Variable '{args.kernel_var}' nicht in {kernel_mat_path} gefunden.")
    kernel_mat = kernel_md[args.kernel_var].astype(np.float32)

    # Forward projection (AP/PA)
    # Gamma-Kamera-Projektionen simulieren (AP/PA), MBq-aequivalent
    ap_raw, pa_raw = gamma_camera_core(
        act_data=A_xyz_MBq.astype(np.float32),
        atn_data=spect_mu_xyz.astype(np.float32),
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
    print("ap_raw: min/max/mean="
          f"{float(ap_raw.min()):.4f}/"
          f"{float(ap_raw.max()):.4f}/"
          f"{float(ap_raw.mean()):.4f}, "
          f"sum={float(ap_raw.sum()):.4e}")
    print("pa_raw: min/max/mean="
          f"{float(pa_raw.min()):.4f}/"
          f"{float(pa_raw.max()):.4f}/"
          f"{float(pa_raw.mean()):.4f}, "
          f"sum={float(pa_raw.sum()):.4e}")
    print("ratio sum ap/pa =",
          float(ap_raw.sum()) / (float(pa_raw.sum()) + 1e-12))

    # Convert raw projection to MBq-equivalent and then to counts
    ap_mbq = ap_raw
    pa_mbq = pa_raw
    # MBq-aequivalent -> erwartete Counts ueber Sensitivitaet und Akquisitionszeit
    ap_lam = np.clip(ap_mbq, 0.0, None) * float(args.sensitivity_cps_per_mbq) * float(args.acq_time_s)
    pa_lam = np.clip(pa_mbq, 0.0, None) * float(args.sensitivity_cps_per_mbq) * float(args.acq_time_s)
    print("ap_lam: min/max/mean="
          f"{float(ap_lam.min()):.4f}/"
          f"{float(ap_lam.max()):.4f}/"
          f"{float(ap_lam.mean()):.4f}, "
          f"sum={float(ap_lam.sum()):.4e}  (expected total counts)")
    print("pa_lam: min/max/mean="
          f"{float(pa_lam.min()):.4f}/"
          f"{float(pa_lam.max()):.4f}/"
          f"{float(pa_lam.mean()):.4f}, "
          f"sum={float(pa_lam.sum()):.4e}")

    # Counts = erwartete Counts pro Pixel
    ap_counts = ap_lam.astype(np.float32)
    pa_counts = pa_lam.astype(np.float32)

    print("[COUNTS] AP counts: sum/min/max/mean = "
          f"{ap_counts.sum():.4e} / {ap_counts.min():.1f} / {ap_counts.max():.1f} / {ap_counts.mean():.1f}")
    print("[COUNTS] PA counts: sum/min/max/mean = "
          f"{pa_counts.sum():.4e} / {pa_counts.min():.1f} / {pa_counts.max():.1f} / {pa_counts.mean():.1f}")
    print("ap_counts: min/max/mean="
          f"{float(ap_counts.min()):.4f}/"
          f"{float(ap_counts.max()):.4f}/"
          f"{float(ap_counts.mean()):.4f}, "
          f"sum={float(ap_counts.sum()):.4e}")
    print("pa_counts: min/max/mean="
          f"{float(pa_counts.min()):.4f}/"
          f"{float(pa_counts.max()):.4f}/"
          f"{float(pa_counts.mean()):.4f}, "
          f"sum={float(pa_counts.sum()):.4e}")

    # Normalize projections for NN input
    # AP/PA robust normalisieren (rein skaliert, MBq-aequivalent/Counts)
    ap_norm, pa_norm, scale_auto = normalize_projections(
        ap_raw=ap_counts,
        pa_raw=pa_counts,
    )

    # QA-Check: p99.9 nach joint-Scaling muss nicht exakt gleich fuer AP/PA sein
    ap_p999 = float(np.quantile(ap_norm.ravel(), 0.999)) if ap_norm.size > 0 else float("nan")
    pa_p999 = float(np.quantile(pa_norm.ravel(), 0.999)) if pa_norm.size > 0 else float("nan")
    print("[RANGE] AP norm:", ap_norm.min(), ap_norm.max(),
          "PA norm:", pa_norm.min(), pa_norm.max())
    print(f"[INFO] verwendeter Normalisierungsfaktor (joint p99.9): {scale_auto:.4e}")
    print(f"[CHECK] p99.9(AP_norm)={ap_p999:.4f} | p99.9(PA_norm)={pa_p999:.4f}", flush=True)
    tol = 0.10
    if (abs(ap_p999 - 1.0) > tol) or (abs(pa_p999 - 1.0) > tol) or (abs(ap_p999 - pa_p999) > tol):
        print(
            "[INFO] p99.9(AP_norm) und p99.9(PA_norm) sind nicht konsistent "
            f"(tol={tol:.2f}). Unterschiede zwischen AP und PA sind bei joint p99.9 moeglich.",
            flush=True,
        )

    # Kein zusätzlicher globaler LR-Flip mehr: orient_patch in gamma_camera_core liefert bereits das korrekte
    # Detektor-Koordinatensystem. Ein weiterer Flip hätte die AP/PA-Projektionen gegenüber den Volumina gespiegelt.
    ap_out = ap_norm
    pa_out = pa_norm

    # Optional global 90° CCW rotation for network/world alignment.
    if args.apply_global_rot90:
        # Volumina sind in Shape (x, y, z) = (LR, AP/Depth, SI);
        # die coronal-Ebene ist (x, z), daher rotieren wir nur in axes=(0, 2) und lassen AP/Depth (y) unangetastet.
        rot_axes = (0, 2)
        spect_mu_xyz = np.rot90(spect_mu_xyz, k=1, axes=rot_axes)
        ct_mu_xyz = np.rot90(ct_mu_xyz, k=1, axes=rot_axes)
        act_xyz = np.rot90(act_xyz, k=1, axes=rot_axes)
        ap_out = np.rot90(ap_out, k=1)
        pa_out = np.rot90(pa_out, k=1)

    # Normalize 3D volumes (separate scales per modality)
    spect_norm, spect_scale_p999, spect_norm_p999 = normalize_volume_p999(spect_mu_xyz)
    print(f"[INFO] spect_scale_p99.9={spect_scale_p999:.6e}")
    ct_norm, ct_scale_p999, ct_norm_p999 = normalize_volume_p999(ct_mu_xyz)
    print(f"[INFO] ct_scale_p99.9={ct_scale_p999:.6e}")
    act_norm, act_scale_p999, act_norm_p999 = normalize_volume_p999(act_xyz)
    print(f"[INFO] act_scale_p99.9={act_scale_p999:.6e}")

    # Save outputs and metadata
    # Speichern als .npy im out-Ordner
    np.save(out_dir / "spect_att.npy", spect_mu_xyz.astype(np.float32))
    np.save(out_dir / "ct_att.npy",    ct_mu_xyz.astype(np.float32))
    np.save(out_dir / "act.npy",       act_xyz.astype(np.float32))
    np.save(out_dir / "spect_att_norm.npy", spect_norm)
    np.save(out_dir / "ct_att_norm.npy",    ct_norm)
    np.save(out_dir / "act_norm.npy",       act_norm)
    np.save(out_dir / "ap_counts.npy", ap_counts.astype(np.float32))
    np.save(out_dir / "pa_counts.npy", pa_counts.astype(np.float32))
    np.save(out_dir / "ap.npy",        ap_out.astype(np.float32))
    np.save(out_dir / "pa.npy",        pa_out.astype(np.float32))

    # Orientierungskontrolle als PNGs (keine zusätzlichen Flips)
    orientation_dir = out_dir / "orientation_check"
    orientation_dir.mkdir(parents=True, exist_ok=True)
    mid_y = spect_mu_xyz.shape[1] // 2
    save_png(spect_mu_xyz[:, mid_y, :], orientation_dir / "spect_att_coronal_mid.png")
    save_png(ct_mu_xyz[:, mid_y, :],    orientation_dir / "ct_att_coronal_mid.png")
    save_png(act_xyz[:, mid_y, :],   orientation_dir / "act_coronal_mid.png")
    save_png(ap_out, orientation_dir / "ap.png")
    save_png(pa_out, orientation_dir / "pa.png")

    # Meta-Info: Summen/Skalen fuer QA, Reproduzierbarkeit und Kalibrier-Tracking.
    # Raw-Outputs sind physikalisch interpretierbar; norm-Outputs fuer NN-Input.
    # Meta-Info
    meta = {
        "shape_xyz": list(map(int, spect_xyz.shape)),
        "spect_bin": str(spect_bin_path),
        "ct_bin":    str(ct_bin_path),
        "mask_bin":  str(mask_bin_path),
        "organ_ids": str(organ_ids_path),
        "activity_unit": "kBq/mL",
        "lu177_psma_like": True,
        "organ_activity_info": organ_info,
        "mu_unit_in": args.mu_unit,
        "mu_unit_out": args.mu_target_unit,
        "sd_mm": float(args.sd_mm),
        "step_len": float(step_len),
        "kernel_mat": str(kernel_mat_path),
        "kernel_var": args.kernel_var,
        "sensitivity_cps_per_mbq": float(args.sensitivity_cps_per_mbq),
        "acq_time_s": float(args.acq_time_s),
        "V_voxel_ml": float(V_voxel_ml),
        "sum_activity_MBq": float(A_xyz_MBq.sum()),
        "sum_activity_Bq": float(A_xyz_MBq.sum() * 1e6),
        "sum_ap_raw": float(ap_raw.sum()),
        "sum_pa_raw": float(pa_raw.sum()),
        "sum_ap_mbq": float(ap_mbq.sum()),
        "sum_pa_mbq": float(pa_mbq.sum()),
        "sum_ap_expected_counts": float(ap_lam.sum()),
        "sum_pa_expected_counts": float(pa_lam.sum()),
        "sum_ap_counts": float(ap_counts.sum()),
        "sum_pa_counts": float(pa_counts.sum()),
        "max_ap_counts": float(ap_counts.max()),
        "max_pa_counts": float(pa_counts.max()),
        "mean_ap_counts": float(ap_counts.mean()),
        "mean_pa_counts": float(pa_counts.mean()),
        "proj_scale_joint_p99": float(scale_auto),
        "spect_scale_p99_9": float(spect_scale_p999),
        "ct_scale_p99_9": float(ct_scale_p999),
        "act_scale_p99_9": float(act_scale_p999),
        "spect_norm_p99_9": float(spect_norm_p999),
        "ct_norm_p99_9": float(ct_norm_p999),
        "act_norm_p99_9": float(act_norm_p999),
        "projection_norm_stats": {
            "ap_raw_min": float(ap_raw.min()),
            "ap_raw_max": float(ap_raw.max()),
            "pa_raw_min": float(pa_raw.min()),
            "pa_raw_max": float(pa_raw.max()),
            "ap_norm_min": float(ap_norm.min()),
            "ap_norm_max": float(ap_norm.max()),
            "ap_norm_p99_9": float(ap_p999),
            "pa_norm_min": float(pa_norm.min()),
            "pa_norm_max": float(pa_norm.max()),
            "pa_norm_p99_9": float(pa_p999),
        },
    }
    with open(out_dir / "meta_simple.json", "w") as f:
        json.dump(meta, f, indent=2)

    if args.manifest is not None:
        manifest_id = args.patient_id or base.name
        if args.patient_id is None:
            print(
                f"[WARN] --patient-id nicht gesetzt, nutze Ordnername '{manifest_id}' fuer manifest-Update.",
                flush=True,
            )
        update_manifest_scale(args.manifest, manifest_id, scale_auto, id_column=args.manifest_id_column)
        print(
            f"[MANIFEST] proj_scale_joint_p99={scale_auto:.4e} geschrieben fuer {manifest_id} in {args.manifest}",
            flush=True,
        )
    else:
        print(
            "[WARN] Kein --manifest angegeben; proj_scale_joint_p99 wurde nur in meta_simple.json gespeichert.",
            flush=True,
        )

    print(f"[OUT] Gespeichert in {out_dir}: "
          f"spect_att.npy, ct_att.npy, act.npy, ap.npy, pa.npy, meta_simple.json")


if __name__ == "__main__":
    main()
