#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run TotalSegmentator on a clinical CT DICOM series and create semi-transparent
overlay previews (coronal, sagittal, axial).

Usage:
    python ct_segmentation.py data/example_01/ct/3D-ABDROUTINE-1.5-B31S \
  --out_dir data/example_01/results \
  --split_segments \
  --also_individuals

Outputs (in the same directory as the DICOM folder):
    - ct.nii.gz
    - seg/segmentator_output.nii.gz
    - preview_coronal.png
    - preview_sagittal.png
    - preview_axial.png
"""

import argparse
import logging
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt


# --------------------------- Utility & I/O helpers ---------------------------

def setup_logging():
    # richtet Logging-Format ein (Zeit, Level, Nachricht)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def assert_dir_exists(p: Path, what: str):
    # prüft, ob ein Ordner existiert
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"{what} existiert nicht oder ist kein Ordner: {p}")


def ensure_cmd_available(cmd: str, hint: str = ""):
    # stellt sicher, dass externes Kommando (z.B. 'TotalSegmentator' im Path liegt)
    if shutil.which(cmd) is None:
        extra = f" ({hint})" if hint else ""
        raise EnvironmentError(
            f"Benötigtes Kommando '{cmd}' wurde nicht gefunden{extra}.\n"
            f"Stelle sicher, dass es installiert ist und im PATH liegt."
        )


# --------------------------- Step 1: DICOM → NIfTI ---------------------------

def convert_dicom_to_nifti(dicom_folder: Path, out_path: Path) -> Path:
    """
    Convert a DICOM series (first valid SeriesUID) to NIfTI using SimpleITK.
    Saves to out_path (e.g., ct.nii.gz) and returns out_path.

    Parameters
    ----------
    dicom_folder : Path
        Folder containing clinical DICOM files.
    out_path : Path
        Output NIfTI path (e.g., /.../ct.nii.gz)
    """
    logging.info("Suche DICOM-Serien mit SimpleITK...")
    reader = sitk.ImageSeriesReader()

    series_ids = reader.GetGDCMSeriesIDs(str(dicom_folder))
    if not series_ids:
        raise RuntimeError(
            f"Es wurden keine DICOM-Serien in '{dicom_folder}' gefunden. "
            "Prüfe den Pfad bzw. die Leserechte."
        )

    # Heuristik: erste gefundene Serie verwenden (typischerweise CT)
    selected_series = series_ids[0]
    logging.info(f"Gewählte SeriesInstanceUID: {selected_series}")

    file_names = reader.GetGDCMSeriesFileNames(str(dicom_folder), selected_series)
    if not file_names:
        raise RuntimeError("Die gewählte DICOM-Serie enthält keine Dateien.")

    reader.SetFileNames(file_names)

    logging.info("Lese DICOM-Serie und konvertiere nach NIfTI (dies kann etwas dauern)...")
    image = reader.Execute()

    # Viele klinische CTs sind bereits in HU (über RescaleSlope/Intercept kodiert).
    # SimpleITK wendet dies üblicherweise automatisch an. Wir schreiben direkt heraus.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(out_path), useCompression=True)
    logging.info(f"NIfTI gespeichert: {out_path}")

    return out_path

from nibabel.processing import resample_from_to

def _is_gzip_file(path: Path) -> bool:
    """Prüft anhand der ersten Bytes, ob die Datei wirklich gzip-komprimiert ist."""
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except Exception:
        return False

def _compress_with_nib(src: Path, dst_gz: Path):
    """Liest ein NIfTI (komprimiert oder unkomprimiert) und speichert gzip-komprimiert."""
    img = nib.load(str(src))
    dst_gz.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(dst_gz))

def _find_seg_candidate_files(base_dirs: list[Path]) -> list[Path]:
    """
    Sucht in den angegebenen Verzeichnissen und Unterordner-Struktur nach möglichen Segmentierungsdateien (*.nii / *.nii.gz).
    """
    candidates: list[Path] = []
    patterns = ["*.nii", "*.nii.gz"]
    for root in base_dirs:
        if not root.exists():
            continue
        # nicht-rekursiv
        for pat in patterns:
            candidates.extend(sorted(root.glob(pat)))
        # leicht rekursiv (Tiefe 2)
        for sub in root.glob("*"):
            if sub.is_dir():
                for pat in patterns:
                    candidates.extend(sorted(sub.glob(pat)))
    return candidates

def _score_candidate(p: Path) -> int:
    """
    Scoring für gefundene NIfTI-Kandidaten: je höher, desto besser.
    Bevorzugt Dateien mit 'seg'/'segment' im Namen und im gewünschten seg/-Ordner.
    """
    name = p.name.lower()
    score = 0
    if "segment" in name or "seg" in name or "total" in name or "ts_" in name:
        score += 10
    if p.suffix == ".gz":
        score += 2
    # leicht bevorzugen, wenn im seg-Ordner
    parts = [x.lower() for x in p.parts]
    if "seg" in parts:
        score += 3
    return score

def _normalize_seg_output(ct_path: Path, seg_dir: Path) -> Path:
    """
    Sucht nach einer Segmentdatei (verschiedene Namen/Lagen),
    repariert 'fake .nii.gz' und vereinheitlicht nach seg/segmentator_output.nii.gz.
    """
    expected_gz = seg_dir / "segmentator_output.nii.gz"
    expected_nii = seg_dir / "segmentator_output.nii"

    # 1) Bereits vorhanden?
    if expected_gz.exists():
        if not _is_gzip_file(expected_gz):
            logging.warning("'%s' ist kein echtes gzip – wird neu geschrieben ...", expected_gz)
            _compress_with_nib(expected_gz, expected_gz)
        return expected_gz
    if expected_nii.exists():
        logging.info("Gefunden: %s → komprimiere nach .nii.gz", expected_nii)
        _compress_with_nib(expected_nii, expected_gz)
        return expected_gz

    # 2) Kandidaten suchen (seg_dir und dessen Parent – also out_dir)
    out_dir = seg_dir.parent
    candidates = _find_seg_candidate_files([seg_dir, out_dir])

    if not candidates:
        raise FileNotFoundError("Keine Segmentdateien (*.nii / *.nii.gz) gefunden.")

    # 3) Best candidate wählen
    cand = max(candidates, key=_score_candidate)
    logging.info("Gefundene Segmentdatei: %s", cand)

    # 4) 'fake gz' reparieren oder umkopieren
    if cand.suffix == ".gz":
        if not _is_gzip_file(cand):
            logging.warning("'%s' ist kein echtes gzip – wird neu geschrieben ...", cand)
            _compress_with_nib(cand, expected_gz)
        else:
            expected_gz.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(cand), str(expected_gz))
        return expected_gz
    else:
        _compress_with_nib(cand, expected_gz)
        return expected_gz

def _load_seg_aligned_to_ct(seg_path: Path, ct_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Lädt die Segmentierung und resampelt sie bei Bedarf (nearest) auf das Grid des CT.
    """
    seg_img = nib.load(str(seg_path))
    if seg_img.shape == ct_img.shape and np.allclose(seg_img.affine, ct_img.affine, atol=1e-4):
        return seg_img
    logging.warning("Segmentation-Grid ≠ CT-Grid → resample (nearest)")
    return resample_from_to(seg_img, ct_img, order=0)


def _collect_individual_segments(seg_dir: Path):
    """
    Sucht einzelne Segment-NIfTIs (z. B. 'liver.nii.gz') in typischen TS-Ausgabeorten
    und legt sie unter seg/individual/ ab. Überspringt die kombinierte Datei.
    """
    targets = seg_dir / "individual"
    targets.mkdir(parents=True, exist_ok=True)

    # Typische Orte/Pattern je nach TS-Version
    search_roots = [
        seg_dir,
        seg_dir / "segmentations",
        seg_dir.parent,                 # manche Versionen schreiben fälschlich in out_dir
        seg_dir.parent / "segmentations",
    ]
    patterns = ["*.nii", "*.nii.gz"]

    moved = 0
    for root in search_roots:
        if not root.exists():
            continue
        for pat in patterns:
            for f in root.glob(pat):
                name = f.name.lower()
                # kombinierte Datei und ct nicht einsammeln
                if "segmentator_output" in name or f.name == "ct.nii.gz":
                    continue
                # nur echte Einzelsegmente (Heuristik)
                if any(k in name for k in ["seg", "segment", "liver", "kidney", "aorta", "spleen", "heart"]) or root.name == "segmentations":
                    dest = targets / f.name
                    if f.resolve() == dest.resolve():
                        continue
                    try:
                        shutil.move(str(f), str(dest))
                        moved += 1
                    except Exception:
                        # wenn move fehlschlägt (z. B. Cross-device), kopieren
                        shutil.copy2(str(f), str(dest))
                        moved += 1
    if moved:
        logging.info("Einzel-Segmente gesammelt: %d Dateien unter %s", moved, targets)
    else:
        logging.info("Keine Einzel-Segmente zum Einsammeln gefunden.")


# ---------------------- Step 2: Run TotalSegmentator ------------------------

def run_totalseg(ct_path: Path, seg_dir: Path, also_individuals: bool = False) -> Path:
    """
    Führt TotalSegmentator aus:
      1. Multilabel (--ml) → seg/segmentator_output.nii.gz
      2. Optional: Einzel-Segmente → seg/individual/*.nii.gz
    """
    seg_dir.mkdir(parents=True, exist_ok=True)
    ensure_cmd_available("TotalSegmentator")

    # -------------------------------------------------------------
    # 1️⃣ Multilabel-Segmentierung (eine Datei, für Overlays)
    # -------------------------------------------------------------
    combined_path = seg_dir / "segmentator_output.nii.gz"
    if not combined_path.exists():
        logging.info("Starte Multilabel-Segmentierung (TotalSegmentator --ml)...")
        cmd = [
            "TotalSegmentator",
            "-i", str(ct_path),
            "-o", str(seg_dir),
            "--ml"
        ]
        logging.info("Kommando: %s", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logging.error("TotalSegmentator stderr:\n%s", res.stderr)
            res.check_returncode()
        combined_path = _normalize_seg_output(ct_path, seg_dir)
    else:
        logging.info("Multilabel-Segmentierung bereits vorhanden, überspringe TS-Lauf.")

    # -------------------------------------------------------------
    # 2️⃣ Einzel-Segmente (optional, ohne --ml)
    # -------------------------------------------------------------
    if also_individuals:
        indiv_dir = seg_dir / "individual"
        indiv_dir.mkdir(parents=True, exist_ok=True)

        if not any(indiv_dir.glob("*.nii*")):
            logging.info("Starte Einzel-Segmentierung (ohne --ml)...")
            cmd = [
                "TotalSegmentator",
                "-i", str(ct_path),
                "-o", str(indiv_dir),
            ]
            logging.info("Kommando: %s", " ".join(cmd))
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                logging.error("TotalSegmentator stderr:\n%s", res.stderr)
                res.check_returncode()
            else:
                logging.info("Einzel-Segmente gespeichert unter: %s", indiv_dir)
        else:
            logging.info("Einzel-Segmente bereits vorhanden, überspringe TS-Lauf.")

    # -------------------------------------------------------------
    # 3️⃣ Rückgabe (kombinierte Datei für Overlays)
    # -------------------------------------------------------------
    logging.info("Segmentation erfolgreich: %s", combined_path)
    return combined_path


def _run_totalseg_once(ct_path: Path, out_dir: Path, use_ml: bool, fast: bool = True):
    ensure_cmd_available("TotalSegmentator")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["TotalSegmentator", "-i", str(ct_path), "-o", str(out_dir), "--ml"] if use_ml else \
          ["TotalSegmentator", "-i", str(ct_path), "-o", str(out_dir)]
    if fast:
        cmd.append("--fast")
    logging.info("Starte TotalSegmentator: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
        



# ---------------------- Step 3: Overlay-Previews (PNG) ----------------------

def _window_image_hu(img: np.ndarray, wmin: float, wmax: float) -> np.ndarray:
    """Apply HU windowing and normalize to [0,1]."""
    img = np.clip(img, wmin, wmax)
    if wmax == wmin:
        return np.zeros_like(img, dtype=np.float32)
    img = (img - wmin) / (wmax - wmin)
    return img.astype(np.float32)


def _label_to_rgb(mask_slice: np.ndarray) -> np.ndarray:
    """
    Map integer labels to RGB using a qualitative colormap.
    Background (0) -> transparent handled later (alpha).
    Returns float RGB in [0,1], shape (H, W, 3).
    """
    # Use a large qualitative map for diverse labels
    cmap = plt.get_cmap("tab20", 20)
    labels = mask_slice.astype(np.int32)
    rgb = np.zeros((*labels.shape, 3), dtype=np.float32)
    nonzero = labels > 0
    if np.any(nonzero):
        # Wrap labels into available colors deterministically
        idx = (labels[nonzero] - 1) % cmap.N
        rgb_vals = cmap(idx)[:, :3]  # drop alpha from colormap
        rgb[nonzero] = rgb_vals.astype(np.float32)
    return rgb


def _extract_middle_slices(vol: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return middle sagittal (x-mid), coronal (y-mid), axial (z-mid) slices as 2D arrays
    without rotation/flips. Shapes:
      - sagittal: (Y, Z)
      - coronal:  (X, Z)
      - axial:    (X, Y)
    """
    x, y, z = vol.shape
    sag = vol[x // 2, :, :]   # (Y, Z)
    cor = vol[:, y // 2, :]   # (X, Z)
    axi = vol[:, :, z // 2]   # (X, Y)
    return sag, cor, axi


def _save_overlay(
    ct_2d: np.ndarray,
    seg_2d: np.ndarray,
    out_png: Path,
    alpha: float,
    wmin: float,
    wmax: float,
    add_colorbar: bool = True,
    voxel_mm: Tuple[float, float] = (1.0, 1.0),  # (dx, dy)
    rotated: bool = False,
    dpi: int = 200,
):
    """
    Plot 2D overlay with correct physical aspect ratio (no stretching).
    voxel_mm: (pixel spacing in mm) along x and y axes of ct_2d.
    """
    ct_w = _window_image_hu(ct_2d, wmin, wmax)
    rgb = _label_to_rgb(seg_2d)

    ny, nx = ct_w.shape
    dx, dy = voxel_mm

    # falls rotiert (90°/270°), Achsen tauschen
    if rotated:
        dx, dy = dy, dx

    width_mm = nx * dx
    height_mm = ny * dy
    aspect = width_mm / height_mm

    fig_h = 6
    fig_w = fig_h * aspect
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    # extent = physikalische mm-Range
    extent = [0, width_mm, 0, height_mm]

    im = ax.imshow(
        ct_w,
        cmap="gray",
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    overlay = np.zeros((ny, nx, 4), dtype=np.float32)
    overlay[..., :3] = rgb
    overlay[..., 3] = (seg_2d > 0).astype(np.float32) * float(alpha)
    ax.imshow(overlay, origin="lower", extent=extent, aspect="equal")

    ax.set_axis_off()

    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("HU (Fenster)")
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels([f"{int(wmin)}", f"{int((wmin+wmax)/2)}", f"{int(wmax)}"])

    fig.tight_layout(pad=0)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    logging.info(f"Preview gespeichert: {out_png.name}")



from nibabel.affines import voxel_sizes
def make_preview_slices(ct_path: Path, seg_path: Path, out_dir: Path,
                        alpha: float, wmin: float, wmax: float):
    logging.info("Lade CT-Volumen (nibabel)...")
    ct_img = nib.load(str(ct_path))
    ct = np.asanyarray(ct_img.dataobj).astype(np.float32)

    logging.info("Lade/angleiche Segmentations-Volumen (nibabel)...")
    seg_img = _load_seg_aligned_to_ct(seg_path, ct_img)
    seg = np.asanyarray(seg_img.dataobj).astype(np.int32)

    sag_ct, cor_ct, axi_ct = _extract_middle_slices(ct)
    sag_seg, cor_seg, axi_seg = _extract_middle_slices(seg)

    # Voxelgrößen
    from nibabel.affines import voxel_sizes
    sx, sy, sz = voxel_sizes(ct_img.affine)
    sx, sy, sz = abs(sx), abs(sy), abs(sz)

    # Orientierung: coronal/sagittal um 270° CW (k=3)
    cor_ct = np.rot90(cor_ct, k=3)
    cor_seg = np.rot90(cor_seg, k=3)
    sag_ct = np.rot90(sag_ct, k=3)
    sag_seg = np.rot90(sag_seg, k=3)
    cor_ct = np.fliplr(cor_ct)
    cor_seg = np.fliplr(cor_seg)

    # ---- Axial ----
    _save_overlay(
        axi_ct, axi_seg, out_dir / "preview_axial.png",
        alpha, wmin, wmax,
        voxel_mm=(sy, sx),
        rotated=False,
    )

    # ---- Coronal ----
    _save_overlay(
        cor_ct, cor_seg, out_dir / "preview_coronal.png",
        alpha, wmin, wmax,
        voxel_mm=(sz, sx),
        rotated=True,  # nach rot90 -> mm-Achsen getauscht
    )

    # ---- Sagittal ----
    _save_overlay(
        sag_ct, sag_seg, out_dir / "preview_sagittal.png",
        alpha, wmin, wmax,
        voxel_mm=(sz, sy),
        rotated=True,
    )


# ------------------------------ CLI & Orchestration -------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline: DICOM CT → NIfTI → TotalSegmentator → 3 Overlay-Previews",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "dicom_folder",
        type=str,
        help="Pfad zum Ordner mit klinischen DICOM-Dateien (eine Serie).",
    )
    p.add_argument(
        "--also_individuals",
        action="store_true",
        help="Zusätzlich zu segmentator_output.nii.gz auch alle Einzel-Segmente unter seg/individual/ speichern."
    )
    p.add_argument(
        "--split_segments",
        action="store_true",
        help="Zusätzlich alle Einzel-Segmente speichern (seg/individual/*.nii.gz).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optionaler Ausgabeordner (Standard: im selben Ordner wie DICOM-Serie).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Alphawert der halbtransparenten Segmentmaske (0..1).",
    )
    p.add_argument(
        "--wmin",
        type=float,
        default=-1000.0,
        help="Fenster-Untergrenze in HU (z. B. -1000).",
    )
    p.add_argument(
        "--wmax",
        type=float,
        default=1000.0,
        help="Fenster-Obergrenze in HU (z. B. 1000).",
    )
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    dicom_dir = Path(args.dicom_folder).expanduser().resolve()
    assert_dir_exists(dicom_dir, "Der DICOM-Ordner")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else dicom_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else dicom_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_ct = out_dir / "ct.nii.gz"
    seg_dir = out_dir / "seg"

    # Plausibilitätsprüfungen
    if not (0.0 <= args.alpha <= 1.0):
        logging.error("Ungültiger Alphawert: %s (muss zwischen 0 und 1 liegen).", args.alpha)
        sys.exit(2)
    if args.wmax <= args.wmin:
        logging.error("wmax (%.2f) muss größer sein als wmin (%.2f).", args.wmax, args.wmin)
        sys.exit(2)

    try:
        # 1) DICOM -> NIfTI
        ct_path = convert_dicom_to_nifti(dicom_dir, out_ct)

        # 2) TotalSegmentator
        seg_path = run_totalseg(ct_path, seg_dir, also_individuals=args.also_individuals)

        # 3) Overlays
        make_preview_slices(ct_path, seg_path, out_dir, alpha=args.alpha,
                            wmin=args.wmin, wmax=args.wmax)

        logging.info("Fertig. Ergebnisse liegen in: %s", dicom_dir)

    except Exception as e:
        logging.error("Fehler: %s", e)
        sys.exit(1)

    logging.info("Fertig. Ergebnisse liegen in: %s", out_dir)

if __name__ == "__main__":
    main()