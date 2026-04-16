#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
show_real_scinti.py

Liest planare Szintigraphie-DICOMs (Multi-Frame oder Einzelbilder) aus einem Ordner,
speichert die Rohdaten als .npy und erzeugt Previews:

- Alle Bilder nach ViewLabel (AP / PA / UNKNOWN)
- Bei Multi-Frame-UNKNOWN mit genau 4 Frames:
    * pro Frame ein PNG (frame_0..3)
    * AP_guess = Frame 0 + Frame 2
    * PA_guess = Frame 1 + Frame 3
      jeweils als .npy + PNG

Beispielaufruf:

python show_real_scinti.py \
  --dicom_dir /home/mnguest12/projects/thesis/SPECTCT_Pipeline/data/example_01/planar/LU-177-PSMA-GANZKOERPER19-H \
  --out_dir   /home/mnguest12/projects/thesis/SPECTCT_Pipeline/data/example_01/results/planar_images
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    import pydicom
except ImportError as e:
    raise ImportError(
        "Das Modul 'pydicom' wird benötigt. Bitte im venv installieren:\n"
        "  pip install pydicom"
    ) from e


def find_dicom_files(dicom_dir: Path) -> list[Path]:
    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"DICOM-Ordner existiert nicht: {dicom_dir}")
    files: list[Path] = []
    for root, _, fnames in os.walk(dicom_dir):
        for fname in fnames:
            p = Path(root) / fname
            if p.suffix.lower() in (".dcm", ""):
                files.append(p)
    if not files:
        raise RuntimeError(f"Keine DICOM-Dateien in {dicom_dir} gefunden.")
    return sorted(files)


def get_view_label(ds: "pydicom.dataset.Dataset") -> str:
    """
    Versucht, eine View-Position aus dem DICOM zu lesen.
    Rückgabe: 'AP', 'PA' oder 'UNKNOWN'.
    """
    view = None
    if "ViewPosition" in ds:
        view = str(ds.ViewPosition).upper().strip()

    if view is None:
        return "UNKNOWN"

    if view in ("AP", "ANT", "A"):
        return "AP"
    if view in ("PA", "POST", "P"):
        return "PA"
    return "UNKNOWN"


def load_planar_images_grouped(dicom_dir: Path) -> Dict[str, List[np.ndarray]]:
    files = find_dicom_files(dicom_dir)
    print(f"[INFO] Gefundene DICOM-Dateien: {len(files)}")

    grouped: Dict[str, List[np.ndarray]] = {"AP": [], "PA": [], "UNKNOWN": []}

    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
        except Exception as e:
            print(f"[WARN] Konnte {f} nicht lesen: {e}")
            continue

        try:
            arr = ds.pixel_array
        except Exception as e:
            print(f"[WARN] Kein pixel_array in {f}: {e}")
            continue

        view_label = get_view_label(ds)
        print(f"[DEBUG] Datei: {f.name}, View: {view_label}, pixel_array.shape={arr.shape}")

        if arr.ndim == 3:
            # Multi-Frame: (nframes, H, W)
            for i in range(arr.shape[0]):
                frame = arr[i, :, :].astype(np.float32)
                grouped.setdefault(view_label, []).append(frame)
        elif arr.ndim == 2:
            grouped.setdefault(view_label, []).append(arr.astype(np.float32))
        else:
            print(f"[WARN] Unerwartige Dimension in {f}: arr.ndim={arr.ndim} -> ignoriere")

    for key in list(grouped.keys()):
        if len(grouped[key]) == 0:
            print(f"[INFO] Keine Bilder für View='{key}' gefunden.")
    return grouped


def stack_and_save_npy(grouped: Dict[str, List[np.ndarray]], out_dir: Path) -> Dict[str, np.ndarray]:
    out_arrays: Dict[str, np.ndarray] = {}
    for view, imgs in grouped.items():
        if not imgs:
            continue
        shapes = {img.shape for img in imgs}
        if len(shapes) > 1:
            print(f"[WARN] Unterschiedliche Shapes für View='{view}': {shapes}. "
                  f"Versuche trotzdem zu stapeln.")
        arr = np.stack(imgs, axis=0)  # (n, H, W)
        out_arrays[view] = arr
        out_path = out_dir / f"scinti_{view}.npy"
        np.save(out_path, arr.astype(np.float32))
        print(f"[INFO] Gespeichert: {out_path}  shape={arr.shape}")
    return out_arrays



def save_preview_png(view: str, arr: np.ndarray, out_dir: Path, name_suffix: str = "",
                     clip_percentile: float = 99.5, use_log: bool = False):
    """
    arr: (n, H, W) oder (H, W)
    erzeugt ein Summenbild und speichert PNG mit Colorbar.
    """
    # --- Summenbild erzeugen ---
    if arr.ndim == 3:
        img = arr.sum(axis=0)
    else:
        img = arr
    img = img.astype(np.float32)

    # --- Wertebereich bestimmen ---
    img_min = float(img.min())
    img_max = float(img.max())

    # oberes Perzentil als vmax (robust gegen Hotspots)
    vmax = float(np.percentile(img, clip_percentile))
    if vmax <= 0:
        vmax = img_max if img_max > 0 else 1.0

    # --- Figure & Achsen ---
    fig, ax = plt.subplots(figsize=(5, 7))

    # --- Darstellung (linear oder log) ---
    if use_log:
        vmin = max(img_min, vmax * 1e-4, 1e-3)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        im = ax.imshow(img, cmap="inferno", norm=norm, interpolation="nearest")
    else:
        im = ax.imshow(img, cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")

    ax.set_title(f"Planare Szinti – {view}{name_suffix}")
    ax.axis("off")

    # --- Colorbar rechts außen (gut platziert!) ---
    cbar = fig.colorbar(im, ax=ax, shrink=0.80, pad=0.03)
    cbar.set_label("Intensity (a.u.)", rotation=90)

    # --- Speichern ---
    out_path = out_dir / f"preview_{view}{name_suffix}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] Preview gespeichert: {out_path} (vmax={vmax:.1f}, img_max={img_max:.1f})")


def handle_unknown_fourframe(arr_unknown: np.ndarray, out_dir: Path):
    """
    Spezialfall:
    - arr_unknown: (4, H, W) aus einer Multi-Frame-Datei ohne ViewPosition.
    - Interpretation (FIX):
        * Frame 0 = AP
        * Frame 1 = PA
        * Frames 2 und 3 werden ignoriert.
    - Speichert:
        * scinti_AP.npy (Frame 0)
        * scinti_PA.npy (Frame 1)
        * preview_AP.png
        * preview_PA.png
    """
    n, H, W = arr_unknown.shape
    if n < 2:
        print(f"[INFO] UNKNOWN hat nur {n} Frames (<2) -> kann AP/PA nicht ableiten.")
        return

    print("[INFO] UNKNOWN: Interpretiere Frame 0 als AP, Frame 1 als PA. Frames 2/3 werden ignoriert.")

    ap = arr_unknown[0]  # (H, W)
    pa = arr_unknown[1]  # (H, W)

    # NPY speichern
    np.save(out_dir / "scinti_AP.npy", ap.astype(np.float32))
    np.save(out_dir / "scinti_PA.npy", pa.astype(np.float32))
    print(f"[INFO] scinti_AP.npy und scinti_PA.npy gespeichert.")

    # Previews (jeweils EIN 2D-Bild, mit Perzentil-Clipping)
    save_preview_png("AP", ap, out_dir, clip_percentile=99.5, use_log=False)
    save_preview_png("PA", pa, out_dir, clip_percentile=99.5, use_log=False)

    # Wenn du zusätzlich Log-Variante willst, einfach einkommentieren:
    # save_preview_png("AP_log", ap, out_dir, clip_percentile=99.5, use_log=True)
    # save_preview_png("PA_log", pa, out_dir, clip_percentile=99.5, use_log=True)


def main():
    parser = argparse.ArgumentParser(
        description="Liest planare Szinti-DICOMs, speichert NPYs und erzeugt AP/PA-Previews (mit Heuristik bei 4 Frames)."
    )
    parser.add_argument(
        "--dicom_dir",
        required=True,
        type=str,
        help="Pfad zum Ordner mit planaren Szinti-DICOMs.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="Ausgabeordner für .npy und Preview-PNGs.",
    )
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Lese DICOMs aus: {dicom_dir}")
    grouped = load_planar_images_grouped(dicom_dir)

    print(f"[INFO] Speichere NPY-Dateien nach: {out_dir}")
    arrays = stack_and_save_npy(grouped, out_dir)

    print("[INFO] Erzeuge Preview-PNGs ...")
    for view, arr in arrays.items():
        # Für UNKNOWN mit 4 Frames erzeugen wir keine Summen-Preview,
        # sondern behandeln das in handle_unknown_fourframe separat.
        if view == "UNKNOWN" and arr.ndim == 3 and arr.shape[0] == 4:
            continue
        save_preview_png(view, arr, out_dir)

    # Spezialfall: UNKNOWN mit genau 4 Frames → AP/PA aus Frame 0/1
    if "UNKNOWN" in arrays:
        arr_unknown = arrays["UNKNOWN"]
        if arr_unknown.ndim == 3 and arr_unknown.shape[0] == 4:
            handle_unknown_fourframe(arr_unknown, out_dir)

    print("[INFO] Fertig.")


if __name__ == "__main__":
    main()