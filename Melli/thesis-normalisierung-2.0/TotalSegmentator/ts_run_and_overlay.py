#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TotalSegmentator (optional) → Coronal(AP)-Overlay (mm-Extents)
- Erwartet HU-CT in RAS (X,Y,Z).
- Segmentdatei wird fest unter ts_output.nii (oder .nii.gz) erwartet.
- Kein HU-Check, kein Ordnerhandling mehr.

Beispiel:
  python ts_run_and_overlay.py \
    --in-hu runs_360/ct_recon_rtk360_HU.nii \
    --seg runs_360/ts_output.nii \
    --overlay runs_360/ts_preview/coronal_overlay.png \
    --run-ts --ts-args "--fast"
"""

import os, sys, argparse, subprocess
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- Hilfsfunktionen ----------------

def ensure_same_grid(seg_path: str, ref_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Falls Segment-Shape oder Affine ≠ HU-Bild:
    → resample (nearest neighbor) auf das Grid des HU-Bildes.
    """
    seg_img = nib.load(seg_path)
    if seg_img.shape == ref_img.shape and np.allclose(seg_img.affine, ref_img.affine, atol=1e-4):
        return seg_img  # passt exakt
    print("[WARN] Segmentation shape/affine ≠ HU → resample (nearest)")
    return resample_from_to(seg_img, ref_img, order=0)


def save_coronal_overlay_ras(hu_img: nib.Nifti1Image,
                             seg_img: nib.Nifti1Image,
                             out_png: str,
                             base_h_in: float = 8.0,
                             coronal_orient: str = "AP",
                             cmap_name: str = "turbo",
                             alpha: float = 0.35) -> None:
    """
    Zeichnet Coronal(AP)-Overlay:
      - mittlere Coronal-Scheibe (X fix),
      - HU in Graustufen (2–98 %),
      - Labels halbtransparent farbig.
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    vol = np.asarray(hu_img.dataobj, dtype=np.float32)  # HU-Volumen
    lab = np.asarray(seg_img.dataobj, dtype=np.int32)   # Labelvolumen
    aff = hu_img.affine
    dx, dy, dz = float(aff[0, 0]), float(aff[1, 1]), float(aff[2, 2])
    X, Y, Z = vol.shape

    # mittlere Coronal-Scheibe (XZ)
    sl_hu  = vol[:, Y//2, :]
    sl_seg = lab[:, Y//2, :]

    # anatomisch korrekt darstellen (Z nach oben, X horizontal)
    img2 = np.flipud(sl_hu)
    seg2 = np.flipud(sl_seg)

    # ggf. AP-orientiert (horizontal spiegeln)
    if coronal_orient.upper() == "AP":
        img2 = np.fliplr(img2)
        seg2 = np.fliplr(seg2)

    # HU-Fensterung (2–98 %)
    finite = np.isfinite(img2)
    p2, p98 = np.percentile(img2[finite], [2, 98]) if finite.any() else (0, 1)
    shown = np.clip((img2 - p2) / (p98 - p2 + 1e-6), 0, 1)

    # diskrete Farbtabelle für Labels
    vmax = int(seg2.max()) if seg2.size > 0 else 0
    cmap = plt.cm.get_cmap(cmap_name, max(vmax, 1) + 1)
    seg_norm = seg2 / (max(vmax, 1) + 1.0)
    seg_rgba = cmap(seg_norm)
    seg_rgba[..., 3] = (seg2 > 0) * float(alpha)

    # mm-Extents und Bildgröße
    width_mm, height_mm = dz * Z, dy * Y
    extent = [0, width_mm, 0, height_mm]
    aspect = height_mm / max(width_mm, 1e-6)
    h_in = base_h_in
    w_in = h_in / max(aspect, 1e-6)

    # Plot
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=200)
    ax.imshow(shown, cmap="gray", origin="lower", extent=extent, aspect="equal")
    ax.imshow(seg_rgba, origin="lower", extent=extent, aspect="equal")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Z [mm]")
    ax.set_title(f"Coronal ({coronal_orient}) Overlay")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[OK] Overlay gespeichert: {out_png}  | extent={width_mm:.1f}×{height_mm:.1f} mm")


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-hu", required=True, help="Pfad zur HU-CT-Datei (RAS, X,Y,Z)")
    ap.add_argument("--seg", required=True, help="Pfad zur Segmentdatei (z. B. runs_360/ts_output.nii)")
    ap.add_argument("--overlay", default="ts_preview/coronal_overlay.png", help="Zielbild (PNG)")
    ap.add_argument("--base-h-in", type=float, default=8.0, help="Bildhöhe in Inches")
    ap.add_argument("--run-ts", action="store_true", help="TotalSegmentator ausführen (überschreibt --seg)")
    ap.add_argument("--ts-args", type=str, default="", help="Zusatzargumente für TS, z. B. \"--fast\"")
    ap.add_argument("--cmap", default="turbo", help="Colormap für Labels")
    ap.add_argument("--alpha", type=float, default=0.35, help="Label-Transparenz")
    ap.add_argument("--coronal_orient", choices=["AP", "PA"], default="AP")
    args = ap.parse_args()

    seg_path = args.seg

    # 1) ggf. TotalSegmentator ausführen
    if args.run_ts:
        seg_dir = os.path.dirname(seg_path) or "."
        os.makedirs(seg_dir, exist_ok=True)
        cmd = ["TotalSegmentator", "-i", args.in_hu, "-o", seg_path, "--ml"]
        if args.ts_args:
            cmd += args.ts_args.split()
        print("[INFO] Running TotalSegmentator:\n  " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"[OK] Segmentation finished → {seg_path}")

    # 2) Segment + HU laden, ggf. resamplen
    hu_img = nib.load(args.in_hu)
    seg_img = ensure_same_grid(seg_path, hu_img)

    # 3) Overlay speichern
    save_coronal_overlay_ras(
        hu_img, seg_img, args.overlay,
        base_h_in=args.base_h_in,
        coronal_orient=args.coronal_orient,
        cmap_name=args.cmap,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()