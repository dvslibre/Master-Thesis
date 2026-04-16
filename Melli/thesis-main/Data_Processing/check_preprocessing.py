#!/usr/bin/env python3
"""
check_preprocessing.py — Sanity-Check für simples Preprocessing

Erwartete Struktur:
  <base>/
    out/
      spect_att.npy   (SPECT-µ-Volumen, z.B. bei 208 keV, Einheiten je nach XCAT-Export)
      ct_att.npy      (CT-µ-Volumen, z.B. bei 80 keV)
      act.npy         (synthetische Aktivitätskonzentration aus Maske, in kBq/mL)
      [mask.npy]      (optional, falls gespeichert)
      [ap.npy, pa.npy] (optional, falls Gamma-Kamera-Modell verwendet wurde)

Erzeugt:
  - check_coronal_slices.png   (Koronalschnitte von spect_att, ct_att, act, optional mask)
  - check_projections.png      (AP/PA-Projektionen, falls ap.npy/pa.npy existieren)

Aufruf:
python check_preprocessing.py \
  --base /home/mnguest12/projects/thesis/Data_Processing/phantom_01
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Phantom-Basisordner (der mit out/ drin, z.B. phantom_01)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    base = args.base
    out_dir = base / "out"

    if not out_dir.exists():
        raise FileNotFoundError(f"out-Ordner nicht gefunden: {out_dir}")

    spect_path = out_dir / "spect_att.npy"
    ct_path    = out_dir / "ct_att.npy"
    act_path   = out_dir / "act.npy"
    mask_path  = out_dir / "mask.npy"   # optional
    ap_path    = out_dir / "ap.npy"     # optional
    pa_path    = out_dir / "pa.npy"     # optional

    for pth in [spect_path, ct_path, act_path]:
        if not pth.exists():
            raise FileNotFoundError(f"Pflichtdatei fehlt: {pth}")

    spect = np.load(spect_path)  # (x,y,z)
    ct    = np.load(ct_path)     # (x,y,z)
    act   = np.load(act_path)    # (x,y,z)  – Aktivität in kBq/mL

    mask = np.load(mask_path) if mask_path.exists() else None

    print("[INFO] Shapes:")
    print("  spect_att:", spect.shape)
    print("  ct_att:   ", ct.shape)
    print("  act (kBq/mL):", act.shape)
    if mask is not None:
        print("  mask:     ", mask.shape)

    print(f"[INFO] Activity stats: min={act.min():.1f} kBq/mL, "
          f"max={act.max():.1f} kBq/mL, mean={act.mean():.1f} kBq/mL")

    # ---------------------------------------------------------
    # 1) Koronale Slices (µ_SPECT, µ_CT, Aktivität, optional Maske)
    #    Volumen: (x,y,z) -> wir interpretieren (D,H,W) = (x,y,z)
    #    Koronalebene: (D,W) = vol[:, mid_H, :]
    # ---------------------------------------------------------
    D, H, W = spect.shape
    mid_H = H // 2
    print(f"[INFO] Koronalschnitt bei y={mid_H}")

    def make_coronal(vol):
        """
        Erzeugt koronal orientierte AP-Darstellung:
        - slice bei y = mid_H
        - rotate + flips für sinnvolle Anatomie (Kopf oben, Füße unten, AP)
        """
        cor = vol[:, mid_H, :]          # (D,W)

        # 1) 90° CW rotieren
        cor_plot = np.rot90(cor, k=-1)

        # 2) Links/Rechts flippen → AP statt PA
        cor_plot = np.fliplr(cor_plot)

        # 3) Kopf/Fuß korrigieren → nicht auf dem Kopf
        cor_plot = np.flipud(cor_plot)

        return cor_plot

    spect_cor = make_coronal(spect)
    ct_cor    = make_coronal(ct)
    act_cor   = make_coronal(act)
    mask_cor  = make_coronal(mask) if mask is not None else None

    n_cols = 3 + (1 if mask_cor is not None else 0)
    fig, axs = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    ax_idx = 0
    im0 = axs[ax_idx].imshow(spect_cor, cmap="gray")
    axs[ax_idx].set_title("SPECT μ – coronal")
    axs[ax_idx].axis("off")
    c0 = plt.colorbar(im0, ax=axs[ax_idx])
    c0.set_label("μ (arb. units)")   # falls du weißt: z.B. „μ (1/cm)“
    ax_idx += 1

    im1 = axs[ax_idx].imshow(ct_cor, cmap="gray")
    axs[ax_idx].set_title("CT μ – coronal")
    axs[ax_idx].axis("off")
    c1 = plt.colorbar(im1, ax=axs[ax_idx])
    c1.set_label("μ (arb. units)")
    ax_idx += 1

    im2 = axs[ax_idx].imshow(act_cor, cmap="inferno")
    axs[ax_idx].set_title("Activity – coronal")
    axs[ax_idx].axis("off")
    c2 = plt.colorbar(im2, ax=axs[ax_idx])
    c2.set_label("Activity (kBq/mL)")   # <<< wichtige Änderung: Einheit klar
    ax_idx += 1

    if mask_cor is not None:
        im3 = axs[ax_idx].imshow(mask_cor, cmap="tab20")
        axs[ax_idx].set_title("Mask – coronal")
        axs[ax_idx].axis("off")
        c3 = plt.colorbar(im3, ax=axs[ax_idx])
        c3.set_label("Label ID")

    plt.tight_layout()
    out1 = base / "check_coronal_slices.png"
    plt.savefig(out1, dpi=200)
    plt.close()
    print(f"[OUT] Saved {out1}")

    # ---------------------------------------------------------
    # 2) AP/PA-Projektionen aus ap.npy/pa.npy (falls vorhanden)
    # ---------------------------------------------------------
    if ap_path.exists() and pa_path.exists():
        ap = np.load(ap_path)
        pa = np.load(pa_path)

        print("[INFO] Projections:")
        print("  ap:", ap.shape, "min/max:", ap.min(), ap.max())
        print("  pa:", pa.shape, "min/max:", pa.min(), pa.max())

        # Rotiert für sinnvolle anatomische Ansicht
        ap_plot = np.rot90(ap, k=1)
        pa_plot = np.rot90(pa, k=1)

        # Gemeinsame Farbskala
        vmin = min(ap_plot.min(), pa_plot.min())
        vmax = max(ap_plot.max(), pa_plot.max())

        # Figure + GridSpec: zwei Bilder dicht nebeneinander + schmale Colorbar
        fig = plt.figure(figsize=(10, 4))
        gs = fig.add_gridspec(
            1, 3,
            width_ratios=[1, 1, 0.05],  # zwei gleich breite Bilder + schmale Colorbar
            wspace=0.02                 # kleiner Abstand zwischen AP und PA
        )

        ax_ap = fig.add_subplot(gs[0, 0])
        ax_pa = fig.add_subplot(gs[0, 1])
        cax   = fig.add_subplot(gs[0, 2])

        im_ap = ax_ap.imshow(ap_plot, origin="lower",
                             aspect="equal", vmin=vmin, vmax=vmax,
                             cmap="plasma")
        ax_ap.set_title("AP Projection")
        ax_ap.axis("off")

        im_pa = ax_pa.imshow(pa_plot, origin="lower",
                             aspect="equal", vmin=vmin, vmax=vmax,
                             cmap="plasma")
        ax_pa.set_title("PA Projection")
        ax_pa.axis("off")

        cbar = fig.colorbar(im_pa, cax=cax)
        cbar.set_label("Intensity (a.u.)")  # hier sind es weiterhin „relative counts“ o. skaliert

        plt.tight_layout()
        out2 = base / "check_projections.png"
        plt.savefig(out2, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OUT] Saved {out2}")
    else:
        print("[INFO] ap.npy/pa.npy nicht gefunden – Projections-Check wird übersprungen.")


if __name__ == "__main__":
    main()