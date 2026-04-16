#!/usr/bin/env python3
"""
check_preprocessing.py — Sanity-Check für orientierte Volumina und Projektionen

Erzeugt:
  - check_coronal_slices.png
  - check_ray_intensities_from_npz.png
in <base>/ gespeichert.

Aufruf: 
python check_preprocessing.py \
  --base /home/mnguest12/projects/thesis/RayNet/phantom_04
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
        help="Phantom-Basisordner (der mit data/ und out/ drin, z.B. phantom_04)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    base = args.base
    data_dir = base / "data"
    out_dir  = base / "out"

    mu_path   = data_dir / "mu.npy"
    act_path  = data_dir / "act.npy"
    mask_path = data_dir / "mask.npy"
    ap_path   = data_dir / "ap.npy"
    pa_path   = data_dir / "pa.npy"
    rays_path = out_dir  / "rays_train.npz"

    for pth in [mu_path, act_path, mask_path, ap_path, pa_path, rays_path]:
        if not pth.exists():
            raise FileNotFoundError(pth)

    mu   = np.load(mu_path)   # (D,H,W)
    act  = np.load(act_path)  # (D,H,W)
    mask = np.load(mask_path) # (D,H,W)
    ap   = np.load(ap_path)   # (Hproj,Wproj) = (256,651)
    pa   = np.load(pa_path)

    print("[INFO] Shapes:",
          "mu",   mu.shape,
          "act",  act.shape,
          "mask", mask.shape,
          "ap",   ap.shape,
          "pa",   pa.shape)

    # ---------------------------------------------------------
    # 1) Koronale Slices (µ, Aktivität, Maske)
    #    Volumen: (D,H,W)
    #    Koronalebene: (D,W) = mu[:, mid_H, :]
    #    Rotation: 90° im Uhrzeigersinn (k=-1)
    # ---------------------------------------------------------
    D, H, W = mu.shape
    mid_H = H // 2

    mu_cor   = np.flip(mu[:, mid_H, :], axis=0)
    act_cor  = np.flip(act[:, mid_H, :], axis=0)
    mask_cor = np.flip(mask[:, mid_H, :], axis=0)

    mu_cor_plot   = np.flipud(np.rot90(mu_cor,   k=-1))
    act_cor_plot  = np.flipud(np.rot90(act_cor,  k=-1))
    mask_cor_plot = np.flipud(np.rot90(mask_cor, k=-1))

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axs[0].imshow(mu_cor_plot, cmap="gray")
    axs[0].set_title(f"µ – coronal (H={mid_H})")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(act_cor_plot, cmap="inferno")
    axs[1].set_title("Activity – coronal")
    plt.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(mask_cor_plot, cmap="tab20")
    axs[2].set_title("Mask – coronal")
    plt.colorbar(im2, ax=axs[2])

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    out1 = base / "check_coronal_slices.png"
    plt.savefig(out1, dpi=200)
    plt.close()
    print(f"[OUT] Saved {out1}")


    # ---------------------------------------------------------
    # 2) AP/PA-Projektionen (volle 2D-Ansichten)
    # ---------------------------------------------------------
    rays_npz = np.load(rays_path)
    I_pairs  = rays_npz["I_pairs"]   # [M,2]
    xy_pairs = rays_npz.get("xy_pairs", None)

    Hproj, Wproj = ap.shape
    M = I_pairs.shape[0]
    print(f"[INFO] rays_train.npz: M={M}, Hproj={Hproj}, Wproj={Wproj}")

    # Nur obere zwei Plots: volle Projektionen AP/PA
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    vmin_ap = ap.min()
    vmax_ap = ap.max()
    vmin_pa = pa.min()
    vmax_pa = pa.max()

    im0 = axes[0].imshow(np.rot90(ap, k=1), cmap="plasma", origin="lower", aspect="equal",
                         vmin=vmin_ap, vmax=vmax_ap)
    axes[0].set_title("AP (aus ap.npy)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(np.rot90(pa, k=1), cmap="plasma", origin="lower", aspect="equal",
                         vmin=vmin_pa, vmax=vmax_pa)
    axes[1].set_title("PA (aus pa.npy)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    out2 = base / "check_ray_intensities_from_npz.png"
    plt.savefig(out2, dpi=200)
    plt.close(fig)
    print(f"[OUT] Saved {out2}")




if __name__ == "__main__":
    main()