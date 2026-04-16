#!/usr/bin/env python3
"""
fit_mu_mapping.py

Ziel:
Aus XCAT-Phantomen eine empirische Abbildung
    µ_208keV = f(µ_80keV)
lernen.

Voraussetzung:
Im Phantom-Ordner <base>/out/ liegen:
  - ct_att.npy    -> µ bei ~80 keV (Proxy: CT)
  - spect_att.npy -> µ bei ~208 keV (Proxy: SPECT)

Dieses Skript:
  - lädt beide Volumina
  - extrahiert gültige Voxel (µ_80 > 0, µ_208 > 0)
  - optional: clippt extreme Ausreißer (Perzentil-Grenzen)
  - sampled zufällig max_samples Punkte
  - fitten eines Polynoms µ_208 = P(µ_80)
  - speichert:
      * fit_mu_mapping.png (Scatter + Fit-Kurve)
      * fit_mu_mapping.json (Polynom-Koeffizienten und Metadaten)
      * gibt die Koeffizienten auf stdout aus

Beispielaufruf:

python fit_mu_mapping.py \
  --base /home/mnguest12/projects/thesis/Data_Processing/phantom_01 \
  --degree 2 \
  --max_samples 200000 \
  --scatter_samples 20000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Fit µ_208keV = f(µ_80keV) aus spect_att.npy und ct_att.npy.")
    p.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Phantom-Basisordner (der mit out/ drin, z.B. phantom_01)",
    )
    p.add_argument(
        "--degree",
        type=int,
        default=1,
        help="Polynomgrad für µ208 = f(µ80) (z.B. 1=linear, 2=quadratisch).",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=200_000,
        help="Maximale Anzahl Punkte für das Fitting (zufälliges Subsampling).",
    )
    p.add_argument(
        "--scatter_samples",
        type=int,
        default=20_000,
        help="Anzahl der Punkte, die im Scatterplot angezeigt werden.",
    )
    p.add_argument(
        "--lower_percentile",
        type=float,
        default=0.1,
        help="Unteres Perzentil zum Clipping der µ80/µ208-Werte (z.B. 0.1).",
    )
    p.add_argument(
        "--upper_percentile",
        type=float,
        default=99.9,
        help="Oberes Perzentil zum Clipping der µ80/µ208-Werte (z.B. 99.9).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    base = args.base.resolve()
    out_dir = base / "out"

    if not out_dir.exists():
        raise FileNotFoundError(f"out-Ordner nicht gefunden: {out_dir}")

    ct_path    = out_dir / "ct_att.npy"     # ~ µ(80 keV)
    spect_path = out_dir / "spect_att.npy"  # ~ µ(208 keV)

    if not ct_path.exists():
        raise FileNotFoundError(ct_path)
    if not spect_path.exists():
        raise FileNotFoundError(spect_path)

    print(f"[INFO] Lade µ(80 keV) aus:  {ct_path}")
    print(f"[INFO] Lade µ(208 keV) aus: {spect_path}")

    mu80 = np.load(ct_path).astype(np.float32).ravel()
    mu208 = np.load(spect_path).astype(np.float32).ravel()

    if mu80.shape != mu208.shape:
        raise ValueError(f"Shape mismatch nach Flatten: mu80={mu80.shape}, mu208={mu208.shape}")

    # Gültige Voxel: positive µ-Werte
    valid = (mu80 > 0) & (mu208 > 0)
    mu80_valid = mu80[valid]
    mu208_valid = mu208[valid]

    print(f"[INFO] Gültige Voxel (µ80>0 & µ208>0): {mu80_valid.size}")

    if mu80_valid.size < 1000:
        print("[WARN] Sehr wenige gültige Voxel, Fit könnte instabil sein.")

    # Perzentil-Clipping gegen Ausreißer
    lp = args.lower_percentile
    up = args.upper_percentile

    mu80_low, mu80_high = np.percentile(mu80_valid, [lp, up])
    mu208_low, mu208_high = np.percentile(mu208_valid, [lp, up])

    mask_clip = (
        (mu80_valid >= mu80_low) & (mu80_valid <= mu80_high) &
        (mu208_valid >= mu208_low) & (mu208_valid <= mu208_high)
    )

    mu80_clip = mu80_valid[mask_clip]
    mu208_clip = mu208_valid[mask_clip]

    print(f"[INFO] Nach Clipping auf Perzentile [{lp}, {up}]: {mu80_clip.size} Punkte")
    print(f"       µ80 in [{mu80_clip.min():.4f}, {mu80_clip.max():.4f}]")
    print(f"       µ208 in [{mu208_clip.min():.4f}, {mu208_clip.max():.4f}]")

    # Zufälliges Subsampling für das Fitting
    np.random.seed(0)
    n = mu80_clip.size
    n_fit = min(args.max_samples, n)
    idx_fit = np.random.choice(n, size=n_fit, replace=False)

    x_fit = mu80_clip[idx_fit]
    y_fit = mu208_clip[idx_fit]

    print(f"[INFO] Fitting mit {n_fit} Punkten, Polynomgrad={args.degree}")

    # Polynom-Fit µ208 = P(µ80)
    coeffs = np.polyfit(x_fit, y_fit, deg=args.degree)
    # coeffs: [a_n, a_{n-1}, ..., a0]
    print(f"[FIT] Koeffizienten (höchste Potenz zuerst):")
    for i, c in enumerate(coeffs):
        power = args.degree - i
        print(f"       a_{power} = {c:.6e}")

    # Achsenbereich für Plot (gemeinsamer Bereich beider µ)
    vmin = min(mu80_clip.min(), mu208_clip.min())
    vmax = max(mu80_clip.max(), mu208_clip.max())

    # Für Plot: Fit-Linie über gemeinsamen Bereich
    x_line = np.linspace(vmin, vmax, 200)
    y_line = np.polyval(coeffs, x_line)

    # Scatterpunkte für Plot
    n_scatter = min(args.scatter_samples, n)
    idx_scatter = np.random.choice(n, size=n_scatter, replace=False)
    x_sc = mu80_clip[idx_scatter]
    y_sc = mu208_clip[idx_scatter]

    # Plot: Scatter + Fitkurve + Diagonale, gleiche Achsen
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_sc, y_sc, s=8, alpha=0.5, label="Voxels (Sample)")
    ax.plot(x_line, y_line, "r", linewidth=2, label=f"Poly fit (deg={args.degree})")

    # Diagonale y=x als Referenz
    ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5, label="y = x")

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"$\mu_{80\ \mathrm{keV}}$  (CT-Attenuation)")
    ax.set_ylabel(r"$\mu_{208\ \mathrm{keV}}$  (SPECT-Attenuation)")
    ax.set_title(r"Fit: $\mu_{208} = f(\mu_{80})$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    png_path = base / "fit_mu_mapping.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"[OUT] Plot gespeichert: {png_path}")


    # JSON mit Fit-Infos
    fit_info = {
        "base": str(base),
        "ct_att_path": str(ct_path),
        "spect_att_path": str(spect_path),
        "degree": int(args.degree),
        "coeffs_highest_first": [float(c) for c in coeffs],
        "lower_percentile": float(lp),
        "upper_percentile": float(up),
        "n_valid": int(mu80_valid.size),
        "n_after_clipping": int(mu80_clip.size),
        "n_fit": int(n_fit),
        "mu80_range_clipped": [float(mu80_clip.min()), float(mu80_clip.max())],
        "mu208_range_clipped": [float(mu208_clip.min()), float(mu208_clip.max())],
        "note": "Use mu208 = polyval(coeffs_highest_first, mu80)",
    }

    json_path = base / "fit_mu_mapping.json"
    with open(json_path, "w") as f:
        json.dump(fit_info, f, indent=2)
    print(f"[OUT] Fit-Parameter gespeichert: {json_path}")

    print("\n[INFO] Beispiel: Umrechnung im Code")
    print("  import numpy as np")
    print(f"  coeffs = np.array({fit_info['coeffs_highest_first']})")
    print("  mu208 = np.polyval(coeffs, mu80)")


if __name__ == "__main__":
    main()
