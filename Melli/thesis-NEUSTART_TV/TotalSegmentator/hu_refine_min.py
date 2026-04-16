#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal: lineare HU-Feinkalibrierung + einfache Preview.

- Modus A (empfohlen): direkt gemessene Mediane (--measured-*)
- Modus B: räumliche Anker "x_mm,y_mm,z_mm:HU"

Beispiel (nur Coronal-Preview):
python hu_refine_min.py \
  --in runs_360/ct_recon_rtk_HU_iso1.0mm.nii \
  --out runs_360/ct_recon_rtk_HU_finecal.nii.gz \
  --measured-lung -569 --measured-liver 125 --measured-kidney 102 \
  --preview --view coronal --png runs_360/ts_preview/HU_finecal_coronal.png
"""

import os, re, sys, argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Helpers ----------
def mm_to_idx(mm, spacing_mm):
    return int(round(float(mm) / float(spacing_mm)))

def parse_mm(token, total_mm):
    token = str(token).strip().lower()
    if token in ("x/2","y/2","z/2"):
        axis = token[0]
        return dict(x=total_mm[0]/2, y=total_mm[1]/2, z=total_mm[2]/2)[axis]
    m = re.fullmatch(r'(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)', token)
    return float(token) if not m else total_mm[0]*(float(m.group(1))/float(m.group(2)))

def fit_ab(x_vals, hu_targets):
    x = np.asarray(x_vals, float); y = np.asarray(hu_targets, float)
    A = np.stack([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    rmse = float(np.sqrt(np.mean((a*x + b - y)**2)))
    return float(a), float(b), rmse

# ---------- Preview ----------
def render_slice_png(nifti_path, out_png, view="coronal",
                     hu_window=(-1000,1000), base_h_in=7.0,
                     coronal_orient="AP"):
    img = nib.load(nifti_path)
    img_ras = nib.as_closest_canonical(img)
    vol = np.asarray(img_ras.dataobj, dtype=np.float32)
    aff = img_ras.affine
    X,Y,Z = vol.shape
    dx,dy,dz = map(float, np.abs(np.diag(aff)[:3]))

    v = view.lower()
    if v=="coronal":
        sl = vol[X//2,:,:]
        if coronal_orient.upper()=="AP":
            im = np.flipud(np.fliplr(sl))
        else:
            im = np.flipud(sl)
        extent = [0, dz*Z, 0, dy*Y]; xlabel,ylabel,ttl="Z [mm]","Y [mm]",f"Coronal ({coronal_orient})"
    elif v=="axial":
        sl = vol[:,Y//2,:]; im=np.flipud(sl.T)
        extent = [0, dx*X, 0, dz*Z]; xlabel,ylabel,ttl="X [mm]","Z [mm]","Axial"
    elif v=="sagittal":
        sl = vol[:,:,Z//2]; im=np.flipud(sl.T)
        extent = [0, dx*X, 0, dy*Y]; xlabel,ylabel,ttl="X [mm]","Y [mm]","Sagittal"
    else:
        raise ValueError("view must be coronal|axial|sagittal")

    width_mm, height_mm = extent[1]-extent[0], extent[3]-extent[2]
    aspect = height_mm/max(width_mm,1e-6)
    h_in=base_h_in; w_in=h_in/max(aspect,1e-6)
    vmin,vmax = hu_window

    fig, ax = plt.subplots(figsize=(w_in,h_in), dpi=150)
    imh = ax.imshow(im, origin="lower", extent=extent, cmap="turbo",
                    vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(ttl)
    cb = fig.colorbar(imh, ax=ax, shrink=0.8); cb.set_label("Hounsfield Units [HU]")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(fig)
    print(f"[OK] Preview saved: {out_png} | window=[{vmin:.0f},{vmax:.0f}]")

# ---------- Core ----------
def main():
    ap = argparse.ArgumentParser(description="Minimal HU fine calibration + preview")
    ap.add_argument("--in",  dest="infile",  required=True)
    ap.add_argument("--out", dest="outfile", required=True)

    # Modus A: direkt gemessene Mediane
    ap.add_argument("--measured-lung",   type=float, dest="meas_lung")
    ap.add_argument("--measured-liver",  type=float, dest="meas_liver")
    ap.add_argument("--measured-kidney", type=float, dest="meas_kidney")
    ap.add_argument("--target-lung",   type=float, default=-850.0)
    ap.add_argument("--target-liver",  type=float, default=55.0)
    ap.add_argument("--target-kidney", type=float, default=35.0)

    # Modus B: räumliche Anker
    ap.add_argument("--anchor", action="append", default=[],
        help='Format "x_mm,y_mm,z_mm:HU" (mehrfach). Beispiel: "x/2,450,90:-850"')
    ap.add_argument("--radius_mm", type=float, default=8.0)

    # Clip + Preview
    ap.add_argument("--clip", nargs=2, type=float, default=[-1024,2000])
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--png", default=None)
    ap.add_argument("--view", choices=["coronal","axial","sagittal"], default="coronal")
    ap.add_argument("--hu_window", nargs=2, type=float, default=[-1000,1000])
    ap.add_argument("--base_h_in", type=float, default=7.0)
    ap.add_argument("--coronal_orient", choices=["AP","PA"], default="AP")
    args = ap.parse_args()

    # Lade Volumen
    img = nib.load(args.infile)
    img_ras = nib.as_closest_canonical(img)
    vol = np.asarray(img_ras.dataobj).astype(np.float32)
    aff = img_ras.affine
    dx,dy,dz = map(float, np.abs(np.diag(aff)[:3]))
    X,Y,Z = vol.shape
    size_mm = (dx*X, dy*Y, dz*Z)

    # Eingaben sammeln
    x_vals, hu_targets = [], []

    # Modus A: gemessene Mediane
    meas = []
    if args.meas_lung   is not None: meas.append(("lung",   float(args.meas_lung),   float(args.target_lung)))
    if args.meas_liver  is not None: meas.append(("liver",  float(args.meas_liver),  float(args.target_liver)))
    if args.meas_kidney is not None: meas.append(("kidney", float(args.meas_kidney), float(args.target_kidney)))
    if len(meas) >= 2:
        for name, m, t in meas:
            print(f"[PAIR] measured {name}: µ_median={m:.3f} → target HU={t:.1f}")
            x_vals.append(m); hu_targets.append(t)

    # Modus B: räumliche Anker
    if len(x_vals) < 2 and args.anchor:
        for a_str in args.anchor:
            coord, hu_t = a_str.split(":")
            xs,ys,zs = [t.strip() for t in coord.split(",")]
            # mm-Koord interpretieren
            x_mm = parse_mm(xs, size_mm)
            y_mm = parse_mm(ys, (size_mm[1],)*3)
            z_mm = parse_mm(zs, (size_mm[2],)*3)
            ix,iy,iz = mm_to_idx(x_mm,dx), mm_to_idx(y_mm,dy), mm_to_idx(z_mm,dz)
            r_vox = max(1, int(round(args.radius_mm/min(dy,dz))))
            yy, zz = np.meshgrid(
                np.arange(max(0, iy-r_vox), min(Y, iy+r_vox+1)),
                np.arange(max(0, iz-r_vox), min(Z, iz+r_vox+1)),
                indexing="ij")
            mask = (yy-iy)**2 + (zz-iz)**2 <= r_vox**2
            mu_med = float(np.median(vol[ix, yy[mask], zz[mask]]))
            x_vals.append(mu_med); hu_targets.append(float(hu_t))
            print(f"[ANCHOR] {a_str} → idx=({ix},{iy},{iz}) r={r_vox} → µ_median={mu_med:.6f}")

    if len(x_vals) < 2:
        print("[ERR] Mindestens 2 Paare nötig: entweder --measured-* oder --anchor.", file=sys.stderr)
        sys.exit(1)

    # Fit & Anwenden
    a,b,rmse = fit_ab(x_vals, hu_targets)
    print(f"[FIT] HU = {a:.6f} * µ + {b:.2f} | RMSE={rmse:.2f} HU")

    hu = a*vol + b
    hu = np.clip(hu, args.clip[0], args.clip[1]).astype(np.float32)

    out = nib.Nifti1Image(hu, img_ras.affine, img_ras.header)
    out.header.set_qform(img_ras.affine, code=1); out.header.set_sform(img_ras.affine, code=1)
    nib.save(out, args.outfile)
    print(f"[DONE] saved: {args.outfile}")

    # Preview
    if args.preview:
        png = args.png or (os.path.splitext(args.outfile)[0] + "_preview.png")
        render_slice_png(args.outfile, png, view=args.view,
                         hu_window=tuple(args.hu_window),
                         base_h_in=args.base_h_in,
                         coronal_orient=args.coronal_orient)

if __name__ == "__main__":
    main()