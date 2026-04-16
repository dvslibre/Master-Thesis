#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply known linear µ→HU mapping (HU = a*µ + b) with optional:
- robust outside-air removal
- smoothing
- keep-box cropping
- isotropic resampling
- single or 3-view preview PNG

Based on mu_to_hu_and_preview_4.py (no calibration fitting).
"""

import os, argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
from nibabel.processing import resample_to_output

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Outside-Air Removal ----------
def remove_outside_air_keep_inside(hu, air_thr=-750.0, tissue_thr=-550.0, close_iter=2):
    """Removes external air while keeping internal cavities (lung, bowel)."""
    hu = hu.copy()
    X, Y, Z = hu.shape
    outside = np.zeros((X, Y, Z), dtype=bool)

    for y in range(Y):  # axial slices
        sl = hu[:, y, :]
        tissue = sl > tissue_thr
        tissue_closed = binary_closing(tissue, structure=np.ones((3,3), bool), iterations=close_iter)
        passable = (sl < air_thr) & (~tissue_closed)

        seeds = np.zeros_like(passable, bool)
        seeds[0,:]=True; seeds[-1,:]=True; seeds[:,0]=True; seeds[:,-1]=True

        reach = seeds & passable
        last = -1
        while True:
            neigh = np.zeros_like(reach, bool)
            neigh[:-1,:] |= reach[1:,:]
            neigh[1: ,:] |= reach[:-1,:]
            neigh[:, :-1] |= reach[:, 1:]
            neigh[:, 1: ] |= reach[:, :-1]
            new_reach = (neigh | reach) & passable
            s = int(new_reach.sum())
            if s == last:
                break
            reach = new_reach; last = s
        outside[:, y, :] = reach

    outside = binary_dilation(outside, iterations=1)
    hu[outside] = -1024.0
    return hu


# ---------- Keep-Box ----------
def apply_keep_box_mm(hu, aff, keep):
    """Sets everything outside given box [Y0 Y1 Z0 Z1 X0 X1] to -1024 HU."""
    dx, dy, dz = float(aff[0,0]), float(aff[1,1]), float(aff[2,2])
    X, Y, Z = hu.shape
    if len(keep) not in (4,6):
        raise ValueError("--keep_box_mm expects 4 or 6 values (Y0 Y1 Z0 Z1 [X0 X1])")
    Y0,Y1,Z0,Z1 = keep[:4]
    X0,X1 = (keep[4], keep[5]) if len(keep)==6 else (0.0, dx*X)
    def clamp_idx(a,lo,hi): return int(max(lo, min(hi, round(a))))
    ix0,ix1 = clamp_idx(X0/dx,0,X), clamp_idx(X1/dx,0,X)
    iy0,iy1 = clamp_idx(Y0/dy,0,Y), clamp_idx(Y1/dy,0,Y)
    iz0,iz1 = clamp_idx(Z0/dz,0,Z), clamp_idx(Z1/dz,0,Z)
    mask = np.zeros_like(hu, bool)
    mask[ix0:ix1, iy0:iy1, iz0:iz1] = True
    hu[~mask] = -1024.0
    return hu


# ---------- Core µ→HU ----------
def apply_mu_to_hu(in_path, out_path, a, b,
                   clip=(-1024,2000), smooth_sigma=0.0, resample_mm=None,
                   air_thr=-750.0, tissue_thr=-550.0, close_iter=2,
                   keep_box_mm=None):
    print(f"[INFO] Lade: {in_path}")
    img = nib.load(in_path)
    vol = np.asarray(img.dataobj).astype(np.float32)
    aff = img.affine

    print(f"[MAP] HU = {a:.6f} * µ + {b:.2f}")
    hu = a * vol + b

    # Außenluft entfernen
    hu = remove_outside_air_keep_inside(hu, air_thr=air_thr, tissue_thr=tissue_thr, close_iter=close_iter)

    # optional glätten
    if smooth_sigma > 0:
        hu = gaussian_filter(hu, sigma=float(smooth_sigma))

    # optional harte Keep-Box
    if keep_box_mm is not None:
        hu = apply_keep_box_mm(hu, aff, keep_box_mm)

    # Clip & speichern
    hu_min, hu_max = clip
    hu = np.clip(hu, hu_min, hu_max).astype(np.float32)
    out_img = nib.Nifti1Image(hu, aff)
    out_img.header.set_qform(aff, code=1)
    out_img.header.set_sform(aff, code=1)
    nib.save(out_img, out_path)
    print(f"[DONE] gespeichert: {out_path}")

    # optional isotropes Resample
    if resample_mm and resample_mm > 0:
        print(f"[INFO] Resample auf isotrop {resample_mm:.2f} mm …")
        out_float = nib.Nifti1Image(hu.astype(np.float32), aff)
        rs = resample_to_output(out_float, voxel_sizes=(resample_mm,)*3, order=1)
        rs_data = np.clip(np.asarray(rs.dataobj), hu_min, hu_max).astype(np.int16)
        rs_img = nib.Nifti1Image(rs_data, rs.affine)
        rs_img.header.set_qform(rs.affine, code=1)
        rs_img.header.set_sform(rs.affine, code=1)
        root, ext = os.path.splitext(out_path)
        if ext.lower()==".gz":
            root, _ = os.path.splitext(root)
        resampled_path = root + f"_iso{resample_mm:.1f}mm.nii"
        nib.save(rs_img, resampled_path)
        print(f"[DONE] resampled: {resampled_path}")
        return resampled_path

    return out_path


# ---------- Preview ----------
def render_slice_png(nifti_path, out_png, view="coronal", base_h_in=7.0,
                     hu_window=(-1000,1000), percentile=None,
                     cmap="turbo", coronal_orient="AP", three_views=False):
    img = nib.load(nifti_path)
    img_ras = nib.as_closest_canonical(img)
    vol = np.asarray(img_ras.dataobj, dtype=np.float32)
    aff = img_ras.affine
    X, Y, Z = vol.shape
    dx, dy, dz = map(float, np.abs(np.diag(aff)[:3]))

    def get_view(v):
        v=v.lower()
        if v=="coronal":
            sl = vol[X//2,:,:]; im = np.flipud(sl)
            if coronal_orient.upper()=="AP": im = np.fliplr(im)
            extent=[0, dz*Z, 0, dy*Y]; xlabel,ylabel,ttl="Z [mm]","Y [mm]",f"Coronal ({coronal_orient})"
        elif v=="axial":
            sl = vol[:,Y//2,:]; im=np.flipud(sl.T)
            extent=[0, dx*X, 0, dz*Z]; xlabel,ylabel,ttl="X [mm]","Z [mm]","Axial"
        elif v=="sagittal":
            sl = vol[:,:,Z//2]; im=np.flipud(sl.T)
            extent=[0, dx*X, 0, dy*Y]; xlabel,ylabel,ttl="X [mm]","Y [mm]","Sagittal"
        else: raise ValueError("view must be coronal|axial|sagittal")
        return im, extent, xlabel, ylabel, ttl

    def window_limits(arr):
        if percentile:
            p1,p99 = np.percentile(arr, percentile); return float(p1), float(p99)
        return hu_window

    if not three_views:
        img2, ex, xlabel, ylabel, ttl = get_view(view)
        width_mm, height_mm = ex[1]-ex[0], ex[3]-ex[2]
        aspect = height_mm/max(width_mm,1e-6)
        h_in=base_h_in; w_in=h_in/max(aspect,1e-6)
        vmin,vmax = window_limits(img2)
        fig, ax = plt.subplots(figsize=(w_in,h_in), dpi=150)
        im = ax.imshow(img2, origin="lower", extent=ex, cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(ttl)
        cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("Hounsfield Units [HU]")
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(fig)
        print(f"[OK] Preview saved: {out_png} | window=[{vmin:.0f},{vmax:.0f}]")
        return

    # three-views mode
    views=["coronal","axial","sagittal"]
    metas=[get_view(v) for v in views]
    all_vals=np.concatenate([m[0].ravel() for m in metas])
    vmin,vmax = (np.percentile(all_vals, percentile) if percentile else hu_window)
    h_in=base_h_in; widths=[]
    for _,ex,*_ in metas:
        width_mm, height_mm = ex[1]-ex[0], ex[3]-ex[2]
        widths.append(h_in/max(height_mm/max(width_mm,1e-6),1e-6))
    w_in=sum(widths)+2.0
    fig,axes=plt.subplots(1,3, figsize=(w_in,h_in), dpi=150)
    for ax,(img2,ex,xlabel,ylabel,ttl) in zip(axes, metas):
        im=ax.imshow(img2, origin="lower", extent=ex, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(ttl)
    cbar=fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8); cbar.set_label("Hounsfield Units [HU]")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(fig)
    print(f"[OK] Preview (3) saved: {out_png} | window=[{vmin:.0f},{vmax:.0f}]")


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Apply µ→HU mapping with known a,b + options")
    ap.add_argument("--in",  dest="infile",  required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--b", type=float, required=True)
    ap.add_argument("--clip", nargs=2, type=float, default=[-1024,2000])
    ap.add_argument("--smooth_sigma", type=float, default=0.0)
    ap.add_argument("--resample_mm", type=float, default=None)
    ap.add_argument("--air_thr", type=float, default=-750.0)
    ap.add_argument("--tissue_thr", type=float, default=-550.0)
    ap.add_argument("--close_iter", type=int, default=2)
    ap.add_argument("--keep_box_mm", nargs="+", type=float)
    ap.add_argument("--png", dest="pngfile", help="PNG output path for preview")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--view", choices=["coronal","axial","sagittal"], default="coronal")
    ap.add_argument("--percentile", nargs=2, type=float)
    ap.add_argument("--hu_window", nargs=2, type=float, default=[-1000,1000])
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--base_h_in", type=float, default=7.0)
    ap.add_argument("--tri", action="store_true")
    ap.add_argument("--coronal_orient", choices=["AP","PA"], default="AP")
    args = ap.parse_args()

    hu_path = apply_mu_to_hu(
        in_path=args.infile, out_path=args.outfile, a=args.a, b=args.b,
        clip=tuple(args.clip), smooth_sigma=args.smooth_sigma, resample_mm=args.resample_mm,
        air_thr=args.air_thr, tissue_thr=args.tissue_thr, close_iter=args.close_iter,
        keep_box_mm=args.keep_box_mm
    )

    if args.preview or args.tri:
        # robust PNG name handling for .nii.gz
        if args.pngfile:
            png = args.pngfile if not args.tri else args.pngfile.replace(".png","_3views.png")
        else:
            root, ext = os.path.splitext(args.outfile)
            if ext.lower()==".gz":
                root, _ = os.path.splitext(root)
            png = root + ("_3views.png" if args.tri else ".png")

        render_slice_png(
            nifti_path=hu_path, out_png=png, view=args.view,
            base_h_in=args.base_h_in, hu_window=tuple(args.hu_window),
            percentile=tuple(args.percentile) if args.percentile else None,
            cmap=args.cmap, coronal_orient=args.coronal_orient, three_views=bool(args.tri)
        )

if __name__ == "__main__":
    main()