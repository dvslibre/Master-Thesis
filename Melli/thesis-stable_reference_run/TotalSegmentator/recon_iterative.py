#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative CT-Rekonstruktion (SART / optional OS-SART, falls vorhanden) mit ITK-RTK
aus xCAT/ct_projector-Projektionen.

- Detektorbreite/-höhe aus Half_fan_angle + SDD (Breite=2*SDD*tan(half_fan))
- Optional: dv=du (--square_pixels)
- Optional: FOV in Iso-Ebene automatisch (--auto_iso_fov)
- Optional: Projektionen ausdünnen (--proj_stride)
- Optional: v-Cropping des Detektors (--keep_rows) → reduziert vert. Cone-Angle
- Ausgabe: NIfTI, kanonisch RAS (X,Y,Z) mit korrekter Affine
"""

import argparse, re, os
import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- I/O ----------
def parse_proj_txt(path):
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()

    def grab(keys):
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            m = re.search(rf'^\s*([+-]?\d+(?:\.\d+)?)\s*:\s*{re.escape(key)}', txt, re.M | re.I)
            if m:
                return float(m.group(1))
        return None

    geom = {
        "width_mm":  grab(["width(mms)", "width"]),
        "height_mm": grab(["height(mms)", "height"]),
        "nr": int(grab("num_rows") or 0),
        "nc": int(grab("num_channels") or 0),
        "DSO": grab(["distance_to_source(mms)", "distance_to_source"]),
        "DTD": grab(["distance_to_detector(mms)", "distance_to_detector"]),
        "det_shift_u": grab("detector_shift") or 0.0,
        "half_fan_deg": grab(["Half_fan_angle", "half_fan_angle", "half_fan"]),
    }
    assert geom["DSO"] is not None and geom["DTD"] is not None, "DSO/DTD fehlen in proj.txt"
    return geom


def load_mu_stack_any(mat_path):
    mu, ang = None, None
    try:
        from scipy.io import loadmat
        M = loadmat(mat_path, squeeze_me=True, simplify_cells=True)
        mu = np.asarray(M["mu_stack"], np.float32)
        if "angles_out" in M:
            ang = np.asarray(M["angles_out"], np.float32)
        elif "angles_deg" in M:
            ang = np.asarray(M["angles_deg"], np.float32)
    except Exception:
        import h5py
        with h5py.File(mat_path, "r") as f:
            mu = np.array(f["mu_stack"], dtype=np.float32)
            if mu.shape[0] != mu.shape[-1]:
                mu = np.transpose(mu, (2, 1, 0))  # (nr,nc,nProj) -> (nProj,nr,nc)
                print(f"[FIX] mu_stack transposed → {mu.shape} (nProj, nr, nc)")
            if "angles_out" in f:
                ang = np.array(f["angles_out"], dtype=np.float32).squeeze()
            elif "angles_deg" in f:
                ang = np.array(f["angles_deg"], dtype=np.float32).squeeze()
    if mu.ndim != 3:
        raise ValueError(f"mu_stack hat ndims={mu.ndim}, erwarte 3.")
    return mu.astype(np.float32, copy=False), (None if ang is None else ang.astype(np.float32, copy=False))


# ---------- Preview ----------
def save_preview_png(nifti_path, out_png, view="coronal", base_h_in=7.0, title=None, coronal_orient="AP"):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    img = nib.load(nifti_path); img = nib.as_closest_canonical(img)
    vol = np.asarray(img.dataobj, dtype=np.float32); aff = img.affine
    X, Y, Z = vol.shape; dx, dy, dz = float(aff[0,0]), float(aff[1,1]), float(aff[2,2])

    v = view.lower()
    if v == "coronal":
        sl = vol[X//2, :, :]
        img2 = np.flipud(sl)
        if str(coronal_orient).upper() == "AP":
            img2 = np.fliplr(img2)
        ex = [0, dz*Z, 0, dy*Y]; xlabel, ylabel = "Z [mm]", "Y [mm]"; ttl = title or f"Coronal ({coronal_orient})"
    elif v == "axial":
        sl = vol[:, Y//2, :]; img2 = np.flipud(sl.T)
        ex = [0, dx*X, 0, dz*Z]; xlabel, ylabel, ttl = "X [mm]", "Z [mm]", "Axial"
    elif v == "sagittal":
        sl = vol[:, :, Z//2]; img2 = np.flipud(sl.T)
        ex = [0, dx*X, 0, dy*Y]; xlabel, ylabel, ttl = "X [mm]", "Y [mm]", "Sagittal"
    else:
        raise ValueError("view must be coronal|axial|sagittal")

    p2,p98 = np.percentile(img2,[2,98]); shown = np.clip((img2-p2)/(p98-p2+1e-6),0,1)
    aspect = (ex[3]-ex[2]) / max((ex[1]-ex[0]),1e-6); h_in = base_h_in; w_in = h_in / max(aspect,1e-6)
    fig = plt.figure(figsize=(w_in,h_in), dpi=150); ax = plt.gca()
    ax.imshow(shown, cmap="gray", origin="lower", extent=ex, aspect="equal")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(ttl)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(fig)
    print(f"[INFO] Preview ({view}) gespeichert: {out_png}")


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="ct_proj_stack.mat")
    ap.add_argument("--proj", default="proj.txt")
    ap.add_argument("--out", default="iter_recon.nii")
    ap.add_argument("--out_ras", default=None)

    # Volumen-Raster (kann mit --auto_iso_fov ersetzt werden)
    ap.add_argument("--nx", type=int, default=320)
    ap.add_argument("--ny", type=int, default=512)
    ap.add_argument("--nz", type=int, default=320)
    ap.add_argument("--sx", type=float, default=500.0)
    ap.add_argument("--sy", type=float, default=800.0)
    ap.add_argument("--sz", type=float, default=500.0)
    ap.add_argument("--auto_iso_fov", action="store_true",
                    help="Setzt sx/sy/sz automatisch auf Iso-FOV (aus Detektor + DSO/SDD).")

    # Detektor / Geometrie
    ap.add_argument("--detw", type=float, default=None, help="Detektorbreite [mm] override")
    ap.add_argument("--deth", type=float, default=None, help="Detektorhöhe [mm] override")
    ap.add_argument("--square_pixels", action="store_true",
                    help="dv=du setzen und height_det = width_det * (nr/nc) ableiten")
    ap.add_argument("--angle_offset", type=float, default=0.0)
    ap.add_argument("--reverse_angles", action="store_true")
    ap.add_argument("--flip_u", action="store_true")
    ap.add_argument("--shift_v", type=float, default=0.0, help="Detektor-Versatz v [mm]")
    ap.add_argument("--scale_du", type=float, default=1.0, help="Skalierung u-Pitch")
    ap.add_argument("--scale_dso", type=float, default=1.0, help="Skaliert DSO & DTD gemeinsam")

    # Iterativ
    ap.add_argument("--iters", type=int, default=20, help="SART Iterationen")
    ap.add_argument("--relax", type=float, default=None, help="Relaxationsfaktor (Lambda), z.B. 0.7")
    ap.add_argument("--gauss_sigma_mm", type=float, default=0.0, help="3D-Gauss σ [mm] (0=aus)")

    # Vorschau / Reduktionen
    ap.add_argument("--proj_stride", type=int, default=1, help="jede k-te Projektion verwenden (1=alle)")
    ap.add_argument("--keep_rows", type=int, default=None, help="nur mittlere v-Zeilen behalten (z.B. 440, 420)")

    # Preview
    ap.add_argument("--quick_ap_png", type=str, default=None)
    ap.add_argument("--preview_view", choices=["coronal","axial","sagittal"], default="coronal")
    ap.add_argument("--coronal_orient", choices=["AP","PA"], default="AP")
    ap.add_argument("--base_h_in", type=float, default=6.0)
    args = ap.parse_args()

    # Daten laden
    mu_raw, ang = load_mu_stack_any(args.mat)

    # optional: Projektionen ausdünnen (z.B. jede 2.)
    stride = int(getattr(args, "proj_stride", 1))
    if stride > 1:
        mu_raw = mu_raw[::stride].copy()
        if ang is not None:
            ang = ang[::stride].copy()
        print(f"[SPEED] proj_stride={stride} → nProj={mu_raw.shape[0]}")
    print(f"[INFO] mu_raw shape: {mu_raw.shape}")

    G = parse_proj_txt(args.proj)

    DSO = float(G["DSO"]) * args.scale_dso
    DTD = float(G["DTD"]) * args.scale_dso
    SDD = DSO + DTD

    nproj, nr, nc = mu_raw.shape
    if G["nr"] and G["nc"] and (G["nr"] != nr or G["nc"] != nc):
        raise ValueError(f"Detektorgröße uneinheitlich: mu={nr}x{nc}, proj.txt={G['nr']}x{G['nc']}")

    # Detektorbreite/Höhe bestimmen
    if args.detw is not None and args.deth is not None:
        width_mm, height_mm, src = float(args.detw), float(args.deth), "CLI overrides"
    elif G.get("half_fan_deg") is not None:
        half = float(G["half_fan_deg"])
        width_mm = 2.0 * SDD * np.tan(np.deg2rad(half))
        height_mm = width_mm * (nr / nc) if args.square_pixels else (float(G["height_mm"]) or width_mm * (nr / nc))
        src = f"half_fan={half}°" + (" + square_pixels" if args.square_pixels else " (+height fallback)")
    else:
        width_mm  = float(args.detw) if args.detw else (float(G["width_mm"]) or args.sx)
        height_mm = float(args.deth) if args.deth else (float(G["height_mm"]) or args.sy)
        src = "proj.txt fallback"

    du = (width_mm / nc) * args.scale_du
    dv = height_mm / nr
    if args.square_pixels:
        dv = du
        height_mm = dv * nr

    shift_u_raw = float(G["det_shift_u"])
    shift_u_mm  = shift_u_raw * du if abs(shift_u_raw) < 2.0 else shift_u_raw

    print(f"[GEOM] Quelle: {src}")
    print(f"[GEOM] width_det={width_mm:.2f} mm, height_det={height_mm:.2f} mm | du={du:.4f} mm, dv={dv:.4f} mm")
    print(f"[GEOM] DSO={DSO:.1f} mm, DTD={DTD:.1f} mm, SDD={SDD:.1f} mm | det_shift_u(mm)={shift_u_mm:.3f}")

    # Iso-FOV
    M = SDD / DSO
    width_iso, height_iso = width_mm / M, height_mm / M
    print(f"[SUGGEST] Iso-FOV(mm): sx/sz≈{width_iso:.1f}, sy≈{height_iso:.1f}  (M={M:.3f})")

    # FOV übernehmen
    sx, sy, sz_mm = args.sx, args.sy, args.sz
    if args.auto_iso_fov:
        sx, sy, sz_mm = width_iso, height_iso, width_iso
        print(f"[AUTO] set FOV(mm) → sx={sx:.1f}, sy={sy:.1f}, sz={sz_mm:.1f}")

    # Winkel
    if ang is None or len(ang) != nproj:
        ang = np.linspace(0, 360, nproj, endpoint=False, dtype=np.float32)
        print("[WARN] angles_* fehlte – setze 0..360° gleichmäßig.")
    if args.reverse_angles:
        ang = ang[::-1]; print("[INFO] reverse_angles aktiv")
    if args.angle_offset != 0.0:
        ang = (ang + args.angle_offset) % 360.0; print(f"[INFO] angle_offset = {args.angle_offset}°")
    print(f"[INFO] angles[0:5]={np.asarray(ang[:5]).round(1)} ... angles[-5:]=[{np.asarray(ang[-5:]).round(1)}] (n={len(ang)})")

    # Checks (vor Crop)
    cone_u = np.degrees(np.arctan(0.5 * du * nc / SDD))
    cone_v = np.degrees(np.arctan(0.5 * dv * nr / SDD))
    print(f"[CHECK] cone half-angles u/v ≈ {cone_u:.2f}° / {cone_v:.2f}°, magnification={M:.3f}")

    # Projektionen → ITK
    mu = np.asarray(mu_raw, dtype=np.float32, order="C")

    # --- optionales v-Cropping (Cone-Angle reduzieren) ---
    if args.keep_rows:
        k = int(args.keep_rows)
        assert 0 < k <= nr, "--keep_rows muss <= num_rows sein"
        start = (nr - k) // 2
        mu = mu[:, start:start+k, :].copy()
        nr = mu.shape[1]
        height_mm = dv * nr  # neue phys. Detektorhöhe
        cone_v2 = np.degrees(np.arctan(0.5 * dv * nr / SDD))
        print(f"[REDUCE] keep_rows={k} → nr={nr}, new height_det={height_mm:.1f} mm, cone_v≈{cone_v2:.2f}°")

    if args.flip_u:
        mu = mu[:, :, ::-1].copy()
        shift_u_mm = -shift_u_mm
        print("[INFO] u-axis flipped → det_shift_u sign inverted")

    # --- ITK / RTK ---
    import itk
    from itk import RTK as rtk

    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for a in ang:
        geometry.AddProjection(DSO, SDD, float(a), shift_u_mm, float(args.shift_v))

    proj_img = itk.GetImageFromArray(mu)      # (nProj, nr, nc) -> ITK (z,y,x)
    proj_img.SetSpacing((du, dv, 1.0))
    proj_img.SetOrigin((-width_mm/2.0, -height_mm/2.0, 0.0))
    Mdir = itk.Matrix[itk.D, 3, 3](); Mdir.SetIdentity()
    proj_img.SetDirection(Mdir)

    # Volumen
    nx, ny, nz = args.nx, args.ny, args.nz
    dx, dy, dz = sx/nx, sy/ny, sz_mm/nz
    vol_np = np.zeros((nz, ny, nx), dtype=np.float32)
    vol_img = itk.GetImageFromArray(vol_np)
    vol_img.SetSpacing((dx, dy, dz))
    vol_img.SetOrigin((-sx/2.0, -sy/2.0, -sz_mm/2.0))
    vol_img.SetDirection(Mdir)

    print(f"[INFO] Vol: ({nx},{ny},{nz}) vox | FOV(mm)=({sx:.1f},{sy:.1f},{sz_mm:.1f}) → voxel(mm)=({dx:.3f},{dy:.3f},{dz:.3f})")

    # Prefer OS-SART if available; else SART
    if hasattr(rtk, "OSConeBeamReconstructionFilter"):
        sart = rtk.OSConeBeamReconstructionFilter.New()
        print("[INFO] Using OS-SART (ordered subsets).")
        # Falls verfügbar: Projektionen/Subset einstellen
        if hasattr(sart, "SetNumberOfProjectionsPerSubset"):
            n_per_subset = max(1, mu.shape[0] // 36)
            sart.SetNumberOfProjectionsPerSubset(int(n_per_subset))
            print(f"[INFO] proj/subset={n_per_subset}")
    elif hasattr(rtk, "SARTConeBeamReconstructionFilter"):
        sart = rtk.SARTConeBeamReconstructionFilter.New()
        print("[INFO] Using SART.")
    else:
        raise RuntimeError("Weder OS- noch SART-Filter im RTK-Python-Build gefunden.")

    sart.SetInput(0, vol_img)
    sart.SetInput(1, proj_img)
    sart.SetGeometry(geometry)
    if hasattr(sart, "SetNumberOfIterations"):
        sart.SetNumberOfIterations(int(args.iters))
    if args.relax is not None and hasattr(sart, "SetLambda"):
        sart.SetLambda(float(args.relax))
        print(f"[INFO] Relaxation (lambda) = {float(args.relax)}")
    # Displaced-Detector-Filter NICHT zwangsweise deaktivieren:
    if hasattr(sart, "SetDisableDisplacedDetectorFilter"):
        sart.SetDisableDisplacedDetectorFilter(False)

    print(f"[INFO] Iterative Rekonstruktion: iters={args.iters}")
    sart.Update()
    out_itk = sart.GetOutput()

    # optionale Glättung in mm
    if args.gauss_sigma_mm and args.gauss_sigma_mm > 0:
        out_itk = itk.smoothing_recursive_gaussian_image_filter(
            out_itk, sigma=float(args.gauss_sigma_mm)
        )
        print(f"[INFO] Gaussian smoothing σ={args.gauss_sigma_mm:.2f} mm")

    # ITK → NumPy → RAS speichern
    out_np = itk.array_from_image(out_itk).astype(np.float32)    # (z,y,x)
    vol_ras = out_np.transpose(2,1,0).copy()                     # (x,y,z)
    aff_ras = np.diag([dx,dy,dz,1.0]).astype(np.float32)

    out_ras_path = args.out_ras if args.out_ras else os.path.splitext(args.out)[0] + "_RAS.nii"
    nib.save(nib.Nifti1Image(vol_ras, aff_ras), out_ras_path)
    print(f"[OK] saved (canonical RAS): {out_ras_path}")

    # Preview
    if args.quick_ap_png:
        base = os.path.splitext(os.path.basename(out_ras_path))[0]
        os.makedirs(args.quick_ap_png, exist_ok=True)
        fname = f"{base}_{args.preview_view}" + (f"_{args.coronal_orient}" if args.preview_view=='coronal' else "")
        out_png = os.path.join(args.quick_ap_png, f"{fname}.png")
        save_preview_png(out_ras_path, out_png, view=args.preview_view, base_h_in=args.base_h_in, coronal_orient=args.coronal_orient)
        print(f"[OK] Preview saved: {out_png}")


if __name__ == "__main__":
    main()