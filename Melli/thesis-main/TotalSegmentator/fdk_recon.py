#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDK-Rekonstruktion mit ITK-RTK aus xCAT/ct_projector-Projektionen (mu_stack).

Wesentlich:
- Detektorbreite/-höhe werden (falls vorhanden) aus Half_fan_angle und SDD
  abgeleitet: width_det = 2*SDD*tan(half_fan).
- Optional: Volumen-FOV automatisch auf FOV in der Iso-Ebene setzen (--auto_iso_fov):
    width_iso  = width_det  * (DSO/SDD)
    height_iso = height_det * (DSO/SDD)

Beispiel:
python3 fdk_recon.py \
  --mat ct_proj_stack_400.mat \
  --proj proj_400.txt \
  --auto_iso_fov \
  --quick_ap_png preview/
"""

import argparse, re, os
import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- I/O-Helper ----------------
def parse_proj_txt(path):
    """proj.txt einlesen und Geometrie-Parameter extrahieren (robust bzgl. Keys)."""
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()

    def grab(keys):
        """Hilfsfunktion: akzeptiert mehrere mögliche Key-Schreibweisen (robust gegen Varianten)."""
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            m = re.search(rf'^\s*([+-]?\d+(?:\.\d+)?)\s*:\s*{re.escape(key)}', txt, re.M | re.I)
            if m:
                return float(m.group(1))
        return None

    geom = {
        # Falls width/height(mms) vorhanden, sind das die physikalischen Abmessungen des aktiven Panelbereichs
        "width_mm":  grab(["width(mms)", "width"]),
        "height_mm": grab(["height(mms)", "height"]),
        # Detektorpixel-Anzahl (Zeilen = v-Richtung, Spalten = u-Richtung)
        "nr": int(grab("num_rows") or 0),
        "nc": int(grab("num_channels") or 0),
        # DSO/DTD => SDD = DSO + DTD
        "DSO": grab(["distance_to_source(mms)", "distance_to_source"]),
        "DTD": grab(["distance_to_detector(mms)", "distance_to_detector"]),
        # u-Versatz des Detektors (Pixel oder mm; wird unten konsistent in mm gebracht)
        "det_shift_u": grab("detector_shift") or 0.0,
        # optionaler Half-Fan-Angle (in Grad); wenn vorhanden, kann ich damit width ableiten
        "half_fan_deg": grab(["Half_fan_angle", "half_fan_angle", "half_fan"]),
    }
    assert geom["DSO"] is not None and geom["DTD"] is not None, "DSO/DTD fehlen in proj.txt"
    return geom


def load_mu_stack_any(mat_path):
    """
    Lädt mu_stack aus .mat (v7/v7.3) und liefert (mu [nProj,nr,nc], angles or None).

    WICHTIG:
    - In v7.3 (HDF5) kommen die Daten oft als (nr, nc, nProj).
      Für RTK/ITK will ich aber (nProj, nr, nc) → also transponiere ich ggf.
    - nProj = Anzahl Projektionen (Winkel), nr = Detektor-Zeilen (v), nc = Detektor-Spalten (u).
    """
    mu, ang = None, None
    try:
        from scipy.io import loadmat
        M = loadmat(mat_path, squeeze_me=True, simplify_cells=True)  # .mat → Dict
        mu = np.asarray(M["mu_stack"], np.float32)
        if "angles_out" in M:
            ang = np.asarray(M["angles_out"], np.float32)
        elif "angles_deg" in M:
            ang = np.asarray(M["angles_deg"], np.float32)
    except Exception:
        import h5py
        with h5py.File(mat_path, "r") as f:
            mu = np.array(f["mu_stack"], dtype=np.float32)
            # HDF5 (v7.3) kommt häufig als (nr,nc,nProj) → für RTK brauche ich (nProj,nr,nc)
            mu = np.transpose(mu, (2, 1, 0))
            print(f"[FIX] mu_stack transposed → {mu.shape} (nProj, nr, nc)")
            if "angles_out" in f:
                ang = np.array(f["angles_out"], dtype=np.float32).squeeze()
            elif "angles_deg" in f:
                ang = np.array(f["angles_deg"], dtype=np.float32).squeeze()
    if mu.ndim != 3:
        raise ValueError(f"mu_stack hat ndims={mu.ndim}, erwarte 3.")
    return mu.astype(np.float32, copy=False), (None if ang is None else ang.astype(np.float32, copy=False))


# ---------------- Preview (RAS) ----------------
def save_preview_png(nifti_path, out_png, view="coronal", base_h_in=7.0, title=None, coronal_orient="AP"):
    """
    Preview direkt aus der gespeicherten NIfTI (in RAS).
    ACHSEN (RAS!):
      X = links↔rechts (lateral), Y = anterior↔posterior, Z = cranio↔caudal (longitudinal)
    Für die Ansichten:
      - coronal: Y fix, Ebene X–Z
      - axial:   Z fix, Ebene X–Y
      - sagittal: X fix, Ebene Y–Z
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    img = nib.load(nifti_path)
    img_ras = nib.as_closest_canonical(img)  # Sicherheit: in kanonisches RAS bringen (falls nicht schon so gespeichert)
    vol = np.asarray(img_ras.dataobj, dtype=np.float32)
    aff = img_ras.affine
    X, Y, Z = vol.shape
    # Achtung: in meiner RAS-Datei ist die Affine diagonal mit den Spacings (dx,dy,dz)
    dx, dy, dz = float(aff[0, 0]), float(aff[1, 1]), float(aff[2, 2])

    v = view.lower()
    if v == "coronal":
        # CORONAL: Y fix → nehme mittlere Y-Scheibe → erhalte ein X–Z Bild
        sl = vol[:, Y//2, :]            # (X, Z)
        # Transponieren, damit X horizontal läuft und Z vertikal, und flipup für Bildkoordinaten
        img2 = np.flipud(sl.T)
        # Achsenskalierung: X → [0, dx*X], Z → [0, dz*Z]
        ex = [0, dx * X, 0, dz * Z]
        xlabel, ylabel = "X [mm]", "Z [mm]"
        ttl = title or "Coronal"

    elif v == "axial":
        # AXIAL: Z fix → mittlere Z-Scheibe → X–Y Bild
        sl = vol[:, :, Z // 2]          # (X, Y)
        img2 = np.flipud(sl.T)
        ex = [0, dx * X, 0, dy * Y]
        xlabel, ylabel, ttl = "X [mm]", "Y [mm]", "Axial"

    elif v == "sagittal":
        # SAGITTAL: X fix → mittlere X-Scheibe → Y–Z Bild
        sl = vol[X // 2, :, :]          # (Y, Z)
        img2 = np.flipud(sl.T)
        ex = [0, dy * Y, 0, dz * Z]
        xlabel, ylabel, ttl = "Y [mm]", "Z [mm]", "Sagittal"
    else:
        raise ValueError("view must be coronal|axial|sagittal")

    # einfache Windowing-Normalisierung (robust)
    p2, p98 = np.percentile(img2, [2, 98])
    shown = np.clip((img2 - p2) / (p98 - p2 + 1e-6), 0, 1)

    # angenehmes Seitenverhältnis der Abbildung wählen
    width_mm, height_mm = ex[1] - ex[0], ex[3] - ex[2]
    aspect = height_mm / max(width_mm, 1e-6)
    h_in = base_h_in
    w_in = h_in / max(aspect, 1e-6)

    fig = plt.figure(figsize=(w_in, h_in), dpi=150)
    ax = plt.gca()
    ax.imshow(shown, cmap="gray", origin="lower", extent=ex, aspect="equal")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(ttl)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(fig)
    print(f"[INFO] Preview ({view}) gespeichert: {out_png}")


# ---------------- Hauptprogramm ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="ct_proj_stack.mat")
    ap.add_argument("--proj", default="proj.txt")
    ap.add_argument("--out", default="ct_recon_rtk.nii")
    ap.add_argument("--out_ras", default=None)
    # Preview
    ap.add_argument("--quick_ap_png", type=str, default=None)
    ap.add_argument("--preview_view", choices=["coronal", "axial", "sagittal"], default="coronal")
    ap.add_argument("--base_h_in", type=float, default=6.0)
    # Volumengitter (FOV in mm; kann mit --auto_iso_fov automatisch gesetzt werden)
    ap.add_argument("--nx", type=int, default=352)
    ap.add_argument("--ny", type=int, default=560)     # WICHTIG: RTK/ITK interpretiert diese Achse als "longitudinal"
    ap.add_argument("--nz", type=int, default=352)     # RTK-Z (Detektor v-Richtung) → wird später als Y_RAS enden
    ap.add_argument("--sx", type=float, default=333.3)
    ap.add_argument("--sy", type=float, default=533.3)
    ap.add_argument("--sz", type=float, default=333.3)
    ap.add_argument("--auto_iso_fov", action="store_true",
                    help="Setzt sx/sy/sz automatisch auf FOV in der Iso-Ebene (aus Detektor + DSO/SDD).")
    # Detektor-Override
    ap.add_argument("--detw", type=float, default=None, help="Detektorbreite [mm] override")
    ap.add_argument("--deth", type=float, default=None, help="Detektorhöhe [mm] override")
    # Geometrie- & Filter-Flags
    ap.add_argument("--angle_offset", type=float, default=270.0)  # meine Angles starten bei ~90° → +270° = +(-90°) Mod 360 → coronal/AP direkt richtig
    ap.add_argument("--reverse_angles", action="store_true")
    ap.add_argument("--flip_u", action="store_true")
    ap.add_argument("--shift_v", type=float, default=0.0, help="Detektor-Versatz v [mm] (vertikal)")
    ap.add_argument("--scale_du", type=float, default=1.0, help="Skalierung u-Pitch (Experimentierschraube)")
    ap.add_argument("--scale_dso", type=float, default=1.0, help="Skaliert DSO & DTD gemeinsam")
    ap.add_argument("--hann", type=float, default=0.7, help="Hann Cut Frequency [0..1] (0.7=weicher, 1.0=schärfer)")
    args = ap.parse_args()

    # --- Projektionen & Geometrie laden ---
    mu_raw, ang = load_mu_stack_any(args.mat)
    print(f"[INFO] mu_raw shape: {mu_raw.shape}")
    G = parse_proj_txt(args.proj)

    DSO = float(G["DSO"]) * args.scale_dso
    DTD = float(G["DTD"]) * args.scale_dso
    SDD = DSO + DTD

    nproj, nr, nc = mu_raw.shape  # (Anzahl Projektionen, Detektor-Zeilen v, Detektor-Spalten u)
    if G["nr"] and G["nc"] and (G["nr"] != nr or G["nc"] != nc):
        raise ValueError(f"Detektorgröße uneinheitlich: mu={nr}x{nc}, proj.txt={G['nr']}x{G['nc']}")

    # --- width/height bestimmen ---
    # Logik: falls Half-Fan vorhanden → Breite = 2*SDD*tan(half) (konsistent zur Geometrie).
    # Höhe: aus proj.txt, falls vorhanden; sonst proportional zur Pixelanzahl.
    width_mm = height_mm = None
    src = ""
    if args.detw is not None and args.deth is not None:
        width_mm, height_mm = float(args.detw), float(args.deth)
        src = "CLI overrides"
    elif G.get("half_fan_deg") is not None:
        half_fan = float(G["half_fan_deg"])
        width_mm = 2.0 * SDD * np.tan(np.deg2rad(half_fan))  # Detektorbreite aus Half-Fan (am Detektor!)
        height_mm = (float(G["height_mm"]) if G["height_mm"] else width_mm * (nr / nc))
        src = f"half_fan={half_fan}° (+height fallback)"
    else:
        width_mm  = float(args.detw) if args.detw else (float(G["width_mm"]) or args.sx)
        height_mm = float(args.deth) if args.deth else (float(G["height_mm"]) or args.sy)
        src = "proj.txt fallback"

    # Pixelpitches (mm/px) auf dem Detektor
    du = (width_mm / nc) * args.scale_du   # u-Pitch
    dv = (height_mm / nr)                  # v-Pitch

    # det_shift_u: Falls in Pixel angegeben (|val|<2), konvertiere in mm, sonst bleibt mm
    det_shift_u_raw = float(G["det_shift_u"])
    det_shift_u_mm = det_shift_u_raw * du if abs(det_shift_u_raw) < 2.0 else det_shift_u_raw

    print(f"[GEOM] Quelle: {src}")
    print(f"[GEOM] width_det={width_mm:.2f} mm, height_det={height_mm:.2f} mm | du={du:.4f} mm, dv={dv:.4f} mm")
    print(f"[GEOM] DSO={DSO:.1f} mm, DTD={DTD:.1f} mm, SDD={SDD:.1f} mm | det_shift_u(mm)={det_shift_u_mm:.3f}")

    # --- Iso-FOV (nur Info oder auto setzen) ---
    # Vergrößerung: M = SDD/DSO. Iso-FOV = Detektor-FOV / M.
    # WICHTIG: hier meine Konvention fürs spätere Volumen:
    #   sx (X_RAS) ← projizierte Detektorbreite (u)
    #   sy (Z_RAS) ← projizierte Detektorhöhe (v)  [longitudinal]
    #   sz (Y_RAS) ← projizierte Detektorbreite (u)  (für quadratischen Querschnitt X/Z; kann ich aber auch frei wählen)
    M = SDD / DSO
    width_iso  = width_mm  / M
    height_iso = height_mm / M
    print(f"[SUGGEST] Iso-FOV(mm): sx≈{width_iso:.1f}, sy≈{height_iso:.1f}, sz≈{width_iso:.1f}  ...")

    # Volumen-FOV holen/ggf. automatisch setzen
    sx, sy, sz = args.sx, args.sy, args.sz
    if args.auto_iso_fov:
        sx = width_iso          # X_RAS ~ u_iso
        sy = height_iso         # Z_RAS ~ v_iso (longitudinal, längste Achse)
        sz = width_iso          # Y_RAS ~ u_iso (in-plane passend zu X_RAS)
        # Hinweis: Ich kann sx/sz später größer wählen (gegen Trunkierung), unabhängig vom Iso-FOV

    # --- Winkel vorbereiten ---
    if ang is None or len(ang) != nproj:
        # Fallback: gleichmäßig 0..360°
        ang = np.linspace(0, 360, nproj, endpoint=False, dtype=np.float32)
        print("[WARN] angles_* fehlte – setze 0..360° gleichmäßig.")
    if args.reverse_angles:
        ang = ang[::-1]; print("[INFO] reverse_angles aktiv")
    if args.angle_offset != 0.0:
        # Offset modular auf 0..360 bringen
        ang = (ang + args.angle_offset) % 360.0
        print(f"[INFO] angle_offset = {args.angle_offset}°")
    print(f"[INFO] angles[0:5]={np.asarray(ang[:5]).round(1)} ... angles[-5:]={np.asarray(ang[-5:]).round(1)} (n={len(ang)})")

    # --- Plausibilitäts-Checks ---
    # Abschätzung der cone half-angles direkt aus Pitch, Pixelzahl und SDD
    cone_half_u = np.degrees(np.arctan(0.5 * du * nc / SDD))
    cone_half_v = np.degrees(np.arctan(0.5 * dv * nr / SDD))
    print(f"[CHECK] cone half-angles u/v ≈ {cone_half_u:.2f}° / {cone_half_v:.2f}°, magnification={M:.3f}")

    # --- Projektionen ggf. spiegeln ---
    mu = np.asarray(mu_raw, dtype=np.float32, order="C")
    if args.flip_u:
        mu = mu[:, :, ::-1].copy()
        det_shift_u_mm = -det_shift_u_mm  # Spiegelung → Vorzeichen vom u-Offset anpassen
        print("[INFO] u-axis flipped → det_shift_u sign inverted")

    # ---------------- ITK-RTK ----------------
    import itk
    from itk import RTK as rtk

    # RTK-Geometrie aufbauen: für jeden Winkel eine Projektion mit u/v-Offsets
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for a in ang:
        geometry.AddProjection(DSO, SDD, float(a), det_shift_u_mm, float(args.shift_v))

    # Achtung ITK-Array-Interpretation:
    # itk.GetImageFromArray(mu) interpretiert mu als (z,y,x).
    # mu liegt hier als (nProj, nr, nc) → wird zu (z= nProj, y= nr, x= nc) interpretiert.
    # Für Projektionen ist das okay (RTK erwartet so gestapelt).
    proj_img = itk.GetImageFromArray(mu)
    proj_img.SetSpacing((du, dv, 1.0))                         # (x=u-Pitch, y=v-Pitch, z=1 pro Projektion)
    proj_img.SetOrigin((-width_mm / 2.0, -height_mm / 2.0, 0.0))  # Detektormitte als (0,0)
    Mdir = itk.Matrix[itk.D, 3, 3](); Mdir.SetIdentity()
    proj_img.SetDirection(Mdir)

    # Zielvolumen
    # WICHTIG (mein großes Aha): ITK sieht NumPy-Arrays als (z,y,x),
    # d.h. vol_np.shape = (nz, ny, nx) → physikalisch x,y,z.
    nx, ny, nz = args.nx, args.ny, args.nz
    dx, dy, dz = sx / nx, sy / ny, sz / nz
    vol_np = np.zeros((nz, ny, nx), dtype=np.float32)  # (z_itk, y_itk, x_itk)
    vol_img = itk.GetImageFromArray(vol_np)
    vol_img.SetSpacing((dx, dy, dz))
    vol_img.SetOrigin((-sx / 2.0, -sy / 2.0, -sz / 2.0))
    vol_img.SetDirection(Mdir)

    print(f"[INFO] Vol: ({nx},{ny},{nz}) vox | FOV(mm)=({sx:.1f},{sy:.1f},{sz:.1f}) → voxel(mm)=({dx:.3f},{dy:.3f},{dz:.3f})")

    # FDK-Reko
    fdk = rtk.FDKConeBeamReconstructionFilter.New()
    if hasattr(fdk, "SetHannCutFrequency"):
        fdk.SetHannCutFrequency(float(args.hann))  # 0.7 = weicher (glatter), 1.0 = schärfer (rauschiger)
        print(f"[INFO] Hann apodization cut = {args.hann:.2f}")
    if hasattr(fdk, "SetTruncationCorrection"):
        # bei randnahem FOV hilfreich (kann leichte Helligkeits-/Streifenartefakte dämpfen)
        fdk.SetTruncationCorrection(True)

    fdk.SetInput(0, vol_img)
    fdk.SetInput(1, proj_img)
    fdk.SetGeometry(geometry)
    fdk.Update()

    # --- ITK → NumPy → RAS speichern ---
    out_np = itk.array_from_image(fdk.GetOutput()).astype(np.float32)  # (z_itk,y_itk,x_itk)
    # Jetzt in RAS-Achsen umordnen:
    #   X_RAS ← x_itk
    #   Y_RAS ← z_itk   (Detektor-v-Ebene)
    #   Z_RAS ← y_itk   (longitudinal)
    vol_ras = np.transpose(out_np, (2, 0, 1)).copy()                   # (x_ras,y_ras,z_ras) = (x_itk, z_itk, y_itk)
    # Affine entsprechend der RAS-Reihenfolge: diag([dx, dz, dy])
    aff_ras = np.diag([dx, dz, dy, 1.0]).astype(np.float32)

    out_ras_path = args.out_ras if args.out_ras else os.path.splitext(args.out)[0] + "_RAS.nii"
    nib.save(nib.Nifti1Image(vol_ras, aff_ras), out_ras_path)
    print(f"[OK] saved (canonical RAS): {out_ras_path}")

    # Preview optional erzeugen (nutzt die RAS-Datei, daher stimmen die Achsen/Lables)
    if args.quick_ap_png:
        base = os.path.splitext(os.path.basename(out_ras_path))[0]
        os.makedirs(args.quick_ap_png, exist_ok=True)
        fname = f"{base}_{args.preview_view}"
        out_png = os.path.join(args.quick_ap_png, f"{fname}.png")
        save_preview_png(
            out_ras_path,
            out_png,
            view=args.preview_view,
            base_h_in=args.base_h_in
        )
        print(f"[OK] Preview saved: {out_png}")

    # --- quick sanity auf dem gespeicherten Volumen ---
    nii = nib.load(out_ras_path)
    print("[SANITY] RAS shape:", nii.shape)
    print("[SANITY] spacing diag:", np.diag(nii.affine)[:3])
    try:
        print("[SANITY] axcodes:", nib.aff2axcodes(nii.affine))
    except Exception as e:
        print("[SANITY] axcodes failed:", e)


if __name__ == "__main__":
    main()