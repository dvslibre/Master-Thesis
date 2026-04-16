#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDK-Rekonstruktion mit ITK-RTK aus xCAT/ct_projector-Projektionen (mu_stack),
plus optionaler Preview (coronal/axial/sagittal) mit mm-Extents.
Headless (Agg). Preview ist strikt RAS (keine Heuristik).

Eingang:
  - ct_proj_stack.mat:
      mu_stack: [nProj, nr, nc]
      angles_out (optional; sonst 0..360° gleichmäßig)
  - proj.txt: width_mm, height_mm, num_rows (nr), num_channels (nc),
              distance_to_source (DSO), distance_to_detector (DTD), detector_shift (u)

NEU/Änderung:
  - Ausgabe ist jetzt grundsätzlich **kanonisch RAS**: Array=(X,Y,Z), Affine=diag([dx,dy,dz]).
    Sowohl --out als auch --out_ras sind identisch (falls beide angegeben).
    Preview wird immer aus der kanonischen Datei erzeugt.
"""

import argparse, re, os
import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- I/O-Helper ----------------
def parse_proj_txt(path):
    """proj.txt einlesen und Geometrie-Parameter extrahieren"""
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()                                             # Dateitext als string einlesen

    def grab(key):                                                                                              # innere Hilfsfunktion: sucht in txt nach Zeile, die wie <Zahl> : <key> aussieht
        m = re.search(rf'^\s*([+-]?\d+(?:\.\d+)?)\s*:\s*{re.escape(key)}', txt, re.M | re.I)
        return float(m.group(1)) if m else None

    geom = {                                                                                                    # Dictionary mit allen nötigen Geometrieparametern
        "width_mm":  grab("width(mms)"),            # Detektorbreite
        "height_mm": grab("height(mms)"),           # Detektorhöhe
        "nr": int(grab("num_rows") or 0),           # Zeilen HÖHE Detektor
        "nc": int(grab("num_channels") or 0),       # Kanäle BREITE Detektor
        "DSO": grab("distance_to_source(mms)"),
        "DTD": grab("distance_to_detector(mms)"),
        "det_shift_u": grab("detector_shift") or 0.0,
    }
    assert geom["DSO"] is not None and geom["DTD"] is not None, "DSO/DTD fehlen in proj.txt"
    return geom


def load_mu_stack_any(mat_path):
    """Lädt mu_stack aus .mat (v7/v7.3) und liefert (mu, angles)"""
    mu = None                                                                                                   # Initialisierung der Platzhalter
    ang = None
    try:
        from scipy.io import loadmat
        M = loadmat(mat_path, squeeze_me=True, simplify_cells=True)                                             # Matlab cell arrays werden in Python-Struktur umgewandelt
        mu = np.asarray(M["mu_stack"], np.float32)
        if "angles_out" in M:                                                                                   # Winkel einlesen --> NumPy-Array
            ang = np.asarray(M["angles_out"], np.float32)
        elif "angles_deg" in M:
            ang = np.asarray(M["angles_deg"], np.float32)
    except Exception:
        import h5py
        with h5py.File(mat_path, "r") as f:
            mu = np.array(f["mu_stack"], dtype=np.float32)
            mu = np.transpose(mu, (2, 1, 0))  # (310,512,360) → (360,512,310)
            print(f"[FIX] mu_stack transposed → {mu.shape} (nProj, nr, nc)")
            if "angles_out" in f:
                ang = np.array(f["angles_out"], dtype=np.float32).squeeze()
            elif "angles_deg" in f:
                ang = np.array(f["angles_deg"], dtype=np.float32).squeeze()

    if mu.ndim != 3:                                                                                            # Formvalidierung (muss dreidimensional sein)
        raise ValueError(f"mu_stack hat ndims={mu.ndim}, erwarte 3.")
    return mu.astype(np.float32, copy=False), (None if ang is None else ang.astype(np.float32, copy=False))


# ---------------- Preview (deterministisch in RAS, Mapping gemäß deiner Sichtprüfung) ----------------
def save_preview_png(nifti_path, out_png, view="coronal",
                     base_h_in=7.0, title=None, coronal_orient="AP"):
    """
    Erwartet: RAS-kanonisches NIfTI (Array=(X,Y,Z), Affine diag(dx,dy,dz)).
    Mapping gemäß deiner visuellen Diagnose:
      - coronal  = fixiere X → Ebene (Y,Z)
      - axial    = fixiere Y → Ebene (X,Z)
      - sagittal = fixiere Z → Ebene (X,Y)
    """

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    # NIfTI laden (sollte bereits RAS sein)
    img = nib.load(nifti_path)
    img_ras = nib.as_closest_canonical(img)                                                                     # bringt Bild in RAS-Layout, falls es noch nciht passt
    vol = np.asarray(img_ras.dataobj, dtype=np.float32)                                                         # vol = NumPy Array mit Shape (X,Y,Z)
    aff = img_ras.affine                                                                                        # diag(dx,dy,dz)
    X, Y, Z = vol.shape
    dx, dy, dz = float(aff[0, 0]), float(aff[1, 1]), float(aff[2, 2])

    v = view.lower()
    if v == "coronal":
        # Coronal: X mittig fixieren → Ebene (Y,Z)
        sl = vol[X // 2, :, :]

        # Standarddarstellung (PA): Blickrichtung von posterior → anterior
        img2 = np.flipud(sl)  # Kopf oben behalten, Z horizontal, Y vertikal

        # Für AP-Ansicht: horizontal spiegeln (Y-Achse = anterior↔posterior)
        if str(coronal_orient).upper() == "AP":
            img2 = np.fliplr(img2)

        # Extents: Z horizontal, Y vertikal
        ex = [0, dz * Z, 0, dy * Y]
        xlabel, ylabel = "Z [mm]", "Y [mm]"
        ttl = title or f"Coronal ({str(coronal_orient).upper()})"

    elif v == "axial":
        # AXIAL: Y fixieren → (X,Z)
        sl = vol[:, Y // 2, :]                             # (X,Z)
        img2 = np.flipud(sl.T)
        ex = [0, dx * X, 0, dz * Z]
        xlabel, ylabel = "X [mm]", "Z [mm]"
        ttl = title or "Axial"

    elif v == "sagittal":
        # SAGITTAL: Z fixieren → (X,Y)
        sl = vol[:, :, Z // 2]                             # (X,Y)
        img2 = np.flipud(sl.T)
        ex = [0, dx * X, 0, dy * Y]
        xlabel, ylabel = "X [mm]", "Y [mm]"
        ttl = title or "Sagittal"

    else:
        raise ValueError("view must be coronal|axial|sagittal")

    # Fensterung (2..98 %) + normiert auf [0,1]
    p2, p98 = np.percentile(img2, [2, 98])
    shown = np.clip((img2 - p2) / (p98 - p2 + 1e-6), 0, 1)

    # Bildgröße so wählen, dass das mm-Verhältnis bewahrt bleibt
    width_mm, height_mm = ex[1] - ex[0], ex[3] - ex[2]
    aspect = height_mm / max(width_mm, 1e-6)
    h_in = float(base_h_in)
    w_in = h_in / max(aspect, 1e-6)

    fig = plt.figure(figsize=(w_in, h_in), dpi=150)
    ax = plt.gca()
    ax.imshow(shown, cmap="gray", origin="lower", extent=ex, aspect="equal")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(ttl)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[INFO] Preview ({view}) gespeichert: {out_png} | extent(mm)={width_mm:.1f}×{height_mm:.1f} | fig≈{w_in:.2f}×{h_in:.2f} in")


# ---------------- Hauptprogramm ----------------
def main():
    # ---------- Argumente/CLI ----------
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="ct_proj_stack.mat")
    ap.add_argument("--proj", default="proj.txt")
    ap.add_argument("--out", default="ct_recon_rtk.nii")
    ap.add_argument("--out_ras", default=None,
                    help="optional: zusätzliches kanonisches NIfTI (X,Y,Z + Affine diag(dx,dy,dz))")
    ap.add_argument("--coronal_orient", choices=["AP","PA"], default="AP",
                help="Nur für coronal: AP=Anterior→Posterior, PA=Posterior→Anterior")
    # ---- Standardraster: isotrop 1.5625 mm ----
    ap.add_argument("--nx", type=int, default=320)      # X (links-rechts)
    ap.add_argument("--ny", type=int, default=512)      # Y (kraniokaudal)
    ap.add_argument("--nz", type=int, default=320)      # Z (vorn-hinten)
    ap.add_argument("--sx", type=float, default=500.0)
    ap.add_argument("--sy", type=float, default=800.0)
    ap.add_argument("--sz", type=float, default=500.0)
    # Optionale Überschreibung der Detektorgröße (falls proj.txt leer oder unvollständig)
    ap.add_argument("--detw", type=float, default=None)
    ap.add_argument("--deth", type=float, default=None)
    # Steuerflags
    ap.add_argument("--reverse_angles", action="store_true")
    ap.add_argument("--angle_offset", type=float, default=0.0)
    # Preview Optionen
    ap.add_argument("--quick_ap_png", type=str, default=None)
    ap.add_argument("--preview_view", choices=["coronal", "axial", "sagittal"], default="coronal")
    ap.add_argument("--base_h_in", type=float, default=6.0)
    args = ap.parse_args()

    # ----------- Projektionen & Geometrie -----------
    mu_raw, ang = load_mu_stack_any(args.mat)
    print(f"[INFO] mu_raw shape: {mu_raw.shape}")

    # --- Geometrie lesen ---
    G = parse_proj_txt(args.proj)
    DSO = float(G["DSO"])                             # Source --> Isocenter Abstand
    DTD = float(G["DTD"])                             # Isocenter --> Detektor Abstand
    SDD = DSO + DTD                                   # Source --> Detektor Abstand
    # Detektorgröße in mm
    width_mm  = float(args.detw) if args.detw else (float(G["width_mm"])  if G["width_mm"]  else args.sx)
    height_mm = float(args.deth) if args.deth else (float(G["height_mm"]) if G["height_mm"] else args.sy)
    nr_txt = int(G["nr"] or 0)
    nc_txt = int(G["nc"] or 0)
    det_shift_u = float(G["det_shift_u"])

    # --- Zielform ausgeben ---
    nproj, nr, nc = mu_raw.shape
    print(f"[INFO] Projektionen: nProj={nproj}, Detektor={nr}×{nc}")


    # ---------- Winkel ----------
    if ang is None or len(ang) != nproj:
        ang = np.linspace(0, 360, nproj, endpoint=False, dtype=np.float32)
        print("[WARN] angles_* fehlte – setze 0..360° gleichmäßig.")
    if args.reverse_angles:
        ang = ang[::-1]; print("[INFO] reverse_angles aktiv")
    if args.angle_offset != 0.0:
        ang = (ang + args.angle_offset) % 360.0; print(f"[INFO] angle_offset = {args.angle_offset}°")
    print(f"[INFO] angles[0:5]={np.asarray(ang[:5]).round(1)} ... angles[-5:]={np.asarray(ang[-5:]).round(1)} (n={len(ang)})")


    # ---------- mu nach [nProj, nr, nc] bringen (deterministisch & schlank) ----------
    # Erwartet: mu_raw.shape == (nProj, nr, nc)
    assert mu_raw.ndim == 3, "mu_stack muss 3D sein (erwarte (nProj, nr, nc))."
    nproj, nr, nc = mu_raw.shape

    # Optional: gegen proj.txt validieren (falls dort nr/nc stehen)
    if nr_txt and nc_txt:
        assert (nr, nc) == (nr_txt, nc_txt), (
            f"Detektor-Maße unerwartet: mu={nr}×{nc}, proj.txt={nr_txt}×{nc_txt}"
        )

    # In float32 konvertieren und für C-kontiguen Speicher sorgen (robust für ITK/RTK)
    mu = np.asarray(mu_raw, dtype=np.float32, order="C")

    # Detektor-Pixelgrößen (mm/Pixel) in u (Breite, nc) und v (Höhe, nr)
    du = width_mm / nc
    dv = height_mm / nr
    print(f"[INFO] du={du:.4f} mm, dv={dv:.4f} mm, DSO={DSO} SDD={SDD} (u-shift={det_shift_u})")


    # ---------------- ITK-RTK ----------------
    import itk
    from itk import RTK as rtk                                                                          # RTK liefert Cone-Beam-Geometrie und FDK-Filter

    # RTK-Geometrie
    geometry = rtk.ThreeDCircularProjectionGeometry.New()                                               # erzeugt leeres Geometrie-Objekt
    for a in ang:
        geometry.AddProjection(DSO, SDD, float(a), det_shift_u, 0.0)                                    # fügt für jeden Winkel a eine Projektion zur Geometrie hinzu (geometry weiß dann für jeden Index, wie Quelle/Detektor standen)

    # Projektionen (Stack) als ITK-Bild (Spacing = (du,dv,1), Origin so, dass Detektorzentrum bei (0,0) liegt) --> (z,y,x) = (nProj,nr,nc) im ITK-Sinn
    proj_img = itk.GetImageFromArray(mu)                                                                # umwandlung von mu als Zahlenarray in ITK-Struktur mit Metadaten
    proj_img.SetSpacing((du, dv, 1.0))                                                                  # setzt physikalisches Pixelmaß in mm
    proj_img.SetOrigin((-width_mm / 2.0, -height_mm / 2.0, 0.0))
    M = itk.Matrix[itk.D, 3, 3](); M.SetIdentity()                                                      # erstellt 3x3-Double-Matrix in ITK --> beschreibt, wie die Bildachsen im Raum orientiert sind
    proj_img.SetDirection(M)                                                                            # weist dem ITK-Image die Orientierungsmatrix zu

    # Zielvolumen (ITK braucht (z,y,x)-Reihenfolge)
    nx, ny, nz = args.nx, args.ny, args.nz                                                              # Anzahl der Voxel in x, y, z Richtung
    sx, sy, sz_mm = args.sx, args.sy, args.sz                                                           # Field of View in mm in jede Richtung
    dx, dy, dz = sx / nx, sy / ny, sz_mm / nz                                                           # Voxelausdehnung in mm (--> Spacing des Zielvolumens)
    vol_np = np.zeros((nz, ny, nx), dtype=np.float32)                                                   # erstellt leeres NumPy Array mit Nullen für das Volumen (z,y,x)
    vol_img = itk.GetImageFromArray(vol_np)                                                             # Umwandlung NumPy-Array in ITK-Image, damit RTK rechnen kann
    vol_img.SetSpacing((dx, dy, dz))                                                                    # setzt physikalisches Voxel-Spacing in mm entlang der Achsen
    vol_img.SetOrigin((-sx / 2.0, -sy / 2.0, -sz_mm / 2.0))
    vol_img.SetDirection(M)

    print(f"[INFO] Vol: nVoxel=({nx},{ny},{nz}), FOV(mm)=({sx},{sy},{sz_mm}) → voxel(mm)=({dx:.4f},{dy:.4f},{dz:.4f})")

    # FDK Rekonstruktion (RTK)
    fdk = rtk.FDKConeBeamReconstructionFilter.New()                                                     # erstellt FDK-Filter, um aus Projektionen das 3D-CT zu berechnen
    # Sanfte Rampenfensterung (typisch 0.6–0.9)
    if hasattr(fdk, "SetHannCutFrequency"):
        fdk.SetHannCutFrequency(0.7)
        print("[INFO] Hann apodization: cut freq = 0.70")
    # Trunkierungs-Korrektur (falls vorhanden; hilft bei abgeschnittenem FOV)
    if hasattr(fdk, "SetTruncationCorrection"):
        fdk.SetTruncationCorrection(True)
    fdk.SetInput(0, vol_img)                                                                            # die RTK schreibt die Rekonstruktion in das vorbereitete vol_img
    fdk.SetInput(1, proj_img)                                                                           # übergibt die 2D-Projektionsbilder (proj_img)
    fdk.SetGeometry(geometry)                                                                           # übergibt die Scanner-Geometrie
    fdk.Update()                                                                                        # startet die Rekonstruktion (intern: Rampenfilterung der Projektionen --> Rückprojektion entlang definierter Geometrie --> Summierung ins Volumen)

    out_img = fdk.GetOutput()                                                                           # extrahiert ITK-Image (Ergebnis)
    out_np = itk.array_from_image(out_img).astype(np.float32)                                           # Umwandlung des ITK-Ergebnisses in NumPy-Array mit (z,y,x)    
    print(f"[DEBUG] out_np shape after RTK: {out_np.shape}")

    # --------- KANONISIEREN & SPEICHERN (einheitlich RAS: (X,Y,Z)) ---------
    # ITK → NumPy: (z,y,x)  → wir wollen (x,y,z)
    vol_ras = out_np.transpose(2, 1, 0).copy()  # (z,y,x) → (x,y,z)
    print(f"[FIX] vol_ras shape (x,y,z) = {vol_ras.shape}")
    aff_ras = np.diag([dx, dy, dz, 1.0]).astype(np.float32)

    # Zielpfade
    out_ras_path = args.out_ras if args.out_ras else os.path.splitext(args.out)[0] + "_RAS.nii"

     # --------- SPEICHERN (nur kanonische RAS-Version) ---------
    nib.save(nib.Nifti1Image(vol_ras, aff_ras), out_ras_path)
    print(f"[OK] saved (canonical RAS): {out_ras_path}   voxel(mm)=({dx:.4f},{dy:.4f},{dz:.4f})")

    # Optional: Preview-PNG speichern
    if args.quick_ap_png:
        target_for_preview = out_ras_path  # immer die kanonische Datei
        out_path = args.quick_ap_png

        # Basename der NIfTI-Datei (ohne Extension)
        base = os.path.splitext(os.path.basename(out_ras_path))[0]

        # Falls nur ein Verzeichnis angegeben wurde (oder ein Pfad mit / am Ende)
        if out_path.endswith(os.sep) or os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)
            # Dateiname je nach Ansicht
            if args.preview_view == "coronal":
                fname = f"{base}_coronal_{args.coronal_orient}.png"
            else:
                fname = f"{base}_{args.preview_view}.png"
            out_png = os.path.join(out_path, fname)
        else:
            # Wenn eine Datei angegeben wurde → nur Endung & Verzeichnis verwenden
            d, b = os.path.split(out_path)
            os.makedirs(d or ".", exist_ok=True)
            # Einheitliche Benennung basierend auf NIfTI-Namen
            if args.preview_view == "coronal":
                out_png = os.path.join(d or ".", f"{base}_coronal_{args.coronal_orient}.png")
            else:
                out_png = os.path.join(d or ".", f"{base}_{args.preview_view}.png")

        # Rendern
        save_preview_png(
            target_for_preview,
            out_png,
            view=args.preview_view,
            base_h_in=args.base_h_in,
            coronal_orient=args.coronal_orient,
        )
        print(f"[OK] Preview saved: {out_png}")


if __name__ == "__main__":
    main()