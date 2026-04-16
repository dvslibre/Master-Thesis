#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ziel des Skripts:
- Binärdatei (*.par_atn_1.bin) mit µ-Werten (linearen Schwächungskoeffizienten in 1/mm)
  aus einer XCAT-Simulation
- Daraus --> 3D-Bild im NIfTI-Format
- Zusätzlich: µ-Werte in HU (Hounsfield Units) umrechnen
- Optional: 
    * Resampling (auf isotrope Voxelgröße, z.B. 1x1x1 mm) mit SimpleITK
    * Automatische Organ-/Struktur-Segmentierung via TotalSegmentator
    * Ein Vorschaubild (Overlay): HU-Graubild + farbige Segment-Konturen
    * HU-Statistiken global und (falls Segmentierung vorhanden) pro Label/Organ.
"""

# --- Standard-Python-Bibliotheken und externe Pakete importieren ---
import argparse      # liest Parameter aus der Kommandozeile (z.B. --in-bin ...)
import os            # für Pfade/Funktionen rund um Dateien/Ordner
import json          # um Ergebnisse/Statistiken als .json zu speichern
import subprocess    # um externe Programme (TotalSegmentator) aufzurufen

import numpy as np   # Numerik: Arrays, Mathefunktionen
import nibabel as nib # NIfTI lesen/schreiben

# SimpleITK (für Resampling) ist optional. Falls nicht installiert: sitk = None
try:
    import SimpleITK as sitk
except Exception:
    sitk = None


# ======== I/O & Geometry ========
def load_bin_with_order(path, shape_xyz, bin_order="XYZ", array_order="C", dtype="<f4"):
    """
    Lädt eine rohe Binärdatei mit Fließkommawerten (µ) als 3D-Volumen.

    Parameter:
    - path: Pfad zur .bin-Datei
    - shape_xyz: (nx, ny, nz) = Anzahl Voxel in X-, Y-, Z-Richtung
    - bin_order: In welcher Reihenfolge liegen die Achsen in der Datei? (z.B. "XYZ", "ZXY", ...)
                 -> wichtig, damit man später in die Zielreihenfolge (Z, Y, X) umsortieren kann
    - array_order: "C" (row-major, wie NumPy standardmäßig) oder "F" (Fortran-Order)
    - dtype: Datentyp in der Datei, hier little-endian float32 ("<f4")

    Rückgabe:
    - 3D-Array als NumPy-Array in der **einheitlichen** Reihenfolge (Z, Y, X)
    """
    nx, ny, nz = shape_xyz
    # Wir lesen genau nx*ny*nz Werte aus der Datei ein.
    arr = np.fromfile(path, dtype=dtype, count=nx * ny * nz)
    if arr.size != nx * ny * nz:
        # Falls weniger/mehr Werte als erwartet vorhanden sind -> Fehler.
        raise ValueError(f"Binärdatei hat {arr.size} Werte, erwartet {nx*ny*nz}")

    # Sicherstellen, dass bin_order eine Permutation von X,Y,Z ist (also z.B. XYZ, XZY, YXZ, ...)
    order = bin_order.upper()
    if set(order) != set("XYZ"):
        raise ValueError("bin_order muss Permutation von XYZ sein")

    # Wir bauen die Form in der Reihenfolge, wie die Datei organisiert ist.
    # Beispiel: bin_order="ZXY" -> shape_in = (nz, nx, ny)
    shape_in = tuple({"X": nx, "Y": ny, "Z": nz}[c] for c in order)

    # Das 1D-Array wird als 3D-Volumen interpretiert (mit vorgegebenem Speicherlayout).
    vol_in = arr.reshape(shape_in, order=array_order.upper())

    # Wir wollen am Ende immer (Z, Y, X) zurückgeben.
    # Dazu bestimmen wir, an welcher Position in 'vol_in' die Achsen Z, Y, X sitzen...
    idx = {c: i for i, c in enumerate(order)}
    # ...und transponieren in die Zielreihenfolge.
    axes = [idx["Z"], idx["Y"], idx["X"]]
    return np.transpose(vol_in, axes=axes)


def save_nifti(vol, spacing, out_path):
    """
    Speichert ein 3D-Array als NIfTI (.nii.gz).

    Parameter:
    - vol: 3D-Array (Z, Y, X)
    - spacing: (dz, dy, dx) in Millimetern (Abstand zwischen Voxelkanten)
    - out_path: Zielpfad
    """
    dz, dy, dx = spacing
    affine = np.array([[dx, 0, 0, 0],
                       [0, dy, 0, 0],
                       [0, 0, dz, 0],
                       [0, 0, 0, 1]], dtype=np.float32)
    nib.save(nib.Nifti1Image(vol.astype(np.float32), affine), out_path)
    return out_path


def resample_isotropic(in_nii, out_nii, target_mm=1.0):
    """
    Erstellt eine zweite NIfTI-Datei, bei der die Voxelwürfel isotrop sind
    (z.B. 1 x 1 x 1 mm), damit spätere Analysen gleichmäßiger arbeiten.

    - Wenn SimpleITK nicht installiert ist, wird das Resampling übersprungen (Warnung).
    """
    if sitk is None:
        print("[WARN] SimpleITK fehlt – kein Resampling.")
        return in_nii

    # Bild laden
    img = sitk.ReadImage(in_nii)

    # Aktuelle Voxelabstände und Bildgröße herausfinden
    spacing = img.GetSpacing()  # (dx, dy, dz) in SimpleITK-Konvention
    size = img.GetSize()        # (nx, ny, nz)

    # Ziel-Abstände definieren (alle Achsen = target_mm)
    new_spacing = (float(target_mm),) * 3

    # Neue Bildgröße so berechnen, dass das physische Volumen gleich bleibt:
    # neue Größe ~ alte Größe * (alte Abstände / neue Abstände)
    new_size = [int(round(osz * ospc / nspc))
                for osz, ospc, nspc in zip(size, spacing, new_spacing)]

    # Resampling mit linearer Interpolation (gut für Grauwerte)
    img2 = sitk.Resample(
        img, new_size, sitk.Transform(),
        sitk.sitkLinear, img.GetOrigin(),
        new_spacing, img.GetDirection(), 0.0, img.GetPixelID()
    )

    # Schreiben und Pfad zurückgeben
    sitk.WriteImage(img2, out_nii)
    return out_nii


# ======== Labelmap-Loader ========
def _label_map_from_seg(seg_path: str | None, user_path: str | None = None) -> dict[int, str]:
    """
    Sucht eine Zuordnungstabelle „Label-ID -> Name“ für Segmentierungen.

    Suchreihenfolge:
    1) Falls der Nutzer explizit eine Datei angegeben hat (--ts-labels), nimm diese.
       Akzeptiert JSON oder CSV (Spalten z.B. id,name).
    2) Im gleichen Ordner wie die Segmentierung „labels.json“ oder „label_map.csv“.
    3) Versuche, in Paketdaten von TotalSegmentator (totalseg/TotalSegmentator) passende JSONs zu finden.
    4) Falls alles scheitert: keine Rückgabe
    """
    import csv, importlib.util
    from pathlib import Path

    def try_json(p: Path):
        # Prüfe: gibt es die Datei? Wenn ja, lade JSON.
        if not p.is_file(): return None
        d = json.load(open(p, "r", encoding="utf-8"))
        # Variante A: {"labels": [{"id": 1, "name": "Spleen"}, ...]}
        if isinstance(d, dict) and "labels" in d and isinstance(d["labels"], list):
            return {int(x["id"]): str(x["name"]) for x in d["labels"] if "id" in x}
        # Variante B: {"1": "Spleen", "2": "Right kidney", ...}
        if isinstance(d, dict):
            out = {}
            for k, v in d.items():
                try:
                    out[int(k)] = str(v)
                except:
                    pass
            return out or None

    def try_csv(p: Path):
        # CSV mit Kopfzeile (z.B. id,name)
        if not p.is_file(): return None
        out = {}
        with open(p, newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                k = row.get("id") or row.get("label")
                n = row.get("name") or row.get("Label")
                if k and n:
                    try:
                        out[int(k)] = str(n)
                    except:
                        pass
        return out or None

    # 1) Nutzerpfad explizit
    if user_path:
        p = Path(user_path)
        m = try_json(p) or try_csv(p)
        if m: return m

    # 2) Neben der Segmentierung suchen
    if seg_path:
        sp = Path(seg_path)
        d = sp.parent if sp.is_file() else sp
        for c in ["labels.json", "label_map.csv"]:
            m = try_json(d / c) or try_csv(d / c)
            if m: return m

    # 3) Paketdaten durchsuchen
    try:
        import importlib.resources as r
        for pkg in ("totalseg", "TotalSegmentator"):
            if importlib.util.find_spec(pkg):
                data_dir = r.files(pkg).joinpath("data")
                for c in data_dir.glob("*.json"):
                    m = try_json(c)
                    if m: return m
    except Exception:
        pass

    # 4) Fallback
    return ""


def _color_map_for_labels(labels: np.ndarray):
    """
    Weist einzelnen Label-IDs (1,2,3,...) gut unterscheidbare Farben zu,
    damit die Legende/Overlays schön bunt und lesbar sind.
    """
    import matplotlib.pyplot as plt
    # Wir kombinieren drei Paletten, damit wir viele verschiedene Farben bekommen.
    palette = list(plt.cm.get_cmap("tab20").colors) \
           + list(plt.cm.get_cmap("tab20b").colors) \
           + list(plt.cm.get_cmap("tab20c").colors)
    # Nur positive Labels (0 = Hintergrund) und sortiert
    uniq_sorted = sorted({int(l) for l in labels if int(l) > 0})
    # Ordne jeder ID eine Farbe zu (bei vielen Labels „rotieren“ wir durch die Palette)
    return {lab: palette[i % len(palette)] for i, lab in enumerate(uniq_sorted)}


# ======== Overlay + Legende ========
def make_preview_slice(
    nii_path, png_path, seg_path=None, plane="coronal",
    rotate_k=0, flip_lr=False, flip_ud=False,
    add_legend=False, legend_max=15, legend_outside=False,
    vmin=None, vmax=None, add_colorbar=False,
    fill_alpha=0.35, label_map=None, export_labels_path=None
):
    """
    Erstellt eine 2D-Ansicht (einen Schnitt) aus dem 3D-HU-Volumen als PNG:
    - Graustufenbild = HU
    - Optional: farbige Überlagerung der Segmentierung + Legende + Farbleiste

    Wichtige Begriffe:
    - 'plane' bestimmt die Schnittebene:
        * coronal, sagittal, axial
    - rotate_k/flips: Bild drehen/spiegeln (nur für die Vorschau, nicht die Daten!)
    - vmin/vmax: Wertebereich für die Graustufen (Standard: automatische 0.5/99.5-Perzentile)
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.lines import Line2D
    from skimage.measure import find_contours

    # Hilfsfunktion: nimmt aus einem 3D-Volumen die mittlere Scheibe entlang der gewünschten Ebene
    def extract(vol, plane):
        if plane == "coronal":   # Mitte entlang Y -> Ergebnis ist (Z,X)
            return vol[:, vol.shape[1] // 2, :].T
        if plane == "sagittal":  # Mitte entlang X -> Ergebnis ist (Z,Y)
            return vol[:, :, vol.shape[2] // 2].T
        if plane == "axial":     # Mitte entlang Z -> Ergebnis ist (Y,X)
            return vol[vol.shape[0] // 2, :, :]
        raise ValueError("plane must be coronal/sagittal/axial")

    # Hilfsfunktion: dreht/spiegelt ein 2D-Bild abhängig von den Schaltern
    def xf(a):
        if rotate_k:
            a = np.rot90(a, int(rotate_k))
        if flip_lr:
            a = np.fliplr(a)
        if flip_ud:
            a = np.flipud(a)
        return a

    # HU-Volumen laden und mittleren Schnitt entnehmen
    vol = np.asarray(nib.load(nii_path).dataobj)
    sl  = xf(extract(vol, plane))

    # Falls vmin/vmax nicht vorgegeben sind: automatisch ausrobusten Perzentilen wählen
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(sl, [0.5, 99.5])

    # ---------- Layout der Abbildung vorbereiten ----------
    want_leg  = bool(add_legend and legend_outside)
    want_cbar = bool(add_colorbar)

    IMG_W  = 1.00  # großer Bildbereich
    CBAR_W = 0.04  # schmale Farbleiste
    GAP_W  = 0.03  # kleiner Abstand
    LEG_W  = 0.40  # Bereich für Legende
    WSPACE = 0.01  # Luft zwischen den Bereichen

    # Je nach Wunsch bauen wir (Bild+Colorbar+Legende) oder nur Teile davon auf.
    if want_leg and want_cbar:
        fig = plt.figure(figsize=(8.3, 6.0))
        gs = gridspec.GridSpec(1, 4, width_ratios=[IMG_W, CBAR_W, GAP_W, LEG_W], wspace=WSPACE)
        ax_img  = fig.add_subplot(gs[0, 0])
        ax_cbar = fig.add_subplot(gs[0, 1])
        ax_gap  = fig.add_subplot(gs[0, 2]); ax_gap.axis("off")
        ax_leg  = fig.add_subplot(gs[0, 3]); ax_leg.axis("off")
    elif want_cbar and not want_leg:
        fig = plt.figure(figsize=(6.8, 6.0))
        gs = gridspec.GridSpec(1, 2, width_ratios=[IMG_W, CBAR_W], wspace=0.02)
        ax_img  = fig.add_subplot(gs[0, 0])
        ax_cbar = fig.add_subplot(gs[0, 1])
        ax_leg  = None
    elif want_leg and not want_cbar:
        fig = plt.figure(figsize=(8.0, 6.0))
        gs = gridspec.GridSpec(1, 2, width_ratios=[IMG_W, LEG_W], wspace=0.02)
        ax_img  = fig.add_subplot(gs[0, 0])
        ax_cbar = None
        ax_leg  = fig.add_subplot(gs[0, 1]); ax_leg.axis("off")
    else:
        fig, ax_img = plt.subplots(figsize=(6.2, 6.0))
        ax_cbar = None
        ax_leg  = None

    # Graubild zeichnen (HU-Werte als Graustufen)
    im = ax_img.imshow(sl, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    ax_img.axis("off")

    # ---------- Optional: Segmente überlagern ----------
    legend_items = []  # für die Legende (Name, Farbe)
    if seg_path and os.path.isfile(seg_path):
        seg   = np.asarray(nib.load(seg_path).dataobj)  # 3D Labels (0=Hintergrund, >0=Struktur)
        seg2d = xf(extract(seg, plane))                 # passender 2D-Schnitt der Labels
        labels_present = np.unique(seg2d[seg2d > 0])    # alle vorhandenen Labels > 0

        # Farbkarte erzeugen (jede ID bekommt eine Farbe)
        colmap = _color_map_for_labels(labels_present)
        # Flächeninhalt je Label im 2D-Schnitt (für sinnvolle Zeichenreihenfolge, große zuerst)
        areas = {int(l): int((seg2d == l).sum()) for l in labels_present}
        draw_order = sorted(labels_present, key=lambda l: -areas[int(l)])[:legend_max]

        # Für jedes zu zeichnende Label:
        for lab in map(int, draw_order):
            mask = (seg2d == lab)   # Maske: wo dieses Label im 2D-Bild liegt
            if not np.any(mask):
                continue
            color = colmap[lab]

            # Halbtransparente Flächenfüllung
            ax_img.imshow(
                np.ma.masked_where(~mask, mask),
                cmap=plt.cm.colors.ListedColormap([color]),
                origin="lower", alpha=fill_alpha, interpolation="nearest"
            )

            # Konturen finden und als Linie zeichnen (macht die Grenzen scharf sichtbar)
            try:
                for cnt in find_contours(mask.astype(np.uint8), 0.5):
                    ax_img.plot(cnt[:, 1], cnt[:, 0], linewidth=1.2, color=color)
            except Exception:
                pass

            # Namen für die Legende nachschlagen (oder „Label 123“ falls unbekannt)
            name = (label_map or {}).get(lab, f"Label {lab}")
            legend_items.append((name, color))

    # ---------- Optional: Farbleiste (zeigt, welcher Grauwert wieviel HU ist) ----------
    if ax_cbar is not None:
        cb = fig.colorbar(im, cax=ax_cbar)
        cb.set_label("HU")

    # ---------- Optional: Legende mit Farbbalken + Namen der Labels ----------
    if ax_leg is not None and legend_items:
        from matplotlib.lines import Line2D
        y = 0.98; dy = 0.075
        for name, col in legend_items:
            ax_leg.add_line(Line2D([0.05, 0.22], [y, y], transform=ax_leg.transAxes, color=col, lw=5))
            ax_leg.text(0.26, y, name, transform=ax_leg.transAxes, va="center", fontsize=8)
            y -= dy
            if y < 0.06:
                break
    elif (not legend_outside) and add_legend and legend_items:
        # Alternative: Legende direkt über das Bild legen (oben rechts)
        handles = [Line2D([0], [0], color=c, lw=2, label=n) for n, c in legend_items]
        ax_img.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.6)

    # Bild speichern und schließen (Speicher aufräumen)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ======== TotalSegmentator-Wrapper ========
def run_totalseg(in_hu, out_seg, fast=False, more_args=None):
    """
    Startet TotalSegmentator im ML-Modus, um Organe/Strukturen
    im HU-Volumen automatisch zu erkennen.

    Parameter:
    - in_hu: Pfad zur NIfTI-Datei (HU-Volumen)
    - out_seg: Pfad zur Ausgabe (Segmentierungs-NIfTI)
    - fast: True = schnellere, ggf. weniger genaue Variante
    - more_args: Liste mit zusätzlichen Kommandozeilen-Argumenten

    Rückgabe:
    - Der Rückgabecode des Programms (0 = Erfolg). Alles andere = Fehler.
    """
    args = ["TotalSegmentator", "-i", in_hu, "-o", out_seg, "--ml", "--quiet"]
    if fast:
        args.append("--fast")
    if more_args:
        args += more_args
    print("[CMD]", " ".join(args))
    # subprocess.call startet den externen Prozess und liefert dessen Exit-Code
    return subprocess.call(args)


# ======== HU-Statistiken ========
def hu_stats(hu, seg=None):
    """
    Berechnet einfache Statistiken der HU-Werte:
    - global (über alle Voxel)
    - optional: pro Label/Struktur (wenn eine Segmentierung 'seg' übergeben wird)

    Rückgabe:
    - dict mit "global" und ggf. "per_label"
    """
    stats = {
        "global": {
            "min": float(np.min(hu)),
            "max": float(np.max(hu)),
            "mean": float(np.mean(hu)),
            "std": float(np.std(hu)),
        }
    }

    # Wenn eine Segmentierung vorliegt, für jedes Label separat rechnen
    if seg is not None:
        labels = np.unique(seg[seg > 0])  # alle vorhandenen Label-IDs > 0
        per = {}
        for l in labels.astype(int):
            m = (seg == l)  # Maske: Voxel, die zu Label l gehören
            v = hu[m]       # HU-Werte nur in dieser Maske
            if v.size:
                per[int(l)] = {
                    "voxels": int(v.size),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v)),
                }
        stats["per_label"] = per
    return stats


# ======== main() ========
def main():
    """
    Liest die Kommandozeilen-Argumente, führt die einzelnen Schritte aus
    (Laden, Umrechnen, Speichern, optional Segmentieren/Resamplen/Preview/Stats)
    und schreibt die Ergebnisse in den angegebenen Ordner.
    """
    # --- Argumente definieren (was der/die Nutzer:in angeben kann) ---
    ap = argparse.ArgumentParser(description="XCAT µ(1/cm) -> NIfTI (HU) + optional TS + Overlay + Stats")
    ap.add_argument("--in-bin", required=True)                  # Eingabedatei (Binärdatei mit µ)
    ap.add_argument("--shape-xyz", required=True)               # 3D-Form, als "X,Y,Z" (z.B. "256,256,300")
    ap.add_argument("--bin-order", default="XYZ")               # Reihenfolge der Achsen in der Datei
    ap.add_argument("--array-order", default="C")               # Speicherlayout: "C"=row-major, "F"=Fortran
    ap.add_argument("--in-units", default="per_cm")             # Einheit der µ-Werte in der Datei
    ap.add_argument("--spacing", default="1.0,1.0,1.0")         # Voxelabstände (dz,dy,dx) in mm
    ap.add_argument("--mu-water", type=float, required=True)    # µ von Wasser [1/cm] (für HU-Umrechnung)
    ap.add_argument("--out-dir", default=".")                   # Ausgabeordner
    ap.add_argument("--tag", default="xcat")                    # Namens-Vorsilbe für Ausgabedateien

    # Anzeige / Overlay (für das Vorschaubild)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--plane", default="coronal", choices=["coronal","sagittal","axial"])
    ap.add_argument("--rotate", type=int, default=0)
    ap.add_argument("--flip-lr", action="store_true")
    ap.add_argument("--flip-ud", action="store_true")
    ap.add_argument("--legend", action="store_true")
    ap.add_argument("--legend-outside", action="store_true")
    ap.add_argument("--legend-max", type=int, default=15)
    ap.add_argument("--fill-alpha", type=float, default=0.35)
    ap.add_argument("--hu-png", action="store_true")
    ap.add_argument("--clip-hu-min", type=float, default=-1024.0)  # typische CT-Untergrenze
    ap.add_argument("--clip-hu-max", type=float, default=3071.0)   # typische CT-Obergrenze

    # Resampling (z.B. auf 1.0 mm isotrop)
    ap.add_argument("--resample-mm", type=float, default=1.0)

    # TotalSegmentator (automatische Organ-Segmentierung)
    ap.add_argument("--run-ts", action="store_true")
    ap.add_argument("--ts-fast", action="store_true")
    ap.add_argument("--ts-args", default="")      # zusätzliche TS-Argumente, z.B. "--task body"
    ap.add_argument("--ts-labels", default="")    # explizite Labeltabelle (JSON/CSV)
    ap.add_argument("--export-labels", action="store_true")  # die benutzten Labels als JSON speichern

    # Statistiken (HU global + pro Label)
    ap.add_argument("--stats", action="store_true")

    # --- Argumente einlesen ---
    args = ap.parse_args()

    # Sicherstellen, dass der Ausgabeordner existiert
    os.makedirs(args.out_dir, exist_ok=True)

    # Strings wie "256,256,300" in Tupel von Zahlen umwandeln
    shape_xyz = tuple(map(int, args.shape_xyz.split(",")))
    spacing_zyx = tuple(map(float, args.spacing.split(",")))  # (dz, dy, dx)
    print(f"[INFO] shape={shape_xyz}, order={args.bin_order}, spacing={spacing_zyx}")

    # === 1) µ laden (→ Ergebnis-Array hat Achsreihenfolge (Z, Y, X)) ===
    mu_cm = load_bin_with_order(args.in_bin, shape_xyz, args.bin_order, args.array_order)

    # NaN/Inf-Werte (falls vorhanden) auf 0 setzen, damit Rechnungen stabil sind
    mu_cm = np.nan_to_num(mu_cm, nan=0.0, posinf=0.0, neginf=0.0)

    # Falls die Eingabeeinheit 1/mm war, in 1/cm umrechnen (1 cm = 10 mm)
    if args.in_units == "per_mm":
        mu_cm *= 10.0  # 1/mm → 1/cm

    # === 2) µ → HU umrechnen ===
    # Formel: HU = 1000 * (µ - µ_Wasser) / µ_Wasser
    # Idee: Wasser hat HU = 0, Luft ca. -1000 HU, Knochen stark positiv.
    mu_w = float(args.mu_water)
    hu = 1000.0 * (mu_cm - mu_w) / mu_w

    # HU auf sinnvollen Bereich begrenzen (sonst könnten Ausreißer die Anzeige sprengen)
    hu = np.clip(hu, args.clip_hu_min, args.clip_hu_max)

    # === 3) NIfTI speichern (HU) + optional Resampling ===
    nii_hu = os.path.join(args.out_dir, f"{args.tag}_HU.nii.gz")
    save_nifti(hu, spacing_zyx, nii_hu)

    # Falls SimpleITK verfügbar und resample_mm gesetzt ist: isotrop resamplen
    if sitk is not None and args.resample_mm:
        nii_iso = os.path.join(args.out_dir, f"{args.tag}_HU_iso{args.resample_mm:.1f}mm.nii.gz")
        resample_isotropic(nii_hu, nii_iso, args.resample_mm)
        nii_hu = nii_iso  # ab hier mit der resampleten Datei weiterarbeiten

    # === 4) TotalSegmentator (optional) ===
    seg_path = None
    if args.run_ts:
        seg_path = os.path.join(args.out_dir, f"{args.tag}_ts_seg.nii.gz")
        # Zusätzliche Argumente als Liste (falls der String nicht leer ist)
        more = args.ts_args.split() if args.ts_args.strip() else None
        rc = run_totalseg(nii_hu, seg_path, fast=args.ts_fast, more_args=more)
        if rc != 0:
            print("[ERR] TotalSegmentator fehlgeschlagen.")
            seg_path = None  # Segmentierung ignorieren, damit Folgefunktionen nicht crashen

    # === 5) Vorschau (Overlay) erzeugen (optional) ===
    if args.preview:
        # Labelnamen automatisch finden (oder aus --ts-labels nehmen)
        label_map = _label_map_from_seg(seg_path, args.ts_labels or None)
        # Optional: die verwendete Labeltabelle als JSON wegschreiben
        export_path = (os.path.join(args.out_dir, f"{args.tag}_labels_used.json")
                       if args.export_labels else None)

        png = os.path.join(args.out_dir, f"{args.tag}_overlay_{args.plane}.png")
        make_preview_slice(
            nii_hu, png, seg_path=seg_path, plane=args.plane,
            rotate_k=args.rotate, flip_lr=args.flip_lr, flip_ud=args.flip_ud,
            add_legend=args.legend, legend_max=args.legend_max,
            legend_outside=args.legend_outside, add_colorbar=args.hu_png,
            vmin=args.clip_hu_min, vmax=args.clip_hu_max,
            fill_alpha=args.fill_alpha, label_map=label_map,
            export_labels_path=export_path
        )
        print(f"[OK] Overlay -> {png}")

    # === 6) HU-Statistiken (optional) ===
    if args.stats:
        seg_arr = None
        if seg_path and os.path.isfile(seg_path):
            seg_arr = np.asarray(nib.load(seg_path).dataobj)
        stats = hu_stats(hu, seg_arr)
        out_json = os.path.join(args.out_dir, f"{args.tag}_HU_stats.json")
        with open(out_json, "w") as f:
            json.dump(stats, f, indent=2)
        print("[OK] HU-Stats:", stats["global"])


# Starte das Programm nur, wenn die Datei direkt ausgeführt wird (nicht beim Import als Modul)
if __name__ == "__main__":
    main()