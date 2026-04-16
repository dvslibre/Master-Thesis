#!/usr/bin/env python3
"""
Erzeugt Overlay-Previews aus CT (HU) + Segmentierungen:
- CT wird mit Fensterung (WL/WW) nach 0..1 abgebildet.
- Segmentierungen können als Multi-Label-NIfTI oder als Ordner mit Einzelmasken vorliegen.
- Es werden drei Mittelschnitte (axial, koronal, sagittal) als PNG exportiert.
"""

import argparse, json
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi


def load_nifti(p: Path):
    """
    Lädt eine NIfTI-Datei und liefert (Datenarray, affine Matrix, Voxelgrößen/zooms).
    dataobj wird lazy gelesen; hier in ein NumPy-Array materialisiert.
    """
    img = nib.load(str(p))
    data = np.asarray(img.dataobj)
    return data, img.affine, img.header.get_zooms()


def window_hu(x, wl=-200, ww=400):
    """
    Einfache CT-Fensterung (HU → 0..1):
    - wl: window level (Mitte), ww: window width (Breite)
    - Werte außerhalb des Fensters werden auf [0,1] geclippt.
    """
    lo, hi = wl - ww/2.0, wl + ww/2.0
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0, 1)


def find_seg_source(seg_path: Path):
    """
    Ermittelt, woher die Segmentierung kommt:
    - Datei (Multi-Label NIfTI) → ("multi", pfad)
    - Ordner mit bekannten Multi-Label-Dateinamen → ("multi", pfad)
    - Sonst: Ordner mit Einzelmasken (rglob auf *.nii*) → ("dir", ordner)
    """
    if seg_path.is_file():
        return ("multi", seg_path)
    if seg_path.is_dir():
        for c in ["ts_total.nii.gz", "segmentations.nii.gz", "TotalSegmentator.nii.gz", "labels.nii.gz"]:
            p = seg_path / c
            if p.exists(): return ("multi", p)
        nii_files = sorted([p for p in seg_path.rglob("*.nii*")])
        if not nii_files:
            raise FileNotFoundError(f"Keine NIfTI-Dateien in {seg_path} gefunden.")
        return ("dir", seg_path)
    raise FileNotFoundError(f"Seg-Pfad nicht gefunden: {seg_path}")


def build_label_volume_from_dir(shape, seg_dir: Path):
    """
    Baut aus vielen Einzelmasken (je Datei eine binäre Maske) ein gemeinsames Labelvolumen:
    - Jede gefundene Maske, die shape-matching ist und >0 Voxels hat, bekommt eine neue Label-ID.
    - Rückgabe: (vol, names) mit vol=int16-Labelbild und names={id: name}.
    """
    files = sorted([p for p in seg_dir.rglob("*.nii*")])
    vol = np.zeros(shape, dtype=np.int16); names = {}; next_id = 1
    for f in files:
        try:
            d, _, _ = load_nifti(f)
            if d.shape != shape:  # falsche Dimension → überspringen
                continue
            m = (d > 0.5)        # binarisieren (robust gegen float/bool)
            if not np.any(m):
                continue
            vol[m] = next_id
            names[next_id] = f.stem  # Dateiname als Labelname
            next_id += 1
        except Exception:
            # Einzeldatei fehlerhaft? Ignorieren und weitermachen.
            continue
    if vol.max() == 0:
        raise RuntimeError("Aus Einzelmasken konnte kein Label-Volumen gebaut werden.")
    return vol, names


def try_load_label_names(ts_out_dir: Path):
    """
    Versucht, eine labels.json neben der Segmentierung zu laden.
    Erwartet { "<int>": "name", ... } → Keys werden zu int gecastet.
    """
    j = ts_out_dir / "labels.json"
    if j.exists():
        try:
            d = json.load(open(j, "r"))
            return {int(k): v for k, v in d.items() if str(k).isdigit()}
        except Exception:
            pass
    return {}


def color_for_label(i):
    """
    Wählt zyklisch Farben aus tab20/tab20b/tab20c (viele gut unterscheidbare Töne).
    """
    base = list(plt.cm.get_cmap("tab20").colors) \
         + list(plt.cm.get_cmap("tab20b").colors) \
         + list(plt.cm.get_cmap("tab20c").colors)
    return base[i % len(base)]


def make_overlay(base2d, label2d, names, out_png, alpha=0.45, filled=True):
    """
    Rendert ein Overlay-Bild:
    - base2d: 2D-Graubild bereits in 0..1 (z.B. CT nach Fensterung)
    - label2d: 2D-Labelbild (0=Hintergrund, >0 Klassen-IDs)
    - names: Mapping {id: name} für Legende
    - out_png: Zielpfad
    - alpha/filled: Deckkraft und ob Flächen gefüllt werden (neben Kontur)
    """
    h, w = base2d.shape
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Grundbild in Graustufen; .T + origin="lower" sorgt für radiologische Konvention
    plt.figure(figsize=(7, 7))
    plt.imshow(base2d.T, origin="lower", cmap="gray", interpolation="nearest")

    # Nur vorhandene Labels berücksichtigen
    labels_in = np.unique(label2d)
    labels_in = labels_in[labels_in > 0]
    if labels_in.size == 0:
        plt.title("Keine Label im Mittelschnitt")
        plt.axis("off")
        plt.savefig(out_png, bbox_inches="tight", dpi=220)
        plt.close()
        return

    # RGBA-Canvas für zusammengesetztes Overlay (wir zeichnen alles in einem Rutsch)
    overlay_rgba = np.zeros((w, h, 4), dtype=np.float32)  # Achtung: wir plotten transponiert

    legend_items, legend_colors = [], []

    # Große Flächen zuerst, damit kleine später obenauf liegen
    areas = {int(l): (label2d == l).sum() for l in labels_in}
    labels_sorted = np.array(sorted(labels_in, key=lambda l: -areas[int(l)]), dtype=int)

    for idx, lab in enumerate(labels_sorted):
        mask = (label2d == lab)
        if not np.any(mask):
            continue

        # Farbwahl + Deckkraft
        col = color_for_label(idx)  # (r,g,b) in 0..1
        r, g, b = col
        a = float(alpha)

        if filled:
            # Alpha-Compositing nur an maskierten Pixeln (Quelle über Ziel)
            src = np.zeros_like(overlay_rgba)
            src[..., 0] = r; src[..., 1] = g; src[..., 2] = b; src[..., 3] = a
            mT = mask.T  # wir rendern transponiert
            dst = overlay_rgba

            Sa, Da = src[..., 3], dst[..., 3]
            out_a = Sa * mT + Da * (1 - Sa * mT)
            eps = 1e-6  # numerische Stabilität
            out_r = (src[..., 0]*Sa*mT + dst[..., 0]*Da*(1 - Sa*mT)) / (out_a + eps)
            out_g = (src[..., 1]*Sa*mT + dst[..., 1]*Da*(1 - Sa*mT)) / (out_a + eps)
            out_b = (src[..., 2]*Sa*mT + dst[..., 2]*Da*(1 - Sa*mT)) / (out_a + eps)
            overlay_rgba[..., 0] = np.where(out_a > eps, out_r, overlay_rgba[..., 0])
            overlay_rgba[..., 1] = np.where(out_a > eps, out_g, overlay_rgba[..., 1])
            overlay_rgba[..., 2] = np.where(out_a > eps, out_b, overlay_rgba[..., 2])
            overlay_rgba[..., 3] = np.maximum(overlay_rgba[..., 3], out_a)

        # Kontur klar zeichnen (immer obendrauf)
        try:
            plt.contour(mask.T, levels=[0.5], colors=[col], linewidths=2.0)
        except Exception:
            pass

        # Legende begrenzen, damit es nicht ausufert
        if len(legend_items) < 15:
            legend_items.append(names.get(int(lab), f"Label {int(lab)}"))
            legend_colors.append(col)

    # RGBA-Overlay einmalig drüberlegen
    if overlay_rgba[..., 3].max() > 0:
        plt.imshow(overlay_rgba, origin="lower", interpolation="nearest")

    # Kompakte Legende (unten rechts)
    if legend_items:
        patches = [plt.Line2D([0], [0], marker='s', linestyle='',
                              markerfacecolor=c, markeredgecolor='none', markersize=8)
                   for c in legend_colors]
        plt.legend(patches, legend_items, loc="lower right", fontsize=8, framealpha=0.6)

    plt.axis("off")
    plt.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close()


def pick_middle_slices(vol):
    """
    Wählt Mittelschnitte entlang x/y/z (einfacher Heuristik: jeweils Index // 2).
    """
    x = vol.shape[0] // 2; y = vol.shape[1] // 2; z = vol.shape[2] // 2
    return x, y, z


def main():
    """
    CLI-Parsing, Laden der Daten, Ableiten der Labels und Erzeugen der drei Preview-PNGs.
    """
    ap = argparse.ArgumentParser(description="Overlay TotalSegmentator Masken (Mittelschnitt)")
    ap.add_argument("--in_hu", required=True, type=Path, help="CT als NIfTI (HU-Werte)")
    ap.add_argument("--seg", required=True, type=Path, help="Multi-Label NIfTI ODER Ordner mit Einzelmasken")
    ap.add_argument("--outdir", required=True, type=Path, help="Zielordner für Previews")
    ap.add_argument("--alpha", type=float, default=0.35, help="Deckkraft der gefüllten Flächen")
    ap.add_argument("--filled", action="store_true", help="Flächen füllen (zusätzlich zur Kontur)")
    ap.add_argument("--wl", type=float, default=-200.0, help="Window Level für CT-Fensterung")
    ap.add_argument("--ww", type=float, default=400.0, help="Window Width für CT-Fensterung")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # CT laden und fenstern (0..1)
    ct, _, _ = load_nifti(args.in_hu)
    base = window_hu(ct.astype(np.float32), wl=args.wl, ww=args.ww)

    # Segmentierungsquelle ermitteln und laden
    mode, src = find_seg_source(args.seg)
    if mode == "multi":
        seg, _, _ = load_nifti(src)
        if seg.shape != base.shape:
            raise ValueError(f"Shape mismatch: CT {base.shape} vs SEG {seg.shape}")
        # label_names aus labels.json, wenn vorhanden (Datei oder Ordner)
        label_names = try_load_label_names(args.seg.parent if args.seg.is_file() else args.seg)
        labels = seg.astype(np.int32)
    else:
        # Einzelmasken zu einem Labelvolumen zusammensetzen
        seg_vol, names = build_label_volume_from_dir(base.shape, args.seg)
        label_names = names
        labels = seg_vol.astype(np.int32)

    # Mittelschnitte bestimmen
    xc, yc, zc = pick_middle_slices(base)

    # Drei Ansichten rendern
    make_overlay(base[:, :, zc], labels[:, :, zc], label_names, args.outdir / "preview_axial.png",
                 alpha=args.alpha, filled=args.filled)
    make_overlay(base[:, yc, :], labels[:, yc, :], label_names, args.outdir / "preview_coronal.png",
                 alpha=args.alpha, filled=args.filled)
    make_overlay(base[xc, :, :], labels[xc, :, :], label_names, args.outdir / "preview_sagittal.png",
                 alpha=args.alpha, filled=args.filled)

    print("[OK] Previews:", args.outdir)


if __name__ == "__main__":
    main()