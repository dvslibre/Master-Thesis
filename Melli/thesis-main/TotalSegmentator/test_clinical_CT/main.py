#!/usr/bin/env python3
"""
Kleines CLI-Tool für den Workflow:
DICOM-Serie laden → NIfTI speichern → optional isotrop resamplen → TotalSegmentator ausführen.

Abhängigkeiten:
- SimpleITK (DICOM lesen/schreiben, Resampling)
- nibabel (NIfTI-Inspektion)
- TotalSegmentator (externes CLI-Tool)
"""

import argparse, os, sys, subprocess, shutil
from pathlib import Path
import numpy as np

# SimpleITK ist Pflicht. Falls es fehlt, sauber abbrechen.
try:
    import SimpleITK as sitk
except ImportError:
    print("[ERROR] SimpleITK fehlt: `pip install SimpleITK`", file=sys.stderr)
    sys.exit(1)


def log(*a):
    """Kleine Hilfsfunktion: schreibt sofort (ohne Puffern) auf STDOUT."""
    print(*a, flush=True)


def ensure_dir(p: Path):
    """Legt einen Ordner an (rekursiv), wenn er noch nicht existiert."""
    p.mkdir(parents=True, exist_ok=True)


def load_dicom_series(dicom_dir: Path, series_uid: str = None) -> sitk.Image:
    """
    Liest eine DICOM-Serie mit SimpleITK ein und liefert ein 3D-Image zurück.

    - Wenn mehrere Serien im Ordner liegen, kann per series_uid gezielt gewählt werden.
    - Rescale Slope/Intercept wird automatisch angewandt → Hounsfield Units bei CT.
    """
    r = sitk.ImageSeriesReader()
    # Sorgt dafür, dass auch private DICOM-Tags gelesen werden (hilfreich bei Herstellervarianten).
    r.MetaDataDictionaryArrayUpdateOn(); r.LoadPrivateTagsOn()

    # Alle Serien-IDs (UIDs) im Ordner abfragen.
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError(f"Keine DICOM-Serie gefunden in: {dicom_dir}")

    # Entweder die gewünschte UID nutzen oder die erste gefundene Serie nehmen.
    use_uid = series_uid or series_ids[0]

    # Liste der DICOM-Dateien, die zu dieser Serie gehören.
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), use_uid)
    if not files:
        raise RuntimeError("Serie hat keine Dateien.")

    r.SetFileNames(files)
    return r.Execute()  # Enthält CT-Daten in HU dank Slope/Intercept


def save_nifti(img: sitk.Image, out_path: Path):
    """
    Speichert ein SimpleITK-Image als NIfTI. Typ wird auf Int16 gecastet (üblich für CT in HU).
    """
    sitk.WriteImage(sitk.Cast(img, sitk.sitkInt16), str(out_path), True)


def resample_isotropic(img: sitk.Image, iso_mm: float) -> sitk.Image:
    """
    Resampling auf isotrope Voxelgröße (z.B. 1.0 mm).
    - Berechnet neue Bildgröße konsistent zur neuen Auflösung.
    - Linearer Interpolator ist für CT ausreichend.
    - Luft als Default-Pixelwert (-1000 HU), falls Bereiche außerhalb entstehen.
    """
    sp_old = np.array(list(img.GetSpacing()), dtype=float)  # alte Auflösung (mm)
    sz_old = np.array(list(img.GetSize()), dtype=float)     # alte Größe (Voxel)
    sp_new = np.array([iso_mm, iso_mm, iso_mm], dtype=float)
    # neue Größe so bestimmen, dass physische Ausdehnung erhalten bleibt
    sz_new = np.rint(sz_old * (sp_old / sp_new)).astype(int).tolist()

    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(tuple(sp_new))
    f.SetSize(sz_new)
    f.SetOutputDirection(img.GetDirection())  # Orientierung beibehalten
    f.SetOutputOrigin(img.GetOrigin())        # Ursprung beibehalten
    f.SetInterpolator(sitk.sitkLinear)
    f.SetDefaultPixelValue(-1000.0)           # Luft in HU
    return f.Execute(img)


def hu_stats(nifti_path: Path):
    """
    Öffnet eine NIfTI-Datei und liefert einfache Statistik über die Hounsfield-Werte.
    Praktisch, um grobe Fehler (falsches Rescaling etc.) zu erkennen.
    """
    import nibabel as nib
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    return dict(
        shape=tuple(int(x) for x in data.shape),                 # (Z, Y, X) bzw. (H, W, D) je nach Ordnung
        spacing=tuple(float(z) for z in img.header.get_zooms()), # Voxelabstände (mm)
        min=float(np.nanmin(data)),
        median=float(np.nanmedian(data)),
        max=float(np.nanmax(data)),
    )


def run_totalsegmentator(nifti_path: Path, out_dir: Path, device: str, fast: bool, extra_args: str):
    """
    Startet das TotalSegmentator-CLI für die Segmentierung.

    - device: 'cuda' wird zu 'gpu' normalisiert, 'cuda:X' zu 'gpu:X'
    - --fast: schnellere, aber etwas grobere Segmentierung
    - extra_args: frei durchgereichte zusätzliche CLI-Argumente
    """
    def normalize_device(d):
        if not d: return None
        d = d.lower()
        if d == "cuda": return "gpu"
        if d.startswith("cuda:"): return "gpu:" + d.split(":", 1)[1]
        return d  # akzeptiert auch 'gpu', 'gpu:X', 'cpu', 'mps'

    dev = normalize_device(device)
    cmd = ["TotalSegmentator", "-i", str(nifti_path), "-o", str(out_dir)]
    if fast: cmd += ["--fast"]
    if dev:  cmd += ["--device", dev]
    if extra_args: cmd += extra_args.split()

    log("[CMD]", " ".join(cmd))
    # stdout/stderr direkt durchreichen, damit der Fortschritt sichtbar ist
    res = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"TotalSegmentator fehlgeschlagen (exit {res.returncode})")


def main():
    """
    CLI-Parsing und Orchestrierung der Verarbeitungsschritte.
    """
    ap = argparse.ArgumentParser(description="DICOM → NIfTI → (optional 1mm) → TotalSegmentator")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dicom_dir", type=Path, help="Pfad DICOM-Serie")
    g.add_argument("--in_nifti",  type=Path, help="fertiges NIfTI")
    ap.add_argument("--series_uid", type=str, default=None, help="DICOM SeriesInstanceUID (optional)")
    ap.add_argument("--out_dir", type=Path, required=True, help="Zielordner für Ergebnisse")
    ap.add_argument("--resample_mm", type=float, default=None, help="Isotropes Resampling (z.B. 1.0)")
    ap.add_argument("--device", type=str, default="cuda",
                    choices=["cuda","cpu","gpu","mps"], help="Rechen-Device für TotalSegmentator")
    ap.add_argument("--fast", action="store_true", help="Schneller Modus von TotalSegmentator")
    ap.add_argument("--extra_ts_args", type=str, default="", help="Weitere CLI-Argumente für TotalSegmentator")
    ap.add_argument("--keep_intermediate", action="store_true", help="work/-Ordner behalten (Debug)")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    work = args.out_dir / "work"; ensure_dir(work)

    # 1) Input laden
    if args.in_nifti:
        # Bereits vorhandenes NIfTI nutzen
        nifti = args.in_nifti.resolve()
        if not nifti.exists():
            raise FileNotFoundError(nifti)
        log(f"[INFO] Verwende NIfTI: {nifti}")
        img = sitk.ReadImage(str(nifti))
    else:
        # DICOM-Serie einlesen und als NIfTI ablegen
        log(f"[STEP] Lade DICOM-Serie aus: {args.dicom_dir}")
        img = load_dicom_series(args.dicom_dir, args.series_uid)
        nifti = work / "ct_input.nii.gz"
        save_nifti(img, nifti)
        log(f"[OK] NIfTI geschrieben: {nifti}")

    # 2) Optionales Resampling auf isotrope Voxel
    if args.resample_mm is not None:
        log(f"[STEP] Resample auf isotrop {args.resample_mm} mm …")
        img = resample_isotropic(img, args.resample_mm)
        nifti = work / f"ct_input_iso{args.resample_mm:.1f}mm.nii.gz"
        save_nifti(img, nifti)
        log(f"[OK] Resampled NIfTI: {nifti}")

    # 3) Grobe Plausibilitätschecks (Größe, Abstände, HU-Verteilung)
    try:
        stats = hu_stats(nifti)
        log("[CHECK] shape:", stats["shape"])
        log("[CHECK] spacing (mm):", stats["spacing"])
        log("[CHECK] HU stats:", f"min={stats['min']:.1f}  median={stats['median']:.1f}  max={stats['max']:.1f}")
        # Sehr grobe Heuristik: CT-Luft typischerweise ~ -1000 HU, Knochen hoch → median>>0 kann auf Fehler hindeuten.
        if stats["min"] > -800 or stats["median"] > 200:
            log("[WARN] HU ungewoehnlich. Serie/Rescale pruefen.")
    except Exception as e:
        # Statistik ist nur "nice to have" – Fehler hier sollen den Lauf nicht abbrechen.
        log("[WARN] HU-Check fehlgeschlagen:", e)

    # 4) TotalSegmentator ausführen
    ts_out = args.out_dir / "ts_output"; ensure_dir(ts_out)
    run_totalsegmentator(nifti, ts_out, args.device, args.fast, args.extra_ts_args)

    # 5) Aufräumen (work/-Ordner nur behalten, wenn explizit gewünscht)
    if not args.keep_intermediate and work.exists():
        try:
            shutil.rmtree(work)
        except Exception as e:
            log("[WARN] work/ nicht geloescht:", e)

    log("[DONE] TotalSegmentator fertig.")
    log(f"→ Labels: {ts_out}")


if __name__ == "__main__":
    main()