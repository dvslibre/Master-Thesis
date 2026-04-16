"""
Aufruf:
python VOI_evaluation.py \
    --spect_dicom_dir /home/mnguest12/projects/thesis/SPECTCT_Pipeline/data/example_01/spect/LU-177-SPECT-CT-DOSIMETRIE-RECON-AC \
    --ct_dicom_dir  /home/mnguest12/projects/thesis/SPECTCT_Pipeline/data/example_01/ct/3D-ABDROUTINE-1.5-B31S \
    --mask_dir      /home/mnguest12/projects/thesis/SPECTCT_Pipeline/data/example_01/results/seg/individual \
    --output_csv    /home/mnguest12/projects/thesis/SPECTCT_Pipeline/data/example_01/results/spect/spect_stats.tsv
"""

#!/usr/bin/env python

import os
import argparse
import csv

import numpy as np
import SimpleITK as sitk

import matplotlib
matplotlib.use("Agg")  # kein GUI nötig
import matplotlib.pyplot as plt


# Welche VOIs ausgewertet werden sollen und wie die Dateien heißen
VOI_FILES = {
    "kidney_left": "kidney_left.nii.gz",
    "kidney_right": "kidney_right.nii.gz",
    "liver": "liver.nii.gz",
    "prostate": "prostate.nii.gz",
    "small_bowel": "small_bowel.nii.gz",
    "spleen": "spleen.nii.gz",
}


def read_dicom_series(dicom_dir: str) -> sitk.Image:
    """
    Liest eine DICOM-Serie (z.B. spect oder CT) mit SimpleITK ein.
    """
    if not os.path.isdir(dicom_dir):
        raise RuntimeError(f"Verzeichnis existiert nicht (oder ist kein Ordner): {dicom_dir}")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise RuntimeError(f"Keine DICOM-Serien gefunden in: {dicom_dir}")

    # Wenn mehrere Serien im Ordner sind, nimm die erste
    series_id = series_ids[0]
    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)
    reader.SetFileNames(file_names)

    image = reader.Execute()
    return image


def resample_mask_to_spect(mask_img: sitk.Image, spect_img: sitk.Image) -> sitk.Image:
    """
    Resample der VOI-Maske (CT-Space) in den spect-Space.
    Nearest-Neighbor-Interpolation, damit es eine saubere Binärmaske bleibt.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(spect_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())  # Identität
    resampler.SetDefaultPixelValue(0)

    mask_resampled = resampler.Execute(mask_img)
    return mask_resampled


def resample_image_to_target(
    img: sitk.Image,
    reference_img: sitk.Image,
    interpolator=sitk.sitkLinear,
    default_value=0,
) -> sitk.Image:
    """
    Allgemeines Resampling eines Bildes (z.B. CT) in den Raum eines Referenzbildes (z.B. spect).
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(img)


def compute_statistics(spect_img: sitk.Image, mask_img: sitk.Image):
    """
    Berechnet Summe, Mittelwert, Min, Max für alle spect-Werte innerhalb der Maske.
    """
    spect_arr = sitk.GetArrayFromImage(spect_img)      # shape: [z, y, x]
    mask_arr = sitk.GetArrayFromImage(mask_img)

    # Maske: alles > 0 als 'innerhalb'
    mask_bool = mask_arr > 0

    # spect-Werte innerhalb der VOI
    values = spect_arr[mask_bool]

    if values.size == 0:
        return {
            "num_voxels": 0,
            "sum": np.nan,
            "mean": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "num_voxels": int(values.size),
        "sum": float(np.sum(values)),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def fmt_int(value):
    """Hilfsfunktion: float -> ganze Zahl, NaN -> leere Zelle."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        return int(round(value))
    if value is None:
        return ""
    return value

def write_readable_table(results, out_path):
    """
    Schreibt eine schön ausgerichtete Texttabelle (fixed width) nach out_path.
    Gut zum Lesen im Editor/Terminal.
    """
    headers = ["series", "organ", "num_voxels", "sum", "mean", "min", "max"]

    # Alle Zeilen vorbereiten
    rows = []
    rows.append(headers)
    for r in results:
        rows.append([
            r["series"],
            r["organ"],
            fmt_int(r["num_voxels"]),
            fmt_int(r["sum"]),
            fmt_int(r["mean"]),
            fmt_int(r["min"]),
            fmt_int(r["max"]),
        ])

    # Spaltenbreiten bestimmen
    col_widths = [
        max(len(str(row[i])) for row in rows)
        for i in range(len(headers))
    ]

    # Schreiben
    with open(out_path, "w") as f:
        for idx, row in enumerate(rows):
            parts = []
            for col_idx, value in enumerate(row):
                text = str(value)
                # series & organ linksbündig, Rest rechtsbündig
                if col_idx <= 1:
                    parts.append(text.ljust(col_widths[col_idx]))
                else:
                    parts.append(text.rjust(col_widths[col_idx]))
            line = "  ".join(parts)
            f.write(line + "\n")
            if idx == 0:
                f.write("-" * len(line) + "\n")


def create_organ_visualizations(
    organ: str,
    spect_img: sitk.Image,
    mask_img_spect_space: sitk.Image,
    out_dir: str,
    ct_img_spect_space: sitk.Image | None = None,
):
    """
    Erzeugt für ein Organ:
      - Overlay-Bild (CT grau + spect Heatmap + Maskenkontur)
      - Histogramm der spect-Counts im Organ
    und speichert beides in einer gemeinsamen Figure als PNG.
    """
    os.makedirs(out_dir, exist_ok=True)

    spect_arr = sitk.GetArrayFromImage(spect_img)
    mask_arr = sitk.GetArrayFromImage(mask_img_spect_space) > 0

    if not mask_arr.any():
        # Nichts in der Maske -> keine Visualisierung nötig
        return

    if ct_img_spect_space is not None:
        ct_arr = sitk.GetArrayFromImage(ct_img_spect_space)
    else:
        ct_arr = None

    # z-Slices, in denen das Organ vorkommt
    z_indices = np.where(mask_arr.any(axis=(1, 2)))[0]
    if z_indices.size == 0:
        return

    # Nehme den "mittleren" Slice
    z = int(z_indices[len(z_indices) // 2])

    spect_slice = spect_arr[z, :, :]
    mask_slice = mask_arr[z, :, :]

    if ct_arr is not None:
        ct_slice = ct_arr[z, :, :]
    else:
        ct_slice = None

    # Werte im Organ für Histogramm
    organ_values = spect_arr[mask_arr]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Overlay-Plot ---
    ax0 = axes[0]
    if ct_slice is not None:
        ax0.imshow(ct_slice, cmap="gray")
    else:
        ax0.imshow(spect_slice, cmap="gray")

    ax0.imshow(spect_slice, cmap="hot", alpha=0.5)
    ax0.contour(mask_slice, levels=[0.5], colors="cyan", linewidths=1)
    ax0.set_title(f"{organ} – Slice {z}")
    ax0.axis("off")

    # --- Histogramm ---
    ax1 = axes[1]
    ax1.hist(organ_values, bins=50)
    ax1.set_title(f"Histogram {organ}")
    ax1.set_xlabel("spect Counts")
    ax1.set_ylabel("Voxel-Anzahl")

    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{organ}_overlay_hist.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)



def collapse_4d_to_3d(img: sitk.Image, mode: str = "sum") -> sitk.Image:
    """
    Reduziert ein 4D-Bild (t,z,y,x) auf 3D (z,y,x), indem über t aggregiert wird.
    Spacing/Origin bleiben für die ersten 3 Dimensionen gleich.
    Die 3x3-Orientierungsmatrix wird korrekt aus der 4x4-Orientierung extrahiert,
    indem die Zeit-Spalte entfernt wird.
    """
    if img.GetDimension() == 3:
        return img
    if img.GetDimension() != 4:
        raise ValueError(f"Unexpected dimension: {img.GetDimension()}")

    # SimpleITK-Array: (t,z,y,x)
    arr4 = sitk.GetArrayFromImage(img)

    if mode == "sum":
        arr3 = arr4.sum(axis=0)
    elif mode == "mean":
        arr3 = arr4.mean(axis=0)
    elif mode == "first":
        arr3 = arr4[0]
    else:
        raise ValueError("mode must be 'sum', 'mean', or 'first'.")

    img3 = sitk.GetImageFromArray(arr3)

    spacing4 = img.GetSpacing()   # (sx, sy, sz, st)
    origin4 = img.GetOrigin()     # (ox, oy, oz, ot)
    dir4_flat = img.GetDirection()  # 16 Werte
    dir4 = np.array(dir4_flat, dtype=float).reshape(4, 4)

    # räumliche 3x3-Matrix: nimm die ersten 3 Zeilen und die ersten 3 Spalten
    dir3 = dir4[:3, :3].reshape(-1)

    img3.SetSpacing(spacing4[:3])
    img3.SetOrigin(origin4[:3])
    img3.SetDirection(tuple(dir3))

    return img3


def run_evaluation(
    spect_dicom_dir: str,
    mask_dir: str,
    output_csv: str | None = None,
    csv_mode: str = "overwrite",
    ct_dicom_dir: str | None = None,
):
    # Serienlabel aus dem DICOM-Verzeichnis (z.B. spect_WB_TRUEX2I14S0MM_AC_0102)
    series_label = os.path.basename(os.path.normpath(spect_dicom_dir))

    print(f"Lese spect-DICOM-Serie aus: {spect_dicom_dir}")
    spect_img = read_dicom_series(spect_dicom_dir)
    print("DEBUG SPECT:")
    print("  dim:", spect_img.GetDimension())
    print("  size:", spect_img.GetSize())
    print("  spacing:", spect_img.GetSpacing())
    print("  direction len:", len(spect_img.GetDirection()))
    if spect_img.GetDimension() == 4:
        print("SPECT ist 4D – reduziere auf 3D (sum).")
        spect_img = collapse_4d_to_3d(spect_img, mode="sum")

    # CT optional einlesen und in spect-Space bringen
    ct_img_spect_space = None
    if ct_dicom_dir is not None:
        print(f"Lese CT-DICOM-Serie aus: {ct_dicom_dir}")
        ct_img = read_dicom_series(ct_dicom_dir)
        print("DEBUG CT:")
        print("  dim:", ct_img.GetDimension())
        print("  size:", ct_img.GetSize())
        print("  spacing:", ct_img.GetSpacing())
        print("  direction len:", len(ct_img.GetDirection()))
        print("Resample CT -> spect-Space ...")
        ct_img_spect_space = resample_image_to_target(
            ct_img, spect_img, interpolator=sitk.sitkLinear, default_value=-1024
        )

    # Output-Verzeichnis für Figuren
    fig_dir = None
    if output_csv is not None:
        fig_dir = os.path.join(os.path.dirname(output_csv), "figures")

    results = []

    for organ, filename in VOI_FILES.items():
        mask_path = os.path.join(mask_dir, filename)

        if not os.path.exists(mask_path):
            print(f"[WARNUNG] Maske für {organ} nicht gefunden: {mask_path}")
            continue

        print(f"Verarbeite Organ: {organ}")
        print(f"  Lese Maske: {mask_path}")
        mask_img_ct_space = sitk.ReadImage(mask_path)

        # Maske in spect-Space bringen
        mask_resampled = resample_mask_to_spect(mask_img_ct_space, spect_img)

        stats = compute_statistics(spect_img, mask_resampled)
        stats["organ"] = organ
        stats["series"] = series_label
        results.append(stats)

        # Ausgabe auf der Konsole
        print(f"  Serie:        {series_label}")
        print(f"  Anzahl Voxel: {stats['num_voxels']}")
        print(f"  Summe:        {stats['sum']}")
        print(f"  Mittelwert:   {stats['mean']}")
        print(f"  Minimum:      {stats['min']}")
        print(f"  Maximum:      {stats['max']}")
        print("-" * 40)

        # Visualisierung (Overlay + Histogramm)
        if fig_dir is not None:
            create_organ_visualizations(
                organ=organ,
                spect_img=spect_img,
                mask_img_spect_space=mask_resampled,
                out_dir=fig_dir,
                ct_img_spect_space=ct_img_spect_space,
            )

    # Optional als TSV (Tab-getrennt) speichern, ohne Nachkommastellen
    if output_csv is not None:
        fieldnames = ["series", "organ", "num_voxels", "sum", "mean", "min", "max"]

        file_exists = os.path.exists(output_csv)
        if csv_mode == "overwrite":
            mode = "w"
            write_header = True
        elif csv_mode == "append":
            mode = "a"
            write_header = not file_exists
        else:
            raise ValueError(f"Ungültiger csv_mode: {csv_mode}. Erlaubt: overwrite, append")

        print(f"Schreibe Ergebnisse in TSV: {output_csv} (Modus: {csv_mode})")

        with open(output_csv, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            if write_header:
                writer.writeheader()
            for r in results:
                writer.writerow({
                    "series": r["series"],
                    "organ": r["organ"],
                    "num_voxels": fmt_int(r["num_voxels"]),
                    "sum": fmt_int(r["sum"]),
                    "mean": fmt_int(r["mean"]),
                    "min": fmt_int(r["min"]),
                    "max": fmt_int(r["max"]),
                })
        # Zusätzlich eine hübsch formatierte Texttabelle schreiben
        readable_path = os.path.splitext(output_csv)[0] + "_readable.txt"
        print(f"Schreibe lesbare Texttabelle nach: {readable_path}")
        write_readable_table(results, readable_path)


def main():
    parser = argparse.ArgumentParser(
        description="Auswertung von spect-Werten innerhalb von VOI-Masken"
    )
    parser.add_argument(
        "--spect_dicom_dir",
        required=True,
        help="Pfad zum DICOM-Ordner der spect-Serie",
    )
    parser.add_argument(
        "--mask_dir",
        required=True,
        help="Pfad zum Ordner mit den VOI-NIfTIs (aus TotalSegmentator, CT-Space)",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional: Pfad zu einer TSV/CSV-Datei für die Ergebnisse",
    )
    parser.add_argument(
        "--csv_mode",
        choices=["overwrite", "append"],
        default="overwrite",
        help="Wie mit der TSV/CSV-Datei verfahren werden soll: "
             "'overwrite' = Datei neu schreiben, "
             "'append' = an bestehende Datei anhängen.",
    )
    parser.add_argument(
        "--ct_dicom_dir",
        default=None,
        help="Optional: Pfad zum DICOM-Ordner der CT-Serie "
             "(für Overlays CT+spect+Maske).",
    )

    args = parser.parse_args()

    run_evaluation(
        spect_dicom_dir=args.spect_dicom_dir,
        mask_dir=args.mask_dir,
        output_csv=args.output_csv,
        csv_mode=args.csv_mode,
        ct_dicom_dir=args.ct_dicom_dir,
    )


if __name__ == "__main__":
    main()
