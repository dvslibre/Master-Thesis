"""
python ROI_evaluation.py \
  --pet_dicom_dir /home/mnguest12/projects/thesis/PETCT_Pipeline/data/example_01/PET_WB_TRUEX2I14S0MM_AC_0102 \
  --mask_dir     /home/mnguest12/projects/thesis/PETCT_Pipeline/data/example_01/results/seg/individual \
  --ap           /home/mnguest12/projects/thesis/PETCT_Pipeline/data/example_01/results/scintigraphy_sim/projection_AP.npy \
  --pa           /home/mnguest12/projects/thesis/PETCT_Pipeline/data/example_01/results/scintigraphy_sim/projection_PA.npy \
  --output_csv   /home/mnguest12/projects/thesis/PETCT_Pipeline/data/example_01/results/scintigraphy_sim/roi_stats.tsv
"""


#!/usr/bin/env python

import os
import argparse
import csv
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Welche ROIs ausgewertet werden sollen (wie im 3D-Skript)
ROI_FILES = {
    "kidney_left": "kidney_left.nii.gz",
    "kidney_right": "kidney_right.nii.gz",
    "liver": "liver.nii.gz",
    "prostate": "prostate.nii.gz",
    "small_bowel": "small_bowel.nii.gz",
    "spleen": "spleen.nii.gz",
}


def read_dicom_series(dicom_dir: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise RuntimeError(f"Keine Serie in {dicom_dir}")
    fns = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(fns)
    return reader.Execute()


def resample_mask_to_pet(mask_img: sitk.Image, pet_img: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(pet_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(mask_img)


def project_mask_to_2d(mask_arr_zyx: np.ndarray, view: str) -> np.ndarray:
    """
    3D-Maske (z,y,x) -> 2D-Maske (z,x) für AP/PA,
    analog zur Projektion der Aktivität.
    """
    if view == "AP":
        return np.any(mask_arr_zyx, axis=1).astype(np.uint8)      # für jeden Pixel im 2D-Pixel wird geschaut, ob in ges. Tiefe irgendwo Maske = 1 war
    elif view == "PA":
        return np.any(mask_arr_zyx[:, ::-1, :], axis=1).astype(np.uint8)
    else:
        raise ValueError("view must be 'AP' or 'PA'")


def compute_2d_stats(im2d: np.ndarray, mask2d: np.ndarray):
    values = im2d[mask2d > 0]

    if values.size == 0:
        return dict(pixel_count=0, sum=np.nan, mean=np.nan, min=np.nan, max=np.nan)

    return dict(
        pixel_count=int(values.size),
        sum=float(np.sum(values)),
        mean=float(np.mean(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
    )


def fmt(value):
    """
    Werte für Tabellen sauber formatiert:
    - NaN -> ""
    - floats -> ganze Zahl (gerundet)
    - ints -> int
    """
    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        return str(int(round(value)))  # auf ganze Zahl runden

    return value


def write_readable_table(results, out_path):
    headers = ["series", "organ", "view", "pixel_count", "sum", "mean", "min", "max"]

    rows = [headers]
    for r in results:
        rows.append([
            r["series"], r["organ"], r["view"],
            fmt(r["pixel_count"]), fmt(r["sum"]), fmt(r["mean"]),
            fmt(r["min"]), fmt(r["max"])
        ])

    col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]

    with open(out_path, "w") as f:
        for idx, row in enumerate(rows):
            line = "  ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row))
            f.write(line + "\n")
            if idx == 0:
                f.write("-" * len(line) + "\n")


def create_roi_visualizations(
    organ: str,
    view: str,
    proj_2d: np.ndarray,
    mask_2d: np.ndarray,
    out_dir: str,
    spacing_x: float,
    spacing_z: float,
):
    """
    2D-Visualisierung:
      - Overlay: Szintigrafie (proj_2d) + ROI-Kontur (mask_2d)
      - Histogramm der Counts in der ROI.
      Geometrie wie in scintigraphy_sim.py (mm-basiert, aspect='equal').
    """
    if not np.any(mask_2d):
        return

    os.makedirs(out_dir, exist_ok=True)

    roi_values = proj_2d[mask_2d > 0]

    nz, nx = proj_2d.shape  # (z, x)
    x_min, x_max = 0.0, nx * spacing_x
    z_min, z_max = 0.0, nz * spacing_z

    # Koordinatengitter für die Kontur in mm
    z_coords = np.linspace(z_min, z_max, nz)
    x_coords = np.linspace(x_min, x_max, nx)
    X, Z = np.meshgrid(x_coords, z_coords)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Overlay ---
    ax0 = axes[0]
    im = ax0.imshow(
        proj_2d,
        cmap="inferno",
        origin="lower",
        extent=[x_min, x_max, z_min, z_max],
        aspect="equal",
    )
    ax0.contour(X, Z, mask_2d, levels=[0.5], colors="cyan", linewidths=1)
    ax0.set_title(f"{organ} – {view}")
    ax0.set_xlabel("x [mm]")
    ax0.set_ylabel("z [mm]")

    # --- Histogramm ---
    ax1 = axes[1]
    ax1.hist(roi_values, bins=50)
    ax1.set_title(f"Histogram {organ} – {view}")
    ax1.set_xlabel("Counts")
    ax1.set_ylabel("Pixel-Anzahl")

    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{organ}_{view}_overlay_hist.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)




def run_roi_evaluation(
    pet_dicom_dir,
    mask_dir,
    ap_path,
    pa_path,
    output_csv
):
    # Serienlabel
    series_label = os.path.basename(os.path.normpath(pet_dicom_dir))

    print("[INFO] Lese PET DICOM (für Geometrie / Masken-Resampling)...")
    pet_img = read_dicom_series(pet_dicom_dir)
    spacing_x, spacing_y, spacing_z = pet_img.GetSpacing()

    # Output-Verzeichnis für Figuren
    fig_dir = None
    if output_csv is not None:
        fig_dir = os.path.join(os.path.dirname(output_csv), "figures")
        os.makedirs(fig_dir, exist_ok=True)

    # 2D-Projektionen laden
    print("[INFO] Lade 2D-Projektionen...")
    proj_ap = np.load(ap_path)   # (z,x)
    proj_pa = np.load(pa_path)   # (z,x)

    results = []

    for organ, fname in ROI_FILES.items():
        path_nii = os.path.join(mask_dir, fname)
        if not os.path.exists(path_nii):
            print(f"[WARNUNG] Maske fehlt: {organ}")
            continue

        print(f"[INFO] Verarbeite {organ}...")

        # Maske CT-Space -> PET-Space
        mask_ct = sitk.ReadImage(path_nii)
        mask_pet = resample_mask_to_pet(mask_ct, pet_img)
        mask_arr = sitk.GetArrayFromImage(mask_pet)  # (z,y,x)

        # 2D-Projektionsmasken
        mask2d_ap = project_mask_to_2d(mask_arr, "AP")  # (z,x)
        mask2d_pa = project_mask_to_2d(mask_arr, "PA")  # (z,x)

        # --- AP ---
        stats_ap = compute_2d_stats(proj_ap, mask2d_ap)
        stats_ap |= dict(organ=organ, series=series_label, view="AP")
        results.append(stats_ap)

        if fig_dir is not None:
            create_roi_visualizations(
                organ=organ,
                view="AP",
                proj_2d=proj_ap,
                mask_2d=mask2d_ap,
                out_dir=fig_dir,
                spacing_x=spacing_x,
                spacing_z=spacing_z,
            )

        # --- PA ---
        stats_pa = compute_2d_stats(proj_pa, mask2d_pa)
        stats_pa |= dict(organ=organ, series=series_label, view="PA")
        results.append(stats_pa)

        if fig_dir is not None:
            create_roi_visualizations(
                organ=organ,
                view="PA",
                proj_2d=proj_pa,
                mask_2d=mask2d_pa,
                out_dir=fig_dir,
                spacing_x=spacing_x,
                spacing_z=spacing_z,
            )

    # CSV schreiben
    if output_csv is not None:
        fieldnames = ["series", "organ", "view", "pixel_count", "sum", "mean", "min", "max"]

        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        readable_path = output_csv.replace(".tsv", "_readable.txt")
        write_readable_table(results, readable_path)
        print(f"[INFO] Tabellen gespeichert als:\n  - {output_csv}\n  - {readable_path}")
        if fig_dir is not None:
            print(f"[INFO] Figuren gespeichert in: {fig_dir}")


def main():
    parser = argparse.ArgumentParser(description="2D ROI Auswertung auf synthetischen Szintigrafien")
    parser.add_argument("--pet_dicom_dir", required=True,
                        help="DICOM-Ordner der PET-Serie (für Geometrie / Masken-Resampling)")
    parser.add_argument("--mask_dir", required=True,
                        help="Ordner mit VOI-NIfTIs (CT-Space, z.B. aus TotalSegmentator)")
    parser.add_argument("--ap", required=True, help="Pfad zu projection_AP.npy")
    parser.add_argument("--pa", required=True, help="Pfad zu projection_PA.npy")
    parser.add_argument("--output_csv", required=True,
                        help="Pfad zu TSV mit 2D-ROI-Statistiken")

    args = parser.parse_args()

    run_roi_evaluation(
        args.pet_dicom_dir,
        args.mask_dir,
        args.ap,
        args.pa,
        args.output_csv,
    )


if __name__ == "__main__":
    main()