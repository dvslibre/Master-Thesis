#!/usr/bin/env python

"""
stratos_applied.py

Überträgt die Stratos-Idee auf klinische PET/CT-Daten:

- Verwendung derselben Gamma-Kamera-Physik wie in scintigraphy_simulation.py
- Aufbau einer Systemmatrix A, deren Spalten den Beitrag einzelner Organe
  (3D-Masken) zur AP/PA-Projektion beschreiben
- Zusätzlich: automatische "others"-Maske (Body minus Union(Organe))
- Lösung des linearen Systems A x ≈ b (b = beobachtete AP/PA-Projektion),
  z.B. mittels NNLS, um Organaktivitäten zu schätzen

Beispielaufruf:

python stratos_applied.py \
  --pet_dicom_dir /home/mnguest12/projects/thesis/PETCT_Pipeline/data/example_01/PET_WB_TRUEX2I14S0MM_AC_0102 \
  --ct_dicom_dir  /home/mnguest12/projects/thesis/PETCT_Pipeline/data/example_01/CT_WB_5_0_B30F_0005 \
  --mask_dir      /home/mnguest12/projects/thesis/PETCT_Pipeline/data/example_01/results/seg/individual \
  --ap            /home/mnguest12/projects/thesis/PETCT_Pipeline/data/example_01/results/scintigraphy_sim/projection_AP.npy \
  --pa            /home/mnguest12/projects/thesis/PETCT_Pipeline/data/example_01/results/scintigraphy_sim/projection_PA.npy \
  --output_dir    /home/mnguest12/projects/thesis/PETCT_Pipeline/data/example_01/results/stratos_applied \
    --scatter_sigma_xy 2.0 \
  --coll_sigma_xy 0.0 \
  --collimator_kernel_mat /home/mnguest12/projects/thesis/PhantomGenerator/LEAP_Kernel.mat \
  --z0_slices 1 \
  --use_nnls
"""

import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.optimize import nnls

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Wir nutzen die Funktionen aus deinem bestehenden Skript
import scintigraphy_simulation as scinti


# Welche Organe (Masken) explizit einbezogen werden sollen
ROI_FILES: Dict[str, str] = {
    # bisherige Organe
    "kidney_left":   "kidney_left.nii.gz",
    "kidney_right":  "kidney_right.nii.gz",
    "liver":         "liver.nii.gz",
    "prostate":      "prostate.nii.gz",
    "small_bowel":   "small_bowel.nii.gz",
    "spleen":        "spleen.nii.gz",

    # neue Organe
    "brain":                 "brain.nii.gz",
    "colon":                 "colon.nii.gz",
    "duodenum":              "duodenum.nii.gz",
    "esophagus":             "esophagus.nii.gz",
    "gallbladder":           "gallbladder.nii.gz",
    "heart":                 "heart.nii.gz",
    "lung_lower_lobe_left":  "lung_lower_lobe_left.nii.gz",
    "lung_lower_lobe_right": "lung_lower_lobe_right.nii.gz",
    "lung_middle_lobe_right":"lung_middle_lobe_right.nii.gz",
    "lung_upper_lobe_left":  "lung_upper_lobe_left.nii.gz",
    "lung_upper_lobe_right": "lung_upper_lobe_right.nii.gz",
    "pancreas":              "pancreas.nii.gz",
    "spinal_cord":           "spinal_cord.nii.gz",
    "stomach":               "stomach.nii.gz",
    "thyroid_gland":         "thyroid_gland.nii.gz",
    "urinary_bladder":       "urinary_bladder.nii.gz",
}


# -------------------------------------------------------------
# Hilfsfunktionen: Body-Maske & Masken-Resampling
# -------------------------------------------------------------

def create_body_mask_from_ct(ct_resampled: sitk.Image, hu_threshold: float = -400.0) -> sitk.Image:
    """Erzeugt eine grobe Body-Maske (alles außer Luft) aus CT im PET-Space.

    HU > hu_threshold -> Körper
    HU <= hu_threshold -> Luft/Außen
    """
    ct_arr = sitk.GetArrayFromImage(ct_resampled).astype(np.float32)
    body = (ct_arr > hu_threshold).astype(np.uint8)
    body_img = sitk.GetImageFromArray(body)
    body_img.CopyInformation(ct_resampled)
    return body_img


def resample_mask_to_pet(mask_img: sitk.Image, pet_img: sitk.Image) -> sitk.Image:
    """Resample der VOI-Maske (CT-Space) in den PET-Space.
    Nearest-Neighbor-Interpolation, damit es eine saubere Binärmaske bleibt.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(pet_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(mask_img)


def build_group_mask_from_substrings(
    pet_img: sitk.Image,
    mask_dir: str,
    substrings: List[str],
    min_voxels: int = 10,
    label: str = "group",
) -> np.ndarray | None:
    """
    Sucht alle NIfTIs im mask_dir, deren Dateiname einen der substrings enthält,
    resampelt sie in den PET-Space und bildet die logische Union.

    Rückgabe:
      bool-Array (z,y,x) oder None, falls nichts gefunden / zu wenig Voxel.
    """
    pet_shape = sitk.GetArrayFromImage(pet_img).shape
    union = np.zeros(pet_shape, dtype=bool)
    count_files = 0
    total_vox = 0

    all_files = sorted(
        f for f in os.listdir(mask_dir)
        if (f.endswith(".nii") or f.endswith(".nii.gz"))
    )

    for fname in all_files:
        if not any(sub in fname for sub in substrings):
            continue

        mask_path = os.path.join(mask_dir, fname)
        try:
            mask_ct = sitk.ReadImage(mask_path)
        except Exception as e:
            print(f"[WARNUNG] Konnte {mask_path} nicht lesen: {e}")
            continue

        mask_pet = resample_mask_to_pet(mask_ct, pet_img)
        arr = sitk.GetArrayFromImage(mask_pet) > 0

        if arr.shape != pet_shape:
            print(f"[WARNUNG] {label}: Shape-Mismatch bei {fname}, ignoriere.")
            continue

        vox = int(arr.sum())
        if vox < min_voxels:
            print(f"[WARNUNG] {label}: {fname} hat nur {vox} Voxel, ignoriere.")
            continue

        union |= arr
        count_files += 1
        total_vox += vox

    if count_files == 0 or total_vox < min_voxels:
        print(f"[INFO] {label}: keine sinnvollen Masken gefunden.")
        return None

    print(f"[INFO] {label}: {count_files} Dateien, insgesamt {total_vox} Voxel")
    return union


def load_organ_masks_with_others(
    pet_img: sitk.Image,
    ct_resampled: sitk.Image,
    mask_dir: str,
    roi_files: Dict[str, str],
    hu_threshold_body: float = -400.0,
    min_voxels: int = 10,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Lädt explizite Organmasken (ROI_FILES), resampelt sie in PET-Space,
    erzeugt eine Body-Maske aus CT und daraus eine "others"-Maske
    (Body minus Union aller Organe + Knochen + Muskeln).

    Rückgabe:
      organ_masks_zyx: dict organ_name -> bool-Array (z,y,x), inkl. "bones", "muscles", "others"
      body_mask_zyx: bool-Array (z,y,x)
    """
    pet_shape = sitk.GetArrayFromImage(pet_img).shape  # (z,y,x)

    # Body-Maske aus CT
    print("[INFO] Erzeuge Body-Maske aus CT (HU-Threshold)...")
    body_img = create_body_mask_from_ct(ct_resampled, hu_threshold=hu_threshold_body)
    body_arr = sitk.GetArrayFromImage(body_img) > 0
    if not body_arr.any():
        raise RuntimeError("Body-Maske ist leer – HU-Threshold ggf. anpassen.")
    print(f"[INFO] Body-Maske: {int(body_arr.sum())} Voxel")

    organ_masks: Dict[str, np.ndarray] = {}
    union_organs = np.zeros_like(body_arr, dtype=bool)

    # 1) Explizite Organe aus ROI_FILES
    for organ, fname in roi_files.items():
        mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(mask_path):
            print(f"[WARNUNG] Maske für {organ} nicht gefunden: {mask_path}")
            continue

        print(f"[INFO] Lese/Resample Maske: {organ} ...")
        mask_ct = sitk.ReadImage(mask_path)
        mask_pet = resample_mask_to_pet(mask_ct, pet_img)
        arr = sitk.GetArrayFromImage(mask_pet) > 0

        if arr.shape != pet_shape:
            raise RuntimeError(
                f"Maskenshape passt nicht zum PET: {organ}, mask shape={arr.shape}, pet shape={pet_shape}"
            )

        voxels = int(arr.sum())
        if voxels < min_voxels:
            print(f"[WARNUNG] Maske {organ} hat nur {voxels} Voxel – ignoriere.")
            continue

        organ_masks[organ] = arr
        union_organs |= arr
        print(f"[INFO]   -> {organ}: {voxels} Voxel")

    if not organ_masks:
        raise RuntimeError("Keine gültigen Organmasken gefunden – Systemmatrix leer.")

    # 2) Knochen-Gruppe (bones)
    bone_substrings = [
        "vertebrae_", "rib_left_", "rib_right_",
        "femur_", "humerus_", "clavicula_",
        "scapula_", "sacrum", "skull",
        "sternum", "hip_", "costal_cartilages",
    ]
    bones_mask = build_group_mask_from_substrings(
        pet_img=pet_img,
        mask_dir=mask_dir,
        substrings=bone_substrings,
        min_voxels=min_voxels,
        label="bones",
    )
    if bones_mask is not None:
        organ_masks["bones"] = bones_mask
        union_organs |= bones_mask

    # 3) Muskel-Gruppe (muscles)
    muscle_substrings = [
        "gluteus_", "iliopsoas_", "autochthon_",
    ]
    muscles_mask = build_group_mask_from_substrings(
        pet_img=pet_img,
        mask_dir=mask_dir,
        substrings=muscle_substrings,
        min_voxels=min_voxels,
        label="muscles",
    )
    if muscles_mask is not None:
        organ_masks["muscles"] = muscles_mask
        union_organs |= muscles_mask

    # 4) Others = Body minus Union(Organe + bones + muscles)
    others = np.logical_and(body_arr, np.logical_not(union_organs))
    others_vox = int(others.sum())
    print(f"[INFO] 'others'-Maske: {others_vox} Voxel")
    if others_vox > 0:
        organ_masks["others"] = others

    return organ_masks, body_arr



# -------------------------------------------------------------
# Systemmatrix A
# -------------------------------------------------------------

def build_system_matrix(
    pet_img: sitk.Image,
    mu_zyx: np.ndarray,
    organ_masks_zyx: Dict[str, np.ndarray],
    scatter_sigma_xy: float,
    coll_sigma_xy: float,
    use_scatter: bool,
    use_attenuation: bool,
    use_collimator: bool,
    coll_kernel: np.ndarray | None = None,
    z0_slices: int = 0,
) -> Tuple[np.ndarray, List[str]]:
    """Baut die Systemmatrix A:

    - jede Spalte entspricht einer 3D-Organmaske (im PET-Space),
      die mit 1 gefüllt (Aktivität = 1) durch das Gamma-Kamera-Modell
      geschickt wird
    - AP und PA werden jeweils geflattet und aneinandergehängt

    A hat die Form: (N_pixel_AP + N_pixel_PA, N_organe)
    """
    spacing_x, spacing_y, spacing_z = pet_img.GetSpacing()

    organs: List[str] = []
    columns: List[np.ndarray] = []

    proj_shape = None

    for organ, mask_arr_zyx in organ_masks_zyx.items():
        print(f"[INFO] Systemmatrix: Organ {organ} ...")
        mask_bool = mask_arr_zyx.astype(bool)
        if not mask_bool.any():
            print(f"[WARNUNG] Maske für {organ} ist im PET-Space leer.")
            continue

        # Aktivität = 1 innerhalb der Maske, 0 sonst
        act_zyx = mask_bool.astype(np.float32)

        proj_ap, proj_pa = scinti.gamma_camera_forward_zyx(
            act_zyx=act_zyx,
            mu_zyx=mu_zyx,
            # unsere Parameter heißen hier *_xy, scinti erwartet *_xz → einfach gemappt:
            scatter_sigma_xz=scatter_sigma_xy,
            coll_sigma_xz=coll_sigma_xy,
            use_scatter=use_scatter,
            use_attenuation=use_attenuation,
            use_collimator=use_collimator,
            spacing_y_mm=spacing_y,
            coll_kernel=coll_kernel,
            z0_slices=z0_slices,
            use_fft_conv=True,  # oder scinti.USE_FFT_CONV, falls definiert
        )

        if proj_shape is None:
            proj_shape = proj_ap.shape
            print(f"[INFO]   -> Projektion shape: {proj_shape}")
        else:
            if proj_ap.shape != proj_shape or proj_pa.shape != proj_shape:
                raise RuntimeError(
                    f"Inkonstistente Projektionsshape bei Organ {organ}: {proj_ap.shape} vs {proj_shape}"
                )

        col = np.concatenate([proj_ap.ravel(), proj_pa.ravel()], axis=0)
        columns.append(col)
        organs.append(organ)

    if not columns:
        raise RuntimeError("Keine Spalten in der Systemmatrix erzeugt – alles leer.")

    A = np.stack(columns, axis=1)  # shape (N_pixels*2, N_organe)
    print(f"[INFO] Systemmatrix A aufgebaut: shape = {A.shape}")
    return A, organs


# -------------------------------------------------------------
# Plot-Helper (für rekonstruierte Projektionen)
# -------------------------------------------------------------

def save_projection_png_simple(proj: np.ndarray, path: str, title: str,
                               spacing_x: float, spacing_z: float):
    """Speichert ein 2D-Projektionsbild mit physikalischer Skalierung."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nz, nx = proj.shape
    x_min, x_max = 0.0, nx * spacing_x
    z_min, z_max = 0.0, nz * spacing_z

    plt.figure(figsize=(5, 7))
    im = plt.imshow(
        proj,
        cmap="inferno",
        origin="lower",
        extent=[x_min, x_max, z_min, z_max],
        aspect="equal",
    )
    plt.title(title)
    plt.xlabel("x [mm]")
    plt.ylabel("z [mm]")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# -------------------------------------------------------------
# Hauptfunktion: Stratos-Ansatz auf klinische Daten
# -------------------------------------------------------------

def run_stratos_applied(
    pet_dicom_dir: str,
    ct_dicom_dir: str,
    mask_dir: str,
    ap_path: str,
    pa_path: str,
    output_dir: str,
    scatter_sigma_xy: float = 2.0,
    coll_sigma_xy: float = 2.0,
    use_scatter: bool = True,
    use_attenuation: bool = True,
    use_collimator: bool = True,
    use_nnls: bool = True,
    hu_threshold_body: float = -400.0,
    collimator_kernel_mat: str | None = None,
    z0_slices: int = 0,
):
    os.makedirs(output_dir, exist_ok=True)
    # ggf. Kollimatorkernel aus .mat laden (LEAP-Kernel)
    coll_kernel = None
    if collimator_kernel_mat is not None:
        print(f"[INFO] Lade Kollimatorkernel aus: {collimator_kernel_mat}")
        coll_kernel = scinti.load_leap_kernel(collimator_kernel_mat)
        print(f"[INFO] Kollimatorkernel-Shape: {coll_kernel.shape}")

    # --- PET/CT einlesen und µ-Map erzeugen ---
    print(f"[INFO] Lese PET-DICOM aus: {pet_dicom_dir}")
    pet_img = scinti.read_dicom_series(pet_dicom_dir)
    spacing_x, spacing_y, spacing_z = pet_img.GetSpacing()

    print(f"[INFO] Lese CT-DICOM aus:  {ct_dicom_dir}")
    ct_img = scinti.read_dicom_series(ct_dicom_dir)

    print("[INFO] Resample CT -> PET-Space ...")
    ct_resampled = scinti.resample_to_reference(
        ct_img,
        reference=pet_img,
        interpolator=sitk.sitkLinear,
        default_value=-1024.0,
    )

    ct_arr_zyx = sitk.GetArrayFromImage(ct_resampled).astype(np.float32)
    print("[INFO] Erzeuge µ-Map aus CT-HU ...")
    mu_zyx = scinti.ct_hu_to_mu(ct_arr_zyx, energy_keV=140.0)

    # --- Organmasken + Body + Others laden ---
    print(f"[INFO] Lade Organmasken aus: {mask_dir}")
    organ_masks_zyx, body_mask_zyx = load_organ_masks_with_others(
        pet_img=pet_img,
        ct_resampled=ct_resampled,
        mask_dir=mask_dir,
        roi_files=ROI_FILES,
        hu_threshold_body=hu_threshold_body,
    )
    print(f"[INFO] Gruppen (inkl. 'others'): {list(organ_masks_zyx.keys())}")

    # --- Beobachtete Projektionen laden ---
    print(f"[INFO] Lade beobachtete Projektionen:\n  AP = {ap_path}\n  PA = {pa_path}")
    proj_ap_meas = np.load(ap_path)  # (z,x)
    proj_pa_meas = np.load(pa_path)  # (z,x)

    if proj_ap_meas.shape != proj_pa_meas.shape:
        raise RuntimeError("AP- und PA-Projektion haben unterschiedliche Shapes.")

    b = np.concatenate([proj_ap_meas.ravel(), proj_pa_meas.ravel()], axis=0)

    # --- Systemmatrix A aus Organmasken (+others) ---
    A, organs = build_system_matrix(
        pet_img=pet_img,
        mu_zyx=mu_zyx,
        organ_masks_zyx=organ_masks_zyx,
        scatter_sigma_xy=scatter_sigma_xy,
        coll_sigma_xy=coll_sigma_xy,
        use_scatter=use_scatter,
        use_attenuation=use_attenuation,
        use_collimator=use_collimator,
        coll_kernel=coll_kernel,
        z0_slices=z0_slices,
    )

    # --- Lösen: A x ≈ b ---
    print(f"[INFO] Löse System A x ≈ b mit {'NNLS' if use_nnls else 'Least Squares'} ...")

    # leichte Skalierung für numerische Stabilität
    scale = 1.0 / max(np.max(A), np.max(b), 1e-6)
    A_scaled = A * scale
    b_scaled = b * scale

    if use_nnls:
        x_est, nnls_resid = nnls(A_scaled, b_scaled)
    else:
        x_est, *_ = np.linalg.lstsq(A_scaled, b_scaled, rcond=None)
        nnls_resid = np.nan

    # --- Reprojektion aus geschätzten Organaktivitäten ---
    b_hat = A @ x_est
    n_pix_ap = proj_ap_meas.size
    proj_ap_rec = b_hat[:n_pix_ap].reshape(proj_ap_meas.shape)
    proj_pa_rec = b_hat[n_pix_ap:].reshape(proj_pa_meas.shape)

    # --- Fehlermaße (global) ---
    rmse = float(np.sqrt(np.mean((b - b_hat) ** 2)))
    rrmse = float(np.linalg.norm(b - b_hat) / max(np.linalg.norm(b), 1e-12))
    print(f"[INFO] RMSE(b, b_hat)  = {rmse:.4g}")
    print(f"[INFO] RRMSE(b, b_hat) = {rrmse:.4g}")

    # --- Lokaler RRMSE nur in Body-Region (AP+PA) ---
    body_ap = np.any(body_mask_zyx, axis=1)  # (z,x)
    body_pa = body_ap.copy()
    body_mask_2d_flat = np.concatenate([body_ap.ravel(), body_pa.ravel()], axis=0)

    b_body = b[body_mask_2d_flat]
    b_hat_body = b_hat[body_mask_2d_flat]
    rrmse_body = float(
        np.linalg.norm(b_body - b_hat_body) / max(np.linalg.norm(b_body), 1e-12)
    )
    print(f"[INFO] RRMSE nur in Body-Region = {rrmse_body:.4g}")

    # --- Ergebnisse speichern ---
    # 1) Organaktivitäten als TSV
    organ_stats_path = os.path.join(output_dir, "organ_activities.tsv")
    with open(organ_stats_path, "w") as f:
        f.write("organ\tactivity_est\n")
        for organ, val in zip(organs, x_est):
            f.write(f"{organ}\t{val:.6g}\n")
    print(f"[INFO] Organaktivitäten gespeichert in: {organ_stats_path}")

    # 2) Reprojezierte Bilder als NPY + PNG
    np.save(os.path.join(output_dir, "proj_AP_rec.npy"), proj_ap_rec.astype(np.float32))
    np.save(os.path.join(output_dir, "proj_PA_rec.npy"), proj_pa_rec.astype(np.float32))

    save_projection_png_simple(
        proj_ap_rec,
        os.path.join(output_dir, "proj_AP_rec.png"),
        "HybridSTRATOS reconstruction – AP",
        spacing_x=spacing_x,
        spacing_z=spacing_z,
    )
    save_projection_png_simple(
        proj_pa_rec,
        os.path.join(output_dir, "proj_PA_rec.png"),
        "HybridSTRATOS reconstruction – PA",
        spacing_x=spacing_x,
        spacing_z=spacing_z,
    )

    # 3) Kurze Textzusammenfassung
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Stratos-Ansatz auf klinische Daten\n")
        f.write(f"RMSE(b, b_hat)           = {rmse:.6g}\n")
        f.write(f"RRMSE(b, b_hat)          = {rrmse:.6g}\n")
        f.write(f"RRMSE (Body-Region)      = {rrmse_body:.6g}\n")
        f.write(f"Solver: {'NNLS' if use_nnls else 'Least Squares'}\n")
        f.write("Organe & Aktivitäten:\n")
        for organ, val in zip(organs, x_est):
            f.write(f"  {organ:20s}: {val:.6g}\n")

    print(f"[INFO] Zusammenfassung gespeichert in: {summary_path}")

    return {
        "organs": organs,
        "x_est": x_est,
        "rmse": rmse,
        "rrmse": rrmse,
        "rrmse_body": rrmse_body,
        "proj_AP_rec": proj_ap_rec,
        "proj_PA_rec": proj_pa_rec,
    }


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stratos-Ansatz auf klinische PET/CT-Daten (Organbasierte Szinti-Faktorisierung mit 'others')."
    )
    parser.add_argument("--pet_dicom_dir", required=True,
                        help="DICOM-Ordner der PET-Serie (Aktivitätsvolumen).")
    parser.add_argument("--ct_dicom_dir", required=True,
                        help="DICOM-Ordner der CT-Serie (für µ-Map und Body-Maske).")
    parser.add_argument("--mask_dir", required=True,
                        help="Ordner mit VOI-NIfTIs (CT-Space, z.B. aus TotalSegmentator).")
    parser.add_argument("--ap", required=True,
                        help="Pfad zu projection_AP.npy (beobachtete AP-Szintigrafie).")
    parser.add_argument("--pa", required=True,
                        help="Pfad zu projection_PA.npy (beobachtete PA-Szintigrafie).")
    parser.add_argument("--output_dir", required=True,
                        help="Ausgabeordner für A, x, rekonstruierte Projektionen usw.")
    parser.add_argument("--scatter_sigma_xy", type=float, default=2.0,
                        help="Sigma der Gauß-Streuung in (z,x)-Ebene (Pixel).")
    parser.add_argument("--coll_sigma_xy", type=float, default=2.0,
                        help="Sigma der Kollimatorunschärfe in (z,x)-Ebene (Pixel).")
    parser.add_argument("--no_scatter", action="store_true",
                        help="Streuung deaktivieren.")
    parser.add_argument("--no_attenuation", action="store_true",
                        help="Abschwächung deaktivieren.")
    parser.add_argument("--no_collimator", action="store_true",
                        help="Kollimatorunschärfe deaktivieren.")
    parser.add_argument("--use_nnls", action="store_true",
                        help="Nichtnegative Lösung (NNLS) statt gewöhnlicher Least Squares.")
    parser.add_argument("--hu_threshold_body", type=float, default=-400.0,
                        help="HU-Schwelle für Body-Maske (Standard: -400 HU).")
    parser.add_argument(
        "--collimator_kernel_mat",
        type=str,
        default=None,
        help="Pfad zu einer .mat-Datei mit Kollimatorkernel (z.B. LEAP_Kernel.mat). "
             "Wenn gesetzt, wird dieser Kernel statt eines Gaußfilters verwendet.",
    )
    parser.add_argument(
        "--z0_slices",
        type=int,
        default=0,
        help="Anzahl der Slices (in y-Richtung), vor denen keine Kollimatorunschärfe angewandt wird.",
    )

    args = parser.parse_args()

    run_stratos_applied(
        pet_dicom_dir=args.pet_dicom_dir,
        ct_dicom_dir=args.ct_dicom_dir,
        mask_dir=args.mask_dir,
        ap_path=args.ap,
        pa_path=args.pa,
        output_dir=args.output_dir,
        scatter_sigma_xy=args.scatter_sigma_xy,
        coll_sigma_xy=args.coll_sigma_xy,
        use_scatter=not args.no_scatter,
        use_attenuation=not args.no_attenuation,
        use_collimator=not args.no_collimator,
        use_nnls=args.use_nnls,
        hu_threshold_body=args.hu_threshold_body,
        collimator_kernel_mat=args.collimator_kernel_mat,
        z0_slices=args.z0_slices,
    )


if __name__ == "__main__":
    main()
