#!/usr/bin/env python

"""
scintigraphy_simulation.py

Simuliert planare Szintigrafie-Projektionen (AP/PA) aus einem SPECT-Volumen,
unter Verwendung einer CT-Serie als Abschwächungskarte.

- SPECT-DICOM -> Aktivitätsvolumen (counts o.ä.)
- CT-DICOM  -> HU-Volumen -> µ-Map (140 keV, stark vereinfacht)
- Gamma-Kamera-Vorwärtsmodell:
    (1) xy-Gauß-Streuung pro Schicht (optional)
    (2) exponentielle Abschwächung entlang y (optional)
    (3) xy-Gauß-Kollimatorunschärfe pro Schicht (optional)
    (4) Projektion = Summe über z

AP/PA werden separat berechnet (einmal „von vorne“, einmal „von hinten“).

Das ist bewusst ein vereinfachtes physikalisches Modell, reicht aber gut
für synthetische Szintigrafien aus klinischen SPECT/CT-Daten.

Aufruf:

python scintigraphy_simulation.py \
  --spect_dicom_dir /home/mnguest12/projects/thesis/SPECTCT_Pipeline/data/example_01/spect/LU-177-SPECT-CT-DOSIMETRIE-RECON-AC \
  --ct_dicom_dir  /home/mnguest12/projects/thesis/SPECTCT_Pipeline/data/example_01/ct/3D-ABDROUTINE-1.5-B31S \
  --output_dir    /home/mnguest12/projects/thesis/SPECTCT_Pipeline/data/example_01/results/scintigraphy_sim \
  --scatter_sigma_xz 1.0 \
  --coll_sigma_xz 1.0 \
  --collimator_kernel_mat /home/mnguest12/projects/thesis/PhantomGenerator/LEAP_Kernel.mat \
  --z0_slices 1

"""

import os
import argparse
from typing import Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.io import loadmat
from scipy.signal import fftconvolve, convolve2d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Soll für die Kollimator-Faltung FFT benutzt werden?
USE_FFT_CONV = True

# -------------------------------------------------------------
# DICOM I/O
# -------------------------------------------------------------
def read_dicom_series(dicom_dir: str) -> sitk.Image:
    """
    Liest eine DICOM-Serie (z.B. SPECT oder CT) mit SimpleITK ein.
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


def resample_to_reference(
    img: sitk.Image,
    reference: sitk.Image,
    interpolator=sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Resample 'img' in den Raum von 'reference' (gleiche Größe, Spacing, Origin, Direction).
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(img)


def load_leap_kernel(mat_path: str, key: str = None) -> np.ndarray:
    """
    Lädt den Kollimatorkern aus einer .mat-Datei.
    Erwartet ein Array der Form (nz_kernel, nx_kernel, n_depth).

    - mat_path: Pfad zu LEAP_Kernel.mat
    - key: Name der Variable im .mat (falls bekannt).
           Wenn None, wird das erste nicht-Meta-Array genommen.
    """
    m = loadmat(mat_path)
    if key is None:
        # erstes "normales" Array nehmen (alles, was nicht __header__/__version__/__globals__ ist)
        data_keys = [k for k in m.keys() if not k.startswith("__")]
        if not data_keys:
            raise RuntimeError(f"Keine nutzbare Variable in {mat_path} gefunden.")
        key = data_keys[0]

    kernel_mat = np.asarray(m[key], dtype=np.float32)

    # Optional: jede Tiefenscheibe auf Summe 1 normalisieren
    if kernel_mat.ndim == 3:
        for i in range(kernel_mat.shape[2]):
            s = kernel_mat[:, :, i].sum()
            if s > 0:
                kernel_mat[:, :, i] /= s
    else:
        raise ValueError(f"Erwartete 3D-Kernelmatrix, bekommen: shape={kernel_mat.shape}")

    return kernel_mat


def collapse_4d_to_3d(img: sitk.Image, mode: str = "sum") -> sitk.Image:
    """
    Reduziert ein 4D-Bild (t,z,y,x) auf 3D (z,y,x), indem über t aggregiert wird.
    Spacing/Origin/Direction werden korrekt auf 3D reduziert.
    """
    if img.GetDimension() == 3:
        return img
    if img.GetDimension() != 4:
        raise ValueError(f"Unexpected dimension: {img.GetDimension()}")

    arr = sitk.GetArrayFromImage(img)  # (t,z,y,x)

    if mode == "sum":
        arr3 = arr.sum(axis=0)
    elif mode == "mean":
        arr3 = arr.mean(axis=0)
    elif mode == "first":
        arr3 = arr[0]
    else:
        raise ValueError("mode must be 'sum', 'mean', or 'first'.")

    img3 = sitk.GetImageFromArray(arr3)

    spacing4 = img.GetSpacing()
    origin4 = img.GetOrigin()
    direction4 = img.GetDirection()

    img3.SetSpacing(spacing4[:3])
    img3.SetOrigin(origin4[:3])
    img3.SetDirection(direction4[:9])  # oberes 3x3 aus 4x4

    return img3


def reorient_to_lps(img: sitk.Image) -> sitk.Image:
    """
    Reorientiert ein 3D-Bild auf DICOM-Standard LPS, damit Achsen/Spacing konsistent sind.
    """
    if img.GetDimension() != 3:
        raise ValueError(f"reorient_to_lps erwartet 3D, bekam {img.GetDimension()}D")
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("LPS")
    return orienter.Execute(img)


def fix_direction_if_degenerate(img: sitk.Image, name: str = "image", det_threshold: float = 0.5):
    """
    Falls die Directions-Matrix degeneriert ist (z.B. det≈0), ersetze sie durch Identität.
    """
    direction = img.GetDirection()
    if len(direction) != 9:
        return img
    det = np.linalg.det(np.reshape(direction, (3, 3)))
    if not np.isfinite(det) or abs(det) < det_threshold:
        print(f"[WARN] {name}: Direction determinant {det:.6f} < {det_threshold}; setze auf Identität.")
        img.SetDirection((1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0))
    return img


def log_image_info(img: sitk.Image, name: str):
    size = img.GetSize()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    print(f"[DEBUG] {name}: dim={img.GetDimension()} size={size} spacing={spacing} origin={origin} direction={direction}")
    print(f"        direction matrix determinant ~ {np.linalg.det(np.reshape(direction, (3,3))):.6f}")
    try:
        corners = {
            "idx(0,0,0)": (0, 0, 0),
            "idx(xmax,0,0)": (size[0]-1, 0, 0),
            "idx(0,ymax,0)": (0, size[1]-1, 0),
            "idx(0,0,zmax)": (0, 0, size[2]-1),
        }
        for label, idx in corners.items():
            pt = img.TransformIndexToPhysicalPoint(idx)
            print(f"        {label} -> phys {pt}")
        # Gesamt-Extents über alle 8 Ecken
        all_corners = [
            (0, 0, 0),
            (size[0]-1, 0, 0),
            (0, size[1]-1, 0),
            (0, 0, size[2]-1),
            (size[0]-1, size[1]-1, 0),
            (size[0]-1, 0, size[2]-1),
            (0, size[1]-1, size[2]-1),
            (size[0]-1, size[1]-1, size[2]-1),
        ]
        pts = np.array([img.TransformIndexToPhysicalPoint(c) for c in all_corners])
        xyz_min = pts.min(axis=0)
        xyz_max = pts.max(axis=0)
        print(f"        phys bbox min {tuple(xyz_min)} max {tuple(xyz_max)}")
    except Exception as e:
        print(f"        [WARN] phys point transform failed: {e}")


def log_array_stats(name: str, arr: np.ndarray):
    print(f"[DEBUG] {name}: shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()} mean={arr.mean()} sum={arr.sum()}")


def log_center_of_mass(img: sitk.Image, arr_zyx: np.ndarray, name: str):
    # Falls negative Werte vorhanden sind (z.B. CT), verschieben wir auf >=0.
    arr_nonneg = arr_zyx - arr_zyx.min()
    if arr_nonneg.sum() == 0:
        print(f"[DEBUG] {name} center of mass: skipped (sum==0)")
        return
    com_zyx = center_of_mass(arr_nonneg)
    com_phys = None
    try:
        # TransformContinuousIndex erwartet (x,y,z)
        com_idx_xyz = (com_zyx[2], com_zyx[1], com_zyx[0])
        com_phys = img.TransformContinuousIndexToPhysicalPoint(com_idx_xyz)
    except Exception:
        pass
    print(f"[DEBUG] {name} center of mass (zyx idx): {com_zyx}")
    if com_phys is not None:
        print(f"        {name} center of mass (phys): {com_phys}")


def rigid_register_ct_to_spect(ct_img: sitk.Image, spect_img: sitk.Image, debug: bool = False) -> sitk.Transform:
    """
    Führe starre Registrierung (Euler3D) CT -> SPECT durch (Mattes MI).
    """
    initial_tx = sitk.CenteredTransformInitializer(
        spect_img,
        ct_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetInitialTransform(initial_tx, inPlace=False)

    final_tx = reg.Execute(spect_img, ct_img)
    if debug:
        print("[DEBUG] Registrierung CT->SPECT beendet.")
        print("        Final metric:", reg.GetMetricValue())
        print("        Final parameters:", list(final_tx.GetParameters()))
    return final_tx


def save_quickcheck_overlays(spect_img: sitk.Image, ct_img: sitk.Image, out_dir: str):
    """
    Speichert einfache Overlay-PNGs (axial, coronal, sagittal) zum schnellen Sichtcheck.
    """
    os.makedirs(out_dir, exist_ok=True)
    spect_arr = sitk.GetArrayFromImage(spect_img)  # z,y,x
    ct_arr = sitk.GetArrayFromImage(ct_img)
    nz, ny, nx = spect_arr.shape
    slices = {
        "axial_zmid": ("axial", nz // 2),
        "coronal_ymid": ("coronal", ny // 2),
        "sagittal_xmid": ("sagittal", nx // 2),
    }

    for name, (plane, idx) in slices.items():
        if plane == "axial":
            ct_slice = np.flipud(ct_arr[idx, :, :])
            spect_slice = np.flipud(spect_arr[idx, :, :])
        elif plane == "coronal":
            ct_slice = np.flipud(ct_arr[:, idx, :])
            spect_slice = np.flipud(spect_arr[:, idx, :])
        elif plane == "sagittal":
            ct_slice = np.flipud(ct_arr[:, :, idx])
            spect_slice = np.flipud(spect_arr[:, :, idx])
        else:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(ct_slice, cmap="gray", interpolation="nearest")
        ax.imshow(spect_slice, cmap="hot", alpha=0.5, interpolation="nearest")
        ax.set_title(f"{plane} slice {idx}")
        ax.axis("off")
        out_path = os.path.join(out_dir, f"quickcheck_{name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# -------------------------------------------------------------
# CT -> µ-Map (stark vereinfacht)
# -------------------------------------------------------------
def ct_hu_to_mu(ct_hu: np.ndarray, energy_keV: float = 140.0) -> np.ndarray:
    """
    Sehr einfache HU -> µ-Abbildung.

    Annahme:
      - Wasser bei 140 keV: µ_water ≈ 0.15 1/cm (ca.)
      - daraus: µ = µ_water * (HU/1000 + 1)

    """
    mu_water = 0.15  # 1/cm, nur als Maßstab
    rho_rel = (ct_hu / 1000.0) + 1.0
    mu = mu_water * rho_rel
    mu = np.clip(mu, 0.0, None)  # keine negativen µ
    return mu.astype(np.float32)


# -------------------------------------------------------------
# Gamma-Kamera-Vorwärtsmodell (zyx, AP/PA = koronal)
# -------------------------------------------------------------

def _forward_single_view_zyx(
    act_zyx: np.ndarray,
    mu_zyx: np.ndarray,
    scatter_sigma_xz: float,
    coll_sigma_xz: float,
    use_scatter: bool,
    use_attenuation: bool,
    use_collimator: bool,
    spacing_y_mm: float,
    coll_kernel: np.ndarray | None = None,
    z0_slices: int = 0,
    use_fft_conv: bool = True,
) -> np.ndarray:
    """
    Vorwärtsmodell für EINE Blickrichtung entlang der y-Achse (AP ODER PA).

    Input-Shape: (nz, ny, nx) = (z, y, x)
    - Scatter/Kollimator: Gauß in der Bildebene (z,x) für jede y-Schicht
    - Abschwächung: exp(-∫ µ dy) entlang y
    - Projektion: Summe über y -> Bild (nz, nx) = koronal
    """
    assert act_zyx.shape == mu_zyx.shape
    nz, ny, nx = act_zyx.shape

    # (1) Streuung in der Detektorebene (z,x) pro y-Schicht
    if use_scatter and scatter_sigma_xz > 0:
        act_sc = np.empty_like(act_zyx, dtype=np.float32)           # neues Array für gestreute Werte
        for j in range(ny):
            act_sc[:, j, :] = gaussian_filter(
                act_zyx[:, j, :],
                sigma=scatter_sigma_xz,
                mode="nearest",                                     # verhindert Rand-Artefakte
            )
    else:
        act_sc = act_zyx.astype(np.float32, copy=True)

    # (2) Abschwächung entlang y (µ in 1/cm, dy in cm)
    if use_attenuation:
        spacing_y_cm = spacing_y_mm / 10.0
        mu_cum = np.cumsum(mu_zyx * spacing_y_cm, axis=1)           # kumulative Absorption entlang der y-Achse
        vol_atn = act_sc * np.exp(-mu_cum)                          # quasi Anwendung Lamber-Beer-Gleichung
    else:
        vol_atn = act_sc

    # (3) Kollimatorunschärfe / -PSF
    if use_collimator:
        # Fall A: echter LEAP-Kernel vorhanden → depth-dependent PSF
        if coll_kernel is not None:
            vol_coll = np.empty_like(vol_atn, dtype=np.float32)
            nz, ny, nx = vol_atn.shape
            conv2 = fftconvolve if use_fft_conv else convolve2d
            n_kernels = coll_kernel.shape[2]

            # wir gehen über die Projektionsrichtung (y) und
            # falten je y-Schicht in der Detektorebene (z,x)
            for j in range(ny):
                if j <= z0_slices:
                    # vor z0: noch keine Kollimatorunschärfe anwenden
                    vol_coll[:, j, :] = vol_atn[:, j, :]
                else:
                    kk = min(j - z0_slices, n_kernels - 1)
                    K = coll_kernel[:, :, kk]
                    vol_coll[:, j, :] = conv2(
                        vol_atn[:, j, :],   # shape (nz, nx)
                        K,                  # shape (kz, kx)
                        mode="same",
                    ).astype(np.float32)
        # Fall B: kein Kernel → fallback auf Gauß, wie bisher
        elif coll_sigma_xz > 0:
            vol_coll = np.empty_like(vol_atn, dtype=np.float32)
            for j in range(ny):
                vol_coll[:, j, :] = gaussian_filter(
                    vol_atn[:, j, :],
                    sigma=coll_sigma_xz,
                    mode="nearest",
                )
        else:
            vol_coll = vol_atn
    else:
        vol_coll = vol_atn

    # (4) Projektion: Summe über y -> koronales Bild (z,x)
    proj = np.sum(vol_coll, axis=1)  # shape (nz, nx)
    return proj


def gamma_camera_forward_zyx(
    act_zyx: np.ndarray,
    mu_zyx: np.ndarray,
    scatter_sigma_xz: float = 2.0,
    coll_sigma_xz: float = 2.0,
    use_scatter: bool = True,
    use_attenuation: bool = True,
    use_collimator: bool = True,
    spacing_y_mm: float = 4.0,
    coll_kernel: np.ndarray | None = None,
    z0_slices: int = 0,
    use_fft_conv: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    AP/PA-Projektion aus Volumen (nz, ny, nx) mit Integration entlang y.
    Ergebnis: proj_AP, proj_PA jeweils (nz, nx) = koronal.
    """
    assert act_zyx.shape == mu_zyx.shape

    # In LPS-Koordinaten ist +y posterior. Für eine AP-Projektion (Kamera anterior)
    # integrieren wir entlang -y ⇒ reverse der y-Achse.
    proj_ap = _forward_single_view_zyx(
        act_zyx=act_zyx[:, ::-1, :],
        mu_zyx=mu_zyx[:, ::-1, :],
        scatter_sigma_xz=scatter_sigma_xz,
        coll_sigma_xz=coll_sigma_xz,
        use_scatter=use_scatter,
        use_attenuation=use_attenuation,
        use_collimator=use_collimator,
        spacing_y_mm=spacing_y_mm,
        coll_kernel=coll_kernel,
        z0_slices=z0_slices,
        use_fft_conv=use_fft_conv,
    )

    # PA (Kamera posterior) integriert entlang +y ⇒ Originalreihenfolge.
    proj_pa = _forward_single_view_zyx(
        act_zyx=act_zyx,
        mu_zyx=mu_zyx,
        scatter_sigma_xz=scatter_sigma_xz,
        coll_sigma_xz=coll_sigma_xz,
        use_scatter=use_scatter,
        use_attenuation=use_attenuation,
        use_collimator=use_collimator,
        spacing_y_mm=spacing_y_mm,
        coll_kernel=coll_kernel,
        z0_slices=z0_slices,
        use_fft_conv=use_fft_conv,
    )

    return proj_ap, proj_pa


# -------------------------------------------------------------
# Plot-Helper
# -------------------------------------------------------------
def save_projection_png(proj: np.ndarray, path: str, title: str,
                        spacing_x: float, spacing_z: float):
    """
    proj hat Shape (nz, nx) = (z, x) im SPECT-Space.
    Darstellung wie Quickcheck: einmal flipud, origin-Default ('upper').
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    nz, nx = proj.shape  # (z, x)

    # Orientierung wie im Quickcheck: flipud
    proj_disp = np.flipud(proj)

    x_min, x_max = 0.0, nx * spacing_x
    z_min, z_max = 0.0, nz * spacing_z

    plt.figure(figsize=(5, 7))
    im = plt.imshow(
        proj_disp,
        cmap="inferno",
        interpolation="nearest",
        # origin NICHT setzen -> Default 'upper' wie im Quickcheck
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
# Haupt-Pipeline (Baseline-Version: CT -> SPECT-Space)
# -------------------------------------------------------------
def run_scintigraphy_simulation(
    spect_dicom_dir: str,
    ct_dicom_dir: str,
    output_dir: str,
    scatter_sigma_xz: float = 2.0,
    coll_sigma_xz: float = 2.0,
    use_scatter: bool = True,
    use_attenuation: bool = True,
    use_collimator: bool = True,
    collimator_kernel_mat: str | None = None,
    z0_slices: int = 0,
    debug: bool = False,
    save_overlays: bool = False,
    com_shift_ct: bool = False,
    rigid_register_ct: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    # ggf. Kollimatorkernel laden
    coll_kernel = None
    if collimator_kernel_mat is not None:
        print(f"[INFO] Lade Kollimatorkernel aus: {collimator_kernel_mat}")
        coll_kernel = load_leap_kernel(collimator_kernel_mat)
        print(f"[INFO] Kollimatorkernel-Shape: {coll_kernel.shape}")

    print(f"[INFO] Lese SPECT-DICOM aus: {spect_dicom_dir}")
    spect_img = read_dicom_series(spect_dicom_dir)
    if spect_img.GetDimension() == 4:
        print("[INFO] SPECT ist 4D – reduziere auf 3D (sum).")
        spect_img = collapse_4d_to_3d(spect_img, mode="sum")
    # SPECT belassen, aber Direction prüfen/korrigieren (manche Header enthalten degenerierte Matrix)
    spect_img = fix_direction_if_degenerate(spect_img, name="SPECT")
    if debug:
        log_image_info(spect_img, "SPECT (nach Load/Check)")
    # SPECT-Voxelabstände (SimpleITK: (x,y,z) in mm)
    spacing = spect_img.GetSpacing()
    spacing_x, spacing_y, spacing_z = spacing[0], spacing[1], spacing[2]

    print(f"[INFO] Lese CT-DICOM aus:  {ct_dicom_dir}")
    ct_img = read_dicom_series(ct_dicom_dir)
    ct_img = reorient_to_lps(ct_img)
    ct_img = fix_direction_if_degenerate(ct_img, name="CT")
    if debug:
        log_image_info(ct_img, "CT (nach Reorient)")

    print("[INFO] Resample CT -> SPECT-Space ...")
    if rigid_register_ct:
        tx = rigid_register_ct_to_spect(ct_img, spect_img, debug=debug)
    elif com_shift_ct:
        # Grobe Translation per COM-Verschiebung CT->SPECT
        ct_arr_tmp = sitk.GetArrayFromImage(ct_img)
        spect_arr_tmp = sitk.GetArrayFromImage(spect_img)
        com_ct = center_of_mass(ct_arr_tmp - ct_arr_tmp.min())
        com_spect = center_of_mass(spect_arr_tmp - spect_arr_tmp.min())
        com_ct_phys = ct_img.TransformContinuousIndexToPhysicalPoint((com_ct[2], com_ct[1], com_ct[0]))
        com_spect_phys = spect_img.TransformContinuousIndexToPhysicalPoint((com_spect[2], com_spect[1], com_spect[0]))
        shift = tuple(com_spect_phys[i] - com_ct_phys[i] for i in range(3))
        print(f"[DEBUG] COM CT phys    : {com_ct_phys}")
        print(f"[DEBUG] COM SPECT phys : {com_spect_phys}")
        print(f"[DEBUG] COM-Shift (mm) : {shift}")
        tx = sitk.TranslationTransform(3, shift)
    else:
        tx = sitk.Transform()

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(spect_img)
    resampler.SetTransform(tx)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024.0)
    ct_resampled = resampler.Execute(ct_img)

    # Arrays im SPECT-Space (SimpleITK: [z, y, x])
    spect_arr_zyx = sitk.GetArrayFromImage(spect_img).astype(np.float32)
    ct_arr_zyx = sitk.GetArrayFromImage(ct_resampled).astype(np.float32)
    if debug:
        log_array_stats("SPECT array (zyx)", spect_arr_zyx)
        log_array_stats("CT array (zyx, resampled)", ct_arr_zyx)
        log_center_of_mass(spect_img, spect_arr_zyx, "SPECT")
        log_center_of_mass(ct_resampled, ct_arr_zyx, "CT resampled")
        if save_overlays:
            qc_dir = os.path.join(output_dir, "quickcheck_overlays")
            save_quickcheck_overlays(spect_img, ct_resampled, qc_dir)
    

    # HU -> µ
    print("[INFO] Erzeuge µ-Map aus CT-HU ...")
    mu_zyx = ct_hu_to_mu(ct_arr_zyx, energy_keV=140.0)

    # Forward-Modell direkt auf (z,y,x)
    print("[INFO] Starte Gamma-Kamera-Vorwärtsmodell (AP/PA, koronal) ...")
    proj_ap, proj_pa = gamma_camera_forward_zyx(
        act_zyx=spect_arr_zyx,
        mu_zyx=mu_zyx,
        scatter_sigma_xz=scatter_sigma_xz,
        coll_sigma_xz=coll_sigma_xz,
        use_scatter=use_scatter,
        use_attenuation=use_attenuation,
        use_collimator=use_collimator,
        spacing_y_mm=spacing_y,
        coll_kernel=coll_kernel,
        z0_slices=z0_slices,
        use_fft_conv=USE_FFT_CONV,
    )
    if debug:
        log_array_stats("Proj AP (z,x)", proj_ap)
        log_array_stats("Proj PA (z,x)", proj_pa)

    # Speichern als NPY
    np.save(os.path.join(output_dir, "projection_AP.npy"), proj_ap.astype(np.float32))
    np.save(os.path.join(output_dir, "projection_PA.npy"), proj_pa.astype(np.float32))
    print(f"[INFO] NPYs gespeichert in: {output_dir}")

    # PNG-Heatmaps
    save_projection_png(
        proj_ap,
        os.path.join(output_dir, "projection_AP.png"),
        "Simulated Scintigraphy – AP",
        spacing_x=spacing_x,
        spacing_z=spacing_z,
    )
    save_projection_png(
        proj_pa,
        os.path.join(output_dir, "projection_PA.png"),
        "Simulated Scintigraphy – PA",
        spacing_x=spacing_x,
        spacing_z=spacing_z,
    )
    print(f"[INFO] PNGs gespeichert in: {output_dir}")

    return proj_ap, proj_pa


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Simulierte Szintigrafie aus SPECT+CT mittels einfachem Gamma-Kamera-Modell."
    )
    parser.add_argument(
        "--spect_dicom_dir",
        required=True,
        help="Pfad zum DICOM-Ordner der SPECT-Serie (wird als Aktivitätsvolumen genutzt).",
    )
    parser.add_argument(
        "--ct_dicom_dir",
        required=True,
        help="Pfad zum DICOM-Ordner der CT-Serie (wird als Abschwächungskarte genutzt).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Ausgabeordner für Projektionen (NPY + PNG).",
    )
    parser.add_argument(
        "--scatter_sigma_xz",
        type=float,
        default=2.0,
        help="Sigma der Gauß-Streuung in der (z,x)-Ebene (in Pixeln). 0 = aus.",
    )
    parser.add_argument(
        "--coll_sigma_xz",
        type=float,
        default=2.0,
        help="Sigma der Gauß-Kollimatorunschärfe in der (z,x)-Ebene (in Pixeln). 0 = aus.",
    )
    parser.add_argument(
        "--no_scatter",
        action="store_true",
        help="Streuung abschalten.",
    )
    parser.add_argument(
        "--no_attenuation",
        action="store_true",
        help="Abschwächung abschalten.",
    )
    parser.add_argument(
        "--no_collimator",
        action="store_true",
        help="Kollimatorunschärfe abschalten.",
    )
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Zusätzliche Debug-Ausgaben (Orientierung, Min/Max, physikalische Punkte).",
    )
    parser.add_argument(
        "--save_overlays",
        action="store_true",
        help="Speichere einfache Overlay-PNGs (axial/coronal/sagittal) zum schnellen Sichtcheck.",
    )
    parser.add_argument(
        "--com_shift_ct",
        action="store_true",
        help="Verschiebe CT grob per COM-Differenz auf SPECT, vor dem Resample (Translation-only).",
    )
    parser.add_argument(
        "--rigid_register_ct",
        action="store_true",
        help="Starre Registrierung CT->SPECT (Mattes MI, Euler3D) vor dem Resample.",
    )

    args = parser.parse_args()

    run_scintigraphy_simulation(
        spect_dicom_dir=args.spect_dicom_dir,
        ct_dicom_dir=args.ct_dicom_dir,
        output_dir=args.output_dir,
        scatter_sigma_xz=args.scatter_sigma_xz,
        coll_sigma_xz=args.coll_sigma_xz,
        use_scatter=not args.no_scatter,
        use_attenuation=not args.no_attenuation,
        use_collimator=not args.no_collimator,
        collimator_kernel_mat=args.collimator_kernel_mat,
        z0_slices=args.z0_slices,
        debug=args.debug,
        save_overlays=args.save_overlays,
        com_shift_ct=args.com_shift_ct,
        rigid_register_ct=args.rigid_register_ct,
    )


if __name__ == "__main__":
    main()
