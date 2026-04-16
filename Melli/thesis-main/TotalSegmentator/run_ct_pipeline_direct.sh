#!/usr/bin/env bash
#SBATCH --job-name=ct_pipeline
#SBATCH --output=/home/mnguest12/slurm/ct_pipeline.%j.out
#SBATCH --error=/home/mnguest12/slurm/ct_pipeline.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G

set -euo pipefail
set -x

echo "========================================================"
echo "[INFO] Starting CT pipeline (FDK → fixed crop → µ→HU (apply_ab) → TS → Overlay)"
echo "========================================================"
echo "GPUs available: ${SLURM_JOB_GPUS:-unknown}"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "--------------------------------------------------------"

# === 1) Environment ===
source /home/mnguest12/mambaforge/bin/activate totalseg
BASE="/home/mnguest12/projects/thesis/TotalSegmentator"
cd "$BASE"

# === 2) Parameter/Defaults ===
RUN_TAG="${1:-360}"                                                                     # erstes Argument bei sbatch run_ct_pipeline_direct.sh 360 wird hier genommen (in dem Fall 360; das wäre aber auch der default, wenn nicht gesetzt)
MAT_FILE="${2:-${BASE}/runs_${RUN_TAG}/ct_proj_stack${RUN_TAG}.mat}"
PROJ_FILE="${3:-${BASE}/runs_${RUN_TAG}/proj${RUN_TAG}.txt}"

OUTDIR="${BASE}/runs_${RUN_TAG}"
PREV_DIR="${OUTDIR}/ts_preview"
mkdir -p "$OUTDIR" "$PREV_DIR"

RECON_RAS="${OUTDIR}/ct_recon_rtk_RAS.nii"
RECON_CROP="${OUTDIR}/ct_recon_rtk_RAS_crop.nii.gz"
HU_NII="${OUTDIR}/ct_recon_rtk_HU_from_ab.nii.gz"
HU_ISO="${OUTDIR}/ct_recon_rtk_HU_from_ab_iso1.0mm.nii.gz"
PNG_HU_COR="${PREV_DIR}/HU_from_ab_coronal.png"
TS_OUTPUT="${OUTDIR}/ts_output_from_ab.nii.gz"
PNG_OVERLAY="${PREV_DIR}/overlay_from_ab_coronal.png"

echo "[INFO] Using paths:"
echo "  RECON_RAS   = ${RECON_RAS}"
echo "  RECON_CROP  = ${RECON_CROP}"
echo "  HU_NII      = ${HU_NII}"
echo "  HU_ISO      = ${HU_ISO}"
echo "  TS_OUTPUT   = ${TS_OUTPUT}"

# === 3) FDK Reconstruction → RAS ===
echo "[STEP 1] FDK reconstruction (RTK) → RAS ..."
python recon_fdk_rtk.py \
  --mat  "${MAT_FILE}" \
  --proj "${PROJ_FILE}" \
  --out_ras "${RECON_RAS}" \
  --quick_ap_png "${PREV_DIR}/" \
  --preview_view coronal \
  --coronal_orient AP

test -f "${RECON_RAS}" || { echo "[FATAL] missing: ${RECON_RAS}"; exit 2; }

# === 4) Fixes Cropping in physikalischen mm-Koordinaten ===
echo "[STEP 2] Apply fixed crop box (Y=150–650 mm, Z=100–400 mm, X=100–400 mm) ..."
python - <<PY
import nibabel as nib, numpy as np
in_path = "${RECON_RAS}"
out_path = "${RECON_CROP}"
print(f"[Crop] in={in_path} -> out={out_path}")

img = nib.load(in_path)
vol = np.asarray(img.dataobj, dtype=np.float32)
aff = img.affine
dx,dy,dz = map(float, np.abs(np.diag(aff)[:3]))
X,Y,Z = vol.shape

# Crop-Box in mm (Y0 Y1 Z0 Z1 X0 X1)
Y0,Y1 = 150,650
Z0,Z1 = 100,400
X0,X1 = 100,400

ix0,ix1 = int(X0/dx), min(X, int(X1/dx))
iy0,iy1 = int(Y0/dy), min(Y, int(Y1/dy))
iz0,iz1 = int(Z0/dz), min(Z, int(Z1/dz))

cropped = vol[ix0:ix1, iy0:iy1, iz0:iz1]

# Affine neu setzen (Origin verschieben)
new_aff = aff.copy()
shift = aff @ np.array([ix0, iy0, iz0, 1.0])
new_aff[:3,3] = shift[:3]

out_img = nib.Nifti1Image(cropped.astype(np.float32), new_aff, img.header)
out_img.header.set_qform(new_aff, code=1); out_img.header.set_sform(new_aff, code=1)
nib.save(out_img, out_path)
print(f"[OK] Cropped saved: {out_path}  shape={cropped.shape}")
PY

test -f "${RECON_CROP}" || { echo "[FATAL] cropped RAS missing: ${RECON_CROP}"; exit 3; }

# === 5) µ → HU (direkte lineare Umrechnung mit fixem a,b) ===
echo "[STEP 3] µ→HU apply_ab (a=52221.34, b=-1222.24) ..."
python mu_to_hu_apply_ab.py \
  --in  "${RECON_CROP}" \
  --out "${HU_NII}" \
  --a 52221.34 --b -1222.24 \
  --preview --view coronal \
  --png "${PNG_HU_COR}" \
  --hu_window -1024 1500 \
  --smooth_sigma 0.0

test -f "${HU_NII}" || { echo "[FATAL] HU file missing: ${HU_NII}"; exit 4; }

# === 6) Isotropes Resample (1.0 mm) ===
echo "[INFO] Resampling HU volume to isotropic 1.0 mm ..."
python - <<PY
import nibabel as nib
from nibabel.processing import resample_to_output
hu_path="${HU_NII}"
out_path="${HU_ISO}"
print(f"[Resample] in={hu_path} -> out={out_path}")
img=nib.load(hu_path)
rs=resample_to_output(img, voxel_sizes=(1.0,1.0,1.0), order=1)
nib.save(rs, out_path)
print(f"[OK] saved isotropic: {out_path}")
PY

test -f "${HU_ISO}" || { echo "[FATAL] resampled HU missing: ${HU_ISO}"; exit 5; }

# === 7) TotalSegmentator + Overlay ===
echo "[STEP 4] TotalSegmentator (multi-label) + coronal overlay ..."
rm -f "${TS_OUTPUT}"

python ts_run_and_overlay.py \
  --in-hu "${HU_ISO}" \
  --seg    "${TS_OUTPUT}" \
  --overlay "${PNG_OVERLAY}" \
  --run-ts \
  --coronal_orient AP

test -f "${TS_OUTPUT}" || { echo "[FATAL] TS output missing: ${TS_OUTPUT}"; exit 6; }
test -f "${PNG_OVERLAY}" || { echo "[FATAL] overlay missing: ${PNG_OVERLAY}"; exit 7; }

echo "--------------------------------------------------------"
echo "[INFO] Pipeline finished successfully at $(date)"
echo "Results in: ${OUTDIR}/"
echo "  - RAS recon (cropped): ${RECON_CROP}"
echo "  - HU volume:           ${HU_NII}"
echo "  - HU (iso 1mm):        ${HU_ISO}"
echo "  - TS multilabel:       ${TS_OUTPUT}"
echo "  - Previews:            ${PREV_DIR}/"
echo "--------------------------------------------------------"

# === 8) HU-Histogramm ===
echo "[STEP 5] Creating HU histogram ..."
python - <<PY
import nibabel as nib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, os
hu_path="${HU_ISO}"
out_png=os.path.join("${PREV_DIR}","HU_histogram.png")
print(f"[Hist] from {hu_path} -> {out_png}")
img=nib.load(hu_path)
arr=np.asarray(img.dataobj).astype(np.float32)
vals=arr[np.isfinite(arr)]
plt.figure(figsize=(7,5))
plt.hist(vals, bins=400, range=(-1200,2500), color='steelblue', alpha=0.85)
plt.title("HU Histogram (iso volume)")
plt.xlabel("Hounsfield Units [HU]")
plt.ylabel("Voxel count")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_png, dpi=150)
plt.close()
print(f"[OK] HU histogram saved → {out_png}")
PY