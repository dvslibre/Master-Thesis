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
echo "[INFO] Starting CT pipeline (FDK → µ→HU → TS → Overlay)"
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
# Optional: Übergabe per CLI: <RUN_TAG> <MAT> <PROJ>
RUN_TAG="${1:-360}"
MAT_FILE="${2:-${BASE}/runs_${RUN_TAG}/ct_proj_stack${RUN_TAG}.mat}"
PROJ_FILE="${3:-${BASE}/runs_${RUN_TAG}/proj${RUN_TAG}.txt}"

OUTDIR="${BASE}/runs_${RUN_TAG}"
PREV_DIR="${OUTDIR}/ts_preview"
mkdir -p "$OUTDIR" "$PREV_DIR"

# Zielpfade (einheitlich)
RECON_RAS="${OUTDIR}/ct_recon_rtk_RAS.nii"
HU_NII="${OUTDIR}/ct_recon_rtk_HU.nii"
HU_ISO="${OUTDIR}/ct_recon_rtk_HU_iso1.0mm.nii"
PNG_HU_COR="${OUTDIR}/ct_recon_rtk_HU_coronal.png"
TS_OUTPUT="${OUTDIR}/ts_output.nii"
PNG_OVERLAY="${PREV_DIR}/coronal_overlay.png"

echo "[INFO] Using:"
echo "  RUN_TAG     = ${RUN_TAG}"
echo "  MAT_FILE    = ${MAT_FILE}"
echo "  PROJ_FILE   = ${PROJ_FILE}"
echo "  OUTDIR      = ${OUTDIR}"

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

# === 4) µ → HU (Anchors + Keep-Box + iso 1.0 mm + Preview) ===
echo "[STEP 2] µ→HU calibration + outside-air removal + keep-box ..."
python mu_to_hu_and_preview_2.py \
  --in  "${RECON_RAS}" \
  --out "${HU_NII}" \
  --anchor "x/2,480,180:60" \
  --anchor "x/2,300,210:30" \
  --anchor "x/2,300,100:-1000" \
  --keep_box_mm 110 690 95 405 95 405 \
  --smooth_sigma 0.0 \
  --resample_mm 1.0 \
  --preview \
  --view coronal \
  --coronal_orient AP \
  --png "${PNG_HU_COR}" \
  --percentile 2 98

# Das Skript legt bei --resample_mm zusätzlich ${HU_ISO} an
test -f "${HU_ISO}" || { echo "[FATAL] expected isotropic HU not found: ${HU_ISO}"; exit 3; }

# === 5) TotalSegmentator + Coronal Overlay (AP) ===
echo "[STEP 3] TotalSegmentator (multi-label) + coronal overlay ..."
# Vorsichtshalber altes Ergebnis entfernen
rm -f "${TS_OUTPUT}"

python ts_run_and_overlay.py \
  --in-hu "${HU_ISO}" \
  --seg    "${TS_OUTPUT}" \
  --overlay "${PNG_OVERLAY}" \
  --run-ts

test -f "${TS_OUTPUT}" || { echo "[FATAL] TS output missing: ${TS_OUTPUT}"; exit 4; }
test -f "${PNG_OVERLAY}" || { echo "[FATAL] overlay missing: ${PNG_OVERLAY}"; exit 5; }

echo "--------------------------------------------------------"
echo "[INFO] Pipeline finished successfully at $(date)"
echo "Results in: ${OUTDIR}/"
echo "  - RAS recon:       ${RECON_RAS}"
echo "  - HU volume:       ${HU_NII}"
echo "  - HU (iso 1.0mm):  ${HU_ISO}"
echo "  - TS multilabel:   ${TS_OUTPUT}"
echo "  - Previews:        ${PREV_DIR}/"
echo "--------------------------------------------------------"

# === 6) HU-Histogramm erzeugen ===
echo "[STEP 6] Creating HU histogram ..."

# Pfade aus vorherigen Schritten
HU_ISO="${OUTDIR}/ct_recon_rtk_HU_iso1.0mm.nii"
PREV_DIR="${OUTDIR}/ts_preview"
mkdir -p "${PREV_DIR}"

python - <<EOF
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys

hu_path = "${HU_ISO}"
out_png = os.path.join("${PREV_DIR}", "HU_histogram.png")

if not os.path.exists(hu_path):
    print(f"[WARN] HU_ISO not found: {hu_path}")
    sys.exit(1)

# HU-Daten laden
img = nib.load(hu_path)
arr = np.asarray(img.dataobj).astype(np.float32)
vals = arr[np.isfinite(arr)]

# Histogramm plotten
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
EOF