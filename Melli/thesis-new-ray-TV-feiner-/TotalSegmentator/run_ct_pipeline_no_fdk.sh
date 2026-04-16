#!/usr/bin/env bash
#SBATCH --job-name=ct_pipeline_no_fdk_v2
#SBATCH --output=/home/mnguest12/slurm/ct_pipeline.%j.out
#SBATCH --error=/home/mnguest12/slurm/ct_pipeline.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G

set -euo pipefail
set -x

echo "========================================================"
echo "[INFO] CT pipeline (crop → µ→HU (apply_ab) → TS-friendly → 1.0mm → TS → Overlay)"
echo "========================================================"
echo "Node: $(hostname) | GPUs: ${SLURM_JOB_GPUS:-unknown}"
echo "Start: $(date)"
echo "--------------------------------------------------------"

# === 1) Env ===
source /home/mnguest12/mambaforge/bin/activate totalseg
BASE="/home/mnguest12/projects/thesis/TotalSegmentator"
cd "$BASE"

# === 2) Inputs/Outputs ===
RUN_TAG="${1:-360}"

OUTDIR="${BASE}/runs_${RUN_TAG}"
PREV_DIR="${OUTDIR}/ts_preview"
mkdir -p "$OUTDIR" "$PREV_DIR"

IN_RAS="${OUTDIR}/ct_recon_rtk_RAS.nii"

RECON_CROP="${OUTDIR}/ct_recon_rtk_RAS_crop.nii.gz"
HU_NII="${OUTDIR}/ct_recon_rtk_HU_from_ab.nii.gz"
HU_TSF="${OUTDIR}/ct_recon_rtk_HU_for_TS.nii.gz"
HU_ISO="${OUTDIR}/ct_recon_rtk_HU_for_TS_iso1.0mm.nii.gz"

TS_OUTPUT="${OUTDIR}/ts_output_from_ab.nii.gz"
PNG_HU_COR="${PREV_DIR}/HU_from_ab_coronal.png"
PNG_OVERLAY="${PREV_DIR}/overlay_from_ab_coronal.png"
PNG_HIST="${PREV_DIR}/HU_histogram.png"

echo "[INFO] Paths:"
printf "  IN_RAS   = %s\n  CROP     = %s\n  HU       = %s\n  HU_TSF   = %s\n  HU_ISO   = %s\n  TS_OUT   = %s\n" \
  "$IN_RAS" "$RECON_CROP" "$HU_NII" "$HU_TSF" "$HU_ISO" "$TS_OUTPUT"

test -f "${IN_RAS}" || { echo "[FATAL] input missing: ${IN_RAS}"; exit 2; }

# === 3) Fixed crop (Y=150–650 mm, Z=100–400 mm, X=100–400 mm) ===
echo "[STEP 1] Cropping volume ..."
python - <<PY
import nibabel as nib, numpy as np
p_in="${IN_RAS}"; p_out="${RECON_CROP}"
img=nib.load(p_in); vol=np.asarray(img.dataobj,np.float32); aff=img.affine
dx,dy,dz=map(float,np.abs(np.diag(aff)[:3])); X,Y,Z=vol.shape
Y0,Y1=150,650; Z0,Z1=100,400; X0,X1=100,400
ix0,ix1=int(X0/dx),min(X,int(X1/dx))
iy0,iy1=int(Y0/dy),min(Y,int(Y1/dy))
iz0,iz1=int(Z0/dz),min(Z,int(Z1/dz))
cropped=vol[ix0:ix1,iy0:iy1,iz0:iz1]
new_aff=aff.copy(); new_aff[:3,3]=(aff@np.array([ix0,iy0,iz0,1.0]))[:3]
nib.save(nib.Nifti1Image(cropped,np.array(new_aff),img.header),p_out)
print(f"[OK] Cropped volume saved: {p_out} shape={cropped.shape}")
PY
test -f "${RECON_CROP}" || { echo "[FATAL] cropped volume missing"; exit 3; }

# === 4) µ → HU (apply a,b) ===
echo "[STEP 2] µ→HU conversion (a=52221.34, b=-1222.24) ..."
python mu_to_hu_apply_ab.py \
  --in  "${RECON_CROP}" \
  --out "${HU_NII}" \
  --a 52221.34 --b -1222.24 \
  --preview --view coronal \
  --png "${PNG_HU_COR}" \
  --hu_window -1024 1500 \
  --smooth_sigma 0.0
test -f "${HU_NII}" || { echo "[FATAL] HU file missing"; exit 4; }

# === 5) TS-friendly (CRISP: band-limited noise, no blur, stronger unsharp) ===
echo "[STEP 3] TS-friendly (CRISP): clip [-1024,1500] + band-noise (σ=20) + unsharp(amount=0.35, radius≈0.8) ..."
python - <<PY
import nibabel as nib, numpy as np
from scipy.ndimage import gaussian_filter
p_in="${HU_NII}"; p_out="${HU_TSF}"
img=nib.load(p_in); arr=np.asarray(img.dataobj, np.float32)

# 1) Clip auf TS-Bereich
arr = np.clip(arr, -1024, 1500)

# 2) Korn: überwiegend bandbegrenztes Noise (feinstrukturiert), wenig weißes Noise
np.random.seed(0)
sigma_total = 20.0
white = np.random.normal(0.0, sigma_total, size=arr.shape).astype(np.float32)
band  = gaussian_filter(np.random.normal(0.0, 1.0, size=arr.shape).astype(np.float32), sigma=0.6)
band *= sigma_total / (band.std() + 1e-6)
arr = arr + (0.2*white + 0.8*band)   # mehr „klinische“ Textur, weniger Pixelrauschen

# 3) KEIN globales Blur

# 4) Kräftigere Kantenbetonung (Unsharp mit kleinerem Radius)
blurred = gaussian_filter(arr, sigma=0.8)  # „Radius“
arr = np.clip(arr + 0.35*(arr - blurred), -1024, 1500)

nib.save(nib.Nifti1Image(arr.astype(np.float32), img.affine, img.header), p_out)
print(f"[OK] TS-friendly HU (CRISP) -> {p_out}")
PY

# === 6) Resample to 1.0 mm isotropic (cubic) ===
python - <<PY
import nibabel as nib
from nibabel.processing import resample_to_output
p_in="${HU_TSF}"; p_out="${HU_ISO}"
img=nib.load(p_in)
rs=resample_to_output(img, voxel_sizes=(1.0,1.0,1.0), order=3)  # cubic
nib.save(rs, p_out)
print(f"[OK] Isotropic HU (1.0 mm, cubic) -> {p_out}")
PY

# === 7) TotalSegmentator + Overlay ===
echo "[STEP 5] TotalSegmentator (multi-label) + overlay ..."
rm -f "${TS_OUTPUT}"
python ts_run_and_overlay.py \
  --in-hu  "${HU_ISO}" \
  --seg    "${TS_OUTPUT}" \
  --overlay "${PNG_OVERLAY}" \
  --run-ts \
  --coronal_orient AP
test -f "${TS_OUTPUT}"   || { echo "[FATAL] TS output missing"; exit 7; }
test -f "${PNG_OVERLAY}" || { echo "[FATAL] overlay missing"; exit 8; }

# === 8) HU-Histogramm ===
echo "[STEP 6] Creating HU histogram ..."
python - <<PY
import nibabel as nib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, os
p="${HU_ISO}"; out="${PNG_HIST}"
img=nib.load(p); a=np.asarray(img.dataobj,np.float32)
v=a[np.isfinite(a)]
plt.figure(figsize=(7,5))
plt.hist(v,bins=400,range=(-1200,2000),color='steelblue',alpha=0.85)
plt.title("HU Histogram (1.0 mm isotropic, TS-friendly)")
plt.xlabel("Hounsfield Units [HU]"); plt.ylabel("Voxel count")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(out,dpi=150); plt.close()
print(f"[OK] HU histogram saved → {out}")
PY

echo "--------------------------------------------------------"
echo "[INFO] Pipeline finished successfully at $(date)"
echo "Results in: ${OUTDIR}/"
echo "  - Cropped RAS:     ${RECON_CROP}"
echo "  - HU volume:       ${HU_NII}"
echo "  - TS-friendly HU:  ${HU_TSF}"
echo "  - Isotropic (1mm): ${HU_ISO}"
echo "  - TS multilabel:   ${TS_OUTPUT}"
echo "  - Previews:        ${PREV_DIR}/"
echo "--------------------------------------------------------"