#!/bin/bash
#SBATCH --job-name=qplanar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=dgx
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/mnguest12/slurm/qplanar.%j.out
#SBATCH --error=/home/mnguest12/slurm/qplanar.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/PhantomGenerator

set -euo pipefail

PROJECT_ROOT=/home/mnguest12/projects/thesis
PYTHON_BIN=${PYTHON_BIN:-python3}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/Data_Processing}
PHANTOM_ROOT=${PHANTOM_ROOT:-${PROJECT_ROOT}/PhantomGenerator}
KERNEL_MAT=${KERNEL_MAT:-${PHANTOM_ROOT}/LEAP_Kernel.mat}
MASK_ROBUSTNESS_SUITE=${MASK_ROBUSTNESS_SUITE:-1}
PHANTOM_LIST=${PHANTOM_LIST:-"16 24 30"}
MASK_ROBUSTNESS_VARIANT=${MASK_ROBUSTNESS_VARIANT:-baseline+dilation2d_r1_xy+shift_x1cm+shift_x2cm}

echo "🚀 Starting QPlanar evaluations on $(hostname)"
echo "📅 Job started at: $(date)"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "Using kernel: ${KERNEL_MAT}"
echo "MASK_ROBUSTNESS_SUITE=${MASK_ROBUSTNESS_SUITE}"
echo "MASK_ROBUSTNESS_VARIANT=${MASK_ROBUSTNESS_VARIANT}"
echo "PHANTOM_LIST=${PHANTOM_LIST}"

source /home/mnguest12/mambaforge/bin/activate totalseg

cd "${PHANTOM_ROOT}"

if [[ ! -f "${KERNEL_MAT}" ]]; then
  echo "[ERROR] Kernel file not found: ${KERNEL_MAT}"
  exit 1
fi

for NUM in ${PHANTOM_LIST}; do
  PHANTOM_NAME="phantom_${NUM}"
  BASE_DIR="${DATA_ROOT}/${PHANTOM_NAME}"
  SRC_DIR="${BASE_DIR}/src"
  OUT_DIR="${PHANTOM_ROOT}/qplanar_results/${PHANTOM_NAME}"
  META_JSON="${BASE_DIR}/out/meta_simple.json"

  echo "\n=== QPlanar: ${PHANTOM_NAME} ==="
  if [[ ! -d "${SRC_DIR}" ]]; then
    echo "[WARNING] Missing src directory for ${PHANTOM_NAME}; skipping"
    continue
  fi

  SPECT_BIN=$(find "${SRC_DIR}" -maxdepth 1 -type f -name "*spect*par_atn*.bin" -print -quit || true)
  MASK_BIN=$(find "${SRC_DIR}" -maxdepth 1 -type f -name "*mask*.bin" -print -quit || true)

  if [[ -z "${SPECT_BIN}" || -z "${MASK_BIN}" ]]; then
    echo "[ERROR] Required binaries not found for ${PHANTOM_NAME}"
    echo "       spect_bin=${SPECT_BIN:-<missing>}"
    echo "       mask_bin=${MASK_BIN:-<missing>}"
    exit 1
  fi

  if [[ ! -f "${META_JSON}" ]]; then
    echo "[ERROR] Meta JSON missing: ${META_JSON}"
    exit 1
  fi

  mkdir -p "${OUT_DIR}"

  echo "Using spect bin: ${SPECT_BIN}"
  echo "Using mask bin: ${MASK_BIN}"
  echo "Writing results to: ${OUT_DIR}"

  EXTRA_ARGS=()
  if [[ "${MASK_ROBUSTNESS_SUITE}" == "1" ]]; then
    echo "[INFO] Robustness suite enabled (${MASK_ROBUSTNESS_VARIANT})"
    EXTRA_ARGS+=(--mask-robustness-suite)
  else
    echo "[INFO] Robustness suite disabled (baseline only)"
  fi

  srun "${PYTHON_BIN}" -u qplanar.py \
    --base "${BASE_DIR}" \
    --spect_bin "${SPECT_BIN}" \
    --mask_bin "${MASK_BIN}" \
    --meta_json "${META_JSON}" \
    --kernel_mat "${KERNEL_MAT}" \
    --shape 256,256,651 \
    --pixel_size_mm 1.5 \
    --poisson \
    --counts_per_pixel 30 \
    --out_dir "${OUT_DIR}" \
    "${EXTRA_ARGS[@]}"

  echo "✅ Finished ${PHANTOM_NAME} at $(date)"
done

echo "📅 Job finished at: $(date)"
