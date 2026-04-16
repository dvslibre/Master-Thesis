#!/bin/bash
#SBATCH --job-name=postproc_sweep_trainfwd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/mnguest12/slurm/postprocessing.%j.out
#SBATCH --error=/home/mnguest12/slurm/postprocessing.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pieNeRF

set -euo pipefail

SWEEP_ROOT="${SWEEP_ROOT:-/home/mnguest12/projects/thesis/pieNeRF/results_spect_test_1/projSchedule_sweep_run_692}"
SWEEP_TAGS="${SWEEP_TAGS:-}"
MANIFEST="/home/mnguest12/projects/thesis/pieNeRF/data/manifest_abs.csv"

MASK_PATTERN='/home/mnguest12/projects/thesis/Data_Processing/{phantom}/out/mask.npy'
DEVICE="cuda"

PRED_SLICES_DIR="${PRED_SLICES_DIR:-test_slices}"
CHECKPOINT_REL="${CHECKPOINT_REL:-checkpoints/checkpoint_step08000.pt}"
CHECKPOINT_GLOB_FALLBACK="${CHECKPOINT_GLOB_FALLBACK:-checkpoints/checkpoint_step*.pt}"

CONDA_ENV="totalseg"
CONDA_ACTIVATE="/home/mnguest12/mambaforge/bin/activate"
PYTHON_BIN="python"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "🛠 Starting sweep postprocessing"
echo "sweep_root=${SWEEP_ROOT}"

source "${CONDA_ACTIVATE}"
conda activate "${CONDA_ENV}"

test -d "${SWEEP_ROOT}" || { echo "[ERROR] SWEEP_ROOT not found"; exit 2; }
test -f "${MANIFEST}" || { echo "[ERROR] MANIFEST not found"; exit 2; }

export PYTHONPATH="/home/mnguest12/projects/thesis:/home/mnguest12/projects/thesis/pieNeRF:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Discover tag directories
# ------------------------------------------------------------------

declare -a TAG_DIRS=()

if [[ -n "${SWEEP_TAGS}" ]]; then
  for tag in ${SWEEP_TAGS}; do
    TAG_DIRS+=("${SWEEP_ROOT}/${tag}")
  done
else
  while IFS= read -r d; do
    TAG_DIRS+=("${d}")
  done < <(find "${SWEEP_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
fi

FAIL_COUNT=0

# ------------------------------------------------------------------
# Loop over sweep tags
# ------------------------------------------------------------------

for tag_dir in "${TAG_DIRS[@]}"; do

  SWEEP_TAG="$(basename "${tag_dir}")"
  RUN_DIR="${tag_dir}"
  SPLIT_JSON="${RUN_DIR}/split.json"

  echo ""
  echo "============================================================"
  echo "🔁 Processing ${SWEEP_TAG}"
  echo "run_dir=${RUN_DIR}"
  echo "============================================================"

  if [[ ! -f "${SPLIT_JSON}" ]]; then
    echo "[ERROR] split.json missing"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi

  # --------------------------------------------------------------
  # Extract config from command.sh
  # --------------------------------------------------------------

  CMD_SH="${tag_dir}/command.sh"
  CONFIG=""

  if [[ -f "${CMD_SH}" ]]; then
    CONFIG="$(grep -oE -- '--config[[:space:]]+[^[:space:]]+' "${CMD_SH}" | head -n1 | awk '{print $2}')"
  fi

  if [[ -z "${CONFIG}" ]]; then
    echo "[ERROR] Could not extract --config from command.sh"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi

  if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] CONFIG not found: ${CONFIG}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi

  # --------------------------------------------------------------
  # Slice directory check
  # --------------------------------------------------------------

  if [[ ! -d "${RUN_DIR}/${PRED_SLICES_DIR}" ]]; then
    echo "[ERROR] Slice directory missing: ${RUN_DIR}/${PRED_SLICES_DIR}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi

  # For --pred-act-per-phantom, postprocessing formats {phantom} per test id.
  # Training outputs are in test_slices/<phantom>/activity_pred.npy.
  PRED_ACT_PATTERN="${PRED_SLICES_DIR}/{phantom}/activity_pred.npy"

  # --------------------------------------------------------------
  # Checkpoint detection
  # --------------------------------------------------------------

  CHECKPOINT="${RUN_DIR}/${CHECKPOINT_REL}"

  if [[ ! -f "${CHECKPOINT}" ]]; then
    CHECKPOINT="$(ls -1 ${RUN_DIR}/${CHECKPOINT_GLOB_FALLBACK} 2>/dev/null | sort -V | tail -n1 || true)"
    if [[ -z "${CHECKPOINT}" ]]; then
      echo "[ERROR] No checkpoint found"
      FAIL_COUNT=$((FAIL_COUNT + 1))
      continue
    fi
    echo "[WARN] Using fallback checkpoint: ${CHECKPOINT}"
  fi

  OUT_DIR="${RUN_DIR}/postproc_trainfwd_${PRED_SLICES_DIR}"
  mkdir -p "${OUT_DIR}"

  echo "config=${CONFIG}"
  echo "checkpoint=${CHECKPOINT}"
  echo "pattern=${PRED_ACT_PATTERN}"

  # --------------------------------------------------------------
  # Execute postprocessing
  # --------------------------------------------------------------

  set -x
  if ! srun /usr/bin/time -v "${PYTHON_BIN}" -u postprocessing.py \
      --run-dir "${RUN_DIR}" \
      --split-json "${SPLIT_JSON}" \
      --manifest "${MANIFEST}" \
      --config "${CONFIG}" \
      --out-dir "${OUT_DIR}" \
      --mask-path-pattern "${MASK_PATTERN}" \
      --device "${DEVICE}" \
      --pred-act-per-phantom \
      --pred-act-pattern "${PRED_ACT_PATTERN}" \
      --proj-forward-model train \
      --checkpoint "${CHECKPOINT}" \
      --render-projections \
      --save-proj-npy \
      --save-proj-png \
      --timing \
      --save-act-compare-5slices \
      --skip-plots \
      --debug-orientation-search \
      --save-orientation-debug-volumes \
      --save-active-organ-plots; then

      set +x
      echo "[ERROR] Postprocessing failed for ${SWEEP_TAG}"
      FAIL_COUNT=$((FAIL_COUNT + 1))
      continue
  fi
  set +x

done

if [[ ${FAIL_COUNT} -ne 0 ]]; then
  echo "[ERROR] Finished with ${FAIL_COUNT} failed sweep tag(s)."
  exit 1
fi

echo "✅ Sweep postprocessing finished"
