#!/bin/bash
#SBATCH --job-name=postproc_baseline_calib
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

RUN_DIR="${RUN_DIR:-/home/mnguest12/projects/thesis/pieNeRF/results_spect_test_1/run_661}"
SPLIT_JSON="${SPLIT_JSON:-${RUN_DIR}/split.json}"
CONFIG="${CONFIG:-/home/mnguest12/projects/thesis/pieNeRF/configs/spect_encoderNorm_projCounts.yaml}"
MANIFEST="/home/mnguest12/projects/thesis/pieNeRF/data/manifest_abs.csv"

MASK_PATTERN='/home/mnguest12/projects/thesis/Data_Processing/{phantom}/out/mask.npy'
DEVICE="cuda"
FORCE_USE_ATTENUATION="${FORCE_USE_ATTENUATION:-1}"
PRED_SLICES_DIR="${PRED_SLICES_DIR:-test_slices}"
PRED_ACT_PATTERN_TEMPLATE="${PRED_ACT_PATTERN_TEMPLATE:-}"
OUT_DIR="${OUT_DIR:-}"

if [[ -z "${PRED_ACT_PATTERN_TEMPLATE}" ]]; then
  PRED_ACT_PATTERN_TEMPLATE='{phantom}/activity_pred.npy'
fi

CONDA_ENV="totalseg"
CONDA_ACTIVATE="/home/mnguest12/mambaforge/bin/activate"
PYTHON_BIN="python"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "🛠 Starting postprocessing job on ${HOSTNAME}"
echo "📅 Job started at: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "run_dir=${RUN_DIR}"
echo "split_json=${SPLIT_JSON}"
echo "config=${CONFIG}"
echo "manifest=${MANIFEST}"
echo "mask_pattern=${MASK_PATTERN}"
echo "device=${DEVICE}"
echo "pred_slices_dir=${PRED_SLICES_DIR}"

source "${CONDA_ACTIVATE}"
conda activate "${CONDA_ENV}"

echo "python=$(which "${PYTHON_BIN}")"
"${PYTHON_BIN}" -V

echo "PWD=$(pwd)"
nvidia-smi || true

# --- sanity checks ---
test -d "${RUN_DIR}" || { echo "[ERROR] RUN_DIR not found: ${RUN_DIR}"; exit 2; }
test -f "${SPLIT_JSON}" || { echo "[ERROR] SPLIT_JSON not found: ${SPLIT_JSON}"; exit 2; }
test -f "${CONFIG}" || { echo "[ERROR] CONFIG not found: ${CONFIG}"; exit 2; }
test -f "${MANIFEST}" || { echo "[ERROR] MANIFEST not found: ${MANIFEST}"; exit 2; }

# ensure imports work (Data_Processing + repo)
export PYTHONPATH="/home/mnguest12/projects/thesis:/home/mnguest12/projects/thesis/pieNeRF:${PYTHONPATH:-}"

EFFECTIVE_SLICES_DIR="${PRED_SLICES_DIR}"
if [[ ! -d "${RUN_DIR}/${EFFECTIVE_SLICES_DIR}" ]]; then
  if [[ "${EFFECTIVE_SLICES_DIR}" == "rest_slices" && -d "${RUN_DIR}/test_slices" ]]; then
    echo "[WARN] ${RUN_DIR}/rest_slices not found -> using test_slices instead."
    EFFECTIVE_SLICES_DIR="test_slices"
  else
    echo "[ERROR] Slice directory not found: ${RUN_DIR}/${EFFECTIVE_SLICES_DIR}"
    exit 2
  fi
fi

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${RUN_DIR}/postproc_baseline_${EFFECTIVE_SLICES_DIR}"
fi

PRED_ACT_PATTERN="${EFFECTIVE_SLICES_DIR}/${PRED_ACT_PATTERN_TEMPLATE}"

echo ""
echo "============================================================"
echo "🔁 Running postprocessing for run_661"
echo "run_dir=${RUN_DIR}"
echo "split_json=${SPLIT_JSON}"
echo "config=${CONFIG}"
echo "out_dir=${OUT_DIR}"
echo "pred_act_pattern=${PRED_ACT_PATTERN}"
echo "force_use_attenuation=${FORCE_USE_ATTENUATION}"
echo "============================================================"

mkdir -p "${OUT_DIR}"
echo "[DBG] Listing ${RUN_DIR}/${EFFECTIVE_SLICES_DIR}:"
ls -lah "${RUN_DIR}/${EFFECTIVE_SLICES_DIR}" || true

set -x
cmd=(
  srun /usr/bin/time -v "${PYTHON_BIN}" -u postprocessing.py
  --run-dir "${RUN_DIR}"
  --split-json "${SPLIT_JSON}"
  --manifest "${MANIFEST}"
  --config "${CONFIG}"
  --out-dir "${OUT_DIR}"
  --mask-path-pattern "${MASK_PATTERN}"
  --device "${DEVICE}"
  --pred-act-per-phantom
  --pred-act-pattern "${PRED_ACT_PATTERN}"
  --proj-forward-model train
  --checkpoint "${RUN_DIR}/checkpoints/checkpoint_step08000.pt"
  --render-projections
  --save-proj-npy
  --save-proj-png
  --timing
  --save-act-compare-5slices
  --skip-plots
  --debug-orientation-search
  --save-orientation-debug-volumes
  --save-active-organ-plots
)
if [[ "${FORCE_USE_ATTENUATION,,}" =~ ^(1|true|yes|on)$ ]]; then
  cmd+=(--force-use-attenuation)
fi
"${cmd[@]}"
set +x

echo "✅ Postprocessing finished at: $(date)"
