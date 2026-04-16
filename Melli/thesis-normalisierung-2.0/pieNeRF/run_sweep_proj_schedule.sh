#!/bin/bash
#SBATCH --job-name=projSchedule_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mnguest12/slurm/sweep_projSchedule.%j.out
#SBATCH --error=/home/mnguest12/slurm/sweep_projSchedule.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pieNeRF

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}

echo "🚀 Starting projection scheduling sweep on $HOSTNAME"
echo "📅 Job started at: $(date)"
echo "🧠 GPUs assigned: ${SLURM_JOB_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# 1) Conda env (match baseline)
source /home/mnguest12/mambaforge/bin/activate totalseg

# 2) Project dir
cd /home/mnguest12/projects/thesis/pieNeRF
ROOT="/home/mnguest12/projects/thesis/pieNeRF"

nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

TEST_CFG="configs/spect_encoderNorm_projCounts.yaml"

# ------------------------------
# Sweep definitions
# ------------------------------

# Each entry: "warmup ramp label"
SCHEDULES=(
  "1500 20000 baseline"
  "0 0 no_warmup"
  "500 3000 short_ramp"
  "500 6000 mid_ramp"
  "0 8000 full_proj_from_start"
)

# ------------------------------
# Baseline fixed params
# ------------------------------

SEED=0
MAX_STEPS=8000
LOG_EVERY=200
SAVE_EVERY=1000

PROJ_TARGET_SOURCE="counts"
POISSON_RATE_MODE="identity"
PROJ_LOSS_TYPE="poisson"
PROJ_LOSS_WEIGHT="0.005"
PROJ_WEIGHT_MIN="0.0005"

ACT_LOSS_WEIGHT="3.0"
ACT_POS_FRACTION="0.05"
ACT_POS_WEIGHT="10.0"
ACT_SPARSITY_WEIGHT="2e-3"
ACT_TV_WEIGHT="1e-4"
ACT_SAMPLES="32768"
CT_LOSS_WEIGHT="1e-4"

Z_ENC_ALPHA="0.5"
ENCODER_PROJ_TRANSFORM="none"
PROJ_SCALE_SOURCE="compute_p99"

DEBUG_GRAD_EVERY=50
CLIP_GRAD_DECODER="1.0"

# ------------------------------
# Output root
# ------------------------------

BASE_DIR="${ROOT}/results_spect_test_1/projSchedule_sweep_run_${SLURM_JOB_ID}"
mkdir -p "${BASE_DIR}"
echo "📁 BASE_DIR=${BASE_DIR}"

RESTORE_DIR=""
cleanup_link() {
  rm -f "${ROOT}/results_spect" || true
  if [ -n "${RESTORE_DIR:-}" ] && [ -e "${RESTORE_DIR}" ]; then
    mv "${RESTORE_DIR}" "${ROOT}/results_spect"
  fi
  RESTORE_DIR=""
}
trap cleanup_link EXIT

# ------------------------------
# Sweep loop
# ------------------------------

for entry in "${SCHEDULES[@]}"; do

  read WARMUP RAMP LABEL <<< "${entry}"
  TAG="sched_${LABEL}"
  RESULTS_DIR="${BASE_DIR}/${TAG}"
  mkdir -p "${RESULTS_DIR}/checkpoints" "${RESULTS_DIR}/logs"

  echo ""
  echo "============================================================"
  echo "🔁 ${TAG} (warmup=${WARMUP}, ramp=${RAMP})"
  echo "============================================================"

  {
    echo "date: $(date)"
    echo "warmup_steps: ${WARMUP}"
    echo "ramp_steps: ${RAMP}"
    echo "proj_loss_weight: ${PROJ_LOSS_WEIGHT}"
  } > "${RESULTS_DIR}/run_meta.txt"

  if [ -f "${RESULTS_DIR}/checkpoints/checkpoint_step08000.pt" ]; then
    echo "[sweep] SKIP ${TAG}"
    continue
  fi

  if [ -e "${ROOT}/results_spect" ] && [ ! -L "${ROOT}/results_spect" ]; then
    RESTORE_DIR="${ROOT}/results_spect.__backup__.$(date +%Y%m%d_%H%M%S)"
    mv "${ROOT}/results_spect" "${RESTORE_DIR}"
  fi

  rm -f "${ROOT}/results_spect"
  ln -s "${RESULTS_DIR}" "${ROOT}/results_spect"

  exit_code=0

  CMD=(srun ${PYTHON_BIN} -u train_emission.py
    --config "${TEST_CFG}"
    --hybrid
    --encoder-use-ct
    --seed "${SEED}"
    --max-steps "${MAX_STEPS}"
    --log-every "${LOG_EVERY}"
    --save-every "${SAVE_EVERY}"

    --proj-target-source "${PROJ_TARGET_SOURCE}"
    --poisson-rate-mode "${POISSON_RATE_MODE}"
    --proj-loss-type "${PROJ_LOSS_TYPE}"
    --proj-loss-weight "${PROJ_LOSS_WEIGHT}"
    --proj-weight-min "${PROJ_WEIGHT_MIN}"
    --proj-warmup-steps "${WARMUP}"
    --proj-ramp-steps "${RAMP}"

    --act-loss-weight "${ACT_LOSS_WEIGHT}"
    --act-pos-fraction "${ACT_POS_FRACTION}"
    --act-pos-weight "${ACT_POS_WEIGHT}"
    --act-sparsity-weight "${ACT_SPARSITY_WEIGHT}"
    --act-tv-weight "${ACT_TV_WEIGHT}"
    --act-samples "${ACT_SAMPLES}"

    --ct-loss-weight "${CT_LOSS_WEIGHT}"

    --z-enc-alpha "${Z_ENC_ALPHA}"
    --encoder-proj-transform "${ENCODER_PROJ_TRANSFORM}"
    --proj-scale-source "${PROJ_SCALE_SOURCE}"

    --debug-latent-stats
    --debug-grad-terms-every "${DEBUG_GRAD_EVERY}"
    --clip-grad-decoder "${CLIP_GRAD_DECODER}"

    --final-act-compare
  )

  printf '%q ' "${CMD[@]}" > "${RESULTS_DIR}/command.sh"
  echo >> "${RESULTS_DIR}/command.sh"

  set -x
  "${CMD[@]}" || exit_code=$?
  set +x

  cleanup_link

  if [ $exit_code -ne 0 ]; then
    echo "[ERROR] ${TAG} failed"
  else
    echo "[sweep] finished ${TAG}"
  fi

done

echo "✅ Projection scheduling sweep finished"
echo "📁 Outputs under: ${BASE_DIR}"fdd