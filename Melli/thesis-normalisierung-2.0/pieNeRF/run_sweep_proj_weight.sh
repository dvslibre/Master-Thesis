#!/bin/bash
#SBATCH --job-name=projW_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mnguest12/slurm/sweep_projW.%j.out
#SBATCH --error=/home/mnguest12/slurm/sweep_projW.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pieNeRF

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}

echo "🚀 Starting proj-loss-weight sweep on $HOSTNAME"
echo "📅 Job started at: $(date)"
echo "🧠 GPUs assigned: ${SLURM_JOB_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# 1) Conda env (match baseline)
source /home/mnguest12/mambaforge/bin/activate totalseg

# 2) Project dir
cd /home/mnguest12/projects/thesis/pieNeRF
ROOT="/home/mnguest12/projects/thesis/pieNeRF"

# 3) GPU info
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Ensure repo imports are fine
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# --- Baseline config behavior: ensure encoderNorm cfg exists ---
TEST_CFG="configs/spect_encoderNorm_projCounts.yaml"
if [ ! -f "${TEST_CFG}" ]; then
  echo "🛠️ Creating ${TEST_CFG} from configs/spect.yaml (set data.proj_input_source=normalized)"
  ${PYTHON_BIN} - <<'PY'
import yaml
src="configs/spect.yaml"
dst="configs/spect_encoderNorm_projCounts.yaml"
with open(src,"r") as f:
    cfg=yaml.safe_load(f)
cfg.setdefault("data", {})
cfg["data"]["proj_input_source"] = "normalized"
with open(dst,"w") as f:
    yaml.safe_dump(cfg,f,sort_keys=False)
print("wrote", dst)
PY
fi

# --- Sweep values (only thing that changes) ---
PROJ_WEIGHTS=(0.0 0.001 0.0025 0.005 0.01)

# --- Fixed baseline parameters (must match baseline) ---
SEED=0
MAX_STEPS=8000
LOG_EVERY=200
SAVE_EVERY=1000

PROJ_TARGET_SOURCE="counts"
POISSON_RATE_MODE="identity"
PROJ_LOSS_TYPE="poisson"
PROJ_WEIGHT_MIN="0.0005"
PROJ_WARMUP_STEPS=1500
PROJ_RAMP_STEPS=20000

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

# --- Where to write sweep results ---
BASE_DIR="${ROOT}/results_spect_test_1/projW_sweep_run_${SLURM_JOB_ID}"
mkdir -p "${BASE_DIR}"
echo "📁 BASE_DIR=${BASE_DIR}"

# --- robust cleanup for results_spect symlink (baseline-style) ---
RESTORE_DIR=""
cleanup_link() {
  rm -f "${ROOT}/results_spect" || true
  if [ -n "${RESTORE_DIR:-}" ] && [ -e "${RESTORE_DIR}" ]; then
    mv "${RESTORE_DIR}" "${ROOT}/results_spect"
  fi
  RESTORE_DIR=""
}
trap cleanup_link EXIT

for W in "${PROJ_WEIGHTS[@]}"; do
  TAG="projW_${W}"
  RESULTS_DIR="${BASE_DIR}/${TAG}"
  mkdir -p "${RESULTS_DIR}/checkpoints" "${RESULTS_DIR}/logs"

  echo ""
  echo "============================================================"
  echo "🔁 Starting sweep run: ${TAG}  (proj-loss-weight=${W})"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  echo "============================================================"

  # Record basic metadata (baseline-like)
  {
    echo "date: $(date)"
    echo "host: ${HOSTNAME}"
    echo "job_id: ${SLURM_JOB_ID}"
    echo "tag: ${TAG}"
    echo "proj_loss_weight: ${W}"
    echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "git_rev: $(git rev-parse --short HEAD 2>/dev/null || echo '<no-git>')"
    echo "python: $(${PYTHON_BIN} -V 2>&1 || true)"
  } > "${RESULTS_DIR}/run_meta.txt"

  # If already finished, skip
  if [ -f "${RESULTS_DIR}/checkpoints/checkpoint_step08000.pt" ]; then
    echo "[sweep] SKIP ${TAG}: checkpoint_step08000.pt exists"
    continue
  fi

  # --- Symlink redirect for hardcoded results_spect paths (baseline-style) ---
  if [ -e "${ROOT}/results_spect" ] && [ ! -L "${ROOT}/results_spect" ]; then
    RESTORE_DIR="${ROOT}/results_spect.__backup__.$(date +%Y%m%d_%H%M%S)"
    echo "⚠️ Backing up existing ./results_spect -> ${RESTORE_DIR}"
    mv "${ROOT}/results_spect" "${RESTORE_DIR}"
  fi

  rm -f "${ROOT}/results_spect"
  ln -s "${RESULTS_DIR}" "${ROOT}/results_spect"
  echo "🔗 Linked ./results_spect -> ${RESULTS_DIR}"

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
    --proj-loss-weight "${W}"
    --proj-weight-min "${PROJ_WEIGHT_MIN}"
    --proj-warmup-steps "${PROJ_WARMUP_STEPS}"
    --proj-ramp-steps "${PROJ_RAMP_STEPS}"

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

  # Save command for reproducibility (baseline-like)
  printf '%q ' "${CMD[@]}" > "${RESULTS_DIR}/command.sh"
  echo >> "${RESULTS_DIR}/command.sh"

  set -x
  "${CMD[@]}" || exit_code=$?
  set +x

  echo "python_exit=$exit_code"
  if [ $exit_code -ne 0 ]; then
    echo "[ERROR] ${TAG} failed with exit_code=$exit_code"
    # keep outputs for debugging; continue to next W
  fi

  # Restore results_spect between runs
  cleanup_link

  # Guard: ensure outputs exist where expected
  if [ -d "${RESULTS_DIR}/checkpoints" ]; then
    echo "[guard] checkpoints dir exists ✅"
  else
    echo "[guard][FATAL] checkpoints dir missing at ${RESULTS_DIR}/checkpoints"
    exit 3
  fi

  FINAL_CKPT="${RESULTS_DIR}/checkpoints/checkpoint_step08000.pt"
  if [ -f "${FINAL_CKPT}" ]; then
    echo "[sweep] finished ${TAG} ✅ (${FINAL_CKPT})"
  else
    last_ckpt="$(ls -1 "${RESULTS_DIR}/checkpoints"/checkpoint_step*.pt 2>/dev/null | tail -n 1 || true)"
    if [ -n "${last_ckpt}" ]; then
      echo "[sweep][WARN] ${TAG}: checkpoint_step08000.pt missing; last ckpt: ${last_ckpt}"
    else
      echo "[sweep][WARN] ${TAG}: no checkpoints found"
    fi
  fi
done

echo "✅ SWEEP FINISHED (training only) at: $(date)"
echo "📁 Outputs under: ${BASE_DIR}"