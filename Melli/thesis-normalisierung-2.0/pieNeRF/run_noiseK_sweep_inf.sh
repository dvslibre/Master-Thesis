#!/bin/bash
#SBATCH --job-name=noiseK_sweep_inf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/mnguest12/slurm/sweep_noiseK_inf.%j.out
#SBATCH --error=/home/mnguest12/slurm/sweep_noiseK_inf.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pieNeRF

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}

# -----------------------------
# User knobs
# -----------------------------
ROOT="/home/mnguest12/projects/thesis/pieNeRF"
TEST_CFG="${TEST_CFG:-configs/spect_encoderNorm_projCounts.yaml}"

NOISE_KAPPAS=(1.0 0.5 0.25 0.1)
NOISE_SEEDS=(0)

BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-${ROOT}/results_spect_test_1/run_661}"
CHECKPOINT="${CHECKPOINT:-${BASELINE_RUN_DIR}/checkpoints/checkpoint_step08000.pt}"

BASE_DIR="${BASE_DIR:-${ROOT}/results_spect_test_1/noiseK_sweep_run_${SLURM_JOB_ID}}"

# -----------------------------
# Environment
# -----------------------------
echo "🚀 Starting noise κ sweep (inference/export) on $HOSTNAME"
echo "Job started: $(date)"
echo "BASE_DIR=${BASE_DIR}"
echo "CHECKPOINT=${CHECKPOINT}"

source /home/mnguest12/mambaforge/bin/activate totalseg
cd "${ROOT}"

nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

mkdir -p "${BASE_DIR}"

# -----------------------------
# Checkpoint flag
# -----------------------------
CKPT_FLAG="--checkpoint"
if [ ! -f "${CHECKPOINT}" ]; then
  echo "[FATAL] Checkpoint not found: ${CHECKPOINT}"
  exit 2
fi
echo "[INFO] Using checkpoint flag: ${CKPT_FLAG}"

# -----------------------------
# results_spect symlink handling
# -----------------------------
RESTORE_DIR=""
cleanup_link() {
  rm -f "${ROOT}/results_spect" || true
  if [ -n "${RESTORE_DIR:-}" ] && [ -e "${RESTORE_DIR}" ]; then
    mv "${RESTORE_DIR}" "${ROOT}/results_spect"
  fi
  RESTORE_DIR=""
}
trap cleanup_link EXIT

# -----------------------------
# Sweep loop
# -----------------------------
for K in "${NOISE_KAPPAS[@]}"; do
  for S in "${NOISE_SEEDS[@]}"; do
    TAG="noiseK_${K}_seed_${S}"
    RESULTS_DIR="${BASE_DIR}/${TAG}"
    mkdir -p "${RESULTS_DIR}/checkpoints" "${RESULTS_DIR}/logs"

    echo ""
    echo "============================================================"
    echo "🔁 ${TAG}"
    echo "RESULTS_DIR=${RESULTS_DIR}"
    echo "============================================================"

    # Robust skip: require latent_stats.json + noisy projections for all test phantoms
    if [ -f "${RESULTS_DIR}/test_slices/latent_stats.json" ]; then
      echo "[sweep] SKIP ${TAG}: test_slices/latent_stats.json exists"
      continue
    fi

    # Backup existing results_spect if it's a real dir
    if [ -e "${ROOT}/results_spect" ] && [ ! -L "${ROOT}/results_spect" ]; then
      RESTORE_DIR="${ROOT}/results_spect.__backup__.$(date +%Y%m%d_%H%M%S)"
      echo "⚠️ Backing up existing ./results_spect -> ${RESTORE_DIR}"
      mv "${ROOT}/results_spect" "${RESTORE_DIR}"
    fi

    rm -f "${ROOT}/results_spect"
    ln -s "${RESULTS_DIR}" "${ROOT}/results_spect"
    echo "🔗 Linked ./results_spect -> ${RESULTS_DIR}"

    # Store metadata
    {
      echo "date: $(date)"
      echo "host: ${HOSTNAME}"
      echo "job_id: ${SLURM_JOB_ID}"
      echo "tag: ${TAG}"
      echo "noise_kappa: ${K}"
      echo "noise_seed: ${S}"
      echo "checkpoint_flag: ${CKPT_FLAG}"
      echo "checkpoint_path: ${CHECKPOINT}"
      echo "git_rev: $(git rev-parse --short HEAD 2>/dev/null || echo '<no-git>')"
      echo "python: $(${PYTHON_BIN} -V 2>&1 || true)"
    } > "${RESULTS_DIR}/run_meta.txt"

    # --- Run export/inference ---
    CMD=(
      srun ${PYTHON_BIN} -u train_emission.py
      --config "${TEST_CFG}"
      --hybrid
      --encoder-use-ct
      --seed 0
      --max-steps 0
      "${CKPT_FLAG}" "${CHECKPOINT}"

      # MUST match baseline (avoid default drift)
      --encoder-proj-transform none
      --proj-scale-source compute_p99
      --z-enc-alpha 0.5

      --proj-target-source counts
      --poisson-rate-mode identity
      --proj-loss-type poisson
      --proj-gain-source z_enc

      --test-noise-mode poisson_counts
      --test-noise-kappa "${K}"
      --test-noise-seed "${S}"

      # NOTE: will likely skip when max-steps=0 (act_vol not set)
      --final-act-compare
    )

    printf '%q ' "${CMD[@]}" > "${RESULTS_DIR}/command.sh"
    echo >> "${RESULTS_DIR}/command.sh"

    set -x
    "${CMD[@]}"
    set +x

    cleanup_link
  done
done

echo "✅ Noise κ sweep (inference/export) finished: $(date)"
echo "📁 Outputs under: ${BASE_DIR}"