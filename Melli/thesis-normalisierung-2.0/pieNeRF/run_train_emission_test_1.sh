#!/bin/bash
#SBATCH --job-name=emission-train-test1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/mnguest12/slurm/emission_train_test1.%j.out
#SBATCH --error=/home/mnguest12/slurm/emission_train_test1.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pieNeRF

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}

echo "🚀 Starting Emission-NeRF TEST 1 (encoder input: normalized, proj target: counts) on $HOSTNAME"
echo "📅 Job started at: $(date)"
echo "🧠 GPUs assigned: ${SLURM_JOB_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# 1) Conda env
source /home/mnguest12/mambaforge/bin/activate totalseg

# 2) Project dir
cd /home/mnguest12/projects/thesis/pieNeRF

# 3) GPU info
nvidia-smi

# 4) Unique results dir for this run (job-id based)
RESULTS_DIR="/home/mnguest12/projects/thesis/pieNeRF/results_spect_test_1/run_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}/checkpoints" "${RESULTS_DIR}/logs"

echo "📁 RESULTS_DIR=${RESULTS_DIR}"

# Record basic metadata
{
  echo "date: $(date)"
  echo "host: ${HOSTNAME}"
  echo "job_id: ${SLURM_JOB_ID}"
  echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "git_rev: $(git rev-parse --short HEAD 2>/dev/null || echo '<no-git>')"
  echo "python: $(${PYTHON_BIN} -V 2>&1 || true)"
} > "${RESULTS_DIR}/run_meta.txt"

# 5) Symlink redirect for hardcoded results_spect paths
RESTORE_DIR=""
if [ -e "results_spect" ] && [ ! -L "results_spect" ]; then
  RESTORE_DIR="results_spect.__backup__.$(date +%Y%m%d_%H%M%S)"
  echo "⚠️ Backing up existing ./results_spect -> ${RESTORE_DIR}"
  mv "results_spect" "${RESTORE_DIR}"
fi

rm -f "results_spect"
ln -s "${RESULTS_DIR}" "results_spect"
echo "🔗 Linked ./results_spect -> ${RESULTS_DIR}"

# 6) Ensure test config exists: encoder input normalized (proj targets still counts via CLI)
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

echo "🏋️ Running train_emission.py (TEST 1)..."

exit_code=0

# --- core schedule knobs (after your proj-ramp fix) ---
# max proj weight = --proj-loss-weight
# min proj weight = --proj-weight-min  (starts at min after warmup, then ramps to max)
# warmup: 1500 steps (proj inactive)
# ramp: 20000 steps (slow, stabilizing)

CMD=(srun ${PYTHON_BIN} -u train_emission.py
  --config "${TEST_CFG}"
  --hybrid
  --encoder-use-ct
  --seed 0
  --max-steps 8000
  --log-every 200
  --save-every 1000

  --proj-target-source counts
  --poisson-rate-mode identity
  --proj-loss-type poisson
  --proj-loss-weight 0.005
  --proj-weight-min 0.0005
  --proj-warmup-steps 1500
  --proj-ramp-steps 20000

  --act-loss-weight 3.0
  --act-pos-fraction 0.05
  --act-pos-weight 10.0
  --act-sparsity-weight 2e-3
  --act-tv-weight 1e-4
  --act-samples 32768

  --ct-loss-weight 1e-4

  --z-enc-alpha 0.5
  --encoder-proj-transform none
  --proj-scale-source compute_p99

  --debug-latent-stats
  --debug-grad-terms-every 50
  --clip-grad-decoder 1.0

  --final-act-compare
)

# Save command for reproducibility
printf '%q ' "${CMD[@]}" > "${RESULTS_DIR}/command.sh"
echo >> "${RESULTS_DIR}/command.sh"

"${CMD[@]}" || exit_code=$?

echo "python_exit=$exit_code"

# 7) Restore results_spect path
echo "🧹 Restoring results_spect path..."
rm -f "results_spect"
if [ -n "${RESTORE_DIR}" ]; then
  echo "↩️ Restoring original ./results_spect from ${RESTORE_DIR}"
  mv "${RESTORE_DIR}" "results_spect"
fi

if [ $exit_code -ne 0 ]; then
  echo "[ERROR] Training failed with exit_code=$exit_code"
  exit $exit_code
fi

echo "✅ TEST 1 finished at: $(date)"
echo "📁 Outputs under: ${RESULTS_DIR}"