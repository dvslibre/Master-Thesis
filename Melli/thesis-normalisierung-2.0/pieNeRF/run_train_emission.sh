#!/bin/bash
#SBATCH --job-name=emission-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/mnguest12/slurm/emission_train.%j.out
#SBATCH --error=/home/mnguest12/slurm/emission_train.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pieNeRF

set -e

PYTHON_BIN=${PYTHON_BIN:-python3}

echo "🚀 Starting Emission-NeRF training job on $HOSTNAME"
echo "📅 Job started at: $(date)"
echo "🧠 GPUs assigned: ${SLURM_JOB_GPUS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# 1️⃣ Conda-Umgebung aktivieren
source /home/mnguest12/mambaforge/bin/activate totalseg

# 2️⃣ Ins Projektverzeichnis
cd /home/mnguest12/projects/thesis/pieNeRF

# 3️⃣ Optional: GPU-Info ausgeben
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi

# 4️⃣ Training starten
echo "🏋️ Running train_emission.py..."
exit_code=0
srun ${PYTHON_BIN} -u train_emission.py \
  --config configs/spect.yaml \
  --hybrid \
  --seed 0 \
  --max-steps 8000 \
  --log-every 200 \
  --save-every 500 \
  --proj-target-source counts \
  --poisson-rate-mode identity \
  --proj-loss-type poisson \
  --proj-loss-weight 0.05 \
  --proj-warmup-steps 300 \
  --proj-ramp-steps 3000 \
  --act-loss-weight 3.0 \
  --act-pos-fraction 0.05 \
  --act-pos-weight 10.0 \
  --act-sparsity-weight 2e-3 \
  --act-tv-weight 1e-4 \
  --act-samples 32768 \
  --ct-loss-weight 1e-4 \
  --z-enc-alpha 0.5 \
  --debug-latent-stats \
  --grad-stats-every 50 \
  --final-act-compare|| exit_code=$?
echo "python_exit=$exit_code"
if [ $exit_code -ne 0 ]; then
  echo "[ERROR] Training failed with exit_code=$exit_code"
  exit $exit_code
fi
echo "✅ Training finished at: $(date)"