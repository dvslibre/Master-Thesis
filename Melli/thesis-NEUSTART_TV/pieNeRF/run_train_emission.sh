#!/bin/bash
#SBATCH --job-name=emission-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/mnguest12/slurm/emission_train.%j.out
#SBATCH --error=/home/mnguest12/slurm/emission_train.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pieNeRF

echo "🚀 Starting Emission-NeRF training job on $HOSTNAME"
echo "📅 Job started at: $(date)"
echo "🧠 GPUs assigned: ${SLURM_JOB_GPUS}"

# 1️⃣ Conda-Umgebung aktivieren
source /home/mnguest12/mambaforge/bin/activate totalseg

# 2️⃣ Ins Projektverzeichnis
cd /home/mnguest12/projects/thesis/pieNeRF

# 3️⃣ Optional: GPU-Info ausgeben
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi

# 4️⃣ Training starten
echo "🏋️ Running train_emission.py..."
srun python -u train_emission.py \
  --config configs/spect.yaml \
  --max-steps 2000 \
  --rays-per-step 4096 \
  --log-every 10 \
  --preview-every 100 \
  --save-every 100 \
  --ct-padding-mode zeros \
  --log-proj-metrics-physical \
  --bg-weight 1.0 \
  --weight-threshold 0.0 \
  --act-loss-weight 0.02 \
  --act-samples 16384 \
  --act-pos-weight 3.0 \
  --ct-loss-weight 0.005 \
  --ct-threshold-norm 0.02 \
  --ct-samples 8192 \
  --z-reg-weight 0 \
  --tv-weight 0.0005 \
  --ray-tv-weight 1e-5 \
  --ray-tv-edge-aware False \
  --bg-depth-mass-weight 5e-4 \
  --bg-depth-eps-norm 1e-12 \
  --bg-depth-mode integral \
  --grad-stats-every 10 \
  --ray-split-enable \
  --ray-split-mode stratified_intensity \
  --ray-fg-quantile 0.90 \
  --ray-split 0.8 \
  --ray-split-seed 123 \
  --ray-train-fg-frac 0.9 \
  --log-quantiles-final-only True \
  --export-vol-res 128 \
  --atten-scale 1.0 \
  --inputs-normalized True
echo "✅ Training finished at: $(date)"
