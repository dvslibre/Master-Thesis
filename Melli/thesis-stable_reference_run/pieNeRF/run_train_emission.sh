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

# 4️⃣ Training starten
echo "🏋️ Running train_emission.py..."
srun python -u train_emission.py \
  --config configs/spect.yaml \
  --max-steps 2000 \
  --rays-per-step 16384 \
  --log-every 10 \
  --preview-every 100 \
  --save-every 100 \
  --bg-weight 1.0 \
  --weight-threshold 0.0 \
  --act-loss-weight 0.005 \
  --act-samples 4096 \
  --act-pos-weight 1.0 \
  --ct-loss-weight 0.002 \
  --ct-threshold 0.002 \
  --ct-samples 4096 \
  --z-reg-weight 5e-4 \
  --tv-weight 2e-3 \
  --ray-tv-weight 2e-5 \
  --ray-tv-oversample 2 \
  --ray-tv-fg-only true \
  --ray-tv-edge-aware \
  --ray-tv-alpha 30 \
  --peak-loss-weight 5e-2 \
  --grad-stats-every 10 \
  --ray-split-enable \
  --ray-split 0.8 \
  --ray-split-seed 123 \
  --ray-split-tile 32 \
  --ray-fg-thr 0.0 \
  --ray-train-fg-frac 0.7 \
  --export-act-volume \
  --export-act-out results_spect/postprocessing/phantom_01/pred_act_step02000.npy \
  --export-act-chunk 131072
echo "✅ Training finished at: $(date)"
