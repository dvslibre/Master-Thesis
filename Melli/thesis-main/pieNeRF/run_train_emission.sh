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
    --rays-per-step 16384 \
    --log-every 10 \
    --preview-every 50 \
    --save-every 100 \
    --bg-weight 0.1 \
    --weight-threshold 0.0 \
    --act-loss-weight 0.03 \
    --act-samples 4096 \
    --ct-loss-weight 0.05 \
    --ct-threshold 0.05 \
    --ct-samples 8192 \
    --z-reg-weight 1e-4 \
    --tv-weight 0.001 \
    --use-organ-mask \
    --mask-loss-weight 0.05
echo "✅ Training finished at: $(date)"
