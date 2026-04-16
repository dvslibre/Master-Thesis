#!/bin/bash
#SBATCH --job-name=pixelpienerf-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/mnguest12/slurm/pixelpienerf.%j.out
#SBATCH --error=/home/mnguest12/slurm/pixelpienerf.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pixelNeRF+pieNeRF

echo "🚀 Starting pixelNeRF+pieNeRF training on $HOSTNAME"
echo "📅 Job started at: $(date)"
echo "🧠 GPUs assigned: ${SLURM_JOB_GPUS}"

# 1️⃣ Conda-Umgebung aktivieren
source /home/mnguest12/mambaforge/bin/activate pixelnerf-med

# 2️⃣ Ins Projektverzeichnis
cd /home/mnguest12/projects/thesis/pixelNeRF+pieNeRF

# 2a️⃣ PYTHONPATH setzen, damit das lokale `model` gefunden wird
export PYTHONPATH="/home/mnguest12/projects/thesis/pixelNeRF+pieNeRF:/home/mnguest12/projects/thesis/pixelNeRF+pieNeRF/pixel-nerf:/home/mnguest12/projects/thesis/pixelNeRF+pieNeRF/pixel-nerf/src:${PYTHONPATH}"

# 3️⃣ Optional: GPU-Info ausgeben
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi

# 4️⃣ Training starten (Defaults können via CLI-Args überschrieben werden)
echo "🏋️ Running train_spect_pienerf_pixelnerf.py..."
srun python -u train/train_spect_pienerf_pixelnerf.py \
    --data_root thesis_med/pieNeRF/data \
    --epochs 2000 \
    --lr 3e-4 \
    --batch_size 2 \
    --gpu_id 0 \
    --run_name dirfix_lrflip \
    --log_interval 50 \
    --vis_interval 100 \
    --depth_profile_interval 100 \
    --vis_index 0 \
    --use_attenuation \
    --mu_scale 0.01 \
    --step_len 1.0 \
    --w_act 0.0 \
    --w_reg_sigma 0.0 \
    "$@"

echo "✅ Training finished at: $(date)"
