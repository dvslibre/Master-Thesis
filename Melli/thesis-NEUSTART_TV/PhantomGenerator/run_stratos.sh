#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=stratos
#SBATCH --output=/home/mnguest12/slurm/%x.%j.out
#SBATCH --error=/home/mnguest12/slurm/%x.%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G

# Threads passend setzen
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "=== STRATOS Job auf $(hostname) ==="
echo "GPUs: ${SLURM_JOB_GPUS}"
echo "Start: $(date)"

# Environment
source /home/mnguest12/mambaforge/bin/activate totalseg

cd /home/mnguest12/projects/thesis/PhantomGenerator

# Fester Phantom-Name (ein Lauf)
export PHANTOM_NAME="phantom_05"

python3 - <<'PYCODE'
import os
from stratos import run_lgs_and_nnls

base_dir="/home/mnguest12/projects/thesis/PhantomGenerator"
phantom_name=os.environ.get("PHANTOM_NAME","phantom_05")

res = run_lgs_and_nnls(
    base_dir=base_dir,
    phantom_name=phantom_name,
    dims=(256,256,651),        # ← wichtig: 651 Slices
    sigma=2.0,
    z0_slices=29,
    roi_border_px=16,
    lambda_reg=0.0,
    scalingFactor=1e5,
    results_subdir="results_poisson",
    # Nur Poisson-Noise auf b (falls in stratos.py implementiert)
    realistic_poisson=True,
    counts_per_pixel=20000
)

print("\n=== Ergebnisse ===")
for key in ["rmse", "rrmse", "rrmse_lb", "rrmse_ap", "rrmse_pa"]:
    print(f"{key:>12s}: {res[key]:.6f}")
print("x_true:", res["true_x"])
print("x_est :", res["x_est"])
print("Organe:", res["organs"])
print("Gespeichert unter:", os.path.join(res["phantom_dir"], "results_poisson"))
PYCODE

echo "Ende: $(date)"