#!/bin/bash
#SBATCH --job-name=postproc_noiseK_trainfwd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/mnguest12/slurm/postprocessing.%j.out
#SBATCH --error=/home/mnguest12/slurm/postprocessing.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pieNeRF

set -euo pipefail

# -----------------------------
# User knobs
# -----------------------------
ROOT="/home/mnguest12/projects/thesis/pieNeRF"
SWEEP_ROOT="${SWEEP_ROOT:-${ROOT}/results_spect_test_1/noiseK_sweep_run_714}"

# Optional: explicit tags (space-separated), otherwise auto-discover
SWEEP_TAGS="${SWEEP_TAGS:-}"
DRY_RUN="${DRY_RUN:-0}"

MANIFEST="${MANIFEST:-${ROOT}/data/manifest_abs.csv}"
DEVICE="${DEVICE:-cuda}"
FORCE_USE_ATTENUATION="${FORCE_USE_ATTENUATION:-1}"

# Robust mask path pattern: keep exactly one placeholder {phantom}
MASK_PATTERN="/home/mnguest12/projects/thesis/Data_Processing/{phantom}/out/mask.npy"

PRED_SLICES_DIR="${PRED_SLICES_DIR:-test_slices}"
DEFAULT_PRED_ACT_PATTERN="${PRED_SLICES_DIR}/{phantom}/activity_pred.npy"
PRED_ACT_PATTERN="${PRED_ACT_PATTERN:-${DEFAULT_PRED_ACT_PATTERN}}"

# We use baseline checkpoint for train-forward rendering (as in your log)
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-${ROOT}/results_spect_test_1/run_661}"
BASELINE_CKPT="${BASELINE_CKPT:-${BASELINE_RUN_DIR}/checkpoints/checkpoint_step08000.pt}"

# Projection metrics target mode (new flag you added)
PROJ_METRICS_TARGET="${PROJ_METRICS_TARGET:-noisy_counts}"  # or clean_counts

# Conda
CONDA_ENV="${CONDA_ENV:-totalseg}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-/home/mnguest12/mambaforge/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "🛠 Starting sweep postprocessing"
echo "ROOT=${ROOT}"
echo "SWEEP_ROOT=${SWEEP_ROOT}"
echo "MANIFEST=${MANIFEST}"
echo "BASELINE_CKPT=${BASELINE_CKPT}"
echo "PROJ_METRICS_TARGET=${PROJ_METRICS_TARGET}"
echo "FORCE_USE_ATTENUATION=${FORCE_USE_ATTENUATION}"
echo "DRY_RUN=${DRY_RUN}"

source "${CONDA_ACTIVATE}"
conda activate "${CONDA_ENV}"

test -d "${SWEEP_ROOT}" || { echo "[FATAL] SWEEP_ROOT not found: ${SWEEP_ROOT}"; exit 2; }
test -f "${MANIFEST}" || { echo "[FATAL] MANIFEST not found: ${MANIFEST}"; exit 2; }
test -f "${BASELINE_CKPT}" || { echo "[FATAL] BASELINE_CKPT not found: ${BASELINE_CKPT}"; exit 2; }

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

ok=0
skip=0
failed=0
would_run=0
declare -A SKIP_REASONS=()

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

log_skip() {
  local reason="$1"
  local message="$2"
  echo "[SKIP] ${message}"
  skip=$((skip + 1))
  SKIP_REASONS["${reason}"]=$(( ${SKIP_REASONS["${reason}"]:-0} + 1 ))
}

normalize_pred_act_pattern() {
  local p="$1"
  # Auto-fix common typo: "{phantom/activity_pred.npy}" -> "{phantom}/activity_pred.npy"
  if [[ "${p}" == *"{phantom/"* ]]; then
    local fixed="${p/\{phantom\//\{phantom\}\/}"
    echo "[WARN] Auto-fixing malformed PRED_ACT_PATTERN: ${p} -> ${fixed}" >&2
    p="${fixed}"
  fi
  # Enforce required placeholder token.
  if [[ "${p}" != *"{phantom}"* ]]; then
    echo "[WARN] Invalid PRED_ACT_PATTERN (missing {phantom}): ${p}" >&2
    echo "[WARN] Falling back to default: ${DEFAULT_PRED_ACT_PATTERN}" >&2
    p="${DEFAULT_PRED_ACT_PATTERN}"
  fi
  echo "${p}"
}

PRED_ACT_PATTERN="$(normalize_pred_act_pattern "${PRED_ACT_PATTERN}")"

# -----------------------------
# Discover run dirs
# -----------------------------
declare -a TAG_DIRS=()
RUN_NAME_RE='^noiseK_[0-9.]+_seed_[0-9]+$'

if [[ -n "${SWEEP_TAGS}" ]]; then
  for tag in ${SWEEP_TAGS}; do
    TAG_DIRS+=("${SWEEP_ROOT}/${tag}")
  done
else
  # Auto-discover only true sweep runs:
  # 1) dir name matches ^noiseK_[0-9.]+_seed_[0-9]+$
  # 2) split.json exists
  while IFS= read -r d; do
    tag="$(basename "${d}")"
    if [[ ! "${tag}" =~ ${RUN_NAME_RE} ]]; then
      log_skip "name_mismatch" "Ignoring helper/meta dir (name mismatch): ${d}"
      continue
    fi
    if [[ ! -f "${d}/split.json" ]]; then
      log_skip "split_json_missing" "Ignoring ${tag}: split.json missing (${d}/split.json)"
      continue
    fi
    TAG_DIRS+=("${d}")
  done < <(find "${SWEEP_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [[ ${#TAG_DIRS[@]} -eq 0 ]]; then
  echo "[FATAL] No valid sweep run dirs found under ${SWEEP_ROOT}"
  exit 3
fi

# -----------------------------
# Loop over sweep tags
# -----------------------------
for tag_dir in "${TAG_DIRS[@]}"; do
  SWEEP_TAG="$(basename "${tag_dir}")"
  RUN_DIR="${tag_dir}"
  SPLIT_JSON="${RUN_DIR}/split.json"

  echo ""
  echo "============================================================"
  echo "🔁 Processing ${SWEEP_TAG}"
  echo "run_dir=${RUN_DIR}"
  echo "============================================================"

  if [[ ! -d "${RUN_DIR}" ]]; then
    log_skip "run_dir_missing" "Run dir missing: ${RUN_DIR}"
    continue
  fi

  if [[ -z "${SWEEP_TAGS}" && ! "${SWEEP_TAG}" =~ ${RUN_NAME_RE} ]]; then
    # Defensive guard in case discover logic changes.
    log_skip "name_mismatch" "Skipping ${SWEEP_TAG}: does not match run pattern ${RUN_NAME_RE}"
    continue
  fi

  if [[ ! -f "${SPLIT_JSON}" ]]; then
    echo "[WARN] split.json missing: ${SPLIT_JSON}"
    log_skip "split_json_missing" "Skipping ${SWEEP_TAG}: split.json missing"
    continue
  fi

  # Extract config from command.sh (fallback to default config)
  CMD_SH="${RUN_DIR}/command.sh"
  CONFIG=""
  if [[ -f "${CMD_SH}" ]]; then
    CONFIG="$(grep -oE -- '--config[[:space:]]+[^[:space:]]+' "${CMD_SH}" | head -n1 | awk '{print $2}' || true)"
  fi
  if [[ -z "${CONFIG}" ]]; then
    CONFIG="configs/spect_encoderNorm_projCounts.yaml"
    echo "[WARN] Could not extract --config from command.sh; using fallback: ${CONFIG}"
  fi
  if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] CONFIG not found: ${CONFIG}"
    failed=$((failed + 1))
    continue
  fi

  # Slice directory check
  if [[ ! -d "${RUN_DIR}/${PRED_SLICES_DIR}" ]]; then
    echo "[WARN] Slice directory missing: ${RUN_DIR}/${PRED_SLICES_DIR}"
    log_skip "test_slices_missing" "Skipping ${SWEEP_TAG}: ${PRED_SLICES_DIR}/ missing"
    continue
  fi

  OUT_DIR="${RUN_DIR}/postproc_trainfwd_${PRED_SLICES_DIR}"
  mkdir -p "${OUT_DIR}"

  # Skip if already done
  if [[ -f "${OUT_DIR}/metrics.csv" ]]; then
    log_skip "metrics_exists" "metrics.csv already exists: ${OUT_DIR}/metrics.csv"
    continue
  fi

  echo "config=${CONFIG}"
  echo "checkpoint(baseline)=${BASELINE_CKPT}"
  echo "pattern=${PRED_ACT_PATTERN}"
  echo "out_dir=${OUT_DIR}"
  echo "proj_metrics_target=${PROJ_METRICS_TARGET}"
  echo "mask_pattern=${MASK_PATTERN}"

  cmd=(
    srun /usr/bin/time -v "${PYTHON_BIN}" -u postprocessing.py
    --run-dir "${RUN_DIR}"
    --split-json "${SPLIT_JSON}"
    --manifest "${MANIFEST}"
    --config "${CONFIG}"
    --out-dir "${OUT_DIR}"
    --mask-path-pattern "${MASK_PATTERN}"
    --device "${DEVICE}"
    --pred-act-per-phantom
    --pred-act-pattern "${PRED_ACT_PATTERN}"
    --proj-forward-model train
    --checkpoint "${BASELINE_CKPT}"
    --render-projections
    --save-proj-npy
    --save-proj-png
    --proj-metrics-target "${PROJ_METRICS_TARGET}"
    --timing
    --save-act-compare-5slices
    --skip-plots
  )
  if is_truthy "${FORCE_USE_ATTENUATION}"; then
    cmd+=(--force-use-attenuation)
  fi

  if is_truthy "${DRY_RUN}"; then
    echo "[DRY-RUN] Would execute:"
    printf '  %q' "${cmd[@]}"
    echo ""
    would_run=$((would_run + 1))
    continue
  fi

  set -x
  if "${cmd[@]}"; then
    set +x
    ok=$((ok + 1))
  else
    set +x
    echo "[ERROR] Postprocessing failed for ${SWEEP_TAG}"
    failed=$((failed + 1))
  fi
done

echo "============================================================"
echo "✅ Sweep postprocessing finished"
echo "ok=${ok} skip=${skip} failed=${failed} would_run=${would_run}"
if [[ ${#SKIP_REASONS[@]} -gt 0 ]]; then
  echo "skip_reasons:"
  for reason in "${!SKIP_REASONS[@]}"; do
    echo "  - ${reason}: ${SKIP_REASONS[$reason]}"
  done
fi
echo "============================================================"

if [[ ${failed} -ne 0 ]]; then
  exit 1
fi
