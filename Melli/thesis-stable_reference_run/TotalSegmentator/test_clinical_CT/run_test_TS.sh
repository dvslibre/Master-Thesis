#!/bin/bash
# ------------------------------------------------------------------------------
# SLURM-Job: TotalSegmentator-Workflow auf Cluster
# - Lädt eine CT-DICOM-Serie
# - konvertiert/optional resampled via main.py (SimpleITK)
# - führt TotalSegmentator aus (GPU, falls vorhanden)
# - erzeugt Overlay-Previews der Mittelschnitte
# ------------------------------------------------------------------------------

#SBATCH --nodes=1                  # ein Knoten
#SBATCH --ntasks=1                 # ein Task
#SBATCH --cpus-per-task=16         # CPU-Threads für Python/ITK usw.
#SBATCH --job-name=test_TS
#SBATCH --output=/home/mnguest12/slurm/test.%j.out
#SBATCH --error=/home/mnguest12/slurm/test.%j.err
#SBATCH --gres=gpu:1               # eine GPU anfordern

echo "GPUs: ${SLURM_JOB_GPUS}"

# --- Projektpfad --------------------------------------------------------------
PROJ_DIR="/home/mnguest12/projects/thesis/TotalSegmentator/test_clinical_CT"
cd "${PROJ_DIR}"

# --- Eingangsserie (DICOM) ----------------------------------------------------
# DICOM-Ordner der gewünschten Serie + zugehörige SeriesInstanceUID
INPUT_DICOM_DIR="${PROJ_DIR}/manifest-1761144529424/Adrenal-ACC-Ki67-Seg/Adrenal_Ki67_Seg_039/02-08-2010-NA-CT Abdomen-18373/3.000000-3 AX WO-61678"
SERIES_UID="1.3.6.1.4.1.14519.5.2.1.169260120790039889716250759344671861678"

# --- Ausgabe & Optionen -------------------------------------------------------
OUT_DIR="${PROJ_DIR}/run_out"
RESAMPLE_MM=1.0           # isotropes Resampling in main.py (z.B. 1.0 mm)
DEVICE="cuda"             # wird in main.py auf 'gpu'/'gpu:X' gemappt
FAST=0                    # 1 → schneller TS-Mode

# --- Conda-Umgebung aktivieren ------------------------------------------------
# Benötigt: env "totalseg" mit SimpleITK, nibabel, TotalSegmentator etc.
source /home/mnguest12/mambaforge/etc/profile.d/conda.sh
conda activate totalseg

# --- Kurzcheck: SimpleITK verfügbar? -----------------------------------------
python - <<'PY'
try:
    import SimpleITK as sitk
    print("[CHECK] SimpleITK", sitk.Version_VersionString())
except Exception as e:
    import sys
    print("[ERROR] SimpleITK fehlt:", e, file=sys.stderr); sys.exit(1)
PY

# Ab hier bei Fehlern abbrechen
set -e
mkdir -p "${OUT_DIR}"

# --- TotalSegmentator-Run via main.py ----------------------------------------
# Argumentliste dynamisch aufbauen (resample & fast optional)
ARGS=( --dicom_dir "${INPUT_DICOM_DIR}" --series_uid "${SERIES_UID}" \
       --out_dir "${OUT_DIR}" --device "${DEVICE}" )
if [[ -n "${RESAMPLE_MM}" ]]; then ARGS+=( --resample_mm "${RESAMPLE_MM}" ); fi
if [[ "${FAST}" == "1" ]]; then ARGS+=( --fast ); fi
ARGS+=( --keep_intermediate )  # work/-Ordner behalten (für Previews/Fallbacks)

echo "Running: python main.py ${ARGS[*]}"
# srun sorgt dafür, dass der Prozess unter SLURM-Ressourcen läuft (GPU/CPUs)
srun python main.py "${ARGS[@]}"

# --- Overlays erzeugen --------------------------------------------------------
# Bevorzugt das resample-CT; sonst Roh-CT; wenn beides fehlt, wird ein CT auf
# die TS-Label-Geometrie resampled (Fallback).
HU_ISO="${OUT_DIR}/work/ct_input_iso${RESAMPLE_MM}mm.nii.gz"
HU_RAW="${OUT_DIR}/work/ct_input.nii.gz"
SEG_DIR="${OUT_DIR}/ts_output"
PREV_DIR="${OUT_DIR}/ts_preview"

if [[ ! -f "${HU_ISO}" && ! -f "${HU_RAW}" ]]; then
  echo "[INFO] Kein work-CT gefunden, baue ct_match_ts.nii.gz als Fallback…"
  # Mini-Python-Snippet: CT aus DICOM lesen und auf Label-Grid resamplen
  python - <<PY
import SimpleITK as sitk, sys
from pathlib import Path
dicom_dir = Path("${INPUT_DICOM_DIR}")
ts_dir    = Path("${SEG_DIR}")
out_path  = Path("${OUT_DIR}/ct_match_ts.nii.gz")
# irgendeine TS-NIfTI-Datei als Referenz-Geometrie nehmen
cands = list(ts_dir.glob("*.nii*")) + list(ts_dir.rglob("*.nii*"))
if not cands: raise SystemExit("Keine NIfTI-Labels im ts_output gefunden.")
label_path = sorted(cands, key=lambda p: len(str(p)))[0]
# DICOM-Serie lesen
r = sitk.ImageSeriesReader(); r.MetaDataDictionaryArrayUpdateOn(); r.LoadPrivateTagsOn()
ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), ids[0])
r.SetFileNames(files)
img = r.Execute()
# CT auf Label-Geometrie resamplen (linear, Luft=-1000 HU)
lbl = sitk.ReadImage(str(label_path))
res = sitk.Resample(img, lbl, sitk.Transform(), sitk.sitkLinear, -1000.0, img.GetPixelID())
res = sitk.Cast(res, sitk.sitkInt16)
out_path.parent.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(res, str(out_path), True)
print("[OK] geschrieben:", out_path)
PY
  HU_FOR_PREVIEW="${OUT_DIR}/ct_match_ts.nii.gz"
else
  # Wenn vorhanden, das isotrop resample nehmen; sonst Roh-CT
  if [[ -f "${HU_ISO}" ]]; then HU_FOR_PREVIEW="${HU_ISO}"; else HU_FOR_PREVIEW="${HU_RAW}"; fi
fi

# Overlay-Previews (axial/koronal/sagittal) erzeugen; Fehler hier brechen den Job nicht ab
mkdir -p "${PREV_DIR}"
echo "[Overlay] HU=${HU_FOR_PREVIEW}  SEG=${SEG_DIR}"
python overlay_ts.py \
  --in_hu "${HU_FOR_PREVIEW}" \
  --seg   "${SEG_DIR}" \
  --outdir "${PREV_DIR}" \
  --alpha 0.35 --filled || true

echo "[DONE] Overlays unter: ${PREV_DIR}"