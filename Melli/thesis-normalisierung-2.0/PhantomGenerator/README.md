# PhantomGenerator

Dieses Repo enthält physikbasierte Vorwärtsprojektion und modellbasierte Quantifizierung für Phantomdaten.

Zentrale Komponenten:
- `qplanar.py`: QPlanar-ähnliche Organquantifizierung mit NNLS
- `stratos.py`: älterer/alternativer LGS+NNLS-Workflow (auch Forward-only nutzbar)
- `eval_qplanar_results.py`: Aggregation und Auswertung mehrerer QPlanar-Läufe

## 1. Ziel und Funktionsweise

Die Pipeline arbeitet auf Phantom-Volumina (`mask` + `mu`) und simuliert AP/PA-Projektionen mit:
1. Scatter (2D-Gauss pro Slice),
2. Attenuation (`exp(-cumsum(mu * dz))`),
3. kollimatorabhängiger Tiefen-PSF (Kernel pro Tiefe),
4. Summation entlang der Projektionsrichtung.

Für die Quantifizierung wird ein lineares Modell gebaut:
- Systemmatrix `A`: Projektionen von Einheitsaktivität pro Organ
- Messvektor `b`: Projektion des gesamten GT-Aktivitätsvolumens
- Lösung: `min ||Ax-b||` mit Nebenbedingung `x >= 0` (NNLS)

## 2. Struktur

- `qplanar.py`: Hauptskript mit CLI, inklusive optionalem Robustness-Suite-Modus
- `run_qplanar.sh`: Slurm-Wrapper für mehrere Phantome
- `stratos.py`: Funktionen `run_lgs_and_nnls(...)` und `run_forward_projection(...)`
- `run_stratos.sh`: Slurm-Beispielaufruf für `stratos.py`
- `eval_qplanar_results.py`: fasst `quantification_results.json` über viele Läufe zusammen
- `LEAP_Kernel.mat`: PSF-Kernel
- `organ_ids.txt`: Mapping Organname -> Label-ID

## 3. Eingaben

`qplanar.py` erwartet:
- `--base`: Phantom-Basisordner (typisch aus `Data_Processing/phantom_xx`)
- `--spect_bin`: Attenuation-BIN
- `--mask_bin`: Masken-BIN (Organ-IDs)
- `--meta_json`: `meta_simple.json` mit `organ_activity_info`
- `--kernel_mat`: LEAP-Kernel
- `--shape`: z. B. `256,256,651`

Optional wichtig:
- `--poisson --counts_per_pixel ...` (Rauschsimulation auf `b`)
- `--exclude_organs ...` (Default: `small_intest`)
- `--mask-robustness-suite` (baseline + Dilation + Shifts)
- `--save_pngs`

## 4. Beispiel: Einzelner QPlanar-Lauf

```bash
cd /home/mnguest12/projects/thesis/PhantomGenerator
python3 qplanar.py \
  --base /home/mnguest12/projects/thesis/Data_Processing/phantom_16 \
  --spect_bin /home/mnguest12/projects/thesis/Data_Processing/phantom_16/src/phantom_16_spect208keV.par_atn_1.bin \
  --mask_bin /home/mnguest12/projects/thesis/Data_Processing/phantom_16/src/phantom_16_mask.par_act_1.bin \
  --meta_json /home/mnguest12/projects/thesis/Data_Processing/phantom_16/out/meta_simple.json \
  --kernel_mat /home/mnguest12/projects/thesis/PhantomGenerator/LEAP_Kernel.mat \
  --shape 256,256,651 \
  --pixel_size_mm 1.5 \
  --poisson \
  --counts_per_pixel 30 \
  --out_dir qplanar_results/phantom_16 \
  --save_pngs
```

## 5. Robustness-Suite (Maskenfehler simulieren)

Mit `--mask-robustness-suite` laufen automatisch:
- `baseline`
- `dilation2d_r1_xy`
- `shift_x1cm...`
- `shift_x2cm...`

Konvention im Code:
- `b` bleibt baseline-basiert,
- `A` wird mit pertubierter Maske aufgebaut.

So lässt sich Sensitivität gegen Segmentierungsfehler testen.

## 6. Outputs von `qplanar.py`

Im angegebenen `out_dir`:
- `quantification_results.json` (GT/REC/RelErr pro Organ)
- optional PNGs (`proj_GT_AP.png`, `proj_GT_PA.png`, `proj_Rec_AP.png`, `proj_Rec_PA.png`)
- bei Suite: Unterordner je Variante + `mask_robustness_suite_summary.json`

## 7. Batch-Ausführung per Slurm

`run_qplanar.sh` verarbeitet mehrere Phantome (Default `16 24 30`) und liest Daten aus `Data_Processing`.

Typischer Start:
```bash
cd /home/mnguest12/projects/thesis/PhantomGenerator
sbatch run_qplanar.sh
```

Nützliche Env-Overrides:
- `PHANTOM_LIST="16 17 18"`
- `MASK_ROBUSTNESS_SUITE=0` (nur baseline)
- `KERNEL_MAT=/pfad/zu/LEAP_Kernel.mat`

## 8. Ergebnisse aggregieren

`eval_qplanar_results.py` sucht rekursiv nach `quantification_results.json` und erzeugt Tabellen/Plots.

```bash
cd /home/mnguest12/projects/thesis/PhantomGenerator
python3 eval_qplanar_results.py \
  --root /home/mnguest12/projects/thesis/PhantomGenerator/qplanar_results \
  --out /home/mnguest12/projects/thesis/PhantomGenerator/qplanar_eval
```

Outputs:
- `all_results.csv`
- `summary_per_organ.csv`
- `summary_per_organ.tex`
- `gt_vs_rec.png`
- `rel_error_boxplot.png`
- `bland_altman.png`

## 9. `stratos.py` (Legacy/Alternative)

`stratos.py` bietet:
- `run_lgs_and_nnls(...)`: kompletter Quantifizierungsdurchlauf
- `run_forward_projection(...)`: nur Vorwärtsprojektion

Ein Beispielaufruf liegt in `run_stratos.sh` (über eingebetteten Python-Block).

## 10. Häufige Fehler

- `FileNotFoundError` bei BIN/Meta:
  - Pfade prüfen (`base/src` und `base/out/meta_simple.json`)
- Falsche Geometrie:
  - `--shape` muss exakt zum BIN-Export passen
- Schlechte Quantifizierung:
  - `organ_ids.txt` und Maskenlabel-Konsistenz prüfen
  - ausgeschlossene Organe (`--exclude_organs`) beachten
- Fehlende Pakete:
  - mindestens `numpy`, `scipy`, `matplotlib`, `pandas` (für Evaluation)
