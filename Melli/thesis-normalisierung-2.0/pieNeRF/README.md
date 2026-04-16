# pieNeRF

`pieNeRF` trainiert ein Emission-NeRF für AP/PA-SPECT-Projektionen und rekonstruiert daraus ein 3D-Aktivitätsvolumen.  
Optional kann CT-basierte Schwächung (Attenuation) berücksichtigt und ein ACT-Volumen als Zusatz-Supervision genutzt werden.

## 1. Was das Repo macht

Eingaben pro Fall (über Manifest):
- `ap.npy`, `pa.npy`: 2D-Projektionen
- `ap_counts.npy`, `pa_counts.npy`: Counts (optional, aber empfohlen)
- `spect_att.npy`/`ct.npy`: 3D-µ-Volumen für Attenuation
- `act.npy`: 3D-Aktivitäts-Referenz (optional)
- `mask.npy`: nur für Postprocessing-Metriken

Kernidee:
1. Für AP/PA werden orthografische Rays durch das 3D-Volumen gelegt.
2. Das NeRF-MLP sagt pro Ray-Sample eine nichtnegative Emission vorher.
3. Entlang des Rays wird integriert (`sum(lambda * ds)`), optional mit Transmission `exp(-∫µ ds)`.
4. Die resultierenden Projektionen werden gegen AP/PA-Targets optimiert (Poisson-basiert).
5. Optional kommen ACT- und CT-Regularisierungsterme hinzu.

## 2. Repo-Struktur

- `train_emission.py`: Haupttraining inkl. Split, Logging, Checkpoints, Preview-Renderings
- `configs/spect.yaml`: Standard-Setup (Datenpfade, Geometrie, NeRF- und Trainingsparameter)
- `graf/datasets.py`: CSV/Manifest-Loader und Tensoraufbereitung
- `graf/config.py`: Dataset-/Modellaufbau (`get_data`, `build_models`)
- `graf/generator.py`: AP/PA-Posen, Ray-Erzeugung, Render-Aufrufe
- `nerf/run_nerf_mod.py`: Emission-Rendering, Attenuation, Ray-Integration
- `postprocessing.py`: Metrikberechnung auf Volumen-/Organ-/Projektions-Ebene
- `run_train_emission.sh`: Slurm-Startskript fürs Training
- `run_postprocessing.sh`: Slurm-Startskript für Sweep-Postprocessing

## 3. Voraussetzungen

Benötigt:
- Python 3.10+ (empfohlen)
- CUDA-fähige GPU für Training
- Pakete aus den Imports:
  - `numpy`, `torch`, `torchvision`, `pyyaml`, `matplotlib`
  - für Postprocessing zusätzlich u. a. `scipy` (abhängig vom gewählten Modus)

Hinweis: In der aktuellen Shell-Umgebung waren `numpy`/weitere Pakete nicht installiert. Für die Ausführung muss die passende Conda/venv aktiv sein.

## 4. Datenformat (Manifest)

Standard ist `data/manifest_abs.csv`, z. B. mit Spalten:
- `patient_id`
- `ap_path`, `pa_path`
- `ap_counts_path`, `pa_counts_path`
- `ct_path` (in diesem Projekt meist `spect_att.npy`)
- `act_path`
- `mask_path`
- `proj_scale_joint_p99`

Die Pfade können absolut sein (wie im aktuellen Manifest) oder relativ auflösbar.

## 5. Training: Schritt für Schritt

### 5.1 Basislauf

```bash
cd /home/mnguest12/projects/thesis/pieNeRF
python3 train_emission.py \
  --config configs/spect.yaml \
  --max-steps 8000 \
  --seed 0
```

### 5.2 Typische wichtige Optionen

- Projektions-Loss:
  - `--proj-loss-type poisson`
  - `--proj-loss-weight 0.05`
  - `--proj-warmup-steps 300`
  - `--proj-ramp-steps 3000`
  - `--proj-target-source counts`
- ACT-Loss:
  - `--act-loss-weight ...`
  - `--act-samples ...`
  - `--act-pos-fraction ...`
  - `--act-pos-weight ...`
- CT-Loss:
  - `--ct-loss-weight ...`
- Logging/Save:
  - `--log-every ...`
  - `--save-every ...`
- Hybrid/Encoder-Modus:
  - `--hybrid`

Ein vollständiges Beispiel ist in `run_train_emission.sh`.

### 5.3 Was beim Training intern passiert

1. `spect.yaml` wird geladen, optional von CLI überschrieben.
2. `SpectDataset` liest AP/PA/CT/ACT aus dem Manifest.
3. `Generator` setzt feste AP/PA-Posen und baut orthografische Rays.
4. `render_minibatch` ruft NeRF-Forward auf und integriert entlang der Rays.
5. Hauptterm ist Poisson-NLL auf AP/PA; optionale Zusatzterme werden addiert.
6. Adam-Update, periodisches Logging (`train_log.csv`), Checkpoints, Previews.

## 6. Ausgaben des Trainings

Im `training.outdir` (Default `./results_spect`):
- `checkpoints/checkpoint_stepXXXXX.pt`
- `train_log.csv`
- Preview-Bilder und optionale Debug-Ausgaben
- Split-Informationen (`split.json`) je nach Lauf

## 7. Konfigurationslogik in `configs/spect.yaml`

Wichtige Blöcke:
- `data`:
  - Manifestpfad, AP/PA-Setup, Split nach `patient_id`
  - anisotrope Weltbox (`radius_xyz_cm`) und `voxel_mm`
  - `proj_input_source` (`counts` oder `normalized`)
- `nerf`:
  - `N_samples`, Netzgröße, `use_attenuation`, `atten_scale`
- `training`:
  - `outdir`, Batch/Chunk-Größen, LR, Intervall-Parameter
  - Basis-Gewichte für TV/ACT/CT-Terme

## 8. Postprocessing

`postprocessing.py` wertet Vorhersagen gegen GT aus und kann Projektionen aus Volumen nachrendern.

Beispiel:
```bash
cd /home/mnguest12/projects/thesis/pieNeRF
python3 postprocessing.py \
  --run-dir results_spect \
  --split-json results_spect/split.json \
  --manifest data/manifest_abs.csv \
  --config configs/spect.yaml \
  --out-dir results_spect/postproc \
  --mask-path-pattern /home/mnguest12/projects/thesis/Data_Processing/{phantom}/out/mask.npy \
  --render-projections \
  --save-proj-npy \
  --save-proj-png \
  --device cuda
```

Wichtige Outputs:
- `metrics.csv` (pro Phantom)
- optionale Projektions-NPY/PNGs
- organ-/maskenbasierte Kennzahlen (wenn Masken vorhanden)

## 9. Zusammenspiel mit `Data_Processing`

Dieses Repo erwartet bereits erzeugte `.npy`-Daten.  
Die vorgelagerte Erzeugung passiert in `/home/mnguest12/projects/thesis/Data_Processing`:
- dort werden `ap.npy`, `pa.npy`, `ap_counts.npy`, `pa_counts.npy`, `spect_att.npy`, `act.npy`, `mask.npy` erzeugt
- anschließend referenziert das Manifest diese Dateien für das Training hier

## 10. Fehlersuche (kurz)

- `ModuleNotFoundError`:
  - richtige Conda/venv aktivieren, Pakete nachinstallieren
- CUDA/Memory-Probleme:
  - `training.chunk`, `training.netchunk`, `--act-samples` reduzieren
- Schlechte Projektionsergebnisse:
  - prüfen, ob `proj_input_source` und `--proj-target-source` zusammenpassen
  - prüfen, ob `proj_scale_joint_p99` im Manifest korrekt gesetzt ist
