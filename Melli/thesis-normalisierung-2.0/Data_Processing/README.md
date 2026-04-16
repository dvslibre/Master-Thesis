# Data_Processing

Dieses Repo erzeugt aus XCAT-BIN-Dateien die Trainingsdaten für `pieNeRF`.

Hauptskript:
- `preprocessing.py`: liest BIN-Volumina, erzeugt Aktivität, simuliert AP/PA-Projektionen, speichert `.npy` und Metadaten

Hilfsskripte:
- `check_preprocessing.py`: visuelle Sanity-Checks
- `fit_mu_mapping.py`: Fit einer empirischen Abbildung `mu_208 = f(mu_80)`

## 1. Eingabe- und Ordnerstruktur

Erwartet pro Phantom:

```text
phantom_XX/
  src/
    <spect_bin>
    <ct_bin>
    <mask_bin>
  out/   (wird automatisch angelegt)
```

Zusätzlich im `Data_Processing`-Wurzelordner:
- `organ_ids.txt`
- `LEAP_Kernel.mat`

## 2. Was `preprocessing.py` konkret macht

Pipeline:
1. BIN-Dateien laden (`spect`, `ct`, `mask`) mit gegebener Shape und Speicherordnung (`F`/`C`).
2. Aus `mask` wird ein Aktivitätsvolumen (`act.npy`) in `kBq/mL` erzeugt (organabhängige Wertebereiche).
3. µ-Einheiten werden ggf. konvertiert (`per_mm` <-> `per_cm`).
4. Aktivität wird in `MBq/voxel` umgerechnet (über `sd_mm`).
5. Gamma-Kamera-Forwardmodell:
   - Scatter (Gauss pro Slice)
   - Attenuation entlang Projektionsrichtung
   - Kollimatorfaltung mit LEAP-Kernel
   - AP/PA-Projektionssummation
6. Projektionen werden in erwartete Counts umgerechnet (`sensitivity * acq_time`).
7. AP/PA werden robust per gemeinsamem p99.9-Faktor normalisiert.
8. Volumina werden zusätzlich separat normiert (`*_norm.npy`).
9. Alles wird nach `out/` geschrieben; optional wird ein Manifest-Eintrag (`proj_scale_joint_p99`) aktualisiert.

## 3. Wichtige CLI-Parameter (`preprocessing.py`)

- Pflicht:
  - `--base`
  - `--spect_bin`, `--ct_bin`, `--mask_bin`
  - `--shape`
  - `--kernel_mat`
- Häufig wichtig:
  - `--mu_unit`, `--mu_target_unit`
  - `--sd_mm`
  - `--bin_order`
  - `--activity_seed` (`<0` => Seed aus Phantomname)
  - `--sensitivity_cps_per_mbq`, `--acq_time_s`
  - `--manifest`, `--patient-id`, `--manifest-id-column`
  - `--apply_global_rot90`

## 4. Beispielaufruf

```bash
cd /home/mnguest12/projects/thesis/Data_Processing
python3 preprocessing.py \
  --base /home/mnguest12/projects/thesis/Data_Processing/phantom_28 \
  --spect_bin phantom_28_spect208keV.par_atn_1.bin \
  --ct_bin phantom_28_ct80keV.par_atn_1.bin \
  --mask_bin phantom_28_mask.par_act_1.bin \
  --shape 256,256,651 \
  --spect_dtype float32 \
  --ct_dtype float32 \
  --mask_dtype float32 \
  --mu_unit per_mm \
  --mu_target_unit per_cm \
  --sd_mm 1.5 \
  --kernel_mat LEAP_Kernel.mat \
  --kernel_var kernel_mat \
  --bin_order F \
  --activity_seed -1 \
  --sensitivity_cps_per_mbq 65 \
  --acq_time_s 300 \
  --manifest /home/mnguest12/projects/thesis/pieNeRF/data/manifest_abs.csv \
  --patient-id phantom_28 \
  --manifest-id-column patient_id
```

## 5. Ausgaben in `out/`

- `spect_att.npy`, `ct_att.npy`, `act.npy`
- `spect_att_norm.npy`, `ct_att_norm.npy`, `act_norm.npy`
- `ap_counts.npy`, `pa_counts.npy`
- `ap.npy`, `pa.npy` (normalisierte NN-Inputs)
- `mask.npy`
- `meta_simple.json` (Skalen, Parameter, Aktivitätsinfos)
- `orientation_check/*.png` (schnelle Sichtkontrolle)

## 6. Qualitätssicherung

### 6.1 Visueller Check

```bash
python3 check_preprocessing.py \
  --base /home/mnguest12/projects/thesis/Data_Processing/phantom_28
```

Erzeugt:
- `check_coronal_slices.png`
- `check_projections.png` (falls AP/PA vorhanden)

### 6.2 µ-Mapping-Fit

```bash
python3 fit_mu_mapping.py \
  --base /home/mnguest12/projects/thesis/Data_Processing/phantom_28 \
  --degree 2 \
  --max_samples 200000 \
  --scatter_samples 20000
```

Erzeugt:
- `fit_mu_mapping.png`
- `fit_mu_mapping.json`

## 7. Übergabe an `pieNeRF`

Die generierten Dateien werden im Manifest (`pieNeRF/data/manifest_abs.csv`) referenziert.  
`pieNeRF` liest danach direkt:
- `ap/pa` (normalisiert oder counts-basiert je nach Config),
- `ct_path` (i. d. R. `spect_att.npy`),
- `act_path`,
- optional `mask_path` für Postprocessing.

## 8. Typische Fehler

- `FileNotFoundError` bei `organ_ids.txt` oder `LEAP_Kernel.mat`:
  - die Dateien müssen im übergeordneten `Data_Processing`-Ordner liegen
- falsche Shape/Order:
  - `--shape` und `--bin_order` müssen zum Exportformat passen
- fehlende Python-Module:
  - mindestens `numpy`, `scipy`, `matplotlib` installieren/Umgebung aktivieren
