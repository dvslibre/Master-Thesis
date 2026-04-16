# RayNet – Überblick & Referenz

Dokumentation in zwei Teilen: erst der logische Gesamtblick, dann ein technisches Nachschlagewerk pro Datei.

---

## 1) Hohe Ebene: Wie funktioniert RayNet logisch?

- **Grundidee**  
  RayNet verarbeitet vorgerenderte SPECT-Rays. Aus einem CT/Masken-Volumen werden AP/PA-Projektionen simuliert, Transmissionen entlang der Rays berechnet und alles als Sequenzen gespeichert. Ein Segmentierungs-Netz (Segment-RayNet) lernt darauf, stückweise konstante Aktivität entlang jedes Rays zu schätzen und die gemessenen Projektionen zu rekonstruieren.

- **Datenfluss: Preprocessing → Rays → Training**  
  1) **Preprocessing**: Rohdaten (µ-Volumen, Organmaske, LEAP-Kernel) werden eingelesen. Aus der Maske wird ein synthetisches Aktivitätsvolumen erzeugt. Eine Gamma-Kamera-Simulation liefert AP/PA-Projektionen. µ/ACT/AP/PA werden als `.npy` gespeichert, Meta als `meta.json`.  
  2) **Ray-Bau**: Für jedes Bildpixel (ggf. subsampled) wird das µ-Profil entlang der Tiefe resampelt (N Samples), Transmissionen AP/PA und gemessene Intensitätspaare (I_AP/I_PA) werden gespeichert. Optional: Aktivität entlang des Rays. Alles landet in `rays_train.npz`.  
  3) **Training**: `raynet_test.py` lädt das NPZ, splittet Rays in Train/Test, und trainiert Segment-RayNet. Das Netz segmentiert µ in wenige Abschnitte, sagt pro Abschnitt eine konstante Aktivität voraus, multipliziert mit Transmission und Schrittweite → rekonstruiert I_AP/I_PA. Loss umfasst Messfehler, optional Supervision auf Aktivität, Regularisierungen.

- **Forward-Pass Segment-RayNet**  
  - Eingaben pro Ray: µ-Profil, Transmissionen T_ap/T_pa, gemessene I_ap/I_pa, Schrittweite ds, optional Ray-Koordinaten (Fourier-Encoded) und GT-Aktivität.  
  - Encoder (1D-Convs) extrahiert Features; FiLM moduliert sie mit globalen Messwerten (und optional Koordinaten).  
  - Kanten-basierte Segmentierung auf µ erzeugt K Segmente; pro Segment wird eine Aktivität geschätzt (Softplus ≥ 0).  
  - Segment-Pooling legt die Aktivität wieder auf N Samples. Physik-Aggregator integriert a·T·ds → I_ap_hat/I_pa_hat.  
  - Loss: Mess-Loss (MAE/MSE/Huber im Linearen oder Log-Raum), optional L2 auf a_gt, Sparsity/BG/Offset/Neighbor/Anti-Korrelation.

- **Train vs. Eval**  
  - **Train**: Gradients an, Batch-Sampling aus NPZ, optional Maske auf aktive Rays, Segmentierung/Kanten adaptiv.  
  - **Eval**: Kein Update, gleiche Forward-Pipeline, nur Metriken/Plots/Heatmaps.  
  - Checkpoint (`segment_raynet_test.pt`) speichert Modellgewichte und Config.

---

## 2) Technisches Nachschlagewerk: Welche Funktion macht was in welchem Skript?

### preprocessing.py  
End-to-end Konverter: Rohvolumina + Kernel rein, `act/mu/mask/ap/pa` + `rays_train.npz` raus. Nur parallele Geometrie. CLI-Einstieg `main()`.
- `ensure_dirs(base)`: Legt `src/data/out` an, liefert Paths.  
- `resolve_into_src(base, src_dir, p)`: Verschiebt relative Dateien ins `src/` (oder passt absolute Pfade durch).  
- Loader: `load_mat_2d` (MATLAB v7.2/7.3), `load_act_bin`, `load_mu_bin_and_align`, `load_mu_nii_and_align`, `load_bin_xyz` – lesen .bin/.nii, permutieren in (D,H,W), optional resamplen.  
- Gamma-Kamera: `_process_view_phys` (Scatter→Attenuation→Kollimator→Summe), `gamma_camera_core` (AP/PA mit stratos-kompatibler Orientierung). Inputs: act/µ (nx,ny,nz), Kernel; Output: AP/PA 2D.  
- `normalize_projections(ap, pa, percentile, clip_to_one, no_clip_low)`: Perzentil-Skalierung mit optionalem Clamping.  
- `build_rays_parallel(ap, pa, mu_dhw, act_dhw, N, subsample, zscore, use_physical_ds, sd_mm, mu_unit)`: Baut Sequenzen pro Ray: µ_seq, Transmissionen AP/PA, I_pairs, optional a_seq, xy_pairs, ds. Inputs Shapes: ap/pa (H,W), mu_dhw (D,H,W).  
- Helpers: `_resample_1d` (lineares Resampling auf N), `convert_mu_units`, `build_activity_from_mask` (synthetische Aktivitäten aus Organmaske + organ_ids.txt).  
- CLI `parse_args()`: Pfade/Shapes/Dtypes, Kernel, Einheiten (per_mm/per_cm), physikalische Schrittweite, Normalisierung, Samplings, Kalibrierung, Seeds.  
- `main()`: Orchestriert Laden, Einheit-Konvertierung, Aktivitätsbau, Gamma-Projektion, Normalisierung, Speichern (`ap/pa/mu/act/mask.npy`, `meta.json`, `rays_train.npz`, optional kalibriertes NPZ). Gerät: NumPy/CPU; benötigt scipy/h5py/nibabel je nach Input.

### check_preprocessing.py  
Sanity-Check für erzeugte Daten in `<base>/data` und `<base>/out`.  
- `parse_args()`: `--base` (Phantom-Ordner).  
- `main()`: Lädt `mu/act/mask/ap/pa` + `rays_train.npz`, druckt Shapes. Speichert Koronalschnitt-Panel (`check_coronal_slices.png`) und AP/PA-Heatmaps (`check_ray_intensities_from_npz.png`). CPU/Matplotlib-only.

### raynet_test.py  
Trainings-/Auswerteskript für Segment-RayNet mit Train/Test-Split. CLI-Einstieg `main()`.  
- Datasets:  
  - `NPZRays(npz_path)`: Lädt `mu_seq/T_ap_seq/T_pa_seq/I_pairs/a_seq/xy_pairs/mask_meas/ds`; setzt Default-Masken (|I|>eps), normiert xy in [-1,1].  
  - `NPZRaysSubset(base, indices)`: Schneidet alle Tensors auf gegebene Indizes, baut neue Masken.  
- Helper: `zscore_1d`, `fourier_encode_xy` (sin/cos-PE für xy), `segment_from_mu` (Kanten-basierte Segmentierung mit min_seg/K_max).  
- Physik: `PhysicsAggregator.forward(a, T_ap, T_pa, ds)`: Summiert a·T·ds → I_ap_hat/I_pa_hat.  
- Modell: `SegmentRayNet(in_ch=3, hidden=64, kernel_size=5, dropout, num_fourier)`: 1D-Conv-Encoder → FiLM (I_ap/I_pa [+ Fourier-xy]) → Segmentierung über µ → Segment-MLP → a_hat → Physik. Optionen: FiLM/Gain deaktivierbar. Outputs: a_eff [B,N], I_ap_hat/I_pa_hat, gain, Segmentanzahlen.  
- Loss/Regs: `meas_loss` (MAE/MSE/Huber, optional log-Raum, Masken), `pearson_corr_per_ray`/`pearson_corr_per_ray_grad` (Korrelation a vs. µ), `neighbor_coherence` (ähnliche Rays koppeln).  
- Training:  
  - `TrainConfig`: Hyperparameter (Epochen, LR, alpha/beta, Masken, edge_tau, min_seg, K_max, Reg-Gewichte, Ablationen, Fourier-xy).  
  - `train_one_epoch(model, loader, opt, cfg)`: Forward+Loss+Backprop+Stats, Gradient-Clipping.  
  - `evaluate_on_loader(model, loader, cfg)`: Nur Metriken.  
  - `main()`: CLI-Args (Datenpfad, Epochen, Loss, Regs, num_rays etc.), Seed, Train/Test-Split (80/20) per Zufall, DataLoader, Modell/Optimizer, Epoch-Loop mit Logs, finale Auswertung, Checkpoint `segment_raynet_test.pt`, Visuals (Ray-Profile Train/Test, Heatmaps). Gerät: CPU/GPU je nach `--device`. Erwartet NPZ aus preprocessing.

### Weitere Dateien

- `segment_raynet_test.pt`: Beispiel-Checkpoint (PyTorch state_dict + Config).  
- `__pycache__/`: Bytecode-Cache, keine Logik.  
- `README.md` (diese Datei): Projektbeschreibung im Ordner `RayNet/code`.

---

### Datenverarbeitung (Kurzreferenz)
- µ-Volumen (`mu.bin`/`mu.npy`/`mu_nii`): Standardlauf (ohne zusätzliche Flags) liest `--mu_bin` als float32, permutiert zu (D,H,W), belässt die Einheit bei 1/mm (`mu_target_unit=per_mm`), clamped ab `clamp_mu_min=0.0`. Wenn `--mu_target_unit per_cm` gesetzt ist (z. B. bei phantom_04/06), wird auf 1/cm skaliert (Faktor 10) – daher kommen die 0.05 → 0.5.  
- Masken-Volumen (`mask.bin`): als int, gleiche Geometrie; `organ_ids.txt` mappt Namen→IDs, daraus wird ein synthetisches Aktivitätsvolumen mit homogenen Zufallswerten je Organ gebaut (Hintergrund=0).  
- Gamma-Kamera-Simulation: scatter (Gauss) an/aus per Flags, Attenuation exp(-cumsum(µ·step_len)) optional, Kollimator-Faltung (LEAP-Kernel) optional; `step_len`=sd_mm (physikalisch), bei per_cm-µ automatisch in cm umgesetzt; Orientierung wie stratos (Rot/Flip).  
- Projektionen AP/PA: aus der Simulation; per Default keine Normierung (`--normalize_projections` fehlt), `proj_scale=1.0`. Nur wenn `--normalize_projections` gesetzt, wird 99.9%-Perzentil skaliert und ggf. geclippt (`clip_to_one`/`no_clip_low`). Gespeichert als float32 `.npy`.  
- Ray-Bau: Für jedes Pixel (y,x) (ggf. `--subsample`) werden µ (in der Ziel-Einheit) und optional Aktivität entlang D auf N Bins resampelt (`_resample_1d`); Transmissionen AP/PA = exp(-∫µ ds) mit ds=1.0 (Voxel) oder physikalisch, wenn `--use_physical_ds`; I_pairs direkt aus AP/PA-Pixeln; optional z-Score auf µ (`--zscore`); Rays ohne Aktivität werden per Schwellwert gefiltert; `xy_pairs` bleiben Pixelkoordinaten.  
- NPZ (`rays_train.npz`): enthält `mu_seq`, `T_ap_seq`, `T_pa_seq`, `I_pairs`, optional `a_seq`, `xy_pairs`, `mask_meas`, `ds`.  
- raynet_test: normiert xy in [-1,1]; Default-Maske: |I|>eps; FiLM-Eingaben (I_ap/I_pa) werden robust skaliert (`_robust_unit` per Quantil); Loss kann im Log-Raum arbeiten (log1p). Globaler Gain (Softplus) + Bias justieren Skalen; Mess-Masken/Background-Thresholds steuern Gewichtung.

**Kurz zusammengefasst (aktueller Standardlauf):**  
- µ wird beim Preprocessing auf 1/cm skaliert (per_mm → per_cm), danach nur clamp≥0, keine z-Score-Normierung (weil `--use_physical_ds`).  
- Aktivität bleibt roh (synthetisch aus Maske), nur Rays mit max(a)<1e-3 werden verworfen.  
- AP/PA bleiben roh (keine Perzentil-Normierung, `proj_scale=1.0`), gehen so in `I_pairs`.  
- Im Modell: I_AP/I_PA für FiLM intern robust skaliert, sonst roh im Loss; xy für FiLM nach [-1,1].
