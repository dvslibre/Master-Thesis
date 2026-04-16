# pieNeRF – Überblick & Referenz

Dokumentation in zwei Teilen: Erst der logische Gesamtblick, dann ein technisches Nachschlagewerk pro Datei.

---

## 1) Wie funktioniert pieNeRF logisch?

- **Grundidee**  
  pieNeRF rekonstruiert ein 3D-Emissionsfeld (z. B. SPECT-Aktivität) aus zwei Projektionen (AP/PA). Ein NeRF-MLP gibt pro 3D-Punkt eine Emissionsstärke aus. Entlang jedes Rays wird diese Emission integriert und mit den gemessenen Projektionen verglichen. Optional dämpft ein CT-Volumen die Emission (Attenuation), und ein optionales act-Volumen liefert Zusatz-Supervision.

- **Prinzip des Emission-NeRF**  
  Für jeden Strahl werden gleichmäßig verteilte Samplepunkte zwischen near/far erzeugt. Das MLP erhält positional encodete Koordinaten plus einen latenten Code `z` (Geometrie/Appearance). Der MLP-Output ist nur die Emission `e(x)` (kein RGB). Die Projektion entsteht als Line-Integral ∑ e(xᵢ)·Δsᵢ. Falls Attenuation aktiv ist, wird zusätzlich μ(xᵢ) aus dem CT interpoliert und pro Segment mit exp(-∫μ ds) gewichtet.

- **Rays, Samples, MLP, Integration, CT-Attenuation**  
  - **Rays**: Orthografisch (parallele Strahlen) mit fixer AP-/PA-Pose; Trainingsmodus nutzt Teilraster (Patch-Sampling), Eval rendert das volle Bild.  
  - **Sampling**: Gleichmäßige z-Samples entlang jedes Rays; optional Jitter (perturb) und Hierarchical Sampling (nicht standardmäßig genutzt).  
  - **MLP**: 8×256 ReLU-NeRF, optional zweites Fine-Netz (nicht standardmäßig drin); nimmt Positional Encoding + latente Features (z, ggf. z_appearance).  
  - **Integration**: Softplus auf Emission, Segmentlängen Δs berücksichtigen Ray-Länge; Summe ergibt Projektion, plus Disparity/Acc als Nebenprodukte.  
  - **CT-Attenuation (optional)**: CT wird in denselben Würfel [-radius, radius]³ gelegt und trilinear abgetastet; Gewichtung e·T·Δs mit Transmission T=exp(-∑μ·Δs).

- **Datenfluss Dataset → Generator → Rendering → Loss → Backprop**  
  1) **Dataset** (`SpectDataset`) liefert AP/PA-Bilder (optional CT/act) als Tensoren.  
  2) **Generator** kapselt NeRF und Ray-Sampling, hält AP/PA-Posen, baut CT-Kontext und reicht latent code `z` als Features ins Rendering.  
  3) **Rendering** (`render_minibatch`/`render_from_pose`) erzeugt Rays, sampelt Punkte, läuft durch das MLP, integriert zu proj_map.  
  4) **Loss**: Poisson-NLL zwischen proj_map und GT-Projektionen, optional L1 auf act-Samples, Glättungs-Loss entlang CT-Plateaustrukturen, L2 auf `z`.  
  5) **Backprop**: GradScaler (optional AMP) skaliert den Loss, Optimizer (Adam) updatet NeRF + `z`.

- **Welche Daten fließen wohin?**  
  - **AP/PA**: Flattened Projektionen als Targets; werden ggf. per Bild auf [0,1] normalisiert.  
  - **CT**: Optionales Volumen; geht durch `build_ct_context` in `render_*` und liefert μ-Samples für Attenuation.  
  - **ACT**: Optionales Aktivitäts-Volumen; zufällige Voxel werden abgetastet und mit NeRF-Emission verglichen (L1).  
  - **Latenter Code z**: Trainierbarer Vektor, als `features` an das NeRF angehängt; optional Appearance-Split (`z_appearance`).  
  - **Rays/Samples**: Aus AP/PA-Posen generiert; Trainingspatch vs. Vollbild je nach Modus.

- **Forward-Pass & Training grob**  
  1) Batch laden (AP/PA [+ CT/ACT]), ggf. normalisieren.  
  2) Zufällige Pixel-Indizes ziehen, zugehörige Rays aus Cache wählen.  
  3) Rendering für AP- und PA-Teilrays (mit/ohne CT-Kontext).  
  4) Poisson-NLL berechnen, optionale ACT-/CT-Losses + z-Reg addieren.  
  5) Backward, Grad-Clipping, Optimizer-Step, Logging/CSV.  
  6) Periodisch: Previews rendern, Depth-Profile speichern, Checkpoints schreiben.

- **Training-Mode vs. Eval-Mode**  
  - **Training**: `use_test_kwargs=False`, Perturb/Noise aktiv, Patch-Sampling (FlexGridRaySampler), Gradients an.  
  - **Eval**: `use_test_kwargs=True`, kein Noise/Jitter, FullRaySampler (volle Projektionen), `torch.no_grad()`, Ausgabe inkl. Disparity/Acc und vollständiger Extras; dient Previews/Abschlussrendering.

---

## 2) Zum Nachschlagen: Welche Funktion macht was in welchem Skript?

### Top-Level

**README.md**  
Beschreibung des Projekts (diese Datei).

**run_train_emission.sh**  
Slurm-Wrapper: Aktiviert Conda-Env, wechselt ins Projekt, zeigt GPU-Info, startet `train_emission.py` mit Standard-CLI-Flags. Inputs: keine direkten; konfigurierbar über Script-Parameter. Outputs: Slurm-Logs, Trainingsergebnisse im Config-Outdir. Läuft auf GPU-Knoten.

**configs/spect.yaml**  
YAML-Config mit Datenpfaden (manifest), Bild-/FOV-Parametern, NeRF/Training-Settings (Chunkgrößen, Attenuation-Flag, N_samples, LR, Scheduler, z-Dim). Wird von `train_emission.py` und `test_dataset.py` geladen.

**__init__.py (Projektwurzel)**  
Leer, dient nur als Paketmarker.

### Training & Evaluierung

**train_emission.py**  
Mini-Trainingsskript für das Emission-NeRF.  
Funktionen/Konstanten:  
- `__VERSION__`: String für Logging.  
- `parse_args()`: CLI-Parser (Config-Pfad, Steps, Rays/Step, Preview/Save-Intervalle, Normalisierung, bg-/act-/ct-/z-Reg-Flags, Debug-Optionen, Seed).  
- `set_seed(seed)`: Setzt Seeds für Torch/NumPy/CUDA.  
- `save_img(arr, path, title=None)`: Robust PNG-Speicherer mit Log-Stretch; nimmt NumPy-Array, schreibt Datei.  
- `adjust_projection(arr, view)`: Flip+Rotate zur Anzeige wie `data_check.py`; Input 2D-Array, Output gedrehtes Array.  
- `poisson_nll(pred, target, eps, clamp_max, weight)`: Poisson-NLL über Strahlen, optional gewichtet; Inputs `pred/target` (Tensor ≥0), Output Skalar-Loss.  
- `normalize_counts(x, return_params=False)`: Per-Bild-Min/Max-Normierung; Input Tensor, Output normiertes Tensor (und Min/Max/Flag).  
- `build_loss_weights(target, bg_weight, threshold)`: Optional Strahl-Gewichte für Hintergrund-Dämpfung; gibt Weight-Tensor oder `None` zurück.  
- `build_pose_rays(generator, pose)`: Precomputet alle Rays für Pose, legt sie auf Generator-Device; Output Tensor `[2, H*W, 3]`.  
- `slice_rays(rays_full, ray_idx)`: Wählt Teilrays nach Indizes.  
- `render_minibatch(generator, z_latent, rays_subset, need_raw, ct_context)`: Rendert Teilrays mit Train-/Test-Settings; gibt proj_map-Flat und Extras zurück.  
- `maybe_render_preview(step, args, generator, z_eval, outdir, ct_volume, act_volume, ct_context)`: Rendert und speichert volle AP/PA-Previews und Depth-Profile in Eval-Mode in festen Intervallen.  
- `init_log_file(path)` / `append_log(path, row)`: CSV-Header anlegen bzw. Zeile anhängen.  
- `save_checkpoint(step, generator, z_train, optimizer, scaler, ckpt_dir)`: Speichert Step, z, Optimizer, AMP-Scaler, coarse/fine Netze.  
- `dump_debug_tensor(outpath, tensor)`: Speichert Tensor auf CPU.  
- `compute_psnr(pred, target)`: PSNR-Berechnung.  
- `sample_act_points(act, nsamples, radius)`: Zieht zufällige Voxel aus act-Volumen, gibt Koordinaten (Welt) + Werte zurück; erwartet `act` als [D,H,W] oder [1,D,H,W].  
- `query_emission_at_points(generator, z_latent, coords)`: Fragt NeRF an spezifischen Weltpunkten ab; Output Emissionswerte.  
- `idx_to_coord(idx, size, radius)`: Mappt Index → Weltkoordinate [-radius, radius].  
- `normalize_curve(arr)`: Normiert 1D-Kurve auf [0,1].  
- `save_depth_profile(...)`: Visualisiert Emission/CT/ACT entlang Strahl-Tiefen für ausgewählte Pixel, speichert PNG.  
- `log_attenuation_profile(step, view, extras)`: Druckt λ/μ/T-Verlauf für Debug bei Attenuation.  
- `sample_ct_pairs(ct, nsamples, thresh, radius)`: Sucht z/z+1-Paare mit kleinem Gradient für CT-Glättungs-Loss; Output Koordinatenpaare + Gewichte.  
- `train()`: Hauptroutine: Config laden, Dataset/Generator bauen, AP/PA-Rays cachen, Smoke-Test-Render, Training-Schleife (Ray-Sampling, Rendering, Losses, Backprop, Logging, Previews, Checkpoints), Abschlussrendering und finaler Checkpoint. Device: erwartet CUDA; nutzt GradScaler bei `use_amp`.  
Nutzung: CLI-Einstiegspunkt (`if __name__ == "__main__": train()`), läuft auf GPU; erwartet `configs/spect.yaml` und Daten unter `data/`.

**test_dataset.py**  
Hilfsskript zum Laden/Visualisieren des Datensatzes.  
- `main(config_path)`: Lädt YAML, ruft `get_data`, druckt Shapes/Stats, erzeugt einfache Plots für AP/PA/CT (mit Flip/Rot), speichert `data_check.png`. CPU-basiert; zum schnellen Datencheck.

### Konfiguration & Datenzugriff (`graf/`)

**graf/config.py**  
Config-Helper und Model-Bau.  
- `save_config(outpath, config)`: YAML speichern.  
- `update_config(config, unknown)`: CLI-Overrides (`--data:imsize 128` etc.) in bestehende Config schreiben.  
- `get_data(config)`: Baut `SpectDataset`, liest echte H/W aus AP-Bild, berechnet formale focal aus FOV, setzt radius, gibt Dataset + `hwfr` (H,W,focal,radius) + `render_poses` (None) zurück.  
- `build_models(config)`: Konfiguriert NeRF-Args (Chunks, feat_dim, Appearance-Split, Emission/Attenuation), ruft `create_nerf`, setzt near/far, instanziiert `FlexGridRaySampler`, baut `Generator`, verschiebt auf CUDA.  
- `build_lr_scheduler(optimizer, config, last_epoch)`: Step- oder MultiStep-LR-Scheduler basierend auf `lr_anneal_every` und `lr_anneal`.  
Device: Generator und Netze auf CUDA.

**graf/datasets.py**  
Dataset-Implementierung für SPECT-Phantome/Patienten.  
- `SpectDataset(manifest_path, imsize=None, transform_img=None, transform_ct=None)`: Liest CSV (patient_id, ap_path, pa_path, ct_path, optional act_path), lädt `.npy`-Dateien. AP/PA werden auf Max=1 normiert, CT permutiert nach (D,H,W) und skaliert, ACT permutiert und normiert. Returns in `__getitem__`: Dict mit `ap` [1,H,W], `pa` [1,H,W], `ct` [D,H,W] (oder leerer Tensor), `act` (oder leer), `meta`.  
Device: Daten werden als CPU-Tensoren geliefert, Training verschiebt sie auf CUDA.

**graf/generator.py**  
Wrapper um NeRF-Rendering und Ray-Sampling.  
- Klasse `Generator`: Hält H/W/focal/radius, Sampler, AP/PA-Posen, Device, Render-Kwargs.  
  - `__call__(z, y=None, rays=None)`: Rendert Mini-Batch; wählt Train-/Test-Args, hängt `features=z` an, passt near/far bei Radius-Intervall, ruft `render`; gibt proj_flat (Train) oder (proj_flat, disp_flat, acc_flat, extras) (Eval). Erwartet Rays `[2, N, 3]` und `z` auf GPU.  
  - `decrease_nerf_noise(it)`: Reduziert `raw_noise_std` linear bis Iteration 5000.  
  - `sample_pose()`: Toggle zwischen festen AP/PA-Posen (falls gesetzt), sonst Zufalls-Punkt auf Kugel.  
  - `sample_rays()`: Baut Rays für aktuelle Pose; orthografisch → FullRaySampler (volle Raster), sonst Trainingssampler oder Full.  
  - `to(device)`, `train()`, `eval()`: Gerät/Mode-Handling für coarse/fine Netze.  
  - `set_fixed_ap_pa(radius, up)`: Definiert AP(+Z)/PA(-Z)-Posen, korrigiert Spiegelung, setzt ortho_size=2*radius.  
  - `render_from_pose(z, pose, ct_context)`: Rendert volles Bild aus gegebener Pose, flatten proj/disp/acc; erlaubt CT-Kontext (sonst Attenuation-Off).  
  - `build_ct_context(ct_volume)`: Formatiert CT zu [1,1,D,H,W], flipt x-Achse, speichert radius und value_range; legt Tensor auf Generator-Device.  
Inputs/Outputs: arbeitet auf CUDA, erwartet `z` Batch-Shape [B, feat_dim], Rays von Samplern.

**graf/transforms.py**  
Ray-Sampling- und GT-Patch-Helfer.  
- Klasse `ImgToPatch(ray_sampler, hwf)`: Extrahiert GT-Pixel passend zu gesampelten Rays; nutzt `ray_sampler` und Dummy-Pose, gibt RGBA/Intensitäten pro Ray zurück.  
- Klasse `RaySampler(N_samples, orthographic=False)`: Basis-Sampler; `__call__` baut Rays (perspektivisch oder ortho über `get_rays`/`get_rays_ortho`), ruft `sample_rays` (abstrakt) und gibt `(rays, select_inds, hw)` zurück.  
- Klasse `FullRaySampler`: `sample_rays` → alle Indizes 0..H*W-1 (volles Bild).  
- Klasse `FlexGridRaySampler(N_samples, random_shift, random_scale, min_scale, max_scale, scale_anneal, orthographic=False)`: Erzeugt NxN-Grid in [-1,1], skaliert/verschiebt zufällig (annealbar) für Patch-Sampling; `return_indices=False`, damit `grid_sample` genutzt wird.  
- Funktionen `get_rays`, `get_rays_ortho` werden aus `nerf.run_nerf_helpers_mod` importiert.

**graf/utils.py**  
Kamera-/Sampling-Helfer.  
- `to_sphere`, `sample_on_sphere`: Zufallspunkte auf Einheitskugel (derzeit ungenutzt).  
- `look_at(eye, at, up, eps)`: Berechnet Look-at-Rotationsmatrix; genutzt für Posen in `generator.py`.

### NeRF-Kern (`nerf/`)

**nerf/run_nerf_mod.py**  
Modifizierte NeRF-Pipeline für Emission/Attenuation.  
- `batchify(fn, chunk)`: Chunked-Wrapper zur OOM-Vermeidung.  
- `run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, features, netchunk, feat_dim_appearance)`: Positional Encoding, Anhängen von Features (z), optional Viewdirs/Appearance-Split, Forward durch MLP in Chunks; Output Shape [N_rays, N_samples, out_dim].  
- `batchify_rays(rays_flat, chunk, **kwargs)`: Ruft `render_rays` blockweise auf, konkatenierte Outputs.  
- `sample_ct_volume(pts, context)`: Trilineare Interpolation im CT-Volumen; erwartet `pts` [N_rays, N_samples, 3], `context` mit `volume` [1,1,D,H,W], `grid_radius`. Output μ-Werte [N_rays, N_samples].  
- `render(H, W, focal, chunk, rays, c2w, near, far, use_viewdirs, c2w_staticcam, **kwargs)`: High-Level-Render: Rays erzeugen/verwenden, near/far anhängen, Features per Ray expandieren, chunked rendern, Outputs reshape, proj/disp/acc extrahieren; verwendet `render_rays`. Erwartet NeRF-Netzwerk in `kwargs["network_fn"]` (Device-Referenz).  
- `raw2outputs_emission(raw, z_vals, rays_d, raw_noise_std, pytest, mu_vals, use_attenuation, attenuation_debug)`: Emission-Pfad: Softplus auf Emission, Segmentlängen, optional Noise; bei Attenuation: kumulative μ·Δs → Transmission → Gewichte e·T·Δs; Summen zu proj_map, depth/disp, acc, optional Debug (λ, μ, Δs, T, weights).  
- `raw2outputs(...)`: Dummy, wirft Fehler bei falschem Aufruf (RGB-Pfad).  
- `render_rays(ray_batch, network_fn, network_query_fn, N_samples, features, retraw, lindisp, perturb, N_importance, network_fine, white_bkgd, raw_noise_std, verbose, pytest, emission, **kwargs)`: Zerlegt Rays, sampelt z_vals (optional jitter/disparity), berechnet Punkte, ruft `network_query_fn`; bei Attenuation sammelt μ; Emission-Pfad über `raw2outputs_emission` (inkl. raw-Dump/Debug), optional Fine-Netz; Rückgabe dict mit proj/disp/acc (+ Debug/raw).  
- `create_nerf(args)`: Baut Embedder für Punkte/Viewdirs, NeRF-MLP(s), network_query_fn, render_kwargs_train/test; setzt output_ch=1 bei Emission, appearance-Split, perturb/noise-Settings.  
Gerätehandling: orientiert sich am Device von `network_fn`; Features und Rays werden darauf verschoben.

**nerf/run_nerf_helpers_mod.py**  
Hilfsfunktionen und MLP-Definition.  
- Klasse `Embedder`: Fourier-Features (sin/cos) über mehrere Frequenzen; `embed(x)` konkatenierter Output.  
- `get_embedder(multires, i)`: Liefert Embedder + Output-Dim (oder Identity bei i=-1).  
- Klasse `NeRF(D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs)`: Kern-MLP; Punkt-MLP mit Skip bei Layer 4, optional Viewdir-Zweig (RGB+Alpha) oder direkte Output-Layer (Emission). Initialisiert Output-Layer mit kleinem Bias/Gewicht. Forward splittet pos/view Inputs, gibt Emission oder RGB+Sigma.  
- `get_rays(H, W, focal, c2w)`: Perspektivische Rays (nicht genutzt im orthografischen Setup).  
- `get_rays_ortho(H, W, c2w, size_h, size_w)`: Orthografische parallele Rays mit Gitter in Weltkoordinaten; genutzt für SPECT-Rendering.

**nerf/__init__.py**  
Leer, Paketmarker.

### Weitere Dateien/Ordner

**data/**  
Beispieldaten und `manifest.csv` (Pfad-Listing). `.npy`-Volumes/-Projektionen werden von `SpectDataset` geladen.

**results_spect/**  
Standard-Ausgabeordner (Logs, Checkpoints, Previews, Evaluation). Keine Code-Logik.

**__pycache__/**  
Bytecode-Cache, ohne Relevanz.

**nicht vorhanden: model/spect_pixel_pienerf.py**  
In diesem Repository gibt es keine Datei mit diesem Namen; falls benötigt, müsste sie separat hinzugefügt werden.

---

## Hinweise zur Nutzung
- Training starten: `python train_emission.py --config configs/spect.yaml --max-steps 1000 ...` (siehe `run_train_emission.sh` für sinnvolle Defaults).  
- GPU erforderlich; Attenuation benötigt ein CT-Volumen (`ct.npy`) im Manifest.  
- Previews/CSV/Checkpoints landen standardmäßig unter `results_spect/`.  
- Debug: `--debug-attenuation-ray` loggt λ/μ/T für einen Beispielstrahl; `--debug-zero-var` speichert Tensors, falls Vorhersagen kollabieren.

---

### Datenverarbeitung (Kurzreferenz)
- AP/PA aus `SpectDataset`: `.npy` werden als float32 geladen und pro Bild auf Max=1 normiert; optional per-Bild-Min/Max-Normierung zur Stabilisierung (`--normalize-targets`).  
- CT (`ct.npy`): permutiert zu (D,H,W), ggf. x-Achse geflippt; in `build_ct_context` zu [1,1,D,H,W] auf GPU, Wertebereich gemerkt; bei Attenuation trilinear gesampelt, µ auf ≥0 geklemmt.  
- ACT (`act.npy`): permutiert zu (D,H,W), auf Max=1 normiert; optional zufällige Voxel werden gesampelt, Koordinaten via `idx_to_coord` nach [-radius,radius] skaliert.  
- Rays: Orthografisch, fixe AP/PA-Posen; Trainings-Patches via `FlexGridRaySampler` (Grid in [-1,1], Zufallsscale/-shift), Eval volle Raster. Pixel-Indizes → Weltkoordinaten über NeRF-Ray-Sampling.  
- Emission MLP-Output: Softplus → ≥0; Segmentlängen Δs berücksichtigen Ray-Länge.  
- Attenuation: µ·Δs kumuliert, Transmission T=exp(-∑µΔs) multipliziert Emission; Debug kann λ/µ/T/Dists speichern.  
- Logging/Scales: Poisson-NLL clamp pred∈[1e-8,1e6]; optional bg-Weight dämpft Strahlen mit Target ≤ threshold; CT/ACT-Losses nutzen resampelte Koordinaten und Gewichte; ds/radius in Weltkoordinaten aus Config.
