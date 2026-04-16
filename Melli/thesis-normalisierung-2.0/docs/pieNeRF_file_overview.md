# pieNeRF Uebersicht: wichtigste Python-Dateien

Ziel: Klar zeigen, wo was implementiert ist, welche Datei fuer Feature X relevant ist und welche typischen Fallstricke (Flip/Scaling/Clamp) es gibt. Alle Aussagen basieren auf dem Code.

## Core-Callgraph (train_emission Pfad)
- `pieNeRF/train_emission.py`
  -> `pieNeRF/graf/config.py` (`get_data`, `build_models`)
  -> `pieNeRF/graf/datasets.py` (`SpectDataset`)
  -> `pieNeRF/graf/generator.py` (`Generator`)
  -> `pieNeRF/graf/transforms.py` (`FlexGridRaySampler`, `FullRaySampler`)
  -> `pieNeRF/graf/utils.py` (`look_at` fuer Posen)
  -> `pieNeRF/nerf/run_nerf_mod.py` (`render`, `render_rays`, `raw2outputs_emission`)
  -> `pieNeRF/nerf/run_nerf_helpers_mod.py` (`NeRF`, `get_embedder`, `get_rays_ortho`)
  -> `pieNeRF/utils/ray_split.py` (`make_pixel_split_from_ap_pa`, `sample_train_indices`)

Im Core-Abschnitt unten stehen nur Dateien, die im aktuellen `train_emission`-Pfad tatsaechlich verwendet werden.

## A) Core pieNeRF Emission-NeRF Pfad

### pieNeRF/train_emission.py
**Ziel (1-2 Saetze):**
- Trainingseintritt fuer Emissions-NeRF: Daten laden, Rays sampeln, Loss berechnen, optimieren, loggen, previewen und speichern.

**Worum geht's / was findet man darin? (4-7 Bulletpoints):**
- CLI-Flags fuer Attenuation (`--atten-scale`, `--use_attenuation` indirekt via config), Ray-Split, TV und z-Regularisierung.
- Poisson-NLL fuer Projektionen, plus optionale ACT/CT-basierte Zusatz-Losses.
- Ray-Sampling fuer AP/PA und optionaler stratifizierter Pixel-Split.
- Preview-Renderings (AP/PA) und Depth-Profile als PNG.
- AMP/GradScaler optional via Config.
- z-Latent als trainierbarer Parameter (Optimierung zusammen mit NeRF).

**Kernelemente (Top 3-8 Stellen):**
- `parse_args()` – CLI-Argumente; Input: argv; Output: `argparse.Namespace`.
- `poisson_nll(pred, target, eps=1e-8, clamp_max=1e6, weight=None)` – Poisson-NLL; Input: Tensors; Output: Skalar-Tensor.
- `render_minibatch(generator, z_latent, rays_subset, ct_context=None, return_raw=False)` – rendert Mini-Batch Rays; Output: `(proj_map_flat, extras)`.
- `evaluate_pixel_subsets(...)` – Eval auf Ray-Subsets (Loss/PSNR/MAE); Output: Dict.
- `sample_act_points(act, nsamples, radius, pos_fraction=0.5, pos_threshold=1e-8)` – ACT-Sampling; Output: `(coords, values, pos_flags)`.
- `sample_ct_pairs(ct, nsamples, thresh, radius)` – CT-Paare fuer Glattheitsloss; Output: `(coords1, coords2, weights)` oder `None`.
- `train()` – Hauptloop; Output: None.

Codeauszug (Poisson-NLL):
```python
pred = pred.clamp_min(eps).clamp_max(clamp_max)
nll = pred - target * torch.log(pred)
if weight is not None:
    nll = nll * weight
return nll.mean()
```

**Typische Fragen, die diese Datei beantwortet:**
- "Wo wird der Loss berechnet?"
- "Wo wird z initialisiert und optimiert?"
- "Wo werden Preview-Renderings gespeichert?"
- "Wo werden AP/PA Rays fuer Training gesampelt?"

**Danger Zones / haeufige Bugs:**
- `pa_xflip` und `map_pa_indices_torch` (Pixel-Mapping zwischen AP/PA).
- `atten_scale` und CT-Kontext (Skalierung der Attenuation).
- Flatten-Order (AP/PA `view(-1)` muss zu Ray-Indizes passen).

---

### pieNeRF/graf/config.py
**Ziel (1-2 Saetze):**
- Konfig-Loader und Builder fuer Dataset, Generator, NeRF und Ray-Sampler.

**Worum geht's / was findet man darin? (4-7 Bulletpoints):**
- `get_data()` liest H/W aus echten AP-Bildern und setzt `focal`/`radius`.
- `build_models()` erzeugt NeRF via `create_nerf` und wrappt in `Generator`.
- Konfig-Flags: `data.*`, `nerf.*`, `training.*`, `ray_sampler.*`, `z_dist.*`.
- Orthographic Rendering fuer SPECT.
- LR-Scheduler Helfer (`build_lr_scheduler`).

**Kernelemente (Top 3-8 Stellen):**
- `get_data(config)` – Dataset + `hwfr`; Output: `(SpectDataset, hwfr, render_poses)`.
- `build_models(config)` – Generator + Render-Kwargs; Output: `Generator`.
- `update_config(config, unknown)` – CLI-Overrides; Output: `dict`.
- `build_lr_scheduler(optimizer, config, last_epoch=-1)` – LR-Scheduler; Output: Scheduler.

**Typische Fragen, die diese Datei beantwortet:**
- "Wo wird das Dataset gebaut?"
- "Wo werden NeRF-Modelle erzeugt?"
- "Welche Config-Flags steuern orthographic vs. pinhole?"

**Danger Zones / haeufige Bugs:**
- `data.radius` (tuple vs. float) beeinflusst near/far-Handling.
- `nerf.atten_scale` wird in `render_kwargs_*` propagiert.
- `data.orthographic` schaltet Ray-Setup um.

---

### pieNeRF/graf/datasets.py
**Ziel (1-2 Saetze):**
- Laden von AP/PA-Projektionen und CT/ACT-Volumina via CSV-Manifest.

**Worum geht's / was findet man darin? (4-7 Bulletpoints):**
- `SpectDataset` liest CSV (patient_id, ap_path, pa_path, ct_path, optional act_path).
- AP/PA als `[1, H, W]`, CT/ACT als `[D, H, W]` nach Transpose.
- CT-Scaling mit `vol *= 10.0`.
- ACT optional, skaliert via `act_scale`.
- Projektionen werden pro Bild auf [0,1] normalisiert.

**Kernelemente (Top 3-8 Stellen):**
- `SpectDataset.__init__(manifest_path, imsize=None, transform_img=None, transform_ct=None, act_scale=1.0)` – init; Output: Dataset.
- `SpectDataset.__getitem__(idx)` – lade AP/PA/CT/ACT; Output: Dict.
- `_load_npy_ct(path)` – CT-Load + Transpose + Scale; Output: Tensor `[D,H,W]`.
- `_load_npy_act(path)` – ACT-Load + Transpose + Scale; Output: Tensor `[D,H,W]`.
- `_normalize_projection(tensor)` – per-Projektion Normierung; Output: Tensor.

**Typische Fragen, die diese Datei beantwortet:**
- "Wo werden Projektionen normalisiert?"
- "Wo wird CT skaliert (CT * 10)?"
- "Welche Tensor-Shapes liefert das Dataset?"

**Danger Zones / haeufige Bugs:**
- Transpose-Order `(1, 0, 2)` fuer CT/ACT.
- CT-Scaling `vol *= 10.0` (Skalierung wirkt auf Attenuation).
- ACT kann leer sein (`torch.empty(0)`), muss abgefangen werden.

---

### pieNeRF/graf/generator.py
**Ziel (1-2 Saetze):**
- Rendering-Wrapper fuer NeRF: Rays, feste AP/PA-Posen, CT-Kontext und Rendering-Helfer.

**Worum geht's / was findet man darin? (4-7 Bulletpoints):**
- `Generator.__call__` rendert Rays mit z-Latent und gibt Projektion zurueck.
- `set_fixed_ap_pa` definiert AP/PA-Posen und korrigiert PA-Spiegelung.
- `render_from_pose` rendert eine komplette Projektion (AP oder PA).
- `build_ct_context` packt CT in `[1,1,D,H,W]` inkl. `grid_radius`.
- Orthographic Rays via `FullRaySampler`.

**Kernelemente (Top 3-8 Stellen):**
- `Generator.__call__(z, y=None, rays=None)` – render mit Rays; Output: `proj_flat` oder `(proj_flat, disp_flat, acc_flat, extras)`.
- `Generator.render_from_pose(z, pose, ct_context=None)` – full-frame; Output: `(proj_flat, disp_flat, acc_flat, extras)`.
- `Generator.set_fixed_ap_pa(radius=None, up=(0,1,0))` – Posen-Setup; Output: None.
- `Generator.build_ct_context(ct_volume)` – CT-Kontext; Output: dict mit `volume`, `grid_radius`, `value_range`.

**Typische Fragen, die diese Datei beantwortet:**
- "Wo werden AP/PA-Posen definiert?"
- "Wie wird CT-Kontext fuer Attenuation vorbereitet?"
- "Wo wird der Unterschied train/eval rendering gesteuert?"

**Danger Zones / haeufige Bugs:**
- `pose_pa[:, 0] *= -1.0` (PA-Spiegelkorrektur).
- Tuple-Radius beeinflusst near/far pro Ray.
- `ct_context` nur auf Device des Netzes.

---

### pieNeRF/graf/transforms.py
**Ziel (1-2 Saetze):**
- Ray-Sampling und Patch-Sampling (inkl. orthographic Support) fuer Trainingsrays.

**Worum geht's / was findet man darin? (4-7 Bulletpoints):**
- `RaySampler` baut Rays mit `get_rays` oder `get_rays_ortho`.
- `FullRaySampler` liefert alle Pixel-Indices fuer full-frame.
- `FlexGridRaySampler` erzeugt ein zufaelliges Patch-Grid fuer Training.
- `ImgToPatch` mappt Rays auf GT-Pixelwerte via `grid_sample`.

**Kernelemente (Top 3-8 Stellen):**
- `RaySampler.__call__(H, W, focal, pose)` – rays + indices/grid; Output: `(rays, select_inds, hw)`.
- `FullRaySampler.sample_rays(H, W)` – alle Pixels; Output: Tensor.
- `FlexGridRaySampler.sample_rays(H, W)` – zufaelliges Grid; Output: Tensor `[N,N,2]`.
- `ImgToPatch.__call__(img)` – extrahiert GT-Pixel; Output: Tensor.

**Typische Fragen, die diese Datei beantwortet:**
- "Wo werden Rays/Posen gebaut?"
- "Wie wird Patch-Sampling fuer Rays gemacht?"

**Danger Zones / haeufige Bugs:**
- `grid_sample` erwartet Koordinaten in [-1,1] (Grid-Shape/Range).
- `orthographic` schaltet zwischen `get_rays` und `get_rays_ortho`.

---

### pieNeRF/graf/utils.py
**Ziel (1-2 Saetze):**
- Kamera- und Geometrie-Helfer fuer Posen (v. a. `look_at`).

**Worum geht's / was findet man darin? (4-7 Bulletpoints):**
- `look_at` erzeugt eine Rotationsmatrix fuer eine Kamera, die auf den Ursprung blickt.
- `to_sphere` und `sample_on_sphere` sind im Core-Pfad nicht verwendet (siehe Kommentare) [OPTIONAL/LEGACY].
- Tensor-Inputs sind NumPy Arrays (nicht Torch).

**Kernelemente (Top 3-8 Stellen):**
- `look_at(eye, at=np.array([0,0,0]), up=np.array([0,0,1]), eps=1e-5)` – Blickmatrix; Output: `r_mat` (N,3,3).
- `sample_on_sphere(range_u=(0,1), range_v=(0,1))` – Kugel-Sampling; Output: (3,) [OPTIONAL/LEGACY].
- `to_sphere(u, v)` – (u,v) -> Punkt auf Kugel; Output: (3,) [OPTIONAL/LEGACY].

**Typische Fragen, die diese Datei beantwortet:**
- "Wo wird die look_at-Logik fuer Posen berechnet?"

**Danger Zones / haeufige Bugs:**
- `up`-Vektor und `eps` beeinflussen Stabilitaet der Achsen.

---

### pieNeRF/nerf/run_nerf_mod.py
**Ziel (1-2 Saetze):**
- Kern-Rendering fuer Emission: Sampling entlang Rays, Integration, optionale Attenuation via CT.

**Worum geht's / was findet man darin? (4-7 Bulletpoints):**
- `render` glueing: Rays erzeugen/flatten, Features erweitern, `render_rays` in Chunks.
- `render_rays` sampelt `z_vals` und ruft NeRF MLP via `network_query_fn`.
- `raw2outputs_emission` integriert Emission entlang des Rays (mit optionaler Attenuation).
- `sample_ct_volume` nutzt `torch.nn.functional.grid_sample` fuer CT-Interpolation.
- Flags: `use_attenuation`, `atten_scale`, `N_samples`, `raw_noise_std`, `emission`.

**Kernelemente (Top 3-8 Stellen):**
- `render(...)` – High-Level-Rendering; Output: `[proj_map, disp_map, acc_map, extras]`.
- `render_rays(ray_batch, network_fn, network_query_fn, N_samples, ..., emission=False, **kwargs)` – Ray-Integration; Output: dict mit `proj_map/disp_map/acc_map`.
- `raw2outputs_emission(raw, z_vals, rays_d, raw_noise_std=0.0, pytest=False, mu_vals=None, use_attenuation=False, atten_scale=25.0, return_dists=False)` – Emission + Attenuation; Output: `(proj_map, disp_map, acc_map, tv_base[, dists])`.
- `sample_ct_volume(pts, context)` – CT-Interpolation; Output: `mu` [N_rays, N_samples].
- `create_nerf(args)` – MLPs + render kwargs; Output: `(render_kwargs_train, render_kwargs_test, grad_vars, named_params)`.

Codeauszug (Attenuation Kern):
```python
mu_dists = mu * dists
attenuation = torch.cumsum(mu_dists, dim=-1) * float(atten_scale)
attenuation = F.pad(attenuation[..., :-1], (1, 0), mode="constant", value=0.0)
attenuation = torch.clamp(attenuation, min=0.0, max=60.0)
transmission = torch.exp(-attenuation)
weights = lambda_vals * transmission * dists
```

**Typische Fragen, die diese Datei beantwortet:**
- "Wo wird Attenuation angewendet?"
- "Wo passiert CT-Interpolation (grid_sample)?"
- "Wie werden `z_vals` entlang Rays gesampelt?"

**Danger Zones / haeufige Bugs:**
- `atten_scale` (physikalische Skalierung) und Clamp (0..60).
- `grid_sample` mit `align_corners=True` und `padding_mode="border"`.
- `ct_context.grid_radius` bestimmt Normalisierung von Weltkoordinaten.

---

### pieNeRF/nerf/run_nerf_helpers_mod.py
**Ziel (1-2 Saetze):**
- Positional Encoding, NeRF-MLP und Ray-Helpers inkl. orthographic Rays.

**Worum geht's / was findet man darin? (4-7 Bulletpoints):**
- `Embedder` baut Fourier-Features fuer xyz.
- `NeRF` MLP mit Skip-Connections und optionalem Viewdir-Zweig.
- Emissions-Init: Output-Bias/Weights leicht positiv.
- `get_rays_ortho` erzeugt parallele Rays fuer SPECT.

**Kernelemente (Top 3-8 Stellen):**
- `get_embedder(multires, i=0)` – PosEnc-Factory; Output: `(embed_fn, out_dim)`.
- `Embedder.embed(inputs)` – concat sin/cos; Output: Tensor.
- `NeRF.forward(x)` – MLP-Forward; Output: Tensor `[N, output_ch]`.
- `get_rays_ortho(H, W, c2w, size_h, size_w)` – orthographic rays; Output: `(rays_o, rays_d)`.

**Typische Fragen, die diese Datei beantwortet:**
- "Wo ist das Positional Encoding definiert?"
- "Wie sieht die NeRF-MLP aus?"
- "Wie werden orthographic Rays gebaut?"

**Danger Zones / haeufige Bugs:**
- `get_rays_ortho` invertiert x-Achse (`grid_x = -grid_x`).
- Output-Init im Emissionsfall (Bias/Weights klein positiv).

---

### pieNeRF/utils/ray_split.py
**Ziel (1-2 Saetze):**
- Stratifizierter Train/Test-Split fuer Ray-Indizes (FG/BG) basierend auf AP/PA.

**Worum geht's / was findet man darin? (4-7 Bulletpoints):**
- `PixelSplit` speichert train/test + fg/bg Indizes.
- Threshold-Logik via fester Schwellwert oder Quantil (`thr < 0`).
- Split erfolgt kachelweise (`tile`) mit Seed.
- `sample_train_indices` mischt FG/BG gemaess `fg_frac`.

**Kernelemente (Top 3-8 Stellen):**
- `make_pixel_split_from_ap_pa(target_ap, target_pa, train_frac, tile, thr, seed, pa_xflip, topk_frac=0.10)` – Split; Output: `PixelSplit`.
- `sample_train_indices(split, n, fg_frac, rng)` – Sampling; Output: `np.ndarray`.
- `_resolve_threshold(score_img, thr)` – thr/Quantil; Output: float.

Codeauszug (Threshold-Logik):
```python
if thr < 0.0:
    q = abs(float(thr))
    positives = score_img[score_img > 0]
    if positives.size == 0:
        return 0.0
    return float(np.quantile(positives, q))
```

**Typische Fragen, die diese Datei beantwortet:**
- "Wo ist der Ray-Split implementiert?"
- "Wie wird FG/BG definiert?"

**Danger Zones / haeufige Bugs:**
- `pa_xflip` beeinflusst FG/BG (PA wird gespiegelt).
- `thr` als Quantil (negativ) vs. absolut.
- Tile-Groesse beeinflusst FG-Verteilung.

---

## B) Optionaler Hybrid-Branch (pixelNeRF+pieNeRF)
Anderer Pfad, anderer Architektur-Ansatz (Encoder + forward operator). Nicht Teil des Core-`train_emission`-Pfads.

### pixelNeRF+pieNeRF/train/train_spect_pienerf_pixelnerf.py
**Ziel (1-2 Saetze):**
- Minimaler Trainingsloop fuer das Hybrid-Modell `SpectPixelPieNeRF` mit MSE auf AP/PA.

**Worum geht's / was findet man darin? (4-7 Bulletpoints):**
- CLI-Args fuer Datenpfad, Attenuation, Visuals, Checkpoints.
- DataLoader fuer `src.data.SpectDataset` (optionales Downsampling).
- Training mit `nn.MSELoss` auf AP/PA.
- Optional: ACT-L1 und sigma-L2 Regularisierung.
- Visuals und Depth-Profile als PNG/Log.

**Kernelemente (Top 3-8 Stellen):**
- `get_dataloader(...)` – Dataset + Loader; Output: `(dataset, loader, first_batch)`.
- `make_model(first_batch, device, mu_scale, step_len)` – Modellaufbau; Output: `SpectPixelPieNeRF`.
- `save_visual(...)` – AP/PA Visuals; Output: None.
- `save_depth_profile(...)` – Depth-Profile; Output: bool.
- `train()` – Hauptloop; Output: None.

**Typische Fragen, die diese Datei beantwortet:**
- "Wo wird das Hybrid-Training gestartet?"
- "Wie werden Visuals/Checkpoints gespeichert?"

**Danger Zones / haeufige Bugs:**
- `target_hw`/`target_depth` beeinflussen Volumen- und Projektionsgroesse.
- `use_attenuation`, `mu_scale`, `step_len` beeinflussen Forward-Operator.

---

### pixelNeRF+pieNeRF/model/spect_pixel_pienerf.py
**Ziel (1-2 Saetze):**
- End-to-End Hybrid-Modell: CT/AP/PA -> Encoder -> sigma -> Forward-Operator.

**Worum geht's / was findet man darin? (4-7 Bulletpoints):**
- `CTPixelNeRFEncoder` erzeugt latentes `z_feat`.
- `sample_spect_rays` erzeugt xyz-Punkte fuer AP/PA.
- `get_embedder` fuer positional encoding.
- `PieNeRFConditionalMLP` sagt sigma pro Punkt voraus.
- `forward_spect` integriert sigma (optional Attenuation).

**Kernelemente (Top 3-8 Stellen):**
- `SpectPixelPieNeRF.__init__(...)` – Model-Setup; Output: Modul.
- `SpectPixelPieNeRF.forward(ct, ap, pa, mu_volume=None, use_attenuation=False, step_len=None)` – Forward; Output: dict (`ap_pred`, `pa_pred`, `sigma_volume`).

**Typische Fragen, die diese Datei beantwortet:**
- "Wo wird der Hybrid-Forward definiert?"
- "Wie werden AP/PA zusammengefuehrt?"

**Danger Zones / haeufige Bugs:**
- Annahme `num_samples == H` (AP-Dimension).
- `sigma` reshape/permute Reihenfolge.

---

### pixelNeRF+pieNeRF/model/encoders/ct_pixelnerf_encoder.py
**Ziel (1-2 Saetze):**
- Encoder fuer CT/AP/PA, der ein globales Latent `z_feat` aus PixelNeRF `SpatialEncoder` ableitet.

**Worum geht's / was findet man darin? (4-7 Bulletpoints):**
- Erstellt `SpatialEncoder` (pixel-nerf) und passt ersten Conv auf 5 Kanaele an.
- Baut drei CT-Slices (axial/coronal/sagittal) und concat mit AP/PA.
- Global Average Pooling erzeugt `z_feat` (B, C).
- Abhaengigkeit auf `pixel-nerf/src/model/encoder.py`.

**Kernelemente (Top 3-8 Stellen):**
- `CTPixelNeRFEncoder.__init__(backbone="resnet18", num_layers=4, latent_channels=None)` – Backbone Setup; Output: Modul.
- `CTPixelNeRFEncoder.forward(ct, ap, pa)` – z_feat; Output: Tensor (B, C).

**Typische Fragen, die diese Datei beantwortet:**
- "Wie wird der Encoder aus CT/AP/PA gebaut?"
- "Woher kommt das latente z im Hybrid-Pfad?"

**Danger Zones / haeufige Bugs:**
- CT-Slice-Layout (axial/coronal/sagittal) muss zu CT-Orientierung passen.
- Conv1 erwartet 5 Kanaele (3 CT Slices + AP + PA).

---

### pixelNeRF+pieNeRF/model/nerf/pienerf_mlp_cond.py
**Ziel (1-2 Saetze):**
- Conditional MLP fuer sigma-Vorhersage, concat mit latentem `z_feat`.

**Worum geht's / was findet man darin? (4-7 Bulletpoints):**
- Concatenation von positional-encoded xyz und z_feat.
- NeRF-typische Skip-Connections (`skips`).
- Output: sigma (1 Kanal).

**Kernelemente (Top 3-8 Stellen):**
- `PieNeRFConditionalMLP.__init__(input_ch_xyz, latent_dim=512, D=8, W=256, skips=None)` – MLP-Aufbau.
- `PieNeRFConditionalMLP.forward(xyz_pe, z_feat)` – sigma; Output: Tensor (N,1) oder (B,N,1).

**Typische Fragen, die diese Datei beantwortet:**
- "Wo wird sigma aus xyz+z berechnet?"

**Danger Zones / haeufige Bugs:**
- Batch-Handling (xyz als [N,C] vs [B,N,C]).

---

### pixelNeRF+pieNeRF/utils/geometry/ray_sampler_spect.py
**Ziel (1-2 Saetze):**
- Uniformes Ray-Sampling entlang AP-Achse fuer Hybrid-Forward.

**Worum geht's / was findet man darin? (4-7 Bulletpoints):**
- Rays fuer jedes (SI, LR) mit `num_samples` entlang AP.
- Rueckgabe `xyz_points` und `ray_dirs` (optional Batch-Dim).
- Koordinaten um Ursprung zentriert, skaliert durch `voxel_size`.

**Kernelemente (Top 3-8 Stellen):**
- `sample_spect_rays(volume_shape, voxel_size, view, num_samples, batch_size=None, device=None, dtype=None)` – Output: `(xyz_points, ray_dirs)`.

**Typische Fragen, die diese Datei beantwortet:**
- "Wo werden Rays im Hybrid-Branch gebaut?"

**Danger Zones / haeufige Bugs:**
- `view` (AP vs PA) aendert Richtung der Rays.
- `voxel_size` skaliert die Weltkoordinaten.

---

### pixelNeRF+pieNeRF/forward/spect_operator_wrapper.py
**Ziel (1-2 Saetze):**
- Forward-Projektion fuer Hybrid-Branch, ausgerichtet an pieNeRF-Orientierung.

**Worum geht's / was findet man darin? (4-7 Bulletpoints):**
- Volumenreihenfolge (SI, AP, LR) wird zu (AP, SI, LR) permutiert.
- Optional Attenuation via `mu_volume` mit kumulativer Summe + Shift.
- AP integriert entlang umgekehrter AP-Achse, PA entlang normaler Achse.
- LR-Flip am Ende zur Ausrichtung der Projektionen.

**Kernelemente (Top 3-8 Stellen):**
- `forward_spect(sigma_volume, mu_volume=None, use_attenuation=False, step_len=1.0)` – Output: `(proj_ap, proj_pa)`.

**Typische Fragen, die diese Datei beantwortet:**
- "Wie sieht der Hybrid-Forward-Operator aus?"
- "Wo wird Attenuation im Hybrid-Pfad angewendet?"

**Danger Zones / haeufige Bugs:**
- LR-Flip am Ende (orientierungsabhaengig).
- AP/PA Integrationsrichtung (reverse AP vs. normal).

---

### pixelNeRF+pieNeRF/pixel-nerf/src/data/SpectDataset.py
**Ziel (1-2 Saetze):**
- Dataset fuer Hybrid-Training mit optionalem Downsampling.

**Worum geht's / was findet man darin? (4-7 Bulletpoints):**
- Manifest-CSV fuer AP/PA/CT/ACT mit Pfad-Resolution.
- CT/ACT Layout-Check -> (SI, AP, LR).
- AP/PA werden normalisiert und optional rotiert.
- Optionales Downsampling via `F.interpolate`.

**Kernelemente (Top 3-8 Stellen):**
- `SpectDataset.__init__(datadir, stage="train", manifest=None, scale_ct=10.0, target_hw=(128,320), target_depth=128, rotate_projections=False)` – Dataset.
- `SpectDataset.__getitem__(idx)` – lade + resample; Output: Dict.
- `_reorder_to_siaplr(vol, name)` – Layout-Heuristik; Output: ndarray.

**Typische Fragen, die diese Datei beantwortet:**
- "Wie werden CT/ACT im Hybrid-Branch geladen und umgeordnet?"

**Danger Zones / haeufige Bugs:**
- Layout-Heuristik in `_reorder_to_siaplr`.
- Resampling kann Intensitaeten veraendern.

---

## Cross-Reference Index (Frage/Feature -> Datei + Funktion)

| Frage/Feature | Datei + Funktion(en) |
| --- | --- |
| Wo wird Attenuation angewendet (Core)? | `pieNeRF/nerf/run_nerf_mod.py` `raw2outputs_emission(...)` |
| Wo wird Attenuation angewendet (Hybrid)? | `pixelNeRF+pieNeRF/forward/spect_operator_wrapper.py` `forward_spect(...)` |
| Wo werden Rays/Posen gebaut (Core)? | `pieNeRF/graf/transforms.py` `RaySampler.__call__(...)`, `pieNeRF/nerf/run_nerf_helpers_mod.py` `get_rays_ortho(...)` |
| Wo werden Rays gebaut (Hybrid)? | `pixelNeRF+pieNeRF/utils/geometry/ray_sampler_spect.py` `sample_spect_rays(...)` |
| Wo passiert CT-Interpolation via grid_sample? | `pieNeRF/nerf/run_nerf_mod.py` `sample_ct_volume(...)` |
| Wo wird der Loss berechnet (Core)? | `pieNeRF/train_emission.py` `poisson_nll(...)` |
| Wo ist der Ray-Split implementiert? | `pieNeRF/utils/ray_split.py` `make_pixel_split_from_ap_pa(...)`, `sample_train_indices(...)` |
| Wo wird z initialisiert und optimiert (Core)? | `pieNeRF/train_emission.py` `train()` (z_train + Optimizer) |
| Wo wird CT-Kontext gebaut? | `pieNeRF/graf/generator.py` `build_ct_context(...)` |
| Wo wird das Dataset fuer Core geladen? | `pieNeRF/graf/datasets.py` `SpectDataset.__getitem__(...)` |
| Wo wird das Dataset fuer Hybrid geladen? | `pixelNeRF+pieNeRF/pixel-nerf/src/data/SpectDataset.py` `SpectDataset.__getitem__(...)` |
| Wo wird preview rendering gespeichert (Core)? | `pieNeRF/train_emission.py` `maybe_render_preview(...)` |
| Wo wird preview rendering gespeichert (Hybrid)? | `pixelNeRF+pieNeRF/train/train_spect_pienerf_pixelnerf.py` `save_visual(...)` |
| Wo wird Positional Encoding definiert? | `pieNeRF/nerf/run_nerf_helpers_mod.py` `Embedder`, `get_embedder(...)` |
| Wo wird die NeRF-MLP definiert (Core)? | `pieNeRF/nerf/run_nerf_helpers_mod.py` `NeRF` |
| Wo wird sigma-MLP definiert (Hybrid)? | `pixelNeRF+pieNeRF/model/nerf/pienerf_mlp_cond.py` `PieNeRFConditionalMLP` |
| Wo werden AP/PA-Posen gesetzt? | `pieNeRF/graf/generator.py` `set_fixed_ap_pa(...)` |
| Wo werden Trainingsrays fuer AP/PA geschnitten? | `pieNeRF/train_emission.py` `slice_rays(...)` |
| Wo werden CT/ACT Transposes gemacht (Core)? | `pieNeRF/graf/datasets.py` `_load_npy_ct(...)`, `_load_npy_act(...)` |
| Wo werden CT/ACT Transposes gemacht (Hybrid)? | `pixelNeRF+pieNeRF/pixel-nerf/src/data/SpectDataset.py` `_reorder_to_siaplr(...)` |

