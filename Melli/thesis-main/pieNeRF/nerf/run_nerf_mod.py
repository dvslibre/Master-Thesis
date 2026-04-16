"""Modified NeRF implementation for emission rendering: embeds coordinates, supports latent features, optional attenuation, and renders SPECT rays."""

import numpy as np
import warnings
import random
import torch
import torch.nn.functional as F

from .run_nerf_helpers_mod import *

np.random.seed(0)                   # fixiere Zufallszahlen (Reproduzierbarkeit)
_ATTENUATION_WARNED = False         # Flag um eine Warnung nur einmal auszugeben


# ---------------------------
# Hilfsfunktionen
# ---------------------------
def batchify(fn, chunk):
    """Wrappt eine Funktion mit 'fn' so, dass sie große Eingaben in kleineren Blöcken
    (Chunks) verarbeitet, um GPU Out-of-Memory zu vermeiden"""           
    if chunk is None:               # wenn kein Chunk-Limit: direkt durchlaufen lassen
        return fn
    def ret(inputs):
        # Verarbeite inputs in Blöcken [i : i+chunk], führe fn auf jeden Block aus
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn,
                features=None, netchunk=1024*64, feat_dim_appearance=0):
    """
    Führt das NeRF-MLP 'fn' auf gegebenen 3D-Punkten aus:
    - positional encoding (embed_fn)
    - optional viewdirs
    - optional latente Features (z)
    - Chunks zu OOM-Vermeidung
    """
    # Device ermitteln (CPU/GPU des NeRF-Modells)
    device = next(fn.parameters()).device

    # 3D-Punkte in 2D-Form flatten und auf Device legen
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]).to(device)
    embedded = embed_fn(inputs_flat)     # positional Encoding anwenden

    # ------------------------------
    # Latenten code z anhängen
    # ------------------------------
    features_shape = None
    features_appearance = None

    if features is not None:
        features = features.to(device)

        # z-Code auf alle Samples eines Rays ausweiten
        features = features.unsqueeze(1).expand(-1, inputs.shape[1], -1).flatten(0, 1)      # ergibt [B*N_samples, feat_dim]

        # Aufteilen in shape-related Teil und appearance-Teil
        if viewdirs is not None and feat_dim_appearance > 0:
            features_shape = features[:, :-feat_dim_appearance]
            features_appearance = features[:, -feat_dim_appearance:]
        else:
            features_shape = features
            features_appearance = None

        # shape-beeinflussende Features an positional encoding anhängen
        embedded = torch.cat([embedded, features_shape], -1)


    # ------------------------------
    # Viewdirs + appeareance codes
    # ------------------------------
    if viewdirs is not None:
        # Jede Viewdir für jeden Samplepunkt kodieren
        input_dirs = viewdirs[:, None].expand(inputs.shape).to(device)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
        if features_appearance is not None:
            embedded = torch.cat([embedded, features_appearance.to(device)], dim=-1)
    else:
        if features_appearance is not None:
            embedded = torch.cat([embedded, features_appearance.to(device)], dim=-1)

    # ------------------------------
    # Forward durch das NeRF mit Chunking
    # ------------------------------
    outputs_flat = batchify(fn, netchunk)(embedded)
    # zurückformen in [N_rays, N_samples, output_dim]
    outputs = torch.reshape(outputs_flat,
                            list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Wendet 'render_rays' blockweise auf große Ray-Mengen an. Nutzt Chunking, damit GPU verträglich"""
    all_ret = {}
    features = kwargs.get('features')
    scalar_accum = {k: [] for k in ("tv_loss", "tv_base_loss", "tv_mu_loss", "mu_gate_loss", "mask_loss")}
    # blockweise über die Rays iterieren
    for i in range(0, rays_flat.shape[0], chunk):
        # z-Features passend mitschneiden
        if features is not None:
            kwargs['features'] = features[i:i+chunk]
        # rendern eines Blockes
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for key in scalar_accum:
            val = ret.pop(key, None)
            if val is not None:
                scalar_accum[key].append(val)
        # Ergebnisse sammeln
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    # alle Blöcke wieder zu vollständigen Outputs zusammenfügen
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    for key, vals in scalar_accum.items():
        if vals:
            all_ret[key] = torch.stack(vals).mean()
    return all_ret


def sample_ct_volume(pts, context):
    """
    Interpoliert das CT-Volumen trilinear an beliebigen Weltpunkten 'pts'.
    Erwartet: 
        pts:        [N_rays, N_samples, 3]
        context:    dict mit {'volume': [1,1,D,H,W], 'grid_radius': r}
    """
    if context is None:
        return None
    volume = context.get("volume")
    if volume is None:
        return None
    if volume.dim() != 5:
        raise ValueError(f"ct_volume must be [1,1,D,H,W], got {tuple(volume.shape)}")
    radius = float(context.get("grid_radius", 1.0))
    if radius <= 0:
        raise ValueError("grid_radius must be > 0 for CT sampling.")

    if volume.device != pts.device:
        raise ValueError("ct_volume must live on the same device as ray samples for interpolation.")

    coords = pts / radius                               # Weltkoordinaten normalisieren in [-1,1]
    coords = torch.clamp(coords, -1.0, 1.0)
    N_rays, N_samples = coords.shape[0], coords.shape[1]
    grid = coords.view(1, 1, N_rays, N_samples, 3)      # grid_sample erwartet 5D: [N, C, D, H, W]
    mu = F.grid_sample(                                 # Trilineare Interpolation im CT-Würfel    
        volume,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return mu.view(N_rays, N_samples)                   # Ergebnis in [N_rays, N_samples]    


# ---------------------------
# Haupt-Renderfunktion
# ---------------------------
def render(H, W, focal, chunk=1024*32, rays=None, c2w=None,
           near=0., far=1., use_viewdirs=False, c2w_staticcam=None, **kwargs):
    """
    High-Level-Rendering:
    - Rays erzeugen / übernehmen
    - near/far + viewdirs anhängen
    - z-Features expandieren
    - batchify_rays -> render_rays
    - outputs zurück als Projektion
    """
    network = kwargs.get("network_fn")
    if network is None:
        raise ValueError("render() benötigt ein 'network_fn' in kwargs.")
    # Nutze das Device des NeRF-Netzes als Referenz für alle Tensoren.
    net_device = next(network.parameters()).device

    # ------------------------------
    # Rays erzeugen (falls c2w gegeben)
    # ------------------------------
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        rays_o, rays_d = rays                       # bereits vorbereitete Rays

    # ------------------------------
    # Optionale Viewdirs
    # ------------------------------
    if use_viewdirs:
        viewdirs = rays_d
        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
    else:
        viewdirs = None

    sh = rays_d.shape                               # (H, W, 3)
    
    # Rays flatten und auf device
    rays_o = torch.reshape(rays_o, [-1, 3]).float().to(net_device)      
    rays_d = torch.reshape(rays_d, [-1, 3]).float().to(net_device)
    if viewdirs is not None:
        viewdirs = viewdirs.to(net_device)

    # near/far an jeden Ray anhängen
    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # ---------------------------
    # z-Features pro Ray expandieren
    # ---------------------------
    features = kwargs.get('features')
    if features is not None:
        # Features vor dem Expand auf das gleiche Device wie die Rays legen.
        features = features.to(net_device, non_blocking=True)
        bs = features.shape[0]
        N_rays = sh[0] // bs
        kwargs['features'] = features.unsqueeze(1).expand(-1, N_rays, -1).flatten(0, 1)

    # ---------------------------
    # Renderin in Chunks aufteilen
    # ---------------------------
    all_ret = batchify_rays(rays, chunk, **kwargs)
    scalar_keys = ("tv_loss", "tv_base_loss", "tv_mu_loss", "mu_gate_loss", "mask_loss")
    scalar_ret = {k: all_ret.pop(k, None) for k in scalar_keys}
    # zurück in Bildform [H, W, ...]
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)
    for k, v in scalar_ret.items():
        if v is not None:
            all_ret[k] = v

    # ---------------------------
    # Projektion extrahieren
    # ---------------------------
    if 'rgb_map' in all_ret:
        k_extract = ['rgb_map', 'disp_map', 'acc_map']
    elif 'proj_map' in all_ret:
        k_extract = ['proj_map', 'disp_map', 'acc_map']
    else:
        raise KeyError(f"No 'rgb_map' or 'proj_map' in render outputs: {list(all_ret.keys())}")
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


# ---------------------------
# Emission-only Variante
# ---------------------------

# ---- Emission-only raw2outputs ----
def raw2outputs_emission(
    raw,
    z_vals,
    rays_d,
    raw_noise_std=0.0,
    pytest=False,
    mu_vals=None,
    mask_vals=None,
    use_attenuation=False,
    attenuation_debug=False,
    debug_prints: bool = False,
    tv_mu_sigma: float = 1.0,
    mu_gate_mode: str = "none",
    mu_gate_center: float = 0.2,
    mu_gate_width: float = 0.1,
):
    """
    Emissions-NeRF:
      - raw[...,0] = e(x): Emissionsstärke im Samplepunkt (ungebundener Ausgang des MLP)
      - Projektion: I = sum_i e_i * Δs_i   (bzw. mit Attenuation: e_i * T_i * Δs_i)

    Parameter:
      raw        : [N_rays, N_samples, C]    MLP-Outputs (mind. Kanal 0 = Emission)
      z_vals     : [N_rays, N_samples]       Tiefenpositionen entlang des Strahls
      rays_d     : [N_rays, 3]               Richtungen der Strahlen
      raw_noise_std     : Rauschstd. (Training-Stabilisierung)
      mu_vals    : [N_rays, N_samples]       CT-basiertes µ (optional, für Attenuation)
      use_attenuation  : bool                Ob Attenuation einbezogen wird
      attenuation_debug: bool                Extra Debug-Infos für Attenuation

    Rückgaben:
      proj_map : [N_rays]   Intensität der Projektion (Line-Integral)
      disp_map : [N_rays]   einfache "Disparity" = 1 / gewichteter Tiefe
      acc_map  : [N_rays]   hier identisch zu proj_map (Summe als "Akkumulation")
      debug_payload : dict oder None, optional Debug-Tensoren
    """
    # Δs entlang des Strahls (Abstand zwischen aufeinanderfolgenden z-Samples)
    dists = z_vals[..., 1:] - z_vals[..., :-1]                  # [N_rays, N_samples-1]
    dists = torch.cat([dists, dists[..., -1:].clone()], dim=-1) # letzte Distanz duplizieren → [N_rays, N_samples]

    # Ray-Länge (Norm von d) berücksichtigen, damit Δs in Weltkoordinaten skaliert ist
    ray_norm = torch.norm(rays_d[..., None, :], dim=-1)         # [N_rays, 1]
    dists = dists * ray_norm                                    # [N_rays, N_samples]

    # Emission >= 0 erzwingen (Softplus statt ReLU, um small-gradients bei kleinen Werten zu vermeiden)
    lambda_vals = F.softplus(raw[..., 0])                       # [N_rays, N_samples]

    # Optionales Trainingsrauschen auf Emissionen (wie in Original-NeRF bei sigma)
    if raw_noise_std > 0.0:
        noise = torch.randn_like(lambda_vals) * raw_noise_std
        if pytest:
            # deterministisches Rauschen bei Tests (NumPy-Seed)
            np.random.seed(0)
            noise_np = np.random.rand(*list(lambda_vals.shape)) * raw_noise_std
            noise = torch.tensor(noise_np, dtype=lambda_vals.dtype, device=lambda_vals.device)
        # Rauschen addieren und unten bei 0 abschneiden
        lambda_vals = torch.clamp(lambda_vals + noise, min=0.0)

    # 1D-TV entlang der Samples (Charbonnier-Variante für stabile Gradienten)
    diff_lambda = lambda_vals[..., 1:] - lambda_vals[..., :-1]
    eps = 1e-6
    tv_ray = torch.sqrt(diff_lambda * diff_lambda + eps)
    tv_base = tv_ray.mean()

    # Edge-aware TV, gewichtet mit μ-Differenzen (kleine Δμ => starke TV, große Δμ => schwächer)
    if mu_vals is not None:
        diff_mu = mu_vals[..., 1:] - mu_vals[..., :-1]
    else:
        diff_mu = torch.zeros_like(diff_lambda)
    sigma = max(float(tv_mu_sigma), 1e-8)
    w_mu = torch.exp(-torch.abs(diff_mu) / (sigma + 1e-8))
    tv_mu = (w_mu * tv_ray).mean()

    # μ-basierter Prior auf Emissionen (weiches Gate)
    mu_gate_loss = lambda_vals.new_tensor(0.0)
    gate_mode = (mu_gate_mode or "none").lower()
    if gate_mode != "none" and mu_vals is not None:
        center = float(mu_gate_center)
        width = float(mu_gate_width)
        if gate_mode == "bandpass":
            dist = torch.clamp(torch.abs(mu_vals - center) - width, min=0.0)
        elif gate_mode == "lowpass":
            dist = torch.clamp(mu_vals - center, min=0.0)
        elif gate_mode == "highpass":
            dist = torch.clamp(center - mu_vals, min=0.0)
        else:
            dist = None
        if dist is not None:
            penalty = dist * dist * lambda_vals
            mu_gate_loss = penalty.mean()

    # Organmasken-Loss: penalizes emission außerhalb der Maske
    mask_loss = lambda_vals.new_tensor(0.0)
    if mask_vals is not None:
        mask_vals = torch.clamp(mask_vals, 0.0, 1.0)
        violation = (1.0 - mask_vals) * lambda_vals
        mask_loss = violation.mean()

    transmission = None
    # Grundfall: keine Attenuation → einfache Gewichte = e * Δs
    weights = lambda_vals * dists

    if use_attenuation:
        if mu_vals is None:
            # Flag gesetzt, aber kein µ-Volumen übergeben → Warnung
            warnings.warn("use_attenuation=True aber ohne ct_context – falle zurück auf reine Emission.")
        else:
            # µ nicht negativ werden lassen (physikalisch: keine Verstärkung)
            mu = torch.clamp(mu_vals, min=0.0)
            if mu.shape != lambda_vals.shape:
                raise ValueError(f"CT samples have wrong shape {mu.shape}, expected {lambda_vals.shape}.")

            # µ * Δs → lineare Dämpfung pro Segment
            mu_dists = mu * dists                                # [N_rays, N_samples]

            # Kumulative Attenuation ∫ µ ds (diskret: kumulierte Summe)
            attenuation = torch.cumsum(mu_dists, dim=-1)         # [N_rays, N_samples]
            # Für Segment i soll die Attenuation nur bis Sample i-1 gehen → ein Sample nach links verschieben
            attenuation = F.pad(attenuation[..., :-1], (1, 0), mode="constant", value=0.0)

            # Clipping der Exponenten, um exp(-attenuation) numerisch stabil zu halten
            attenuation = torch.clamp(attenuation, min=0.0, max=60.0)

            # Transmission T = exp(-∫ µ ds)
            transmission = torch.exp(-attenuation)               # [N_rays, N_samples]

            # Gewichte jetzt: e * T * Δs
            weights = lambda_vals * transmission * dists

    # Line-Integral entlang des Strahls
    proj_map = torch.sum(weights, dim=-1)                        # [N_rays]

    if debug_prints and random.randint(0, 50) == 0:
        lam_min, lam_max = lambda_vals.min().item(), lambda_vals.max().item()
        w_min, w_max = weights.min().item(), weights.max().item()
        mu_min = mu_max = float("nan")
        trans_min = trans_max = float("nan")
        if mu_vals is not None:
            mu_min, mu_max = mu_vals.min().item(), mu_vals.max().item()
        if use_attenuation and transmission is not None:
            trans_min, trans_max = transmission.min().item(), transmission.max().item()
        print(
            f"[DEBUG][raw2outputs_emission] λ min/max: {lam_min:.3e}/{lam_max:.3e} | "
            f"μ min/max: {mu_min:.3e}/{mu_max:.3e} | "
            f"T min/max: {trans_min:.3e}/{trans_max:.3e} | "
            f"weights min/max: {w_min:.3e}/{w_max:.3e} | "
            f"proj min/max: {proj_map.min().item():.3e}/{proj_map.max().item():.3e}",
            flush=True,
        )

    # "Tiefe" = gewichtetes Mittel der z-Positionen
    depth_map = torch.sum(z_vals * weights, dim=-1) / (proj_map + 1e-8)
    disp_map  = 1.0 / torch.clamp(depth_map, min=1e-8)           # einfache Disparity (optional)

    # "acc" – hier als Summenmaß einfach die projizierte Intensität
    acc_map   = proj_map.clone()

    # Optionales Debug-Paket für Analyse
    debug_payload = None
    if attenuation_debug:
        debug_payload = {
            "debug_lambda": lambda_vals.detach(),
            "debug_dists": dists.detach(),
            "debug_weights": weights.detach(),
        }
        if mu_vals is not None:
            debug_payload["debug_mu"] = torch.clamp(mu_vals, min=0.0).detach()
        if transmission is not None:
            debug_payload["debug_transmission"] = transmission.detach()

    return proj_map, disp_map, acc_map, debug_payload, tv_base, tv_mu, mu_gate_loss, mask_loss


# ---------------------------
# Ray Rendering
# ---------------------------

# Dummy für den Fall, dass irgendwo noch raw2outputs (RGB-Version) aufgerufen wird
def raw2outputs(*args, **kwargs):
    raise RuntimeError(
        "raw2outputs() (RGB) wurde aufgerufen, aber Emission ist aktiv – "
        "bitte emission=True in create_nerf setzen."
    )


def render_rays(ray_batch, network_fn, network_query_fn, N_samples,
                features=None, retraw=False, lindisp=False, perturb=0.,
                N_importance=0, network_fine=None, white_bkgd=False,
                raw_noise_std=0., verbose=False, pytest=False, emission=False,
                **kwargs):
    """
    Rendert eine Menge von Rays:
      - nimmt Ray-Parameter (rays_o, rays_d, near, far, optional viewdirs)
      - sampelt N_samples Tiefen z_vals entlang jedes Rays
      - ruft network_query_fn (→ NeRF) auf
      - ruft raw2outputs_emission (oder raw2outputs) auf, um Projektionen zu erhalten
    """
    N_rays = ray_batch.shape[0]

    # Zerlegen des Ray-Batches:
    # ray_batch: [N_rays, 8] oder [N_rays, 11] abhängig von viewdirs
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])  # [N_rays, 1, 2] mit (near,far)
    near, far = bounds[..., 0], bounds[..., 1]               # [N_rays, 1]

    # Tiefen-Sampling t in [0,1] → wird auf [near,far] gemappt
    t_vals = torch.linspace(0., 1., steps=N_samples, device=near.device)
    if not lindisp:
        # linear in der Tiefe
        z_vals = near * (1. - t_vals) + far * t_vals
    else:
        # linear in Disparität (1/z)
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    z_vals = z_vals.expand([N_rays, N_samples])               # [N_rays, N_samples]

    # Optionales „jittering“ der Samplepositionen (Stratified Sampling) zur Regularisierung
    if perturb > 0.:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])     # Mittelpunkte zwischen z_i
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)
        if pytest:
            # deterministisches Jittering im Testmodus
            np.random.seed(0)
            t_rand_np = np.random.rand(*list(z_vals.shape))
            t_rand = torch.tensor(t_rand_np, dtype=z_vals.dtype, device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand             # zufälliger Punkt im Intervall [lower,upper]

    # 3D-Samplepunkte entlang der Rays: p(s) = o + d * s
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]   # [N_rays, N_samples, 3]

    # Netzabfrage: MLP-Forward auf allen Samplepunkten
    raw = network_query_fn(pts, viewdirs, network_fn, features)

    # Attenuation-Setup aus kwargs holen
    ct_context = kwargs.get("ct_context")
    use_attenuation = bool(kwargs.get("use_attenuation", False))
    attenuation_debug = bool(kwargs.get("attenuation_debug", False))
    debug_prints = bool(kwargs.get("debug_prints", False))
    tv_mu_sigma = float(kwargs.get("tv_mu_sigma", 1.0))
    mu_gate_mode = kwargs.get("mu_gate_mode", "none")
    mu_gate_center = float(kwargs.get("mu_gate_center", 0.2))
    mu_gate_width = float(kwargs.get("mu_gate_width", 0.1))
    use_organ_mask = bool(kwargs.get("use_organ_mask", False))
    mu_vals = None
    mask_vals = None

    # Falls Attenuation aktiviert und CT-Context vorhanden: µ aus CT sampeln
    if use_attenuation and ct_context is not None:
        mu_vals = sample_ct_volume(pts, ct_context)           # [N_rays, N_samples]
        if not torch.isfinite(mu_vals).all():
            raise ValueError("CT samples contain NaN/Inf values.")
        # Bereichscheck gegen den tatsächlichen μ-Wertebereich (nur Clamping-Warnung, keine Normierung).
        value_range = ct_context.get("value_range")
        if value_range is not None:
            vmin, vmax = value_range
            tol_low = float(vmin) - 1e-2 * max(1.0, abs(float(vmin)))
            tol_high = float(vmax) + 1e-2 * max(1.0, abs(float(vmax)))
            out_of_range = (mu_vals < tol_low) | (mu_vals > tol_high)
            if out_of_range.any() and not ct_context.get("_range_warned", False):
                warnings.warn(f"CT attenuation samples outside expected range [{vmin:.3g}, {vmax:.3g}] (after scaling).")
                ct_context["_range_warned"] = True
        elif ((mu_vals < -1e-3) | (mu_vals > 1.1)).any():
            if not ct_context.get("_range_warned", False):
                warnings.warn("CT attenuation samples outside expected range (no reference range available).")
                ct_context["_range_warned"] = True
    elif use_attenuation and ct_context is None:
        # use_attenuation=True, aber kein CT → einmalige Warnung, dann deaktiviere Attenuation intern
        global _ATTENUATION_WARNED
        if not _ATTENUATION_WARNED:
            warnings.warn("use_attenuation=True aber render() erhielt kein ct_context – deaktiviere Attenuation.")
            _ATTENUATION_WARNED = True

    if use_organ_mask and ct_context is not None:
        mask_volume = ct_context.get("mask_volume")
        if mask_volume is not None:
            mask_context = {
                "volume": mask_volume,
                "grid_radius": ct_context.get("grid_radius", 1.0),
            }
            mask_vals = sample_ct_volume(pts, mask_context)
            if not torch.isfinite(mask_vals).all():
                raise ValueError("Mask samples contain NaN/Inf values.")

    # Emissions-Pfad (SPECT)
    if emission:
        proj_map, disp_map, acc_map, debug_payload, tv_base_loss, tv_mu_loss, mu_gate_loss, mask_loss = raw2outputs_emission(
            raw,
            z_vals,
            rays_d,
            raw_noise_std=raw_noise_std,
            pytest=pytest,
            mu_vals=mu_vals,
            mask_vals=mask_vals,
            use_attenuation=use_attenuation,
            attenuation_debug=attenuation_debug,
            debug_prints=debug_prints,
            tv_mu_sigma=tv_mu_sigma,
            mu_gate_mode=mu_gate_mode,
            mu_gate_center=mu_gate_center,
            mu_gate_width=mu_gate_width,
        )
        # Standard-Outputs
        ret = {
            'proj_map': proj_map,
            'disp_map': disp_map,
            'acc_map': acc_map,
            'tv_loss': tv_base_loss,
            'tv_base_loss': tv_base_loss,
            'tv_mu_loss': tv_mu_loss,
            'mu_gate_loss': mu_gate_loss,
            'mask_loss': mask_loss,
        }
        # Debug-Infos aus raw2outputs_emission übernehmen
        if debug_payload:
            ret.update(debug_payload)
        # Falls Attenuation-Debug an, aber debug_mu noch nicht drin, hänge µ an
        if attenuation_debug and mu_vals is not None and "debug_mu" not in ret:
            ret["debug_mu"] = mu_vals.detach()
        # Optional: raw-Outputs für spätere Auswertungen zurückgeben
        if retraw:
            ret['raw'] = raw
        return ret

    # Fallback: klassischer RGB-NeRF-Pfad (sollte in meinem Setup NIE aufgerufen werden)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    return ret


# ---------------------------
# NeRF-Erstellung
# ---------------------------

def create_nerf(args):
    """
    Baut:
      - Positional Encoder (Embedder) für Punkte und ggf. Viewdirs
      - NeRF-MLP (coarse) und optional NeRF-MLP (fine)
      - render_kwargs_{train,test} mit allen nötigen Parametern

    Emissions-Spezialfall:
      - output_ch = 1 (nur Emissionskanal)
      - emission-Flag wird in render_rays / raw2outputs_emission ausgewertet
    """
    # Positional Encoding für 3D-Punkte
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    # Latenter Code fließt in die Input-Dim mit ein:
    # feat_dim - feat_dim_appearance = rein geometriebezogene Features
    input_ch += args.feat_dim - args.feat_dim_appearance

    # Viewdir-Encoder (optional)
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    # Appearance-Features an Viewdir-Zweig anhängen
    input_ch_views += args.feat_dim_appearance

    # Emission-Flag: steuert, ob emission-only Pfad verwendet wird
    emission = getattr(args, "emission", False)
    # Bei Emission: nur 1 Kanal Output (e(x)); sonst 4 (RGB+Sigma)
    output_ch = 1 if emission else 4

    skips = [4]  # Skip-Connection nach Layer 4 wie im Original-NeRF

    # Coarse-Netz (Hauptmodell)
    model = NeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs,
    )
    grad_vars = list(model.parameters())
    named_params = list(model.named_parameters())

    # Optional: Fine-Netz (Hierarchical Sampling, wird bei dir evtl. nicht genutzt)
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(
            D=args.netdepth_fine,
            W=args.netwidth_fine,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
        )
        grad_vars += list(model_fine.parameters())
        named_params += list(model_fine.named_parameters())

    # network_query_fn kapselt run_network mit embed_fn und embeddirs_fn
    network_query_fn = lambda inputs, viewdirs, network_fn, features: run_network(
        inputs,
        viewdirs,
        network_fn,
        features=features,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk,
        feat_dim_appearance=args.feat_dim_appearance,
    )

    # Render-Argumente für Training
    render_kwargs_train = dict(
        network_query_fn=network_query_fn,
        perturb=args.perturb,
        N_importance=args.N_importance,
        network_fine=model_fine,
        N_samples=args.N_samples,
        network_fn=model,
        use_viewdirs=args.use_viewdirs,
        white_bkgd=args.white_bkgd,
        raw_noise_std=args.raw_noise_std,
        lindisp=False,
        emission=emission,  # <<< wichtig: schaltet Emission-Pfad in render_rays ein
    )

    # Für Test/Validation: kein Jitter, kein Noise
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, grad_vars, named_params
