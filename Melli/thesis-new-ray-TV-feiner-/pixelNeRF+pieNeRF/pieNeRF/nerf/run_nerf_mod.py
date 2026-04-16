import os, sys
import numpy as np
import imageio
import json
import random
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt

from .run_nerf_helpers_mod import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False
_ATTENUATION_WARNED = False
relu = partial(F.relu, inplace=True)


# ---------------------------
# Hilfsfunktionen
# ---------------------------
def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn,
                features=None, netchunk=1024*64, feat_dim_appearance=0):
    """
    Stellt sicher, dass alle Tensors auf dem selben Device liegen wie 'fn' (das NeRF-Netz).
    """
    # 📌 Device vom Netz holen (cuda oder cpu)
    device = next(fn.parameters()).device

    # Eingaben auf dieses Device bringen
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]).to(device)
    embedded = embed_fn(inputs_flat)

    features_shape = None
    features_appearance = None

    if features is not None:
        # Features ebenfalls auf dieses Device bringen
        features = features.to(device)

        # expand features to shape of flattened inputs
        features = features.unsqueeze(1).expand(-1, inputs.shape[1], -1).flatten(0, 1)

        if viewdirs is not None and feat_dim_appearance > 0:
            features_shape = features[:, :-feat_dim_appearance]
            features_appearance = features[:, -feat_dim_appearance:]
        else:
            features_shape = features
            features_appearance = None

        embedded = torch.cat([embedded, features_shape], -1)

    if viewdirs is not None:
        # viewdirs auch auf dasselbe Device bringen
        input_dirs = viewdirs[:, None].expand(inputs.shape).to(device)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
        if features_appearance is not None:
            embedded = torch.cat([embedded, features_appearance.to(device)], dim=-1)
    else:
        if features_appearance is not None:
            embedded = torch.cat([embedded, features_appearance.to(device)], dim=-1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat,
                            list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    all_ret = {}
    features = kwargs.get('features')
    for i in range(0, rays_flat.shape[0], chunk):
        if features is not None:
            kwargs['features'] = features[i:i+chunk]
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def sample_ct_volume(pts, context):
    """
    Trilineare Interpolation des CT-Volumens an frei gewählten Weltpunkten.
    Annahme: CT deckt [-radius, radius]^3 ab und ist achsenausgerichtet.
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

    coords = pts / radius
    coords = torch.clamp(coords, -1.0, 1.0)
    N_rays, N_samples = coords.shape[0], coords.shape[1]
    grid = coords.view(1, 1, N_rays, N_samples, 3)
    mu = F.grid_sample(
        volume,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return mu.view(N_rays, N_samples)


# ---------------------------
# Haupt-Renderfunktion
# ---------------------------
def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
           near=0., far=1., use_viewdirs=False, c2w_staticcam=None, **kwargs):
    network = kwargs.get("network_fn")
    if network is None:
        raise ValueError("render() benötigt ein 'network_fn' in kwargs.")
    # Nutze das Device des NeRF-Netzes als Referenz für alle Tensoren.
    net_device = next(network.parameters()).device

    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
    else:
        viewdirs = None

    sh = rays_d.shape
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    rays_o = torch.reshape(rays_o, [-1, 3]).float().to(net_device)
    rays_d = torch.reshape(rays_d, [-1, 3]).float().to(net_device)
    if viewdirs is not None:
        viewdirs = viewdirs.to(net_device)

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    features = kwargs.get('features')
    if features is not None:
        # Features vor dem Expand auf das gleiche Device wie die Rays legen.
        features = features.to(net_device, non_blocking=True)
        bs = features.shape[0]
        N_rays = sh[0] // bs
        kwargs['features'] = features.unsqueeze(1).expand(-1, N_rays, -1).flatten(0, 1)

    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # robustes Key-Handling
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
    use_attenuation=False,
    attenuation_debug=False,
):
    """
    Emissions-NeRF:
      - raw[...,0] = e(x) (unbounded)
      - Projektion: I = sum_i e_i * Δs_i
    Rückgaben:
      proj_map : [N_rays]   (Graustufenprojektion)
      disp_map : [N_rays]   (einfaches 1/depth)
      acc_map  : [N_rays]   (hier = proj_map als "Akkumulation")
    """
    # Δs entlang des Strahls
    dists = z_vals[..., 1:] - z_vals[..., :-1]                     # [N_rays, N_samples-1]
    dists = torch.cat([dists, dists[..., -1:].clone()], dim=-1)    # [N_rays, N_samples]
    # Ray-Länge (Norm von d) berücksichtigen
    ray_norm = torch.norm(rays_d[..., None, :], dim=-1)
    dists = dists * ray_norm

    # Emission >= 0 (keine zusätzliche Sättigung, damit der Dynamikbereich erhalten bleibt)
    e = F.softplus(raw[..., 0])

    if raw_noise_std > 0.0:
        noise = torch.randn_like(e) * raw_noise_std
        if pytest:
            np.random.seed(0)
            noise_np = np.random.rand(*list(e.shape)) * raw_noise_std
            noise = torch.tensor(noise_np, dtype=e.dtype, device=e.device)
        e = torch.clamp(e + noise, min=0.0)

    transmission = None
    weights = e * dists
    if use_attenuation:
        if mu_vals is None:
            warnings.warn("use_attenuation=True aber ohne ct_context – falle zurück auf reine Emission.")
        else:
            mu = torch.clamp(mu_vals, min=0.0)
            if mu.shape != e.shape:
                raise ValueError(
                    f"CT samples have wrong shape {mu.shape}, expected {e.shape}."
                )
            mu_dists = mu * dists
            attenuation = torch.cumsum(mu_dists, dim=-1)
            attenuation = F.pad(attenuation[..., :-1], (1, 0), mode="constant", value=0.0)
            attenuation = torch.clamp(attenuation, min=0.0, max=60.0)
            transmission = torch.exp(-attenuation)
            weights = e * transmission * dists

    # Line-Integral (Riemann)
    proj_map = torch.sum(weights, dim=-1)         # [N_rays]

    # Depth (gewichtetes Mittel der z-Positionen)
    depth_map = torch.sum(z_vals * weights, dim=-1) / (proj_map + 1e-8)
    disp_map  = 1.0 / torch.clamp(depth_map, min=1e-8)

    # "acc" – hier einfach proj_map als Summenmaß
    acc_map   = proj_map.clone()

    debug_payload = None
    if attenuation_debug:
        debug_payload = {
            "debug_lambda": e.detach(),
            "debug_dists": dists.detach(),
            "debug_weights": weights.detach(),
        }
        if mu_vals is not None:
            debug_payload["debug_mu"] = torch.clamp(mu_vals, min=0.0).detach()
        if transmission is not None:
            debug_payload["debug_transmission"] = transmission.detach()

    return proj_map, disp_map, acc_map, debug_payload

# ---------------------------
# Ray Rendering
# ---------------------------
# Dummy für den Fall, dass irgendwo noch raw2outputs erwartet wird
def raw2outputs(*args, **kwargs):
    raise RuntimeError("raw2outputs() (RGB) wurde aufgerufen, aber Emission ist aktiv – "
                       "bitte emission=True in create_nerf setzen.")
def render_rays(ray_batch, network_fn, network_query_fn, N_samples,
                features=None, retraw=False, lindisp=False, perturb=0.,
                N_importance=0, network_fine=None, white_bkgd=False,
                raw_noise_std=0., verbose=False, pytest=False, emission=False,
                **kwargs):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    # Erzwinge, dass die Sampling-Parameter auf demselben Device liegen wie near/far.
    t_vals = torch.linspace(0., 1., steps=N_samples, device=near.device)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * t_vals
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)
        if pytest:
            np.random.seed(0)
            t_rand_np = np.random.rand(*list(z_vals.shape))
            t_rand = torch.tensor(t_rand_np, dtype=z_vals.dtype, device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    raw = network_query_fn(pts, viewdirs, network_fn, features)
    ct_context = kwargs.get("ct_context")
    use_attenuation = bool(kwargs.get("use_attenuation", False))
    attenuation_debug = bool(kwargs.get("attenuation_debug", False))
    mu_vals = None
    if use_attenuation and ct_context is not None:
        mu_vals = sample_ct_volume(pts, ct_context)
        if not torch.isfinite(mu_vals).all():
            raise ValueError("CT samples contain NaN/Inf values.")
        if ((mu_vals < -1e-3) | (mu_vals > 1.1)).any():
            if not ct_context.get("_range_warned", False):
                warnings.warn("CT attenuation samples outside expected [0,1] range.")
                ct_context["_range_warned"] = True
    elif use_attenuation and ct_context is None:
        global _ATTENUATION_WARNED
        if not _ATTENUATION_WARNED:
            warnings.warn("use_attenuation=True aber render() erhielt kein ct_context – deaktiviere Attenuation.")
            _ATTENUATION_WARNED = True

    # 🔥 Emission aktivieren
    if emission:
        proj_map, disp_map, acc_map, debug_payload = raw2outputs_emission(
            raw,
            z_vals,
            rays_d,
            raw_noise_std=raw_noise_std,
            pytest=pytest,
            mu_vals=mu_vals,
            use_attenuation=use_attenuation,
            attenuation_debug=attenuation_debug,
        )
        ret = {'proj_map': proj_map, 'disp_map': disp_map, 'acc_map': acc_map}
        if debug_payload:
            ret.update(debug_payload)
        if attenuation_debug and mu_vals is not None and "debug_mu" not in ret:
            ret["debug_mu"] = mu_vals.detach()
        if retraw:
            ret['raw'] = raw
        return ret

    # sonst Standard-NeRF
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
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    input_ch += args.feat_dim - args.feat_dim_appearance
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    input_ch_views += args.feat_dim_appearance

    # 🔥 Emission-Flag: aktiviert Emissions-Variante
    emission = getattr(args, "emission", False)
    output_ch = 1 if emission else 4

    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    grad_vars = list(model.parameters())
    named_params = list(model.named_parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        grad_vars += list(model_fine.parameters())
        named_params += list(model_fine.named_parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn, features: run_network(
        inputs, viewdirs, network_fn, features=features, embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn, netchunk=args.netchunk,
        feat_dim_appearance=args.feat_dim_appearance)

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
        ndc=False,
        lindisp=False,
        emission=emission,  # <<< wichtig!
    )

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0
    return render_kwargs_train, render_kwargs_test, grad_vars, named_params
