"""Mini-training script for the SPECT emission NeRF."""
import argparse
import csv
import math
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from torch.utils.data import DataLoader

from graf.config import get_data, build_models
from utils.ray_split import (
    PixelSplit,
    make_pixel_split_from_ap_pa,
    make_pixel_split_stratified_intensity,
    sample_train_indices,
)

__VERSION__ = "emission-train v0.3"
DEBUG_PRINTS = False  # Nur Debug-Ausgaben, keine Änderung am Verhalten
ATTEN_SCALE_DEFAULT = 25.0


def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"yes", "true", "t", "1"}:
        return True
    if val in {"no", "false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the emission NeRF on SPECT projections.")
    parser.add_argument("--config", type=str, default="configs/spect.yaml", help="Path to the YAML config.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Number of optimisation steps (mini-batches of rays).",
    )
    parser.add_argument(
        "--rays-per-step",
        type=int,
        default=None,
        help="Number of rays per projection per optimisation step. Defaults to training.chunk in the config.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="How often to print running loss statistics.",
    )
    parser.add_argument(
        "--preview-every",
        type=int,
        default=50,
        help="If >0 renders and stores full AP/PA previews every N steps (slow, full-frame render).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="If >0 stores checkpoints every N steps in addition to the final checkpoint.",
    )
    parser.add_argument(
        "--normalize-targets",
        action="store_true",
        help="(Deprecated) Apply per-projection min/max normalisation to both targets and predictions.",
    )
    parser.add_argument(
        "--bg-weight",
        type=float,
        default=1.0,
        help="Down-weights Hintergrundstrahlen im Loss (<1 reduziert Null-Strahlen, 1 = deaktiviert).",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Deaktiviert die periodische Test-Split-Evaluation (schnellere Läufe).",
    )
    parser.add_argument(
        "--debug-prints",
        action="store_true",
        help="Aktiviert verbosere Debug-Ausgaben (keine Verhaltensänderung).",
    )
    parser.add_argument(
        "--weight-threshold",
        type=float,
        default=1e-5,
        help="Zählrate, unter der ein Strahl als Hintergrund gilt (nur relevant mit bg-weight < 1).",
    )
    parser.add_argument(
        "--bg-depth-mass-weight",
        type=float,
        default=0.0,
        help="Gewicht fuer BG depth mass loss (0 = deaktiviert).",
    )
    parser.add_argument(
        "--bg-depth-eps",
        type=float,
        default=1e-10,
        help="Schwellwert fuer Background-Kriterium (Target < eps).",
    )
    parser.add_argument(
        "--bg-depth-eps-norm",
        type=float,
        default=None,
        help="Background-EPS im normierten Raum (überschreibt bg-depth-eps bei normierten Inputs).",
    )
    parser.add_argument(
        "--bg-depth-mode",
        type=str,
        default="integral",
        choices=["integral", "mean"],
        help="BG depth mass mode: integral (sum lambda*dz) oder mean (mean lambda).",
    )
    parser.add_argument(
        "--act-loss-weight",
        type=float,
        default=0.0,
        help="Gewicht für einen optionalen Volumen-Loss gegen act.npy (0 = deaktiviert).",
    )
    parser.add_argument(
        "--act-samples",
        type=int,
        default=None,
        help="Anzahl zufälliger Voxels zur act-Supervision pro Schritt.",
    )
    parser.add_argument(
        "--act-pos-weight",
        type=float,
        default=None,
        help="Zusatzgewicht für den ACT-Loss in aktiven Voxeln (>0).",
    )
    parser.add_argument(
        "--z-reg-weight",
        type=float,
        default=0.0,
        help="L2-Regularisierung auf dem latenten Code z.",
    )
    parser.add_argument(
        "--ct-loss-weight",
        type=float,
        default=0.0,
        help="Gewicht für den CT-Glättungs-Loss entlang z-Konstanten.",
    )
    parser.add_argument(
        "--ct-threshold",
        type=float,
        default=0.05,
        help="Gradienten-Schwelle in ct.npy, unterhalb derer ein Segment als konstant gilt.",
    )
    parser.add_argument(
        "--ct-threshold-norm",
        type=float,
        default=None,
        help="CT-Schwelle im normierten Raum (überschreibt ct-threshold bei normierten Inputs).",
    )
    parser.add_argument(
        "--ct-samples",
        type=int,
        default=8192,
        help="Anzahl CT-Segmentpaare pro Schritt für den Glättungs-Loss.",
    )
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=0.001,
        help="Gewicht für den 1D-TV-Loss entlang der Rays (0 = deaktiviert).",
    )
    parser.add_argument(
        "--ray-tv-weight",
        type=float,
        default=0.0,
        help="Gewicht für den Ray-1D-TV-Prior entlang der Samples (0 = deaktiviert).",
    )
    parser.add_argument(
        "--ray-tv-edge-aware",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help="Aktiviere edge-aware Ray-TV (benoetigt ray_tv_weight > 0 und ray_tv_alpha > 0).",
    )
    parser.add_argument(
        "--ray-tv-alpha",
        type=float,
        default=0.0,
        help="Alpha fuer edge-aware Ray-TV (Gewicht exp(-alpha*|delta_mu|)).",
    )
    parser.add_argument(
        "--ray-tv-w-clamp-min",
        type=float,
        default=0.0,
        help="Optionales Minimum fuer edge-aware TV-Gewichte (z.B. 0.2).",
    )
    parser.add_argument(
        "--grad-stats-every",
        type=int,
        default=0,
        help="Falls >0: Gradienten-Normen je Loss-Term alle N Schritte (nur z-Latent, retain_graph).",
    )
    parser.add_argument(
        "--atten-scale",
        type=float,
        default=ATTEN_SCALE_DEFAULT,
        help="Globaler Längenskalenfaktor für die Attenuation (μ in 1/cm, Bounding Box ~1).",
    )
    parser.add_argument(
        "--ct-padding-mode",
        type=str,
        default="border",
        choices=["border", "zeros"],
        help="Padding-Mode fuer CT grid_sample (border|zeros).",
    )
    parser.add_argument(
        "--ray-split",
        type=float,
        default=0.8,
        help="Anteil der Rays pro Bild für das Training (Rest = Test) beim stratifizierten Split.",
    )
    parser.add_argument(
        "--inputs-normalized",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
        help="True, wenn AP/PA (und optional CT/ACT) normiert eingespeist werden.",
    )
    parser.add_argument(
        "--ray-split-mode",
        type=str,
        default="tile_random",
        choices=["tile_random", "stratified_intensity"],
        help="Split-Modus: tile_random (Tile-permutiert) oder stratified_intensity (FG/BG stratifiziert).",
    )
    parser.add_argument(
        "--ray-split-seed",
        type=int,
        default=123,
        help="Seed für den stratifizierten Ray-Split.",
    )
    parser.add_argument(
        "--ray-split-tile",
        type=int,
        default=32,
        help="Tile-Kantenlänge in Pixeln für den Ray-Split.",
    )
    parser.add_argument(
        "--ray-fg-thr",
        type=str,
        default="0.0",
        help="Schwellwert für Vordergrund (target>thr). Zahl oder 'quantile'.",
    )
    parser.add_argument(
        "--ray-fg-thr-norm",
        type=float,
        default=None,
        help="Schwellwert für Vordergrund im normierten Raum (überschreibt ray-fg-thr bei normierten Inputs).",
    )
    parser.add_argument(
        "--ray-fg-quantile",
        type=float,
        default=0.90,
        help="Quantil q für FG-Definition, falls ray-fg-thr<=0 oder 'quantile'.",
    )
    parser.add_argument(
        "--ray-train-fg-frac",
        type=float,
        default=0.5,
        help="Anteil Vordergrund-Rays beim Training-Sampling (Rest Hintergrund, mit Fallback).",
    )
    parser.add_argument(
        "--ray-split-enable",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
        help="Aktiviere stratifizierten Ray-Split (False => Legacy-Uniform-Split).",
    )
    parser.add_argument(
        "--pa-xflip",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help="Spiegle PA in x-Richtung, um Pixel zu AP zu mappen.",
    )
    parser.add_argument(
        "--log-quantiles-final-only",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
        help="Logge p50/p80/p95/p99 der rohen AP/PA-Projektionen (Pred + Target) nur am finalen Step.",
    )
    parser.add_argument(
        "--debug-proj-stats",
        action="store_true",
        help="Einmalige AP/PA-Min/Max/p99.9-Statistiken direkt nach dem Laden loggen.",
    )
    parser.add_argument(
        "--log-proj-metrics-physical",
        action="store_true",
        help="Logge PSNR/MAE/Quantiles zusaetzlich im physikalischen Massstab (re-skaliert).",
    )
    parser.add_argument(
        "--export-vol-res",
        type=int,
        default=128,
        help="Grid-Aufloesung fuer Export des finalen Aktivitaetsvolumens (z. B. 128 oder 256).",
    )
    parser.add_argument(
        "--export-vol-every",
        type=int,
        default=0,
        help="Optional: Exportiere Aktivitaetsvolumen alle N Schritte (0 = aus).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def set_seed(seed: int):
    # deterministische Seeds für Torch + NumPy, damit Runs reproduzierbar bleiben
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_img(arr, path, title=None):
    """Robust PNG visualisation with optional logarithmic stretch."""
    import matplotlib.pyplot as plt

    # Nan/Inf-Fälle abfangen, damit matplotlib nicht abstürzt
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Wertebereich auf 0..1 strecken, notfalls via Log-Scaling
    a_min, a_max = float(np.min(arr)), float(np.max(arr))
    if np.isclose(a_min, a_max):
        img = np.zeros_like(arr) if a_max == 0 else arr / (a_max + 1e-8)
    else:
        arr_shift = arr - a_min
        arr_log = np.log1p(arr_shift)
        img = (arr_log - arr_log.min()) / (arr_log.max() - arr_log.min() + 1e-8)

    plt.figure(figsize=(6, 4))
    plt.imshow(img, cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def log_projection_quantiles(ap_pred, pa_pred, ap_target=None, pa_target=None, tag="final"):
    quantiles = [0.5, 0.8, 0.95, 0.99]

    def _q(arr):
        data = arr.detach().float().cpu().numpy().ravel()
        return np.quantile(data, quantiles)

    ap_pred_q = _q(ap_pred)
    pa_pred_q = _q(pa_pred)
    msg = (
        f"[quantiles][{tag}] pred_ap p50={ap_pred_q[0]:.3e} p80={ap_pred_q[1]:.3e} "
        f"p95={ap_pred_q[2]:.3e} p99={ap_pred_q[3]:.3e} | "
        f"pred_pa p50={pa_pred_q[0]:.3e} p80={pa_pred_q[1]:.3e} "
        f"p95={pa_pred_q[2]:.3e} p99={pa_pred_q[3]:.3e}"
    )
    if ap_target is not None and pa_target is not None:
        ap_t_q = np.quantile(ap_target.ravel(), quantiles)
        pa_t_q = np.quantile(pa_target.ravel(), quantiles)
        msg += (
            f" | target_ap p50={ap_t_q[0]:.3e} p80={ap_t_q[1]:.3e} "
            f"p95={ap_t_q[2]:.3e} p99={ap_t_q[3]:.3e} | "
            f"target_pa p50={pa_t_q[0]:.3e} p80={pa_t_q[1]:.3e} "
            f"p95={pa_t_q[2]:.3e} p99={pa_t_q[3]:.3e}"
        )
    print(msg, flush=True)


def log_projection_quantiles_scaled(
    ap_pred,
    pa_pred,
    ap_target=None,
    pa_target=None,
    tag="final",
    ap_scale: float = 1.0,
    pa_scale: float = 1.0,
):
    ap_scale = float(ap_scale)
    pa_scale = float(pa_scale)
    log_projection_quantiles(
        ap_pred * ap_scale,
        pa_pred * pa_scale,
        ap_target=None if ap_target is None else ap_target * ap_scale,
        pa_target=None if pa_target is None else pa_target * pa_scale,
        tag=tag,
    )


def export_activity_volume(generator, z_latent, out_path: Path, res: int, device: torch.device):
    radius = generator.radius
    if isinstance(radius, tuple):
        radius = radius[1]
    radius = float(radius)
    res = int(res)
    if res <= 0:
        raise ValueError("export-vol-res must be > 0")

    x_coords = idx_to_coord(torch.arange(res, device=device), res, radius)
    y_coords = idx_to_coord(torch.arange(res, device=device), res, radius)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
    x_flat = x_grid.reshape(-1)
    y_flat = y_grid.reshape(-1)

    target_points = 262144
    chunk_depth = max(1, min(res, target_points // (res * res) if res * res > 0 else 1))

    vol = np.empty((res, res, res), dtype=np.float32)
    with torch.no_grad():
        for z_start in range(0, res, chunk_depth):
            z_end = min(res, z_start + chunk_depth)
            z_idx = torch.arange(z_start, z_end, device=device)
            z_coords = idx_to_coord(z_idx, res, radius)
            z_rep = z_coords.repeat_interleave(x_flat.numel())
            x_rep = x_flat.repeat(z_coords.numel())
            y_rep = y_flat.repeat(z_coords.numel())
            coords = torch.stack((x_rep, y_rep, z_rep), dim=1)
            pred = query_emission_at_points(generator, z_latent, coords)
            pred = pred.view(z_coords.numel(), res, res).detach().cpu().numpy().astype(np.float32)
            vol[z_start:z_end, :, :] = pred

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, vol)


def build_ray_split(num_pixels: int, split_ratio: float, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Erzeuge einen festen Train/Test-Split über alle Rays einer Ansicht.
    Split ist reproduzierbar, weil der globale Seed (set_seed) bereits gesetzt wurde.

    Rückgabe: {"train": train_idx, "test": test_idx} (jeweils torch.long auf device)
    """
    ratio = float(split_ratio)
    ratio = 0.0 if ratio < 0 else (1.0 if ratio > 1.0 else ratio)
    perm = torch.randperm(num_pixels, device=device)
    n_train = int(math.ceil(num_pixels * ratio))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return {"train": train_idx, "test": test_idx}


def sample_split_indices(split_tensor: torch.Tensor, count: int) -> torch.Tensor:
    """Ziehe zufällige Indizes aus einem vorgegebenen Split (keine neuen Rays von der Gegenseite)."""
    if split_tensor.numel() <= count:
        return split_tensor
    rand_idx = torch.randint(0, split_tensor.numel(), (count,), device=split_tensor.device)
    return split_tensor[rand_idx]


def map_pa_indices_torch(idx: torch.Tensor, W: int, do_flip: bool) -> torch.Tensor:
    if not do_flip:
        return idx
    y = idx // W
    x = idx % W
    return y * W + (W - 1 - x)


def grad_norm_of(loss_term: torch.Tensor, params) -> float:
    """L2-Norm der Gradienten eines Loss-Terms bezogen auf gegebene Parameter (z. B. z_latent)."""
    if loss_term is None or not loss_term.requires_grad:
        return 0.0
    grads = torch.autograd.grad(loss_term, params, retain_graph=True, allow_unused=True)
    grads = [g for g in grads if g is not None]
    if not grads:
        return 0.0
    flat = torch.cat([g.reshape(-1) for g in grads])
    return float(flat.norm().detach().cpu().item())


def safe_git_rev() -> str:
    """Versucht den aktuellen Git-Commit (kurz) zu lesen, fällt andernfalls auf 'unknown' zurück."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent)
        return out.decode().strip()
    except Exception as exc:  # noqa: BLE001 – bewusst breit, nur Debug-Info
        return f"unknown ({exc.__class__.__name__})"


def log_effective_config(outdir: Path, config: dict, args):
    """Einmalige Ausgabe der effektiv genutzten Konfiguration (nach YAML+CLI-Merge)."""
    nerf_cfg = config.get("nerf", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    git_rev = safe_git_rev()
    print(f"[cfg] git_rev={git_rev} | expname={config.get('expname', 'n/a')} | outdir={outdir}", flush=True)
    print(
        f"[cfg][data] act_scale={data_cfg.get('act_scale')} | near={data_cfg.get('near')} | far={data_cfg.get('far')} "
        f"| orthographic={data_cfg.get('orthographic')}",
        flush=True,
    )
    print(
        f"[cfg][nerf] N_samples={nerf_cfg.get('N_samples')} | N_importance={nerf_cfg.get('N_importance')} "
        f"| perturb={nerf_cfg.get('perturb')} | atten_scale={nerf_cfg.get('atten_scale')} "
        f"| use_attenuation={nerf_cfg.get('use_attenuation')}",
        flush=True,
    )
    print(
        f"[cfg][training] lr_g={training_cfg.get('lr_g')} | tv_weight={training_cfg.get('tv_weight')} "
        f"| act_loss_weight={args.act_loss_weight} | act_samples={args.act_samples} | act_pos_weight={args.act_pos_weight} "
        f"| ct_loss_weight={args.ct_loss_weight} | ct_threshold={args.ct_threshold} | z_reg_weight={args.z_reg_weight} "
        f"| ray_tv_weight={args.ray_tv_weight} | ray_tv_edge_aware={args.ray_tv_edge_aware} | ray_tv_alpha={args.ray_tv_alpha} "
        f"| ray_tv_w_clamp_min={args.ray_tv_w_clamp_min} | ct_padding_mode={args.ct_padding_mode} "
        f"| bg_depth_mass_weight={args.bg_depth_mass_weight} | bg_depth_eps={args.bg_depth_eps} | bg_depth_mode={args.bg_depth_mode}",
        flush=True,
    )


def build_loss_weights(target: torch.Tensor, bg_weight: float, threshold: float) -> Optional[torch.Tensor]:
    """Erzeuge optionale Strahl-Gewichte, die Null-Strahlen abschwächen."""
    if bg_weight >= 1.0:
        return None
    weights = torch.ones_like(target)
    weights = weights.masked_fill(target <= threshold, bg_weight)
    return weights


def build_pose_rays(generator, pose):
    """Pre-compute all rays for a fixed pose and keep them on the target device."""
    # Ortho-Kamera nutzt ortho_size statt focal
    focal_or_size = generator.ortho_size if generator.orthographic else generator.focal
    rays_full, _, _ = generator.val_ray_sampler(generator.H, generator.W, focal_or_size, pose)
    return rays_full.to(generator.device, non_blocking=True)


def slice_rays(rays_full: torch.Tensor, ray_idx: torch.Tensor) -> torch.Tensor:
    """Select a subset of rays (by linear indices) for a mini-batch."""
    # rays_full hat Form (2, HW, 3) -> mit Indexliste extrahieren
    return torch.stack(
        (
            rays_full[0, ray_idx],
            rays_full[1, ray_idx],
        ),
        dim=0,
    )


def render_minibatch(generator, z_latent, rays_subset, ct_context=None, return_raw: bool = False):
    """Render a mini-batch of rays from a fixed pose while keeping training kwargs."""
    # train/test kwargs werden durch use_test_kwargs umgeschaltet
    render_kwargs = generator.render_kwargs_train if not generator.use_test_kwargs else generator.render_kwargs_test
    render_kwargs = dict(render_kwargs)
    render_kwargs["features"] = z_latent
    if ct_context is not None:
        render_kwargs["ct_context"] = ct_context
    elif render_kwargs.get("use_attenuation"):
        render_kwargs["use_attenuation"] = False
    if return_raw:
        render_kwargs["retraw"] = True
    if DEBUG_PRINTS:
        render_kwargs["debug_prints"] = True
    proj_map, _, _, extras = generator.render(rays=rays_subset, **render_kwargs)
    return proj_map.view(z_latent.shape[0], -1), extras


def compute_ray_tv(
    raw: torch.Tensor,
    mu_vals: Optional[torch.Tensor] = None,
    edge_aware: bool = False,
    alpha: float = 0.0,
    w_clamp_min: float = 0.0,
    mu_thresh: float = 1e-3,
    return_stats: bool = False,
) -> Tuple[torch.Tensor, Optional[dict]]:
    """Berechnet den (edge-aware) 1D-Total-Variation-Prior entlang jedes Rays."""
    lambda_vals = F.softplus(raw[..., 0])  # [N_rays, N_samples]
    diffs = torch.abs(lambda_vals[..., 1:] - lambda_vals[..., :-1])
    if edge_aware and mu_vals is not None and alpha > 0.0:
        if mu_vals.shape != lambda_vals.shape:
            raise ValueError(f"CT samples have wrong shape {mu_vals.shape}, expected {lambda_vals.shape}.")
        mu = torch.clamp(mu_vals, min=0.0)
        mu_diffs = torch.abs(mu[..., 1:] - mu[..., :-1])
        weights = torch.exp(-alpha * mu_diffs)
        if w_clamp_min > 0.0:
            weights = torch.clamp(weights, min=w_clamp_min)
        tv_per_ray = torch.sum(weights * diffs, dim=-1)
        tv = torch.mean(tv_per_ray)
        if return_stats:
            stats = {
                "w_mean": weights.mean(),
                "w_min": weights.min(),
                "w_max": weights.max(),
            }
            if mu_thresh is not None and mu.numel() > 0:
                mask = mu > mu_thresh
                valid = mask.any(dim=-1)
                if valid.any():
                    first_idx = mask.float().argmax(dim=-1)
                    depths = first_idx[valid].float() / max(1.0, float(mu.shape[-1] - 1))
                    stats["ct_boundary_depth_mean"] = depths.mean()
                    stats["ct_boundary_depth_median"] = depths.median()
            return tv, stats
        return tv, None
    tv_per_ray = torch.sum(diffs, dim=-1)
    tv = torch.mean(tv_per_ray)
    if return_stats:
        return tv, None
    return tv, None


def maybe_render_preview(
    step,
    args,
    generator,
    z_eval,
    outdir,
    ct_volume=None,
    act_volume=None,
    ct_context=None,
):
    # Volle AP/PA-Renderings sind teuer; nur alle N Schritte ausführen
    if args.preview_every <= 0 or (step % args.preview_every) != 0:
        return
    prev_flag = generator.use_test_kwargs
    generator.eval()
    generator.use_test_kwargs = True
    ctx = ct_context or generator.build_ct_context(ct_volume, padding_mode=args.ct_padding_mode)
    with torch.no_grad():
        proj_ap, _, _, _ = generator.render_from_pose(z_eval, generator.pose_ap, ct_context=ctx)
        proj_pa, _, _, _ = generator.render_from_pose(z_eval, generator.pose_pa, ct_context=ctx)
    generator.train()
    generator.use_test_kwargs = prev_flag or False
    H, W = generator.H, generator.W
    ap_np = proj_ap[0].reshape(H, W).detach().cpu().numpy()
    pa_np = proj_pa[0].reshape(H, W).detach().cpu().numpy()
    out_dir = outdir / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_img(ap_np, out_dir / f"step_{step:05d}_AP.png", title=f"AP @ step {step}")
    save_img(pa_np, out_dir / f"step_{step:05d}_PA.png", title=f"PA @ step {step}")
    save_depth_profile(step, generator, z_eval, ct_volume, act_volume, out_dir, proj_ap=proj_ap, proj_pa=proj_pa)
    print("🖼️ Preview gespeichert:", flush=True)
    print("   ", (out_dir / f"step_{step:05d}_AP.png").resolve(), flush=True)
    print("   ", (out_dir / f"step_{step:05d}_PA.png").resolve(), flush=True)


def init_log_file(path: Path):
    # CSV-Header nur einmal schreiben
    if path.exists():
        return
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "loss",
                "loss_ap",
                "loss_pa",
                "loss_act",
                "loss_ct",
                "ray_tv",
                "ray_tv_w",
                "bg_depth_mass",
                "bg_depth_mass_w",
                "bg_depth_frac",
                "loss_tv",
                "zreg",
                "mae_ap",
                "mae_pa",
                "psnr_ap",
                "psnr_pa",
                "pred_mean_ap",
                "pred_mean_pa",
                "pred_std_ap",
                "pred_std_pa",
                "loss_test_all",
                "loss_test_ap",
                "loss_test_pa",
                "psnr_test_all",
                "psnr_test_ap",
                "psnr_test_pa",
                "mae_test_all",
                "mae_test_ap",
                "mae_test_pa",
                "loss_test_fg",
                "psnr_test_fg",
                "mae_test_fg",
                "loss_test_top10",
                "psnr_test_top10",
                "mae_test_top10",
                "iter_ms",
                "lr",
                "ray_tv_mode",
                "ray_tv_w_mean",
            ]
        )


def append_log(path: Path, row):
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_checkpoint(step, generator, z_train, optimizer, scaler, ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Minimal-Checkpoint: coarse/fine Netze, Optimizer, AMP-Scaler
    state = {
        "step": step,
        "z_train": z_train.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "generator_coarse": generator.render_kwargs_train["network_fn"].state_dict(),
        "generator_fine": None,
    }
    if generator.render_kwargs_train["network_fine"] is not None:
        state["generator_fine"] = generator.render_kwargs_train["network_fine"].state_dict()
    ckpt_path = ckpt_dir / f"checkpoint_step{step:05d}.pt"
    torch.save(state, ckpt_path)
    print(f"💾 Checkpoint gespeichert: {ckpt_path}", flush=True)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse + 1e-12)


def save_depth_profile(step, generator, z_latent, ct_vol, act_vol, outdir: Path, proj_ap=None, proj_pa=None):
    """
    Speichert Tiefenprofile (λ/μ/Prediction) entlang ausgewählter Strahlen für Analyse/Debugging.
    """
    H, W = generator.H, generator.W

    ap_img = proj_ap.detach().view(H, W).cpu().numpy() if proj_ap is not None else None
    pa_img = proj_pa.detach().view(H, W).cpu().numpy() if proj_pa is not None else None

    act_data = act_vol.detach().cpu().numpy() if act_vol is not None else None
    act_masks = None
    if act_data is not None:
        if act_data.ndim == 4:
            act_data = act_data.squeeze(0)
        act_zero = act_data < 1e-6
        act_nonzero = act_data > 1e-6
        act_masks = (act_zero.max(axis=0), act_nonzero.max(axis=0))

    def extract_curve(vol: torch.Tensor, y_idx: int, x_idx: int):
        if vol is None:
            return None, None
        vol = vol.detach()
        if vol.dim() == 4:
            vol = vol.squeeze(0)
        if vol.dim() != 3:
            return None, None
        D, H_loc, W_loc = vol.shape[-3:]
        if not (0 <= y_idx < H_loc and 0 <= x_idx < W_loc):
            return None, None
        curve = vol[:, y_idx, x_idx].cpu().numpy()
        z_coords = idx_to_coord(torch.arange(D, device=vol.device), D, generator.radius if not isinstance(generator.radius, tuple) else generator.radius[1])
        return curve, z_coords

    def pick_ray_indices(num_zero: int = 1, num_active: int = 3):
        chosen = []

        def add_unique(idx):
            if idx is None:
                return False
            if idx in chosen:
                return False
            chosen.append(idx)
            return True

        def dist(a, b):
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        def is_far_enough(idx):
            return all(dist(idx, c) > 8 for c in chosen)

        def pick_from_mask(mask, prefer_high: bool):
            if mask is None:
                return None
            mask = mask.copy()
            chosen_mask = np.zeros_like(mask, dtype=bool)
            for y, x in chosen:
                if 0 <= y < H and 0 <= x < W:
                    chosen_mask[y, x] = True
            mask = mask & (~chosen_mask)
            if not mask.any():
                return None
            coords = np.argwhere(mask)
            if coords.size == 0:
                return None

            weight_map = None
            if ap_img is not None and pa_img is not None:
                weight_map = ap_img + pa_img
            if weight_map is not None:
                weights = weight_map[mask]
                if weights.size == 0:
                    weights = None
                else:
                    if prefer_high:
                        weights = weights - weights.min() + 1e-6
                    else:
                        weights = weights.max() - weights + 1e-6
                    if not np.isfinite(weights).any() or np.sum(weights) <= 0:
                        weights = None
            if weights is None:
                np.random.shuffle(coords)
                for y, x in coords:
                    if is_far_enough((int(y), int(x))):
                        return int(y), int(x)
                y, x = coords[0]
                return int(y), int(x)

            for _ in range(min(len(coords), 64)):
                idx = np.random.choice(len(coords), p=weights / weights.sum())
                y, x = coords[idx]
                if is_far_enough((int(y), int(x))):
                    return int(y), int(x)
            y, x = coords[np.argmax(weights)]
            return int(y), int(x)

        def pick_proj_extreme(func):
            if ap_img is None or pa_img is None:
                return None
            combo = (ap_img + pa_img).copy()
            for y, x in chosen:
                if 0 <= y < H and 0 <= x < W:
                    combo[y, x] = np.nan
            try:
                y, x = np.unravel_index(func(combo), combo.shape)
            except ValueError:
                return None
            return int(y), int(x)

        ct_pos_mask = None
        if ct_vol is not None:
            ct_data = ct_vol.detach()
            if ct_data.dim() == 4:
                ct_data = ct_data.squeeze(0)
            if ct_data.dim() == 3:
                ct_depth_max = ct_data.max(dim=0).values.cpu().numpy()
                ct_pos_mask = ct_depth_max > 1e-8

        def combine_mask(base_mask, require_ct: bool):
            if base_mask is None:
                return None
            mask = base_mask.astype(bool)
            if ct_pos_mask is not None and require_ct:
                mask = mask & ct_pos_mask
            return mask

        zero_mask = nonzero_mask = None
        if act_data is not None and act_masks is not None:
            zero_mask, nonzero_mask = act_masks

        zero_needed = max(num_zero, 0)
        active_needed = max(num_active, 0)

        if zero_needed > 0:
            for mask in (combine_mask(zero_mask, True), zero_mask):
                if zero_needed <= 0:
                    break
                if add_unique(pick_from_mask(mask, prefer_high=False)):
                    zero_needed -= 1

        if active_needed > 0:
            for _ in range(active_needed):
                idx = pick_from_mask(combine_mask(nonzero_mask, True), prefer_high=True)
                if not add_unique(idx):
                    break
                active_needed -= 1
            while active_needed > 0:
                idx = pick_from_mask(nonzero_mask, prefer_high=True)
                if idx is None:
                    break
                if add_unique(idx):
                    active_needed -= 1

        if zero_needed > 0:
            if add_unique(pick_proj_extreme(np.nanargmin)):
                zero_needed -= 1

        while active_needed > 0:
            idx = pick_proj_extreme(np.nanargmax)
            if idx is None:
                break
            if add_unique(idx):
                active_needed -= 1

        fixed_coords = [(72, 428), (69, 336)]
        for y_raw, x_raw in fixed_coords:
            if len(chosen) >= num_zero + num_active:
                break
            y = int(np.clip(y_raw, 0, H - 1))
            x = int(np.clip(x_raw, 0, W - 1))
            add_unique((y, x))
        rel_coords = [(0.5, 0.5), (0.25, 0.75), (0.75, 0.25)]
        for ry, rx in rel_coords:
            if len(chosen) >= num_zero + num_active:
                break
            y = int(np.clip(round((H - 1) * ry), 0, H - 1))
            x = int(np.clip(round((W - 1) * rx), 0, W - 1))
            add_unique((y, x))

        if ap_img is not None and pa_img is not None and len(chosen) < num_zero + num_active:
            max_y, max_x = np.unravel_index(np.argmax(ap_img + pa_img), ap_img.shape)
            add_unique((int(max_y), int(max_x)))

        while len(chosen) < num_zero + num_active:
            y = int(np.random.randint(0, H))
            x = int(np.random.randint(0, W))
            add_unique((y, x))

        return chosen[: num_zero + num_active]

    def first_shape(vol_a, vol_b):
        for v in (vol_a, vol_b):
            if v is None:
                continue
            data = v.squeeze(0).detach().cpu().numpy() if v.dim() == 4 else v.detach().cpu().numpy()
            if data.ndim == 3:
                return data.shape
        return None

    target_shape = first_shape(ct_vol, act_vol)
    if target_shape is None:
        return

    D = target_shape[0]
    radius = generator.radius
    if isinstance(radius, tuple):
        radius = radius[1]

    num_zero, num_active = 1, 3
    target_total = max(num_zero + num_active, 1)
    cache_attr = "_depth_profile_rays_cache"
    cache = getattr(generator, cache_attr, None)
    ray_indices_cache = None
    if isinstance(cache, dict):
        cached_indices = cache.get("indices")
        cached_shape = cache.get("shape")
        cached_total = cache.get("total")
        if cached_indices and cached_shape == (generator.H, generator.W) and cached_total == target_total:
            ray_indices_cache = cached_indices

    if ray_indices_cache is None:
        ray_indices = pick_ray_indices(num_zero=num_zero, num_active=num_active)
        setattr(generator, cache_attr, {"indices": list(ray_indices), "shape": (generator.H, generator.W), "total": target_total})
    else:
        ray_indices = ray_indices_cache

    depth_idx = torch.arange(D, device=generator.device)
    z_coords = idx_to_coord(depth_idx, D, radius)
    depth_axis = np.linspace(0.0, 1.0, D)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(ray_indices), figsize=(4 * len(ray_indices), 4), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, (y_idx, x_idx) in zip(axes, ray_indices):
        curves = []
        labels = []
        curve_ct = extract_curve(ct_vol, y_idx, x_idx) if ct_vol is not None else (None, None)
        curve_act = extract_curve(act_vol, y_idx, x_idx) if act_vol is not None else (None, None)

        if curve_ct[0] is not None:
            curves.append(normalize_curve(curve_ct[0].copy()))
            labels.append("μ (CT)")
        if curve_act[0] is not None:
            curves.append(normalize_curve(curve_act[0].copy()))
            labels.append("Aktivität (GT)")

        x_coord = idx_to_coord(torch.tensor(x_idx, device=generator.device), target_shape[2], radius)
        y_coord = idx_to_coord(torch.tensor(y_idx, device=generator.device), target_shape[1], radius)
        coords = torch.stack((x_coord.repeat(D), y_coord.repeat(D), z_coords), dim=1)
        pred = query_emission_at_points(generator, z_latent, coords).detach().cpu().numpy()
        curves.append(normalize_curve(pred.copy()))
        labels.append("Aktivität (NeRF)")

        for curve, label in zip(curves, labels):
            ax.plot(depth_axis, curve, label=label)

        title_extra = []
        if ap_img is not None:
            title_extra.append(f"I_AP={ap_img[y_idx, x_idx]:.2e}")
        if pa_img is not None:
            title_extra.append(f"I_PA={pa_img[y_idx, x_idx]:.2e}")
        aux = " | ".join(title_extra)
        ax.set_title(f"Strahl y={y_idx}, x={x_idx}" + (f"\n{aux}" if aux else ""))
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_ylabel("normierte Intensität")
    for ax in axes:
        ax.set_xlabel("Tiefe (anterior → posterior)")
    fig.suptitle(f"Depth-Profile @ step {step:05d}")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"depth_profile_step_{step:05d}.png", dpi=150)
    plt.close(fig)


def evaluate_pixel_subsets(
    generator,
    z_latent,
    rays_cache,
    subsets: Dict[str, Optional[torch.Tensor]],
    ap_flat_proc: torch.Tensor,
    pa_flat_proc: torch.Tensor,
    rays_per_eval: Optional[int],
    bg_weight: float,
    weight_threshold: float,
    pa_xflip: bool,
    ct_context=None,
    W: int = None,
    scale_ap: Optional[float] = None,
    scale_pa: Optional[float] = None,
):
    """Evaluiert Loss/PSNR/MAE auf gemeinsamen Pixel-Indizes für AP+PA (Loss gemittelt über Views)."""
    prev_flag = generator.use_test_kwargs
    generator.eval()
    generator.use_test_kwargs = True
    results = {}
    with torch.no_grad():
        for name, idx_all in subsets.items():
            if idx_all is None or idx_all.numel() == 0:
                results[name] = None
                continue
            n_sel = idx_all.numel() if rays_per_eval is None else min(idx_all.numel(), rays_per_eval)
            idx_ap = idx_all if rays_per_eval is None else sample_split_indices(idx_all, n_sel)
            idx_pa = map_pa_indices_torch(idx_ap, W, pa_xflip)

            ray_batch_ap = slice_rays(rays_cache["ap"], idx_ap)
            ray_batch_pa = slice_rays(rays_cache["pa"], idx_pa)

            pred_ap, _ = render_minibatch(generator, z_latent, ray_batch_ap, ct_context=ct_context)
            pred_pa, _ = render_minibatch(generator, z_latent, ray_batch_pa, ct_context=ct_context)

            target_ap = ap_flat_proc[0, idx_ap].unsqueeze(0)
            target_pa = pa_flat_proc[0, idx_pa].unsqueeze(0)
            pred_ap = pred_ap.clamp_min(1e-8)
            pred_pa = pred_pa.clamp_min(1e-8)

            weight_ap = build_loss_weights(target_ap, bg_weight, weight_threshold)
            weight_pa = build_loss_weights(target_pa, bg_weight, weight_threshold)
            diff_ap = (pred_ap - target_ap) ** 2
            diff_pa = (pred_pa - target_pa) ** 2
            if weight_ap is not None:
                diff_ap = diff_ap * weight_ap
            if weight_pa is not None:
                diff_pa = diff_pa * weight_pa
            loss_ap = diff_ap.mean()
            loss_pa = diff_pa.mean()
            loss_total = 0.5 * (loss_ap + loss_pa)

            psnr_ap = compute_psnr(pred_ap, target_ap)
            psnr_pa = compute_psnr(pred_pa, target_pa)
            mae_ap = torch.mean(torch.abs(pred_ap - target_ap)).item()
            mae_pa = torch.mean(torch.abs(pred_pa - target_pa)).item()

            phys_metrics = None
            if scale_ap is not None and scale_pa is not None:
                scale_ap_f = float(scale_ap)
                scale_pa_f = float(scale_pa)
                pred_ap_phys = pred_ap * scale_ap_f
                pred_pa_phys = pred_pa * scale_pa_f
                target_ap_phys = target_ap * scale_ap_f
                target_pa_phys = target_pa * scale_pa_f
                psnr_ap_phys = compute_psnr(pred_ap_phys, target_ap_phys)
                psnr_pa_phys = compute_psnr(pred_pa_phys, target_pa_phys)
                mae_ap_phys = torch.mean(torch.abs(pred_ap_phys - target_ap_phys)).item()
                mae_pa_phys = torch.mean(torch.abs(pred_pa_phys - target_pa_phys)).item()
                phys_metrics = {
                    "psnr": 0.5 * (psnr_ap_phys + psnr_pa_phys),
                    "mae": 0.5 * (mae_ap_phys + mae_pa_phys),
                    "view": {
                        "ap": {"psnr": psnr_ap_phys, "mae": mae_ap_phys},
                        "pa": {"psnr": psnr_pa_phys, "mae": mae_pa_phys},
                    },
                }

            results[name] = {
                "loss": loss_total.item(),
                "loss_ap": loss_ap.item(),
                "loss_pa": loss_pa.item(),
                "psnr": 0.5 * (psnr_ap + psnr_pa),
                "mae": 0.5 * (mae_ap + mae_pa),
                "pred_mean": ((float(pred_ap.mean()), float(pred_pa.mean()))),
                "target_mean": ((float(target_ap.mean()), float(target_pa.mean()))),
                "view": {
                    "ap": {"loss": loss_ap.item(), "psnr": psnr_ap, "mae": mae_ap},
                    "pa": {"loss": loss_pa.item(), "psnr": psnr_pa, "mae": mae_pa},
                },
                "phys": phys_metrics,
            }

    if prev_flag:
        generator.eval()
    else:
        generator.train()

    return results


def sample_act_points(
    act: torch.Tensor, nsamples: int, radius: float, pos_fraction: float = 0.5, pos_threshold: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Ziehe zufällige Voxel (coords, values) aus act.npy, halb aus aktiven Voxeln (ACT>0), halb global.
    Gibt zusätzlich einen Bool-Flag pro Sample zurück, der anzeigt, ob es aus ACT>0 stammt.
    """
    if act is None:
        raise ValueError("act tensor missing despite act-loss-weight > 0.")
    # Unterscheide (1,D,H,W) vs (D,H,W)
    if act.dim() == 4:
        act = act.squeeze(0)
    D, H, W = act.shape[-3:]
    flat = act.view(-1)
    nsamples = min(nsamples, flat.numel())
    if nsamples <= 0:
        empty = torch.zeros((0,), device=act.device)
        return empty.reshape(0, 3), empty, empty.bool()

    # Split: pos_fraction aus ACT>pos_threshold, Rest uniform
    num_pos = int(round(float(nsamples) * float(pos_fraction)))
    num_pos = max(0, min(num_pos, nsamples))
    num_all = nsamples - num_pos

    pos_mask = flat > pos_threshold
    pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(-1)

    idx_parts = []
    flag_parts = []

    if num_pos > 0 and pos_idx.numel() > 0:
        perm = torch.randint(0, pos_idx.numel(), (num_pos,), device=act.device)
        idx_pos = pos_idx[perm]
        idx_parts.append(idx_pos)
        flag_parts.append(torch.ones_like(idx_pos, dtype=torch.bool))
    else:
        # Fallback: keine aktiven Voxeln gefunden -> alles aus globalem Sampling ziehen
        num_all = nsamples

    if num_all > 0:
        idx_all = torch.randint(0, flat.numel(), (num_all,), device=act.device)
        idx_parts.append(idx_all)
        flag_parts.append(torch.zeros_like(idx_all, dtype=torch.bool))

    idx = torch.cat(idx_parts, dim=0)
    pos_flags = torch.cat(flag_parts, dim=0)
    values = flat[idx]

    hw = H * W
    z_idx = idx // hw
    y_idx = (idx % hw) // W
    x_idx = idx % W

    coords = torch.stack(
        (
            idx_to_coord(x_idx, W, radius),
            idx_to_coord(y_idx, H, radius),
            idx_to_coord(z_idx, D, radius),
        ),
        dim=1,
    )
    return coords, values, pos_flags


def query_emission_at_points(generator, z_latent, coords: torch.Tensor) -> torch.Tensor:
    """Fragt das NeRF an frei gewählten Koordinaten ab (ohne Integration)."""
    if coords.numel() == 0:
        return torch.tensor([], device=coords.device)
    render_kwargs = generator.render_kwargs_train
    network_fn = render_kwargs["network_fn"]
    network_query_fn = render_kwargs["network_query_fn"]
    pts = coords.unsqueeze(0)
    raw = network_query_fn(pts, None, network_fn, features=z_latent)
    raw = raw.view(-1, raw.shape[-1])
    return F.softplus(raw[:, 0])


def idx_to_coord(idx: torch.Tensor, size: int, radius: float) -> torch.Tensor:
    if size <= 1:
        return torch.zeros_like(idx, dtype=torch.float32)
    return ((idx.float() / (size - 1)) - 0.5) * 2.0 * radius


def normalize_curve(arr: np.ndarray) -> np.ndarray:
    arr = arr - np.min(arr)
    maxv = np.max(arr)
    if maxv > 1e-8:
        arr = arr / maxv
    return arr



def sample_ct_pairs(ct: torch.Tensor, nsamples: int, thresh: float, radius: float):
    """Wählt Voxel-Paare (z,z+1) mit geringer CT-Änderung entlang der Tiefe."""
    if ct.dim() == 4:
        ct = ct.squeeze(0)
    D, H, W = ct.shape[-3:]
    if D < 2:
        return None
    # Differenz entlang z, kleine Gradienten => weiches Gewebe -> Loss erzwingt glatte Emission
    diff = torch.abs(ct[1:, :, :] - ct[:-1, :, :])
    ct_max = torch.max(ct)
    rel_diff = diff / (ct_max + 1e-8) if ct_max > 0 else diff
    mask = diff < thresh
    mask = mask | (rel_diff < thresh)
    valid_idx = mask.nonzero(as_tuple=False)
    if valid_idx.numel() == 0:
        return None
    nsamples = min(nsamples, valid_idx.shape[0])
    perm = torch.randperm(valid_idx.shape[0], device=ct.device)[:nsamples]
    sel = valid_idx[perm]
    z = sel[:, 0]
    y = sel[:, 1]
    x = sel[:, 2]
    z_next = z + 1

    coords1 = torch.stack(
        (idx_to_coord(x, W, radius), idx_to_coord(y, H, radius), idx_to_coord(z, D, radius)),
        dim=1,
    )
    coords2 = torch.stack(
        (idx_to_coord(x, W, radius), idx_to_coord(y, H, radius), idx_to_coord(z_next, D, radius)),
        dim=1,
    )
    weights = torch.clamp(1.0 - diff[sel[:, 0], sel[:, 1], sel[:, 2]], min=0.0)
    return coords1, coords2, weights


def train():
    print(f"▶ {__VERSION__} – starte Training", flush=True)
    args = parse_args()
    global DEBUG_PRINTS
    DEBUG_PRINTS = bool(args.debug_prints)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required – please launch on a GPU node.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    nerf_cfg = config.setdefault("nerf", {})
    nerf_cfg.setdefault("atten_scale", ATTEN_SCALE_DEFAULT)
    if args.atten_scale != ATTEN_SCALE_DEFAULT:
        nerf_cfg["atten_scale"] = float(args.atten_scale)

    data_cfg = config.setdefault("data", {})
    if args.normalize_targets:
        print("⚠️ --normalize-targets ist veraltet und wird ignoriert.", flush=True)
    data_cfg.setdefault("act_scale", 1.0)
    data_cfg["debug_proj_stats"] = bool(args.debug_proj_stats)
    data_cfg["ray_split_ratio"] = float(args.ray_split)
    inputs_normalized = bool(args.inputs_normalized)
    if (not inputs_normalized) and (args.ray_fg_thr_norm is not None):
        raise ValueError("ray-fg-thr-norm set but inputs are not normalized.")
    if (not inputs_normalized) and (args.bg_depth_eps_norm is not None):
        raise ValueError("bg-depth-eps-norm set but inputs are not normalized.")
    if (not inputs_normalized) and (args.ct_threshold_norm is not None):
        raise ValueError("ct-threshold-norm set but inputs are not normalized.")
    bg_depth_eps_used = args.bg_depth_eps_norm if (inputs_normalized and args.bg_depth_eps_norm is not None) else args.bg_depth_eps
    ct_threshold_used = args.ct_threshold_norm if (inputs_normalized and args.ct_threshold_norm is not None) else args.ct_threshold
    ray_fg_thr_norm = args.ray_fg_thr_norm if inputs_normalized else None
    training_cfg = config.setdefault("training", {})
    training_cfg.setdefault("val_interval", 0)
    training_cfg.setdefault("tv_weight", 0.001)
    training_cfg["tv_weight"] = args.tv_weight
    training_cfg.setdefault("ray_tv_weight", 0.0)
    training_cfg["ray_tv_weight"] = args.ray_tv_weight
    training_cfg.setdefault("ray_tv_edge_aware", False)
    training_cfg["ray_tv_edge_aware"] = bool(args.ray_tv_edge_aware)
    training_cfg.setdefault("ray_tv_alpha", 0.0)
    training_cfg["ray_tv_alpha"] = float(args.ray_tv_alpha)
    training_cfg.setdefault("ray_tv_w_clamp_min", 0.0)
    training_cfg["ray_tv_w_clamp_min"] = float(args.ray_tv_w_clamp_min)
    training_cfg.setdefault("bg_depth_mass_weight", 0.0)
    training_cfg["bg_depth_mass_weight"] = float(args.bg_depth_mass_weight)
    training_cfg.setdefault("bg_depth_eps", 1e-10)
    training_cfg["bg_depth_eps"] = float(args.bg_depth_eps)
    training_cfg.setdefault("bg_depth_mode", "integral")
    training_cfg["bg_depth_mode"] = str(args.bg_depth_mode)
    training_cfg.setdefault("act_samples", 16384)
    training_cfg.setdefault("act_pos_weight", 2.0)
    if args.act_samples is None:
        args.act_samples = int(training_cfg.get("act_samples", 16384))
    else:
        training_cfg["act_samples"] = args.act_samples
    if args.act_pos_weight is None:
        args.act_pos_weight = float(training_cfg.get("act_pos_weight", 2.0))
    else:
        training_cfg["act_pos_weight"] = args.act_pos_weight
    training_cfg.setdefault("ct_loss_weight", 0.0)
    training_cfg.setdefault("ct_threshold", 0.05)
    training_cfg.setdefault("ct_samples", 8192)
    training_cfg["ct_loss_weight"] = args.ct_loss_weight
    training_cfg["ct_threshold"] = args.ct_threshold
    training_cfg["ct_samples"] = args.ct_samples
    training_cfg.setdefault("z_reg_weight", 0.0)
    training_cfg["z_reg_weight"] = args.z_reg_weight

    print(f"📂 CWD: {Path.cwd().resolve()}", flush=True)
    outdir = Path(config.get("training", {}).get("outdir", "./results_spect")).expanduser().resolve()
    (outdir / "preview").mkdir(parents=True, exist_ok=True)
    print(f"🗂️ Output-Ordner: {outdir}", flush=True)
    log_effective_config(outdir, config, args)
    ckpt_dir = outdir / "checkpoints"
    log_path = outdir / "train_log.csv"
    init_log_file(log_path)

    dataset, hwfr, _ = get_data(config)
    config["data"]["hwfr"] = hwfr

    batch_size = config["training"]["batch_size"]
    if batch_size != 1:
        raise ValueError("This mini-training script currently assumes batch_size == 1.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["training"]["nworkers"],
        pin_memory=True,
        drop_last=False,
    )

    act_global_scale = float(data_cfg.get("act_scale", 1.0))
    if act_global_scale != 1.0:
        print(f"ℹ️ ACT/λ globaler Faktor (im Loader angewandt): x{act_global_scale}", flush=True)
    if DEBUG_PRINTS:
        print(f"[DEBUG] act_scale={act_global_scale}", flush=True)
    ray_split_ratio = float(data_cfg.get("ray_split_ratio", 0.8))
    ray_split_enabled = bool(args.ray_split_enable)
    ray_split_mode = str(args.ray_split_mode)
    ray_split_seed = int(args.ray_split_seed)
    ray_split_tile = int(max(1, args.ray_split_tile))
    ray_fg_thr = args.ray_fg_thr
    ray_fg_quantile = float(args.ray_fg_quantile)
    pa_xflip = bool(args.pa_xflip)
    ray_train_fg_frac = float(np.clip(args.ray_train_fg_frac, 0.0, 1.0))
    log_proj_metrics_physical = bool(args.log_proj_metrics_physical)
    val_interval = int(training_cfg.get("val_interval", 0) or 0)
    tv_weight = float(training_cfg.get("tv_weight", 0.0))
    ray_tv_weight = float(training_cfg.get("ray_tv_weight", 0.0))
    ray_tv_edge_aware = bool(training_cfg.get("ray_tv_edge_aware", False))
    ray_tv_alpha = float(training_cfg.get("ray_tv_alpha", 0.0))
    ray_tv_w_clamp_min = float(training_cfg.get("ray_tv_w_clamp_min", 0.0))

    generator = build_models(config)
    generator.to(device)
    generator.train()
    generator.use_test_kwargs = False  # enforce training kwargs

    # always provide AP/PA fallback poses if not already configured
    generator.set_fixed_ap_pa(radius=hwfr[3])

    z_dim = config["z_dist"]["dim"]
    z_train = torch.nn.Parameter(torch.zeros(1, z_dim, device=device))
    torch.nn.init.normal_(z_train, mean=0.0, std=1.0)

    # --- Sofortiger Smoke-Test ---
    # Einmal vor dem eigentlichen Training rendern, um Setup/NaNs zu prüfen
    with torch.no_grad():
        generator.eval()
        generator.use_test_kwargs = True
        z_smoke = z_train.detach()
        proj_ap, _, _, _ = generator.render_from_pose(z_smoke, generator.pose_ap)
        proj_pa, _, _, _ = generator.render_from_pose(z_smoke, generator.pose_pa)
        generator.train()
        generator.use_test_kwargs = False

    H, W = generator.H, generator.W
    ap_np = proj_ap[0].reshape(H, W).detach().cpu().numpy()
    pa_np = proj_pa[0].reshape(H, W).detach().cpu().numpy()
    smoke_dir = outdir / "preview"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    save_img(ap_np, smoke_dir / "smoke_AP.png", title="Smoke AP")
    save_img(pa_np, smoke_dir / "smoke_PA.png", title="Smoke PA")
    print("✅ Smoke-Test gespeichert:", flush=True)

    rays_cache = {
        "ap": build_pose_rays(generator, generator.pose_ap),
        "pa": build_pose_rays(generator, generator.pose_pa),
    }
    # Gesamtzahl der Pixel bestimmt die Maximalzahl möglicher Strahlen
    num_pixels = generator.H * generator.W
    pixel_split_np: Optional[PixelSplit] = None
    ray_indices: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}

    def parse_fg_threshold(raw_thr) -> Tuple[float, bool]:
        try:
            return float(raw_thr), False
        except Exception:
            if isinstance(raw_thr, str) and raw_thr.strip().lower() == "quantile":
                return 0.0, True
            raise

    ray_fg_thr_value, ray_fg_force_quantile = parse_fg_threshold(ray_fg_thr)
    if ray_fg_thr_norm is not None:
        if ray_fg_force_quantile:
            raise ValueError("ray-fg-thr-norm cannot be combined with quantile mode.")
        ray_fg_thr_value = float(ray_fg_thr_norm)
    if ray_split_mode not in ("tile_random", "stratified_intensity"):
        raise ValueError(f"Unknown ray_split_mode: {ray_split_mode}")

    def map_pa_indices_torch(idx: torch.Tensor, W: int, do_flip: bool) -> torch.Tensor:
        if not do_flip:
            return idx
        y = idx // W
        x = idx % W
        return y * W + (W - 1 - x)

    def _to_torch(arr: Optional[np.ndarray]):
        if arr is None:
            return None
        return torch.from_numpy(arr.astype(np.int64)).long().to(device, non_blocking=True)

    def _log_split(split: PixelSplit, score_img: np.ndarray, mode: str):
        fg_total = split.train_idx_fg.size + split.test_idx_fg.size
        bg_total = split.train_idx_bg.size + split.test_idx_bg.size
        fg_ratio = fg_total / float(num_pixels) if num_pixels > 0 else 0.0
        bg_ratio = bg_total / float(num_pixels) if num_pixels > 0 else 0.0
        test_total = float(split.test_idx_all.size or 1)
        test_fg_ratio = split.test_idx_fg.size / test_total if test_total > 0 else 0.0
        test_bg_ratio = split.test_idx_bg.size / test_total if test_total > 0 else 0.0
        top10_count = split.test_idx_top10.size if split.test_idx_top10 is not None else 0
        print(
            f"🔀 Pixel split: train={split.train_idx_all.size} | test={split.test_idx_all.size} "
            f"| fg={fg_total} ({fg_ratio:.3f}) | bg={bg_total} ({bg_ratio:.3f}) "
            f"| test_fg={split.test_idx_fg.size} ({test_fg_ratio:.3f}) | test_bg={split.test_idx_bg.size} ({test_bg_ratio:.3f}) "
            f"| test_top10={top10_count} | mode={mode} | tile={ray_split_tile} | thr={split.thr_used:.3e} | seed={ray_split_seed}",
            flush=True,
        )
        score_flat = score_img.reshape(-1)
        fg_all = np.concatenate([split.train_idx_fg, split.test_idx_fg]) if fg_total > 0 else np.array([], dtype=np.int64)
        if fg_all.size > 0:
            top_k = min(5, fg_all.size)
            top_order = np.argpartition(-score_flat[fg_all], top_k - 1)[:top_k]
            top_fg = fg_all[top_order]
            dbg_entries = []
            for idx in top_fg:
                y = int(idx // W)
                x = int(idx % W)
                dbg_entries.append(f"({x},{y},{score_flat[idx]:.3e})")
            print(f"   [pixel-split-debug] FG top-{top_k} (x,y,score): " + ", ".join(dbg_entries), flush=True)
        else:
            print("   [pixel-split-debug] FG top-k: none (no FG pixels).", flush=True)

    fg_thr_used = None
    if ray_split_enabled:
        ref_sample = dataset[0]
        ap_target_np = ref_sample["ap"].squeeze(0).numpy()
        pa_target_np = ref_sample["pa"].squeeze(0).numpy()
        if ap_target_np.shape != (H, W) or pa_target_np.shape != (H, W):
            raise ValueError(f"Unexpected target shape: AP {ap_target_np.shape}, PA {pa_target_np.shape}, expected {(H, W)}")

        if ray_split_mode == "stratified_intensity":
            fg_thr_value = ray_fg_thr_value if not ray_fg_force_quantile else 0.0
            pixel_split_np = make_pixel_split_stratified_intensity(
                ap_target_np,
                pa_target_np,
                train_frac=ray_split_ratio,
                fg_threshold=fg_thr_value,
                fg_quantile=ray_fg_quantile,
                seed=ray_split_seed,
                pa_xflip=pa_xflip,
                topk_frac=0.10,
                fg_threshold_norm=ray_fg_thr_norm,
            )
        else:
            fg_thr_value = ray_fg_thr_value
            if ray_fg_force_quantile:
                fg_thr_value = -abs(ray_fg_quantile)
            pixel_split_np = make_pixel_split_from_ap_pa(
                ap_target_np,
                pa_target_np,
                train_frac=ray_split_ratio,
                tile=ray_split_tile,
                thr=fg_thr_value,
                seed=ray_split_seed,
                pa_xflip=pa_xflip,
                topk_frac=0.10,
                fg_threshold_norm=ray_fg_thr_norm,
            )
        score_img = np.maximum(ap_target_np, pa_target_np[:, ::-1] if pa_xflip else pa_target_np)
        _log_split(pixel_split_np, score_img, ray_split_mode)
        fg_thr_used = float(pixel_split_np.thr_used)

        np.savez(
            outdir / "pixel_split.npz",
            train_idx_all=pixel_split_np.train_idx_all,
            test_idx_all=pixel_split_np.test_idx_all,
            train_idx_fg=pixel_split_np.train_idx_fg,
            train_idx_bg=pixel_split_np.train_idx_bg,
            test_idx_fg=pixel_split_np.test_idx_fg,
            test_idx_bg=pixel_split_np.test_idx_bg,
            test_idx_top10=pixel_split_np.test_idx_top10 if pixel_split_np.test_idx_top10 is not None else np.array([], dtype=np.int64),
            meta=np.array(
                [
                    {
                        "H": H,
                        "W": W,
                        "train_frac": ray_split_ratio,
                        "tile": ray_split_tile,
                        "seed": ray_split_seed,
                        "threshold": pixel_split_np.thr_used,
                        "mode": ray_split_mode,
                        "pa_xflip": pa_xflip,
                    }
                ],
                dtype=object,
            ),
        )

        ray_indices["pixel"] = {
            "train_idx_all": _to_torch(pixel_split_np.train_idx_all),
            "test_idx_all": _to_torch(pixel_split_np.test_idx_all),
            "train_idx_fg": _to_torch(pixel_split_np.train_idx_fg),
            "train_idx_bg": _to_torch(pixel_split_np.train_idx_bg),
            "test_idx_fg": _to_torch(pixel_split_np.test_idx_fg),
            "test_idx_bg": _to_torch(pixel_split_np.test_idx_bg),
            "test_idx_top10": _to_torch(pixel_split_np.test_idx_top10) if pixel_split_np.test_idx_top10 is not None else None,
        }
    else:
        split_uniform = build_ray_split(num_pixels, ray_split_ratio, device)
        ray_indices["pixel"] = {
            "train_idx_all": split_uniform["train"],
            "test_idx_all": split_uniform["test"],
            "train_idx_fg": None,
            "train_idx_bg": None,
            "test_idx_fg": None,
            "test_idx_bg": None,
            "test_idx_top10": None,
        }
        print(
            f"🔀 Legacy Pixel-Split: train={ray_indices['pixel']['train_idx_all'].numel()} / "
            f"test={ray_indices['pixel']['test_idx_all'].numel()} (ratio={ray_split_ratio})",
            flush=True,
        )

    rng_train = np.random.default_rng(ray_split_seed + 12345) if ray_split_enabled else None

    rays_per_proj = args.rays_per_step or config["training"]["chunk"]
    if rays_per_proj <= 0:
        raise ValueError("rays-per-step must be > 0.")
    rays_per_proj = min(rays_per_proj, num_pixels)

    optimizer = torch.optim.Adam(
        list(generator.parameters()) + [z_train],
        lr=config["training"]["lr_g"],
    )
    # MSE-Loss im normierten Raum (Poisson entfernt)

    amp_enabled = bool(config["training"].get("use_amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    data_iter = iter(dataloader)
    ct_context = None
    print(
        f"🚀 Starting emission-NeRF training | steps={args.max_steps} | rays/proj={rays_per_proj} "
        f"| image={generator.H}x{generator.W} | chunk={generator.chunk}"
    )
    scale_ap_used = 1.0
    scale_pa_used = 1.0
    scale_joint_used = 1.0
    scale_missing_warned = False

    for step in range(1, args.max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        ap = batch["ap"].to(device, non_blocking=True).float()
        pa = batch["pa"].to(device, non_blocking=True).float()
        meta = batch.get("meta")
        meta_scale = None
        meta_missing = False
        mu_scale_p99_9 = 1.0
        act_scale_p99_9 = 1.0
        voxel_size_mm = 1.5
        if isinstance(meta, dict):
            meta_scale = meta.get("proj_scale_joint_p99")
            meta_missing = meta.get("proj_scale_joint_p99_missing", False)
            mu_scale_p99_9 = meta.get("mu_scale_p99_9", meta.get("ct_scale_p99_9", 1.0))
            act_scale_p99_9 = meta.get("act_scale_p99_9", 1.0)
            voxel_size_mm = meta.get("voxel_size_mm", 1.5)
            if isinstance(meta_scale, (list, tuple)):
                meta_scale = meta_scale[0] if meta_scale else None
            if torch.is_tensor(meta_scale):
                meta_scale = meta_scale.item() if meta_scale.numel() > 0 else None
            if torch.is_tensor(meta_missing):
                meta_missing = bool(meta_missing.item()) if meta_missing.numel() > 0 else False
            if torch.is_tensor(mu_scale_p99_9):
                mu_scale_p99_9 = mu_scale_p99_9.item() if mu_scale_p99_9.numel() > 0 else 1.0
            if torch.is_tensor(act_scale_p99_9):
                act_scale_p99_9 = act_scale_p99_9.item() if act_scale_p99_9.numel() > 0 else 1.0
            if torch.is_tensor(voxel_size_mm):
                voxel_size_mm = voxel_size_mm.item() if voxel_size_mm.numel() > 0 else 1.5
        if meta_scale is None or (isinstance(meta_scale, float) and math.isnan(meta_scale)) or meta_missing:
            scale_joint_used = 1.0
            if not scale_missing_warned:
                print(
                    "[WARN] proj_scale_joint_p99 fehlt im manifest; physikalische Metriken sind bedeutungslos.",
                    flush=True,
                )
                scale_missing_warned = True
        else:
            scale_joint_used = float(meta_scale)
        if mu_scale_p99_9 is None or not np.isfinite(float(mu_scale_p99_9)) or float(mu_scale_p99_9) <= 0:
            raise ValueError("mu_scale_p99_9 must be > 0.")
        if act_scale_p99_9 is None or not np.isfinite(float(act_scale_p99_9)) or float(act_scale_p99_9) <= 0:
            raise ValueError("act_scale_p99_9 must be > 0.")
        if voxel_size_mm is None or not np.isfinite(float(voxel_size_mm)) or float(voxel_size_mm) <= 0:
            raise ValueError("voxel_size_mm must be > 0.")
        scale_ap_used = scale_joint_used
        scale_pa_used = scale_joint_used
        if step == 1:
            print(
                f"[scale] projections_on_disk_normalized_with_joint_p99: proj_scale_joint_p99={scale_joint_used:.3e}",
                flush=True,
            )
        act_vol = batch.get("act")
        if act_vol is not None:
            if act_vol.numel() == 0:
                act_vol = None
            else:
                act_vol = act_vol.to(device, non_blocking=True)
                if act_scale_p99_9 != 1.0 and float(data_cfg.get("act_scale", 1.0)) != 1.0:
                    raise ValueError("act_scale_p99_9 and data.act_scale both set (double-scaling).")
        ct_vol = batch.get("ct")
        if ct_vol is not None:
            ct_vol = ct_vol.to(device, non_blocking=True).float()
        ct_context = (
            generator.build_ct_context(
                ct_vol,
                padding_mode=args.ct_padding_mode,
                voxel_size_mm=float(voxel_size_mm),
                mu_scale_p99_9=float(mu_scale_p99_9),
            )
            if ct_vol is not None
            else None
        )

        # Wichtig: Flatten-Order ist (y * W + x), identisch zu den Ray-Indizes aus make_stratified_tile_split.
        # Keine permute/transpose zwischen (H, W) und reshape(-1), damit Target/Predict exakt die gleiche Reihenfolge teilen.
        ap_flat = ap.view(batch_size, -1)
        pa_flat = pa.view(batch_size, -1)
        ap_flat_proc = ap_flat
        pa_flat_proc = pa_flat

        z_latent = z_train

        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()

        need_ray_tv = ray_tv_weight != 0.0
        need_bg_depth = args.bg_depth_mass_weight > 0.0
        need_raw = need_ray_tv or need_bg_depth

        if ray_split_enabled and pixel_split_np is not None and rng_train is not None:
            idx_np = sample_train_indices(pixel_split_np, rays_per_proj, ray_train_fg_frac, rng_train)
            idx_ap = torch.from_numpy(idx_np).long().to(device, non_blocking=True)
            idx_pa = map_pa_indices_torch(idx_ap, W, pa_xflip)
        else:
            idx_ap = sample_split_indices(ray_indices["pixel"]["train_idx_all"], rays_per_proj)
            idx_pa = map_pa_indices_torch(idx_ap, W, pa_xflip)

        ray_batch_ap = slice_rays(rays_cache["ap"], idx_ap)
        ray_batch_pa = slice_rays(rays_cache["pa"], idx_pa)


        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred_ap, extras_ap = render_minibatch(
                generator, z_latent, ray_batch_ap, ct_context=ct_context, return_raw=need_raw
            )
            pred_pa, extras_pa = render_minibatch(
                generator, z_latent, ray_batch_pa, ct_context=ct_context, return_raw=need_raw
            )

            target_ap = ap_flat_proc[0, idx_ap].unsqueeze(0)
            target_pa = pa_flat_proc[0, idx_pa].unsqueeze(0)
            if DEBUG_PRINTS and (step % args.log_every == 0):
                with torch.no_grad():
                    q_ap = float(torch.quantile(target_ap.view(-1), 0.999).item()) if target_ap.numel() > 0 else float("nan")
                    q_pa = float(torch.quantile(target_pa.view(-1), 0.999).item()) if target_pa.numel() > 0 else float("nan")
                    min_ap = float(target_ap.min().item()) if target_ap.numel() > 0 else float("nan")
                    max_ap = float(target_ap.max().item()) if target_ap.numel() > 0 else float("nan")
                    min_pa = float(target_pa.min().item()) if target_pa.numel() > 0 else float("nan")
                    max_pa = float(target_pa.max().item()) if target_pa.numel() > 0 else float("nan")
                print(
                    f"[norm-check] target_ap min/max={min_ap:.3e}/{max_ap:.3e} p99.9={q_ap:.3e} | "
                    f"target_pa min/max={min_pa:.3e}/{max_pa:.3e} p99.9={q_pa:.3e}",
                    flush=True,
                )

            # Poisson-NLL erwartet pred >= 0
            pred_ap_raw = pred_ap.clamp_min(1e-8)
            pred_pa_raw = pred_pa.clamp_min(1e-8)

            pred_ap = pred_ap_raw
            pred_pa = pred_pa_raw

            weight_ap = build_loss_weights(target_ap, args.bg_weight, args.weight_threshold)
            weight_pa = build_loss_weights(target_pa, args.bg_weight, args.weight_threshold)

            if DEBUG_PRINTS and fg_thr_used is not None and (step % args.log_every == 0):
                with torch.no_grad():
                    score = torch.maximum(target_ap, target_pa)
                    fg_frac = float((score > fg_thr_used).float().mean().item()) if score.numel() > 0 else 0.0
                print(
                    f"[fg-check] fg_frac={fg_frac:.3f} (thr={fg_thr_used:.3e})",
                    flush=True,
                )

            diff_ap = (pred_ap - target_ap) ** 2
            diff_pa = (pred_pa - target_pa) ** 2
            if weight_ap is not None:
                diff_ap = diff_ap * weight_ap
            if weight_pa is not None:
                diff_pa = diff_pa * weight_pa
            loss_ap = diff_ap.mean()
            loss_pa = diff_pa.mean()
            loss = 0.5 * (loss_ap + loss_pa)
            if DEBUG_PRINTS and (step % 50 == 0):
                print(
                    f"[DEBUG][step {step}] TARGET AP min/max: {target_ap.min().item():.3e}/{target_ap.max().item():.3e} | "
                    f"PRED AP min/max: {pred_ap.min().item():.3e}/{pred_ap.max().item():.3e} | "
                    f"TARGET PA min/max: {target_pa.min().item():.3e}/{target_pa.max().item():.3e} | "
                    f"PRED PA min/max: {pred_pa.min().item():.3e}/{pred_pa.max().item():.3e}",
                    flush=True,
                )

            bg_depth_mass = torch.tensor(0.0, device=device)
            bg_depth_mass_w = torch.tensor(0.0, device=device)
            bg_depth_frac_t = torch.tensor(0.0, device=device)
            if args.bg_depth_mass_weight > 0.0:
                bg_mask = None
                if target_ap is not None and target_pa is not None:
                    bg_mask = (target_ap < bg_depth_eps_used) & (target_pa < bg_depth_eps_used)
                elif target_ap is not None:
                    bg_mask = target_ap < bg_depth_eps_used
                elif target_pa is not None:
                    bg_mask = target_pa < bg_depth_eps_used
                if bg_mask is not None:
                    bg_mask_flat = bg_mask.reshape(-1)
                    bg_depth_frac_t = bg_mask_flat.float().mean()
                    bg_terms = []
                    for extras in (extras_ap, extras_pa):
                        if not isinstance(extras, dict):
                            continue
                        raw_out = extras.get("raw")
                        if raw_out is None:
                            continue
                        lambda_vals = F.softplus(raw_out[..., 0])
                        if args.bg_depth_mode == "integral":
                            dists = extras.get("dists")
                            if dists is None:
                                z_vals = extras.get("z_vals")
                                if z_vals is not None:
                                    dists = z_vals[..., 1:] - z_vals[..., :-1]
                                    dists = torch.cat([dists, dists[..., -1:].clone()], dim=-1)
                            if dists is None:
                                continue
                            m_ray = torch.sum(lambda_vals * dists, dim=-1)
                        else:
                            m_ray = torch.mean(lambda_vals, dim=-1)
                        if m_ray.shape[0] != bg_mask_flat.shape[0]:
                            continue
                        if bg_mask_flat.any():
                            bg_terms.append(m_ray[bg_mask_flat].mean())
                    if bg_terms:
                        bg_depth_mass = torch.stack(bg_terms).mean()
                        bg_depth_mass_w = bg_depth_mass * args.bg_depth_mass_weight
                        loss = loss + bg_depth_mass_w

            loss_act = torch.tensor(0.0, device=device)
            if args.act_loss_weight > 0.0 and act_vol is not None:
                radius = generator.radius
                if isinstance(radius, tuple):
                    radius = radius[1]
                # Stichprobe aus act.npy und direkte Dichteabfrage im NeRF
                coords, act_samples, pos_flags = sample_act_points(
                    act_vol, args.act_samples, radius=radius, pos_fraction=0.5, pos_threshold=1e-8
                )
                pred_act = query_emission_at_points(generator, z_latent, coords)
                if pred_act.numel() > 0:
                    if act_scale_p99_9 != 1.0:
                        pred_act = pred_act * float(act_scale_p99_9)
                        act_samples = act_samples * float(act_scale_p99_9)
                    weights_act = torch.where(
                        pos_flags,
                        torch.full_like(pred_act, args.act_pos_weight),
                        torch.ones_like(pred_act),
                    )
                    diff = torch.abs(pred_act - act_samples)
                    loss_act = torch.mean(weights_act * diff)
                    loss = loss + args.act_loss_weight * loss_act

            loss_ct = torch.tensor(0.0, device=device)
            if args.ct_loss_weight > 0.0 and ct_vol is not None:
                radius = generator.radius
                if isinstance(radius, tuple):
                    radius = radius[1]
                ct_pairs = sample_ct_pairs(ct_vol, args.ct_samples, ct_threshold_used, radius=radius)
                if ct_pairs is not None:
                    coords1, coords2, weights = ct_pairs
                    pred1 = query_emission_at_points(generator, z_latent, coords1)
                    pred2 = query_emission_at_points(generator, z_latent, coords2)
                    # Loss zwingt Emission auf flachen CT-Strecken zur Konstanz
                    loss_ct = torch.mean(torch.abs(pred1 - pred2) * weights)
                    loss = loss + args.ct_loss_weight * loss_ct

            loss_reg = torch.tensor(0.0, device=device)
            if args.z_reg_weight > 0.0:
                loss_reg = z_latent.pow(2).mean()
                loss = loss + args.z_reg_weight * loss_reg

            tv_base_loss = torch.tensor(0.0, device=device)
            loss_tv = torch.tensor(0.0, device=device)
            loss_ray_tv = torch.tensor(0.0, device=device)
            loss_ray_tv_w = torch.tensor(0.0, device=device)
            ray_tv_mode = "plain"
            ray_tv_w_mean = None
            ray_tv_w_min = None
            ray_tv_w_max = None
            ct_boundary_depth_mean = None
            ct_boundary_depth_median = None

            tv_base_terms = []
            if isinstance(extras_ap, dict):
                base_val = extras_ap.get("tv_base_loss") or extras_ap.get("tv_loss")
                if base_val is not None:
                    tv_base_terms.append(base_val)
            if isinstance(extras_pa, dict):
                base_val = extras_pa.get("tv_base_loss") or extras_pa.get("tv_loss")
                if base_val is not None:
                    tv_base_terms.append(base_val)

            if tv_base_terms:
                tv_base_loss = torch.stack(tv_base_terms).mean()

            if tv_weight != 0.0:
                loss_tv = tv_weight * tv_base_loss
                loss = loss + loss_tv

            if ray_tv_weight != 0.0:
                edge_aware_active = ray_tv_edge_aware and ray_tv_alpha > 0.0
                ray_tv_terms = []
                ray_tv_w_terms = []
                for extras in (extras_ap, extras_pa):
                    if not isinstance(extras, dict):
                        continue
                    raw_out = extras.get("raw")
                    if raw_out is None:
                        continue
                    if edge_aware_active:
                        mu_out = extras.get("mu")
                        tv_val, w_stats = compute_ray_tv(
                            raw_out,
                            mu_vals=mu_out,
                            edge_aware=True,
                            alpha=ray_tv_alpha,
                            w_clamp_min=ray_tv_w_clamp_min,
                            return_stats=True,
                        )
                        if isinstance(w_stats, dict):
                            w_mean = w_stats.get("w_mean")
                            if w_mean is not None:
                                ray_tv_w_terms.append(w_mean)
                            ray_tv_w_min = w_stats.get("w_min")
                            ray_tv_w_max = w_stats.get("w_max")
                            ct_boundary_depth_mean = w_stats.get("ct_boundary_depth_mean")
                            ct_boundary_depth_median = w_stats.get("ct_boundary_depth_median")
                        ray_tv_terms.append(tv_val)
                    else:
                        tv_val, _ = compute_ray_tv(raw_out)
                        ray_tv_terms.append(tv_val)
                if ray_tv_terms:
                    loss_ray_tv = torch.stack(ray_tv_terms).mean()
                    loss_ray_tv_w = loss_ray_tv * ray_tv_weight
                    loss = loss + loss_ray_tv_w
                if edge_aware_active and ray_tv_w_terms:
                    ray_tv_w_mean = torch.stack(ray_tv_w_terms).mean().item()
                    ray_tv_mode = "edgeaware"

        if args.grad_stats_every > 0 and (step % args.grad_stats_every) == 0:
            proj_loss_for_grad = 0.5 * (loss_ap + loss_pa)
            grad_stats = {
                "proj": grad_norm_of(proj_loss_for_grad, [z_latent]),
                "act": grad_norm_of(args.act_loss_weight * loss_act, [z_latent]) if args.act_loss_weight > 0 else 0.0,
                "ct": grad_norm_of(args.ct_loss_weight * loss_ct, [z_latent]) if args.ct_loss_weight > 0 else 0.0,
                "zreg": grad_norm_of(args.z_reg_weight * loss_reg, [z_latent]) if args.z_reg_weight > 0 else 0.0,
            }
            print(
                f"[grad][step {step:05d}] ||g_proj||={grad_stats['proj']:.3e} "
                f"| ||g_act||={grad_stats['act']:.3e} | ||g_ct||={grad_stats['ct']:.3e} "
                f"| ||g_zreg||={grad_stats['zreg']:.3e}",
                flush=True,
            )

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        iter_ms = (time.perf_counter() - t0) * 1000.0

        with torch.no_grad():
            mae_ap = torch.mean(torch.abs(pred_ap - target_ap)).item()
            mae_pa = torch.mean(torch.abs(pred_pa - target_pa)).item()
            pred_mean = (pred_ap.mean().item(), pred_pa.mean().item())              # skaliert gemäß Projektnorm
            pred_std = (pred_ap.std().item(), pred_pa.std().item())
            pred_mean_raw = (pred_ap_raw.mean().item(), pred_pa_raw.mean().item())  # physikalischer Maßstab
            pred_std_raw = (pred_ap_raw.std().item(), pred_pa_raw.std().item())
            psnr_ap = compute_psnr(pred_ap, target_ap)
            psnr_pa = compute_psnr(pred_pa, target_pa)
            psnr_ap_phys = None
            psnr_pa_phys = None
            mae_ap_phys = None
            mae_pa_phys = None
            if log_proj_metrics_physical:
                pred_ap_phys = pred_ap * float(scale_ap_used)
                pred_pa_phys = pred_pa * float(scale_pa_used)
                target_ap_phys = target_ap * float(scale_ap_used)
                target_pa_phys = target_pa * float(scale_pa_used)
                psnr_ap_phys = compute_psnr(pred_ap_phys, target_ap_phys)
                psnr_pa_phys = compute_psnr(pred_pa_phys, target_pa_phys)
                mae_ap_phys = torch.mean(torch.abs(pred_ap_phys - target_ap_phys)).item()
                mae_pa_phys = torch.mean(torch.abs(pred_pa_phys - target_pa_phys)).item()
            bg_depth_frac = float(bg_depth_frac_t.detach().cpu().item())
            val_stats = None
            if val_interval > 0 and (step % val_interval) == 0 and (not args.no_val):
                rays_eval = None if ray_split_enabled else rays_per_proj
                # Testmetriken:
                # test_all  → gesamter Test-Split (dominiert von BG, kann “zu gut” aussehen)
                # test_fg   → nur Vordergrund-Rays, misst eigentliche Rekonstruktionsqualität
                # test_top10→ oberste 10% Test-Intensitäten, fokussiert auf stärkste Aktivität
                subsets = {
                    "test_all": ray_indices["pixel"]["test_idx_all"],
                }
                if ray_split_enabled:
                    subsets["test_fg"] = ray_indices["pixel"]["test_idx_fg"]
                    subsets["test_top10"] = ray_indices["pixel"]["test_idx_top10"]
                    subsets["test_bg"] = ray_indices["pixel"]["test_idx_bg"]
                val_stats = evaluate_pixel_subsets(
                    generator,
                    z_latent.detach(),
                    rays_cache,
                    subsets=subsets,
                    ap_flat_proc=ap_flat_proc,
                    pa_flat_proc=pa_flat_proc,
                    rays_per_eval=rays_eval,
                    bg_weight=args.bg_weight,
                    weight_threshold=args.weight_threshold,
                    pa_xflip=pa_xflip,
                    ct_context=ct_context,
                    W=W,
                    scale_ap=scale_ap_used if log_proj_metrics_physical else None,
                    scale_pa=scale_pa_used if log_proj_metrics_physical else None,
                )
        val_all = val_stats.get("test_all") if isinstance(val_stats, dict) else None
        val_fg = val_stats.get("test_fg") if isinstance(val_stats, dict) else None
        val_top10 = val_stats.get("test_top10") if isinstance(val_stats, dict) else None
        val_bg = val_stats.get("test_bg") if isinstance(val_stats, dict) else None

        val_loss = val_all["loss"] if val_all is not None else None
        val_psnr = val_all["psnr"] if val_all is not None else None
        val_mae = val_all["mae"] if val_all is not None else None

        val_loss_fg = val_fg["loss"] if val_fg is not None else None
        val_psnr_fg = val_fg["psnr"] if val_fg is not None else None
        val_mae_fg = val_fg["mae"] if val_fg is not None else None
        val_loss_bg = val_bg["loss"] if val_bg is not None else None
        val_psnr_bg = val_bg["psnr"] if val_bg is not None else None
        val_mae_bg = val_bg["mae"] if val_bg is not None else None
        val_pred_mean_bg = val_bg.get("pred_mean") if val_bg is not None else None
        val_target_mean_bg = val_bg.get("target_mean") if val_bg is not None else None
        val_view_all = val_all.get("view") if val_all is not None else None
        val_phys_all = val_all.get("phys") if val_all is not None else None
        val_phys_fg = val_fg.get("phys") if val_fg is not None else None
        val_phys_bg = val_bg.get("phys") if val_bg is not None else None
        val_phys_top10 = val_top10.get("phys") if val_top10 is not None else None
        val_loss_ap = val_view_all["ap"]["loss"] if val_view_all is not None else None
        val_loss_pa = val_view_all["pa"]["loss"] if val_view_all is not None else None
        val_psnr_ap_val = val_view_all["ap"]["psnr"] if val_view_all is not None else None
        val_psnr_pa_val = val_view_all["pa"]["psnr"] if val_view_all is not None else None
        val_mae_ap_val = val_view_all["ap"]["mae"] if val_view_all is not None else None
        val_mae_pa_val = val_view_all["pa"]["mae"] if val_view_all is not None else None

        val_loss_top10 = val_top10["loss"] if val_top10 is not None else None
        val_psnr_top10 = val_top10["psnr"] if val_top10 is not None else None
        val_mae_top10 = val_top10["mae"] if val_top10 is not None else None

        msg = (
            f"[step {step:05d}] loss={loss.item():.6f} | ap={loss_ap.item():.6f} | pa={loss_pa.item():.6f} "
            f"| act={loss_act.item():.6f} | ct={loss_ct.item():.6f} "
            f"| ray_tv={loss_ray_tv.item():.6f} | ray_tv_w={loss_ray_tv_w.item():.6f} "
            f"| bg_depth_mass={bg_depth_mass.item():.6f} | bg_depth_mass_w={bg_depth_mass_w.item():.6f} | bg_depth_frac={bg_depth_frac:.4f} "
            f"| tv={loss_tv.item():.6f} | zreg={loss_reg.item():.6f} "
            f"| mae_ap={mae_ap:.6f} | mae_pa={mae_pa:.6f} "
            f"| psnr_ap={psnr_ap:.2f} | psnr_pa={psnr_pa:.2f} "
            f"| predμ_raw=({pred_mean_raw[0]:.3e},{pred_mean_raw[1]:.3e}) predσ_raw=({pred_std_raw[0]:.3e},{pred_std_raw[1]:.3e}) "
            f"| predμ=({pred_mean[0]:.3e},{pred_mean[1]:.3e}) predσ=({pred_std[0]:.3e},{pred_std[1]:.3e})"
        )
        if log_proj_metrics_physical and psnr_ap_phys is not None:
            msg += (
                f" | mae_ap_phys={mae_ap_phys:.6f} | mae_pa_phys={mae_pa_phys:.6f} "
                f"| psnr_ap_phys={psnr_ap_phys:.2f} | psnr_pa_phys={psnr_pa_phys:.2f}"
            )
        if ray_tv_weight != 0.0:
            msg += f" | ray_tv_mode={ray_tv_mode}"
            if ray_tv_w_mean is not None:
                msg += f" | ray_tv_w_mean={ray_tv_w_mean:.6f}"
            if ray_tv_w_min is not None and ray_tv_w_max is not None:
                msg += f" | ray_tv_w_min={float(ray_tv_w_min):.6f} | ray_tv_w_max={float(ray_tv_w_max):.6f}"
            if ct_boundary_depth_mean is not None and ct_boundary_depth_median is not None:
                msg += (
                    f" | ct_bnd_mean={float(ct_boundary_depth_mean):.3f}"
                    f" | ct_bnd_med={float(ct_boundary_depth_median):.3f}"
                )
        if val_all is not None:
            msg += (
                f" | test_all_loss={val_loss:.6f} | test_all_psnr={val_psnr:.2f} | test_all_mae={val_mae:.6f}"
            )
        if val_fg is not None:
            msg += (
                f" | test_fg_loss={val_loss_fg:.6f} | test_fg_psnr={val_psnr_fg:.2f} | test_fg_mae={val_mae_fg:.6f}"
            )
        if val_top10 is not None:
            msg += (
                f" | test_top10_loss={val_loss_top10:.6f} | test_top10_psnr={val_psnr_top10:.2f} "
                f"| test_top10_mae={val_mae_top10:.6f}"
            )
        if val_bg is not None:
            msg += (
                f" | test_bg_loss={val_loss_bg:.6f} | test_bg_psnr={val_psnr_bg:.2f} | test_bg_mae={val_mae_bg:.6f}"
            )
        if log_proj_metrics_physical and val_phys_all is not None:
            msg += (
                f" | test_all_psnr_phys={val_phys_all['psnr']:.2f} | test_all_mae_phys={val_phys_all['mae']:.6f}"
            )
        if log_proj_metrics_physical and val_phys_fg is not None:
            msg += (
                f" | test_fg_psnr_phys={val_phys_fg['psnr']:.2f} | test_fg_mae_phys={val_phys_fg['mae']:.6f}"
            )
        if log_proj_metrics_physical and val_phys_top10 is not None:
            msg += (
                f" | test_top10_psnr_phys={val_phys_top10['psnr']:.2f} | test_top10_mae_phys={val_phys_top10['mae']:.6f}"
            )
        if log_proj_metrics_physical and val_phys_bg is not None:
            msg += (
                f" | test_bg_psnr_phys={val_phys_bg['psnr']:.2f} | test_bg_mae_phys={val_phys_bg['mae']:.6f}"
            )
        if val_pred_mean_bg is not None and val_target_mean_bg is not None:
            print(
                f"[ray-split-bg-check] mean target={val_target_mean_bg[0]:.3e}/{val_target_mean_bg[1]:.3e} "
                f"mean pred={val_pred_mean_bg[0]:.3e}/{val_pred_mean_bg[1]:.3e}",
                flush=True,
            )
        if val_view_all is not None:
            msg += (
                f" | test_ap_loss={val_loss_ap:.6f} | test_ap_psnr={val_psnr_ap_val:.2f} | test_ap_mae={val_mae_ap_val:.6f}"
                f" | test_pa_loss={val_loss_pa:.6f} | test_pa_psnr={val_psnr_pa_val:.2f} | test_pa_mae={val_mae_pa_val:.6f}"
            )
        print(msg, flush=True)
        append_log(
            log_path,
            [
                step,
                loss.item(),
                loss_ap.item(),
                loss_pa.item(),
                loss_act.item(),
                loss_ct.item(),
                loss_ray_tv.item(),
                loss_ray_tv_w.item(),
                bg_depth_mass.item(),
                bg_depth_mass_w.item(),
                bg_depth_frac,
                loss_tv.item(),
                loss_reg.item(),
                mae_ap,
                mae_pa,
                psnr_ap,
                psnr_pa,
                pred_mean[0],
                pred_mean[1],
                pred_std[0],
                pred_std[1],
                val_loss,
                val_loss_ap,
                val_loss_pa,
                val_psnr,
                val_psnr_ap_val,
                val_psnr_pa_val,
                val_mae,
                val_mae_ap_val,
                val_mae_pa_val,
                val_loss_fg,
                val_psnr_fg,
                val_mae_fg,
                val_loss_top10,
                val_psnr_top10,
                val_mae_top10,
                iter_ms,
                optimizer.param_groups[0]["lr"],
                ray_tv_mode,
                ray_tv_w_mean,
            ],
        )
        if args.save_every > 0 and (step % args.save_every == 0):
            save_checkpoint(step, generator, z_train, optimizer, scaler, ckpt_dir)
        maybe_render_preview(
            step,
            args,
            generator,
            z_train.detach(),
            outdir,
            ct_vol,
            act_vol,
            ct_context,
        )
        if args.export_vol_every > 0 and (step % args.export_vol_every == 0):
            export_path = outdir / f"activity_pred_step_{step:05d}.npy"
            export_activity_volume(generator, z_train.detach(), export_path, args.export_vol_res, device)

    prev_flag = generator.use_test_kwargs
    generator.eval()
    generator.use_test_kwargs = True
    with torch.no_grad():
        proj_ap, _, _, _ = generator.render_from_pose(z_train.detach(), generator.pose_ap, ct_context=ct_context)
        proj_pa, _, _, _ = generator.render_from_pose(z_train.detach(), generator.pose_pa, ct_context=ct_context)
    generator.train()
    generator.use_test_kwargs = prev_flag or False

    H, W = generator.H, generator.W
    ap_np = proj_ap[0].reshape(H, W).detach().cpu().numpy()
    pa_np = proj_pa[0].reshape(H, W).detach().cpu().numpy()
    fp = outdir / "preview"
    fp.mkdir(parents=True, exist_ok=True)
    if args.log_quantiles_final_only:
        ap_t_np = ap.detach().cpu().numpy()[0] if ap is not None else None
        pa_t_np = pa.detach().cpu().numpy()[0] if pa is not None else None
        log_projection_quantiles(proj_ap, proj_pa, ap_target=ap_t_np, pa_target=pa_t_np, tag="final")
        if log_proj_metrics_physical:
            log_projection_quantiles_scaled(
                proj_ap,
                proj_pa,
                ap_target=ap_t_np,
                pa_target=pa_t_np,
                tag="final_phys",
                ap_scale=scale_ap_used,
                pa_scale=scale_pa_used,
            )
    save_img(ap_np, fp / "final_AP.png", "AP final")
    save_img(pa_np, fp / "final_PA.png", "PA final")
    print("🖼️ Finale Previews gespeichert.", flush=True)
    print("   ", (fp / "final_AP.png").resolve(), flush=True)
    print("   ", (fp / "final_PA.png").resolve(), flush=True)

    export_activity_volume(generator, z_train.detach(), outdir / "activity_pred_final.npy", args.export_vol_res, device)
    save_checkpoint(args.max_steps, generator, z_train, optimizer, scaler, ckpt_dir)
    print("✅ Training run finished.", flush=True)


if __name__ == "__main__":
    train()
