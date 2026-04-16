"""Mini-training script for the SPECT emission NeRF."""
import argparse
import csv
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from torch.utils.data import DataLoader

from graf.config import get_data, build_models

__VERSION__ = "emission-train v0.3"
DEBUG_PRINTS = False  # Nur Debug-Ausgaben, keine Änderung am Verhalten


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
        "--projection-normalization",
        type=str,
        default=None,
        choices=["none", "per_dataset", "per_projection"],
        help="Override data.projection_normalization from the config (none|per_dataset|per_projection).",
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
        "--tv-weight-mu",
        type=float,
        default=0.0,
        help="Gewicht für CT-gewichtete (edge-aware) TV entlang der Rays (0 = deaktiviert).",
    )
    parser.add_argument(
        "--tv-mu-sigma",
        type=float,
        default=1.0,
        help="Skalenparameter für μ-Differenzen im edge-aware TV-Weighting.",
    )
    parser.add_argument(
        "--mu-gate-weight",
        type=float,
        default=0.0,
        help="Gewicht für μ-basierten Emissions-Prior (0 = deaktiviert).",
    )
    parser.add_argument(
        "--mu-gate-mode",
        type=str,
        default="none",
        choices=["none", "bandpass", "lowpass", "highpass"],
        help="μ-Prior-Modus: none | bandpass | lowpass | highpass.",
    )
    parser.add_argument(
        "--mu-gate-center",
        type=float,
        default=0.2,
        help="Zentrum der bevorzugten μ-Region für den μ-Prior.",
    )
    parser.add_argument(
        "--mu-gate-width",
        type=float,
        default=0.1,
        help="Breite/Toleranz der bevorzugten μ-Region für den μ-Prior.",
    )
    parser.add_argument(
        "--tv3d-weight",
        type=float,
        default=0.0,
        help="Gewicht für eine optionale 3D-TV-Hülle (Stub, 0 = deaktiviert).",
    )
    parser.add_argument(
        "--tv3d-grid-size",
        type=int,
        default=32,
        help="Gittergröße für die (Stub-)3D-TV-Berechnung.",
    )
    parser.add_argument(
        "--mask-loss-weight",
        type=float,
        default=0.0,
        help="Gewicht für den Organmasken-Constraint (0 = deaktiviert).",
    )
    parser.add_argument(
        "--use-organ-mask",
        action="store_true",
        help="Aktiviert die Nutzung von mask.npy (Organmaske) für einen Masken-Loss.",
    )
    parser.add_argument(
        "--debug-zero-var",
        action="store_true",
        help="Aktiviere zusätzliche Diagnostik und speichere Zwischenergebnisse, sobald Vorhersagen konstante Werte liefern.",
    )
    parser.add_argument(
        "--debug-attenuation-ray",
        action="store_true",
        help="Logge λ/μ/T für einen Beispielstrahl (benötigt nerf.attenuation_debug=True).",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Rendert nur einen schnellen AP/PA-Preview (Step 0) mit aktuellen Orientierungen und beendet.",
    )
    parser.add_argument(
        "--debug-ap-pa",
        action="store_true",
        help="Aktiviert Debug-Vergleich der ungeflippten AP/PA-Rohprojektionen im Preview-Step.",
    )
    parser.add_argument(
        "--debug-ap-pa-save-npy",
        action="store_true",
        help="Speichert zusätzliche .npy-Dumps der AP/PA-Rohwerte und ihrer Differenz (nur mit --debug-ap-pa).",
    )
    parser.add_argument(
        "--debug-ap-pa-every",
        type=int,
        default=50,
        help="Speicher-Intervall (in Steps) für AP/PA-Debug-Assets (PNG/NPY/Hist) bei aktiviertem --debug-ap-pa.",
    )
    parser.add_argument(
        "--debug-ap-pa-assets",
        type=str,
        default="images+hists",
        choices=["none", "images", "images+hists", "all"],
        help="Welche AP/PA-Debug-Assets gespeichert werden (PNG/NPY/Histogramme), wenn --debug-ap-pa aktiv ist.",
    )
    parser.add_argument(
        "--debug-final-no-atten",
        action="store_true",
        help="Render finale AP/PA zusätzlich ohne Attenuation und vergleiche RMSE/STD.",
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


def adjust_projection(arr: np.ndarray, view: str) -> np.ndarray:
    """
    Bringe die Projektionen in dieselbe Orientierung wie data_check.py:
    erst vertikal flippen, dann 90° im Uhrzeigersinn drehen.
    """
    flipped = np.flipud(arr)
    rotated = np.rot90(flipped, k=-1)
    return rotated


def save_img_linear(arr: np.ndarray, path: Path, vmin=None, vmax=None, symmetric=False, cmap="gray", title=None):
    """Save image with linear scaling and optional symmetric range (debug helper)."""
    import matplotlib.pyplot as plt

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if vmin is None:
        vmin = float(np.min(arr))
    if vmax is None:
        vmax = float(np.max(arr))
    if symmetric:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m
    if np.isclose(vmin, vmax):
        img = np.zeros_like(arr)
    else:
        clipped = np.clip(arr, vmin, vmax)
        img = (clipped - vmin) / (vmax - vmin + 1e-8)

    plt.figure(figsize=(6, 4))
    plt.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _pose_summary(pose: torch.Tensor) -> str:
    """Short, human-readable pose summary for AP/PA debug."""
    if pose is None:
        return "pose=None"
    pose_cpu = pose.detach().cpu()
    pos = pose_cpu[:3, 3] if pose_cpu.shape[-1] >= 4 else pose_cpu[:3, -1]
    z_axis = pose_cpu[:3, 2] if pose_cpu.shape[-1] >= 3 else torch.zeros(3)
    return (
        f"t=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
        f"z=({z_axis[0]:.3f},{z_axis[1]:.3f},{z_axis[2]:.3f})"
    )


def _safe_corrcoef_np(a: np.ndarray, b: np.ndarray) -> float:
    """Numerically safe correlation on flattened arrays."""
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c) + 1e-12
    if denom == 0.0:
        return float("nan")
    corr = float(np.clip(np.dot(a_c, b_c) / denom, -1.0, 1.0))
    return corr


def debug_compare_ap_pa(
    pred_ap_raw: torch.Tensor,
    pred_pa_raw: torch.Tensor,
    extras_ap,
    extras_pa,
    generator,
    step: int,
    tag: str,
    debug_save_npy: bool,
    out_dir: Path,
    save_assets: bool,
    asset_mode: str,
):
    """
    AP/PA Debug:
    This comparison decides whether AP/PA similarity is caused by geometry / transmission / view-switch bugs
    or by visualization / normalization artifacts.
    """
    # Vergleich erfolgt auf den Rohprojektionen (ohne Rotate/Flip/Normierung)
    assert pred_ap_raw.shape == pred_pa_raw.shape
    assert pred_ap_raw.data_ptr() != pred_pa_raw.data_ptr()

    debug_dir = out_dir / "ap_pa_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    ap_np_raw = pred_ap_raw.detach().float().cpu().numpy()
    pa_np_raw = pred_pa_raw.detach().float().cpu().numpy()

    ap_flat = pred_ap_raw.detach().float().cpu()
    pa_flat = pred_pa_raw.detach().float().cpu()
    diff_flat = ap_flat - pa_flat
    std_ap = torch.std(ap_flat).item()
    std_pa = torch.std(pa_flat).item()
    std_diff = torch.std(diff_flat).item()
    max_abs = torch.max(torch.abs(diff_flat)).item()
    mean_abs = torch.mean(torch.abs(diff_flat)).item()
    rmse = torch.sqrt(torch.mean(diff_flat * diff_flat)).item()
    ap_abs_max = torch.max(torch.abs(ap_flat)).item()
    corr = _safe_corrcoef_np(ap_flat.numpy(), pa_flat.numpy())
    ap0 = ap_flat - ap_flat.mean()
    pa0 = pa_flat - pa_flat.mean()
    diff0 = ap0 - pa0
    corr_centered = _safe_corrcoef_np(ap0.numpy(), pa0.numpy())
    rmse_centered = torch.sqrt(torch.mean(diff0 * diff0)).item()
    practically_equal = (max_abs < 1e-6) or (max_abs / (ap_abs_max + 1e-8) < 1e-6)
    print(
        f"[debug ap/pa] eq={practically_equal} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rmse={rmse:.3e} corr={corr:.3f} "
        f"| corr_centered={corr_centered:.3f} rmse_centered={rmse_centered:.3e} "
        f"| std_ap={std_ap:.3e} std_pa={std_pa:.3e} std_diff={std_diff:.3e}",
        flush=True,
    )
    if debug_save_npy and save_assets:
        np.save(debug_dir / f"step_{step:05d}_{tag}_ap_raw.npy", ap_np_raw)
        np.save(debug_dir / f"step_{step:05d}_{tag}_pa_raw.npy", pa_np_raw)
        np.save(debug_dir / f"step_{step:05d}_{tag}_ap_pa_diff.npy", ap_np_raw - pa_np_raw)

    if save_assets:
        # Zusätzliche Visualisierungen, um kleine Unterschiede sichtbar zu machen
        shared_vmin = float(min(ap_np_raw.min(), pa_np_raw.min()))
        shared_vmax = float(max(ap_np_raw.max(), pa_np_raw.max()))
        diff_np = ap_np_raw - pa_np_raw
        abs_diff_np = np.abs(diff_np)
        p_signed = float(np.percentile(abs_diff_np, 99.5))
        p_abs = float(np.percentile(abs_diff_np, 99.0))
        if asset_mode in ("images", "images+hists", "all"):
            save_img_linear(
                ap_np_raw,
                debug_dir / f"step_{step:05d}_{tag}_AP_shared.png",
                vmin=shared_vmin,
                vmax=shared_vmax,
                title=f"AP shared scale @ step {step} ({tag})",
            )
            save_img_linear(
                pa_np_raw,
                debug_dir / f"step_{step:05d}_{tag}_PA_shared.png",
                vmin=shared_vmin,
                vmax=shared_vmax,
                title=f"PA shared scale @ step {step} ({tag})",
            )
            save_img_linear(
                diff_np,
                debug_dir / f"step_{step:05d}_{tag}_DIFF_signed.png",
                vmin=-p_signed,
                vmax=+p_signed,
                symmetric=True,
                cmap="seismic",
                title=f"AP-PA signed diff @ step {step} ({tag})",
            )
            save_img_linear(
                abs_diff_np,
                debug_dir / f"step_{step:05d}_{tag}_DIFF_abs.png",
                vmin=0.0,
                vmax=p_abs,
                symmetric=False,
                cmap="hot",
                title=f"AP-PA abs diff @ step {step} ({tag})",
            )
        if asset_mode in ("images+hists", "all"):
            # Histogramme für AP, PA, Diff
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(9, 3))
                plt.subplot(1, 3, 1)
                plt.hist(ap_np_raw.ravel(), bins=50, color="blue", alpha=0.7)
                plt.title("AP")
                plt.subplot(1, 3, 2)
                plt.hist(pa_np_raw.ravel(), bins=50, color="orange", alpha=0.7)
                plt.title("PA")
                plt.subplot(1, 3, 3)
                plt.hist(diff_np.ravel(), bins=50, color="red", alpha=0.7)
                plt.title("AP-PA")
                plt.tight_layout()
                plt.savefig(debug_dir / f"step_{step:05d}_{tag}_hists.png", dpi=150)
                plt.close()
            except Exception as exc:  # pragma: no cover - best effort
                print(f"[debug ap/pa] histogram generation failed: {exc}", flush=True)

    if practically_equal:
        # --- View-Switch / Transmission Debug ---
        rays_ap = extras_ap.get("_rays") if isinstance(extras_ap, dict) else None
        rays_pa = extras_pa.get("_rays") if isinstance(extras_pa, dict) else None

        def _ensure_rays(pose):
            focal_or_size = generator.ortho_size if generator.orthographic else generator.focal
            rays_full, _, _ = generator.val_ray_sampler(generator.H, generator.W, focal_or_size, pose)
            return rays_full

        if rays_ap is None:
            rays_ap = _ensure_rays(generator.pose_ap)
        if rays_pa is None:
            rays_pa = _ensure_rays(generator.pose_pa)

        rays_ap_cpu = rays_ap.detach().cpu()
        rays_pa_cpu = rays_pa.detach().cpu()
        num_rays = rays_ap_cpu.shape[1]
        sample_idx = torch.randperm(num_rays)[: min(3, num_rays)]
        print(f"[debug ap/pa][pose] AP {_pose_summary(generator.pose_ap)} | PA {_pose_summary(generator.pose_pa)}", flush=True)
        for i, idx in enumerate(sample_idx.tolist()):
            o_ap, d_ap = rays_ap_cpu[0, idx], rays_ap_cpu[1, idx]
            o_pa, d_pa = rays_pa_cpu[0, idx], rays_pa_cpu[1, idx]
            print(
                f"[debug ap/pa][rays] #{i} idx={idx} "
                f"AP o=({o_ap[0]:.3f},{o_ap[1]:.3f},{o_ap[2]:.3f}) d=({d_ap[0]:.3f},{d_ap[1]:.3f},{d_ap[2]:.3f}) | "
                f"PA o=({o_pa[0]:.3f},{o_pa[1]:.3f},{o_pa[2]:.3f}) d=({d_pa[0]:.3f},{d_pa[1]:.3f},{d_pa[2]:.3f})",
                flush=True,
            )

        if torch.allclose(rays_ap_cpu, rays_pa_cpu, atol=1e-6, rtol=0):
            print("⚠️ [debug ap/pa] POTENTIAL BUG: AP/PA rays identical.", flush=True)

        def _log_attenuation(label: str, extras):
            if not isinstance(extras, dict):
                print(f"[debug ap/pa][atten] {label}: extras missing.", flush=True)
                return
            mu_vals = extras.get("debug_mu")
            dists_vals = extras.get("debug_dists")
            transmission_vals = extras.get("debug_transmission")
            if mu_vals is None or dists_vals is None:
                print(f"[debug ap/pa][atten] {label}: no mu/dists in extras.", flush=True)
                return
            mu_clamped = torch.clamp(mu_vals, min=0.0)
            mu_min, mu_mean, mu_max = mu_clamped.min().item(), mu_clamped.mean().item(), mu_clamped.max().item()
            mu_dists = mu_clamped * dists_vals
            tau = torch.cumsum(mu_dists, dim=-1)
            tau = F.pad(tau[..., :-1], (1, 0), mode="constant", value=0.0)
            atten = transmission_vals if transmission_vals is not None else torch.exp(-tau)
            tau_min, tau_mean, tau_max = tau.min().item(), tau.mean().item(), tau.max().item()
            att_min, att_mean, att_max = atten.min().item(), atten.mean().item(), atten.max().item()
            print(
                f"[debug ap/pa][atten] {label}: mu(min/mean/max)={mu_min:.3e}/{mu_mean:.3e}/{mu_max:.3e} "
                f"| tau(min/mean/max)={tau_min:.3e}/{tau_mean:.3e}/{tau_max:.3e} "
                f"| atten(min/mean/max)={att_min:.3e}/{att_mean:.3e}/{att_max:.3e}",
                flush=True,
            )

        _log_attenuation("AP", extras_ap)
        _log_attenuation("PA", extras_pa)
    else:
        vmin_ap, vmax_ap = float(ap_np_raw.min()), float(ap_np_raw.max())
        vmin_pa, vmax_pa = float(pa_np_raw.min()), float(pa_np_raw.max())
        shared_vmin = min(vmin_ap, vmin_pa)
        shared_vmax = max(vmax_ap, vmax_pa)
        print(
            f"[debug ap/pa][vmin/vmax] ap=({vmin_ap:.3e},{vmax_ap:.3e}) | pa=({vmin_pa:.3e},{vmax_pa:.3e}) | shared=({shared_vmin:.3e},{shared_vmax:.3e})",
            flush=True,
        )
        ap_np_vis = adjust_projection(ap_np_raw, "ap")
        pa_np_vis = adjust_projection(pa_np_raw, "pa")
        save_img_linear(
            ap_np_vis,
            debug_dir / f"step_{step:05d}_{tag}_AP_shared.png",
            vmin=shared_vmin,
            vmax=shared_vmax,
            title=f"AP shared scale @ step {step} ({tag})",
        )
        save_img_linear(
            pa_np_vis,
            debug_dir / f"step_{step:05d}_{tag}_PA_shared.png",
            vmin=shared_vmin,
            vmax=shared_vmax,
            title=f"PA shared scale @ step {step} ({tag})",
        )
        diff_vis = ap_np_vis - pa_np_vis
        save_img_linear(
            diff_vis,
            debug_dir / f"step_{step:05d}_{tag}_AP_PA_diff.png",
            symmetric=True,
            cmap="seismic",
            title=f"AP-PA diff @ step {step} ({tag})",
        )
        best_transform, best_rmse, best_corr = _find_best_pa_transform(pa_np_raw, ap_np_raw)
        print(
            f"[debug ap/pa][transform] best={best_transform} best_rmse={best_rmse:.3e} best_corr={best_corr:.3f} | outputs highly similar; narrow dynamic range may make previews look identical",
            flush=True,
        )



def _find_best_pa_transform(pa_raw: np.ndarray, ap_raw: np.ndarray):
    """Try 8 simple 2D transforms to align PA to AP and report best RMSE/corr."""
    transforms = [
        ("identity", lambda x: x),
        ("flip_ud", np.flipud),
        ("flip_lr", np.fliplr),
        ("rot90", lambda x: np.rot90(x, 1)),
        ("rot180", lambda x: np.rot90(x, 2)),
        ("rot270", lambda x: np.rot90(x, 3)),
        ("transpose", lambda x: np.transpose(x)),
        ("transpose_flip_lr", lambda x: np.fliplr(np.transpose(x))),
    ]
    best = ("none", float("inf"), float("nan"))
    for name, fn in transforms:
        cand = fn(pa_raw)
        if cand.shape != ap_raw.shape:
            continue
        diff = cand - ap_raw
        rmse = float(np.sqrt(np.mean(diff * diff)))
        corr = _safe_corrcoef_np(cand, ap_raw)
        if rmse < best[1]:
            best = (name, rmse, corr)
    return best


def poisson_nll(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    clamp_max: float = 1e6,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Poisson Negative Log-Likelihood Loss für Emissions- oder Zähl-Daten.
    Erwartet nichtnegative 'pred' und 'target' (z. B. Intensitäten).
    Falls projizierte Zählraten global skaliert werden, müssen pred/target
    konsistent dieselbe Skalierung durchlaufen – der Loss bleibt physikalisch
    äquivalent (nur numerische Reskalierung).
    """
    # Stabilisierung über clamping, damit log() definiert bleibt
    pred = pred.clamp_min(eps).clamp_max(clamp_max)
    nll = pred - target * torch.log(pred)
    if weight is not None:
        nll = nll * weight
    return nll.mean()


def build_ray_split(num_pixels: int, split_ratio: float, device: torch.device):
    """
    Erzeuge einen festen Train/Test-Split über alle Rays einer Ansicht.
    Split ist reproduzierbar, weil der globale Seed (set_seed) bereits gesetzt wurde.
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


def render_minibatch(generator, z_latent, rays_subset, need_raw: bool = False, ct_context=None):
    """Render a mini-batch of rays from a fixed pose while keeping training kwargs."""
    # train/test kwargs werden durch use_test_kwargs umgeschaltet
    render_kwargs = generator.render_kwargs_train if not generator.use_test_kwargs else generator.render_kwargs_test
    render_kwargs = dict(render_kwargs)
    render_kwargs["features"] = z_latent
    if need_raw:
        render_kwargs["retraw"] = True
    if ct_context is not None:
        render_kwargs["ct_context"] = ct_context
    elif render_kwargs.get("use_attenuation"):
        render_kwargs["use_attenuation"] = False
    if DEBUG_PRINTS:
        render_kwargs["debug_prints"] = True
    proj_map, _, _, extras = generator.render(rays=rays_subset, **render_kwargs)
    return proj_map.view(z_latent.shape[0], -1), extras


# Preview renderers:
# - maybe_render_preview: periodic previews during training / preview_only (step-based).
# - Smoke-test preview before training loop (quick AP/PA render for NaN check).
# - Final preview block at end of train(): writes final_AP.png / final_PA.png after the training loop.
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
    debug_ap_pa = bool(getattr(args, "debug_ap_pa", False))
    debug_save_npy = debug_ap_pa and bool(getattr(args, "debug_ap_pa_save_npy", False))
    debug_every = max(1, int(getattr(args, "debug_ap_pa_every", 50)))
    asset_mode = getattr(args, "debug_ap_pa_assets", "images+hists")
    debug_final_no_atten = bool(getattr(args, "debug_final_no_atten", False))
    do_preview = args.preview_every > 0 and (step % args.preview_every) == 0

    # Volle AP/PA-Renderings sind teuer; ohne Debug nur zu den geplanten Intervallen rendern.
    if not (do_preview or debug_ap_pa):
        return
    if debug_ap_pa:
        print(f"[debug ap/pa] ENTER preview path (preview) at step {step}", flush=True)

    prev_flag = generator.use_test_kwargs
    generator.eval()
    generator.use_test_kwargs = True

    ctx = ct_context or generator.build_ct_context(ct_volume)

    debug_override = None
    return_rays = False
    if debug_ap_pa:
        debug_override = {
            "attenuation_debug": True,
            "retraw": True,
        }
        if DEBUG_PRINTS:
            debug_override["debug_prints"] = True
        return_rays = True

    with torch.no_grad():
        proj_ap, _, _, extras_ap = generator.render_from_pose(
            z_eval, generator.pose_ap, ct_context=ctx, debug_override=debug_override, return_rays=return_rays
        )
        proj_pa, _, _, extras_pa = generator.render_from_pose(
            z_eval, generator.pose_pa, ct_context=ctx, debug_override=debug_override, return_rays=return_rays
        )

    # ursprünglichen Modus wiederherstellen
    generator.train()
    generator.use_test_kwargs = prev_flag or False

    H, W = generator.H, generator.W

    pred_ap_raw = proj_ap[0].reshape(H, W)
    pred_pa_raw = proj_pa[0].reshape(H, W)
    ap_np_raw = pred_ap_raw.detach().cpu().numpy()
    pa_np_raw = pred_pa_raw.detach().cpu().numpy()

    out_dir = outdir / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[preview step {step:05d}] "
        f"AP min/max={ap_np_raw.min():.3e}/{ap_np_raw.max():.3e} | "
        f"PA min/max={pa_np_raw.min():.3e}/{pa_np_raw.max():.3e}",
        flush=True,
    )
    if args.debug_ap_pa:
        save_assets = (step % debug_every) == 0
        debug_compare_ap_pa(
            pred_ap_raw,
            pred_pa_raw,
            extras_ap,
            extras_pa,
            generator,
            step,
            tag="preview",
            debug_save_npy=debug_save_npy,
            out_dir=out_dir,
            save_assets=save_assets,
            asset_mode=asset_mode,
        )

    if do_preview:
        # -------- VIS (mit adjust_projection) --------
        ap_np_vis = adjust_projection(ap_np_raw, "ap")
        pa_np_vis = adjust_projection(pa_np_raw, "pa")

        save_img(
            ap_np_vis,
            out_dir / f"step_{step:05d}_AP.png",
            title=f"AP @ step {step}",
        )
        save_img(
            pa_np_vis,
            out_dir / f"step_{step:05d}_PA.png",
            title=f"PA @ step {step}",
        )

        # Depth-Profile wieder mitspeichern (teilt Rendering mit den Previews).
        save_depth_profile(
            step,
            generator,
            z_eval,
            ct_volume,
            act_volume,
            out_dir,
            proj_ap=proj_ap,
            proj_pa=proj_pa,
        )

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
                "tv",
                "tv_mu",
                "mu_gate",
                "tv3d",
                "mask",
                "mae_ap",
                "mae_pa",
                "psnr_ap",
                "psnr_pa",
                "pred_mean_ap",
                "pred_mean_pa",
                "pred_std_ap",
                "pred_std_pa",
                "loss_test",
                "psnr_ap_test",
                "psnr_pa_test",
                "mae_ap_test",
                "mae_pa_test",
                "iter_ms",
                "lr",
            ]
        )


def append_log(path: Path, row):
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def compute_tv3d_stub(*args, device=None, **kwargs):
    """
    Platzhalter für eine zukünftige 3D-TV-Regularisierung über ein Hilfsgitter.
    Aktuell wird kein Volumen evaluiert – der Rückgabewert bleibt 0.
    """
    if device is None:
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device = arg.device
                break
    if device is None and "device" in kwargs and isinstance(kwargs["device"], torch.device):
        device = kwargs["device"]
    device = device or torch.device("cpu")
    return torch.tensor(0.0, device=device)


def save_algorithm_orientation_debug(
    outdir: Path,
    H: int,
    W: int,
    ap: Optional[torch.Tensor],
    pa: Optional[torch.Tensor],
    ct_context: Optional[dict],
    ct_vol: Optional[torch.Tensor] = None,
    spect_att_vol: Optional[torch.Tensor] = None,
    act_vol: Optional[torch.Tensor] = None,
    mask_vol: Optional[torch.Tensor] = None,
    prefix: str = "",
):
    """
    Speichere die Daten exakt im Layout, das der Algorithmus nutzt (keine Orientierungs-Kosmetik).
    - Projektionen: wie sie in ap_flat/pa_flat in die Losses gehen.
    - Volumina: nach denselben Flips/Unsqueezes wie build_ct_context, MIP entlang der CT-Dimension.
    """
    debug_dir = outdir / "debug_algorithm_orientation"
    debug_dir.mkdir(parents=True, exist_ok=True)

    def log_and_save(name: str, arr: np.ndarray):
        print(f"[algo-orient] {name} shape={arr.shape} min/max={arr.min():.3e}/{arr.max():.3e}", flush=True)
        save_img(arr, debug_dir / f"{prefix}{name}.png", title=name)

    def prep_volume(vol: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if vol is None or not torch.is_tensor(vol) or vol.numel() == 0:
            return None
        arr = vol.detach().cpu()
        if arr.dim() == 3:
            arr = arr.unsqueeze(0)
        if arr.dim() == 4:
            arr = arr.unsqueeze(0)
        if arr.dim() != 5:
            return None
        arr = torch.flip(arr, dims=[-1])
        return arr.squeeze(0).squeeze(0).numpy()  # -> (D,H,W)

    # Projektionen im Loss-Layout
    if ap is not None and ap.numel() > 0:
        ap_flat = ap.view(ap.shape[0], -1)
        ap_used = ap_flat[0].detach().cpu().view(H, W).numpy()
        log_and_save("ap_used", ap_used)
    if pa is not None and pa.numel() > 0:
        pa_flat = pa.view(pa.shape[0], -1)
        pa_used = pa_flat[0].detach().cpu().view(H, W).numpy()
        log_and_save("pa_used", pa_used)

    # Volumina nach build_ct_context-Logik (Flip letzter Achse, Unsqueeze zu [1,1,D,H,W])
    vol_ct_used = prep_volume(ct_vol)
    vol_spect_used = prep_volume(spect_att_vol)
    vol_mask_used = None
    if ct_context is not None and isinstance(ct_context, dict):
        ctx_vol = ct_context.get("volume")
        if ctx_vol is not None and ctx_vol.numel() > 0:
            vol_ct_used = ctx_vol.detach().cpu().squeeze(0).squeeze(0).numpy()
        ctx_mask = ct_context.get("mask_volume")
        if ctx_mask is not None and ctx_mask.numel() > 0:
            vol_mask_used = ctx_mask.detach().cpu().squeeze(0).squeeze(0).numpy()
    if vol_mask_used is None:
        vol_mask_used = prep_volume(mask_vol)

    def save_mip(name: str, vol_np: Optional[np.ndarray]):
        if vol_np is None or vol_np.size == 0 or vol_np.ndim != 3:
            return
        mip = vol_np.max(axis=0)  # MIP entlang der Sampling-Dimension (D)
        log_and_save(name, mip)

    save_mip("ct_used_mip", vol_ct_used)
    save_mip("spect_att_used_mip", vol_spect_used)

    # ACT wird unverändert genutzt (keine Flips in build_ct_context) -> MIP über erste Achse
    if act_vol is not None and torch.is_tensor(act_vol) and act_vol.numel() > 0:
        act_arr = act_vol.detach().cpu()
        if act_arr.dim() == 5:
            act_arr = act_arr.squeeze(0).squeeze(0)
        elif act_arr.dim() == 4:
            act_arr = act_arr.squeeze(0)
        if act_arr.dim() == 3:
            act_np = act_arr.numpy()
            act_mip = act_np.max(axis=0)
            log_and_save("act_used_mip", act_mip)

    if vol_mask_used is not None:
        save_mip("mask_used_mip", vol_mask_used)
    else:
        print("[algo-orient] mask_used_mip skipped (no mask_volume present)", flush=True)


def save_orientation_debug_sample(dataset, outdir: Path):
    """
    Speichere einmalig alle Eingaben (AP, PA, CT, ACT, Masken, Attenuation) in derselben
    Anzeige-Orientierung wie die Previews, inklusive zusätzlichem flipud für Volumina.
    """

    def orient_projection(tensor: torch.Tensor, view: str) -> Optional[np.ndarray]:
        if tensor is None or tensor.numel() == 0:
            return None
        arr = tensor.squeeze().detach().cpu().numpy()
        return adjust_projection(arr, view=view)

    def orient_volume(vol: torch.Tensor, name: str = "") -> Optional[np.ndarray]:
        if vol is None or vol.numel() == 0:
            return None
        arr = vol.detach().cpu().numpy()
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3:
            return None
        mip = arr.max(axis=0)  # (LR, SI) – gleiche Achse wie orientation_preview (Dataset liefert bereits AP,LR,SI)
        img = adjust_projection(mip, view="ap")  # gleiche Orientierung wie AP-Projektion
        img = np.flipud(img)  # zusätzlicher Flip wie in orientation_preview validiert
        if name == "act":
            img = np.fliplr(img)  # ACT braucht zusätzlich flip LR für korrekte Anzeige
        return img

    try:
        sample = dataset[0]
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Orientierungsvorschau übersprungen (Dataset-Access): {exc}", flush=True)
        return

    # Falls die Maske nicht aus dem Dataset geladen wurde, versuche sie direkt neben ap/ct zu laden (nur Debug/Preview).
    if (sample.get("mask") is None or sample.get("mask").numel() == 0) and hasattr(dataset, "entries"):
        try:
            entry0 = dataset.entries[0]
            base = Path(entry0.get("ct_path") or entry0.get("ap_path", "")).resolve().parent
            mask_path = base / "mask.npy"
            if mask_path.exists():
                mask_np = np.load(mask_path).astype(np.float32)
                # im Loader wird (1,0,2) transponiert → (AP, LR, SI)
                mask_np = np.transpose(mask_np, (1, 0, 2))
                sample["mask"] = torch.from_numpy(mask_np)
        except Exception:
            pass

    orient_dir = outdir / "preview" / "orientation_debug"
    orient_dir.mkdir(parents=True, exist_ok=True)

    items = {
        "ap": orient_projection(sample.get("ap"), view="ap"),
        "pa": orient_projection(sample.get("pa"), view="pa"),
        "ct_att": orient_volume(sample.get("ct"), name="ct"),
        "spect_att": orient_volume(sample.get("spect_att"), name="spect_att"),
        "act": orient_volume(sample.get("act"), name="act"),
        "mask": orient_volume(sample.get("mask"), name="mask"),
    }
    for name, img in items.items():
        if img is None:
            continue
        # Falls Bild im (W,H)-Layout kommt → nach (H,W) bringen
        if img.ndim == 2 and img.shape[0] > img.shape[1]:
            img = img.T
        print(
            f"[orientation] {name} min/max={np.min(img):.3e}/{np.max(img):.3e} shape={img.shape}",
            flush=True,
        )
        save_img(img, orient_dir / f"{name}.png", title=name)
    print(f"[orientation] Debug-PNGs gespeichert unter {orient_dir}", flush=True)


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


def dump_debug_tensor(outpath: Path, tensor: torch.Tensor):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), outpath)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse + 1e-12)


def evaluate_split(
    generator,
    z_latent,
    rays_cache,
    ray_indices,
    split: str,
    ap_flat_proc: torch.Tensor,
    pa_flat_proc: torch.Tensor,
    rays_per_proj_eval: int,
    bg_weight: float,
    weight_threshold: float,
    ct_context=None,
):
    """Evaluiert Poisson-NLL/PSNR auf einem festen Split (train/test) ohne Gradienten."""
    idx_ap_all = ray_indices["ap"].get(split)
    idx_pa_all = ray_indices["pa"].get(split)
    if idx_ap_all is None or idx_pa_all is None or idx_ap_all.numel() == 0 or idx_pa_all.numel() == 0:
        return None

    n_ap = min(idx_ap_all.numel(), rays_per_proj_eval)
    n_pa = min(idx_pa_all.numel(), rays_per_proj_eval)
    idx_ap = sample_split_indices(idx_ap_all, n_ap)
    idx_pa = sample_split_indices(idx_pa_all, n_pa)

    prev_flag = generator.use_test_kwargs
    # Eval-Mode aktivieren (kein Grad, festes Test-Split)
    generator.eval()
    generator.use_test_kwargs = True
    with torch.no_grad():
        ray_batch_ap = slice_rays(rays_cache["ap"], idx_ap)
        ray_batch_pa = slice_rays(rays_cache["pa"], idx_pa)

        pred_ap, _ = render_minibatch(generator, z_latent, ray_batch_ap, need_raw=False, ct_context=ct_context)
        pred_pa, _ = render_minibatch(generator, z_latent, ray_batch_pa, need_raw=False, ct_context=ct_context)

        target_ap = ap_flat_proc[0, idx_ap].unsqueeze(0)
        target_pa = pa_flat_proc[0, idx_pa].unsqueeze(0)

        pred_ap = pred_ap.clamp_min(1e-8)
        pred_pa = pred_pa.clamp_min(1e-8)

        weight_ap = build_loss_weights(target_ap, bg_weight, weight_threshold)
        weight_pa = build_loss_weights(target_pa, bg_weight, weight_threshold)
        loss_ap = poisson_nll(pred_ap, target_ap, weight=weight_ap)
        loss_pa = poisson_nll(pred_pa, target_pa, weight=weight_pa)
        loss_total = loss_ap + loss_pa

        psnr_ap = compute_psnr(pred_ap, target_ap)
        psnr_pa = compute_psnr(pred_pa, target_pa)
        mae_ap = torch.mean(torch.abs(pred_ap - target_ap)).item()
        mae_pa = torch.mean(torch.abs(pred_pa - target_pa)).item()

    # Ursprünglichen Modus wiederherstellen (Train/Eval + use_test_kwargs)
    if prev_flag:
        generator.eval()
    else:
        generator.train()

    return {
        "loss": loss_total.item(),
        "loss_ap": loss_ap.item(),
        "loss_pa": loss_pa.item(),
        "psnr_ap": psnr_ap,
        "psnr_pa": psnr_pa,
        "mae_ap": mae_ap,
        "mae_pa": mae_pa,
    }


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


def save_depth_profile(step, generator, z_latent, ct_vol, act_vol, outdir: Path, proj_ap=None, proj_pa=None):
    # Nur aktiv, wenn Ground Truth CT oder act existieren
    if ct_vol is None and act_vol is None:
        return

    def extract_curve(vol, y_idx: int, x_idx: int):
        if vol is None:
            return None, None
        data = vol.squeeze(0).detach().cpu().numpy() if vol.dim() == 4 else vol.detach().cpu().numpy()
        if data.ndim != 3:
            return None, None
        D, H, W = data.shape
        # Falls H/W vertauscht sind (z. B. 651x256 statt 256x651), tauschen.
        if H == generator.W and W == generator.H:
            data = np.transpose(data, (0, 2, 1))
            D, H, W = data.shape
        y_idx = int(np.clip(y_idx, 0, H - 1))
        x_idx = int(np.clip(x_idx, 0, W - 1))
        return data[:, y_idx, x_idx], (D, H, W)

    def to_np_image(tensor):
        if tensor is None:
            return None
        return tensor.detach().cpu().reshape(generator.H, generator.W).numpy()

    ap_img = to_np_image(proj_ap)
    pa_img = to_np_image(proj_pa)

    act_data = None
    act_masks = None
    if act_vol is not None:
        act_data = act_vol.squeeze(0).detach().cpu().numpy() if act_vol.dim() == 4 else act_vol.detach().cpu().numpy()
        if act_data.ndim != 3:
            act_data = None
        else:
            if act_data.shape[1] == generator.W and act_data.shape[2] == generator.H:
                act_data = np.transpose(act_data, (0, 2, 1))
            depth_max = act_data.max(axis=0)
            if depth_max.shape == (generator.W, generator.H):
                depth_max = depth_max.T
            elif depth_max.shape != (generator.H, generator.W):
                depth_max = None
            if depth_max is not None:
                act_masks = (depth_max <= 1e-8, depth_max > 1e-8)

    ct_data = None
    ct_depth_max = None
    if ct_vol is not None:
        ct_data = ct_vol.squeeze(0).detach().cpu().numpy() if ct_vol.dim() == 4 else ct_vol.detach().cpu().numpy()
        if ct_data.ndim == 3:
            if ct_data.shape[1] == generator.W and ct_data.shape[2] == generator.H:
                ct_data = np.transpose(ct_data, (0, 2, 1))
            ct_depth_max = ct_data.max(axis=0)
            if ct_depth_max.shape == (generator.W, generator.H):
                ct_depth_max = ct_depth_max.T
            elif ct_depth_max.shape != (generator.H, generator.W):
                ct_depth_max = None


    def pick_ray_indices(num_zero: int = 1, num_active: int = 3):
        """Wählt Strahlen für das Depth-Profil: 1 Hintergrundstrahl + 3 aktive."""
        H, W = generator.H, generator.W
        chosen = []
        target_total = max(num_zero + num_active, 1)
        min_dist = max(1, int(0.05 * min(H, W)))  # verhindert, dass Strahlen direkt benachbart sind

        def is_far_enough(idx):
            if idx is None or not chosen:
                return True
            y, x = idx
            for cy, cx in chosen:
                if np.hypot(y - cy, x - cx) < min_dist:
                    return False
            return True

        def add_unique(idx):
            if idx is None:
                return False
            if idx not in chosen and is_far_enough(idx):
                chosen.append(idx)
                return True
            return False

        def pick_from_mask(mask, prefer_high=True):
            if mask is None:
                return None
            mask = mask.astype(bool).copy()
            if mask.shape != (H, W) or not mask.any():
                return None
            for y, x in chosen:
                if 0 <= y < H and 0 <= x < W:
                    mask[y, x] = False
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
        if ct_depth_max is not None:
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

        # (1) Gezielt Null- und Aktiv-Strahlen auswählen
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

        # (2) Fallback über Projektionen (Minimum für 0-Strahl, Maximum für Aktivität)
        if zero_needed > 0:
            if add_unique(pick_proj_extreme(np.nanargmin)):
                zero_needed -= 1

        while active_needed > 0:
            idx = pick_proj_extreme(np.nanargmax)
            if idx is None:
                break
            if add_unique(idx):
                active_needed -= 1

        # (3) Rest mit festen/relativen Koordinaten auffüllen
        fixed_coords = [(72, 428), (69, 336)]
        for y_raw, x_raw in fixed_coords:
            if len(chosen) >= target_total:
                break
            y = int(np.clip(y_raw, 0, H - 1))
            x = int(np.clip(x_raw, 0, W - 1))
            add_unique((y, x))
        rel_coords = [(0.5, 0.5), (0.25, 0.75), (0.75, 0.25)]
        for ry, rx in rel_coords:
            if len(chosen) >= target_total:
                break
            y = int(np.clip(round((H - 1) * ry), 0, H - 1))
            x = int(np.clip(round((W - 1) * rx), 0, W - 1))
            add_unique((y, x))

        if ap_img is not None and pa_img is not None and len(chosen) < target_total:
            max_y, max_x = np.unravel_index(np.argmax(ap_img + pa_img), ap_img.shape)
            add_unique((int(max_y), int(max_x)))

        while len(chosen) < target_total:
            y = int(np.random.randint(0, H))
            x = int(np.random.randint(0, W))
            add_unique((y, x))

        return chosen[:target_total]

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

    depth_idx = torch.arange(D, device=generator.device)
    z_coords = idx_to_coord(depth_idx, D, radius)
    depth_axis = np.linspace(0.0, 1.0, D)
    import matplotlib.pyplot as plt

    # Ray-Auswahl stabil über den gesamten Lauf halten, indem wir sie beim ersten Aufruf cachen.
    ray_indices = getattr(generator, "_depth_profile_indices", None)
    if not ray_indices:
        ray_indices = pick_ray_indices(num_zero=1, num_active=3)
        generator._depth_profile_indices = ray_indices

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
        coords = torch.stack(
            (x_coord.repeat(D), y_coord.repeat(D), z_coords),
            dim=1,
        )
        # Vorhersage entlang der Tiefe an genau diesem Pixel extrahieren
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


def log_attenuation_profile(step: int, view: str, extras: dict):
    """Druckt λ/μ/T über der Strahltiefe für Debugging-Zwecke."""
    if extras is None:
        return
    lambda_vals = extras.get("debug_lambda")
    if lambda_vals is None:
        return
    ray_idx = 0
    to_np = lambda tensor: tensor[ray_idx].detach().cpu().numpy()
    lam = to_np(lambda_vals)
    mu_vals = extras.get("debug_mu")
    mu = to_np(mu_vals) if mu_vals is not None else None
    trans_vals = extras.get("debug_transmission")
    trans = to_np(trans_vals) if trans_vals is not None else None
    dists = extras.get("debug_dists")
    d = to_np(dists) if dists is not None else None
    weights = extras.get("debug_weights")
    contrib = to_np(weights) if weights is not None else None
    intensity = float(np.sum(contrib)) if contrib is not None else float(np.sum(lam))

    def fmt(arr):
        if arr is None:
            return "n/a"
        return np.array2string(arr, precision=4, separator=", ")

    print(
        f"[attenuation-debug][{view}][step {step:05d}] I={intensity:.4e} | "
        f"λ={fmt(lam)} | μ={fmt(mu)} | Δ={fmt(d)} | T={fmt(trans)} | λ·T·Δ={fmt(contrib)}",
        flush=True,
    )


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

    data_cfg = config.setdefault("data", {})
    if args.normalize_targets:
        print("⚠️ --normalize-targets ist veraltet – Projektionen werden bereits im Loader auf [0,1] normiert.", flush=True)
    if args.projection_normalization is not None:
        data_cfg["projection_normalization"] = args.projection_normalization
    proj_mode = data_cfg.setdefault("projection_normalization", "none").lower()
    if proj_mode != "none":
        print("⚠️ projection_normalization != 'none' wird ignoriert – Loader normiert jedes Bild einzeln.", flush=True)
        data_cfg["projection_normalization"] = "none"
    data_cfg.setdefault("act_scale", 1.0)
    data_cfg.setdefault("ray_split_ratio", 0.8)
    training_cfg = config.setdefault("training", {})
    training_cfg.setdefault("val_interval", 0)
    training_cfg.setdefault("tv_weight", 0.001)
    training_cfg["tv_weight"] = args.tv_weight
    training_cfg.setdefault("mask_loss_weight", 0.0)
    training_cfg.setdefault("use_organ_mask", False)
    training_cfg["mask_loss_weight"] = args.mask_loss_weight
    training_cfg["use_organ_mask"] = bool(args.use_organ_mask or training_cfg.get("use_organ_mask", False))
    data_cfg.setdefault("use_organ_mask", False)
    data_cfg["use_organ_mask"] = bool(training_cfg["use_organ_mask"])
    training_cfg.setdefault("tv_weight_mu", 0.0)
    training_cfg.setdefault("tv_mu_sigma", 1.0)
    training_cfg["tv_weight_mu"] = args.tv_weight_mu
    training_cfg["tv_mu_sigma"] = args.tv_mu_sigma
    training_cfg.setdefault("mu_gate_weight", 0.0)
    training_cfg.setdefault("mu_gate_mode", "none")
    training_cfg.setdefault("mu_gate_center", 0.2)
    training_cfg.setdefault("mu_gate_width", 0.1)
    training_cfg["mu_gate_weight"] = args.mu_gate_weight
    training_cfg["mu_gate_mode"] = args.mu_gate_mode
    training_cfg["mu_gate_center"] = args.mu_gate_center
    training_cfg["mu_gate_width"] = args.mu_gate_width
    training_cfg.setdefault("tv3d_weight", 0.0)
    training_cfg.setdefault("tv3d_grid_size", 32)
    training_cfg["tv3d_weight"] = args.tv3d_weight
    training_cfg["tv3d_grid_size"] = args.tv3d_grid_size
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

    print(f"📂 CWD: {Path.cwd().resolve()}", flush=True)
    outdir = Path(config.get("training", {}).get("outdir", "./results_spect")).expanduser().resolve()
    (outdir / "preview").mkdir(parents=True, exist_ok=True)
    print(f"🗂️ Output-Ordner: {outdir}", flush=True)
    ckpt_dir = outdir / "checkpoints"
    log_path = outdir / "train_log.csv"
    init_log_file(log_path)
    debug_dir = outdir / "debug_dump"

    dataset, hwfr, _ = get_data(config)
    config["data"]["hwfr"] = hwfr
    save_orientation_debug_sample(dataset, outdir)

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
    val_interval = int(training_cfg.get("val_interval", 0) or 0)
    tv_weight = float(training_cfg.get("tv_weight", 0.0))
    tv_weight_mu = float(training_cfg.get("tv_weight_mu", 0.0))
    tv_mu_sigma = float(training_cfg.get("tv_mu_sigma", 1.0))
    mu_gate_weight = float(training_cfg.get("mu_gate_weight", 0.0))
    mu_gate_mode = str(training_cfg.get("mu_gate_mode", "none")).lower()
    mu_gate_center = float(training_cfg.get("mu_gate_center", 0.2))
    mu_gate_width = float(training_cfg.get("mu_gate_width", 0.1))
    tv3d_weight = float(training_cfg.get("tv3d_weight", 0.0))
    tv3d_grid_size = int(training_cfg.get("tv3d_grid_size", 32))
    mask_loss_weight = float(training_cfg.get("mask_loss_weight", 0.0))
    use_organ_mask = bool(training_cfg.get("use_organ_mask", False))

    generator = build_models(config)
    generator.to(device)
    generator.train()
    generator.use_test_kwargs = False  # enforce training kwargs
    for kwargs_render in (generator.render_kwargs_train, generator.render_kwargs_test):
        kwargs_render["tv_mu_sigma"] = tv_mu_sigma
        kwargs_render["mu_gate_mode"] = mu_gate_mode
        kwargs_render["mu_gate_center"] = mu_gate_center
        kwargs_render["mu_gate_width"] = mu_gate_width
        kwargs_render["use_organ_mask"] = use_organ_mask
    if args.debug_attenuation_ray:
        generator.render_kwargs_train["attenuation_debug"] = True
        generator.render_kwargs_test["attenuation_debug"] = True

    # always provide AP/PA fallback poses if not already configured
    generator.set_fixed_ap_pa(radius=hwfr[3])

    z_dim = config["z_dist"]["dim"]
    z_train = torch.nn.Parameter(torch.zeros(1, z_dim, device=device))
    torch.nn.init.normal_(z_train, mean=0.0, std=1.0)

    # Optional: Nur einen schnellen Preview rendern und beenden (kein Training).
    if args.preview_only:
        sample = dataset[0]
        ct_vol = sample.get("ct")
        act_vol = sample.get("act")
        ct_att_vol = sample.get("ct_att")
        spect_att_vol = sample.get("spect_att")
        mask_vol_raw = sample.get("mask")
        mask_vol = mask_vol_raw if use_organ_mask else None
        ct_vol = ct_vol.to(device, non_blocking=True).float() if ct_vol is not None and ct_vol.numel() > 0 else None
        act_vol = act_vol.to(device, non_blocking=True).float() if act_vol is not None and act_vol.numel() > 0 else None
        ct_att_vol = ct_att_vol.to(device, non_blocking=True).float() if ct_att_vol is not None and ct_att_vol.numel() > 0 else None
        spect_att_vol = (
            spect_att_vol.to(device, non_blocking=True).float()
            if spect_att_vol is not None and spect_att_vol.numel() > 0
            else None
        )
        mask_vol = mask_vol.to(device, non_blocking=True).float() if mask_vol is not None and mask_vol.numel() > 0 else None
        if bool(config["data"].get("act_debug_marker", False)) and act_vol is not None and act_vol.numel() > 0:
            act_min = act_vol.min().item()
            act_max = act_vol.max().item()
            print(f"[ACT DEBUG][preview_before_ct_context] min/max/shape {act_min:.3e}/{act_max:.3e} {tuple(act_vol.shape)}", flush=True)
        ct_context = generator.build_ct_context(
            ct_vol if ct_vol is not None else spect_att_vol or ct_att_vol,
            mask_volume=mask_vol,
            att_volume=ct_att_vol,
            spect_att_volume=spect_att_vol,
        )
        if bool(config["data"].get("act_debug_marker", False)) and ct_context is not None:
            ctx_vol = ct_context.get("volume")
            if ctx_vol is not None and ctx_vol.numel() > 0:
                ctx_min = ctx_vol.min().item()
                ctx_max = ctx_vol.max().item()
                print(f"[ACT DEBUG][preview_ct_context_volume] min/max/shape {ctx_min:.3e}/{ctx_max:.3e} {tuple(ctx_vol.shape)}", flush=True)
        save_algorithm_orientation_debug(
            outdir,
            generator.H,
            generator.W,
            sample.get("ap"),
            sample.get("pa"),
            ct_context,
            ct_vol=ct_vol,
            spect_att_vol=spect_att_vol,
            act_vol=act_vol,
            mask_vol=mask_vol_raw,
            prefix="step0_",
        )
        maybe_render_preview(
            0,
            args,
            generator,
            z_train.detach(),
            outdir,
            ct_vol,
            act_vol,
            ct_context,
        )
        print("✅ Preview-only Modus: AP/PA-Previews für Step 0 gespeichert, Training übersprungen.", flush=True)
        return

    # --- Sofortiger Smoke-Test ---
    # Preview-Pfad: schneller AP/PA-Render vor dem eigentlichen Training (Setup/NaN-Check)
    debug_ap_pa = bool(getattr(args, "debug_ap_pa", False))
    debug_save_npy = debug_ap_pa and bool(getattr(args, "debug_ap_pa_save_npy", False))
    debug_every = max(1, int(getattr(args, "debug_ap_pa_every", 50)))
    asset_mode = getattr(args, "debug_ap_pa_assets", "images+hists")
    if debug_ap_pa:
        print(f"[debug ap/pa] ENTER preview path (smoke) at step 0", flush=True)
    with torch.no_grad():
        generator.eval()
        generator.use_test_kwargs = True
        z_smoke = z_train.detach()
        debug_override = None
        return_rays = False
        if debug_ap_pa:
            debug_override = {
                "attenuation_debug": True,
                "retraw": True,
            }
            if DEBUG_PRINTS:
                debug_override["debug_prints"] = True
            return_rays = True
        proj_ap, _, _, extras_ap = generator.render_from_pose(
            z_smoke, generator.pose_ap, debug_override=debug_override, return_rays=return_rays
        )
        proj_pa, _, _, extras_pa = generator.render_from_pose(
            z_smoke, generator.pose_pa, debug_override=debug_override, return_rays=return_rays
        )
        generator.train()
        generator.use_test_kwargs = False

    H, W = generator.H, generator.W
    pred_ap_raw = proj_ap[0].reshape(H, W)
    pred_pa_raw = proj_pa[0].reshape(H, W)
    ap_np_raw = pred_ap_raw.detach().cpu().numpy()
    pa_np_raw = pred_pa_raw.detach().cpu().numpy()
    smoke_dir = outdir / "preview"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    if debug_ap_pa:
        debug_compare_ap_pa(
            pred_ap_raw,
            pred_pa_raw,
            extras_ap,
            extras_pa,
            generator,
            tag="smoke",
            debug_save_npy=debug_save_npy,
            out_dir=smoke_dir,
            step=0,
            save_assets=False,
            asset_mode=asset_mode,
        )
    ap_np = adjust_projection(ap_np_raw, "ap")
    pa_np = adjust_projection(pa_np_raw, "pa")
    save_img(ap_np, smoke_dir / "smoke_AP.png", title="Smoke AP")
    save_img(pa_np, smoke_dir / "smoke_PA.png", title="Smoke PA")
    print("✅ Smoke-Test gespeichert:", flush=True)

    rays_cache = {
        "ap": build_pose_rays(generator, generator.pose_ap),
        "pa": build_pose_rays(generator, generator.pose_pa),
    }
    # Gesamtzahl der Pixel bestimmt die Maximalzahl möglicher Strahlen
    num_pixels = generator.H * generator.W
    # Fester Train/Test-Split pro View (reproduzierbar via globalem Seed).
    ray_indices = {
        "ap": build_ray_split(num_pixels, ray_split_ratio, device),
        "pa": build_ray_split(num_pixels, ray_split_ratio, device),
    }
    print(
        f"🔀 Ray-Split (AP/PA): train={ray_indices['ap']['train'].numel()} / test={ray_indices['ap']['test'].numel()} "
        f"(ratio={ray_split_ratio})",
        flush=True,
    )
    if DEBUG_PRINTS:
        print(
            f"[DEBUG] AP train rays: {len(ray_indices['ap']['train'])} | AP test rays: {len(ray_indices['ap']['test'])} | "
            f"PA train rays: {len(ray_indices['pa']['train'])} | PA test rays: {len(ray_indices['pa']['test'])}",
            flush=True,
        )

    rays_per_proj = args.rays_per_step or config["training"]["chunk"]
    if rays_per_proj <= 0:
        raise ValueError("rays-per-step must be > 0.")
    rays_per_proj = min(rays_per_proj, num_pixels)

    optimizer = torch.optim.Adam(
        list(generator.parameters()) + [z_train],
        lr=config["training"]["lr_g"],
    )
    # Poisson-basierter Loss
    loss_fn = poisson_nll

    amp_enabled = bool(config["training"].get("use_amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    data_iter = iter(dataloader)
    ct_context = None
    orientation_dump_done = False
    final_debug_done = False

    print(
        f"🚀 Starting emission-NeRF training | steps={args.max_steps} | rays/proj={rays_per_proj} "
        f"| image={generator.H}x{generator.W} | chunk={generator.chunk}"
    )

    for step in range(1, args.max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        ap = batch["ap"].to(device, non_blocking=True).float()
        pa = batch["pa"].to(device, non_blocking=True).float()
        act_vol = batch.get("act")
        if act_vol is not None:
            if act_vol.numel() == 0:
                act_vol = None
            else:
                act_vol = act_vol.to(device, non_blocking=True)
        ct_att_vol = batch.get("ct_att")
        if ct_att_vol is not None:
            if ct_att_vol.numel() == 0:
                ct_att_vol = None
            else:
                ct_att_vol = ct_att_vol.to(device, non_blocking=True).float()
        spect_att_vol = batch.get("spect_att")
        if spect_att_vol is not None:
            if spect_att_vol.numel() == 0:
                spect_att_vol = None
            else:
                spect_att_vol = spect_att_vol.to(device, non_blocking=True).float()
        mask_vol_raw = batch.get("mask")
        mask_vol = mask_vol_raw
        if mask_vol is not None:
            if mask_vol.numel() == 0:
                mask_vol = None
            else:
                mask_vol = mask_vol.to(device, non_blocking=True).float()
        if not use_organ_mask:
            mask_vol = None
        ct_vol = batch.get("ct")
        if ct_vol is not None:
            if ct_vol.numel() == 0:
                ct_vol = None
            else:
                ct_vol = ct_vol.to(device, non_blocking=True).float()
        # bevorzugt ct_att als CT-Basis nutzen, falls vorhanden
        if ct_att_vol is not None:
            ct_vol = ct_att_vol

        if bool(config["data"].get("act_debug_marker", False)) and act_vol is not None and act_vol.numel() > 0:
            act_min = act_vol.min().item()
            act_max = act_vol.max().item()
            print(f"[ACT DEBUG][before_ct_context] min/max/shape {act_min:.3e}/{act_max:.3e} {tuple(act_vol.shape)}", flush=True)

        ct_context = generator.build_ct_context(
            ct_vol if ct_vol is not None else spect_att_vol or ct_att_vol,
            mask_volume=mask_vol,
            att_volume=ct_att_vol,
            spect_att_volume=spect_att_vol,
        )
        if bool(config["data"].get("act_debug_marker", False)) and ct_context is not None:
            ctx_vol = ct_context.get("volume")
            if ctx_vol is not None and ctx_vol.numel() > 0:
                ctx_min = ctx_vol.min().item()
                ctx_max = ctx_vol.max().item()
                print(f"[ACT DEBUG][ct_context_volume] min/max/shape {ctx_min:.3e}/{ctx_max:.3e} {tuple(ctx_vol.shape)}", flush=True)
        if not orientation_dump_done:
            save_algorithm_orientation_debug(
                outdir,
                generator.H,
                generator.W,
                ap,
                pa,
                ct_context,
                ct_vol=ct_vol,
                spect_att_vol=spect_att_vol,
                act_vol=act_vol,
                mask_vol=mask_vol_raw,
            )
            if bool(config["data"].get("act_debug_marker", False)) and act_vol is not None and act_vol.numel() > 0:
                if bool(config["data"].get("act_flip_lr", False)) or bool(config["data"].get("act_flip_si", False)):
                    assert act_vol.max().item() > 100, "ACT marker missing → wrong ACT tensor used"
            orientation_dump_done = True

        ap_flat = ap.view(batch_size, -1)
        pa_flat = pa.view(batch_size, -1)
        ap_flat_proc = ap_flat
        pa_flat_proc = pa_flat

        z_latent = z_train

        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()

        idx_ap = sample_split_indices(ray_indices["ap"]["train"], rays_per_proj)
        idx_pa = sample_split_indices(ray_indices["pa"]["train"], rays_per_proj)

        ray_batch_ap = slice_rays(rays_cache["ap"], idx_ap)
        ray_batch_pa = slice_rays(rays_cache["pa"], idx_pa)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred_ap, extras_ap = render_minibatch(
                generator, z_latent, ray_batch_ap, need_raw=args.debug_zero_var, ct_context=ct_context
            )
            pred_pa, extras_pa = render_minibatch(
                generator, z_latent, ray_batch_pa, need_raw=args.debug_zero_var, ct_context=ct_context
            )

            target_ap = ap_flat_proc[0, idx_ap].unsqueeze(0)
            target_pa = pa_flat_proc[0, idx_pa].unsqueeze(0)

            # Poisson-NLL erwartet pred >= 0
            pred_ap_raw = pred_ap.clamp_min(1e-8)
            pred_pa_raw = pred_pa.clamp_min(1e-8)

            pred_ap = pred_ap_raw
            pred_pa = pred_pa_raw

            weight_ap = build_loss_weights(target_ap, args.bg_weight, args.weight_threshold)
            weight_pa = build_loss_weights(target_pa, args.bg_weight, args.weight_threshold)

            loss_ap = poisson_nll(pred_ap, target_ap, weight=weight_ap)
            loss_pa = poisson_nll(pred_pa, target_pa, weight=weight_pa)
            loss = loss_ap + loss_pa
            if DEBUG_PRINTS and (step % 50 == 0):
                print(
                    f"[DEBUG][step {step}] TARGET AP min/max: {target_ap.min().item():.3e}/{target_ap.max().item():.3e} | "
                    f"PRED AP min/max: {pred_ap.min().item():.3e}/{pred_ap.max().item():.3e} | "
                    f"TARGET PA min/max: {target_pa.min().item():.3e}/{target_pa.max().item():.3e} | "
                    f"PRED PA min/max: {pred_pa.min().item():.3e}/{pred_pa.max().item():.3e}",
                    flush=True,
                )
            if args.debug_attenuation_ray:
                log_attenuation_profile(step, "AP", extras_ap)
                log_attenuation_profile(step, "PA", extras_pa)

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
                ct_pairs = sample_ct_pairs(ct_vol, args.ct_samples, args.ct_threshold, radius=radius)
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
            tv_mu_loss = torch.tensor(0.0, device=device)
            mu_gate_loss = torch.tensor(0.0, device=device)
            loss_tv = torch.tensor(0.0, device=device)
            loss_tv_mu = torch.tensor(0.0, device=device)
            loss_mu_gate = torch.tensor(0.0, device=device)
            mask_loss = torch.tensor(0.0, device=device)
            loss_mask = torch.tensor(0.0, device=device)

            tv_base_terms = []
            tv_mu_terms = []
            mu_gate_terms = []
            if isinstance(extras_ap, dict):
                base_val = extras_ap.get("tv_base_loss") or extras_ap.get("tv_loss")
                if base_val is not None:
                    tv_base_terms.append(base_val)
                if extras_ap.get("tv_mu_loss") is not None:
                    tv_mu_terms.append(extras_ap["tv_mu_loss"])
                if extras_ap.get("mu_gate_loss") is not None:
                    mu_gate_terms.append(extras_ap["mu_gate_loss"])
            if isinstance(extras_pa, dict):
                base_val = extras_pa.get("tv_base_loss") or extras_pa.get("tv_loss")
                if base_val is not None:
                    tv_base_terms.append(base_val)
                if extras_pa.get("tv_mu_loss") is not None:
                    tv_mu_terms.append(extras_pa["tv_mu_loss"])
                if extras_pa.get("mu_gate_loss") is not None:
                    mu_gate_terms.append(extras_pa["mu_gate_loss"])

            if tv_base_terms:
                tv_base_loss = torch.stack(tv_base_terms).mean()
            if tv_mu_terms:
                tv_mu_loss = torch.stack(tv_mu_terms).mean()
            if mu_gate_terms:
                mu_gate_loss = torch.stack(mu_gate_terms).mean()
            if isinstance(extras_ap, dict) and extras_ap.get("mask_loss") is not None:
                mask_loss = extras_ap["mask_loss"]
            if isinstance(extras_pa, dict) and extras_pa.get("mask_loss") is not None:
                # Mittelwert über AP/PA, falls beide vorhanden
                if mask_loss.numel() == 0 or mask_loss.item() == 0.0:
                    mask_loss = extras_pa["mask_loss"]
                else:
                    mask_loss = torch.stack([mask_loss, extras_pa["mask_loss"]]).mean()

            if tv_weight != 0.0:
                loss_tv = tv_weight * tv_base_loss
                loss = loss + loss_tv
            if tv_weight_mu != 0.0:
                loss_tv_mu = tv_weight_mu * tv_mu_loss
                loss = loss + loss_tv_mu
            if mu_gate_weight != 0.0:
                loss_mu_gate = mu_gate_weight * mu_gate_loss
                loss = loss + loss_mu_gate
            if mask_loss_weight != 0.0 and use_organ_mask:
                # penalizes emission outside organ mask
                loss_mask = mask_loss_weight * mask_loss
                loss = loss + loss_mask

            loss_tv3d = torch.tensor(0.0, device=device)
            if tv3d_weight > 0.0:
                tv3d_loss_unweighted = compute_tv3d_stub(generator, z_latent, grid_size=tv3d_grid_size, device=device)
                loss_tv3d = tv3d_weight * tv3d_loss_unweighted
                loss = loss + loss_tv3d

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
            val_stats = None
            if val_interval > 0 and (step % val_interval) == 0 and (not args.no_val):
                # Test-Split evaluiert strikt auf den fixen Test-Rays, Gradienten bleiben aus.
                rays_eval = rays_per_proj
                val_stats = evaluate_split(
                    generator,
                    z_latent.detach(),
                    rays_cache,
                    ray_indices,
                    split="test",
                    ap_flat_proc=ap_flat_proc,
                    pa_flat_proc=pa_flat_proc,
                    rays_per_proj_eval=rays_eval,
                    bg_weight=args.bg_weight,
                    weight_threshold=args.weight_threshold,
                    ct_context=ct_context,
                )
            if args.debug_zero_var:
                targ_std = (target_ap.std().item(), target_pa.std().item())
                if pred_std[0] < 1e-7 or pred_std[1] < 1e-7:
                    print("⚠️ Zero-Var Vorhersage erkannt – dumppe Debug-Daten ...", flush=True)
                    dump_debug_tensor(debug_dir / f"step_{step:05d}_pred_ap.pt", pred_ap)
                    dump_debug_tensor(debug_dir / f"step_{step:05d}_pred_pa.pt", pred_pa)
                    dump_debug_tensor(debug_dir / f"step_{step:05d}_target_ap.pt", target_ap)
                    dump_debug_tensor(debug_dir / f"step_{step:05d}_target_pa.pt", target_pa)
                    dump_debug_tensor(debug_dir / f"step_{step:05d}_rays_ap.pt", ray_batch_ap)
                    dump_debug_tensor(debug_dir / f"step_{step:05d}_rays_pa.pt", ray_batch_pa)
                    if extras_ap.get("raw") is not None:
                        dump_debug_tensor(debug_dir / f"step_{step:05d}_raw_ap.pt", extras_ap["raw"])
                    if extras_pa.get("raw") is not None:
                        dump_debug_tensor(debug_dir / f"step_{step:05d}_raw_pa.pt", extras_pa["raw"])
                    print(
                        f"   targetσ=({targ_std[0]:.3e},{targ_std[1]:.3e}) "
                        f"| predμ=({pred_mean[0]:.3e},{pred_mean[1]:.3e})",
                        flush=True,
                    )

        val_loss = val_stats["loss"] if val_stats is not None else None
        val_psnr_ap = val_stats["psnr_ap"] if val_stats is not None else None
        val_psnr_pa = val_stats["psnr_pa"] if val_stats is not None else None
        val_mae_ap = val_stats["mae_ap"] if val_stats is not None else None
        val_mae_pa = val_stats["mae_pa"] if val_stats is not None else None

        msg = (
            f"[step {step:05d}] loss={loss.item():.6f} | ap={loss_ap.item():.6f} | pa={loss_pa.item():.6f} "
            f"| act={loss_act.item():.6f} | ct={loss_ct.item():.6f} | tv={loss_tv.item():.6f} | tv_mu={loss_tv_mu.item():.6f} "
            f"| mu_gate={loss_mu_gate.item():.6f} | tv3d={loss_tv3d.item():.6f} | mask={loss_mask.item():.6f} | zreg={loss_reg.item():.6f} "
            f"| mae_ap={mae_ap:.6f} | mae_pa={mae_pa:.6f} "
            f"| psnr_ap={psnr_ap:.2f} | psnr_pa={psnr_pa:.2f} "
            f"| predμ_raw=({pred_mean_raw[0]:.3e},{pred_mean_raw[1]:.3e}) predσ_raw=({pred_std_raw[0]:.3e},{pred_std_raw[1]:.3e}) "
            f"| predμ=({pred_mean[0]:.3e},{pred_mean[1]:.3e}) predσ=({pred_std[0]:.3e},{pred_std[1]:.3e})"
        )
        if val_stats is not None:
            msg += (
                f" | test_loss={val_loss:.6f} | test_psnr_ap={val_psnr_ap:.2f} | test_psnr_pa={val_psnr_pa:.2f} "
                f"| test_mae_ap={val_mae_ap:.6f} | test_mae_pa={val_mae_pa:.6f}"
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
                loss_tv.item(),
                loss_tv_mu.item(),
                loss_mu_gate.item(),
                loss_tv3d.item(),
                loss_mask.item(),
                mae_ap,
                mae_pa,
                psnr_ap,
                psnr_pa,
                pred_mean[0],
                pred_mean[1],
                pred_std[0],
                pred_std[1],
                val_loss,
                val_psnr_ap,
                val_psnr_pa,
                val_mae_ap,
                val_mae_pa,
                iter_ms,
                optimizer.param_groups[0]["lr"],
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

    debug_ap_pa = bool(getattr(args, "debug_ap_pa", False))
    debug_save_npy = debug_ap_pa and bool(getattr(args, "debug_ap_pa_save_npy", False))
    debug_every = max(1, int(getattr(args, "debug_ap_pa_every", 50)))
    asset_mode = getattr(args, "debug_ap_pa_assets", "images+hists")

    prev_flag = generator.use_test_kwargs
    generator.eval()
    generator.use_test_kwargs = True
    with torch.no_grad():
        debug_override = None
        return_rays = False
        if debug_ap_pa:
            debug_override = {
                "attenuation_debug": True,
                "retraw": True,
            }
            if DEBUG_PRINTS:
                debug_override["debug_prints"] = True
            return_rays = True
        proj_ap, _, _, extras_ap = generator.render_from_pose(
            z_train.detach(), generator.pose_ap, ct_context=ct_context, debug_override=debug_override, return_rays=return_rays
        )
        proj_pa, _, _, extras_pa = generator.render_from_pose(
            z_train.detach(), generator.pose_pa, ct_context=ct_context, debug_override=debug_override, return_rays=return_rays
        )
    generator.train()
    generator.use_test_kwargs = prev_flag or False

    H, W = generator.H, generator.W
    pred_ap_raw = proj_ap[0].reshape(H, W)
    pred_pa_raw = proj_pa[0].reshape(H, W)
    ap_np_raw = pred_ap_raw.detach().cpu().numpy()
    pa_np_raw = pred_pa_raw.detach().cpu().numpy()
    fp = outdir / "preview"
    fp.mkdir(parents=True, exist_ok=True)
    if debug_ap_pa and not final_debug_done:
        print(f"[debug ap/pa] ENTER preview path (final) at step {args.max_steps}", flush=True)
        debug_compare_ap_pa(
            pred_ap_raw,
            pred_pa_raw,
            extras_ap,
            extras_pa,
            generator,
            tag="final",
            debug_save_npy=debug_save_npy,
            out_dir=fp,
            step=args.max_steps,
            save_assets=True,
            asset_mode=asset_mode,
        )
        final_debug_done = True

    if debug_final_no_atten:
        with torch.no_grad():
            proj_ap_no_att, _, _, _ = generator.render_from_pose(
                z_train.detach(),
                generator.pose_ap,
                ct_context=ct_context,
                debug_override=debug_override,
                return_rays=False,
                force_disable_atten=True,
            )
            proj_pa_no_att, _, _, _ = generator.render_from_pose(
                z_train.detach(),
                generator.pose_pa,
                ct_context=ct_context,
                debug_override=debug_override,
                return_rays=False,
                force_disable_atten=True,
            )
        ap_no_att = proj_ap_no_att[0].reshape(H, W)
        pa_no_att = proj_pa_no_att[0].reshape(H, W)
        rmse_ap = torch.sqrt(torch.mean((pred_ap_raw - ap_no_att) ** 2)).item()
        rmse_pa = torch.sqrt(torch.mean((pred_pa_raw - pa_no_att) ** 2)).item()
        std_ap_att = pred_ap_raw.std().item()
        std_ap_no = ap_no_att.std().item()
        std_pa_att = pred_pa_raw.std().item()
        std_pa_no = pa_no_att.std().item()
        print(
            f"[debug atten] AP rmse={rmse_ap:.3e} std_with={std_ap_att:.3e} std_no={std_ap_no:.3e} | "
            f"PA rmse={rmse_pa:.3e} std_with={std_pa_att:.3e} std_no={std_pa_no:.3e}",
            flush=True,
        )
    ap_np = adjust_projection(ap_np_raw, "ap")
    pa_np = adjust_projection(pa_np_raw, "pa")
    save_img(ap_np, fp / "final_AP.png", "AP final")
    save_img(pa_np, fp / "final_PA.png", "PA final")
    print("🖼️ Finale Previews gespeichert.", flush=True)
    print("   ", (fp / "final_AP.png").resolve(), flush=True)
    print("   ", (fp / "final_PA.png").resolve(), flush=True)

    save_checkpoint(args.max_steps, generator, z_train, optimizer, scaler, ckpt_dir)
    print("✅ Training run finished.", flush=True)


if __name__ == "__main__":
    train()
