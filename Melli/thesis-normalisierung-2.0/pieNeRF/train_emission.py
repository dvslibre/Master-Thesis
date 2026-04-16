"""Mini-training script for the SPECT emission NeRF."""
import argparse
import csv
import json
import math
import logging
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Mapping, Any
from collections import Counter

import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from torch.utils.data import DataLoader, Subset

from graf.config import get_data, build_models
from graf.encoders import ProjectionEncoder
from utils.ray_split import (
    PixelSplit,
    make_pixel_split_from_ap_pa,
    make_pixel_split_stratified_intensity,
    sample_train_indices,
)

__VERSION__ = "emission-train v0.3"
DEBUG_PRINTS = False  # Nur Debug-Ausgaben, keine Änderung am Verhalten
ATTEN_SCALE_DEFAULT = 25.0
_POISSON_RATE_LEGACY_WARNED = False
_DEPTH_PROFILE_LOGGED = False
_SAMPLE_VOLUME_ALONG_PTS_LOGGED = False

_PRED_ACT_STATS_LOGGED = False

SOFTPLUS_DEBIAS = 0.6931471805599453


def activity_from_raw(raw: torch.Tensor, debias: bool = True) -> torch.Tensor:
    """Convert raw network output into non-negative activity (optional ln2 debias)."""
    act = F.softplus(raw)
    if debias:
        act = act - SOFTPLUS_DEBIAS
    return act.clamp_min(0.0)


ACT_DEBUG_WORLD_COORDS: List[tuple[tuple[float, float, float], tuple[int, int, int]]] = []
ACT_DEBUG_WORLD_STEP: Optional[int] = None
ACT_DEBUG_WORLD_REPORTED = False
ACT_ACT_RADIUS_CACHE: Dict[str, tuple[tuple[float, float, float], str]] = {}
ACT_RADIUS_LOGGED_KEYS: set[str] = set()


def _get_git_short_hash(cwd: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "NA"


def _log_runtime_provenance() -> None:
    cwd = os.getcwd()
    script_path = Path(__file__).resolve()
    git_rev = _get_git_short_hash(cwd)
    python_exec = sys.executable
    print(
        f"[runtime] cwd={cwd} git_rev={git_rev} train_emission_py={script_path} python={python_exec}",
        flush=True,
    )


def _resolve_activity_radius_xyz(generator, radius_xyz_override=None) -> tuple[float, float, float]:
    if radius_xyz_override is not None:
        return tuple(float(r) for r in radius_xyz_override)
    radius_xyz = getattr(generator, "radius_xyz", None)
    if radius_xyz is not None:
        return tuple(float(r) for r in radius_xyz)
    radius = generator.radius
    if isinstance(radius, (tuple, list)):
        radius = radius[-1]
    scalar = float(radius)
    return (scalar, scalar, scalar)


def _normalize_act_path(act_path: Any) -> Optional[str]:
    if act_path is None:
        return None
    if isinstance(act_path, (list, tuple)):
        for candidate in act_path:
            norm = _normalize_act_path(candidate)
            if norm is not None:
                return norm
        return None
    if isinstance(act_path, Path):
        return str(act_path)
    if isinstance(act_path, str):
        stripped = act_path.strip()
        return stripped if stripped else None
    return str(act_path)


def _act_spacing_from_meta(act_meta: Mapping[str, Any]) -> tuple[tuple[float, float, float], str, float] | None:
    normalized_path = _normalize_act_path(act_meta.get("act_path"))
    if normalized_path is None:
        return None
    act_path = Path(normalized_path)
    meta_file = act_path.parent / "meta_simple.json"
    if not meta_file.exists():
        return None
    try:
        with meta_file.open("r") as reader:
            meta_json = json.load(reader)
    except Exception:
        return None
    sd_mm = meta_json.get("sd_mm")
    if sd_mm is None:
        return None
    try:
        sd_mm_val = float(sd_mm)
    except Exception:
        return None
    sd_cm = sd_mm_val / 10.0
    spacing = (sd_cm, sd_cm, sd_cm)
    return spacing, str(meta_file), sd_mm_val


def _describe_active_axis_range(
    volume: np.ndarray | torch.Tensor, thr: float = 1e-8
) -> Optional[dict]:
    """Return the most active axis range (min, max, count) for the given volume."""
    arr = np.asarray(volume)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    mask = arr > thr
    if not mask.any():
        return None
    best_axis = None
    best_count = -1
    best_mask = None
    for axis in range(3):
        axes = tuple(i for i in range(3) if i != axis)
        axis_mask = np.any(mask, axis=axes)
        count = int(np.count_nonzero(axis_mask))
        if count > best_count:
            best_count = count
            best_axis = axis
            best_mask = axis_mask
    if best_mask is None or best_mask.sum() == 0:
        return None
    indices = np.nonzero(best_mask)[0]
    return {
        "axis": best_axis,
        "min": int(indices[0]),
        "max": int(indices[-1]),
        "n_slices": int(best_mask.sum()),
    }


def _maybe_log_act_radius(
    key: str,
    *,
    step: Optional[int],
    act_path: Optional[str],
    meta_file: Optional[str],
    sd_mm: Optional[float],
    sd_cm: tuple[float, float, float] | None,
    act_shape: tuple[int, int, int],
    radius_xyz: tuple[float, float, float],
    generator_radius_xyz: tuple[float, float, float],
    source: str,
) -> None:
    if key in ACT_RADIUS_LOGGED_KEYS:
        return
    ACT_RADIUS_LOGGED_KEYS.add(key)
    step_str = str(step) if step is not None else "n/a"
    act_path_desc = act_path if act_path is not None else "n/a"
    meta_desc = meta_file if meta_file is not None else "n/a"
    sd_mm_desc = f"{sd_mm:.4f}" if sd_mm is not None else "n/a"
    sd_cm_desc = (
        f"{sd_cm[0]:.3f}/{sd_cm[1]:.3f}/{sd_cm[2]:.3f} cm" if sd_cm is not None else "n/a"
    )
    act_shape_desc = f"({act_shape[0]},{act_shape[1]},{act_shape[2]})"
    radius_desc = f"({radius_xyz[0]:.3f},{radius_xyz[1]:.3f},{radius_xyz[2]:.3f})"
    gen_radius_desc = f"({generator_radius_xyz[0]:.3f},{generator_radius_xyz[1]:.3f},{generator_radius_xyz[2]:.3f})"
    print(
        f"[ACT-RADIUS] step={step_str} source={source} act_path={act_path_desc} meta_file={meta_desc} "
        f"sd_mm={sd_mm_desc} sd_cm={sd_cm_desc} act_shape={act_shape_desc} "
        f"radius_xyz={radius_desc} gen_radius_xyz={gen_radius_desc}",
        flush=True,
    )


def _resolve_act_radius_xyz_from_act(
    generator,
    act: torch.Tensor,
    act_meta: Optional[Mapping[str, Any]],
    step: Optional[int] = None,
) -> tuple[float, float, float]:
    D, H, W = act.shape[-3:]
    cache_key = None
    spacing_info = None
    act_path_str = None
    if isinstance(act_meta, Mapping):
        act_path_raw = act_meta.get("act_path")
        normalized_path = _normalize_act_path(act_path_raw)
        if normalized_path:
            act_path_str = normalized_path
            cache_key = normalized_path
            cached = ACT_ACT_RADIUS_CACHE.get(normalized_path)
            if cached is not None:
                radius_xyz, _ = cached
                generator_radius_xyz = _resolve_activity_radius_xyz(generator)
                _maybe_log_act_radius(
                    cache_key,
                    step=step,
                    act_path=act_path_str,
                    meta_file=None,
                    sd_mm=None,
                    sd_cm=None,
                    act_shape=(D, H, W),
                    radius_xyz=radius_xyz,
                    generator_radius_xyz=generator_radius_xyz,
                    source="cached",
                )
                return radius_xyz
            spacing_info = _act_spacing_from_meta(act_meta)
    if spacing_info is not None:
        spacing_cm, meta_file, sd_mm_val = spacing_info
        sx, sy, sz = spacing_cm
        radius_xyz = (
            0.5 * W * sx,
            0.5 * H * sy,
            0.5 * D * sz,
        )
        if cache_key is not None:
            ACT_ACT_RADIUS_CACHE[cache_key] = (radius_xyz, "meta")
        generator_radius_xyz = _resolve_activity_radius_xyz(generator)
        _maybe_log_act_radius(
            cache_key or "meta/full_extent",
            step=step,
            act_path=act_path_str,
            meta_file=meta_file,
            sd_mm=sd_mm_val,
            sd_cm=spacing_cm,
            act_shape=(D, H, W),
            radius_xyz=radius_xyz,
            generator_radius_xyz=generator_radius_xyz,
            source="meta/full_extent",
        )
        return radius_xyz
    radius_xyz = _resolve_activity_radius_xyz(generator)
    source = "fallback"
    if cache_key is not None:
        ACT_ACT_RADIUS_CACHE[cache_key] = (radius_xyz, source)
    generator_radius_xyz = radius_xyz
    _maybe_log_act_radius(
        cache_key or "fallback",
        step=step,
        act_path=act_path_str,
        meta_file=None,
        sd_mm=None,
        sd_cm=None,
        act_shape=(D, H, W),
        radius_xyz=radius_xyz,
        generator_radius_xyz=generator_radius_xyz,
        source=source,
    )
    return radius_xyz


def _world_stats_from_weights(
    weights: torch.Tensor, radius_xyz: tuple[float, float, float]
) -> tuple[
    Optional[tuple[float, float, float]],
    Optional[tuple[float, float, float]],
    Optional[tuple[float, float, float]],
]:
    if weights.numel() == 0:
        return (None, None, None)
    D, H, W = weights.shape[-3:]
    device = weights.device
    dtype = weights.dtype
    rx, ry, rz = radius_xyz
    x_coords = idx_to_coord(torch.arange(W, device=device), W, rx).to(dtype)
    y_coords = idx_to_coord(torch.arange(H, device=device), H, ry).to(dtype)
    z_coords = idx_to_coord(torch.arange(D, device=device), D, rz).to(dtype)

    total = weights.sum()
    total_val = float(total.item())
    if total_val <= 0.0:
        return (None, None, None)

    x_weight = weights.sum(dim=(0, 1))
    y_weight = weights.sum(dim=(0, 2))
    z_weight = weights.sum(dim=(1, 2))

    x_com = float(((x_coords * x_weight).sum() / total).item())
    y_com = float(((y_coords * y_weight).sum() / total).item())
    z_com = float(((z_coords * z_weight).sum() / total).item())

    mask = weights > 0
    if mask.any():
        z_idx, y_idx, x_idx = mask.nonzero(as_tuple=True)
        x_min = float(x_coords[x_idx].min().item())
        x_max = float(x_coords[x_idx].max().item())
        y_min = float(y_coords[y_idx].min().item())
        y_max = float(y_coords[y_idx].max().item())
        z_min = float(z_coords[z_idx].min().item())
        z_max = float(z_coords[z_idx].max().item())
        min_tuple = (x_min, y_min, z_min)
        max_tuple = (x_max, y_max, z_max)
    else:
        min_tuple = None
        max_tuple = None

    return (x_com, y_com, z_com), min_tuple, max_tuple


def _coord_tuple_str(coord: Optional[tuple[float, float, float]]) -> str:
    if coord is None:
        return "n/a"
    return f"({coord[0]:.3f},{coord[1]:.3f},{coord[2]:.3f})"


def _print_align_check(
    act_stats: tuple[
        Optional[tuple[float, float, float]],
        Optional[tuple[float, float, float]],
        Optional[tuple[float, float, float]],
    ],
    ct_stats: tuple[
        Optional[tuple[float, float, float]],
        Optional[tuple[float, float, float]],
        Optional[tuple[float, float, float]],
    ],
    mask_frac: Optional[float] = None,
) -> None:
    act_com, act_min, act_max = act_stats
    ct_com, ct_min, ct_max = ct_stats
    delta_value = None
    if act_com is not None and ct_com is not None:
        delta_value = math.sqrt(sum((a - b) ** 2 for a, b in zip(act_com, ct_com)))
    delta_str = f"{delta_value:.4f}" if delta_value is not None else "n/a"
    mask_desc = f"{mask_frac:.4f}" if mask_frac is not None else "n/a"
    print(
        "[ALIGN-CHECK] "
        f"act_com={_coord_tuple_str(act_com)} act_range_min={_coord_tuple_str(act_min)} act_range_max={_coord_tuple_str(act_max)} "
        f"ct_com={_coord_tuple_str(ct_com)} ct_range_min={_coord_tuple_str(ct_min)} ct_range_max={_coord_tuple_str(ct_max)} "
        f"delta={delta_str} mask_frac={mask_desc}",
        flush=True,
    )


def _world_coord_to_grid_idx(value: float, grid_size: int, radius_axis: float) -> int:
    if grid_size <= 1 or radius_axis == 0.0:
        return 0
    normalized = (value / (2.0 * radius_axis)) + 0.5
    normalized = max(0.0, min(1.0, normalized))
    idx = normalized * (grid_size - 1)
    return int(round(idx))


def _world_coords_to_grid_indices(
    coord: torch.Tensor, res: int, radius_xyz: tuple[float, float, float]
) -> tuple[int, int, int]:
    x, y, z = coord.tolist()
    rx, ry, rz = radius_xyz
    ix = _world_coord_to_grid_idx(x, res, rx)
    iy = _world_coord_to_grid_idx(y, res, ry)
    iz = _world_coord_to_grid_idx(z, res, rz)
    return iz, iy, ix

def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"yes", "true", "t", "1"}:
        return True
    if val in {"no", "false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")


def float_or_none(v):
    if v is None:
        return None
    if isinstance(v, float):
        return v
    val = str(v).strip().lower()
    if val in {"none", "null", ""}:
        return None
    try:
        return float(val)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Float or None expected, got '{v}'.") from exc


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
        default=50,
        help="How often to print running loss statistics.",
    )
    parser.add_argument(
        "--preview-every",
        type=int,
        default=100,
        help="If >0 renders and stores full AP/PA previews every N steps (slow, full-frame render).",
    )
    parser.add_argument(
        "--depth-profile-signal-quantile",
        type=float,
        default=0.99,
        help="Quantil fuer Signal-Kandidaten in Depth-Profile-Plots (auf Target-Score).",
    )
    parser.add_argument(
        "--depth-profile-bg-quantile",
        type=float,
        default=0.10,
        help="Quantil fuer Background-Kandidaten in Depth-Profile-Plots (auf Target-Score).",
    )
    parser.add_argument(
        "--depth-profile-seed",
        type=int,
        default=123,
        help="Seed fuer deterministische Auswahl der Depth-Profile-Strahlen.",
    )
    parser.add_argument(
        "--final-sagittal-viz",
        action="store_true",
        help="Speichere am Trainingsende eine sagittale GT-vs-Depth-Curtain Visualisierung.",
    )
    parser.add_argument(
        "--final-sagittal-axis0-idx",
        type=int,
        nargs=3,
        default=[80, 128, 200],
        metavar=("IDX0", "IDX1", "IDX2"),
        help="Drei feste axis0-Indizes fuer die finale sagittale Visualisierung (wie in test.py).",
    )
    parser.add_argument(
        "--final-sagittal-debug",
        action="store_true",
        help="Speichere zusaetzlich eine Debug-PNG mit E[depth]/argmax pro Pixel.",
    )
    parser.add_argument(
        "--final-act-compare",
        action="store_true",
        help="Speichere am Trainingsende eine GT-vs-Pred Activity-Compare PNG via Volumen-Slicing.",
    )
    parser.add_argument(
        "--final-act-compare-axial",
        action="store_true",
        default=True,
        help="Finale Compare-PNG verwendet axialen Vergleich (axis2). Flag ist aus Kompatibilitaetsgruenden akzeptiert.",
    )
    parser.add_argument(
        "--final-act-compare-axis2-idx",
        type=int,
        nargs=3,
        default=[65, 260, 325],
        metavar=("Z0", "Z1", "Z2"),
        help="Drei feste axis2-Indizes fuer den axialen finalen Compare-Plot.",
    )
    parser.add_argument(
        "--final-act-compare-scale",
        type=str,
        default="separate",
        choices=["shared", "separate"],
        help="Color scaling fuer finale Activity-Compare: shared (vergleichbar) oder separate (strukturbetont).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="If >0 stores checkpoints every N steps in addition to the final checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optionaler Checkpoint-Pfad zum Laden (z. B. .../checkpoints/checkpoint_step08000.pt).",
    )
    parser.add_argument(
        "--checkpoint-load-optimizer",
        action="store_true",
        help="Lade bei --checkpoint auch Optimizer/Scaler-State (Resume-Training).",
    )
    parser.add_argument(
        "--checkpoint-strict",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
        help="Strictes Laden der State-Dicts bei --checkpoint.",
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
        "--debug-sanity-checks",
        action="store_true",
        help="Aktiviert Sanity-Checks (Attenuation/Scaling/Geometry); kann Training verlangsamen.",
    )
    parser.add_argument(
        "--debug-sanity-every",
        type=int,
        default=100,
        help="Intervall in Steps fuer Sanity-Checks (nur bei --debug-sanity-checks).",
    )
    parser.add_argument(
        "--no-final-test",
        action="store_true",
        help="Disable the final test evaluation after training.",
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
        "--act-sparsity-weight",
        type=float,
        default=0.0,
        help="Gewicht für L1-Sparsity auf der vorhergesagten Aktivität (pred_pos.mean()).",
    )
    parser.add_argument(
        "--act-tv-weight",
        type=float,
        default=0.0,
        help="Gewicht für isotrope 3D Total Variation auf pred_pos (ACT Phase A Regularisierung).",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Aktiviert den Hybrid-Ansatz: AP/PA -> Encoder -> z_enc Conditioning + Projection-Loss als Nebenloss.",
    )
    parser.add_argument(
        "--proj-loss-type",
        type=str,
        default="poisson",
        choices=["poisson", "sqrt_mse", "huber"],
        help="Projection-Loss-Typ (poisson oder sqrt_mse).",
    )
    parser.add_argument(
        "--poisson-rate-mode",
        type=str,
        default="identity",
        choices=["softplus_shift", "identity"],
        help="Poisson-Rate-Definition: softplus_shift (legacy) oder identity (rate=pred).",
    )
    parser.add_argument(
        "--poisson-rate-floor",
        type=float,
        default=0.0,
        help="Optionaler Floor fuer Poisson-Rate (Counts-Skala). >0 aktiviert Stabilisierung.",
    )
    parser.add_argument(
        "--poisson-rate-floor-mode",
        type=str,
        default="clamp",
        choices=["clamp", "softplus_hinge"],
        help="Poisson-Rate-Floor-Modus: clamp oder softplus_hinge (smooth clamp, no bias at boundary).",
    )
    parser.add_argument(
        "--lambda-ray-tv-weight",
        type=float,
        default=0.0,
        help="Optionaler TV-Smoothness auf lambda entlang der Ray-Depth-Achse (nur Projection-Pfad).",
    )
    parser.add_argument(
        "--proj-loss-weight",
        type=float,
        default=0.1,
        help="Gewicht fuer den Projection-Loss im Hybrid-Modus (Nebenloss).",
    )
    parser.add_argument(
        "--proj-warmup-steps",
        type=int,
        default=0,
        help="Optionales Warmup: Schritte 0..W nur ACT+TV, Projection-Loss danach aktiv.",
    )
    parser.add_argument(
        "--latent-dropout-prob",
        type=float,
        default=0.0,
        help="Wahrscheinlichkeit, mit der z_latent während des Trainings pro Sample auf 0 gesetzt wird.",
    )
    parser.add_argument(
        "--debug-z-sensitivity",
        action="store_true",
        help="Aktiviere Diagnostik: prüft, wie stark z_latent die Projektionen/ACT beeinflusst (nur einmal am Ende).",
    )
    parser.add_argument(
        "--debug-z-cosine",
        action="store_true",
        help="Logge Cosine-Similarity/L2-Abstand der z_enc_proj der Test-Patients in save_test_volume_slices.",
    )
    parser.add_argument(
        "--debug-latent-stats",
        action="store_true",
        help="Logge z-Conditioning-Statistiken (Normen, Cosines, Gradnormen) waehrend Training.",
    )
    parser.add_argument(
        "--debug-latent-stats-every",
        type=int,
        default=50,
        help="Intervall fuer --debug-latent-stats (in Steps).",
    )
    parser.add_argument(
        "--grad-clip-enabled",
        action="store_true",
        default=True,
        help="Aktiviere grad clipping (clip_grad_norm) für den Generator (standardmäßig an).",
    )
    parser.add_argument(
        "--no-grad-clip",
        action="store_false",
        dest="grad_clip_enabled",
        help="Deaktiviere grad clipping, wenn standardmäßig aktiviert.",
    )
    parser.add_argument(
        "--grad-clip-max-norm",
        type=float,
        default=1.0,
        help="max_norm für grad clipping (nur aktiv wenn --grad-clip-enabled gesetzt).",
    )
    parser.add_argument(
        "--clip-grad-decoder",
        type=float,
        default=0.0,
        help="Optionales decoder-only grad clipping (0=aus).",
    )
    parser.add_argument(
        "--debug-grad-terms-every",
        type=int,
        default=0,
        help="Wenn >0: loggt getrennte Grad-Normen fuer loss_act/loss_proj alle N Steps.",
    )
    parser.add_argument(
        "--proj-weight-min",
        type=float,
        default=0.005,
        help="Unteres Limit fuer proj_loss Gewicht waehrend Warmup/Ramp.",
    )
    parser.add_argument(
        "--proj-ramp-steps",
        type=int,
        default=200,
        help="Anzahl Steps fuer lineare Ramp auf proj_loss_weight nach Warmup.",
    )
    parser.add_argument(
        "--proj-target-source",
        type=str,
        default="counts",
        choices=["counts", "norm"],
        help="Quelle fuer Projection Targets: counts (ap_counts/pa_counts) oder norm (ap/pa).",
    )
    parser.add_argument(
        "--pred-to-counts-scale-override",
        type=float,
        default=-1.0,
        help="Override fuer pred_to_counts_scale (>0 nutzt diesen Wert statt proj_scale_joint_p99).",
    )
    parser.add_argument(
        "--use-gain",
        action="store_true",
        help="Aktiviert lernbaren Gain (multiplikativ auf Projektion). Default: False.",
    )
    parser.add_argument(
        "--proj-gain-source",
        type=str,
        default="z_enc",
        choices=["z_enc", "scalar", "none"],
        help="Gain g fuer counts-Projektion: z_enc-Head, lernbarer scalar oder none.",
    )
    parser.add_argument(
        "--gain-reg-weight",
        type=float,
        default=1e-4,
        help="Gewicht fuer Gain-Regularisierung (log-gain prior).",
    )
    parser.add_argument(
        "--gain-reg-scale",
        type=float,
        default=1.0,
        help="Skalierung fuer Gain-Regularisierung (1.0 = unveraendert).",
    )
    parser.add_argument(
        "--gain-warn-min",
        type=float,
        default=1e-3,
        help="Warnschwelle fuer Gain (min) im Sanity-Check.",
    )
    parser.add_argument(
        "--gain-warn-max",
        type=float,
        default=1e3,
        help="Warnschwelle fuer Gain (max) im Sanity-Check.",
    )
    parser.add_argument(
        "--gain-prior-mode",
        type=str,
        default="ema_init",
        choices=["ema_init", "fixed"],
        help="Gain prior: EMA der ersten Schritte oder fixer Wert.",
    )
    parser.add_argument(
        "--gain-prior-value",
        type=float,
        default=1.0,
        help="Fixer Gain-Prior (nur bei gain-prior-mode=fixed).",
    )
    parser.add_argument(
        "--gain-clamp-min",
        type=float,
        default=0.05,
        help="Optionales Minimum fuer Gain (nur im proj-loss Pfad).",
    )
    parser.add_argument(
        "--gain-clamp-max",
        type=float_or_none,
        default=5.0,
        help="Optionales Maximum fuer Gain (nur im proj-loss Pfad). Setze 'none' fuer aus.",
    )
    parser.add_argument(
        "--gain-warmup-mode",
        type=str,
        default="none",
        choices=["none", "one", "prior"],
        help="Fixiert Gain waehrend proj_warmup_steps (one=1.0, prior=gain_prior_value).",
    )
    parser.add_argument(
        "--encoder-proj-transform",
        type=str,
        default="log1p",
        choices=["log1p", "sqrt", "none"],
        help="Transform fuer Encoder-Input: log1p(y/s), sqrt(y/s) oder none.",
    )
    parser.add_argument(
        "--proj-scale-source",
        type=str,
        default="meta_p99",
        choices=["meta_p99", "compute_p99", "sumcounts", "none"],
        help="Skalenquelle fuer Encoder-Inputs: meta_p99, compute_p99, sumcounts oder none.",
    )
    parser.add_argument(
        "--act-norm-source",
        type=str,
        default="p99_scan",
        choices=["none", "p99_global", "p99_scan", "fixed"],
        help="Normierung fuer ACT-Loss: none, p99_global, p99_scan oder fixed.",
    )
    parser.add_argument(
        "--act-norm-value",
        type=float,
        default=1.0,
        help="Fixer Normierungsfaktor fuer ACT-Loss (nur bei act-norm-source=fixed).",
    )
    parser.add_argument(
        "--act-scale-reg-weight",
        type=float,
        default=1e-6,
        help="Gewicht fuer ACT-Skalenregularisierung (sum(pred_act_raw) ≈ sum(gt_act)).",
    )
    parser.add_argument(
        "--encoder-use-ct",
        action="store_true",
        help="Optional: CT als zusaetzlicher Encoder-Input (Mean-Projektion).",
    )
    parser.add_argument(
        "--test-noise-mode",
        type=str,
        default="none",
        choices=["none", "poisson_counts"],
        help="Test/Inference-only Noise fuer AP/PA vor dem Encoder: none oder poisson_counts.",
    )
    parser.add_argument(
        "--test-noise-kappa",
        type=float,
        default=1.0,
        help="Skalierung fuer Test-Poisson-Noise: noisy = Poisson(kappa*counts)/kappa.",
    )
    parser.add_argument(
        "--test-noise-seed",
        type=int,
        default=0,
        help="Basis-Seed fuer Test-Poisson-Noise (pro Phantom deterministisch abgeleitet).",
    )
    parser.add_argument(
        "--z-enc-alpha",
        type=float,
        default=0.1,
        help="Skalierung fuer z_enc im Hybrid-Conditioning (z_latent = alpha * z_enc_proj ohne trainables z).",
    )
    parser.add_argument(
        "--encoder-lr-mult",
        type=float,
        default=0.1,
        help="Multiplikator fuer Encoder-Lernrate relativ zu lr_g.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fuehrt einen Smoke-Test mit einem Batch (Forward+Backward) aus und beendet.",
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
        "--act-pos-fraction",
        type=float,
        default=0.5,
        help="Anteil positiver ACT-Samples pro Batch.",
    )
    parser.add_argument(
        "--act-pos-threshold",
        type=float,
        default=1e-8,
        help="Threshold für positives ACT-Sampling.",
    )
    parser.add_argument(
        "--act-only",
        action="store_true",
        help="Deaktiviert Projektionsteil (Forward + Loss); nur ACT + Regularizer.",
    )
    parser.add_argument(
        "--debug-enc",
        action="store_true",
        help="Einmalige Debug-Logs für Encoder-Trainierbarkeit (Step 1).",
    )
    parser.add_argument(
        "--debug-act",
        action="store_true",
        help="Einmalige Debug-Logs für ACT-Targets/Normierung/Pred/Grad (Step 1).",
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
        "--depth-sanity-every",
        type=int,
        default=50,
        help="Depth-Checks/Abbruch alle N Schritte (0 = deaktiviert).",
    )
    parser.add_argument(
        "--proj-collapse-patience",
        type=int,
        default=0,
        help="Abbruch erst nach N aufeinanderfolgenden Projection-Collapses (0 = nie aborten).",
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
        "--debug-proj-orientation",
        action="store_true",
        help="Speichere pro Preview ein JSON mit Orientierungs-Stats (shape/min/max/argmax/center-of-mass/hash) fuer AP/PA.",
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


def _extract_patient_id_from_batch(batch) -> Optional[str]:
    """Versucht patient_id aus einem DataLoader-Batch zu lesen (ohne Annahmen zu erzwingen)."""
    if not isinstance(batch, dict):
        return None
    meta = batch.get("meta")
    if isinstance(meta, dict):
        pid = meta.get("patient_id")
        if torch.is_tensor(pid):
            if pid.numel() == 0:
                return None
            return str(pid.view(-1)[0].item())
        if isinstance(pid, (list, tuple)):
            return str(pid[0]) if pid else None
        return str(pid) if pid is not None else None
    if isinstance(meta, (list, tuple)) and meta:
        first = meta[0]
        if isinstance(first, dict):
            pid = first.get("patient_id")
            return str(pid) if pid is not None else None
    return None


def _slugify_patient_id(patient_id: Optional[str]) -> str:
    if patient_id is None:
        return "unknown"
    slug = re.sub(r"[^\w.-]+", "_", patient_id.strip())
    return slug or "unknown"


def _stable_patient_seed(base_seed: int, patient_id: str) -> int:
    digest = hashlib.sha1(str(patient_id).encode("utf-8")).hexdigest()
    return int(base_seed) + int(digest[:8], 16)


def print_dataset_summary(dataset, max_print: int = 20):
    """Gibt eine kompakte Zusammenfassung der Dataset-Struktur aus (ohne Bild/Volumen-Loading)."""
    if dataset is None:
        print("[debug][dataset] dataset=None; skip summary.", flush=True)
        return
    n = len(dataset)
    patient_ids = []
    if hasattr(dataset, "get_patient_id"):
        for idx in range(n):
            pid = dataset.get_patient_id(idx)
            if pid is not None:
                patient_ids.append(pid)
    elif hasattr(dataset, "entries"):
        for e in getattr(dataset, "entries", []):
            pid = e.get("patient_id") if isinstance(e, dict) else None
            if pid is not None:
                patient_ids.append(pid)
    else:
        print("[debug][dataset] No lightweight patient_id accessor found.", flush=True)
        return

    counts = Counter(patient_ids)
    n_unique = len(counts)
    count_vals = list(counts.values())
    if count_vals:
        min_c = int(np.min(count_vals))
        med_c = float(np.median(count_vals))
        max_c = int(np.max(count_vals))
    else:
        min_c = med_c = max_c = float("nan")
    print(
        f"[debug][dataset] #samples={n} | #unique patient_id={n_unique} | "
        f"samples/patient min/median/max={min_c}/{med_c}/{max_c}",
        flush=True,
    )
    if counts:
        top_k = min(10, len(counts))
        top = counts.most_common(top_k)
        top_str = ", ".join([f"{pid}:{cnt}" for pid, cnt in top[:max_print]])
        print(f"[debug][dataset] top patient_id (count): {top_str}", flush=True)


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


def save_img_abs(
    arr,
    path,
    *,
    title=None,
    step=None,
    view=None,
    patient_id=None,
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    allow_arr_max: bool = True,
    cbar_label="counts",
    cmap="viridis",
    header_text: Optional[str] = None,
    include_data_stats: bool = True,
):
    """Save a counts-based preview image (no normalization) with colorbar."""
    import matplotlib.pyplot as plt

    data = np.asarray(arr, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    finite = np.isfinite(data)
    if vmax is None:
        if finite.any():
            try:
                candidate = float(np.nanquantile(data[finite], 0.995))
            except Exception:
                candidate = float(np.nanmax(data[finite]))
        else:
            candidate = 0.0
    else:
        candidate = float(vmax)
    if allow_arr_max and finite.any():
        candidate = max(candidate, float(np.nanmax(data[finite])))
    candidate = max(candidate, vmin + 1e-6, 1e-6)
    vmin = float(vmin)
    if vmin >= candidate:
        candidate = vmin + 1e-6
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, origin="upper", cmap=cmap, vmin=vmin, vmax=candidate)
    cbar = fig.colorbar(im, ax=ax, label=cbar_label)
    if header_text is not None:
        ax.set_title(str(header_text), loc="center", pad=10)
    else:
        title_parts = []
        if title:
            title_parts.append(str(title))
        if view:
            title_parts.append(view)
        if step is not None:
            title_parts.append(f"step {step}")
        if patient_id:
            title_parts.append(f"id {patient_id}")
        if include_data_stats:
            minv = float(np.nanmin(data)) if finite.any() else 0.0
            maxv = float(np.nanmax(data)) if finite.any() else 0.0
            sumv = float(np.nansum(data))
            title_parts.append(f"min={minv:.2e}")
            title_parts.append(f"max={maxv:.2e}")
            title_parts.append(f"sum={sumv:.2e}")
            title_parts.append(f"vmax={candidate:.2e}")
        title_text = " | ".join(title_parts)
        max_chars = max(20, int(fig.get_size_inches()[0] * 10))
        title_wrapped = textwrap.fill(title_text, width=max_chars)
        ax.set_title(title_wrapped, loc="left", pad=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


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


def export_activity_volume(
    generator,
    z_latent,
    out_path: Optional[Path],
    res: int,
    device: torch.device,
    radius_xyz: Optional[Tuple[float, float, float]] = None,
    log_world_range: bool = False,
):
    res = int(res)
    if res <= 0:
        raise ValueError("export-vol-res must be > 0")

    global ACT_DEBUG_WORLD_COORDS, ACT_DEBUG_WORLD_STEP, ACT_DEBUG_WORLD_REPORTED
    rx, ry, rz = _resolve_activity_radius_xyz(generator, radius_xyz)
    x_coords = idx_to_coord(torch.arange(res, device=device), res, rx)
    y_coords = idx_to_coord(torch.arange(res, device=device), res, ry)
    z_coords_all = idx_to_coord(torch.arange(res, device=device), res, rz)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
    x_flat = x_grid.reshape(-1)
    y_flat = y_grid.reshape(-1)

    target_points = 262144
    chunk_depth = max(1, min(res, target_points // (res * res) if res * res > 0 else 1))

    vol = np.empty((res, res, res), dtype=np.float32)
    if log_world_range:
        x_min = float(x_coords.min().item())
        x_max = float(x_coords.max().item())
        y_min = float(y_coords.min().item())
        y_max = float(y_coords.max().item())
        z_min = float(z_coords_all.min().item())
        z_max = float(z_coords_all.max().item())
        print(
            f"[ACT-CHECK][export][world-range] "
            f"x=[{x_min:.6f},{x_max:.6f}] y=[{y_min:.6f},{y_max:.6f}] z=[{z_min:.6f},{z_max:.6f}]",
            flush=True,
        )
    with torch.no_grad():
        for z_start in range(0, res, chunk_depth):
            z_end = min(res, z_start + chunk_depth)
            z_idx = torch.arange(z_start, z_end, device=device)
            z_coords = idx_to_coord(z_idx, res, rz)
            z_rep = z_coords.repeat_interleave(x_flat.numel())
            x_rep = x_flat.repeat(z_coords.numel())
            y_rep = y_flat.repeat(z_coords.numel())
            coords = torch.stack((x_rep, y_rep, z_rep), dim=1)
            _, pred_raw = query_emission_at_points(generator, z_latent, coords, return_raw=True)
            pred = activity_from_raw(pred_raw)
            pred = pred.view(z_coords.numel(), res, res).detach().cpu().numpy().astype(np.float32)
            vol[z_start:z_end, :, :] = pred

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, vol)
    if ACT_DEBUG_WORLD_COORDS and not ACT_DEBUG_WORLD_REPORTED:
        coords_list = [
            torch.tensor(world, device=device, dtype=torch.float32) for world, _ in ACT_DEBUG_WORLD_COORDS
        ]
        if coords_list:
            with torch.no_grad():
                coords_tensor = torch.stack(coords_list, dim=0)
                query_vals = query_emission_at_points(generator, z_latent, coords_tensor)
            for idx, ((world_x, world_y, world_z), (iz, iy, ix)) in enumerate(ACT_DEBUG_WORLD_COORDS):
                export_val = float(vol[iz, iy, ix])
                query_val = float(query_vals[idx].item())
                abs_diff = abs(export_val - query_val)
                rel_diff = abs_diff / (abs(query_val) + 1e-8)
                print(
                    f"[ACT-CHECK][export] step={ACT_DEBUG_WORLD_STEP} coord=({world_x:.6f},{world_y:.6f},{world_z:.6f}) "
                    f"grid=(iz={iz},iy={iy},ix={ix}) export={export_val:.6e} query={query_val:.6e} "
                    f"abs_diff={abs_diff:.6e} rel_diff={rel_diff:.6e}",
                    flush=True,
                )
        ACT_DEBUG_WORLD_REPORTED = True
    return vol


def _radius_to_float(radius):
    if radius is None:
        return None
    if isinstance(radius, (tuple, list)):
        if not radius:
            return None
        radius = radius[-1]
    return float(radius)


def save_test_volume_slices(
    args,
    generator,
    z_latent_base: torch.Tensor,
    test_loader,
    outdir: Path,
    device: torch.device,
    max_patients: int = 4,
    slice_percents=(0.1, 0.5, 0.9),
    encoder: Optional[nn.Module] = None,
    z_fuser: Optional[nn.Module] = None,
    z_enc_alpha: float = 0.0,
):
    if test_loader is None or len(test_loader.dataset) == 0:
        return
    slice_root = outdir / "test_slices"
    slice_root.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    saved_info: list[dict] = []
    prev_use_test = generator.use_test_kwargs
    prev_encoder_training = encoder.training if encoder is not None else None
    generator.eval()
    if encoder is not None:
        encoder.eval()

    hybrid_enabled = bool(getattr(args, "hybrid", False))
    debug_cos_latents: list[torch.Tensor] = []
    debug_cos_patients: list[str] = []
    latent_diag_rows: list[dict] = []
    z_enc_vectors: list[torch.Tensor] = []
    z_enc_patients: list[str] = []

    def _tensor_stats_and_sig(tensor):
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return None, None
        flat = tensor.detach().reshape(-1).float()
        arr = flat.cpu().numpy().astype(np.float32)
        stats = (float(arr.min()), float(arr.mean()), float(arr.max()))
        sig = hashlib.sha1(arr.tobytes()).hexdigest()
        return stats, sig

    def _build_test_latent(batch: dict) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not (hybrid_enabled and encoder is not None):
            raise RuntimeError("Hybrid encoder required for test slices.")
        return build_hybrid_latent_from_batch(
            args=args,
            batch=batch,
            device=device,
            z_latent_base=z_latent_base,
            encoder=encoder,
            z_fuser=z_fuser,
            z_enc_alpha=z_enc_alpha,
        )

    def _first_sample_2d(t: torch.Tensor) -> np.ndarray:
        if t is None:
            return np.empty((0, 0), dtype=np.float32)
        td = t.detach().float().cpu()
        if td.dim() == 4:
            td = td[0, 0]
        elif td.dim() == 3:
            td = td[0]
        elif td.dim() > 2:
            td = td.reshape(td.shape[-2], td.shape[-1])
        return np.asarray(td.numpy(), dtype=np.float32)

    def _vec_stats(t: Optional[torch.Tensor]) -> Optional[dict]:
        if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
            return None
        flat = t.detach().reshape(t.shape[0], -1).float()
        norms = flat.norm(dim=1)
        return {
            "shape": list(flat.shape),
            "norm_mean": float(norms.mean().item()),
            "norm_std": float(norms.std().item()) if norms.numel() > 1 else 0.0,
            "val_mean": float(flat.mean().item()),
            "val_std": float(flat.std().item()),
        }

    patient_iter = iter(test_loader)
    try:
        while len(seen_ids) < max_patients:
            batch = next(patient_iter)
            patient_id = _extract_patient_id_from_batch(batch) or f"patient_{len(seen_ids)+1}"
            if patient_id in seen_ids:
                continue
            seen_ids.add(patient_id)
            patient_dir = slice_root / patient_id
            patient_dir.mkdir(parents=True, exist_ok=True)

            meta = batch.get("meta")
            dataset_index = None
            ap_path = None
            pa_path = None
            ct_path = None
            if isinstance(meta, dict):
                dataset_index = meta.get("dataset_index")
                ap_path = meta.get("ap_path")
                pa_path = meta.get("pa_path")
                ct_path = meta.get("ct_path")
            ap_counts = batch.get("ap_counts")
            pa_counts = batch.get("pa_counts")
            ct_vol = batch.get("ct")
            ap_counts_stats, ap_counts_sig = _tensor_stats_and_sig(ap_counts)
            pa_counts_stats, pa_counts_sig = _tensor_stats_and_sig(pa_counts)
            ct_stats, ct_sig = _tensor_stats_and_sig(ct_vol)
            print(
                f"[test][slices][inputs] patient={patient_id} idx={dataset_index} "
                f"ap_path={ap_path} pa_path={pa_path} ct_path={ct_path} "
                f"ap_counts_stats={ap_counts_stats} sig={ap_counts_sig[:8] if ap_counts_sig else None} "
                f"pa_counts_stats={pa_counts_stats} sig={pa_counts_sig[:8] if pa_counts_sig else None} "
                f"ct_stats={ct_stats} sig={ct_sig[:8] if ct_sig else None}",
                flush=True,
            )
            batch_for_latent = batch
            test_noise_mode = str(getattr(args, "test_noise_mode", "none") or "none").lower()
            if test_noise_mode == "poisson_counts":
                kappa = float(getattr(args, "test_noise_kappa", 1.0))
                if not (kappa > 0.0):
                    raise ValueError(f"--test-noise-kappa must be > 0 for poisson_counts, got {kappa}")
                if ap_counts is None or pa_counts is None or ap_counts.numel() == 0 or pa_counts.numel() == 0:
                    raise RuntimeError(
                        "test-noise-mode=poisson_counts requires ap_counts/pa_counts in the batch. "
                        "Please ensure manifest contains *_counts.npy."
                    )
                ap_counts_t = ap_counts.to(device, non_blocking=True).float().clamp_min(0.0)
                pa_counts_t = pa_counts.to(device, non_blocking=True).float().clamp_min(0.0)
                patient_seed = _stable_patient_seed(int(getattr(args, "test_noise_seed", 0)), patient_id)
                noise_gen = torch.Generator(device="cpu")
                noise_gen.manual_seed(int(patient_seed))
                noisy_ap = torch.poisson(kappa * ap_counts_t.cpu(), generator=noise_gen).to(ap_counts_t.device) / kappa
                noisy_pa = torch.poisson(kappa * pa_counts_t.cpu(), generator=noise_gen).to(pa_counts_t.device) / kappa
                noisy_ap = noisy_ap.clamp_min(0.0)
                noisy_pa = noisy_pa.clamp_min(0.0)
                batch_for_latent = dict(batch)
                batch_for_latent["proj_input_ap"] = noisy_ap
                batch_for_latent["proj_input_pa"] = noisy_pa
                counts_cat = torch.cat([ap_counts_t.reshape(-1), pa_counts_t.reshape(-1)], dim=0)
                noisy_cat = torch.cat([noisy_ap.reshape(-1), noisy_pa.reshape(-1)], dim=0)
                counts_sum = float(counts_cat.sum().item())
                noisy_sum = float(noisy_cat.sum().item())
                counts_mean = float(counts_cat.mean().item())
                noisy_mean = float(noisy_cat.mean().item())
                counts_p99 = float(torch.quantile(counts_cat, 0.99).item())
                noisy_p99 = float(torch.quantile(noisy_cat, 0.99).item())
                print(
                    f"[test][noise] patient={patient_id} mode=poisson_counts kappa={kappa:.6g} seed={patient_seed} "
                    f"sum_counts={counts_sum:.6e} sum_noisy={noisy_sum:.6e} "
                    f"mean_counts={counts_mean:.6e} mean_noisy={noisy_mean:.6e} "
                    f"p99_counts={counts_p99:.6e} p99_noisy={noisy_p99:.6e}",
                    flush=True,
                )
                np.save(patient_dir / "noisy_ap_counts.npy", _first_sample_2d(noisy_ap))
                np.save(patient_dir / "noisy_pa_counts.npy", _first_sample_2d(noisy_pa))
                print(
                    f"[test][noise] patient={patient_id} exported noisy projections to "
                    f"{patient_dir / 'noisy_ap_counts.npy'} and {patient_dir / 'noisy_pa_counts.npy'}",
                    flush=True,
                )
            if hybrid_enabled and encoder is not None:
                z_latent_batch, z_enc_batch, z_enc_proj_batch = _build_test_latent(batch_for_latent)
            else:
                z_latent_batch = z_latent_base.detach()
                z_enc_batch = None
                z_enc_proj_batch = None
            if args.debug_z_cosine and hybrid_enabled and encoder is not None:
                debug_cos_patients.append(patient_id)
                debug_cos_latents.append(z_latent_batch.detach().clone())
            if z_enc_batch is not None and z_enc_batch.numel() > 0:
                z_enc_patients.append(patient_id)
                z_enc_vectors.append(z_enc_batch[0].detach().reshape(-1).cpu())
            latent_diag_rows.append(
                {
                    "patient_id": patient_id,
                    "z_enc": _vec_stats(z_enc_batch),
                    "z_enc_proj": _vec_stats(z_enc_proj_batch),
                    "z_latent": _vec_stats(z_latent_batch),
                }
            )
            z_arr = z_latent_batch.detach().cpu().numpy().astype(np.float32)
            z_stats = (
                (float(z_arr.min()), float(z_arr.mean()), float(z_arr.max()))
                if z_arr.size > 0
                else (float("nan"), float("nan"), float("nan"))
            )
            z_sig = hashlib.sha1(z_arr.ravel().tobytes()).hexdigest()
            latent_sig = hashlib.sha1(z_latent_batch.cpu().numpy().ravel().astype(np.float32).tobytes()).hexdigest()
            print(
                f"[test][slices][latent] patient={patient_id} z_latent_sig={latent_sig[:8]} "
                f"shape={tuple(z_latent_batch.shape)} stats={z_stats} z_sha={z_sig[:8]}",
                flush=True,
            )
            pred_path = patient_dir / "activity_pred.npy"
            pred_vol = export_activity_volume(
                generator,
                z_latent_batch,
                pred_path,
                args.export_vol_res,
                device,
                log_world_range=args.debug_sanity_checks,
            )
            pred_vol = np.asarray(pred_vol, dtype=np.float32)
            pred_stats = (float(pred_vol.min()), float(pred_vol.mean()), float(pred_vol.max()))
            pred_range = _describe_active_axis_range(pred_vol, thr=1e-8)
            range_desc = (
                f"axis{pred_range['axis']} {pred_range['min']}-{pred_range['max']} "
                f"({pred_range['n_slices']} slices)"
                if pred_range
                else "no active slices"
            )
            print(
                f"[test][slices][pred] patient={patient_id} stats(min/mean/max)=("
                f"{pred_stats[0]:.3e}/{pred_stats[1]:.3e}/{pred_stats[2]:.3e}) {range_desc}",
                flush=True,
            )
            act_tensor = batch.get("act")
            gt_vol = None
            act_vol_plot = None
            if isinstance(act_tensor, torch.Tensor) and act_tensor.numel() > 0:
                act_nd = act_tensor.detach()
                if act_nd.dim() == 4:
                    act_nd = act_nd.squeeze(0)
                if act_nd.ndim == 3:
                    gt_vol = act_nd.cpu().numpy().astype(np.float32)
                    act_vol_plot = act_nd
                    gt_range = _describe_active_axis_range(gt_vol, thr=1e-8)
                    gt_desc = (
                        f"axis{gt_range['axis']} {gt_range['min']}-{gt_range['max']} "
                        f"({gt_range['n_slices']} slices)"
                        if gt_range
                        else "no active slices"
                    )
                    print(
                        f"[test][slices][gt] patient={patient_id} gt_range={gt_desc}",
                        flush=True,
                    )
            final_filename = patient_dir / f"final_{patient_id}_act_compare_axial.png"
            save_final_act_compare_volume_slicing(
                args,
                act_vol_plot,
                patient_dir,
                pred_path,
                pred_vol_np=pred_vol,
                out_path_override=final_filename,
                grid_radius=_radius_to_float(generator.radius),
                radius_xyz=tuple(float(r) for r in generator.radius_xyz),
            )
            saved_info.append(
                {
                    "patient_id": patient_id,
                    "files": [str(final_filename)],
                    "gt_included": act_vol_plot is not None,
                }
            )
    except StopIteration:
        pass
    finally:
        if prev_use_test:
            generator.eval()
        else:
            generator.train()
        if encoder is not None and prev_encoder_training is not None:
            if prev_encoder_training:
                encoder.train()
            else:
                encoder.eval()
    if args.debug_z_cosine and debug_cos_latents:
        latents = torch.cat(debug_cos_latents, dim=0).detach().cpu()
        n = latents.shape[0]
        if n > 0:
            names = ", ".join(debug_cos_patients[:n])
            if n > 1:
                normed = latents / (latents.norm(dim=1, keepdim=True) + 1e-8)
                cos_matrix = normed @ normed.t()
                off = ~torch.eye(n, dtype=torch.bool)
                cos_off = cos_matrix[off]
                cos_min = float(cos_off.min().item())
                cos_mean = float(cos_off.mean().item())
                cos_max = float(cos_off.max().item())
                dist_matrix = torch.cdist(latents, latents, p=2)
                dist_off = dist_matrix[off]
                dist_min = float(dist_off.min().item())
                dist_mean = float(dist_off.mean().item())
                dist_max = float(dist_off.max().item())
                print(
                    f"[debug-z][cosine] patients=[{names}] cos_off(min/mean/max)={cos_min:.3e}/{cos_mean:.3e}/{cos_max:.3e} "
                    f"L2_off(min/mean/max)={dist_min:.3e}/{dist_mean:.3e}/{dist_max:.3e}",
                    flush=True,
                )
            else:
                print(
                    f"[debug-z][cosine] patient={debug_cos_patients[0]} only one patient -> no cross-similarity",
                    flush=True,
                )
    if latent_diag_rows:
        diag_payload: dict[str, Any] = {"per_patient": latent_diag_rows}
        if len(z_enc_vectors) > 1:
            z_stack = torch.stack(z_enc_vectors, dim=0)
            z_norm = z_stack / (z_stack.norm(dim=1, keepdim=True) + 1e-8)
            cos_m = (z_norm @ z_norm.t()).cpu()
            pair_rows = []
            for i in range(cos_m.shape[0]):
                for j in range(i + 1, cos_m.shape[1]):
                    pair_rows.append(
                        {
                            "id_a": z_enc_patients[i],
                            "id_b": z_enc_patients[j],
                            "cosine": float(cos_m[i, j].item()),
                        }
                    )
            if pair_rows:
                cos_vals = [p["cosine"] for p in pair_rows]
                print(
                    f"[test][slices][z-enc] pairwise cosine min/mean/max="
                    f"{min(cos_vals):.3e}/{(sum(cos_vals)/len(cos_vals)):.3e}/{max(cos_vals):.3e}",
                    flush=True,
                )
            diag_payload["pairwise_z_enc_cosine"] = pair_rows
        diag_path = slice_root / "latent_stats.json"
        diag_path.write_text(json.dumps(diag_payload, indent=2))
        print(f"[test][slices] wrote latent diagnostics to {diag_path}", flush=True)
    if not saved_info:
        return
    meta_path = slice_root / "slices_meta.json"
    meta_path.write_text(json.dumps(saved_info, indent=2))
    print(
        f"[test][slices] saved {len(saved_info)} patient(s) slices under {slice_root.resolve()}",
        flush=True,
    )

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


def compute_poisson_rate(pred_raw: torch.Tensor, mode: str, eps: float = 1e-6) -> torch.Tensor:
    """Map raw projection output to a Poisson rate (lambda).

    Note: renderer already outputs a non-negative projection (proj_map). "identity" is
    the physically consistent mapping. "softplus_shift" preserves legacy behavior.
    """
    mode = str(mode or "softplus_shift")
    if mode == "softplus_shift":
        global _POISSON_RATE_LEGACY_WARNED
        if not _POISSON_RATE_LEGACY_WARNED:
            print(
                "[WARN] Poisson rate uses legacy softplus_shift on already-positive proj_map. "
                "Consider --poisson-rate-mode identity for physically consistent rates.",
                flush=True,
            )
            _POISSON_RATE_LEGACY_WARNED = True
        rate = F.softplus(pred_raw) - math.log(2.0)
        return rate.clamp_min(eps)
    if mode == "identity":
        return pred_raw.clamp_min(eps)
    raise ValueError(f"Unknown poisson_rate_mode: {mode}")


def apply_poisson_rate_floor(
    rate: torch.Tensor, floor: float, mode: str = "clamp"
) -> Tuple[torch.Tensor, Optional[float]]:
    floor = float(floor)
    if floor <= 0.0:
        return rate, None
    floor_t = torch.tensor(floor, device=rate.device, dtype=rate.dtype)
    floor_frac = float((rate < floor_t).float().mean().item())
    mode = str(mode or "clamp")
    if mode == "clamp":
        return torch.clamp(rate, min=floor_t), floor_frac
    if mode == "softplus_hinge":
        offset = rate.new_tensor(math.log(2.0))
        return floor_t + F.softplus(rate - floor_t) - offset, floor_frac
    raise ValueError(f"Unknown poisson_rate_floor_mode: {mode}")


def sqrt_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sqrt-MSE: ||sqrt(pred) - sqrt(target)||^2, stabilisiert via clamp."""
    pred_s = torch.sqrt(pred.clamp_min(eps))
    target_s = torch.sqrt(target.clamp_min(eps))
    diff2 = (pred_s - target_s) ** 2
    if weight is not None:
        diff2 = diff2 * weight
    return torch.mean(diff2)


def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    diff = pred - target
    abs_diff = torch.abs(diff)
    quad = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device, dtype=pred.dtype))
    lin = abs_diff - quad
    loss = 0.5 * quad * quad + delta * lin
    if weight is not None:
        loss = loss * weight
    return torch.mean(loss)


def _extract_meta_scalar(meta, key: str) -> Optional[float]:
    if not isinstance(meta, dict):
        return None
    val = meta.get(key)
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    if torch.is_tensor(val):
        if val.numel() == 0:
            return None
        return float(val.detach().view(-1)[0].item())
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def compute_joint_p99(ap: torch.Tensor, pa: torch.Tensor) -> torch.Tensor:
    """Joint p99 über AP+PA pro Batch-Sample (returns [B])."""
    if ap.dim() > 2:
        ap_flat = ap.reshape(ap.shape[0], -1)
    else:
        ap_flat = ap.unsqueeze(0)
    if pa.dim() > 2:
        pa_flat = pa.reshape(pa.shape[0], -1)
    else:
        pa_flat = pa.unsqueeze(0)
    joint = torch.cat([ap_flat, pa_flat], dim=1)
    return torch.quantile(joint.float(), 0.99, dim=1)


def compute_proj_scale(
    ap: torch.Tensor,
    pa: torch.Tensor,
    source: str,
    meta: Optional[dict] = None,
) -> torch.Tensor:
    """Bestimmt Skalenfaktor s pro Sample fuer Encoder-Inputs."""
    source = str(source or "none")
    B = ap.shape[0] if ap.dim() >= 3 else 1
    device = ap.device
    if source == "none":
        return torch.ones((B,), device=device)
    if source == "meta_p99":
        meta_scale = _extract_meta_scalar(meta, "proj_scale_joint_p99")
        if meta_scale is not None and math.isfinite(meta_scale) and meta_scale > 0:
            return torch.full((B,), float(meta_scale), device=device)
        # Fallback: compute p99 on the fly
        return compute_joint_p99(ap, pa)
    if source == "compute_p99":
        return compute_joint_p99(ap, pa)
    if source == "sumcounts":
        ap_sum = ap.reshape(B, -1).sum(dim=1)
        pa_sum = pa.reshape(B, -1).sum(dim=1)
        return ap_sum + pa_sum
    raise ValueError(f"Unknown proj_scale_source: {source}")


def apply_proj_transform(
    proj: torch.Tensor,
    scale: torch.Tensor,
    transform: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Skaliert proj mit s und wendet Transform an (log1p/sqrt/none)."""
    if scale is None:
        scale_t = None
    else:
        if torch.is_tensor(scale):
            scale_t = scale.to(device=proj.device, dtype=proj.dtype)
        else:
            scale_t = torch.tensor(scale, device=proj.device, dtype=proj.dtype)

    transform = str(transform or "none")
    if scale_t is None:
        scaled = proj
    else:
        while scale_t.dim() < proj.dim():
            scale_t = scale_t.view(-1, *([1] * (proj.dim() - 1)))
        scaled = proj / torch.clamp(scale_t, min=eps)
    scaled = torch.clamp(scaled, min=0.0)
    if transform == "log1p":
        return torch.log1p(scaled)
    if transform == "sqrt":
        return torch.sqrt(scaled + eps)
    if transform == "none":
        return scaled
    raise ValueError(f"Unknown encoder_proj_transform: {transform}")


def compute_act_norm_factor(
    act_vol: Optional[torch.Tensor],
    source: str,
    fixed_value: float,
    cached_global: Optional[float],
) -> Tuple[float, Optional[float]]:
    """Ermittelt Normierungsfaktor fuer ACT-Loss; gibt ggf. neuen globalen Cache zurueck."""
    source = str(source or "p99_scan")
    if source == "none":
        return 1.0, cached_global
    if source == "fixed":
        val = float(fixed_value)
        return (val if val > 0 else 1.0), cached_global
    if act_vol is None or act_vol.numel() == 0:
        return 1.0, cached_global
    def _approx_quantile(t: torch.Tensor, q: float, max_samples: int = 1_000_000) -> float:
        flat = t.reshape(-1)
        if flat.numel() <= max_samples:
            return float(torch.quantile(flat, q).item())
        # subsample to keep quantile fast/robust on large volumes
        idx = torch.randint(0, flat.numel(), (max_samples,), device=flat.device)
        sample = flat[idx]
        return float(torch.quantile(sample, q).item())

    if source == "p99_global":
        if cached_global is not None and cached_global > 0:
            return cached_global, cached_global
        p99 = _approx_quantile(act_vol.float(), 0.99)
        p99 = p99 if p99 > 0 else 1.0
        return p99, p99
    if source == "p99_scan":
        p99 = _approx_quantile(act_vol.float(), 0.99)
        return (p99 if p99 > 0 else 1.0), cached_global
    raise ValueError(f"Unknown act_norm_source: {source}")


def nonfinite_fraction(t: Optional[torch.Tensor]) -> float:
    if t is None or t.numel() == 0:
        return 0.0
    return float((~torch.isfinite(t)).float().mean().item())


def tensor_stats(t: Optional[torch.Tensor]) -> Optional[dict]:
    if t is None or t.numel() == 0:
        return None
    t = t.detach().float()
    flat = t.reshape(-1)
    # Quantile on very large tensors can error; subsample for robust stats.
    if flat.numel() > 1_000_000:
        idx = torch.randint(0, flat.numel(), (1_000_000,), device=flat.device)
        flat = flat[idx]
    return {
        "min": float(flat.min().item()),
        "mean": float(flat.mean().item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "max": float(flat.max().item()),
    }


def fmt_stats(stats: Optional[dict]) -> str:
    if stats is None:
        return "min/mean/p95/max=nan/nan/nan/nan"
    return (
        "min/mean/p95/max="
        f"{stats['min']:.3e}/{stats['mean']:.3e}/{stats['p95']:.3e}/{stats['max']:.3e}"
    )


def _ratio_stats(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float, float]:
    eps = 1e-8
    sum_pred = float(pred.sum().item())
    sum_target = float(target.sum().item())
    mean_pred = float(pred.mean().item())
    mean_target = float(target.mean().item())
    sum_ratio = sum_pred / (sum_target + eps)
    mean_ratio = mean_pred / (mean_target + eps)
    return sum_ratio, mean_ratio, sum_pred, sum_target


_ENCODER_INPUT_STATS_LOGGED = False
_ENCODER_PIPELINE_TRAIN_LOGGED = False
_ENCODER_PIPELINE_EVAL_LOGGED = False


def _encoder_input_min_mean_max(tensor: torch.Tensor) -> tuple[float, float, float]:
    if tensor is None or tensor.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    flat = tensor.detach().reshape(-1).float()
    return float(flat.min().item()), float(flat.mean().item()), float(flat.max().item())


def _log_encoder_pipeline_debug(
    phase: str,
    ap_enc_input: torch.Tensor,
    pa_enc_input: torch.Tensor,
    proj_transform: str,
    proj_scale_source: str,
):
    ap_stats = _encoder_input_min_mean_max(ap_enc_input)
    pa_stats = _encoder_input_min_mean_max(pa_enc_input)
    print(
        f"[debug][encoder-pipeline][{phase}] pre_build transform={proj_transform} "
        f"scale_source={proj_scale_source} "
        f"AP(min/mean/max)={ap_stats[0]:.3e}/{ap_stats[1]:.3e}/{ap_stats[2]:.3e} "
        f"PA(min/mean/max)={pa_stats[0]:.3e}/{pa_stats[1]:.3e}/{pa_stats[2]:.3e}",
        flush=True,
    )


def build_encoder_input(
    ap: torch.Tensor,
    pa: torch.Tensor,
    ct_vol: Optional[torch.Tensor],
    scale: torch.Tensor,
    transform: str,
    use_ct: bool,
) -> torch.Tensor:
    """Baut den Encoder-Input als [B,C,H,W] aus AP/PA (+optional CT)."""
    ap_enc = apply_proj_transform(ap, scale, transform)
    pa_enc = apply_proj_transform(pa, scale, transform)
    inputs = [ap_enc, pa_enc]
    if use_ct:
        if ct_vol is None or ct_vol.numel() == 0:
            # CT fehlt: Dummy-Channel mit 0
            zeros = torch.zeros_like(ap_enc)
            inputs.append(zeros)
        else:
            # ct_vol: [B,D,H,W] -> Mean-Projektion [B,1,H,W]
            if ct_vol.dim() == 3:
                ct = ct_vol.unsqueeze(0)
            else:
                ct = ct_vol
            ct_mean = ct.mean(dim=1, keepdim=True)
            # an AP/PA-Auflösung anpassen
            if ct_mean.shape[-2:] != ap_enc.shape[-2:]:
                ct_mean = F.interpolate(ct_mean, size=ap_enc.shape[-2:], mode="bilinear", align_corners=False)
            # einfache Standardisierung pro Sample
            ct_flat = ct_mean.reshape(ct_mean.shape[0], -1)
            ct_mu = ct_flat.mean(dim=1).view(-1, 1, 1, 1)
            ct_std = ct_flat.std(dim=1).view(-1, 1, 1, 1)
            ct_norm = (ct_mean - ct_mu) / (ct_std + 1e-6)
        inputs.append(ct_norm)
    global _ENCODER_INPUT_STATS_LOGGED
    if not _ENCODER_INPUT_STATS_LOGGED:
        _ENCODER_INPUT_STATS_LOGGED = True
        def _stats_min_mean_max(tensor: torch.Tensor) -> tuple[float, float, float]:
            if tensor.numel() == 0:
                return float("nan"), float("nan"), float("nan")
            flat = tensor.reshape(-1)
            return float(flat.min().item()), float(flat.mean().item()), float(flat.max().item())

        ap_stats = _stats_min_mean_max(ap_enc)
        pa_stats = _stats_min_mean_max(pa_enc)
        print(
            f"[DEBUG][encoder-input] transform={transform} "
            f"ap(min/mean/max)={ap_stats[0]:.3e}/{ap_stats[1]:.3e}/{ap_stats[2]:.3e} "
            f"| pa(min/mean/max)={pa_stats[0]:.3e}/{pa_stats[1]:.3e}/{pa_stats[2]:.3e}",
            flush=True,
        )
    return torch.cat(inputs, dim=1)


def build_hybrid_latent_from_batch(
    args,
    batch: dict,
    device: torch.device,
    z_latent_base: torch.Tensor,
    encoder: Optional[nn.Module],
    z_fuser: Optional[nn.Module],
    z_enc_alpha: float,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build per-batch hybrid latent: z = z_base + alpha * z_enc_proj."""
    if encoder is None:
        return z_latent_base.detach(), None, None
    ap = batch.get("ap")
    pa = batch.get("pa")
    if ap is None or pa is None:
        raise RuntimeError("Batch missing AP/PA for hybrid latent build.")
    ap = ap.to(device, non_blocking=True).float()
    pa = pa.to(device, non_blocking=True).float()
    ap_enc_input = batch.get("proj_input_ap", ap)
    pa_enc_input = batch.get("proj_input_pa", pa)
    if ap_enc_input is not ap:
        ap_enc_input = ap_enc_input.to(device, non_blocking=True).float()
    if pa_enc_input is not pa:
        pa_enc_input = pa_enc_input.to(device, non_blocking=True).float()
    ct_vol = batch.get("ct")
    if ct_vol is not None and ct_vol.numel() > 0:
        ct_vol = ct_vol.to(device, non_blocking=True).float()
    else:
        ct_vol = None
    with torch.no_grad():
        proj_scale_enc = compute_proj_scale(ap, pa, args.proj_scale_source, batch.get("meta"))
        proj_scale_enc = torch.clamp(proj_scale_enc, min=1e-6)
        global _ENCODER_PIPELINE_EVAL_LOGGED
        if bool(getattr(args, "debug_latent_stats", False)) and (not _ENCODER_PIPELINE_EVAL_LOGGED):
            _ENCODER_PIPELINE_EVAL_LOGGED = True
            _log_encoder_pipeline_debug(
                "val/test",
                ap_enc_input,
                pa_enc_input,
                args.encoder_proj_transform,
                args.proj_scale_source,
            )
        enc_input = build_encoder_input(
            ap_enc_input,
            pa_enc_input,
            ct_vol,
            proj_scale_enc,
            args.encoder_proj_transform,
            args.encoder_use_ct,
        )
        z_enc = encoder(enc_input)
        z_enc_proj = z_fuser(z_enc) if z_fuser is not None else z_enc
        z_base = z_latent_base
        if z_base.shape[0] != z_enc_proj.shape[0]:
            z_base = z_base.expand(z_enc_proj.shape[0], -1)
        z_latent = z_base + (float(z_enc_alpha) * z_enc_proj)
    return z_latent.detach(), z_enc.detach(), z_enc_proj.detach()


def build_hwfr_from_config(data_cfg: dict) -> list:
    """Fallback HWFR fuer Smoke-Tests ohne Datenzugriff."""
    imsize = data_cfg.get("imsize") or data_cfg.get("H") or 128
    H = int(imsize)
    W = int(data_cfg.get("W") or H)
    fov = float(data_cfg.get("fov", 60.0))
    focal = W / 2.0 * 1.0 / np.tan(0.5 * fov * np.pi / 180.0)
    radius = data_cfg.get("radius", 0.5)
    render_radius = radius
    if isinstance(radius, str):
        radius = tuple(float(r) for r in radius.split(","))
        render_radius = max(radius)
    return [H, W, focal, render_radius]


def build_synthetic_batch(
    H: int,
    W: int,
    device: torch.device,
    with_ct: bool = True,
    with_act: bool = True,
) -> dict:
    """Erzeugt ein synthetisches Batch fuer Smoke-Tests (ohne I/O)."""
    B = 1
    ap = torch.rand((B, 1, H, W), device=device) * 5.0
    pa = torch.rand((B, 1, H, W), device=device) * 5.0
    ap_counts = ap * 1000.0
    pa_counts = pa * 1000.0
    D = int(min(32, H))
    ct = torch.rand((D, H, W), device=device) if with_ct else torch.empty(0, device=device)
    act = torch.rand((D, H, W), device=device) if with_act else torch.empty(0, device=device)
    meta = {"proj_scale_joint_p99": float(torch.quantile(torch.cat([ap.reshape(-1), pa.reshape(-1)]), 0.99).item())}
    return {
        "ap": ap,
        "pa": pa,
        "ap_counts": ap_counts,
        "pa_counts": pa_counts,
        "ct": ct,
        "act": act,
        "meta": meta,
    }


def compute_lambda_and_attenuation_stats(
    extras_list,
    atten_scale: float,
    clamp_max: float = 60.0,
) -> Tuple[Optional[dict], Optional[dict], Optional[dict], Optional[float], Optional[float], float, float]:
    """Aggregiert lambda/attenuation-Stats aus Extras (raw/mu/dists)."""
    lambda_vals = []
    atten_vals = []
    mu_vals = []
    for extras in extras_list:
        if not isinstance(extras, dict):
            continue
        raw_out = extras.get("raw")
        if raw_out is None:
            continue
        lambda_vals.append(F.softplus(raw_out[..., 0]))
        mu = extras.get("mu")
        dists = extras.get("dists")
        if mu is None or dists is None:
            continue
        if mu.shape != dists.shape:
            continue
        mu = torch.clamp(mu, min=0.0)
        mu_vals.append(mu)
        mu_dists = mu * dists
        attenuation = torch.cumsum(mu_dists, dim=-1) * float(atten_scale)
        attenuation = F.pad(attenuation[..., :-1], (1, 0), mode="constant", value=0.0)
        attenuation = torch.clamp(attenuation, min=0.0, max=clamp_max)
        atten_vals.append(attenuation)
    lambda_stats = None
    atten_stats = None
    frac_gt20 = None
    frac_clamp = None
    if lambda_vals:
        lambda_all = torch.cat([lv.reshape(-1) for lv in lambda_vals], dim=0)
        lambda_stats = tensor_stats(lambda_all)
    mu_stats = None
    if mu_vals:
        mu_all = torch.cat([mv.reshape(-1) for mv in mu_vals], dim=0)
        mu_stats = tensor_stats(mu_all)
    if atten_vals:
        atten_all = torch.cat([av.reshape(-1) for av in atten_vals], dim=0)
        atten_stats = tensor_stats(atten_all)
        frac_gt20 = float((atten_all > 20.0).float().mean().item())
        frac_clamp = float((atten_all >= clamp_max - 1e-6).float().mean().item())
    if lambda_vals:
        lambda_flat = torch.cat([lv.reshape(-1) for lv in lambda_vals], dim=0)
        nonfinite_lambda = nonfinite_fraction(lambda_flat)
    else:
        nonfinite_lambda = 0.0
    if atten_vals:
        atten_flat = torch.cat([av.reshape(-1) for av in atten_vals], dim=0)
        nonfinite_atten = nonfinite_fraction(atten_flat)
    else:
        nonfinite_atten = 0.0
    return lambda_stats, mu_stats, atten_stats, frac_gt20, frac_clamp, nonfinite_lambda, nonfinite_atten


def log_attenuation_sanity(
    step: int,
    extras_list,
    atten_scale: float,
    label: str = "train",
    trans_near0: float = 1e-4,
    trans_near1: float = 0.999,
):
    """Print attenuation/unit sanity stats based on render extras."""
    dists_vals = []
    sum_dists_vals = []
    ray_norm_vals = []
    near_vals = []
    far_vals = []
    mu_vals = []
    atten_vals = []
    trans_vals = []
    n_samples = None
    for extras in extras_list:
        if not isinstance(extras, dict):
            continue
        dists = extras.get("dists")
        if dists is None:
            continue
        if dists.dim() == 3 and dists.shape[-1] == 1:
            dists = dists.squeeze(-1)
        if n_samples is None:
            n_samples = int(dists.shape[-1])
        dists_vals.append(dists.reshape(-1))
        sum_dists_vals.append(dists.sum(dim=-1).reshape(-1))
        ray_norm = extras.get("ray_norm")
        near = extras.get("near")
        far = extras.get("far")
        if ray_norm is not None:
            ray_norm_vals.append(ray_norm.reshape(-1))
        if near is not None:
            near_vals.append(near.reshape(-1))
        if far is not None:
            far_vals.append(far.reshape(-1))
        mu = extras.get("mu")
        if mu is not None:
            if mu.dim() == 3 and mu.shape[-1] == 1:
                mu = mu.squeeze(-1)
            if mu.shape == dists.shape:
                mu_vals.append(mu.reshape(-1))
                attenuation = extras.get("attenuation")
                transmission = extras.get("transmission")
                if attenuation is None:
                    mu_clamped = torch.clamp(mu, min=0.0)
                    mu_dists = mu_clamped * dists
                    attenuation = torch.cumsum(mu_dists, dim=-1) * float(atten_scale)
                    attenuation = F.pad(attenuation[..., :-1], (1, 0), mode="constant", value=0.0)
                    attenuation = torch.clamp(attenuation, min=0.0, max=60.0)
                if transmission is None:
                    transmission = torch.exp(-attenuation)
                atten_vals.append(attenuation.reshape(-1))
                trans_vals.append(transmission.reshape(-1))
    if not dists_vals:
        print(f"[sanity][{label}][step {step}] no dists available for attenuation stats.", flush=True)
        return

    dists_all = torch.cat(dists_vals, dim=0)
    sum_dists_all = torch.cat(sum_dists_vals, dim=0)
    dists_stats = tensor_stats(dists_all)
    sum_stats = tensor_stats(sum_dists_all)
    ray_norm_stats = tensor_stats(torch.cat(ray_norm_vals, dim=0)) if ray_norm_vals else None
    near_stats = tensor_stats(torch.cat(near_vals, dim=0)) if near_vals else None
    far_stats = tensor_stats(torch.cat(far_vals, dim=0)) if far_vals else None
    mu_stats = tensor_stats(torch.cat(mu_vals, dim=0)) if mu_vals else None
    atten_stats = tensor_stats(torch.cat(atten_vals, dim=0)) if atten_vals else None
    trans_stats = tensor_stats(torch.cat(trans_vals, dim=0)) if trans_vals else None
    trans_near0_frac = (
        float((torch.cat(trans_vals, dim=0) < trans_near0).float().mean().item()) if trans_vals else float("nan")
    )
    trans_near1_frac = (
        float((torch.cat(trans_vals, dim=0) > trans_near1).float().mean().item()) if trans_vals else float("nan")
    )
    sum_mean = float(sum_stats["mean"]) if sum_stats is not None else float("nan")
    sum_cm_est = sum_mean * float(atten_scale) if math.isfinite(sum_mean) else float("nan")
    print(
        f"[sanity][{label}][step {step}] dists(min/mean/p95/max)={fmt_stats(dists_stats)} "
        f"| sum_dists(min/mean/p95/max)={fmt_stats(sum_stats)} | sum_dists_cm_est={sum_cm_est:.3e} "
        f"| ||d||(min/mean/p95/max)={fmt_stats(ray_norm_stats)} "
        f"| near(min/mean/p95/max)={fmt_stats(near_stats)} | far(min/mean/p95/max)={fmt_stats(far_stats)} "
        f"| mu(min/mean/p95/max)={fmt_stats(mu_stats)} | atten(min/mean/p95/max)={fmt_stats(atten_stats)} "
        f"| T(min/mean/p95/max)={fmt_stats(trans_stats)} | T<={trans_near0:.1e}={trans_near0_frac:.3f} "
        f"| T>={trans_near1:.3f}={trans_near1_frac:.3f}",
        flush=True,
    )
    if ray_norm_stats is not None:
        ray_norm_mean = float(ray_norm_stats["mean"])
        if abs(ray_norm_mean - 1.0) > 0.1:
            print(
                f"[sanity][{label}] WARN: ||rays_d|| mean deviates from 1.0 (mean={ray_norm_mean:.3f}).",
                flush=True,
            )
    if ray_norm_vals and near_vals and far_vals:
        ray_norm_all = torch.cat(ray_norm_vals, dim=0)
        near_all = torch.cat(near_vals, dim=0)
        far_all = torch.cat(far_vals, dim=0)
        if dists_all.numel() > 0 and ray_norm_all.numel() == near_all.numel() == far_all.numel():
            expected = (far_all - near_all) * ray_norm_all
            if n_samples is not None and n_samples > 1:
                # dists includes a duplicated last segment -> expected sum is scaled by N/(N-1)
                expected = expected * (float(n_samples) / float(n_samples - 1))
            expected_mean = float(expected.mean().item())
            if math.isfinite(expected_mean) and expected_mean > 0:
                ratio = sum_mean / expected_mean
                if abs(ratio - 1.0) > 0.1:
                    print(
                        f"[sanity][{label}] WARN: sum_dists mean != (far-near)*||d|| "
                        f"(ratio={ratio:.3f}, expected_mean={expected_mean:.3e}).",
                        flush=True,
                    )
    if sum_mean <= 0 or (math.isfinite(sum_mean) and sum_mean < 1e-6):
        print(f"[sanity][{label}] WARN: sum_dists mean is very small ({sum_mean:.3e}).", flush=True)
    if trans_vals and (trans_near1_frac > 0.99):
        print(f"[sanity][{label}] WARN: transmission ~1 for most samples (frac {trans_near1_frac:.3f}).", flush=True)
    if trans_vals and (trans_near0_frac > 0.5):
        print(f"[sanity][{label}] WARN: transmission ~0 for many samples (frac {trans_near0_frac:.3f}).", flush=True)


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


def grad_norm_of_module(loss_term: torch.Tensor, module: Optional[nn.Module]) -> float:
    if module is None:
        return 0.0
    params = [p for p in module.parameters() if p.requires_grad]
    if not params:
        return 0.0
    return grad_norm_of(loss_term, params)


def global_grad_norm(params) -> float:
    """L2-Norm ueber alle vorhandenen Gradienten (logging only)."""
    total = 0.0
    for p in params:
        if p is None or p.grad is None:
            continue
        g = p.grad.detach()
        total += float(g.norm().item()) ** 2
    return math.sqrt(total) if total > 0.0 else 0.0


def module_grad_norm(module: Optional[nn.Module]) -> float:
    if module is None:
        return 0.0
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(g.norm().item()) ** 2
    return math.sqrt(total) if total > 0.0 else 0.0


def module_grad_mean_abs(module: Optional[nn.Module]) -> float:
    if module is None:
        return 0.0
    vals = [p.grad.detach().abs().mean() for p in module.parameters() if p.grad is not None]
    if not vals:
        return 0.0
    return float(torch.stack(vals).mean().item())


def params_grad_norm(params: List[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(g.norm().item()) ** 2
    return math.sqrt(total) if total > 0.0 else 0.0


def sanitize_proj_weight_bounds(weight_min: float, weight_max: float) -> tuple[float, float, bool]:
    w_min = float(weight_min)
    w_max = float(weight_max)
    swapped = False
    if w_min > w_max:
        w_min, w_max = w_max, w_min
        swapped = True
    return w_min, w_max, swapped


def compute_proj_weight_schedule(
    step: int,
    warmup_steps: int,
    ramp_steps: int,
    weight_min: float,
    weight_max: float,
) -> tuple[float, bool, bool, float]:
    """Deterministische Projection-Weight-Schedule (1-indexed step)."""
    step_i = int(step)
    warmup = max(0, int(warmup_steps))
    ramp = max(0, int(ramp_steps))
    w_min = float(weight_min)
    w_max = float(weight_max)

    if step_i <= warmup:
        return 0.0, False, True, 0.0

    if ramp == 0:
        ramp_t = 1.0
    else:
        ramp_t = (step_i - warmup) / float(ramp)
        ramp_t = min(1.0, max(0.0, ramp_t))
    weight = w_min + ramp_t * (w_max - w_min)
    return float(weight), True, False, float(ramp_t)


def _self_test_proj_weight_schedule() -> None:
    w, active, warmup, t = compute_proj_weight_schedule(
        step=10,
        warmup_steps=10,
        ramp_steps=100,
        weight_min=5e-4,
        weight_max=5e-3,
    )
    assert w == 0.0 and (not active) and warmup and t == 0.0
    w, active, warmup, t = compute_proj_weight_schedule(
        step=11,
        warmup_steps=10,
        ramp_steps=100,
        weight_min=5e-4,
        weight_max=5e-3,
    )
    assert active and (not warmup) and 0.0 < t < 1.0 and (5e-4 < w < 5e-3)
    w, active, warmup, t = compute_proj_weight_schedule(
        step=111,
        warmup_steps=10,
        ramp_steps=100,
        weight_min=5e-4,
        weight_max=5e-3,
    )
    assert active and (not warmup) and t == 1.0 and abs(w - 5e-3) < 1e-12
    w, active, warmup, t = compute_proj_weight_schedule(
        step=11,
        warmup_steps=10,
        ramp_steps=0,
        weight_min=5e-4,
        weight_max=5e-3,
    )
    assert active and (not warmup) and t == 1.0 and abs(w - 5e-3) < 1e-12


def _log_latent_conditioning_stats(
    step: int,
    z_base: torch.Tensor,
    z_enc: Optional[torch.Tensor],
    z_enc_proj: Optional[torch.Tensor],
    z_latent: torch.Tensor,
    alpha: float,
) -> None:
    """Logs latent-mix diagnostics for hybrid conditioning."""
    if z_enc is None or z_enc_proj is None:
        return
    with torch.no_grad():
        eps = 1e-8
        delta = z_latent - z_base
        alpha_e = float(alpha) * z_enc_proj

        def _ms(v: torch.Tensor) -> tuple[float, float]:
            if v.numel() == 0:
                return float("nan"), float("nan")
            return float(v.mean().item()), float(v.std(unbiased=False).item())

        n_z_enc = z_enc.norm(dim=1)
        n_e = z_enc_proj.norm(dim=1)
        n_alpha_e = alpha_e.norm(dim=1)
        n_delta = delta.norm(dim=1)
        n_delta_enc = (z_latent - z_enc).norm(dim=1)

        cos_enc_final = F.cosine_similarity(z_enc, z_latent, dim=1, eps=eps)
        cos_proj_final = F.cosine_similarity(z_enc_proj, z_latent, dim=1, eps=eps)
        ratio_alpha_e = n_alpha_e / (n_z_enc + eps)

        m_nz_enc, s_nz_enc = _ms(n_z_enc)
        m_ne, s_ne = _ms(n_e)
        m_nae, s_nae = _ms(n_alpha_e)
        m_nd, s_nd = _ms(n_delta)
        m_nde, s_nde = _ms(n_delta_enc)
        m_c1, s_c1 = _ms(cos_enc_final)
        m_c2, s_c2 = _ms(cos_proj_final)
        m_r, s_r = _ms(ratio_alpha_e)

        print(
            f"[latent][step {step:05d}] "
            f"||z_enc||(m/std)={m_nz_enc:.3e}/{s_nz_enc:.3e} "
            f"||E||(m/std)={m_ne:.3e}/{s_ne:.3e} "
            f"||alphaE||(m/std)={m_nae:.3e}/{s_nae:.3e} "
            f"||z_final-z_base||(m/std)={m_nd:.3e}/{s_nd:.3e} "
            f"||z_final-z_enc||(m/std)={m_nde:.3e}/{s_nde:.3e} "
            f"cos(z_enc,z_final)={m_c1:.3e}±{s_c1:.3e} "
            f"cos(E,z_final)={m_c2:.3e}±{s_c2:.3e} "
            f"ratio||alphaE||/||z_enc||={m_r:.3e}±{s_r:.3e}",
            flush=True,
        )


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
        f"| orthographic={data_cfg.get('orthographic')} | radius={data_cfg.get('radius')} "
        f"| use_anisotropic_radius={data_cfg.get('use_anisotropic_radius')} "
        f"| radius_xyz_cm={data_cfg.get('radius_xyz_cm')}",
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
        f"| act_loss_weight={args.act_loss_weight} | act_samples={args.act_samples} "
        f"| act_pos_weight={args.act_pos_weight} | act_pos_fraction={args.act_pos_fraction} "
        f"| act_pos_threshold={args.act_pos_threshold} "
        f"| ct_loss_weight={args.ct_loss_weight} | ct_threshold={args.ct_threshold} "
        f"| ray_tv_weight={args.ray_tv_weight} | ray_tv_edge_aware={args.ray_tv_edge_aware} | ray_tv_alpha={args.ray_tv_alpha} "
        f"| ray_tv_w_clamp_min={args.ray_tv_w_clamp_min} | lambda_ray_tv_weight={args.lambda_ray_tv_weight} "
        f"| ct_padding_mode={args.ct_padding_mode} "
        f"| bg_depth_mass_weight={args.bg_depth_mass_weight} | bg_depth_eps={args.bg_depth_eps} | bg_depth_mode={args.bg_depth_mode} "
        f"| poisson_rate_mode={args.poisson_rate_mode} | poisson_rate_floor={args.poisson_rate_floor} "
        f"| poisson_rate_floor_mode={args.poisson_rate_floor_mode} "
        f"| grad_clip_enabled={args.grad_clip_enabled} | grad_clip_max_norm={args.grad_clip_max_norm} "
        f"| clip_grad_decoder={args.clip_grad_decoder} | debug_grad_terms_every={args.debug_grad_terms_every} "
        f"| debug_sanity_checks={bool(getattr(args, 'debug_sanity_checks', False))}",
        flush=True,
    )
    if getattr(args, "hybrid", False):
        print(
            f"[cfg][hybrid] proj_loss_type={args.proj_loss_type} | proj_loss_weight={args.proj_loss_weight} "
            f"| proj_warmup_steps={args.proj_warmup_steps} | proj_weight_min={args.proj_weight_min} "
            f"| proj_ramp_steps={args.proj_ramp_steps} | proj_target_source={args.proj_target_source} "
            f"| proj_gain_source={args.proj_gain_source} | gain_reg_weight={args.gain_reg_weight} "
            f"| gain_reg_scale={args.gain_reg_scale} "
            f"| gain_prior_mode={args.gain_prior_mode} | gain_prior_value={args.gain_prior_value} "
            f"| gain_clamp_min={args.gain_clamp_min} | gain_clamp_max={args.gain_clamp_max} "
            f"| encoder_proj_transform={args.encoder_proj_transform} "
            f"| proj_scale_source={args.proj_scale_source} | act_norm_source={args.act_norm_source} "
            f"| act_norm_value={args.act_norm_value} | encoder_use_ct={args.encoder_use_ct} "
            f"| z_enc_alpha={args.z_enc_alpha} | debug_latent_stats={args.debug_latent_stats} "
            f"| debug_latent_stats_every={args.debug_latent_stats_every}",
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


def render_minibatch(
    generator,
    z_latent,
    rays_subset,
    ct_context=None,
    return_raw: bool = False,
    debug_sanity_checks: bool = False,
):
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
    if debug_sanity_checks:
        render_kwargs["debug_sanity_checks"] = True
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


def _projection_orientation_stats(arr: np.ndarray) -> Dict[str, Any]:
    img = np.asarray(arr, dtype=np.float64)
    if img.ndim != 2:
        img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D projection array, got shape={img.shape}")

    finite = np.isfinite(img)
    payload: Dict[str, Any] = {
        "shape": [int(img.shape[0]), int(img.shape[1])],
        "finite_frac": float(finite.mean()) if finite.size > 0 else 0.0,
    }
    if not finite.any():
        payload.update(
            {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "sum": float("nan"),
                "argmax_yx": None,
                "com_yx": None,
                "sha1_f32": None,
            }
        )
        return payload

    finite_vals = img[finite]
    payload.update(
        {
            "min": float(np.min(finite_vals)),
            "max": float(np.max(finite_vals)),
            "mean": float(np.mean(finite_vals)),
            "sum": float(np.sum(finite_vals)),
        }
    )

    argmax_idx = int(np.nanargmax(np.where(finite, img, -np.inf)))
    argmax_y, argmax_x = np.unravel_index(argmax_idx, img.shape)
    payload["argmax_yx"] = [int(argmax_y), int(argmax_x)]

    weights = np.where(finite, np.clip(img, a_min=0.0, a_max=None), 0.0)
    wsum = float(weights.sum())
    if wsum > 0.0:
        yy, xx = np.indices(img.shape)
        com_y = float((yy * weights).sum() / wsum)
        com_x = float((xx * weights).sum() / wsum)
        payload["com_yx"] = [com_y, com_x]
    else:
        payload["com_yx"] = None

    digest_arr = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    payload["sha1_f32"] = hashlib.sha1(digest_arr.tobytes()).hexdigest()
    return payload


def maybe_render_preview(
    step,
    args,
    generator,
    z_eval,
    outdir,
    ct_volume=None,
    act_volume=None,
    ct_context=None,
    target_ap=None,
    target_pa=None,
    target_ap_counts=None,
    target_pa_counts=None,
    patient_id: Optional[str] = None,
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
    def _format_phantom_label(pid: Optional[str]) -> str:
        if pid is None:
            return "Phantom ID n/a"
        pid_s = str(pid)
        m = re.search(r"(\d+)", pid_s)
        if m is None:
            return f"Phantom ID {pid_s}"
        try:
            return f"Phantom ID {int(m.group(1))}"
        except Exception:
            return f"Phantom ID {m.group(1)}"

    phantom_label = _format_phantom_label(patient_id)

    def _proj_to_img(tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if tensor is None:
            return None
        t = tensor.detach()
        if t.numel() == H * W:
            t = t.view(H, W)
        elif t.dim() >= 2 and t.shape[-2:] == (H, W):
            t = t.reshape(-1, H, W)[0]
        else:
            return None
        return t.cpu().numpy()

    def _choose_vmax(reference: Optional[np.ndarray], fallback: float) -> float:
        if reference is not None:
            finite = np.isfinite(reference)
            if finite.any():
                try:
                    val = float(np.quantile(reference[finite], 0.995))
                except Exception:
                    val = float(np.nanmax(reference[finite]))
                if np.isfinite(val) and val > 0.0:
                    return max(val, fallback, 1e-6)
        if np.isfinite(fallback) and fallback > 0.0:
            return max(fallback, 1e-6)
        return 1e-6

    def _first_not_none(*values):
        for v in values:
            if v is not None:
                return v
        return None

    ap_target_img = _first_not_none(_proj_to_img(target_ap_counts), _proj_to_img(target_ap))
    pa_target_img = _first_not_none(_proj_to_img(target_pa_counts), _proj_to_img(target_pa))

    def _save_view(pred_img: np.ndarray, target_img: Optional[np.ndarray], view: str):
        vmax = _choose_vmax(target_img, float(np.nanmax(pred_img)))
        # keep colorscale anchored to the target counts, never expanded by pred max
        save_img_abs(
            pred_img,
            out_dir / f"step_{step:05d}_{view}.png",
            step=step,
            view=view,
            patient_id=patient_id,
            vmax=vmax,
            allow_arr_max=False,
            cbar_label="counts",
            header_text=f"pred | {view} | step {step} | {phantom_label}",
            include_data_stats=False,
        )
        if target_img is None or target_img.shape != pred_img.shape:
            return
        diff = pred_img - target_img
        diff_vmax = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 0.0
        diff_vmax = max(diff_vmax, 1e-6)
        save_img_abs(
            diff,
            out_dir / f"step_{step:05d}_{view}_diff.png",
            title="pred-target",
            step=step,
            view=f"{view} diff",
            patient_id=patient_id,
            vmin=-diff_vmax,
            vmax=diff_vmax,
            cbar_label="counts",
            cmap="RdBu_r",
            header_text=f"pred-target | {view} diff | step {step} | {phantom_label}",
            include_data_stats=False,
        )

    _save_view(ap_np, ap_target_img, "AP")
    _save_view(pa_np, pa_target_img, "PA")
    if getattr(args, "debug_proj_orientation", False):
        orientation_payload = {
            "step": int(step),
            "patient_id": None if patient_id is None else str(patient_id),
            "pred": {
                "ap": _projection_orientation_stats(ap_np),
                "pa": _projection_orientation_stats(pa_np),
            },
            "target": {
                "ap": None if ap_target_img is None else _projection_orientation_stats(ap_target_img),
                "pa": None if pa_target_img is None else _projection_orientation_stats(pa_target_img),
            },
        }
        orientation_path = out_dir / f"step_{step:05d}_orientation.json"
        with orientation_path.open("w", encoding="utf-8") as f:
            json.dump(orientation_payload, f, indent=2)
        print(f"[debug][proj-orientation] saved {orientation_path.resolve()}", flush=True)
    save_depth_profile(
        step,
        generator,
        z_eval,
        ct_volume,
        act_volume,
        out_dir,
        proj_ap=proj_ap,
        proj_pa=proj_pa,
        target_ap=target_ap,
        target_pa=target_pa,
        target_ap_counts=target_ap_counts,
        target_pa_counts=target_pa_counts,
        signal_quantile=args.depth_profile_signal_quantile,
        bg_quantile=args.depth_profile_bg_quantile,
        seed=args.depth_profile_seed,
    )
    print("🖼️ Preview gespeichert:", flush=True)
    print("   ", (out_dir / f"step_{step:05d}_AP.png").resolve(), flush=True)
    print("   ", (out_dir / f"step_{step:05d}_PA.png").resolve(), flush=True)


def init_log_file(path: Path) -> Path:
    # CSV-Header nur einmal schreiben; bei Header-Mismatch neuen Log erstellen.
    header = [
        "step",
        "loss",
        "loss_ap",
        "loss_pa",
        "loss_act",
        "loss_sparsity",
        "loss_act_tv",
        "loss_ct",
        "ray_tv",
        "ray_tv_w",
        "lambda_ray_tv",
        "lambda_floor_frac",
        "lambda_mean",
        "lambda_p95",
        "lambda_eff_mean",
        "lambda_eff_p95",
        "bg_depth_mass",
        "bg_depth_mass_w",
        "bg_depth_frac",
        "loss_tv",
        "mae_ap",
        "mae_pa",
        "rmse_ap",
        "rmse_pa",
        "nll_ap",
        "nll_pa",
        "dev_ap",
        "dev_pa",
        "pred_mean_ap",
        "pred_mean_pa",
        "pred_std_ap",
        "pred_std_pa",
        "val_loss",
        "val_loss_ap",
        "val_loss_pa",
        "val_mae",
        "val_rmse",
        "val_nll",
        "val_dev",
        "val_mae_ap",
        "val_mae_pa",
        "val_rmse_ap",
        "val_rmse_pa",
        "val_nll_ap",
        "val_nll_pa",
        "val_dev_ap",
        "val_dev_pa",
        "val_loss_fg",
        "val_mae_fg",
        "val_rmse_fg",
        "val_nll_fg",
        "val_dev_fg",
        "val_loss_top10",
        "val_mae_top10",
        "val_rmse_top10",
        "val_nll_top10",
        "val_dev_top10",
        "val_loss_bg",
        "val_mae_bg",
        "val_rmse_bg",
        "val_nll_bg",
        "val_dev_bg",
        "loss_test_all",
        "loss_test_ap",
        "loss_test_pa",
        "mae_test_all",
        "mae_test_ap",
        "mae_test_pa",
        "rmse_test_all",
        "rmse_test_ap",
        "rmse_test_pa",
        "nll_test_all",
        "nll_test_ap",
        "nll_test_pa",
        "dev_test_all",
        "dev_test_ap",
        "dev_test_pa",
        "iter_ms",
        "lr",
        "ray_tv_mode",
        "ray_tv_w_mean",
    ]
    if path.exists():
        existing_header = None
        try:
            with path.open("r", newline="") as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
        except Exception:
            existing_header = None
        if existing_header == header:
            return path
        # Header mismatch -> write to a new versioned log file.
        base = path.with_name(f"{path.stem}_v2{path.suffix}")
        candidate = base
        v = 2
        while candidate.exists():
            v += 1
            candidate = path.with_name(f"{path.stem}_v{v}{path.suffix}")
        print(
            f"[log][warn] train_log header mismatch; using new log file: {candidate}",
            flush=True,
        )
        path = candidate
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    return path


def append_log(path: Path, row):
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def init_hybrid_log_file(path: Path):
    if path.exists():
        try:
            header = path.read_text().splitlines()[0]
            if "gain" not in header or "mu_min" not in header:
                print(
                    f"[hybrid][warn] existing hybrid_stats.csv has old header; "
                    f"consider deleting {path} to get new columns.",
                    flush=True,
                )
        except Exception:
            pass
        return
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "proj_weight",
                "loss_proj",
                "loss_ap",
                "loss_pa",
                "loss_act",
                "loss_gain",
                "gain_prior",
                "act_norm_factor",
                "loss_total",
                "proj_scale_enc",
                "target_ap_min",
                "target_ap_mean",
                "target_ap_p95",
                "target_ap_max",
                "target_pa_min",
                "target_pa_mean",
                "target_pa_p95",
                "target_pa_max",
                "pred_ap_min",
                "pred_ap_mean",
                "pred_ap_p95",
                "pred_ap_max",
                "pred_pa_min",
                "pred_pa_mean",
                "pred_pa_p95",
                "pred_pa_max",
                "lambda_ray_min",
                "lambda_ray_mean",
                "lambda_ray_p95",
                "lambda_ray_max",
                "mu_min",
                "mu_mean",
                "mu_p95",
                "mu_max",
                "atten_min",
                "atten_mean",
                "atten_p95",
                "atten_max",
                "atten_frac_gt20",
                "atten_frac_clamp60",
                "gain",
                "nonfinite_pred",
                "nonfinite_lambda",
                "nonfinite_atten",
                "grad_norm_global",
                "grad_norm_gen",
                "clip_event",
                "z_enc_l2",
                "z_latent_l2",
            ]
        )


def append_hybrid_log(path: Path, row):
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_checkpoint(step, generator, optimizer, scaler, ckpt_dir: Path, encoder=None, z_fuser=None, gain_head=None, gain_param=None):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Minimal-Checkpoint: coarse/fine Netze, Optimizer, AMP-Scaler
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "generator_coarse": generator.render_kwargs_train["network_fn"].state_dict(),
        "generator_fine": None,
    }
    if generator.render_kwargs_train["network_fine"] is not None:
        state["generator_fine"] = generator.render_kwargs_train["network_fine"].state_dict()
    if encoder is not None:
        state["encoder"] = encoder.state_dict()
    if z_fuser is not None:
        state["z_fuser"] = z_fuser.state_dict()
    if gain_head is not None:
        state["gain_head"] = gain_head.state_dict()
    if gain_param is not None:
        state["gain_param"] = gain_param.detach().cpu()
    ckpt_path = ckpt_dir / f"checkpoint_step{step:05d}.pt"
    torch.save(state, ckpt_path)
    print(f"💾 Checkpoint gespeichert: {ckpt_path}", flush=True)


def load_checkpoint(
    ckpt_path: Path,
    *,
    device: torch.device,
    generator,
    optimizer=None,
    scaler=None,
    encoder: Optional[nn.Module] = None,
    z_fuser: Optional[nn.Module] = None,
    gain_head: Optional[nn.Module] = None,
    gain_param: Optional[nn.Parameter] = None,
    strict: bool = True,
    load_optimizer: bool = False,
) -> int:
    if ckpt_path is None:
        return 0
    ckpt_file = Path(ckpt_path).expanduser().resolve()
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
    state = torch.load(ckpt_file, map_location=device)
    step = int(state.get("step", 0))

    coarse = state.get("generator_coarse")
    if coarse is None:
        raise KeyError(f"Checkpoint missing key 'generator_coarse': {ckpt_file}")
    generator.render_kwargs_train["network_fn"].load_state_dict(coarse, strict=bool(strict))

    fine_net = generator.render_kwargs_train.get("network_fine")
    fine_state = state.get("generator_fine")
    if fine_net is not None and fine_state is not None:
        fine_net.load_state_dict(fine_state, strict=bool(strict))

    if encoder is not None and "encoder" in state:
        encoder.load_state_dict(state["encoder"], strict=bool(strict))
    if z_fuser is not None and "z_fuser" in state:
        z_fuser.load_state_dict(state["z_fuser"], strict=bool(strict))
    if gain_head is not None and "gain_head" in state:
        gain_head.load_state_dict(state["gain_head"], strict=bool(strict))
    if gain_param is not None and "gain_param" in state:
        with torch.no_grad():
            gain_param.copy_(state["gain_param"].to(device=device, dtype=gain_param.dtype))

    if load_optimizer:
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and "scaler" in state:
            scaler.load_state_dict(state["scaler"])

    print(
        f"[ckpt] loaded {ckpt_file} | step={step} | strict={bool(strict)} | load_optimizer={bool(load_optimizer)}",
        flush=True,
    )
    return step


def compute_projection_metrics(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> dict:
    """Forward-only projection metrics for counts (MAE/RMSE) and Poisson diagnostics."""

    pred_clamped = torch.clamp(pred, min=eps)
    target_clamped = torch.clamp(target, min=eps)
    diff = pred - target
    mae = float(torch.mean(torch.abs(diff)).item())
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    nll = float(torch.mean(pred_clamped - target * torch.log(pred_clamped)).item())
    ratio = target_clamped / pred_clamped
    poisson_deviance = float(torch.mean(2.0 * (diff + target * torch.log(ratio))).item())
    return {
        "mae": mae,
        "rmse": rmse,
        "nll": nll,
        "dev": poisson_deviance,
    }


def save_depth_profile(
    step,
    generator,
    z_latent,
    ct_vol,
    act_vol,
    outdir: Path,
    proj_ap=None,
    proj_pa=None,
    target_ap=None,
    target_pa=None,
    target_ap_counts=None,
    target_pa_counts=None,
    signal_quantile: float = 0.99,
    bg_quantile: float = 0.10,
    seed: int = 123,
):
    """
    Speichert Tiefenprofile (λ/μ/Prediction) entlang ausgewählter Strahlen für Analyse/Debugging.
    """
    H, W = generator.H, generator.W

    ap_img = proj_ap.detach().view(H, W).cpu().numpy() if proj_ap is not None else None
    pa_img = proj_pa.detach().view(H, W).cpu().numpy() if proj_pa is not None else None

    def proj_to_img(t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.detach()
        if t.numel() == H * W:
            t = t.view(H, W)
        elif t.dim() >= 2 and t.shape[-2:] == (H, W):
            t = t.reshape(-1, H, W)[0]
        else:
            return None
        return t.cpu().numpy()

    ap_counts_img = proj_to_img(target_ap_counts)
    pa_counts_img = proj_to_img(target_pa_counts)
    ap_target_img = proj_to_img(target_ap)
    pa_target_img = proj_to_img(target_pa)

    use_counts_targets = ap_counts_img is not None and pa_counts_img is not None
    if use_counts_targets:
        target_ap_img = ap_counts_img
        target_pa_img = pa_counts_img
        target_kind = "counts"
    else:
        target_ap_img = ap_target_img
        target_pa_img = pa_target_img
        target_kind = "norm" if (target_ap_img is not None or target_pa_img is not None) else "none"

    if target_ap_img is not None and target_pa_img is not None:
        score_map = np.maximum(target_ap_img, target_pa_img)
    elif target_ap_img is not None:
        score_map = target_ap_img.copy()
    elif target_pa_img is not None:
        score_map = target_pa_img.copy()
    else:
        score_map = None

    ap_title_img = target_ap_img
    pa_title_img = target_pa_img
    if ap_title_img is None:
        ap_title_img = proj_to_img(proj_ap)
    if pa_title_img is None:
        pa_title_img = proj_to_img(proj_pa)

    signal_quantile = float(np.clip(signal_quantile, 0.0, 1.0))
    bg_quantile = float(np.clip(bg_quantile, 0.0, 1.0))
    rng = np.random.default_rng(int(seed))

    perms = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]

    def _to_dhw(vol: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
        if vol is None:
            return None
        v = vol.detach()
        if v.dim() == 4:
            v = v.squeeze(0)
        if v.dim() != 3:
            print(
                f"[depth-profile][WARN] {name} ist nicht 3D (shape={tuple(v.shape)}); deaktiviere {name}-Kurve.",
                flush=True,
            )
            return None
        raw_shape = tuple(v.shape)
        chosen_perm = None
        v_dhw = None
        for perm in perms:
            v_perm = v.permute(perm)
            shp = tuple(v_perm.shape)
            if shp[1] == H and shp[2] == W:
                chosen_perm = perm
                v_dhw = v_perm.contiguous()
                break
        if chosen_perm is None or v_dhw is None:
            print(
                f"[depth-profile][WARN] {name} raw shape={raw_shape} kann nicht auf (D,H,W)=(*,{H},{W}) gemappt werden; "
                f"deaktiviere {name}-Kurve.",
                flush=True,
            )
            return None
        print(
            f"[depth-profile] {name} raw shape={raw_shape} | perm={chosen_perm} -> DHW shape={tuple(v_dhw.shape)}",
            flush=True,
        )
        if H > 0 and W > 0:
            cy = min(H // 2, v_dhw.shape[1] - 1)
            cx = min(W // 2, v_dhw.shape[2] - 1)
            center_max = float(v_dhw[:, cy, cx].max().item())
            print(
                f"[depth-profile] {name} center-line max @ (y={cy},x={cx}) = {center_max:.3e}",
                flush=True,
            )
        return v_dhw

    act_dhw = _to_dhw(act_vol, "act")
    ct_dhw = _to_dhw(ct_vol, "ct")

    act_sample_vol = None
    if act_vol is not None:
        act_raw = act_vol.detach()
        if act_raw.dim() == 4:
            act_raw = act_raw.squeeze(0)
        if act_raw.dim() != 3:
            print(
                f"[depth-profile][WARN] act raw shape={tuple(act_raw.shape)} ist nicht 3D; GT-Ray-Sampling deaktiviert.",
                flush=True,
            )
        else:
            act_sample_vol = act_raw.float().contiguous()
            if act_sample_vol.device != generator.device:
                act_sample_vol = act_sample_vol.to(generator.device, non_blocking=True)
            print(
                f"[depth-profile] act raw shape fuer Ray-Sampling: {tuple(act_sample_vol.shape)}",
                flush=True,
            )

    act_masks = None

    def extract_curve(vol_dhw: Optional[torch.Tensor], y_idx: int, x_idx: int, name: str):
        if vol_dhw is None:
            return None, None
        if vol_dhw.dim() != 3:
            print(
                f"[depth-profile][WARN] {name} DHW ist nicht 3D (shape={tuple(vol_dhw.shape)}); skippe.",
                flush=True,
            )
            return None, None
        D_loc, H_loc, W_loc = vol_dhw.shape
        if H_loc != H or W_loc != W:
            print(
                f"[depth-profile][WARN] {name} DHW shape={tuple(vol_dhw.shape)} passt nicht zu (H,W)=({H},{W}); skippe.",
                flush=True,
            )
            return None, None
        if not (0 <= y_idx < H_loc and 0 <= x_idx < W_loc):
            return None, None
        curve = vol_dhw[:, y_idx, x_idx].detach().float().cpu().numpy()
        curve = np.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        radius_local = generator.radius[1] if isinstance(generator.radius, tuple) else generator.radius
        z_coords = idx_to_coord(torch.arange(D_loc, device=vol_dhw.device), D_loc, float(radius_local))
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

        ct_good_mask = None
        if ct_dhw is not None:
            ct_depth_max = ct_dhw.max(dim=0).values.detach().cpu().numpy()
            ct_depth_min = ct_dhw.min(dim=0).values.detach().cpu().numpy()
            ct_good_mask = (ct_depth_max - ct_depth_min) > 1e-8
            ct_good_mask = ct_good_mask & (ct_depth_max > 1e-8)

        act_zero_mask = act_nonzero_mask = None
        if act_masks is not None:
            act_zero_mask, act_nonzero_mask = act_masks

        def prefer_mask(base_mask: np.ndarray, preferred_mask: Optional[np.ndarray]):
            if preferred_mask is None:
                return base_mask
            masked = base_mask & preferred_mask
            if masked.any():
                return masked
            return preferred_mask if preferred_mask.any() else base_mask

        def apply_optional_mask(base_mask: np.ndarray, extra_mask: Optional[np.ndarray]):
            if extra_mask is None:
                return base_mask
            masked = base_mask & extra_mask
            return masked if masked.any() else base_mask

        # Deterministische, target-basierte Auswahl: Background (niedrig),
        # Signal (oberstes Quantil), bevorzugt mit Count-Targets.
        if score_map is not None and np.isfinite(score_map).any():
            valid_mask = np.isfinite(score_map)
            scores_valid = score_map[valid_mask]
            if scores_valid.size > 0:
                q_sig = float(np.quantile(scores_valid, signal_quantile))
                q_bg = float(np.quantile(scores_valid, bg_quantile))

                zero_mask = (score_map == 0) & valid_mask
                valid_count = int(scores_valid.size)
                zero_count = int(zero_mask.sum())
                bg_needed = max(1, int(math.ceil(bg_quantile * valid_count)))
                many_zeros = zero_count >= bg_needed or q_bg <= 0.0
                bg_mask = zero_mask if many_zeros else ((score_map <= q_bg) & valid_mask)
                sig_mask = (score_map >= q_sig) & valid_mask
                bg_mask = prefer_mask(bg_mask, act_zero_mask)
                bg_mask = apply_optional_mask(bg_mask, ct_good_mask)
                sig_mask = prefer_mask(sig_mask, act_nonzero_mask)

                def sample_from_coords(coords: np.ndarray):
                    if coords.size == 0:
                        return None
                    idx = int(rng.integers(0, len(coords)))
                    y, x = coords[idx]
                    return int(y), int(x)

                bg_idx = None
                bg_coords = np.argwhere(bg_mask)
                if bg_coords.size > 0:
                    bg_scores = score_map[bg_mask]
                    median_val = float(np.median(bg_scores))
                    diffs = np.abs(bg_scores - median_val)
                    min_diff = float(diffs.min())
                    near_idx = np.flatnonzero(np.isclose(diffs, min_diff, rtol=0.0, atol=1e-12))
                    bg_idx = sample_from_coords(bg_coords[near_idx]) if near_idx.size > 0 else sample_from_coords(bg_coords)
                if bg_idx is None:
                    min_val = float(np.min(scores_valid))
                    min_mask = valid_mask & np.isclose(score_map, min_val, rtol=0.0, atol=1e-12)
                    min_mask = prefer_mask(min_mask, act_zero_mask)
                    min_mask = apply_optional_mask(min_mask, ct_good_mask)
                    fallback_mask = min_mask if min_mask.any() else prefer_mask(valid_mask, act_zero_mask)
                    fallback_mask = apply_optional_mask(fallback_mask, ct_good_mask)
                    bg_idx = sample_from_coords(np.argwhere(fallback_mask))
                add_unique(bg_idx)

                sig_coords = np.argwhere(sig_mask)
                if sig_coords.size > 0:
                    sig_scores = score_map[sig_mask]
                    order = np.argsort(sig_scores)[::-1]
                    for idx in order:
                        if len(chosen) >= num_zero + num_active:
                            break
                        y, x = sig_coords[int(idx)]
                        add_unique((int(y), int(x)))

                if len(chosen) < num_zero + num_active:
                    valid_coords = np.argwhere(valid_mask)
                    valid_scores = score_map[valid_mask]
                    order_global = np.argsort(valid_scores)[::-1]
                    for idx in order_global:
                        if len(chosen) >= num_zero + num_active:
                            break
                        y, x = valid_coords[int(idx)]
                        add_unique((int(y), int(x)))

                attempts = 0
                while len(chosen) < num_zero + num_active and attempts < 128:
                    attempts += 1
                    cand = sample_from_coords(np.argwhere(valid_mask))
                    add_unique(cand)

                if chosen:
                    return chosen[: num_zero + num_active]

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

        ct_pos_mask = ct_good_mask

        def combine_mask(base_mask, require_ct: bool):
            if base_mask is None:
                return None
            mask = base_mask.astype(bool)
            if ct_pos_mask is not None and require_ct:
                mask = mask & ct_pos_mask
            return mask

        zero_mask = nonzero_mask = None
        if act_masks is not None:
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

    target_shape = None
    for v in (ct_dhw, act_dhw):
        if v is not None:
            target_shape = tuple(v.shape)
            break
    if target_shape is None:
        target_shape = first_shape(ct_vol, act_vol)
    if target_shape is None:
        return

    radius = generator.radius
    if isinstance(radius, tuple):
        radius = radius[1]
    radius = float(radius)

    d_candidates = []
    if ct_dhw is not None:
        d_candidates.append(int(ct_dhw.shape[0]))
    if act_dhw is not None:
        d_candidates.append(int(act_dhw.shape[0]))
    if d_candidates:
        D = int(min(d_candidates))
        if len(set(d_candidates)) > 1:
            print(
                f"[depth-profile][WARN] Uneinheitliche Depth-Dims {d_candidates}; nutze D={D}.",
                flush=True,
            )
    else:
        D = int(target_shape[0])

    def _clip_depth(vol_dhw: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
        if vol_dhw is None:
            return None
        if vol_dhw.shape[0] < D:
            print(
                f"[depth-profile][WARN] {name} depth={vol_dhw.shape[0]} < D={D}; deaktiviere {name}-Kurve.",
                flush=True,
            )
            return None
        if vol_dhw.shape[0] != D:
            return vol_dhw[:D]
        return vol_dhw

    act_dhw = _clip_depth(act_dhw, "act")
    ct_dhw = _clip_depth(ct_dhw, "ct")

    act_masks = None
    if act_dhw is not None:
        act_data = act_dhw.detach().float().cpu().numpy()
        act_zero = act_data < 1e-6
        act_nonzero = act_data > 1e-6
        act_masks = (act_zero.max(axis=0), act_nonzero.max(axis=0))
        cy = min(H // 2, act_data.shape[1] - 1)
        cx = min(W // 2, act_data.shape[2] - 1)
        center_max_np = float(np.max(act_data[:, cy, cx]))
        print(
            f"[depth-profile] act_DHW sanity: max over depth @ (y={cy},x={cx}) = {center_max_np:.3e}",
            flush=True,
        )

    num_zero, num_active = 1, 1
    target_total = max(num_zero + num_active, 1)
    cache_attr = "_depth_profile_rays_cache"
    cache = getattr(generator, cache_attr, None)
    ray_indices_cache = None
    if isinstance(cache, dict):
        cached_indices = cache.get("indices")
        cached_shape = cache.get("shape")
        cached_total = cache.get("total")
        cached_sig_q = cache.get("signal_quantile")
        cached_bg_q = cache.get("bg_quantile")
        cached_seed = cache.get("seed")
        cached_target_kind = cache.get("target_kind")
        if (
            cached_indices
            and cached_shape == (generator.H, generator.W)
            and cached_total == target_total
            and cached_sig_q == signal_quantile
            and cached_bg_q == bg_quantile
            and cached_seed == int(seed)
            and cached_target_kind == target_kind
        ):
            ray_indices_cache = cached_indices

    if ray_indices_cache is None:
        ray_indices = pick_ray_indices(num_zero=num_zero, num_active=num_active)
        setattr(
            generator,
            cache_attr,
            {
                "indices": list(ray_indices),
                "shape": (generator.H, generator.W),
                "total": target_total,
                "signal_quantile": signal_quantile,
                "bg_quantile": bg_quantile,
                "seed": int(seed),
                "target_kind": target_kind,
            },
        )
    else:
        ray_indices = ray_indices_cache

    depth_idx = torch.arange(D, device=generator.device)
    z_coords = idx_to_coord(depth_idx, D, radius)
    depth_axis = np.linspace(0.0, 1.0, D)
    import matplotlib.pyplot as plt

    def _depth_axis_from_z(z_vals_t: torch.Tensor) -> np.ndarray:
        z_np = z_vals_t.detach().float().cpu().numpy().reshape(-1)
        if z_np.size == 0:
            return depth_axis
        z_min = float(np.min(z_np))
        z_max = float(np.max(z_np))
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min + 1e-8:
            return np.linspace(0.0, 1.0, z_np.size)
        return (z_np - z_min) / (z_max - z_min + 1e-8)

def _sample_volume_along_pts(vol_3d: torch.Tensor, pts: torch.Tensor, radius: float) -> Optional[torch.Tensor]:
    if vol_3d is None or vol_3d.dim() != 3:
        return None
    if radius <= 0:
        return None
    try:
        vol = vol_3d.view(1, 1, *vol_3d.shape)
        pts_norm = pts / radius
        global _SAMPLE_VOLUME_ALONG_PTS_LOGGED
        if not _SAMPLE_VOLUME_ALONG_PTS_LOGGED:
            _SAMPLE_VOLUME_ALONG_PTS_LOGGED = True
            clip_mask = (pts_norm.abs() > 1.0).any(dim=-1)
            clip_frac = float(clip_mask.float().mean().item())
            before_min = pts_norm.amin(dim=0).cpu().tolist()
            before_max = pts_norm.amax(dim=0).cpu().tolist()
            print(
                f"[sanity][_sample_volume_along_pts] radius={radius:.3e} pts_norm(min/max)={before_min}/{before_max} "
                f"clipped_frac={clip_frac:.3f}",
                flush=True,
            )
        grid = pts_norm.clamp(min=-1.0, max=1.0)
        grid = grid.view(1, grid.shape[0], 1, 1, 3)
        sampled = F.grid_sample(
            vol,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.view(-1)
    except Exception as exc:
        print(f"[depth-profile][WARN] grid_sample fehlgeschlagen: {exc}", flush=True)
        return None

    rays_ap_full = build_pose_rays(generator, generator.pose_ap)
    prev_flag = bool(generator.use_test_kwargs)
    generator.use_test_kwargs = True
    render_kwargs = dict(generator.render_kwargs_test)
    render_kwargs["features"] = z_latent
    render_kwargs["retraw"] = True
    if bool(render_kwargs.get("use_attenuation", False)):
        render_kwargs["use_attenuation"] = False
        print(
            "[depth-profile][WARN] use_attenuation ohne ct_context -> fuer Depth-Profile deaktiviert.",
            flush=True,
        )

    warn_no_act = False
    warn_no_z = False
    warn_render_fail = False
    warn_sampling_fail = False

    # Avoid constrained_layout here: long diagnostic strings can force extreme
    # whitespace and visually "squash" the axes width.
    width_per_panel = 5.0
    fig, axes = plt.subplots(
        1,
        len(ray_indices),
        figsize=(width_per_panel * len(ray_indices), 4.2),
        sharex=True,
        sharey=True,
    )
    fig.suptitle(f"Depth-Profile - Step {step}", fontsize=12)
    fig.subplots_adjust(wspace=0.35, top=0.80)
    axes = np.atleast_1d(axes)
    color_map = {
        "μ (CT)": "black",
        "Aktivität (GT)": "red",
        "Aktivität (NeRF)": "lime",
    }

    try:
        with torch.no_grad():
            generator.eval()
            for ax, (y_idx, x_idx) in zip(axes, ray_indices):
                curves = []
                labels = []

                ray_idx = int(y_idx * W + x_idx)
                ray_batch = rays_ap_full[:, ray_idx : ray_idx + 1, :]
                rays_o = ray_batch[0, 0]
                rays_d = ray_batch[1, 0]

                z_vals_ray = None
                try:
                    _, _, _, extras = generator.render(rays=ray_batch, **render_kwargs)
                    z_vals_ray = extras.get("z_vals")
                except Exception as exc:
                    if not warn_render_fail:
                        print(f"[depth-profile][WARN] render fuer z_vals fehlgeschlagen: {exc}", flush=True)
                        warn_render_fail = True
                    z_vals_ray = None

                if z_vals_ray is None:
                    if not warn_no_z:
                        print(
                            "[depth-profile][WARN] Keine z_vals aus render; nutze uniforme Samples entlang des Rays.",
                            flush=True,
                        )
                        warn_no_z = True
                    z_vals_ray = z_coords
                else:
                    z_vals_ray = z_vals_ray.reshape(-1)

                pts = rays_o.view(1, 3) + rays_d.view(1, 3) * z_vals_ray.view(-1, 1)
                depth_axis_ray = _depth_axis_from_z(z_vals_ray)
                global _DEPTH_PROFILE_LOGGED
                if not _DEPTH_PROFILE_LOGGED:
                    _DEPTH_PROFILE_LOGGED = True
                    z_head = z_vals_ray[:5].cpu().tolist()
                    z_tail = z_vals_ray[-5:].cpu().tolist()
                    z_min = float(z_vals_ray.min().item())
                    z_max = float(z_vals_ray.max().item())
                    depth_axis_head = depth_axis_ray[:5].tolist()
                    depth_axis_tail = depth_axis_ray[-5:].tolist()
                    print(
                        f"[sanity][depth-profile] z_vals(min/max)={z_min:.3e}/{z_max:.3e} head={z_head} tail={z_tail} "
                        f"depth_axis(head/tail)={depth_axis_head}/{depth_axis_tail} (plotted axis=depth_axis_ray)",
                        flush=True,
                    )
                d_ray = int(depth_axis_ray.size)

                curve_ct = extract_curve(ct_dhw, y_idx, x_idx, "ct") if ct_dhw is not None else (None, None)
                ct_curve_raw = curve_ct[0]
                if ct_curve_raw is not None:
                    ct_curve_raw = np.asarray(ct_curve_raw, dtype=np.float32).reshape(-1)
                    if ct_curve_raw.size != d_ray and ct_curve_raw.size > 1:
                        x_src = np.linspace(0.0, 1.0, ct_curve_raw.size)
                        x_dst = np.linspace(0.0, 1.0, d_ray)
                        ct_curve_raw = np.interp(x_dst, x_src, ct_curve_raw).astype(np.float32)
                    if ct_curve_raw.size == d_ray:
                        curves.append(normalize_curve(ct_curve_raw.copy()))
                        labels.append("μ (CT)")

                pred = query_emission_at_points(generator, z_latent, pts).detach().float().cpu().numpy().reshape(-1)
                pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
                if pred.size != d_ray and pred.size > 1:
                    x_src = np.linspace(0.0, 1.0, pred.size)
                    x_dst = np.linspace(0.0, 1.0, d_ray)
                    pred = np.interp(x_dst, x_src, pred).astype(np.float32)
                curves.append(normalize_curve(pred.copy()))
                labels.append("Aktivität (NeRF)")

                gt_curve_raw = None
                if act_sample_vol is None:
                    if not warn_no_act:
                        print("[depth-profile][WARN] act_vol fehlt fuer Ray-Sampling; GT-Kurve wird ausgelassen.", flush=True)
                        warn_no_act = True
                else:
                    gt_curve_t = _sample_volume_along_pts(act_sample_vol, pts, radius)
                    if gt_curve_t is None:
                        if not warn_sampling_fail:
                            print("[depth-profile][WARN] GT-Ray-Sampling fehlgeschlagen; GT-Kurve wird ausgelassen.", flush=True)
                            warn_sampling_fail = True
                    else:
                        gt_curve_raw = gt_curve_t.detach().float().cpu().numpy().reshape(-1)
                        gt_curve_raw = np.nan_to_num(gt_curve_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(
                            np.float32, copy=False
                        )
                        if gt_curve_raw.size != d_ray and gt_curve_raw.size > 1:
                            x_src = np.linspace(0.0, 1.0, gt_curve_raw.size)
                            x_dst = np.linspace(0.0, 1.0, d_ray)
                            gt_curve_raw = np.interp(x_dst, x_src, gt_curve_raw).astype(np.float32)
                        if gt_curve_raw.size == d_ray:
                            curves.insert(0, normalize_curve(gt_curve_raw.copy()))
                            labels.insert(0, "Aktivität (GT)")

                for curve, label in zip(curves, labels):
                    ax.plot(depth_axis_ray, curve, label=label, color=color_map.get(label))

                def _curve_stats(arr: Optional[np.ndarray]) -> Tuple[float, float, float]:
                    if arr is None or arr.size == 0:
                        return float("nan"), float("nan"), float("nan")
                    finite = np.isfinite(arr)
                    if not finite.any():
                        return float("nan"), float("nan"), 0.0
                    vals = arr[finite]
                    return float(np.max(vals)), float(np.quantile(vals, 0.95)), float(np.mean(vals > 1e-8))

                gt_max, gt_p95, gt_nz = _curve_stats(gt_curve_raw)
                pred_max, pred_p95, pred_nz = _curve_stats(pred)
                print(
                    f"[depth-profile][ray y={y_idx} x={x_idx}] gt_max={gt_max:.3e} gt_p95={gt_p95:.3e} gt_nz={gt_nz:.3f} "
                    f"| pred_max={pred_max:.3e} pred_p95={pred_p95:.3e} pred_nz={pred_nz:.3f}",
                    flush=True,
                )

                diag_parts = []
                if np.isfinite(gt_max):
                    diag_parts.append(f"gt_max={gt_max:.2e}")
                    diag_parts.append(f"gt_p95={gt_p95:.2e}")
                    if gt_nz <= 0.0:
                        diag_parts.append("gt_nonzero=0")
                if np.isfinite(pred_max):
                    diag_parts.append(f"pred_max={pred_max:.2e}")
                if ap_title_img is not None:
                    diag_parts.append(f"I_AP={ap_title_img[y_idx, x_idx]:.2e}")
                if pa_title_img is not None:
                    diag_parts.append(f"I_PA={pa_title_img[y_idx, x_idx]:.2e}")

                # Break the diagnostic text into multiple short lines so it
                # stays above the plot area without overlapping curves.
                max_parts_per_line = 3
                diag_lines = [
                    " | ".join(diag_parts[i : i + max_parts_per_line])
                    for i in range(0, len(diag_parts), max_parts_per_line)
                ]
                title_lines = [f"({y_idx},{x_idx})"] + diag_lines
                ax.set_title("\n".join(title_lines), fontsize=8, pad=10)
                ax.set_ylim(0, 1.05)
                ax.set_xlim(0.0, 1.0)
                ax.grid(True, alpha=0.2)
                ax.legend(loc="upper right", fontsize=8)
                # Ensure square axes boxes for depth_profile_step_*.png (equal side lengths).
                # This is about visual aspect, not equal data step sizes.
                if hasattr(ax, "set_box_aspect"):
                    ax.set_box_aspect(1)
    finally:
        generator.use_test_kwargs = prev_flag
        if hasattr(generator, "train"):
            generator.train()

    axes[0].set_ylabel("normierte Intensität")
    for ax in axes:
        ax.set_xlabel("Tiefe (anterior → posterior)")
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"depth_profile_step_{step:05d}.png", dpi=150)
    plt.close(fig)


def robust_norm_np(img: np.ndarray, lo_q: float = 0.01, hi_q: float = 0.99) -> np.ndarray:
    """Robuste Normierung auf [0,1] via Quantile (NaN/Inf-sicher)."""
    arr = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    vals = arr[finite]
    lo_q = float(np.clip(lo_q, 0.0, 1.0))
    hi_q = float(np.clip(hi_q, 0.0, 1.0))
    if hi_q < lo_q:
        lo_q, hi_q = hi_q, lo_q
    try:
        lo = float(np.quantile(vals, lo_q))
        hi = float(np.quantile(vals, hi_q))
    except Exception:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if not np.isfinite(lo):
        lo = float(np.min(vals))
    if not np.isfinite(hi):
        hi = float(np.max(vals))
    if hi <= lo + 1e-8:
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmax <= vmin + 1e-8:
            out = np.zeros_like(arr, dtype=np.float32)
            out[finite] = 0.0
            return out
        lo, hi = vmin, vmax
    clipped = np.clip(arr, lo, hi)
    return ((clipped - lo) / (hi - lo + 1e-8)).astype(np.float32)


def coord_to_index(coord: torch.Tensor, size: int, radius: float) -> torch.Tensor:
    """Inverse von idx_to_coord mit Clamping auf gueltige Indizes."""
    if size <= 1 or radius <= 0:
        return torch.zeros_like(coord, dtype=torch.long)
    idx_f = ((coord / (2.0 * float(radius))) + 0.5) * float(size - 1)
    return torch.clamp(idx_f.round().long(), min=0, max=int(size - 1))


def save_final_sagittal_depth_consistency(
    args,
    generator,
    z_latent: torch.Tensor,
    act_vol: Optional[torch.Tensor],
    ct_context,
    outdir: Path,
):
    """
    Finale sagittale GT-vs-Depth-Curtain Visualisierung.

    Links: GT-Slices exakt wie in test.py (fixe axis0-Indizes, extent/aspect/ticks).
    Rechts: Depth-Curtain pro Slice auf demselben (axis1,axis2)-Raster.
    """
    if not bool(getattr(args, "final_sagittal_viz", False)):
        return
    if act_vol is None or act_vol.numel() == 0:
        print("[final-sagittal] act_vol fehlt; skippe sagittale Visualisierung.", flush=True)
        return

    import matplotlib.pyplot as plt

    preview_dir = outdir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    out_path = preview_dir / "final_sagittal_depth_consistency.png"
    debug_path = preview_dir / "final_sagittal_debug.png"

    act_t = act_vol.detach()
    if act_t.dim() == 4:
        act_t = act_t.squeeze(0)
    if act_t.dim() != 3:
        print(f"[final-sagittal][WARN] act_vol ist nicht 3D (shape={tuple(act_t.shape)}); skippe.", flush=True)
        return

    act_np = act_t.float().cpu().numpy()
    A, B, C = act_np.shape  # axis0, axis1, axis2
    H, W = int(generator.H), int(generator.W)
    print(
        f"[final-sagittal] act_vol.shape={(A, B, C)} | generator.HW={(H, W)}",
        flush=True,
    )

    idx_list_raw = list(getattr(args, "final_sagittal_axis0_idx", [80, 128, 200]))
    idx_list = []
    for idx in idx_list_raw:
        idx_i = int(idx)
        if 0 <= idx_i < A:
            idx_list.append(idx_i)
        else:
            print(f"[final-sagittal][WARN] axis0 idx={idx_i} out of bounds fuer A={A}; ignoriere.", flush=True)
    if not idx_list:
        print("[final-sagittal][WARN] Keine gueltigen axis0-Indizes; skippe.", flush=True)
        return

    # gemeinsame Ticks exakt wie in test.py
    y_step = max(1, B // 10)
    z_step = max(1, C // 10)
    y_ticks = np.arange(0, B, y_step)
    z_ticks = np.arange(0, C, z_step)

    def _save_gt_only():
        fig, axs = plt.subplots(1, len(idx_list), figsize=(5 * len(idx_list), 6), constrained_layout=True)
        if not isinstance(axs, np.ndarray):
            axs = np.asarray([axs])
        last_im = None
        for ax, idx in zip(axs, idx_list):
            img = act_np[idx, :, :].astype(np.float32)
            # Optional: Orientierungskorrekturen wie in test.py.
            # img = np.flipud(img)
            # img = np.fliplr(img)
            last_im = ax.imshow(
                img,
                origin="upper",
                extent=[0, C - 1, B - 1, 0],
                aspect="equal",
                cmap="viridis",
            )
            ax.set_title(f"GT sagittal (act) @ axis0={idx}")
            ax.set_xlabel("axis2 (z-like)")
            ax.set_ylabel("axis1 (y-like)")
            ax.set_xticks(z_ticks)
            ax.set_yticks(y_ticks)
        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axs, shrink=0.9)
            cbar.set_label("Activity (raw units)")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[final-sagittal] Saved GT-only {out_path}", flush=True)

    # Shape-Sanity: fuer korrektes Mapping braucht das Ray-Raster (axis1,axis2).
    if (B != H) or (C != W):
        print(
            "[final-sagittal][WARN] act (axis1,axis2) passt nicht zu generator.HW -> speichere GT-only.",
            flush=True,
        )
        _save_gt_only()
        return

    rays_ap_full = build_pose_rays(generator, generator.pose_ap)
    if rays_ap_full.dim() != 3 or rays_ap_full.shape[0] != 2:
        print(
            f"[final-sagittal][WARN] Unerwartete Ray-Shape: {tuple(rays_ap_full.shape)}; speichere GT-only.",
            flush=True,
        )
        _save_gt_only()
        return

    # Render-Setup mit retraw, um Depth-Gewichte pro Ray zu rekonstruieren.
    prev_flag = bool(generator.use_test_kwargs)
    generator.use_test_kwargs = True
    render_kwargs = dict(generator.render_kwargs_test)
    render_kwargs["features"] = z_latent
    render_kwargs["retraw"] = True
    use_atten = bool(render_kwargs.get("use_attenuation", False))
    atten_scale = float(render_kwargs.get("atten_scale", ATTEN_SCALE_DEFAULT))
    if ct_context is not None:
        render_kwargs["ct_context"] = ct_context
    elif use_atten:
        render_kwargs["use_attenuation"] = False

    eps = 1e-8
    ray_chunk = int(render_kwargs.get("chunk", 1024 * 32))
    ray_chunk = max(1024, min(ray_chunk, 16384))
    n_rays = int(rays_ap_full.shape[1])
    print(
        f"[final-sagittal] rays={n_rays} | ray_chunk={ray_chunk} | atten={use_atten} | atten_scale={atten_scale:.3f}",
        flush=True,
    )

    def _compute_depth_weights(extras: dict) -> Optional[torch.Tensor]:
        raw_out = extras.get("raw")
        dists = extras.get("dists")
        if raw_out is None or dists is None:
            return None
        lambda_vals = F.softplus(raw_out[..., 0])
        weights = lambda_vals * dists
        mu_vals = extras.get("mu")
        if mu_vals is not None:
            mu = torch.clamp(mu_vals, min=0.0)
            mu_dists = mu * dists
            attenuation = torch.cumsum(mu_dists, dim=-1) * float(atten_scale)
            attenuation = F.pad(attenuation[..., :-1], (1, 0), mode="constant", value=0.0)
            attenuation = torch.clamp(attenuation, min=0.0, max=60.0)
            transmission = torch.exp(-attenuation)
            mu_t = mu * transmission
            # Bevorzugt: mu*T (muT). Fallback ist lambda*dists.
            weights = mu_t * dists
        weights = torch.clamp(weights, min=0.0)
        return weights

    expected_map = torch.zeros((B, C), dtype=torch.float32, device="cpu")
    peak_map = torch.zeros((B, C), dtype=torch.float32, device="cpu")
    sumw_map = torch.zeros((B, C), dtype=torch.float32, device="cpu")
    argmax_map = torch.zeros((B, C), dtype=torch.int64, device="cpu")
    valid_map = torch.zeros((B, C), dtype=torch.bool, device="cpu")
    D_samples: Optional[int] = None
    curtain_available = True
    logged_shapes = False
    try:
        with torch.no_grad():
            generator.eval()
            for start in range(0, n_rays, ray_chunk):
                end = min(n_rays, start + ray_chunk)
                rays_chunk = rays_ap_full[:, start:end, :]
                _, _, _, extras = generator.render(rays=rays_chunk, **render_kwargs)
                raw_out = extras.get("raw")
                dists = extras.get("dists")
                mu_vals = extras.get("mu")
                weights_raw = _compute_depth_weights(extras)
                z_vals = extras.get("z_vals")
                if not logged_shapes:
                    raw_shape = tuple(raw_out.shape) if raw_out is not None else None
                    dists_shape = tuple(dists.shape) if dists is not None else None
                    z_shape = tuple(z_vals.shape) if z_vals is not None else None
                    w_shape = tuple(weights_raw.shape) if weights_raw is not None else None
                    print(
                        "[final-sagittal][chunk0] "
                        f"raw={raw_shape} | dists={dists_shape} | z_vals={z_shape} | "
                        f"weights_raw={w_shape} | mu_present={mu_vals is not None}",
                        flush=True,
                    )
                    logged_shapes = True
                if weights_raw is None or z_vals is None:
                    curtain_available = False
                    print(
                        "[final-sagittal][WARN] Extras ohne raw/dists/z_vals -> speichere GT-only.",
                        flush=True,
                    )
                    break

                sum_w_raw = weights_raw.sum(dim=-1, keepdim=True)
                weights = weights_raw / (sum_w_raw + eps)

                D_here = int(weights.shape[-1])
                if D_samples is None:
                    D_samples = D_here
                elif D_samples != D_here:
                    d_use = min(D_samples, D_here)
                    print(
                        f"[final-sagittal][WARN] Inkonsistente Sample-Anzahl ({D_samples} vs {D_here}); nutze {d_use}.",
                        flush=True,
                    )
                    weights = weights[..., :d_use]
                    D_here = d_use
                    D_samples = d_use

                sample_idx = torch.arange(D_here, device=weights.device, dtype=weights.dtype)
                expected_ray = torch.sum(weights * sample_idx.view(1, -1), dim=-1)
                peak_ray = torch.max(weights, dim=-1).values
                sumw_ray = torch.sum(weights, dim=-1)
                argmax_ray = torch.argmax(weights, dim=-1)

                expected_cpu = expected_ray.detach().cpu().float()
                peak_cpu = peak_ray.detach().cpu().float()
                sumw_cpu = sumw_ray.detach().cpu().float()
                argmax_cpu = argmax_ray.detach().cpu().long()

                ray_idx_np = np.arange(start, end, dtype=np.int64)
                y_idx_np = ray_idx_np // C
                x_idx_np = ray_idx_np % C
                y_idx = torch.from_numpy(y_idx_np).long()
                x_idx = torch.from_numpy(x_idx_np).long()

                expected_map[y_idx, x_idx] = expected_cpu
                peak_map[y_idx, x_idx] = peak_cpu
                sumw_map[y_idx, x_idx] = sumw_cpu
                argmax_map[y_idx, x_idx] = argmax_cpu
                valid_map[y_idx, x_idx] = True
    finally:
        # Generator-Zustand robust wiederherstellen (ohne .training Zugriff).
        generator.use_test_kwargs = prev_flag
        if hasattr(generator, "train"):
            generator.train()

    if (not curtain_available) or (D_samples is None):
        _save_gt_only()
        return

    valid_np = valid_map.cpu().numpy()
    expected_np = expected_map.cpu().numpy()
    peak_np = peak_map.cpu().numpy()
    sumw_np = sumw_map.cpu().numpy()
    if not valid_np.any():
        print("[final-sagittal][WARN] Keine gueltigen Rays fuer expected_map; speichere GT-only.", flush=True)
        _save_gt_only()
        return
    # Nur fuer die Visualisierung in float casten.
    argmax_np = argmax_map.cpu().numpy().astype(np.float32)

    def _stats(vals: np.ndarray) -> Tuple[float, float, float, float]:
        return (
            float(np.min(vals)),
            float(np.median(vals)),
            float(np.quantile(vals, 0.95)),
            float(np.max(vals)),
        )

    exp_vals = expected_np[valid_np]
    peak_vals = peak_np[valid_np]
    sumw_vals = sumw_np[valid_np]
    valid_frac = float(np.mean(valid_np))
    frac_peak_sharp = float(np.mean(peak_vals > (3.0 / float(max(D_samples, 1)))))
    exp_stats = _stats(exp_vals)
    peak_stats = _stats(peak_vals)
    sumw_stats = _stats(sumw_vals)
    frac_sumw_low = float(np.mean(sumw_vals < 0.99))
    print(
        f"[final-sagittal] expected_map.shape={expected_np.shape} | "
        f"E[sample] min/median/p95/max="
        f"{exp_stats[0]:.2f}/{exp_stats[1]:.2f}/{exp_stats[2]:.2f}/{exp_stats[3]:.2f} "
        f"| valid_frac={valid_frac:.3f}",
        flush=True,
    )
    print(
        "[final-sagittal] peak_map min/median/p95/max="
        f"{peak_stats[0]:.3f}/{peak_stats[1]:.3f}/{peak_stats[2]:.3f}/{peak_stats[3]:.3f} "
        f"| frac(peak>3/D)={frac_peak_sharp:.3f}",
        flush=True,
    )
    print(
        "[final-sagittal] sumw_map min/median/p95/max="
        f"{sumw_stats[0]:.3f}/{sumw_stats[1]:.3f}/{sumw_stats[2]:.3f}/{sumw_stats[3]:.3f} "
        f"| frac(sumw<0.99)={frac_sumw_low:.3f}",
        flush=True,
    )

    # Sanity-Checks: Index-Mapping + Ray-Summen fuer Testpunkte.
    rng = np.random.default_rng(12345)
    for idx in idx_list:
        points = [(B // 2, C // 2)]
        for _ in range(4):
            points.append((int(rng.integers(0, B)), int(rng.integers(0, C))))
        print(f"[final-sagittal][slice axis0={idx}] Sanity-Checks:", flush=True)
        for (yy, zz) in points[:5]:
            ray_idx = int(yy * C + zz)
            act_val = float(act_np[idx, yy, zz])
            exp_d = float(expected_np[yy, zz])
            peak_v = float(peak_np[yy, zz])
            sum_w = float(sumw_np[yy, zz])
            print(
                f"  (axis1,axis2)=({yy},{zz}) | act={act_val:.3e} | ray_idx=y*C+z={ray_idx} "
                f"| E[sample]={exp_d:.2f} | peak={peak_v:.3f} | sum_w={sum_w:.3f}",
                flush=True,
            )

    cmap_name = "viridis"
    cmap = plt.get_cmap(cmap_name)

    # Hauptfigure: 3 Slices, jeweils GT links, expected_map rechts.
    n_rows = len(idx_list)
    fig, axs = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axs = np.asarray([axs])

    last_im = None
    depth_vmin, depth_vmax = 0.0, float(max(D_samples - 1, 1))
    last_depth_im = None
    for row, idx in enumerate(idx_list):
        gt_ax = axs[row, 0]
        cur_ax = axs[row, 1]
        gt_img = act_np[idx, :, :].astype(np.float32)
        # Optional: Orientierungskorrekturen wie in test.py.
        # gt_img = np.flipud(gt_img)
        # gt_img = np.fliplr(gt_img)

        last_im = gt_ax.imshow(
            gt_img,
            origin="upper",
            extent=[0, C - 1, B - 1, 0],
            aspect="equal",
            cmap=cmap_name,
        )
        gt_ax.set_title(f"GT sagittal (act) @ axis0={idx}")
        gt_ax.set_xlabel("axis2 (z-like)")
        gt_ax.set_ylabel("axis1 (y-like)")
        gt_ax.set_xticks(z_ticks)
        gt_ax.set_yticks(y_ticks)

        if row == 0:
            last_depth_im = cur_ax.imshow(
                expected_np,
                origin="upper",
                extent=[0, C - 1, B - 1, 0],
                aspect="equal",
                cmap="magma",
                vmin=depth_vmin,
                vmax=depth_vmax,
            )
            cur_ax.set_title("Expected sample index (AP, shared across slices)")
            cur_ax.set_xlabel("axis2 (z-like)")
            cur_ax.set_ylabel("axis1 (y-like)")
            cur_ax.set_xticks(z_ticks)
            cur_ax.set_yticks(y_ticks)
        else:
            cur_ax.axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axs.ravel().tolist(), shrink=0.9)
        cbar.set_label("Activity (raw units)")
    if last_depth_im is not None:
        cbar_depth = fig.colorbar(last_depth_im, ax=axs[:, 1].ravel().tolist(), shrink=0.9)
        cbar_depth.set_label("Expected sample index")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[final-sagittal] Saved {out_path}", flush=True)

    # Optionale Debug-Visualisierung: expected_map, peak_map und argmax_map.
    if bool(getattr(args, "final_sagittal_debug", False)):
        fig_dbg, axs_dbg = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        exp_im = axs_dbg[0].imshow(
            expected_np,
            origin="upper",
            extent=[0, C - 1, B - 1, 0],
            aspect="equal",
            cmap="magma",
            vmin=depth_vmin,
            vmax=depth_vmax,
        )
        axs_dbg[0].set_title("Expected sample index (AP)")
        axs_dbg[0].set_xlabel("axis2 (z-like)")
        axs_dbg[0].set_ylabel("axis1 (y-like)")
        axs_dbg[0].set_xticks(z_ticks)
        axs_dbg[0].set_yticks(y_ticks)

        peak_im = axs_dbg[1].imshow(
            peak_np,
            origin="upper",
            extent=[0, C - 1, B - 1, 0],
            aspect="equal",
            cmap="viridis",
        )
        axs_dbg[1].set_title("Peak weight per ray (AP)")
        axs_dbg[1].set_xlabel("axis2 (z-like)")
        axs_dbg[1].set_ylabel("axis1 (y-like)")
        axs_dbg[1].set_xticks(z_ticks)
        axs_dbg[1].set_yticks(y_ticks)

        argmax_im = axs_dbg[2].imshow(
            argmax_np,
            origin="upper",
            extent=[0, C - 1, B - 1, 0],
            aspect="equal",
            cmap="magma",
            vmin=depth_vmin,
            vmax=depth_vmax,
        )
        axs_dbg[2].set_title("Argmax sample index (AP)")
        axs_dbg[2].set_xlabel("axis2 (z-like)")
        axs_dbg[2].set_ylabel("axis1 (y-like)")
        axs_dbg[2].set_xticks(z_ticks)
        axs_dbg[2].set_yticks(y_ticks)

        fig_dbg.colorbar(exp_im, ax=axs_dbg[0], shrink=0.9, label="Expected sample index")
        fig_dbg.colorbar(peak_im, ax=axs_dbg[1], shrink=0.9, label="Peak weight")
        fig_dbg.colorbar(argmax_im, ax=axs_dbg[2], shrink=0.9, label="Argmax sample index")
        fig_dbg.savefig(debug_path, dpi=200)
        plt.close(fig_dbg)
        print(f"[final-sagittal] Saved debug {debug_path}", flush=True)


def save_final_act_compare_volume_slicing(
    args,
    act_vol: Optional[torch.Tensor],
    outdir: Path,
    pred_path: Optional[Path],
    pred_vol_np: Optional[np.ndarray] = None,
    out_path_override: Optional[Path] = None,
    grid_radius: Optional[float] = None,
    radius_xyz: Optional[Tuple[float, float, float]] = None,
    force_save: bool = False,
):
    """Finale GT-vs-Pred Activity-Compare PNG via reinem Volumen-Slicing (wie in test.py)."""
    if not (bool(getattr(args, "final_act_compare", False)) or force_save):
        return

    if act_vol is None or act_vol.numel() == 0:
        print("[final-act-compare] act_vol fehlt; skippe Activity-Compare.", flush=True)
        return

    if pred_vol_np is None and (pred_path is None or (not pred_path.exists())):
        print(
            f"[final-act-compare][WARN] Pred-Datei fehlt: {pred_path}; skippe Activity-Compare.",
            flush=True,
        )
        return

    import matplotlib.pyplot as plt

    preview_dir = outdir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    # Axial ist der Default; das Flag wird aus Kompatibilitaetsgruenden gelesen.
    _ = bool(getattr(args, "final_act_compare_axial", True))
    # Finale Compare-PNG ist axial und kollidiert nicht mit final_sagittal.
    base_out_path = out_path_override if out_path_override is not None else (preview_dir / "final_act_compare_axial.png")
    base_dir = base_out_path.parent
    base_name = base_out_path.stem
    abs_shared_path = base_dir / f"{base_name}_abs_shared.png"

    act_t = act_vol.detach()
    if act_t.dim() == 4:
        act_t = act_t.squeeze(0)
    if act_t.dim() != 3:
        print(
            f"[final-act-compare][WARN] act_vol ist nicht 3D (shape={tuple(act_t.shape)}); skippe.",
            flush=True,
        )
        return

    act_norm_factor, _ = compute_act_norm_factor(
        act_t, args.act_norm_source, args.act_norm_value, None
    )
    act_norm_factor = max(float(act_norm_factor), 1e-8)
    gt_np = act_t.float().cpu().numpy()
    if pred_vol_np is not None:
        pred_np = np.asarray(pred_vol_np, dtype=np.float32)
    else:
        if pred_path is None:
            print("[final-act-compare][WARN] pred_path fehlt; skippe Activity-Compare.", flush=True)
            return
        try:
            pred_np = np.load(pred_path).astype(np.float32, copy=False)
        except Exception as exc:
            print(f"[final-act-compare][WARN] Konnte Pred nicht laden: {exc}; skippe.", flush=True)
            return

    if pred_np.ndim != 3:
        print(
            f"[final-act-compare][WARN] Pred ist nicht 3D (shape={tuple(pred_np.shape)}); skippe.",
            flush=True,
        )
        return
    gt_raw_shape = tuple(int(x) for x in gt_np.shape)
    pred_shape = tuple(int(x) for x in pred_np.shape)
    if radius_xyz is None:
        if grid_radius is not None:
            radius_xyz = (float(grid_radius), float(grid_radius), float(grid_radius))
        else:
            radius_xyz = (1.0, 1.0, 1.0)
    else:
        radius_xyz = tuple(float(r) for r in radius_xyz)

    def _volume_stats(arr: np.ndarray) -> Tuple[float, float, float]:
        finite = np.isfinite(arr)
        if finite.any():
            vals = arr[finite].astype(np.float32, copy=False).ravel()
            if vals.size:
                return float(vals.min()), float(vals.mean()), float(vals.max())
        return float("nan"), float("nan"), float("nan")

    rx, ry, rz = radius_xyz

    def _spacing_for_axis(dim: int, radius_axis: float) -> float:
        if dim <= 1 or radius_axis == 0.0:
            return float("nan")
        return (2.0 * float(radius_axis)) / float(dim - 1)

    def _resample_gt_to_pred(gt_arr: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        if gt_arr.shape == target_shape:
            return gt_arr.astype(np.float32, copy=True)
        tensor = torch.from_numpy(gt_arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            resampled = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)
        return resampled.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    gt_np = _resample_gt_to_pred(gt_np, pred_shape)
    gt_shape = tuple(int(x) for x in gt_np.shape)
    axis_radii = (rz, ry, rx)
    gt_spacing_raw = tuple(_spacing_for_axis(dim, rad) for dim, rad in zip(gt_raw_shape, axis_radii))
    gt_spacing_resampled = tuple(_spacing_for_axis(dim, rad) for dim, rad in zip(gt_shape, axis_radii))
    pred_spacing = tuple(_spacing_for_axis(dim, rad) for dim, rad in zip(pred_shape, axis_radii))

    gt_abs = np.clip(gt_np.astype(np.float32, copy=False), 0.0, None)
    pred_abs = np.clip(pred_np.astype(np.float32, copy=False), 0.0, None)

    print(
        "[final-act-compare][debug] "
        f"GT raw shape={gt_raw_shape} (D,H,W) -> resampled shape={gt_shape}; Pred shape={pred_shape}; "
        "axis meaning: axial depth=axis2 (z-axis).",
        flush=True,
    )
    print(
        "[final-act-compare][debug] World extents (cm): "
        f"axis0(D/z)=[{-rz:.3f},{rz:.3f}] axis1(H/y)=[{-ry:.3f},{ry:.3f}] axis2(W/x)=[{-rx:.3f},{rx:.3f}]; "
        f"voxel spacings raw/resampled/pred={gt_spacing_raw}/{gt_spacing_resampled}/{pred_spacing}.",
        flush=True,
    )
    print(
        "[final-act-compare][debug] Resampling direction: GT -> Pred grid (trilinear, align_corners=False); "
        f"Resampled stats: min/mean/max={_volume_stats(gt_np)}; Pred stats: min/mean/max={_volume_stats(pred_np)}.",
        flush=True,
    )

    def _robust_limits(arr: np.ndarray) -> Tuple[float, float]:
        finite = np.isfinite(arr)
        if not finite.any():
            return 0.0, 1.0
        vals = arr[finite].astype(np.float32, copy=False).ravel()
        if vals.size == 0:
            return 0.0, 1.0
        try:
            lo, hi = np.quantile(vals, [0.01, 0.99])
        except Exception:
            lo = float(np.min(vals))
            hi = float(np.max(vals))
        lo = float(lo)
        hi = float(hi)
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo + 1e-8):
            lo = float(np.min(vals))
            hi = float(np.max(vals))
            if hi <= lo + 1e-8:
                hi = lo + 1e-6
        return lo, hi

    def _slice_stats(img: np.ndarray) -> Tuple[float, float, float, float, float]:
        finite = np.isfinite(img)
        if not finite.any():
            return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        vals = img[finite].astype(np.float32, copy=False).ravel()
        if vals.size == 0:
            return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        nz_frac = float(np.mean(np.abs(vals) > 1e-8))
        return float(np.min(vals)), float(np.mean(vals)), float(np.max(vals)), nz_frac, float(np.std(vals))

    A, B, C = gt_shape
    R = int(pred_shape[0])
    _, _, C_gt_raw = gt_raw_shape
    z_list_raw = list(getattr(args, "final_act_compare_axis2_idx", [65, 260, 325]))

    def _idx_to_world(z_idx: int, size: int, radius_axis: float) -> float:
        if size <= 1 or radius_axis == 0.0:
            return 0.0
        return ((float(z_idx) / float(size - 1)) - 0.5) * 2.0 * radius_axis

    def _world_to_pred_idx(z_world: float, R_pred: int, radius_pred: float) -> int:
        if R_pred <= 1 or radius_pred == 0.0:
            return 0
        normalized = np.clip((z_world / (2.0 * radius_pred)) + 0.5, 0.0, 1.0)
        idx = int(round(normalized * (R_pred - 1)))
        return int(np.clip(idx, 0, R_pred - 1))

    z_pairs: list[Tuple[int, int]] = []
    for z_gt in z_list_raw:
        z_i = int(z_gt)
        if 0 <= z_i < C_gt_raw:
            world_z = _idx_to_world(z_i, C_gt_raw, rz)
            z_pred = _world_to_pred_idx(world_z, R, rz)
            z_pairs.append((z_i, z_pred))
        else:
            print(
                f"[final-act-compare][WARN] axis2 idx_gt={z_i} out of bounds fuer C_gt={C_gt_raw}; ignoriere.",
                flush=True,
            )
    if not z_pairs:
        print("[final-act-compare][WARN] Keine gueltigen axis2-Indizes; skippe.", flush=True)
        return

    x_step_gt = max(1, B // 10)
    y_step_gt = max(1, A // 10)
    x_ticks_gt = np.arange(0, B, x_step_gt)
    y_ticks_gt = np.arange(0, A, y_step_gt)

    x_step_pr = max(1, R // 10)
    y_step_pr = max(1, R // 10)
    x_ticks_pr = np.arange(0, R, x_step_pr)
    y_ticks_pr = np.arange(0, R, y_step_pr)

    rows = len(z_pairs)
    gt_slices: list[np.ndarray] = []
    pred_slices: list[np.ndarray] = []
    for row, (z_gt, z_pred) in enumerate(z_pairs):
        gt_img = gt_abs[:, :, z_pred].astype(np.float32, copy=False)
        pr_img = pred_abs[:, :, z_pred].astype(np.float32, copy=False)
        gt_slices.append(gt_img)
        pred_slices.append(pr_img)

        gt_min, gt_mean, gt_max, gt_nz, gt_std = _slice_stats(gt_img)
        pr_min, pr_mean, pr_max, pr_nz, pr_std = _slice_stats(pr_img)
        print(
            f"[final-act-compare][axial axis2_gt={z_gt} -> axis2_pred={z_pred}] "
            f"GT min/mean/max={gt_min:.3e}/{gt_mean:.3e}/{gt_max:.3e} std={gt_std:.3e} nz_frac={gt_nz:.3f} | "
            f"Pred min/mean/max={pr_min:.3e}/{pr_mean:.3e}/{pr_max:.3e} std={pr_std:.3e} nz_frac={pr_nz:.3f}",
            flush=True,
        )
        if np.isfinite(pr_std) and pr_std < 1e-8:
            print(
                f"[final-act-compare][WARN] Pred axial slice axis2_pred={z_pred} ist nahezu konstant (std={pr_std:.3e}).",
                flush=True,
            )

    finite_gt = gt_abs[np.isfinite(gt_abs)]
    if finite_gt.size > 0:
        vmax_abs = float(np.quantile(finite_gt, 0.995))
        if not np.isfinite(vmax_abs):
            vmax_abs = float(np.nanmax(finite_gt))
    else:
        try:
            vmax_abs = float(np.nanmax(gt_abs))
        except ValueError:
            vmax_abs = 1.0
    vmax_abs = max(vmax_abs, 1e-6)
    vmin_abs = 0.0
    def _render_shared_plot(
        gt_imgs: list[np.ndarray],
        pr_imgs: list[np.ndarray],
        file_path: Path,
        cbar_label: str,
        vmin: float,
        vmax: float,
        title_suffix: str,
    ):
        if not gt_imgs:
            return
        fig, axs = plt.subplots(rows, 2, figsize=(12, 4 * rows), constrained_layout=True)
        axs = np.asarray(axs)
        if axs.ndim == 1:
            axs = axs.reshape(1, 2)
        im_gt_first = None
        for row, (z_gt, z_pred) in enumerate(z_pairs):
            ax_gt = axs[row, 0]
            ax_pr = axs[row, 1]
            gt_img = gt_imgs[row]
            pr_img = pr_imgs[row]
            im_gt = ax_gt.imshow(
                gt_img,
                origin="upper",
                extent=[0, B - 1, A - 1, 0],
                aspect="equal",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            im_pr = ax_pr.imshow(
                pr_img,
                origin="upper",
                extent=[0, R - 1, R - 1, 0],
                aspect="equal",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            if im_gt_first is None:
                im_gt_first = im_gt
            ax_gt.set_title(f"GT axial (act) @ axis2={z_gt}{title_suffix}")
            ax_pr.set_title(f"Pred axial (act_pred) @ axis2_pred={z_pred} (from {z_gt}){title_suffix}")
            ax_gt.set_xlabel("axis1 (x-like)")
            ax_pr.set_xlabel("axis1 (x-like)")
            ax_gt.set_ylabel("axis0 (y-like)")
            ax_pr.set_ylabel("axis0 (y-like)")
            ax_gt.set_xticks(x_ticks_gt)
            ax_gt.set_yticks(y_ticks_gt)
            ax_pr.set_xticks(x_ticks_pr)
            ax_pr.set_yticks(y_ticks_pr)
        if im_gt_first is not None:
            cbar = fig.colorbar(im_gt_first, ax=axs.ravel().tolist(), shrink=0.92)
            cbar.set_label(cbar_label)
        fig.savefig(file_path, dpi=200)
        plt.close(fig)
        print(f"[final-act-compare] Saved {file_path.resolve()}", flush=True)

    _render_shared_plot(gt_slices, pred_slices, abs_shared_path, "Activity (absolute shared)", vmin_abs, vmax_abs, " [abs]")


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
    loss_fn=poisson_nll,
    poisson_rate_mode: str = "softplus_shift",
    poisson_rate_floor: float = 0.0,
    poisson_rate_floor_mode: str = "clamp",
    proj_loss_active: bool = True,
    pred_scale: float = 1.0,
    gain: Optional[torch.Tensor] = None,
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

            pred_ap_raw, _ = render_minibatch(generator, z_latent, ray_batch_ap, ct_context=ct_context)
            pred_pa_raw, _ = render_minibatch(generator, z_latent, ray_batch_pa, ct_context=ct_context)

            if loss_fn == poisson_nll:
                lambda_ap_used = compute_poisson_rate(pred_ap_raw, poisson_rate_mode, eps=1e-6)
                lambda_pa_used = compute_poisson_rate(pred_pa_raw, poisson_rate_mode, eps=1e-6)
            else:
                lambda_ap_used = pred_ap_raw
                lambda_pa_used = pred_pa_raw

            if proj_loss_active and gain is not None:
                lambda_ap_used = lambda_ap_used * gain
                lambda_pa_used = lambda_pa_used * gain
            pred_ap = lambda_ap_used
            pred_pa = lambda_pa_used
            if loss_fn == poisson_nll and proj_loss_active and float(poisson_rate_floor) > 0.0:
                pred_ap, _ = apply_poisson_rate_floor(pred_ap, poisson_rate_floor, poisson_rate_floor_mode)
                pred_pa, _ = apply_poisson_rate_floor(pred_pa, poisson_rate_floor, poisson_rate_floor_mode)

            target_ap = ap_flat_proc[0, idx_ap].unsqueeze(0)
            target_pa = pa_flat_proc[0, idx_pa].unsqueeze(0)
            subset_label = locals().get("name", None)
            subset_label = subset_label if subset_label is not None else "eval"
            ratio_used_ap = _ratio_stats(lambda_ap_used, target_ap)
            ratio_used_pa = _ratio_stats(lambda_pa_used, target_pa)
            scale_val = float(pred_scale) if (pred_scale is not None and math.isfinite(pred_scale)) else float("nan")
            should_log_eval = DEBUG_PRINTS or (subset_label == "test_all")
            if should_log_eval:
                print(
                    f"[scale-probe][eval][{subset_label}] pred_scale={scale_val:.3f} "
                    f"| pred_vs_target_used AP(sum/mean)={ratio_used_ap[0]:.3f}/{ratio_used_ap[1]:.3f} "
                    f"PA(sum/mean)={ratio_used_pa[0]:.3f}/{ratio_used_pa[1]:.3f}",
                    flush=True,
                )

            if (target_ap < 0).any() or (target_pa < 0).any():
                print("[WARN] Negative projection targets detected in eval.", flush=True)
            if loss_fn == poisson_nll:
                if not torch.isfinite(pred_ap).all() or not torch.isfinite(pred_pa).all():
                    raise RuntimeError("Non-finite lambda in Poisson projection loss (eval).")
                if (pred_ap <= 0).any() or (pred_pa <= 0).any():
                    raise RuntimeError("Non-positive lambda in Poisson projection loss (eval).")

            weight_ap = build_loss_weights(target_ap, bg_weight, weight_threshold)
            weight_pa = build_loss_weights(target_pa, bg_weight, weight_threshold)

            # ---------------------------------------------------------------------------
            loss_ap = loss_fn(pred_ap, target_ap, weight=weight_ap)
            loss_pa = loss_fn(pred_pa, target_pa, weight=weight_pa)
            loss_total = 0.5 * (loss_ap + loss_pa)

            metrics_ap = compute_projection_metrics(pred_ap, target_ap)
            metrics_pa = compute_projection_metrics(pred_pa, target_pa)

            phys_metrics = None
            if scale_ap is not None and scale_pa is not None:
                scale_ap_f = float(scale_ap)
                scale_pa_f = float(scale_pa)
                pred_ap_phys = pred_ap * scale_ap_f
                pred_pa_phys = pred_pa * scale_pa_f
                target_ap_phys = target_ap * scale_ap_f
                target_pa_phys = target_pa * scale_pa_f
                metrics_phys_ap = compute_projection_metrics(pred_ap_phys, target_ap_phys)
                metrics_phys_pa = compute_projection_metrics(pred_pa_phys, target_pa_phys)
                phys_metrics = {
                    "mae": 0.5 * (metrics_phys_ap["mae"] + metrics_phys_pa["mae"]),
                    "rmse": 0.5 * (metrics_phys_ap["rmse"] + metrics_phys_pa["rmse"]),
                    "nll": 0.5 * (metrics_phys_ap["nll"] + metrics_phys_pa["nll"]),
                    "dev": 0.5 * (metrics_phys_ap["dev"] + metrics_phys_pa["dev"]),
                    "view": {
                        "ap": metrics_phys_ap,
                        "pa": metrics_phys_pa,
                    },
                }

            results[name] = {
                "loss": loss_total.item(),
                "loss_ap": loss_ap.item(),
                "loss_pa": loss_pa.item(),
                "mae": 0.5 * (metrics_ap["mae"] + metrics_pa["mae"]),
                "rmse": 0.5 * (metrics_ap["rmse"] + metrics_pa["rmse"]),
                "nll": 0.5 * (metrics_ap["nll"] + metrics_pa["nll"]),
                "dev": 0.5 * (metrics_ap["dev"] + metrics_pa["dev"]),
                "pred_mean": ((float(pred_ap.mean()), float(pred_pa.mean()))),
                "target_mean": ((float(target_ap.mean()), float(target_pa.mean()))),
                "view": {
                    "ap": {"loss": loss_ap.item(), **metrics_ap},
                    "pa": {"loss": loss_pa.item(), **metrics_pa},
                },
                "phys": phys_metrics,
            }

    if prev_flag:
        generator.eval()
    else:
        generator.train()

    return results


def _prepare_val_batch_for_eval(batch, generator, device, args):
    ap = batch["ap"].to(device, non_blocking=True).float()
    pa = batch["pa"].to(device, non_blocking=True).float()
    if (ap.shape[-2], ap.shape[-1]) != (generator.H, generator.W):
        raise ValueError(
            f"Val batch shape {tuple(ap.shape[-2:])} inconsistent with generator image size "
            f"({generator.H}, {generator.W})."
        )
    meta = batch.get("meta")
    meta_scale = _extract_meta_scalar(meta, "proj_scale_joint_p99")
    if meta_scale is None or not math.isfinite(meta_scale):
        scale_joint_used = 1.0
    else:
        scale_joint_used = float(meta_scale)

    ap_counts = batch.get("ap_counts")
    pa_counts = batch.get("pa_counts")
    if ap_counts is not None and ap_counts.numel() > 0:
        ap_counts = ap_counts.to(device, non_blocking=True).float()
    else:
        ap_counts = None
    if pa_counts is not None and pa_counts.numel() > 0:
        pa_counts = pa_counts.to(device, non_blocking=True).float()
    else:
        pa_counts = None

    batch_size_val = ap.shape[0] if ap.dim() >= 3 else 1
    if batch_size_val != 1:
        raise ValueError("Val loader currently expects batch_size == 1.")

    use_counts = (
        (ap_counts is not None)
        and (pa_counts is not None)
        and ap_counts.numel() > 0
        and pa_counts.numel() > 0
    )
    target_ap_full = ap_counts if use_counts else ap
    target_pa_full = pa_counts if use_counts else pa
    ap_flat_proc = target_ap_full.reshape(batch_size_val, -1)
    pa_flat_proc = target_pa_full.reshape(batch_size_val, -1)

    pred_to_counts_scale = scale_joint_used if use_counts else 1.0
    pred_to_counts_override = float(args.pred_to_counts_scale_override)
    if use_counts and pred_to_counts_override > 0:
        pred_to_counts_scale = pred_to_counts_override
    if use_counts:
        scale_ap_used = 1.0
        scale_pa_used = 1.0
    else:
        scale_ap_used = scale_joint_used
        scale_pa_used = scale_joint_used

    ct_vol = batch.get("ct")
    if ct_vol is not None and ct_vol.numel() > 0:
        ct_vol = ct_vol.to(device, non_blocking=True).float()
    else:
        ct_vol = None
    ct_context = generator.build_ct_context(ct_vol, padding_mode=args.ct_padding_mode) if ct_vol is not None else None

    return {
        "ap_flat_proc": ap_flat_proc,
        "pa_flat_proc": pa_flat_proc,
        "scale_ap": scale_ap_used,
        "scale_pa": scale_pa_used,
        "pred_to_counts_scale": pred_to_counts_scale,
        "ct_context": ct_context,
    }


def _aggregate_subset_metrics(metrics_list):
    if not metrics_list:
        return None

    def _mean(values):
        return float(np.mean(values)) if values else float("nan")

    aggregated = {
        "loss": _mean([m["loss"] for m in metrics_list]),
        "loss_ap": _mean([m["loss_ap"] for m in metrics_list]),
        "loss_pa": _mean([m["loss_pa"] for m in metrics_list]),
        "mae": _mean([m["mae"] for m in metrics_list]),
        "rmse": _mean([m["rmse"] for m in metrics_list]),
        "nll": _mean([m["nll"] for m in metrics_list]),
        "dev": _mean([m["dev"] for m in metrics_list]),
        "pred_mean": (
            _mean([m["pred_mean"][0] for m in metrics_list]),
            _mean([m["pred_mean"][1] for m in metrics_list]),
        ),
        "target_mean": (
            _mean([m["target_mean"][0] for m in metrics_list]),
            _mean([m["target_mean"][1] for m in metrics_list]),
        ),
        "view": {
            "ap": {
                "loss": _mean([m["view"]["ap"]["loss"] for m in metrics_list]),
                "mae": _mean([m["view"]["ap"]["mae"] for m in metrics_list]),
                "rmse": _mean([m["view"]["ap"]["rmse"] for m in metrics_list]),
                "nll": _mean([m["view"]["ap"]["nll"] for m in metrics_list]),
                "dev": _mean([m["view"]["ap"]["dev"] for m in metrics_list]),
            },
            "pa": {
                "loss": _mean([m["view"]["pa"]["loss"] for m in metrics_list]),
                "mae": _mean([m["view"]["pa"]["mae"] for m in metrics_list]),
                "rmse": _mean([m["view"]["pa"]["rmse"] for m in metrics_list]),
                "nll": _mean([m["view"]["pa"]["nll"] for m in metrics_list]),
                "dev": _mean([m["view"]["pa"]["dev"] for m in metrics_list]),
            },
        },
    }

    phys_list = [m.get("phys") for m in metrics_list if m.get("phys") is not None]
    if phys_list:
        aggregated["phys"] = {
            "mae": _mean([p["mae"] for p in phys_list]),
            "rmse": _mean([p["rmse"] for p in phys_list]),
            "nll": _mean([p["nll"] for p in phys_list]),
            "dev": _mean([p["dev"] for p in phys_list]),
            "view": {
                "ap": {
                    "mae": _mean([p["view"]["ap"]["mae"] for p in phys_list]),
                    "rmse": _mean([p["view"]["ap"]["rmse"] for p in phys_list]),
                    "nll": _mean([p["view"]["ap"]["nll"] for p in phys_list]),
                    "dev": _mean([p["view"]["ap"]["dev"] for p in phys_list]),
                },
                "pa": {
                    "mae": _mean([p["view"]["pa"]["mae"] for p in phys_list]),
                    "rmse": _mean([p["view"]["pa"]["rmse"] for p in phys_list]),
                    "nll": _mean([p["view"]["pa"]["nll"] for p in phys_list]),
                    "dev": _mean([p["view"]["pa"]["dev"] for p in phys_list]),
                },
            },
        }
    else:
        aggregated["phys"] = None

    return aggregated


def _format_val_metrics_keys(keys):
    return sorted(k.replace("test_", "val_", 1) if k.startswith("test_") else k for k in keys)


def evaluate_val_loader(
    val_loader,
    generator,
    z_latent,
    z_latent_base,
    rays_cache,
    subsets,
    device,
    args,
    loss_fn,
    proj_loss_active,
    rays_per_eval,
    bg_weight,
    weight_threshold,
    pa_xflip,
    W,
    gain,
    log_proj_metrics_physical,
    encoder: Optional[nn.Module] = None,
    z_fuser: Optional[nn.Module] = None,
    z_enc_alpha: float = 0.0,
):
    if val_loader is None or len(val_loader.dataset) == 0:
        print("[eval][warn] val_loader empty; skipping evaluation.", flush=True)
        return None

    proj_counts_active = bool(getattr(args, "hybrid", False)) and (args.proj_target_source == "counts")
    prev_encoder_training = encoder.training if encoder is not None else None
    if encoder is not None:
        encoder.eval()
    val_metrics = []
    try:
        for batch in val_loader:
            prepared = _prepare_val_batch_for_eval(batch, generator, device, args)
            z_eval = z_latent
            if bool(getattr(args, "hybrid", False)) and encoder is not None and z_latent_base is not None:
                z_eval, _, _ = build_hybrid_latent_from_batch(
                    args=args,
                    batch=batch,
                    device=device,
                    z_latent_base=z_latent_base,
                    encoder=encoder,
                    z_fuser=z_fuser,
                    z_enc_alpha=z_enc_alpha,
                )
            batch_stats = evaluate_pixel_subsets(
                generator,
                z_eval,
                rays_cache,
                subsets,
                prepared["ap_flat_proc"],
                prepared["pa_flat_proc"],
                rays_per_eval,
                bg_weight,
                weight_threshold,
                pa_xflip,
                ct_context=prepared["ct_context"],
                W=W,
                scale_ap=prepared["scale_ap"] if log_proj_metrics_physical else None,
                scale_pa=prepared["scale_pa"] if log_proj_metrics_physical else None,
                loss_fn=loss_fn,
                poisson_rate_mode=args.poisson_rate_mode,
                poisson_rate_floor=args.poisson_rate_floor,
                poisson_rate_floor_mode=args.poisson_rate_floor_mode,
                proj_loss_active=proj_loss_active,
                pred_scale=1.0,
                gain=gain,
            )
            val_metrics.append(batch_stats)
    finally:
        if encoder is not None and prev_encoder_training is not None:
            if prev_encoder_training:
                encoder.train()
            else:
                encoder.eval()

    if not val_metrics:
        print("[eval][warn] val_loader produced no batches; skipping evaluation.", flush=True)
        return None

    aggregated = {}
    subset_names = set().union(*(m.keys() for m in val_metrics if m is not None))
    for subset in subset_names:
        subset_list = [m.get(subset) for m in val_metrics if m is not None and m.get(subset) is not None]
        aggregated[subset] = _aggregate_subset_metrics(subset_list)

    return aggregated


def sample_act_points(
    act: torch.Tensor,
    nsamples: int,
    radius_xyz: tuple[float, float, float],
    pos_fraction: float = 0.5,
    pos_threshold: float = 1e-8,
    return_indices: bool = False,
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

    rx, ry, rz = radius_xyz
    coords = torch.stack(
        (
            idx_to_coord(x_idx, W, rx),
            idx_to_coord(y_idx, H, ry),
            idx_to_coord(z_idx, D, rz),
        ),
        dim=1,
    )
    if return_indices:
        coord_indices = torch.stack((z_idx, y_idx, x_idx), dim=1)
        return coords, values, pos_flags, coord_indices
    return coords, values, pos_flags


def query_emission_at_points(
    generator, z_latent, coords: torch.Tensor, return_raw: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fragt das NeRF an frei gewählten Koordinaten ab (ohne Integration)."""
    if coords.numel() == 0:
        empty = torch.tensor([], device=coords.device)
        return (empty, empty) if return_raw else empty
    render_kwargs = generator.render_kwargs_train
    network_fn = render_kwargs["network_fn"]
    network_query_fn = render_kwargs["network_query_fn"]
    pts = coords.unsqueeze(0)
    raw = network_query_fn(pts, None, network_fn, features=z_latent)
    raw = raw.view(-1, raw.shape[-1])
    signal = raw[:, 0]
    pred = activity_from_raw(signal)
    if return_raw:
        return pred, signal
    return pred


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
    global ACT_DEBUG_WORLD_COORDS, ACT_DEBUG_WORLD_STEP, ACT_DEBUG_WORLD_REPORTED
    print(f"▶ {__VERSION__} – starte Training", flush=True)
    args = parse_args()
    global DEBUG_PRINTS
    DEBUG_PRINTS = bool(args.debug_prints)
    hybrid_enabled = bool(args.hybrid)

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
    debug_dataset_summary = bool(data_cfg.get("debug_dataset_summary", False))
    training_cfg = config.setdefault("training", {})
    training_cfg.setdefault("val_interval", 0)
    training_cfg.setdefault("tv_weight", 0.001)
    training_cfg["tv_weight"] = args.tv_weight
    training_cfg.setdefault("ray_tv_weight", 0.0)
    training_cfg["ray_tv_weight"] = args.ray_tv_weight
    training_cfg.setdefault("lambda_ray_tv_weight", 0.0)
    training_cfg["lambda_ray_tv_weight"] = args.lambda_ray_tv_weight
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
    training_cfg.setdefault("run_final_test", True)
    training_cfg.setdefault("debug_eval_flow", False)
    training_cfg.setdefault("act_samples", 16384)
    training_cfg.setdefault("act_pos_weight", 2.0)
    training_cfg.setdefault("act_pos_fraction", 0.5)
    training_cfg.setdefault("act_pos_threshold", 1e-8)
    training_cfg.setdefault("act_loss_weight", args.act_loss_weight)
    cli_act_loss_flag = "--act-loss-weight" in sys.argv
    if cli_act_loss_flag:
        training_cfg["act_loss_weight"] = float(args.act_loss_weight)
    else:
        args.act_loss_weight = float(training_cfg.get("act_loss_weight", args.act_loss_weight))
        training_cfg["act_loss_weight"] = args.act_loss_weight
    if args.act_samples is None:
        args.act_samples = int(training_cfg.get("act_samples", 16384))
    else:
        training_cfg["act_samples"] = args.act_samples
    if args.act_pos_weight is None:
        args.act_pos_weight = float(training_cfg.get("act_pos_weight", 2.0))
    else:
        training_cfg["act_pos_weight"] = args.act_pos_weight
    if args.act_pos_fraction is None:
        args.act_pos_fraction = float(training_cfg.get("act_pos_fraction", 0.5))
    else:
        training_cfg["act_pos_fraction"] = args.act_pos_fraction
    if args.act_pos_threshold is None:
        args.act_pos_threshold = float(training_cfg.get("act_pos_threshold", 1e-8))
    else:
        training_cfg["act_pos_threshold"] = args.act_pos_threshold
    training_cfg.setdefault("proj_warmup_steps", args.proj_warmup_steps)
    args.proj_warmup_steps = int(training_cfg.get("proj_warmup_steps", args.proj_warmup_steps))
    training_cfg["proj_warmup_steps"] = args.proj_warmup_steps
    training_cfg.setdefault("ct_loss_weight", 0.0)
    training_cfg.setdefault("ct_threshold", 0.05)
    training_cfg.setdefault("ct_samples", 8192)
    training_cfg["ct_loss_weight"] = args.ct_loss_weight
    training_cfg["ct_threshold"] = args.ct_threshold
    training_cfg["ct_samples"] = args.ct_samples
    if hybrid_enabled and args.act_loss_weight <= 0.0:
        print("[WARN] Hybrid aktiv, aber --act-loss-weight <= 0: ACT-Hauptloss ist deaktiviert.", flush=True)
    if hybrid_enabled and "ct_prefer_raw" not in data_cfg:
        data_cfg["ct_prefer_raw"] = True

    print(f"📂 CWD: {Path.cwd().resolve()}", flush=True)
    outdir = Path(config.get("training", {}).get("outdir", "./results_spect")).expanduser().resolve()
    _log_runtime_provenance()
    (outdir / "preview").mkdir(parents=True, exist_ok=True)
    print(f"🗂️ Output-Ordner: {outdir}", flush=True)
    log_effective_config(outdir, config, args)
    print(f"[cfg][check] training.act_loss_weight={args.act_loss_weight}", flush=True)
    ckpt_dir = outdir / "checkpoints"
    log_path = outdir / "train_log.csv"
    log_path = init_log_file(log_path)
    hybrid_log_path = None
    if hybrid_enabled:
        hybrid_log_path = outdir / "hybrid_stats.csv"
        init_hybrid_log_file(hybrid_log_path)

    dataset = None
    try:
        dataset, hwfr, _ = get_data(config)
    except Exception as exc:
        if args.smoke_test:
            print(f"[smoke-test] get_data failed ({exc.__class__.__name__}): using synthetic batch.", flush=True)
            hwfr = build_hwfr_from_config(data_cfg)
        else:
            raise
    config["data"]["hwfr"] = hwfr
    data_cfg = config["data"]
    radius_xyz_raw = data_cfg.get("radius_xyz_cm")
    radius_xyz_cm = None if radius_xyz_raw is None else tuple(float(r) for r in radius_xyz_raw)
    auto_near_far = bool(data_cfg.get("auto_near_far_from_radius", True))
    if auto_near_far and (radius_xyz_cm is not None):
        data_cfg["near"] = 0.0
        data_cfg["far"] = 2.0 * radius_xyz_cm[2]
        print(
            "[cfg][data] auto_near_far_from_radius enabled → near/far set to "
            f"{data_cfg['near']:.3f}cm / {data_cfg['far']:.3f}cm (rz={radius_xyz_cm[2]:.3f}cm)",
            flush=True,
        )
    if radius_xyz_cm is not None:
        print(
            "[cfg][data] anisotropic radius_xyz_cm available: "
            f"{radius_xyz_cm} (use_anisotropic_radius={data_cfg.get('use_anisotropic_radius')})",
            flush=True,
        )
    path_length_cm = float(data_cfg.get("far", 0.0)) - float(data_cfg.get("near", 0.0))
    atten_scale = float(config.get("nerf", {}).get("atten_scale", 25.0))
    print(
        "[cfg][attenuation] path_length_cm="
        f"{path_length_cm:.3f}cm | atten_scale={atten_scale:.3f} | "
        "units=cm",
        flush=True,
    )
    if abs(atten_scale - 1.0) > 1e-3 and path_length_cm > 0.0:
        print(
            "[WARN][attenuation] atten_scale != 1.0; with cm-world and far-near representing physical length, "
            "consider setting atten_scale=1.0 for direct ∫μ ds.",
            flush=True,
        )

    if debug_dataset_summary and dataset is not None:
        print_dataset_summary(dataset, max_print=20)

    batch_size = config["training"]["batch_size"]
    if batch_size != 1:
        raise ValueError("This mini-training script currently assumes batch_size == 1.")

    train_loader = None
    val_loader = None
    test_loader = None
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=config["training"]["nworkers"],
        pin_memory=True,
        drop_last=False,
    )

    split_flag = bool(data_cfg.get("split_by_patient_id", False))
    dataset_split_stats = None
    if dataset is not None:
        if split_flag:
            split_seed = int(data_cfg.get("split_seed", 0))
            split_train = float(data_cfg.get("split_train", 0.8))
            split_val = float(data_cfg.get("split_val", 0.1))
            split_test = float(data_cfg.get("split_test", 0.1))
            split_mode = str(data_cfg.get("split_mode", "ratios")).lower()
            split_train_count = int(data_cfg.get("split_train_count", -1))
            split_val_count = int(data_cfg.get("split_val_count", -1))
            split_test_count = int(data_cfg.get("split_test_count", -1))
            (
                train_subset,
                val_subset,
                test_subset,
                split_stats,
            ) = split_by_patient_id(
                dataset,
                seed=split_seed,
                split_mode=split_mode,
                train_ratio=split_train,
                val_ratio=split_val,
                test_ratio=split_test,
                train_count=split_train_count,
                val_count=split_val_count,
                test_count=split_test_count,
            )
            print("[debug][dataset] split stats:", flush=True)
            for key, val in split_stats.items():
                print(f"   {key} = {val}", flush=True)
            dataset_split_stats = split_stats
            split_info = {
                "seed": split_seed,
                "mode": split_mode,
                "train_ids": split_stats.get("patient_ids_train_full", []),
                "val_ids": split_stats.get("patient_ids_val_full", []),
                "test_ids": split_stats.get("patient_ids_test_full", []),
                "n_samples_train": split_stats.get("n_samples_train"),
                "n_samples_val": split_stats.get("n_samples_val"),
                "n_samples_test": split_stats.get("n_samples_test"),
                "n_patients_train": split_stats.get("n_patients_train"),
                "n_patients_val": split_stats.get("n_patients_val"),
                "n_patients_test": split_stats.get("n_patients_test"),
            }
            split_path = outdir / "split.json"
            split_path.write_text(json.dumps(split_info, indent=2))

            train_loader = DataLoader(
                train_subset,
                shuffle=True,
                **loader_kwargs,
            )
            if len(val_subset) > 0:
                val_loader = DataLoader(
                    val_subset,
                    shuffle=False,
                    **loader_kwargs,
                )
            if len(test_subset) > 0:
                test_loader = DataLoader(
                    test_subset,
                    shuffle=False,
                    **loader_kwargs,
                )
        else:
            train_loader = DataLoader(
                dataset,
                shuffle=True,
                **loader_kwargs,
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
    debug_eval_flow = bool(training_cfg.get("debug_eval_flow", False))
    n_val_batches_cfg = "NA"
    if val_loader is not None and hasattr(val_loader, "__len__"):
        try:
            n_val_batches_cfg = len(val_loader)
        except Exception:
            n_val_batches_cfg = "NA"
    run_final_test = bool(training_cfg.get("run_final_test", True)) and (not args.no_final_test)
    print(
        f"[cfg][training] val_interval={val_interval} no_val={args.no_val} max_steps={args.max_steps} "
        f"split_by_patient_id={split_flag} n_val_batches={n_val_batches_cfg} "
        f"run_final_test={run_final_test} debug_eval_flow={debug_eval_flow}",
        flush=True,
    )
    tv_weight = float(training_cfg.get("tv_weight", 0.0))
    ray_tv_weight = float(training_cfg.get("ray_tv_weight", 0.0))
    lambda_ray_tv_weight = float(training_cfg.get("lambda_ray_tv_weight", 0.0))
    ray_tv_edge_aware = bool(training_cfg.get("ray_tv_edge_aware", False))
    ray_tv_alpha = float(training_cfg.get("ray_tv_alpha", 0.0))
    ray_tv_w_clamp_min = float(training_cfg.get("ray_tv_w_clamp_min", 0.0))
    depth_sanity_every = int(max(0, args.depth_sanity_every))
    depth_checks_active = depth_sanity_every > 0
    depth_grad_zero_streak = 0
    debug_sanity_checks = bool(getattr(args, "debug_sanity_checks", False))
    debug_sanity_every = int(max(1, getattr(args, "debug_sanity_every", 100)))
    geometry_checked = False

    generator = build_models(config)
    generator.to(device)
    generator.train()
    generator.use_test_kwargs = False  # enforce training kwargs

    def _depth_grad_norm(loss_term: torch.Tensor, module_candidate) -> float:
        def _local_grad_norm(loss_term_local: torch.Tensor, module_local: torch.nn.Module) -> float:
            if loss_term_local is None or not loss_term_local.requires_grad:
                return 0.0
            params = [p for p in module_local.parameters() if p.requires_grad]
            if not params:
                return 0.0
            grads = torch.autograd.grad(loss_term_local, params, retain_graph=True, allow_unused=True)
            grads = [g for g in grads if g is not None]
            if not grads:
                return 0.0
            flat = torch.cat([g.reshape(-1) for g in grads])
            return float(flat.norm().detach().cpu().item())

        target = (
            module_candidate
            if isinstance(module_candidate, torch.nn.Module)
            else (generator if isinstance(generator, torch.nn.Module) else None)
        )
        if target is None:
            print("[WARN][depth] Grad-Target ist kein nn.Module; skippe Depth-Grad-Norm.", flush=True)
            return 0.0
        if "grad_norm_of_module" in globals():
            try:
                return grad_norm_of_module(loss_term, target)
            except Exception as exc:
                print(
                    f"[WARN][depth] grad_norm_of_module failed ({exc.__class__.__name__}); nutze Fallback.",
                    flush=True,
                )
        return _local_grad_norm(loss_term, target)

    # always provide AP/PA fallback poses if not already configured
    generator.set_fixed_ap_pa(radius=hwfr[3])

    z_dim = config["z_dist"]["dim"]
    z_latent_base = torch.zeros(1, z_dim, device=device)
    encoder = None
    z_fuser = None
    z_enc_alpha = float(args.z_enc_alpha)
    gain_head = None
    gain_param = None
    if hybrid_enabled:
        enc_in_ch = 2 + (1 if args.encoder_use_ct else 0)
        encoder = ProjectionEncoder(in_ch=enc_in_ch, z_dim=z_dim, base_ch=32).to(device)
        z_fuser = nn.Sequential(nn.Linear(z_dim, z_dim), nn.LayerNorm(z_dim)).to(device)
        if args.proj_target_source == "counts":
            if args.proj_gain_source == "z_enc":
                gain_head = nn.Linear(z_dim, 1).to(device)
            elif args.proj_gain_source == "scalar":
                gain_param = nn.Parameter(torch.zeros(1, device=device))
        encoder.train()
        print(
            f"[hybrid] Encoder init: in_ch={enc_in_ch}, z_dim={z_dim} | z_enc_alpha={z_enc_alpha}",
            flush=True,
        )

    # --- Sofortiger Smoke-Test ---
    # Einmal vor dem eigentlichen Training rendern, um Setup/NaNs zu prüfen
    with torch.no_grad():
        generator.eval()
        generator.use_test_kwargs = True
        z_smoke = z_latent_base
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

    if ray_split_enabled and dataset is None:
        print("[smoke-test] dataset missing; disabling ray split.", flush=True)
        ray_split_enabled = False

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
            )
        score_img = np.maximum(ap_target_np, pa_target_np[:, ::-1] if pa_xflip else pa_target_np)
        _log_split(pixel_split_np, score_img, ray_split_mode)

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
    
    # --- Fix: Robuste Optimizer-Erstellung mit Parameter-Gruppen ---
    lr_g = config["training"]["lr_g"]
    param_groups = [{'params': generator.parameters(), 'lr': lr_g, 'name': 'generator'}]
    if hybrid_enabled:
        if encoder is not None:
            lr_enc = lr_g * float(args.encoder_lr_mult)
            param_groups.append({'params': encoder.parameters(), 'lr': lr_enc, 'name': 'encoder'})
            print(f"[cfg][optim] Encoder added to optimizer with lr={lr_enc:.3e}", flush=True)
        if z_fuser is not None:
            param_groups.append({'params': z_fuser.parameters(), 'lr': lr_g, 'name': 'z_fuser'})
        if gain_head is not None:
            param_groups.append({'params': gain_head.parameters(), 'lr': lr_g, 'name': 'gain_head'})
        if gain_param is not None:
            param_groups.append({'params': [gain_param], 'lr': lr_g, 'name': 'gain_param'})

    optimizer = torch.optim.Adam(param_groups)
    opt_params = []
    for group in optimizer.param_groups:
        opt_params.extend(group['params'])

    # (1) Instrumentierung: Optimizer-Check (einmalig)
    if args.debug_enc:
        print(f"[DEBUG][ENC_INIT] hybrid_enabled={hybrid_enabled}", flush=True)
        if hybrid_enabled:
            print(f"[DEBUG][ENC_INIT] encoder is None: {encoder is None}", flush=True)
            if encoder is not None:
                print(f"[DEBUG][ENC_INIT] encoder.training: {encoder.training}", flush=True)
                print(f"[DEBUG][ENC_INIT] z_enc_alpha: {z_enc_alpha}", flush=True)
                n_enc_params = sum(p.numel() for p in encoder.parameters())
                enc_ids = {id(p) for p in encoder.parameters()}
                opt_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
                intersection = len(enc_ids.intersection(opt_ids))
                print(f"[DEBUG][ENC_INIT] Encoder params total: {n_enc_params}", flush=True)
                print(f"[DEBUG][ENC_INIT] Encoder params in optimizer: {intersection}/{len(enc_ids)} (opt_total: {len(opt_ids)})", flush=True)
 
    # Projection-Loss
    proj_loss_type = args.proj_loss_type
    if hybrid_enabled:
        if args.proj_target_source == "counts" and proj_loss_type != "poisson":
            print("[WARN] counts target -> set proj_loss_type=poisson.", flush=True)
            proj_loss_type = "poisson"
        if args.proj_target_source == "norm" and proj_loss_type == "poisson":
            print("[WARN] norm target -> set proj_loss_type=sqrt_mse.", flush=True)
            proj_loss_type = "sqrt_mse"
    if proj_loss_type == "poisson":
        loss_fn = poisson_nll
    elif proj_loss_type == "huber":
        loss_fn = huber_loss
    else:
        loss_fn = sqrt_mse_loss

    amp_enabled = bool(config["training"].get("use_amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    loaded_step = 0
    if str(getattr(args, "checkpoint", "") or "").strip():
        loaded_step = load_checkpoint(
            Path(args.checkpoint),
            device=device,
            generator=generator,
            optimizer=optimizer,
            scaler=scaler,
            encoder=encoder,
            z_fuser=z_fuser,
            gain_head=gain_head,
            gain_param=gain_param,
            strict=bool(args.checkpoint_strict),
            load_optimizer=bool(args.checkpoint_load_optimizer),
        )
    _self_test_proj_weight_schedule()

    proj_weight_min_sched, proj_weight_max_sched, proj_bounds_swapped = sanitize_proj_weight_bounds(
        args.proj_weight_min,
        args.proj_loss_weight,
    )
    if proj_bounds_swapped:
        print(
            f"[WARN][proj-schedule] proj_weight_min ({args.proj_weight_min:.3e}) > "
            f"proj_loss_weight ({args.proj_loss_weight:.3e}); swapping bounds.",
            flush=True,
        )

    decoder_module = generator.render_kwargs_train.get("network_fn")
    if isinstance(decoder_module, nn.Module):
        decoder_params = [p for p in decoder_module.parameters() if p.requires_grad]
    else:
        decoder_params = [p for p in generator.parameters() if p.requires_grad]
    encoder_params = [p for p in encoder.parameters() if p.requires_grad] if encoder is not None else []
    z_fuser_params = [p for p in z_fuser.parameters() if p.requires_grad] if z_fuser is not None else []

    if args.smoke_test:
        batch = None
        if train_loader is not None:
            try:
                batch = next(iter(train_loader))
            except Exception as exc:
                print(f"[smoke-test] train_loader failed ({exc.__class__.__name__}); using synthetic batch.", flush=True)
        if batch is None:
            batch = build_synthetic_batch(generator.H, generator.W, device=device)
            ap = batch["ap"]
            pa = batch["pa"]
            ap_enc_input = batch.get("proj_input_ap", ap)
            pa_enc_input = batch.get("proj_input_pa", pa)
            meta = batch.get("meta")
            act_vol = batch.get("act")
            ct_vol = batch.get("ct")
        else:
            ap = batch["ap"].to(device, non_blocking=True).float()
            pa = batch["pa"].to(device, non_blocking=True).float()
            ap_enc_input = batch.get("proj_input_ap", ap)
            pa_enc_input = batch.get("proj_input_pa", pa)
            meta = batch.get("meta")
            act_vol = batch.get("act")
            if act_vol is not None and act_vol.numel() > 0:
                act_vol = act_vol.to(device, non_blocking=True)
            ct_vol = batch.get("ct")
            if ct_vol is not None and ct_vol.numel() > 0:
                ct_vol = ct_vol.to(device, non_blocking=True).float()
        if act_vol is not None and act_vol.numel() == 0:
            act_vol = None
        if ct_vol is not None and ct_vol.numel() == 0:
            ct_vol = None
        ct_context = generator.build_ct_context(ct_vol, padding_mode=args.ct_padding_mode) if ct_vol is not None else None
        if debug_sanity_checks:
            ap_counts_log = batch.get("ap_counts")
            pa_counts_log = batch.get("pa_counts")
            if ap_counts_log is not None and ap_counts_log.numel() > 0:
                ap_counts_log = ap_counts_log.to(device, non_blocking=True).float()
            else:
                ap_counts_log = None
            if pa_counts_log is not None and pa_counts_log.numel() > 0:
                pa_counts_log = pa_counts_log.to(device, non_blocking=True).float()
            else:
                pa_counts_log = None
            ap_min = float(ap.min().item())
            ap_mean = float(ap.mean().item())
            ap_max = float(ap.max().item())
            pa_min = float(pa.min().item())
            pa_mean = float(pa.mean().item())
            pa_max = float(pa.max().item())
            apc_min = float(ap_counts_log.min().item()) if ap_counts_log is not None else float("nan")
            apc_mean = float(ap_counts_log.mean().item()) if ap_counts_log is not None else float("nan")
            apc_max = float(ap_counts_log.max().item()) if ap_counts_log is not None else float("nan")
            pac_min = float(pa_counts_log.min().item()) if pa_counts_log is not None else float("nan")
            pac_mean = float(pa_counts_log.mean().item()) if pa_counts_log is not None else float("nan")
            pac_max = float(pa_counts_log.max().item()) if pa_counts_log is not None else float("nan")
            print(
                f"[sanity][proj-input] ap(min/mean/max)={ap_min:.3e}/{ap_mean:.3e}/{ap_max:.3e} "
                f"| pa(min/mean/max)={pa_min:.3e}/{pa_mean:.3e}/{pa_max:.3e} "
                f"| ap_counts(min/mean/max)={apc_min:.3e}/{apc_mean:.3e}/{apc_max:.3e} "
                f"| pa_counts(min/mean/max)={pac_min:.3e}/{pac_mean:.3e}/{pac_max:.3e}",
                flush=True,
            )
            # Interpretation hint:
            # ratio ≈ 1    -> counts already normalized
            # ratio ≈ p99  -> counts ≈ norm * proj_scale_joint_p99
            # ratio >> p99 -> counts are in different physical units
            eps = 1e-8
            ratio_ap = (apc_mean / (ap_mean + eps)) if ap_counts_log is not None else float("nan")
            ratio_pa = (pac_mean / (pa_mean + eps)) if pa_counts_log is not None else float("nan")
            print(
                f"[sanity][proj-ratio] ratio_ap={ratio_ap:.3e} | ratio_pa={ratio_pa:.3e}",
                flush=True,
            )

            z_base = z_latent_base
        if z_base.shape[0] != ap.shape[0]:
            z_base = z_base.expand(ap.shape[0], -1)
        z_enc = None
        if hybrid_enabled and encoder is not None:
            proj_scale_enc = compute_proj_scale(ap, pa, args.proj_scale_source, meta)
            proj_scale_enc = torch.clamp(proj_scale_enc, min=1e-6)
            enc_input = build_encoder_input(
                ap_enc_input,
                pa_enc_input,
                ct_vol,
                proj_scale_enc,
                args.encoder_proj_transform,
                args.encoder_use_ct,
            )
            z_enc = encoder(enc_input)
            if z_enc.shape[0] != z_base.shape[0]:
                z_base = z_base.expand(z_enc.shape[0], -1)
            z_enc_proj = z_fuser(z_enc) if z_fuser is not None else z_enc
            z_latent = z_base + (z_enc_alpha * z_enc_proj)
        else:
            z_latent = z_base

        z_latent_pre_dropout = z_latent.detach()
        latent_dropout_frac = 0.0
        dropout_prob = float(max(0.0, min(1.0, args.latent_dropout_prob)))
        if dropout_prob > 0.0 and generator.training:
            dropout_mask = torch.rand(z_latent.shape[0], device=z_latent.device) < dropout_prob
            latent_dropout_frac = float(dropout_mask.float().mean().item()) if z_latent.shape[0] > 0 else 0.0
            if dropout_mask.any():
                z_latent = z_latent.clone()
                z_latent[dropout_mask] = 0.0

        scale_joint_used = 1.0
        if isinstance(meta, dict):
            meta_scale = meta.get("proj_scale_joint_p99")
            if torch.is_tensor(meta_scale):
                meta_scale = meta_scale.item() if meta_scale.numel() > 0 else None
            if isinstance(meta_scale, (int, float)) and math.isfinite(meta_scale) and meta_scale > 0:
                scale_joint_used = float(meta_scale)

        idx_ap = torch.randperm(num_pixels, device=device)[:rays_per_proj]
        idx_pa = map_pa_indices_torch(idx_ap, W, pa_xflip)
        ray_batch_ap = slice_rays(rays_cache["ap"], idx_ap)
        ray_batch_pa = slice_rays(rays_cache["pa"], idx_pa)

        proj_weight = 1.0
        if hybrid_enabled:
            proj_weight = float(args.proj_loss_weight)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred_ap_raw, _ = render_minibatch(generator, z_latent, ray_batch_ap, ct_context=ct_context)
            pred_pa_raw, _ = render_minibatch(generator, z_latent, ray_batch_pa, ct_context=ct_context)
            ap_counts = batch.get("ap_counts")
            pa_counts = batch.get("pa_counts")
            use_counts = ap_counts is not None and ap_counts.numel() > 0 and pa_counts is not None and pa_counts.numel() > 0
            if use_counts:
                ap_counts = ap_counts.to(device, non_blocking=True).float()
                pa_counts = pa_counts.to(device, non_blocking=True).float()
                target_ap = ap_counts.reshape(ap_counts.shape[0], -1)[0, idx_ap].unsqueeze(0)
                target_pa = pa_counts.reshape(pa_counts.shape[0], -1)[0, idx_pa].unsqueeze(0)
            else:
                target_ap = ap.reshape(ap.shape[0], -1)[0, idx_ap].unsqueeze(0)
                target_pa = pa.reshape(pa.shape[0], -1)[0, idx_pa].unsqueeze(0)
            if debug_sanity_checks:
                tap_min = float(target_ap.min().item())
                tap_mean = float(target_ap.mean().item())
                tap_max = float(target_ap.max().item())
                tpa_min = float(target_pa.min().item())
                tpa_mean = float(target_pa.mean().item())
                tpa_max = float(target_pa.max().item())
                print(
                    f"[sanity][proj-target] target_ap(min/mean/max)={tap_min:.3e}/{tap_mean:.3e}/{tap_max:.3e} "
                    f"| target_pa(min/mean/max)={tpa_min:.3e}/{tpa_mean:.3e}/{tpa_max:.3e}",
                    flush=True,
                )

            pred_to_counts_scale = scale_joint_used if use_counts else 1.0
            pred_to_counts_override = float(args.pred_to_counts_scale_override)
            if use_counts and pred_to_counts_override > 0:
                pred_to_counts_scale = pred_to_counts_override
            if use_counts:
                print(
                    f"[scale][smoke] pred_to_counts_scale: orig={scale_joint_used:.3e} "
                    f"override={pred_to_counts_override:.3e} used={pred_to_counts_scale:.3e}",
                    flush=True,
                )
            if debug_sanity_checks:
                meta_val = float(meta_scale) if isinstance(meta_scale, (int, float)) and math.isfinite(meta_scale) else float("nan")
                override_val = pred_to_counts_override if pred_to_counts_override > 0 else float("nan")
                orig_val = float(scale_joint_used) if use_counts else float("nan")
                used_val = float(pred_to_counts_scale) if use_counts else float("nan")
                print(
                    f"[sanity][proj-scale] proj_scale_joint_p99={meta_val:.3e} "
                    f"| pred_to_counts_scale_orig={orig_val:.3e} "
                    f"| pred_to_counts_scale_override={override_val:.3e} "
                    f"| pred_to_counts_scale_used={used_val:.3e}",
                    flush=True,
                )

            if proj_loss_type == "poisson":
                pred_ap = compute_poisson_rate(pred_ap_raw, args.poisson_rate_mode, eps=1e-6)
                pred_pa = compute_poisson_rate(pred_pa_raw, args.poisson_rate_mode, eps=1e-6)
            else:
                pred_ap = pred_ap_raw
                pred_pa = pred_pa_raw
            if use_counts:
                if gain_head is not None and z_enc is not None:
                    gain_val = F.softplus(gain_head(z_enc))
                    pred_ap = pred_ap * gain_val
                    pred_pa = pred_pa * gain_val
                elif gain_param is not None:
                    gain_val = F.softplus(gain_param)
                    pred_ap = pred_ap * gain_val
                    pred_pa = pred_pa * gain_val
            if proj_loss_type == "poisson" and float(args.poisson_rate_floor) > 0.0:
                pred_ap, _ = apply_poisson_rate_floor(pred_ap, args.poisson_rate_floor, args.poisson_rate_floor_mode)
                pred_pa, _ = apply_poisson_rate_floor(pred_pa, args.poisson_rate_floor, args.poisson_rate_floor_mode)
            loss_ap = loss_fn(pred_ap, target_ap)
            loss_pa = loss_fn(pred_pa, target_pa)
            loss_proj = 0.5 * (loss_ap + loss_pa)
            if hybrid_enabled:
                loss = proj_weight * loss_proj
            else:
                loss = loss_proj
            loss_act = torch.tensor(0.0, device=device)
            if args.act_loss_weight > 0.0 and act_vol is not None:
                act_radius_xyz = _resolve_act_radius_xyz_from_act(
                    generator, act_vol, batch.get("meta"), step=step
                )
                if args.debug_sanity_checks:
                    act_weights = act_vol.clamp_min(0.0)
                    act_stats = _world_stats_from_weights(act_weights, act_radius_xyz)
                    mask_frac = None
                    ct_stats: tuple[
                        Optional[tuple[float, float, float]],
                        Optional[tuple[float, float, float]],
                        Optional[tuple[float, float, float]],
                    ] = (None, None, None)
                    if ct_vol is not None and ct_vol.numel() > 0:
                        ct_threshold = float(args.ct_threshold) if args.ct_threshold is not None else 0.0
                        if ct_threshold > 0.0:
                            ct_mask = ct_vol > ct_threshold
                        else:
                            ct_mask = ct_vol > 0
                        has_mask = bool(ct_mask.any().item())
                        if not has_mask:
                            ct_mask = ct_vol > 0
                            has_mask = bool(ct_mask.any().item())
                        if has_mask:
                            mask_frac = float(ct_mask.float().mean().item())
                            ct_radius_xyz = _resolve_activity_radius_xyz(generator)
                            ct_stats = _world_stats_from_weights(ct_mask.float(), ct_radius_xyz)
                    if mask_frac is not None:
                        _print_align_check(act_stats, ct_stats, mask_frac=mask_frac)
                coords, act_samples, pos_flags = sample_act_points(
                    act_vol,
                    args.act_samples,
                    radius_xyz=act_radius_xyz,
                    pos_fraction=args.act_pos_fraction,
                    pos_threshold=args.act_pos_threshold,
                )
                pred_act, pred_act_raw = query_emission_at_points(generator, z_latent, coords, return_raw=True)
                if pred_act.numel() > 0:
                    act_norm_factor, _ = compute_act_norm_factor(
                        act_vol, args.act_norm_source, args.act_norm_value, None
                    )
                    act_norm_factor = float(act_norm_factor)
                    pred_pos = pred_act_raw.clamp_min(0.0) / max(act_norm_factor, 1e-8)
                    act_pos = act_samples.clamp_min(0.0) / max(act_norm_factor, 1e-8)
                    pred_log = torch.log1p(pred_pos)
                    act_log = torch.log1p(act_pos)
                    weights_act = torch.where(
                        pos_flags,
                        torch.full_like(pred_log, args.act_pos_weight),
                        torch.ones_like(pred_log),
                    )
                    diff = F.smooth_l1_loss(pred_log, act_log, reduction="none")
                    loss_act = torch.mean(weights_act * diff)
                    scale_loss = torch.tensor(0.0, device=device)
                    if args.act_scale_reg_weight > 0.0:
                        pred_sum = pred_pos.sum()
                        target_sum = act_pos.sum()
                        scale_diff = pred_sum - target_sum
                        scale_loss = args.act_scale_reg_weight * scale_diff * scale_diff
                    loss = loss + args.act_loss_weight * loss_act + scale_loss
                    need_act_scale_log = args.log_every > 0 and (step == 1 or (step % args.log_every) == 0)
                    if need_act_scale_log:
                        pred_sum_val = float(pred_pos.sum().detach().item())
                        target_sum_val = float(act_pos.sum().detach().item())
                        pred_mean_val = float(pred_pos.mean().detach().item())
                        target_mean_val = float(act_pos.mean().detach().item())
                        scale_loss_val = float(scale_loss.detach().item()) if args.act_scale_reg_weight > 0.0 else 0.0
                        print(
                            f"[act-scale][step {step}] pred_sum={pred_sum_val:.3e} mean={pred_mean_val:.3e} "
                            f"target_sum={target_sum_val:.3e} mean={target_mean_val:.3e} "
                            f"scale_loss={scale_loss_val:.3e}",
                            flush=True,
                        )

        if args.debug_z_sensitivity and debug_z_sample is None:
            debug_z_sample = {
                "z_latent": z_latent_pre_dropout.clone(),
                "ray_batch_ap": ray_batch_ap.detach().clone(),
                "ray_batch_pa": ray_batch_pa.detach().clone(),
                "ct_context": ct_context,
                "target_ap": target_ap.detach().clone(),
                "target_pa": target_pa.detach().clone(),
                "pred_ap_normal": pred_ap.detach().clone(),
                "pred_pa_normal": pred_pa.detach().clone(),
                "proj_loss_type": args.proj_loss_type,
                "poisson_rate_mode": args.poisson_rate_mode,
                "poisson_rate_floor": float(args.poisson_rate_floor),
                "poisson_rate_floor_mode": args.poisson_rate_floor_mode,
                "use_counts": use_counts,
                "gain_val": gain_val.detach().clone() if gain_val is not None else None,
                "gain_param": gain_param.detach().clone() if gain_param is not None else None,
                "act_coords": coords.detach().clone() if ('coords' in locals() and coords is not None) else None,
                "act_samples": act_samples.detach().clone() if ('act_samples' in locals() and act_samples is not None) else None,
                "act_pred": pred_act.detach().clone() if ('pred_act' in locals() and pred_act is not None) else None,
                "act_pred_raw": pred_act_raw.detach().clone() if ('pred_act_raw' in locals() and pred_act_raw is not None) else None,
                "latent_dropout_frac": latent_dropout_frac,
                "amp_enabled": amp_enabled,
            }
        scaler.scale(loss).backward()
        print(
            f"[smoke-test] loss={loss.item():.6f} | proj={loss_proj.item():.6f} | act={loss_act.item():.6f} "
            f"| pred_ap shape={tuple(pred_ap.shape)} pred_pa shape={tuple(pred_pa.shape)} "
            f"| finite_pred={torch.isfinite(pred_ap).all().item() and torch.isfinite(pred_pa).all().item()}",
            flush=True,
        )
        return

    data_iter = iter(train_loader) if train_loader is not None else iter([])
    ct_context = None
    max_steps_cfg = None
    if isinstance(training_cfg, dict):
        max_steps_cfg = training_cfg.get("max_steps")
    if max_steps_cfg is None:
        max_steps_cfg = config.get("max_steps")
    if max_steps_cfg is not None and args.max_steps is not None and args.max_steps > 0:
        if int(max_steps_cfg) != int(args.max_steps):
            print(
                f"[cfg] max_steps in config={int(max_steps_cfg)} -> CLI override (using {int(args.max_steps)})",
                flush=True,
            )
    max_steps = int(args.max_steps)
    last_step = 0
    exit_reason = "normal"

    def _signal_handler(signum, frame):
        nonlocal exit_reason, last_step
        exit_reason = "signal"
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        print(
            f"[signal] received {signame} last_step={last_step} max_steps={max_steps}",
            flush=True,
        )
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    print(
        f"🚀 Starting emission-NeRF training | steps={max_steps} | rays/proj={rays_per_proj} "
        f"| image={generator.H}x{generator.W} | chunk={generator.chunk} | ckpt_step={loaded_step}"
    )
    scale_ap_used = 1.0
    scale_pa_used = 1.0
    scale_joint_used = 1.0
    scale_missing_warned = False
    counts_missing_warned = False
    act_norm_global = None
    last_act_tv_value = 0.0
    last_z_latent = z_latent_base
    # Default values for inference-only runs (e.g. --max-steps 0), where the train loop is skipped.
    proj_loss_active = (not bool(args.act_only)) and (not (bool(args.hybrid) and float(args.proj_loss_weight) <= 0.0))
    ap = None
    pa = None
    ap_counts = None
    pa_counts = None
    ct_vol = None
    act_vol = None
    ct_context = None
    ap_enc_input = None
    pa_enc_input = None
    meta = None
    debug_z_sample = None
    last_gain_val = None
    gain_prior_ema = None
    gain_prior_final = None
    gain_prior_decay = 0.9
    gain_prior_steps = 50
    proj_collapse_count = 0

    last_patient_id: Optional[str] = None
    last_safe_patient_id: str = "unknown"

    print(
        f"[sanity] max_steps={max_steps} proj_warmup_steps={args.proj_warmup_steps} "
        f"proj_ramp_steps={max(0, int(args.proj_ramp_steps))} "
        f"proj_weight_min/max={proj_weight_min_sched:.3e}/{proj_weight_max_sched:.3e} "
        f"depth_sanity_every={args.depth_sanity_every}",
        flush=True,
    )
    try:
        for step in range(1, max_steps + 1):
            last_step = step
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
    
            ap = batch["ap"].to(device, non_blocking=True).float()
            pa = batch["pa"].to(device, non_blocking=True).float()
            ap_enc_input = batch.get("proj_input_ap", ap)
            if torch.is_tensor(ap_enc_input):
                ap_enc_input = ap_enc_input.to(device, non_blocking=True).float()
            pa_enc_input = batch.get("proj_input_pa", pa)
            if torch.is_tensor(pa_enc_input):
                pa_enc_input = pa_enc_input.to(device, non_blocking=True).float()

            meta = batch.get("meta")
            meta_scale = None
            meta_missing = False
            if isinstance(meta, dict):
                meta_scale = meta.get("proj_scale_joint_p99")
                meta_missing = meta.get("proj_scale_joint_p99_missing", False)
                if isinstance(meta_scale, (list, tuple)):
                    meta_scale = meta_scale[0] if meta_scale else None
                if torch.is_tensor(meta_scale):
                    meta_scale = meta_scale.item() if meta_scale.numel() > 0 else None
                if torch.is_tensor(meta_missing):
                    meta_missing = bool(meta_missing.item()) if meta_missing.numel() > 0 else False
            current_patient_id = _extract_patient_id_from_batch(batch)
            safe_patient_id = _slugify_patient_id(current_patient_id)
            last_patient_id = current_patient_id
            last_safe_patient_id = safe_patient_id
            if debug_dataset_summary and step == 1:
                meta_obj = batch.get("meta") if isinstance(batch, dict) else None
                path_hint = None
                if isinstance(meta_obj, dict) and "patient_id" in meta_obj:
                    path_hint = "batch['meta']['patient_id']"
                elif isinstance(meta_obj, (list, tuple)) and meta_obj and isinstance(meta_obj[0], dict) and "patient_id" in meta_obj[0]:
                    path_hint = "batch['meta'][0]['patient_id']"
                pid = current_patient_id
                meta_type = type(meta_obj).__name__ if meta_obj is not None else "None"
                print(
                    f"[debug][dataset] batch meta type={meta_type} | patient_id={pid} | path_hint={path_hint}",
                    flush=True,
                )
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
            ct_vol = batch.get("ct")
            if ct_vol is not None:
                ct_vol = ct_vol.to(device, non_blocking=True).float()
            ct_context = generator.build_ct_context(ct_vol, padding_mode=args.ct_padding_mode) if ct_vol is not None else None
    
            # Wichtig: Flatten-Order ist (y * W + x), identisch zu den Ray-Indizes aus make_stratified_tile_split.
            # Keine permute/transpose zwischen (H, W) und reshape(-1), damit Target/Predict exakt die gleiche Reihenfolge teilen.
            ap_flat = ap.reshape(batch_size, -1)
            pa_flat = pa.reshape(batch_size, -1)
            ap_counts = batch.get("ap_counts")
            pa_counts = batch.get("pa_counts")
            if ap_counts is not None and ap_counts.numel() > 0:
                ap_counts = ap_counts.to(device, non_blocking=True).float()
            else:
                ap_counts = None
            if pa_counts is not None and pa_counts.numel() > 0:
                pa_counts = pa_counts.to(device, non_blocking=True).float()
            else:
                pa_counts = None
            if debug_sanity_checks and step == 1:
                ap_min = float(ap.min().item())
                ap_mean = float(ap.mean().item())
                ap_max = float(ap.max().item())
                pa_min = float(pa.min().item())
                pa_mean = float(pa.mean().item())
                pa_max = float(pa.max().item())
                apc_min = float(ap_counts.min().item()) if ap_counts is not None else float("nan")
                apc_mean = float(ap_counts.mean().item()) if ap_counts is not None else float("nan")
                apc_max = float(ap_counts.max().item()) if ap_counts is not None else float("nan")
                pac_min = float(pa_counts.min().item()) if pa_counts is not None else float("nan")
                pac_mean = float(pa_counts.mean().item()) if pa_counts is not None else float("nan")
                pac_max = float(pa_counts.max().item()) if pa_counts is not None else float("nan")
                print(
                    f"[sanity][proj-input] ap(min/mean/max)={ap_min:.3e}/{ap_mean:.3e}/{ap_max:.3e} "
                    f"| pa(min/mean/max)={pa_min:.3e}/{pa_mean:.3e}/{pa_max:.3e} "
                    f"| ap_counts(min/mean/max)={apc_min:.3e}/{apc_mean:.3e}/{apc_max:.3e} "
                    f"| pa_counts(min/mean/max)={pac_min:.3e}/{pac_mean:.3e}/{pac_max:.3e}",
                    flush=True,
                )
                # Interpretation hint:
                # ratio ≈ 1    -> counts already normalized
                # ratio ≈ p99  -> counts ≈ norm * proj_scale_joint_p99
                # ratio >> p99 -> counts are in different physical units
                eps = 1e-8
                ratio_ap = (apc_mean / (ap_mean + eps)) if ap_counts is not None else float("nan")
                ratio_pa = (pac_mean / (pa_mean + eps)) if pa_counts is not None else float("nan")
                print(
                    f"[sanity][proj-ratio] ratio_ap={ratio_ap:.3e} | ratio_pa={ratio_pa:.3e}",
                    flush=True,
                )
    
            use_counts = (
                (ap_counts is not None)
                and (pa_counts is not None)
                and (ap_counts.numel() > 0)
                and (pa_counts.numel() > 0)
            )
            if not use_counts and not counts_missing_warned:
                print("[WARN] Projection-Loss wuerde normierte AP/PA nutzen (Counts fehlen).", flush=True)
                counts_missing_warned = True
            if proj_loss_type == "poisson" and not use_counts:
                raise RuntimeError("Poisson projection loss requires count targets (ap_counts/pa_counts).")
            if use_counts and proj_loss_type != "poisson":
                raise RuntimeError("Counts targets require Poisson projection loss.")
    
            target_ap_full = ap_counts if use_counts else ap
            target_pa_full = pa_counts if use_counts else pa
            ap_flat_proc = target_ap_full.reshape(batch_size, -1)
            pa_flat_proc = target_pa_full.reshape(batch_size, -1)
    
            pred_to_counts_scale = scale_joint_used if use_counts else 1.0
            pred_to_counts_orig = pred_to_counts_scale
            pred_to_counts_override = float(args.pred_to_counts_scale_override)
            if use_counts and pred_to_counts_override > 0:
                pred_to_counts_scale = pred_to_counts_override
            if use_counts:
                scale_ap_used = 1.0
                scale_pa_used = 1.0
            else:
                scale_ap_used = scale_joint_used
                scale_pa_used = scale_joint_used
            if debug_sanity_checks and step == 1:
                meta_val = float(meta_scale) if isinstance(meta_scale, (int, float)) and math.isfinite(meta_scale) else float("nan")
                override_val = pred_to_counts_override if pred_to_counts_override > 0 else float("nan")
                orig_val = float(pred_to_counts_orig) if use_counts else float("nan")
                used_val = float(pred_to_counts_scale) if use_counts else float("nan")
                print(
                    f"[sanity][proj-scale] proj_scale_joint_p99={meta_val:.3e} "
                    f"| pred_to_counts_scale_orig={orig_val:.3e} "
                    f"| pred_to_counts_scale_override={override_val:.3e} "
                    f"| pred_to_counts_scale_used={used_val:.3e}",
                    flush=True,
                )
            if use_counts and step == 1:
                print(
                    f"[scale] pred_to_counts_scale: orig={scale_joint_used:.3e} "
                    f"override={pred_to_counts_override:.3e} used={pred_to_counts_scale:.3e}",
                    flush=True,
                )
    
            z_base = z_latent_base
            if z_base.shape[0] != ap.shape[0]:
                z_base = z_base.expand(ap.shape[0], -1)
            z_enc = None
            z_enc_proj = None
            proj_scale_enc = None
            if hybrid_enabled and encoder is not None:
                proj_scale_enc = compute_proj_scale(ap, pa, args.proj_scale_source, meta)
                proj_scale_enc = torch.clamp(proj_scale_enc, min=1e-6)
                global _ENCODER_PIPELINE_TRAIN_LOGGED
                latent_stats_every = max(1, int(args.debug_latent_stats_every))
                should_log_encoder_pipeline = (
                    bool(args.debug_latent_stats)
                    and (step == 1 or (step % latent_stats_every) == 0)
                    and (step == 1 or (not _ENCODER_PIPELINE_TRAIN_LOGGED))
                )
                if should_log_encoder_pipeline:
                    _ENCODER_PIPELINE_TRAIN_LOGGED = True
                    _log_encoder_pipeline_debug(
                        "train",
                        ap_enc_input,
                        pa_enc_input,
                        args.encoder_proj_transform,
                        args.proj_scale_source,
                    )
                enc_input = build_encoder_input(
                    ap_enc_input,
                    pa_enc_input,
                    ct_vol,
                    proj_scale_enc,
                    args.encoder_proj_transform,
                    args.encoder_use_ct,
                )
                z_enc = encoder(enc_input)
                if z_enc.shape[0] != z_base.shape[0]:
                    z_base = z_base.expand(z_enc.shape[0], -1)
                if z_fuser is not None:
                    z_enc_proj = z_fuser(z_enc)
                else:
                    z_enc_proj = z_enc
                z_latent = z_base + (z_enc_alpha * z_enc_proj)

            # (2) Instrumentierung: z_enc/z_latent Graph-Check (step 1)
            if args.debug_enc and step == 1 and z_enc is not None:
                print(f"[DEBUG][ENC_FWD] z_enc.requires_grad={z_enc.requires_grad}, grad_fn is not None: {z_enc.grad_fn is not None}", flush=True)
                print(f"[DEBUG][ENC_FWD] z_latent.requires_grad={z_latent.requires_grad}, grad_fn is not None: {z_latent.grad_fn is not None}", flush=True)
                s = (z_latent * torch.randn_like(z_latent)).sum()
                g_z_enc = torch.autograd.grad(s, z_enc, retain_graph=True, allow_unused=True)[0]
                print(f"[DEBUG][ENC_FWD] grad(z_latent, z_enc) is None: {g_z_enc is None}", flush=True)
                if g_z_enc is not None:
                    print(f"[DEBUG][ENC_FWD] grad(z_latent, z_enc) norm: {g_z_enc.norm().item():.6e}", flush=True)
            last_z_latent = z_latent
            latent_stats_every = max(1, int(args.debug_latent_stats_every))
            should_log_latent_stats = (
                bool(args.debug_latent_stats)
                and hybrid_enabled
                and (step == 1 or (step % latent_stats_every) == 0)
            )
            if should_log_latent_stats:
                _log_latent_conditioning_stats(
                    step=step,
                    z_base=z_base,
                    z_enc=z_enc,
                    z_enc_proj=z_enc_proj,
                    z_latent=z_latent,
                    alpha=z_enc_alpha,
                )

            # Phase A: ACT-only training.
            # Wenn args.act_loss_weight > 0 und alle projektionsrelevanten Gewichte effektiv 0 sind
            # (proj_loss_weight == 0, ray_tv_weight == 0, bg_depth_mass_weight == 0)
            # dann: Kein Projektions-Forward.
            phase_a_active = (
                args.act_loss_weight > 0.0
                and args.proj_loss_weight == 0.0
                and args.ray_tv_weight == 0.0
                and args.bg_depth_mass_weight == 0.0
            )
            skip_proj = bool(args.act_only) or phase_a_active
            debug_act_step = bool(args.debug_act and step == 1)
            if not skip_proj and hybrid_enabled:
                proj_weight_used, proj_loss_active_sched, proj_warmup_active, proj_ramp_t = compute_proj_weight_schedule(
                    step=step,
                    warmup_steps=args.proj_warmup_steps,
                    ramp_steps=args.proj_ramp_steps,
                    weight_min=proj_weight_min_sched,
                    weight_max=proj_weight_max_sched,
                )
            else:
                proj_warmup_active = bool(args.proj_warmup_steps > 0 and step <= args.proj_warmup_steps)
                proj_ramp_t = 1.0 if not proj_warmup_active else 0.0
                proj_weight_used = 0.0 if proj_warmup_active else (1.0 if not skip_proj else 0.0)
                proj_loss_active_sched = not proj_warmup_active

            proj_loss_active = (
                (not skip_proj)
                and (proj_loss_type == "poisson")
                and proj_loss_active_sched
                and (not hybrid_enabled or proj_weight_used > 0.0)
            )
            proj_metrics_enabled = proj_loss_active
            if args.log_every > 0 and (step == 1 or (step % args.log_every) == 0):
                print(
                    f"[proj-status][step {step}] proj_loss_active={proj_loss_active} "
                    f"proj_warmup_active={proj_warmup_active} proj_weight_used={proj_weight_used:.3e} "
                    f"ramp_t={proj_ramp_t:.4f} min={proj_weight_min_sched:.3e} max={proj_weight_max_sched:.3e} "
                    f"warmup={int(args.proj_warmup_steps)} ramp={max(0, int(args.proj_ramp_steps))} "
                    f"skip_proj={skip_proj}",
                    flush=True,
                )

            if debug_sanity_checks and (not geometry_checked) and step == 1:
                geometry_checked = True
                try:
                    prev_flag = generator.use_test_kwargs
                    generator.eval()
                    generator.use_test_kwargs = True
                    with torch.no_grad():
                        proj_ap_full, _, _, _ = generator.render_from_pose(z_latent.detach(), generator.pose_ap, ct_context=ct_context)
                        proj_pa_full, _, _, _ = generator.render_from_pose(z_latent.detach(), generator.pose_pa, ct_context=ct_context)
                    generator.train()
                    generator.use_test_kwargs = prev_flag or False

                    H, W = generator.H, generator.W
                    pred_ap_full = proj_ap_full.view(1, -1)
                    pred_pa_full = proj_pa_full.view(1, -1)
                    if proj_loss_type == "poisson" and proj_loss_active:
                        pred_ap_full = compute_poisson_rate(pred_ap_full, args.poisson_rate_mode, eps=1e-6)
                        pred_pa_full = compute_poisson_rate(pred_pa_full, args.poisson_rate_mode, eps=1e-6)
                        gain_val_dbg = None
                        if gain_head is not None and z_enc is not None:
                            gain_val_dbg = F.softplus(gain_head(z_enc))
                        elif gain_param is not None:
                            gain_val_dbg = F.softplus(gain_param)
                        if gain_val_dbg is not None:
                            g_min = float(args.gain_clamp_min) if args.gain_clamp_min is not None else None
                            g_max = args.gain_clamp_max
                            if g_min is not None or g_max is not None:
                                gmin = g_min if g_min is not None else -float("inf")
                                gmax = g_max if g_max is not None else float("inf")
                                gain_val_dbg = torch.clamp(gain_val_dbg, min=gmin, max=gmax)
                            pred_ap_full = pred_ap_full * gain_val_dbg
                            pred_pa_full = pred_pa_full * gain_val_dbg

                    tgt_ap_full = target_ap_full.reshape(1, -1)
                    tgt_pa_full = target_pa_full.reshape(1, -1)

                    def _corr(a, b):
                        a = a.reshape(-1).float()
                        b = b.reshape(-1).float()
                        a = a - a.mean()
                        b = b - b.mean()
                        denom = a.std() * b.std() + 1e-8
                        return float((a * b).mean().item() / denom.item())

                    def _nmse(a, b):
                        a = a.reshape(-1).float()
                        b = b.reshape(-1).float()
                        num = torch.mean((a - b) ** 2)
                        den = torch.mean(b ** 2) + 1e-12
                        return float((num / den).item())

                    pa_img = tgt_pa_full.view(H, W)
                    pa_img_flip = torch.flip(pa_img, dims=[1])
                    pa_flip_flat = pa_img_flip.reshape(1, -1)
                    corr_ap = _corr(pred_ap_full, tgt_ap_full)
                    corr_pa = _corr(pred_pa_full, tgt_pa_full)
                    corr_pa_flip = _corr(pred_pa_full, pa_flip_flat)
                    nmse_ap = _nmse(pred_ap_full, tgt_ap_full)
                    nmse_pa = _nmse(pred_pa_full, tgt_pa_full)
                    nmse_pa_flip = _nmse(pred_pa_full, pa_flip_flat)
                    print(
                        f"[sanity][geometry] corr_ap={corr_ap:.3f} | corr_pa={corr_pa:.3f} | corr_pa_flip={corr_pa_flip:.3f} "
                        f"| nmse_ap={nmse_ap:.3e} | nmse_pa={nmse_pa:.3e} | nmse_pa_flip={nmse_pa_flip:.3e}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[sanity][geometry][WARN] check failed: {exc.__class__.__name__}: {exc}", flush=True)
    
            optimizer.zero_grad(set_to_none=True)
            t0 = time.perf_counter()

            idx_ap = None
            idx_pa = None
            ray_batch_ap = None
            ray_batch_pa = None
            need_ray_tv = (not skip_proj) and ray_tv_weight != 0.0
            need_lambda_ray_tv = proj_loss_active and lambda_ray_tv_weight != 0.0
            need_bg_depth = (not skip_proj) and args.bg_depth_mass_weight > 0.0
            need_raw_stats = proj_metrics_enabled and hybrid_enabled and args.log_every > 0 and (step % args.log_every == 0 or step == 1)
            need_raw = (
                need_ray_tv
                or need_lambda_ray_tv
                or need_bg_depth
                or need_raw_stats
                or depth_checks_active
                or debug_sanity_checks
            )

            if not skip_proj:
                if ray_split_enabled and pixel_split_np is not None and rng_train is not None:
                    idx_np = sample_train_indices(pixel_split_np, rays_per_proj, ray_train_fg_frac, rng_train)
                    idx_ap = torch.from_numpy(idx_np).long().to(device, non_blocking=True)
                    idx_pa = map_pa_indices_torch(idx_ap, W, pa_xflip)
                else:
                    idx_ap = sample_split_indices(ray_indices["pixel"]["train_idx_all"], rays_per_proj)
                    idx_pa = map_pa_indices_torch(idx_ap, W, pa_xflip)

                ray_batch_ap = slice_rays(rays_cache["ap"], idx_ap)
                ray_batch_pa = slice_rays(rays_cache["pa"], idx_pa)
    
            loss = torch.tensor(0.0, device=device)
            loss_ap = torch.tensor(0.0, device=device)
            loss_pa = torch.tensor(0.0, device=device)
            loss_proj = torch.tensor(0.0, device=device)
            pred_ap = None
            pred_pa = None
            pred_ap_raw = None
            pred_pa_raw = None
            target_ap = None
            target_pa = None
            extras_ap = None
            extras_pa = None
            gain_val = None
            gain_val_used = None
            lambda_ap_used = None
            lambda_pa_used = None
            lambda_floor_frac = None
    
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                if not skip_proj:
                    pred_ap, extras_ap = render_minibatch(
                        generator,
                        z_latent,
                        ray_batch_ap,
                        ct_context=ct_context,
                        return_raw=need_raw,
                        debug_sanity_checks=bool(args.debug_sanity_checks),
                    )
                    pred_pa, extras_pa = render_minibatch(
                        generator,
                        z_latent,
                        ray_batch_pa,
                        ct_context=ct_context,
                        return_raw=need_raw,
                        debug_sanity_checks=bool(args.debug_sanity_checks),
                    )
    
                    target_ap = ap_flat_proc[0, idx_ap].unsqueeze(0)
                    target_pa = pa_flat_proc[0, idx_pa].unsqueeze(0)
                    if debug_sanity_checks and step == 1:
                        tap_min = float(target_ap.min().item())
                        tap_mean = float(target_ap.mean().item())
                        tap_max = float(target_ap.max().item())
                        tpa_min = float(target_pa.min().item())
                        tpa_mean = float(target_pa.mean().item())
                        tpa_max = float(target_pa.max().item())
                        print(
                            f"[sanity][proj-target] target_ap(min/mean/max)={tap_min:.3e}/{tap_mean:.3e}/{tap_max:.3e} "
                            f"| target_pa(min/mean/max)={tpa_min:.3e}/{tpa_mean:.3e}/{tpa_max:.3e}",
                            flush=True,
                        )
    
                    pred_ap_raw = pred_ap
                    pred_pa_raw = pred_pa
                    for extras in (extras_ap, extras_pa):
                        if not isinstance(extras, dict):
                            continue
                        if extras.get("lambda_ray") is not None and extras.get("lambda_ray_pre_act") is not None:
                            continue
                        raw_out = extras.get("raw")
                        if raw_out is None:
                            continue
                        if raw_out.dim() >= 3:
                            raw_lambda = raw_out[..., 0]
                        elif raw_out.dim() == 2:
                            raw_lambda = raw_out
                        else:
                            continue
                        if extras.get("lambda_ray_pre_act") is None:
                            extras["lambda_ray_pre_act"] = raw_lambda
                        if extras.get("lambda_ray") is None:
                            extras["lambda_ray"] = F.softplus(raw_lambda)

                    if proj_loss_type == "poisson":
                        if proj_loss_active:
                            lambda_ap_used = compute_poisson_rate(pred_ap_raw, args.poisson_rate_mode, eps=1e-6)
                            lambda_pa_used = compute_poisson_rate(pred_pa_raw, args.poisson_rate_mode, eps=1e-6)
                        else:
                            lambda_ap_used = pred_ap_raw
                            lambda_pa_used = pred_pa_raw
                    else:
                        lambda_ap_used = pred_ap_raw
                        lambda_pa_used = pred_pa_raw
    
                    gain_val_raw = None
                    gain_val = None
                    gain_val_used = None
                    
                    # Always compute robust estimate for logging/fallback
                    # Use full sum (no mask) to match loss inputs and debug logs
                    sum_tgt = target_ap.sum() + target_pa.sum()
                    sum_pred_no_gain = lambda_ap_used.sum() + lambda_pa_used.sum()
                    
                    if sum_pred_no_gain > 1e-6:
                        gain_est = sum_tgt / (sum_pred_no_gain + 1e-8)
                    else:
                        gain_est = torch.tensor(1.0, device=device)

                    if use_counts:
                        if args.use_gain:
                            # 1. Learnable Gain Calculation
                            if gain_head is not None and z_enc is not None:
                                gain_raw = gain_head(z_enc)
                                gain_val_raw = F.softplus(gain_raw)
                            elif gain_param is not None:
                                gain_val_raw = F.softplus(gain_param)

                            # 2. Fallback Logic
                            is_finite_learnable = (gain_val_raw is not None) and torch.isfinite(gain_val_raw).all()
                            
                            if is_finite_learnable:
                                gain_val_used = gain_val_raw
                            else:
                                gain_val_used = gain_est
                            
                            # Warmup override
                            if proj_warmup_active and args.gain_warmup_mode != "none":
                                fixed_gain = 1.0 if args.gain_warmup_mode == "one" else float(args.gain_prior_value)
                                if gain_val_used is not None:
                                    gain_val_used = gain_val_used.new_full(gain_val_used.shape, fixed_gain)
                                else:
                                    gain_val_used = torch.tensor(fixed_gain, device=device)
                        else:
                            # Gain disabled -> 1.0
                            gain_val_used = torch.tensor(1.0, device=device)
                            gain_val_raw = torch.tensor(1.0, device=device)
                        
                        gain_val = gain_val_used

                        if proj_loss_active and args.use_gain:
                            if gain_val_used is not None:
                                g_min = float(args.gain_clamp_min) if args.gain_clamp_min is not None else None
                                g_max = args.gain_clamp_max
                                if g_min is not None or g_max is not None:
                                    gmin = g_min if g_min is not None else -float("inf")
                                    gmax = g_max if g_max is not None else float("inf")
                                    gain_val_used = torch.clamp(gain_val_used, min=gmin, max=gmax)
                                lambda_ap_used = lambda_ap_used * gain_val_used
                                lambda_pa_used = lambda_pa_used * gain_val_used

                        if step == 1:
                            sum_pred_with_gain = lambda_ap_used.sum() + lambda_pa_used.sum()
                            g_raw_v = gain_val_raw.mean().item() if (gain_val_raw is not None) else -1.0
                            g_used_v = gain_val_used.mean().item() if (gain_val_used is not None) else 1.0
                            print(f"[DEBUG][gain] use_gain={args.use_gain} sum_tgt={sum_tgt.item():.2f} "
                                  f"sum_pred_no_gain={sum_pred_no_gain.item():.2f} "
                                  f"sum_pred_with_gain={sum_pred_with_gain.item():.2f} "
                                  f"gain_est={gain_est.item():.4f} gain_learnable_mean={g_raw_v:.4f} "
                                  f"gain_used_final={g_used_v:.4f}", flush=True)
                    if args.log_every > 0 and (step == 1 or (step % args.log_every) == 0):
                        ratio_used_ap = _ratio_stats(lambda_ap_used, target_ap)
                        ratio_used_pa = _ratio_stats(lambda_pa_used, target_pa)
                        counts_ap_str = "NA/NA"
                        counts_pa_str = "NA/NA"
                        if (
                            use_counts
                            and ap_counts is not None
                            and ap_counts.numel() > 0
                            and pa_counts is not None
                            and pa_counts.numel() > 0
                        ):
                            target_ap_counts = ap_counts.reshape(ap_counts.shape[0], -1)[0, idx_ap].unsqueeze(0)
                            target_pa_counts = pa_counts.reshape(pa_counts.shape[0], -1)[0, idx_pa].unsqueeze(0)
                            counts_ap = _ratio_stats(lambda_ap_used, target_ap_counts)
                            counts_pa = _ratio_stats(lambda_pa_used, target_pa_counts)
                            counts_ap_str = f"{counts_ap[0]:.3f}/{counts_ap[1]:.3f}"
                            counts_pa_str = f"{counts_pa[0]:.3f}/{counts_pa[1]:.3f}"
                        pts = float(pred_to_counts_scale) if pred_to_counts_scale is not None else float("nan")
                        msg = (
                            f"[scale-probe][train][step {step}] use_counts={use_counts} "
                            f"pred_to_counts_scale={pts:.3e} "
                            f"| pred_vs_target_used AP(sum/mean)={ratio_used_ap[0]:.3f}/{ratio_used_ap[1]:.3f} "
                            f"PA(sum/mean)={ratio_used_pa[0]:.3f}/{ratio_used_pa[1]:.3f}"
                        )
                        if counts_ap_str != "NA/NA":
                            msg += (
                                " | pred_vs_counts "
                                f"AP(sum/mean)={counts_ap_str} PA(sum/mean)={counts_pa_str}"
                            )
                        print(msg, flush=True)
                    lambda_ap_eff = lambda_ap_used
                    lambda_pa_eff = lambda_pa_used
                    if proj_loss_type == "poisson" and proj_loss_active and float(args.poisson_rate_floor) > 0.0:
                        lambda_ap_eff, floor_ap = apply_poisson_rate_floor(
                            lambda_ap_eff, args.poisson_rate_floor, args.poisson_rate_floor_mode
                        )
                        lambda_pa_eff, floor_pa = apply_poisson_rate_floor(
                            lambda_pa_eff, args.poisson_rate_floor, args.poisson_rate_floor_mode
                        )
                        if floor_ap is not None and floor_pa is not None:
                            lambda_floor_frac = 0.5 * (float(floor_ap) + float(floor_pa))
                    pred_ap = lambda_ap_eff
                    pred_pa = lambda_pa_eff
                    last_gain_val = gain_val_used if gain_val_used is not None else gain_val
    
                    if proj_loss_active and ((target_ap < 0).any() or (target_pa < 0).any()):
                        print("[WARN] Negative projection targets detected.", flush=True)
                    if proj_loss_active and proj_loss_type == "poisson":
                        if not torch.isfinite(pred_ap).all() or not torch.isfinite(pred_pa).all():
                            raise RuntimeError("Non-finite lambda in Poisson projection loss.")
                        if (pred_ap <= 0).any() or (pred_pa <= 0).any():
                            raise RuntimeError("Non-positive lambda in Poisson projection loss.")
                        if use_counts:
                            if (target_ap <= 1.0).all() and (target_pa <= 1.0).all():
                                print("[WARN] Count targets appear in [0,1] range; check scaling.", flush=True)
                            if step == 1:
                                p95_ap = float(torch.quantile(pred_ap, 0.95).item()) if pred_ap.numel() > 0 else float("nan")
                                p95_pa = float(torch.quantile(pred_pa, 0.95).item()) if pred_pa.numel() > 0 else float("nan")
                                if p95_ap < 5 or p95_ap > 1e6 or p95_pa < 5 or p95_pa > 1e6:
                                    print("[WARN] Lambda p95 outside expected count scale (5..1e6).", flush=True)
                        if step % 100 == 0:
                            pred_std = float(pred_ap.std().item())
                            target_std = float(target_ap.std().item())
                            if pred_std < 1e-6 and target_std > 1e-3:
                                proj_collapse_count += 1
                                pred_mean = float(pred_ap.mean().item())
                                pred_min = float(pred_ap.min().item())
                                pred_max = float(pred_ap.max().item())
                                target_mean = float(target_ap.mean().item())
                                gain_pre = float(gain_val_raw.mean().item()) if gain_val_raw is not None else float("nan")
                                gain_post_tensor = gain_val_used if gain_val_used is not None else gain_val
                                gain_post = float(gain_post_tensor.mean().item()) if gain_post_tensor is not None else float("nan")
                                print(
                                    "[WARN][proj] Projection lambda collapsed: pred std ~0 while target std is large.",
                                    flush=True,
                                )
                                print(
                                    f"[WARN][proj] pred_std={pred_std:.3e} target_std={target_std:.3e} "
                                    f"| pred_mean={pred_mean:.3e} pred_min/max=({pred_min:.3e},{pred_max:.3e}) "
                                    f"| target_mean={target_mean:.3e} "
                                    f"| pred_to_counts_scale={pred_to_counts_scale:.3e} "
                                    f"| gain_pre={gain_pre:.3e} gain_post={gain_post:.3e} "
                                    f"| collapse_streak={proj_collapse_count} patience={args.proj_collapse_patience}",
                                    flush=True,
                                )
                                if args.proj_collapse_patience > 0 and proj_collapse_count >= args.proj_collapse_patience:
                                    raise RuntimeError(
                                        "Projection lambda collapsed: pred std ~0 while target std is large "
                                        "(patience exceeded)."
                                    )
                            else:
                                proj_collapse_count = 0

                    if debug_sanity_checks and (step == 1 or (step % debug_sanity_every) == 0):
                        atten_scale = float(generator.render_kwargs_train.get("atten_scale", ATTEN_SCALE_DEFAULT))
                        print(
                            f"[sanity][step {step}] near={generator.render_kwargs_train.get('near')} "
                            f"| far={generator.render_kwargs_train.get('far')} | radius={generator.radius} "
                            f"| atten_scale={atten_scale}",
                            flush=True,
                        )
                        log_attenuation_sanity(
                            step,
                            [extras_ap, extras_pa],
                            atten_scale=atten_scale,
                            label="train",
                        )
                        if proj_loss_type == "poisson" and (pred_ap is not None) and (lambda_ap_used is not None):
                            lambda_stats = tensor_stats(lambda_ap_used)
                            lambda_eff_stats = tensor_stats(pred_ap)
                            floor_active = proj_loss_active and float(args.poisson_rate_floor) > 0.0
                            floor_frac = float(lambda_floor_frac) if lambda_floor_frac is not None else 0.0
                            print(
                                f"[sanity][step {step}] lambda(min/mean/p95/max)={fmt_stats(lambda_stats)} "
                                f"| lambda_eff(min/mean/p95/max)={fmt_stats(lambda_eff_stats)} "
                                f"| floor={float(args.poisson_rate_floor):.3e} "
                                f"| floor_mode={args.poisson_rate_floor_mode} "
                                f"| floor_active={bool(floor_active)} "
                                f"| floor_frac={floor_frac:.3f} "
                                f"| proj_warmup_active={bool(proj_warmup_active)} "
                                f"| proj_weight_used={float(proj_weight_used):.3e} "
                                f"| proj_loss_active={bool(proj_loss_active)}",
                                flush=True,
                            )
                        pre_vals = []
                        post_vals = []
                        for extras in (extras_ap, extras_pa):
                            if not isinstance(extras, dict):
                                continue
                            pre = extras.get("lambda_ray_pre_act")
                            post = extras.get("lambda_ray")
                            if pre is not None:
                                pre_vals.append(pre.reshape(-1))
                            if post is not None:
                                post_vals.append(post.reshape(-1))
                        if pre_vals or post_vals:
                            pre_stats = tensor_stats(torch.cat(pre_vals, dim=0)) if pre_vals else None
                            post_stats = tensor_stats(torch.cat(post_vals, dim=0)) if post_vals else None
                            print(
                                f"[sanity][step {step}] lambda_ray_pre_act(min/mean/p95/max)={fmt_stats(pre_stats)} "
                                f"| lambda_ray(min/mean/p95/max)={fmt_stats(post_stats)}",
                                flush=True,
                            )
                        if use_counts:
                            gain_min = float(args.gain_warn_min)
                            gain_max = float(args.gain_warn_max)
                            gain_stats = None
                            if gain_val is not None:
                                gain_stats = tensor_stats(gain_val)
                            scale_gain_stats = None
                            if proj_loss_active and gain_val_used is not None:
                                scale_gain_stats = tensor_stats(gain_val_used)
                            print(
                                f"[sanity][step {step}] pred_to_counts_scale={pred_to_counts_scale:.3e} "
                                f"| gain(min/mean/p95/max)={fmt_stats(gain_stats)} "
                                f"| gain_used(min/mean/p95/max)={fmt_stats(scale_gain_stats)}",
                                flush=True,
                            )
                            gain_raw_mean = float(gain_val_raw.mean().item()) if gain_val_raw is not None else float("nan")
                            gain_used_mean = float(gain_val.mean().item()) if gain_val is not None else float("nan")
                            gain_clamped_mean = (
                                float(gain_val_used.mean().item()) if gain_val_used is not None else float("nan")
                            )
                            print(
                                f"[sanity][step {step}] gain_raw={gain_raw_mean:.3e} "
                                f"| gain_used={gain_used_mean:.3e} "
                                f"| gain_clamped={gain_clamped_mean:.3e}",
                                flush=True,
                            )
                            if gain_stats is not None:
                                gmin = gain_stats["min"]
                                gmax = gain_stats["max"]
                                if (gmin < gain_min) or (gmax > gain_max):
                                    print(
                                        f"[sanity][step {step}] WARN: gain outside [{gain_min:.3e},{gain_max:.3e}] "
                                        f"(min={gmin:.3e}, max={gmax:.3e}).",
                                        flush=True,
                                    )
    
                    weight_ap = None
                    weight_pa = None
                    if proj_loss_active:
                        weight_ap = build_loss_weights(target_ap, args.bg_weight, args.weight_threshold)
                        weight_pa = build_loss_weights(target_pa, args.bg_weight, args.weight_threshold)
    
                    if proj_loss_active and step in (1, 50):
                        pred_raw_mean = float(pred_ap_raw.mean().item())
                        pred_raw_std = float(pred_ap_raw.std().item())
                        pred_mean = float(pred_ap.mean().item())
                        pred_std = float(pred_ap.std().item())
                        target_mean = float(target_ap.mean().item())
                        target_std = float(target_ap.std().item())
                        gain_pre = float(gain_val_raw.mean().item()) if gain_val_raw is not None else float("nan")
                        gain_post_tensor = gain_val_used if gain_val_used is not None else gain_val
                        gain_post = float(gain_post_tensor.mean().item()) if gain_post_tensor is not None else float("nan")
                        print(
                            f"[DEBUG][proj][step {step}] pred_raw_mean={pred_raw_mean:.3e} pred_raw_std={pred_raw_std:.3e} "
                            f"| pred_mean={pred_mean:.3e} pred_std={pred_std:.3e} "
                            f"| target_mean={target_mean:.3e} target_std={target_std:.3e} "
                            f"| sum_pred={pred_ap.sum().item():.3e} sum_target={target_ap.sum().item():.3e} "
                            f"| pred_to_counts_scale={pred_to_counts_scale:.3e} "
                            f"| gain_pre={gain_pre:.3e} gain_post={gain_post:.3e}",
                            flush=True,
                        )
    
                    if proj_loss_active:
                        loss_ap = loss_fn(pred_ap, target_ap, weight=weight_ap)
                        loss_pa = loss_fn(pred_pa, target_pa, weight=weight_pa)
                        loss_proj = 0.5 * (loss_ap + loss_pa)
                        if step == 1:
                            tmin = float(target_ap.min().item()) if target_ap.numel() > 0 else float("nan")
                            tmax = float(target_ap.max().item()) if target_ap.numel() > 0 else float("nan")
                            lmin = float(pred_ap.min().item()) if pred_ap.numel() > 0 else float("nan")
                            lmax = float(pred_ap.max().item()) if pred_ap.numel() > 0 else float("nan")
                            print(
                                f"[DEBUG][proj][step 1] use_counts={use_counts} "
                                f"| target_min/max=({tmin:.3e},{tmax:.3e}) "
                                f"| lambda_min/max=({lmin:.3e},{lmax:.3e})",
                                flush=True,
                            )
                        if hybrid_enabled:
                            if proj_weight_used > 0.0:
                                loss = loss + proj_weight_used * loss_proj
                        else:
                            loss = loss_proj
                        if DEBUG_PRINTS and (step % 50 == 0):
                            print(
                                f"[DEBUG][step {step}] TARGET AP min/max: {target_ap.min().item():.3e}/{target_ap.max().item():.3e} | "
                                f"PRED AP min/max: {pred_ap.min().item():.3e}/{pred_ap.max().item():.3e} | "
                                f"TARGET PA min/max: {target_pa.min().item():.3e}/{target_pa.max().item():.3e} | "
                                f"PRED PA min/max: {pred_pa.min().item():.3e}/{pred_pa.max().item():.3e}",
                                flush=True,
                            )
                    else:
                        loss_ap = torch.tensor(float("nan"), device=device)
                        loss_pa = torch.tensor(float("nan"), device=device)
                        loss_proj = torch.tensor(float("nan"), device=device)
    
                bg_depth_mass = torch.tensor(0.0, device=device)
                bg_depth_mass_w = torch.tensor(0.0, device=device)
                bg_depth_frac_t = torch.tensor(0.0, device=device)
                if (not skip_proj) and args.bg_depth_mass_weight > 0.0:
                    bg_mask = None
                    if target_ap is not None and target_pa is not None:
                        bg_mask = (target_ap < args.bg_depth_eps) & (target_pa < args.bg_depth_eps)
                    elif target_ap is not None:
                        bg_mask = target_ap < args.bg_depth_eps
                    elif target_pa is not None:
                        bg_mask = target_pa < args.bg_depth_eps
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
                loss_sparsity = torch.tensor(0.0, device=device)
                loss_act_tv = torch.tensor(0.0, device=device)
                act_norm_factor = 1.0
                need_act_supervision = act_vol is not None and (
                    args.act_loss_weight > 0.0 or args.act_tv_weight > 0.0
                )
                if need_act_supervision:
                    act_norm_factor, act_norm_global = compute_act_norm_factor(
                        act_vol, args.act_norm_source, args.act_norm_value, act_norm_global
                    )
                    act_norm_factor = float(act_norm_factor)

                act_norm_t = None
                loss_act_tv_updated = False
                if args.act_loss_weight > 0.0 and act_vol is not None:
                    act_radius_xyz = _resolve_act_radius_xyz_from_act(
                        generator, act_vol, batch.get("meta"), step=step
                    )
                    # Stichprobe aus act.npy und direkte Dichteabfrage im NeRF
                    coords, act_samples, pos_flags, act_indices = sample_act_points(
                        act_vol,
                        args.act_samples,
                        radius_xyz=act_radius_xyz,
                        pos_fraction=args.act_pos_fraction,
                        pos_threshold=args.act_pos_threshold,
                        return_indices=True,
                    )
                    pred_act_raw = None
                    pred_act = None
                    pred_act_log = None
                    pred_act, pred_act_raw = query_emission_at_points(
                        generator, z_latent, coords, return_raw=True
                    )
                    if pred_act_raw is not None:
                        act_norm_t = torch.tensor(float(act_norm_factor), device=device, dtype=pred_act_raw.dtype)
                    if debug_act_step and pred_act_raw is not None:
                        print(
                            f"[DEBUG][ACT][pred_act_raw] requires_grad={pred_act_raw.requires_grad}, is_leaf={pred_act_raw.is_leaf}, grad_fn={pred_act_raw.grad_fn is not None}", flush=True
                        )
                    if pred_act.numel() > 0:
                        pred_pos = pred_act / torch.clamp(act_norm_t, min=1e-8)

                        should_log_act_box = (step == 1) or args.debug_sanity_checks
                        if should_log_act_box:
                            res_val = getattr(args, "export_vol_res", 0)
                            try:
                                res_grid = max(1, int(res_val))
                            except Exception:
                                res_grid = 1
                            radius_xyz = act_radius_xyz
                            rx, ry, rz = radius_xyz
                            D, H, W = act_vol.shape[-3:]
                            coord_device = act_vol.device
                            x_coords_all = idx_to_coord(torch.arange(W, device=coord_device), W, rx)
                            y_coords_all = idx_to_coord(torch.arange(H, device=coord_device), H, ry)
                            z_coords_all = idx_to_coord(torch.arange(D, device=coord_device), D, rz)
                            x_min = float(x_coords_all.min().item())
                            x_max = float(x_coords_all.max().item())
                            y_min = float(y_coords_all.min().item())
                            y_max = float(y_coords_all.max().item())
                            z_min = float(z_coords_all.min().item())
                            z_max = float(z_coords_all.max().item())
                            print(
                                f"[ACT-CHECK][step {step}][box-range] "
                                f"x=[{x_min:.6f},{x_max:.6f}] "
                                f"y=[{y_min:.6f},{y_max:.6f}] "
                                f"z=[{z_min:.6f},{z_max:.6f}] "
                                f"box_radius_xyz=({rx:.3f},{ry:.3f},{rz:.3f}) "
                                f"act_shape=({D},{H},{W})",
                                flush=True,
                            )
                            pos_has_flags = bool(pos_flags.any().item())
                            if not pos_has_flags:
                                print(
                                    f"[ACT-CHECK][step {step}] pos_flags.any()=False -> skipping per-sample dump",
                                    flush=True,
                                )
                            else:
                                positive_idx = pos_flags.nonzero(as_tuple=False).reshape(-1)
                                debug_samples = positive_idx[:10]
                                print(
                                    f"[ACT-CHECK][step {step}] logging {len(debug_samples)} positive samples "
                                    f"from act_vol (D,H,W).",
                                    flush=True,
                                )
                                world_subset = coords[debug_samples]
                                for sample_rank, sample_idx in enumerate(debug_samples.tolist(), start=1):
                                    idx_triplet = tuple(int(v) for v in act_indices[sample_idx].tolist())
                                    world_coord = coords[sample_idx]
                                    iz, iy, ix = _world_coords_to_grid_indices(world_coord, res_grid, radius_xyz)
                                    act_gt_val = float(act_samples[sample_idx].clamp_min(0.0).item())
                                    act_pred_val = float(pred_act[sample_idx].item())
                                    print(
                                        f"[ACT-CHECK][step {step}] #{sample_rank} idx_act={idx_triplet} "
                                        f"world=({world_coord[0].item():.6f},{world_coord[1].item():.6f},{world_coord[2].item():.6f}) "
                                        f"gt={act_gt_val:.6e} pred={act_pred_val:.6e} grid=(iz={iz},iy={iy},ix={ix})",
                                        flush=True,
                                    )
                                    if len(ACT_DEBUG_WORLD_COORDS) < 2 and ACT_DEBUG_WORLD_STEP != step:
                                        ACT_DEBUG_WORLD_COORDS.append(
                                            (
                                                (
                                                    float(world_coord[0].item()),
                                                    float(world_coord[1].item()),
                                                    float(world_coord[2].item()),
                                                ),
                                                (iz, iy, ix),
                                            )
                                        )
                                        ACT_DEBUG_WORLD_STEP = step
                                        ACT_DEBUG_WORLD_REPORTED = False
                                if world_subset.numel() > 0:
                                    x_vals = world_subset[:, 0]
                                    y_vals = world_subset[:, 1]
                                    z_vals = world_subset[:, 2]
                                    n_logged = int(debug_samples.numel())
                                    print(
                                        f"[ACT-CHECK][step {step}][sample-range] "
                                        f"x=[{float(x_vals.min().item()):.6f},{float(x_vals.max().item()):.6f}] "
                                        f"y=[{float(y_vals.min().item()):.6f},{float(y_vals.max().item()):.6f}] "
                                        f"z=[{float(z_vals.min().item()):.6f},{float(z_vals.max().item()):.6f}] "
                                        f"n_logged={n_logged}",
                                        flush=True,
                                    )

                        # [DEBUG][ACT-BIAS] Analyse des 0.693-Bias (softplus(0))
                        if step == 1:
                            with torch.no_grad():
                                def _s(name, t):
                                    if t is None: return
                                    t_f = t.detach().float()
                                    print(f"[DEBUG][ACT-BIAS][step 1] {name}: "
                                          f"min={t_f.min():.3e} mean={t_f.mean():.3e} "
                                          f"max={t_f.max():.3e} dtype={t_f.dtype} device={t_f.device}",
                                          flush=True)

                                _s("pred_act_raw (net out)", pred_act_raw)
                                _s("pred_act (debiased)", pred_act)
                                _s("pred_pos (for loss, normalized)", pred_pos)
                                print(f"[DEBUG][ACT-BIAS][step 1] act_norm_factor={act_norm_factor:.6e}", flush=True)
                                act_pos_debug = act_samples.clamp_min(0.0) / torch.clamp(act_norm_t, min=1e-8)
                                _s("act_log (for loss)", torch.log1p(act_pos_debug))
                                _s("pred_act_log (for loss)", torch.log1p(pred_pos))
                                
                                # Gradient check hook
                                if pred_act_raw.requires_grad:
                                    def _hook(grad):
                                        g_norm = grad.norm().item()
                                        g_mean = grad.abs().mean().item()
                                        print(f"[DEBUG][ACT][GRAD] pred_act_raw grad: norm={g_norm:.3e} mean_abs={g_mean:.3e}", flush=True)
                                    pred_act_raw.register_hook(_hook)

                        # GT normalization
                        act_pos = act_samples.clamp_min(0.0) / torch.clamp(act_norm_t, min=1e-8)
                        pred_act_log = torch.log1p(pred_pos)
                        act_log = torch.log1p(act_pos)
                        weights_act = torch.where(
                            pos_flags,
                            torch.full_like(pred_act_log, args.act_pos_weight),
                            torch.ones_like(pred_act_log),
                        )
                        diff = F.smooth_l1_loss(pred_act_log, act_log, reduction="none")

                        # [DEBUG][ACT] Phase A Instrumentation
                        if (step == 1) or (step == max_steps):
                            with torch.no_grad():
                                assert weights_act.numel() == diff.numel(), f"weights_act.numel={weights_act.numel()} != diff.numel={diff.numel()}"
                                assert pos_flags.numel() == diff.numel(), f"pos_flags.numel={pos_flags.numel()} != diff.numel={diff.numel()}"

                                # flatten everything so masks/indexing are guaranteed correct
                                pred_f = pred_act_log.detach().float().reshape(-1)
                                gt_f   = act_log.detach().float().reshape(-1)
                                diff_f = diff.detach().float().reshape(-1)
                                w_f    = weights_act.detach().float().reshape(-1)
                                pos_f  = pos_flags.detach().reshape(-1).bool()

                                n_total = int(pos_f.numel())
                                n_pos = int(pos_f.sum().item())
                                n_neg = int((~pos_f).sum().item())

                                def _stats(name, t):
                                    t = t.float()
                                    p95 = torch.quantile(t, 0.95).item()
                                    nz = (t > 1e-6).float().mean().item()
                                    print(f"[DEBUG][ACT][step {step}] {name}: "
                                          f"min={t.min().item():.3e} mean={t.mean().item():.3e} "
                                          f"p95={p95:.3e} max={t.max().item():.3e} nz_frac={nz:.3f}",
                                          flush=True)

                                print(f"[DEBUG][ACT][step {step}] n_total={n_total} n_pos={n_pos} n_neg={n_neg} pos_frac={n_pos/max(n_total,1):.4f}", flush=True)

                                # show what the loss actually sees
                                _stats("GT_log1p", gt_f)
                                _stats("Pred_log1p", pred_f)
                                _stats("diff", diff_f)
                                _stats("weights", w_f)

                                # weighted diff per class
                                if n_pos > 0:
                                    loss_pos = (w_f[pos_f] * diff_f[pos_f]).mean().item()
                                    w_pos = w_f[pos_f].mean().item()
                                else:
                                    loss_pos, w_pos = 0.0, 0.0
                                if n_neg > 0:
                                    loss_neg = (w_f[~pos_f] * diff_f[~pos_f]).mean().item()
                                    w_neg = w_f[~pos_f].mean().item()
                                else:
                                    loss_neg, w_neg = 0.0, 0.0

                                print(f"[DEBUG][ACT][step {step}] loss_pos={loss_pos:.6f} loss_neg={loss_neg:.6f} (weighted)", flush=True)
                                print(f"[DEBUG][ACT][step {step}] mean_w_pos={w_pos:.6f} mean_w_neg={w_neg:.6f}", flush=True)
                                print(f"[DEBUG][ACT][step {step}] act_norm_source={args.act_norm_source} act_norm_factor={float(act_norm_factor):.6e}", flush=True)

                                # Optional: Linear space stats if available
                                try:
                                    _stats("GT_lin_norm", act_pos.detach().float().reshape(-1))
                                    _stats("Pred_lin_norm", pred_pos.detach().float().reshape(-1))
                                except NameError:
                                    pass

                        # Sparsity Loss (L1 auf normierter Aktivität)
                        if args.act_sparsity_weight > 0.0:
                            loss_sparsity = pred_pos.mean()
                        
                        # 3D TV Regularisierung (auf zufälligem Patch, da pred_pos unstrukturiert ist)
                        ACT_TV_EVERY = 10
                        DO_TV = (args.act_tv_weight > 0.0) and (step % ACT_TV_EVERY == 0)
                        loss_act_tv = torch.zeros((), device=device)

                        if args.act_tv_weight > 0.0:
                            if step == 1:
                                print(f"[DEBUG][ACT_TV] act_vol.shape={act_vol.shape}", flush=True)
                                print(f"[DEBUG][ACT_TV] ACT coords min/max: {coords.min(0).values.cpu().tolist()} {coords.max(0).values.cpu().tolist()}", flush=True)
                            
                            if DO_TV:
                                # Patch-Größe fix 16
                                S = 16
                                # act_vol shape handling: [D,H,W] or [1,D,H,W]
                                shape_src = act_vol.shape
                                D_vol, H_vol, W_vol = shape_src[-3:]
                                
                                # Zufälliger Startpunkt für Patch (Off-by-one safe range [0, Dim-S])
                                d0 = torch.randint(0, max(1, D_vol - S + 1), (1,), device=device).item()
                                h0 = torch.randint(0, max(1, H_vol - S + 1), (1,), device=device).item()
                                w0 = torch.randint(0, max(1, W_vol - S + 1), (1,), device=device).item()
                                
                                z_i = torch.arange(d0, min(d0 + S, D_vol), device=device)
                                y_i = torch.arange(h0, min(h0 + S, H_vol), device=device)
                                x_i = torch.arange(w0, min(w0 + S, W_vol), device=device)
                                
                                grid_z, grid_y, grid_x = torch.meshgrid(z_i, y_i, x_i, indexing='ij')
                                coords_tv = torch.stack([
                                    idx_to_coord(grid_x.flatten(), W_vol, radius),
                                    idx_to_coord(grid_y.flatten(), H_vol, radius),
                                    idx_to_coord(grid_z.flatten(), D_vol, radius)
                                ], dim=1)

                                if step == 1:
                                    print(f"[DEBUG][ACT_TV] TV  coords min/max: {coords_tv.min(0).values.cpu().tolist()} {coords_tv.max(0).values.cpu().tolist()}", flush=True)

                                _, pred_tv_raw = query_emission_at_points(generator, z_latent, coords_tv, return_raw=True)
                                
                                # Konsistente Aktivierung/Normierung wie im Supervised-Pfad
                                pred_tv_pos = activity_from_raw(pred_tv_raw, debias=True) / torch.clamp(
                                    torch.tensor(float(act_norm_factor), device=device, dtype=pred_tv_raw.dtype),
                                    min=1e-8,
                                )
                                pred_tv_vol = pred_tv_pos.view(len(z_i), len(y_i), len(x_i))
                                
                                dx = torch.abs(pred_tv_vol[:, :, 1:] - pred_tv_vol[:, :, :-1]).mean()
                                dy = torch.abs(pred_tv_vol[:, 1:, :] - pred_tv_vol[:, :-1, :]).mean()
                                dz = torch.abs(pred_tv_vol[1:, :, :] - pred_tv_vol[:-1, :, :]).mean()
                                loss_act_tv = dx + dy + dz
                                loss_act_tv_updated = True

                        with torch.no_grad():
                            pred_pos_mean = float(pred_pos.mean().item())
                            pred_nz_frac = float((pred_pos > 1e-6).float().mean().item())

                        loss_act = torch.mean(weights_act * diff)
                        act_loss_weighted = args.act_loss_weight * loss_act
                        loss = loss + act_loss_weighted
                        loss = loss + args.act_sparsity_weight * loss_sparsity
                        loss = loss + args.act_tv_weight * loss_act_tv
                        if loss_act_tv_updated:
                            last_act_tv_value = float(loss_act_tv.item())
                        global _PRED_ACT_STATS_LOGGED
                        if not _PRED_ACT_STATS_LOGGED:
                            _PRED_ACT_STATS_LOGGED = True
                            pred_stats_dbg = tensor_stats(pred_act)
                            eps = 1e-6
                            positive_frac = float((pred_act > eps).float().mean().item())
                            pred_sum = float(pred_act.sum().item())
                            print(
                                f"[sanity][pred_act] stats(min/mean/p95/max)={fmt_stats(pred_stats_dbg)} "
                                f"| sum={pred_sum:.3e} | frac>{eps:.1e}={positive_frac:.3f} | sparsity<eps={(1.0-positive_frac):.3f}",
                                flush=True,
                            )
                        if debug_act_step:
                            act_vol_stats = tensor_stats(act_vol)
                            pos_vol_frac = float((act_vol > 1e-8).float().mean().item()) if act_vol is not None else float("nan")
                            act_stats_pre = tensor_stats(act_samples)
                            act_stats_norm = tensor_stats(act_pos)
                            pred_stats = tensor_stats(pred_act)
                            pred_raw_stats = tensor_stats(pred_act_raw)
                            print(
                                f"[DEBUG][ACT][loss] loss_act={loss_act.item():.6f} (req_grad={loss_act.requires_grad}), weight={args.act_loss_weight}, "
                                f"total_contribution={act_loss_weighted.item():.6f}, total_loss_before={loss.item()-act_loss_weighted.item():.6f}",
                                flush=True
                            )
                            zero_frac = float((act_samples == 0).float().mean().item())
                            tiny_frac = float((act_samples < 1e-6).float().mean().item())
                            pos_frac = float(pos_flags.float().mean().item()) if pos_flags.numel() > 0 else float("nan")
                            print(
                                f"[DEBUG][ACT][step {step}] act_vol={fmt_stats(act_vol_stats)} | pos_vol_frac={pos_vol_frac:.3f} "
                                f"| act_gt={fmt_stats(act_stats_pre)} "
                                f"| act_gt_norm={fmt_stats(act_stats_norm)} "
                                f"| zero_frac={zero_frac:.3f} | lt1e-6_frac={tiny_frac:.3f} | pos_frac={pos_frac:.3f}",
                                flush=True,
                            )
                            print(
                                f"[DEBUG][ACT][step {step}] pred_raw={fmt_stats(pred_raw_stats)} "
                                f"| pred_act={fmt_stats(pred_stats)} "
                                f"| act_norm_source={args.act_norm_source} act_norm_factor={act_norm_factor:.3e} "
                                f"act_norm_global={'set' if act_norm_global is not None else 'none'}",
                                flush=True,
                            )
                            print(
                                f"[DEBUG][ACT][step {step}] requires_grad: pred_act={pred_act_raw.requires_grad} "
                                f"z_latent={z_latent.requires_grad} act_samples={act_samples.requires_grad}",
                                flush=True,
                            )
    
                loss_gain = torch.tensor(0.0, device=device)
                gain_prior = None
                gain_for_reg = gain_val_used if gain_val_used is not None else gain_val
                if hybrid_enabled and gain_for_reg is not None and args.gain_reg_weight > 0.0:
                    gain_mean = float(gain_for_reg.detach().mean().item())
                    if args.gain_prior_mode == "fixed":
                        gain_prior = float(args.gain_prior_value)
                    else:
                        # EMA over first N steps
                        if gain_prior_ema is None:
                            gain_prior_ema = gain_mean
                        else:
                            gain_prior_ema = gain_prior_decay * gain_prior_ema + (1.0 - gain_prior_decay) * gain_mean
                        if step <= gain_prior_steps:
                            gain_prior = gain_prior_ema
                        else:
                            if gain_prior_final is None:
                                gain_prior_final = gain_prior_ema
                            gain_prior = gain_prior_final
                    if gain_prior is not None and gain_prior > 0:
                        log_gain = torch.log(gain_for_reg.clamp_min(1e-12))
                        log_prior = math.log(max(gain_prior, 1e-12))
                        loss_gain = ((log_gain - log_prior) ** 2).mean()
                        loss_gain = loss_gain * float(args.gain_reg_scale)
                        loss = loss + float(args.gain_reg_weight) * loss_gain
    
                loss_ct = torch.tensor(0.0, device=device)
                ct_pairs_valid = False
                if args.ct_loss_weight > 0.0 and ct_vol is not None:
                    radius = generator.radius
                    if isinstance(radius, tuple):
                        radius = radius[1]
                    ct_pairs = sample_ct_pairs(ct_vol, args.ct_samples, args.ct_threshold, radius=radius)
                    if ct_pairs is not None:
                        ct_pairs_valid = True
                        coords1, coords2, weights = ct_pairs
                        pred1 = query_emission_at_points(generator, z_latent, coords1)
                        pred2 = query_emission_at_points(generator, z_latent, coords2)
                        # Loss zwingt Emission auf flachen CT-Strecken zur Konstanz
                        loss_ct = torch.mean(torch.abs(pred1 - pred2) * weights)
                        loss = loss + args.ct_loss_weight * loss_ct
    
                tv_base_loss = torch.tensor(0.0, device=device)
                loss_tv = torch.tensor(0.0, device=device)
                loss_ray_tv = torch.tensor(0.0, device=device)
                loss_ray_tv_w = torch.tensor(0.0, device=device)
                loss_lambda_ray_tv = torch.tensor(0.0, device=device)
                loss_lambda_ray_tv_w = torch.tensor(0.0, device=device)
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

                if proj_loss_active and proj_loss_type == "poisson" and lambda_ray_tv_weight != 0.0:
                    lambda_ray_tv_terms = []
                    lambda_ray_tv_norm_used = False
                    lambda_ray_tv_norm_missing = False
                    lambda_ray_tv_nsamples = None
                    for extras in (extras_ap, extras_pa):
                        if not isinstance(extras, dict):
                            continue
                        lambda_ray = extras.get("lambda_ray")
                        if lambda_ray is None:
                            raw_out = extras.get("raw")
                            if raw_out is None:
                                continue
                            if raw_out.dim() >= 3:
                                raw_lambda = raw_out[..., 0]
                            elif raw_out.dim() == 2:
                                raw_lambda = raw_out
                            else:
                                continue
                            if extras.get("lambda_ray_pre_act") is None:
                                extras["lambda_ray_pre_act"] = raw_lambda
                            if debug_sanity_checks:
                                raw_stats = tensor_stats(raw_lambda)
                                raw_min = raw_stats["min"] if raw_stats is not None else float("nan")
                                raw_mean = raw_stats["mean"] if raw_stats is not None else float("nan")
                                raw_max = raw_stats["max"] if raw_stats is not None else float("nan")
                                print(
                                    f"[sanity][step {step}] lambda_ray fallback=raw shape={tuple(raw_out.shape)} "
                                    f"| raw(min/mean/max)={raw_min:.3e}/{raw_mean:.3e}/{raw_max:.3e}",
                                    flush=True,
                                )
                            lambda_ray = F.softplus(raw_lambda)
                            extras["lambda_ray"] = lambda_ray
                        if lambda_ray.shape[-1] < 2:
                            continue
                        tv_raw = torch.mean(torch.abs(lambda_ray[..., 1:] - lambda_ray[..., :-1]))
                        dz_mean = None
                        dists = extras.get("dists")
                        if torch.is_tensor(dists) and dists.shape == lambda_ray.shape:
                            dz_mean = torch.mean(dists)
                        else:
                            z_vals = extras.get("z_vals")
                            if torch.is_tensor(z_vals):
                                if z_vals.dim() == 1 and z_vals.shape[0] == lambda_ray.shape[-1]:
                                    dz = z_vals[1:] - z_vals[:-1]
                                    dz_mean = torch.mean(torch.abs(dz))
                                elif z_vals.shape[-1] == lambda_ray.shape[-1]:
                                    dz = z_vals[..., 1:] - z_vals[..., :-1]
                                    dz_mean = torch.mean(torch.abs(dz))
                        if dz_mean is not None and torch.isfinite(dz_mean) and float(dz_mean.item()) > 0.0:
                            tv_val = tv_raw / dz_mean
                            lambda_ray_tv_norm_used = True
                        else:
                            tv_val = tv_raw
                            lambda_ray_tv_norm_missing = True
                            lambda_ray_tv_nsamples = int(lambda_ray.shape[-1])
                        lambda_ray_tv_terms.append(tv_val)
                    if lambda_ray_tv_terms:
                        loss_lambda_ray_tv = torch.stack(lambda_ray_tv_terms).mean()
                        loss_lambda_ray_tv_w = loss_lambda_ray_tv * float(lambda_ray_tv_weight)
                        loss = loss + loss_lambda_ray_tv_w
                    if debug_sanity_checks and (step == 1 or (step % debug_sanity_every) == 0):
                        print(
                            f"[sanity][step {step}] lambda_ray_tv={loss_lambda_ray_tv.item():.6f} "
                            f"| lambda_ray_tv_weight={float(lambda_ray_tv_weight):.3e}",
                            flush=True,
                        )
                        if lambda_ray_tv_norm_missing and not lambda_ray_tv_norm_used:
                            print(
                                f"[sanity][step {step}] lambda_ray_tv unnormalized (no dists/z_vals) "
                                f"-> depends on N_samples={lambda_ray_tv_nsamples}",
                                flush=True,
                            )
    
                if depth_checks_active and (step % depth_sanity_every == 0 or step == 1):
                    # Single-Phantom ist inhaltlich stabil, wenn:
                    # - Projektionen plateauieren,
                    # - Depth-Regularizer nicht trivial sind,
                    # - lambda-Std entlang Rays stabil > 1e-4,
                    # - Gain im physikalischen Bereich bleibt.
                    proj_loss_active_depth = (not skip_proj) and (proj_loss_type == "poisson") and (
                        not hybrid_enabled or (proj_weight_used > 0.0 and not proj_warmup_active)
                    )
                    atten_flag = bool(generator.render_kwargs_train.get("use_attenuation", False))
                    atten_active = atten_flag and (ct_context is not None)
                    if proj_loss_active_depth and not atten_active:
                        reason = "use_attenuation=False" if not atten_flag else "ct_context=None"
                        print(
                            f"[WARN][depth] Attenuation inaktiv bei aktivem Projection-Loss -> Depth unterbestimmt. "
                            f"Ursache: {reason}.",
                            flush=True,
                        )
                    if proj_loss_active_depth and atten_active:
                        mu_terms = []
                        atten_terms = []
                        atten_default = globals().get("ATTEN_SCALE_DEFAULT", 1.0)
                        atten_scale = float(generator.render_kwargs_train.get("atten_scale", atten_default))
                        for extras in (extras_ap, extras_pa):
                            if not isinstance(extras, dict):
                                continue
                            mu_out = extras.get("mu")
                            if mu_out is not None and mu_out.dim() > 0 and mu_out.shape[-1] == 1:
                                mu_out = mu_out.squeeze(-1)
                            dists = extras.get("dists")
                            if mu_out is not None:
                                mu_terms.append(mu_out.reshape(-1))
                            if mu_out is not None and dists is not None and mu_out.shape == dists.shape:
                                mu_clamped = torch.clamp(mu_out, min=0.0)
                                mu_dists = mu_clamped * dists
                                attenuation = torch.cumsum(mu_dists, dim=-1) * atten_scale
                                attenuation = F.pad(attenuation[..., :-1], (1, 0), mode="constant", value=0.0)
                                attenuation = torch.clamp(attenuation, min=0.0, max=60.0)
                                atten_terms.append(attenuation.reshape(-1))
                        if not mu_terms:
                            print(
                                "[WARN][depth] Attenuation aktiv, aber mu fehlt in Extras (retraw/ct_context prüfen).",
                                flush=True,
                            )
                        else:
                            mu_all = torch.cat(mu_terms)
                            mu_mean = float(mu_all.mean().item())
                            if mu_mean <= 0.0:
                                print(
                                    "[WARN][depth] mu_mean <= 0 -> Attenuation hat praktisch keinen Einfluss.",
                                    flush=True,
                                )
                        if atten_terms:
                            atten_all = torch.cat(atten_terms)
                            atten_mean = float(atten_all.mean().item())
                            if atten_mean < 1e-3:
                                print(
                                    f"[WARN][depth] Attenuation-Mittelwert sehr klein ({atten_mean:.3e}) -> geringer Einfluss.",
                                    flush=True,
                                )
    
                    depth_zero = []
                    if ray_tv_weight != 0.0 and float(loss_ray_tv.item()) <= 0.0:
                        depth_zero.append("ray_tv")
                    if args.ct_loss_weight > 0.0:
                        if ct_vol is None:
                            depth_zero.append("ct_missing")
                        elif not ct_pairs_valid:
                            depth_zero.append("ct_pairs")
                    if depth_zero:
                        print(
                            f"[WARN][depth] Depth-Terms ohne Signal: {', '.join(depth_zero)}. "
                            "Bitte CT/Ray-TV/BG-Depth pruefen.",
                            flush=True,
                        )
    
                    lam_std_terms = []
                    for extras in (extras_ap, extras_pa):
                        if not isinstance(extras, dict):
                            continue
                        raw_out = extras.get("raw")
                        if raw_out is None:
                            continue
                        lam = F.softplus(raw_out[..., 0]) - math.log(2.0)
                        lam = lam.clamp_min(1e-6)
                        lam_std_terms.append(lam.std(dim=-1).mean())
                    if lam_std_terms:
                        lam_std_mean = float(torch.stack(lam_std_terms).mean().item())
                        if lam_std_mean < 1e-6:
                            print(
                                f"[WARN][depth] Depth-Profil kollabiert (lambda std entlang Ray ~ {lam_std_mean:.3e}).",
                                flush=True,
                            )
                        if lam_std_mean < 1e-4:
                            print(
                                f"[WARN][depth] lambda std entlang Ray sehr klein: {lam_std_mean:.3e}",
                                flush=True,
                            )
    
                    depth_reg_loss = loss_ray_tv_w + (args.ct_loss_weight * loss_ct) + bg_depth_mass_w
                    depth_reg_total = float(depth_reg_loss.detach().item())
                    net_module = generator.render_kwargs_train.get("network_fn")
                    depth_grad_net = _depth_grad_norm(depth_reg_loss, net_module)
                    print(
                        f"[depth][grad] ||g_depth||_net={depth_grad_net:.3e} | depth_reg_total={depth_reg_total:.3e}",
                        flush=True,
                    )
                    if depth_grad_net < 1e-8:
                        depth_grad_zero_streak += 1
                    else:
                        depth_grad_zero_streak = 0
                    if depth_grad_zero_streak >= 3:
                        print(
                            "[WARN][depth] Depth-Gradient ~0 ueber mehrere Checks -> Depth lernt nicht.",
                            flush=True,
                        )
    
                    if gain_val is not None:
                        gain_mean = float(gain_val.detach().mean().item())
                        gain_std = float(gain_val.detach().std().item())
                        print(
                            f"[gain][depth] gain_mean={gain_mean:.3f} gain_std={gain_std:.3f} "
                            f"| depth_reg_total={depth_reg_total:.3e}",
                            flush=True,
                        )
                        if depth_reg_total < 1e-6 and abs(gain_mean - 1.0) > 0.1:
                            print(
                                f"[WARN][gain] gain_mean={gain_mean:.3f} bei depth_reg_total={depth_reg_total:.3e} "
                                "-> Gain kann Strukturfreiheit kompensieren.",
                                flush=True,
                            )
            proj_loss_for_grad = (
                proj_weight_used * 0.5 * (loss_ap + loss_pa)
                if (not skip_proj and proj_loss_active)
                else torch.tensor(0.0, device=device)
            )
    
            if debug_act_step:
                net_module = generator.render_kwargs_train.get("network_fn")
                grad_act_net = grad_norm_of_module(args.act_loss_weight * loss_act, net_module)
                grad_proj_net = grad_norm_of_module(proj_loss_for_grad, net_module)
                grad_act_enc = grad_norm_of_module(args.act_loss_weight * loss_act, encoder)
                grad_proj_enc = grad_norm_of_module(proj_loss_for_grad, encoder)
                print(
                    f"[DEBUG][ACT][gradcomp][step {step}] ||g_act||_net={grad_act_net:.3e} "
                    f"||g_proj||_net={grad_proj_net:.3e} ||g_act||_enc={grad_act_enc:.3e} "
                    f"||g_proj||_enc={grad_proj_enc:.3e}",
                    flush=True,
                )
    
            if args.grad_stats_every > 0 and (step % args.grad_stats_every) == 0:
                grad_stats = {
                    "proj": grad_norm_of(proj_loss_for_grad, [z_latent]),
                    "act": grad_norm_of(args.act_loss_weight * loss_act, [z_latent]) if args.act_loss_weight > 0 else 0.0,
                    "ct": grad_norm_of(args.ct_loss_weight * loss_ct, [z_latent]) if args.ct_loss_weight > 0 else 0.0,
                }
                print(
                    f"[grad][step {step:05d}] ||g_proj||={grad_stats['proj']:.3e} "
                    f"| ||g_act||={grad_stats['act']:.3e} | ||g_ct||={grad_stats['ct']:.3e}",
                    flush=True,
                )

            if args.debug_grad_terms_every > 0 and (step % int(args.debug_grad_terms_every)) == 0:
                act_term = (args.act_loss_weight * loss_act) if args.act_loss_weight > 0.0 else None
                proj_term = (proj_weight_used * loss_proj) if proj_loss_active else None

                def _term_grad_norms(loss_term: Optional[torch.Tensor]) -> Optional[tuple[float, float, float]]:
                    if loss_term is None or (not torch.is_tensor(loss_term)) or (not loss_term.requires_grad):
                        return None
                    optimizer.zero_grad(set_to_none=True)
                    loss_term.backward(retain_graph=True)
                    return (
                        params_grad_norm(decoder_params),
                        params_grad_norm(encoder_params),
                        params_grad_norm(z_fuser_params),
                    )

                act_grad_norms = _term_grad_norms(act_term)
                proj_grad_norms = _term_grad_norms(proj_term) if proj_loss_active else None
                optimizer.zero_grad(set_to_none=True)

                def _fmt_triplet(values: Optional[tuple[float, float, float]]) -> str:
                    if values is None:
                        return "na/na/na"
                    dec_v, enc_v, fus_v = values
                    enc_s = f"{enc_v:.3e}" if encoder_params else "na"
                    fus_s = f"{fus_v:.3e}" if z_fuser_params else "na"
                    return f"{dec_v:.3e}/{enc_s}/{fus_s}"

                print(
                    f"[grad-terms][step {step:05d}] "
                    f"act(dec/enc/fus)={_fmt_triplet(act_grad_norms)} "
                    f"proj(dec/enc/fus)={_fmt_triplet(proj_grad_norms)}",
                    flush=True,
                )

            if debug_act_step and pred_act_raw is not None:
                # Hook für den Gradienten des rohen Netz-Outputs
                pred_act_raw.retain_grad()

            scaler.scale(loss).backward()
            if should_log_latent_stats:
                dec_module = generator.render_kwargs_train.get("network_fn")
                grad_dec = module_grad_norm(dec_module if isinstance(dec_module, nn.Module) else generator)
                grad_enc = module_grad_norm(encoder)
                grad_fuser = module_grad_norm(z_fuser)
                grad_gain_head = module_grad_norm(gain_head)
                grad_gain_param = (
                    float(gain_param.grad.detach().norm().item())
                    if (gain_param is not None and gain_param.grad is not None)
                    else 0.0
                )
                print(
                    f"[latent-grad][step {step:05d}] "
                    f"decoder={grad_dec:.3e} encoder={grad_enc:.3e} "
                    f"z_fuser={grad_fuser:.3e} gain_head={grad_gain_head:.3e} "
                    f"gain_param={grad_gain_param:.3e}",
                    flush=True,
                )

            # (3) Instrumentierung: Gradienten-Check (step 1)
            if args.debug_enc and step == 1:
                if hybrid_enabled and encoder is not None:
                    found_enc_grad = False
                    for p in encoder.parameters():
                        if p.grad is not None:
                            print(f"[DEBUG][ENC_BWD] Encoder grad norm: {p.grad.norm().item():.6e}, absmax: {p.grad.abs().max().item():.6e}", flush=True)
                            found_enc_grad = True
                            break
                    if not found_enc_grad:
                        print("[DEBUG][ENC_BWD] WARN: No gradient found for any encoder parameter.", flush=True)
                
                found_gen_grad = False
                for p in generator.parameters():
                    if p.grad is not None:
                        print(f"[DEBUG][ENC_BWD] Generator grad norm: {p.grad.norm().item():.6e}, absmax: {p.grad.abs().max().item():.6e}", flush=True)
                        found_gen_grad = True
                        break
                if not found_gen_grad:
                    print("[DEBUG][ENC_BWD] WARN: No gradient found for any generator parameter.", flush=True)

            if debug_act_step:
                net_module = generator.render_kwargs_train.get("network_fn")
                if pred_act_raw is not None and pred_act_raw.grad is not None:
                    grad_stats = tensor_stats(pred_act_raw.grad)
                    print(f"[DEBUG][ACT][grad][pred_act_raw] {fmt_stats(grad_stats)}", flush=True)
                else:
                    print(f"[DEBUG][ACT][grad][pred_act_raw] grad is None", flush=True)

                # Gradienten-Normen von ausgewählten Layern
                first_layer_weight = net_module.pts_linears[0].weight
                last_layer_weight = net_module.output_linear.weight
                if first_layer_weight.grad is not None:
                    print(f"[DEBUG][ACT][grad][net_early] {fmt_stats(tensor_stats(first_layer_weight.grad))}", flush=True)
                else:
                    print(f"[DEBUG][ACT][grad][net_early] grad is None", flush=True)
                if last_layer_weight.grad is not None:
                    print(f"[DEBUG][ACT][grad][net_final] {fmt_stats(tensor_stats(last_layer_weight.grad))}", flush=True)
                else:
                    print(f"[DEBUG][ACT][grad][net_final] grad is None", flush=True)

            # --- Instrumentierung: Parameter Update Check (Pre-Step) ---
            do_param_debug = args.debug_enc and step in (1, 2)
            enc_p0, enc_p0_before, enc_p0_norm = None, None, 0.0
            gen_p0, gen_p0_before, gen_p0_norm = None, None, 0.0

            if do_param_debug:
                if hybrid_enabled and encoder is not None:
                    for p in encoder.parameters():
                        if p.requires_grad:
                            enc_p0 = p
                            enc_p0_before = p.detach().clone()
                            enc_p0_norm = p.detach().norm().item()
                            break
                for p in generator.parameters():
                    if p.requires_grad:
                        gen_p0 = p
                        gen_p0_before = p.detach().clone()
                        gen_p0_norm = p.detach().norm().item()
                        break

            grad_norm_global = global_grad_norm(opt_params) if hybrid_enabled else 0.0
            grad_norm_gen_pre = module_grad_norm(generator)
            grad_norm_gen_post = grad_norm_gen_pre
            clip_event = 0
            decoder_clip_before = None
            decoder_clip_after = None
            decoder_clip_applied = False
            need_any_unscale = bool(args.grad_clip_enabled) or (float(args.clip_grad_decoder) > 0.0)
            if need_any_unscale and amp_enabled:
                scaler.unscale_(optimizer)
            if args.grad_clip_enabled:
                grad_norm_gen_post = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=args.grad_clip_max_norm)
                clip_event = float(grad_norm_gen_post) > args.grad_clip_max_norm
            if float(args.clip_grad_decoder) > 0.0 and decoder_params:
                decoder_clip_before = params_grad_norm(decoder_params)
                torch.nn.utils.clip_grad_norm_(decoder_params, max_norm=float(args.clip_grad_decoder))
                decoder_clip_after = params_grad_norm(decoder_params)
                decoder_clip_applied = bool(decoder_clip_before > float(args.clip_grad_decoder))
            scaler.step(optimizer)
            scaler.update()

            if float(args.clip_grad_decoder) > 0.0:
                clip_log_interval = val_interval if val_interval > 0 else 200
                if step == 1 or (step % clip_log_interval) == 0:
                    before_s = f"{decoder_clip_before:.3e}" if decoder_clip_before is not None else "nan"
                    after_s = f"{decoder_clip_after:.3e}" if decoder_clip_after is not None else "nan"
                    print(
                        f"[clip][step {step:05d}] decoder_clip={float(args.clip_grad_decoder):.3e} "
                        f"applied={decoder_clip_applied} measured_norm_before={before_s} measured_norm_after={after_s}",
                        flush=True,
                    )

            if debug_act_step:
                print(
                    f"[DEBUG][ACT][grad] grad_norm_global={grad_norm_global:.6f} "
                    f"gen_pre={grad_norm_gen_pre:.6f} gen_post={grad_norm_gen_post:.6f} clip_max={args.grad_clip_max_norm} clip_event={clip_event}",
                    flush=True,
                )

            # --- Instrumentierung: Parameter Update Check (Post-Step) ---
            if do_param_debug:
                if enc_p0 is not None:
                    dp = (enc_p0.detach() - enc_p0_before)
                    dp_norm = dp.norm().item()
                    dp_ratio = dp_norm / (enc_p0_norm + 1e-12)
                    print(f"[DEBUG][ENC_UPD] step={step} dp_norm={dp_norm:.3e} dp_ratio={dp_ratio:.3e} p_norm={enc_p0_norm:.3e}", flush=True)
                
                if gen_p0 is not None:
                    dq = (gen_p0.detach() - gen_p0_before)
                    dq_norm = dq.norm().item()
                    dq_ratio = dq_norm / (gen_p0_norm + 1e-12)
                    print(f"[DEBUG][GEN_UPD] step={step} dq_norm={dq_norm:.3e} dq_ratio={dq_ratio:.3e} q_norm={gen_p0_norm:.3e}", flush=True)
                
                if step == 1:
                     print("[DEBUG][OPTIM] Param groups LRs:", flush=True)
                     for i, pg in enumerate(optimizer.param_groups):
                         name = pg.get('name', f'group_{i}')
                         lr_val = pg.get('lr', -1.0)
                         print(f"  - {name}: lr={lr_val:.3e}", flush=True)
    
            torch.cuda.synchronize()
            iter_ms = (time.perf_counter() - t0) * 1000.0
    
            with torch.no_grad():
                pred_mean_raw = (float("nan"), float("nan"))
                pred_std_raw = (float("nan"), float("nan"))
                if pred_ap_raw is not None and pred_pa_raw is not None:
                    pred_mean_raw = (pred_ap_raw.mean().item(), pred_pa_raw.mean().item())
                    pred_std_raw = (pred_ap_raw.std().item(), pred_pa_raw.std().item())
                metrics_ap = None
                metrics_pa = None
                metrics_phys_ap = None
                metrics_phys_pa = None
                pred_mean = pred_mean_raw
                pred_std = pred_std_raw
                if proj_metrics_enabled:
                    metrics_ap = compute_projection_metrics(pred_ap, target_ap)
                    metrics_pa = compute_projection_metrics(pred_pa, target_pa)
                    pred_mean = (pred_ap.mean().item(), pred_pa.mean().item())
                    pred_std = (pred_ap.std().item(), pred_pa.std().item())
                    if log_proj_metrics_physical:
                        pred_ap_phys = pred_ap * float(scale_ap_used)
                        pred_pa_phys = pred_pa * float(scale_pa_used)
                        target_ap_phys = target_ap * float(scale_ap_used)
                        target_pa_phys = target_pa * float(scale_pa_used)
                        metrics_phys_ap = compute_projection_metrics(pred_ap_phys, target_ap_phys)
                        metrics_phys_pa = compute_projection_metrics(pred_pa_phys, target_pa_phys)
                mae_ap = metrics_ap["mae"] if metrics_ap else float("nan")
                mae_pa = metrics_pa["mae"] if metrics_pa else float("nan")
                rmse_ap = metrics_ap["rmse"] if metrics_ap else float("nan")
                rmse_pa = metrics_pa["rmse"] if metrics_pa else float("nan")
                dev_ap = metrics_ap["dev"] if metrics_ap else float("nan")
                dev_pa = metrics_pa["dev"] if metrics_pa else float("nan")
                nll_ap = metrics_ap["nll"] if metrics_ap else float("nan")
                nll_pa = metrics_pa["nll"] if metrics_pa else float("nan")
                phys_mae_ap = metrics_phys_ap["mae"] if metrics_phys_ap else None
                phys_mae_pa = metrics_phys_pa["mae"] if metrics_phys_pa else None
                phys_rmse_ap = metrics_phys_ap["rmse"] if metrics_phys_ap else None
                phys_rmse_pa = metrics_phys_pa["rmse"] if metrics_phys_pa else None
                phys_nll_ap = metrics_phys_ap["nll"] if metrics_phys_ap else None
                phys_nll_pa = metrics_phys_pa["nll"] if metrics_phys_pa else None
                phys_dev_ap = metrics_phys_ap["dev"] if metrics_phys_ap else None
                phys_dev_pa = metrics_phys_pa["dev"] if metrics_phys_pa else None
            if debug_sanity_checks and (step == 1 or (step % debug_sanity_every) == 0):
                    if proj_warmup_active and proj_loss_active:
                        print(
                            f"[sanity][step {step}] WARN: proj_warmup_active=True but proj_loss_active=True",
                            flush=True,
                        )
                    if proj_warmup_active:
                        if all(math.isfinite(x) for x in pred_mean) and all(math.isfinite(x) for x in pred_mean_raw):
                            diff_ap = abs(pred_mean[0] - pred_mean_raw[0])
                            diff_pa = abs(pred_mean[1] - pred_mean_raw[1])
                            if max(diff_ap, diff_pa) > 1e-6:
                                print(
                                    f"[sanity][step {step}] WARN: warmup pred_mean != pred_mean_raw "
                                    f"(diff_ap={diff_ap:.3e}, diff_pa={diff_pa:.3e})",
                                    flush=True,
                                )
            bg_depth_frac = float(bg_depth_frac_t.detach().cpu().item())
            if hybrid_enabled and hybrid_log_path is not None and need_raw_stats:
                target_ap_stats = tensor_stats(target_ap)
                target_pa_stats = tensor_stats(target_pa)
                pred_ap_stats = tensor_stats(pred_ap)
                pred_pa_stats = tensor_stats(pred_pa)
                atten_scale = float(generator.render_kwargs_train.get("atten_scale", ATTEN_SCALE_DEFAULT))
                (
                    lambda_stats,
                    mu_stats,
                    atten_stats,
                    atten_frac_gt20,
                    atten_frac_clamp,
                    nonfinite_lambda,
                    nonfinite_atten,
                ) = compute_lambda_and_attenuation_stats([extras_ap, extras_pa], atten_scale=atten_scale)
                nonfinite_pred = nonfinite_fraction(torch.cat([pred_ap, pred_pa], dim=1))
                proj_scale_enc_val = (
                    float(proj_scale_enc.mean().item()) if torch.is_tensor(proj_scale_enc) else float("nan")
                )
                gain_log_src = gain_val_used if gain_val_used is not None else gain_val
                gain_log = float(gain_log_src.mean().item()) if gain_log_src is not None else float("nan")
                z_enc_l2 = (
                    float(z_enc_proj.detach().norm(dim=1).mean().item()) if z_enc_proj is not None else float("nan")
                )
                z_latent_l2 = float(z_latent.detach().norm(dim=1).mean().item())

                def _stat(stats, key):
                    return float(stats[key]) if stats is not None and key in stats else float("nan")

                print(
                    f"[hybrid][step {step:05d}] "
                    f"t_ap(min/mean/p95/max)=({_stat(target_ap_stats,'min'):.3e},"
                    f"{_stat(target_ap_stats,'mean'):.3e},{_stat(target_ap_stats,'p95'):.3e},"
                    f"{_stat(target_ap_stats,'max'):.3e}) "
                    f"t_pa(min/mean/p95/max)=({_stat(target_pa_stats,'min'):.3e},"
                    f"{_stat(target_pa_stats,'mean'):.3e},{_stat(target_pa_stats,'p95'):.3e},"
                    f"{_stat(target_pa_stats,'max'):.3e}) "
                    f"p_ap(min/mean/p95/max)=({_stat(pred_ap_stats,'min'):.3e},"
                    f"{_stat(pred_ap_stats,'mean'):.3e},{_stat(pred_ap_stats,'p95'):.3e},"
                    f"{_stat(pred_ap_stats,'max'):.3e}) "
                    f"p_pa(min/mean/p95/max)=({_stat(pred_pa_stats,'min'):.3e},"
                    f"{_stat(pred_pa_stats,'mean'):.3e},{_stat(pred_pa_stats,'p95'):.3e},"
                    f"{_stat(pred_pa_stats,'max'):.3e}) "
                    f"lambda_ray(min/mean/p95/max)=({_stat(lambda_stats,'min'):.3e},"
                    f"{_stat(lambda_stats,'mean'):.3e},{_stat(lambda_stats,'p95'):.3e},"
                    f"{_stat(lambda_stats,'max'):.3e}) "
                    f"mu(min/mean/p95/max)=({_stat(mu_stats,'min'):.3e},"
                    f"{_stat(mu_stats,'mean'):.3e},{_stat(mu_stats,'p95'):.3e},"
                    f"{_stat(mu_stats,'max'):.3e}) "
                    f"atten(min/mean/p95/max)=({_stat(atten_stats,'min'):.3e},"
                    f"{_stat(atten_stats,'mean'):.3e},{_stat(atten_stats,'p95'):.3e},"
                    f"{_stat(atten_stats,'max'):.3e}) "
                    f"gain={gain_log:.3e} "
                    f"atten>20={atten_frac_gt20 if atten_frac_gt20 is not None else float('nan'):.3f} "
                    f"atten=60={atten_frac_clamp if atten_frac_clamp is not None else float('nan'):.3f} "
                    f"nonfinite(pred/lambda/atten)=({nonfinite_pred:.3e},{nonfinite_lambda:.3e},{nonfinite_atten:.3e}) "
                    f"grad_norm={grad_norm_global:.3e} clip={int(clip_event)}",
                    flush=True,
                )
    
                append_hybrid_log(
                    hybrid_log_path,
                    [
                        step,
                        proj_weight_used,
                        float(loss_proj.item()),
                        float(loss_ap.item()),
                        float(loss_pa.item()),
                        float(loss_act.item()),
                        float(loss_gain.item()),
                        float(gain_prior) if gain_prior is not None else float("nan"),
                        float(act_norm_factor),
                        float(loss.item()),
                        proj_scale_enc_val,
                        _stat(target_ap_stats, "min"),
                        _stat(target_ap_stats, "mean"),
                        _stat(target_ap_stats, "p95"),
                        _stat(target_ap_stats, "max"),
                        _stat(target_pa_stats, "min"),
                        _stat(target_pa_stats, "mean"),
                        _stat(target_pa_stats, "p95"),
                        _stat(target_pa_stats, "max"),
                        _stat(pred_ap_stats, "min"),
                        _stat(pred_ap_stats, "mean"),
                        _stat(pred_ap_stats, "p95"),
                        _stat(pred_ap_stats, "max"),
                        _stat(pred_pa_stats, "min"),
                        _stat(pred_pa_stats, "mean"),
                        _stat(pred_pa_stats, "p95"),
                        _stat(pred_pa_stats, "max"),
                        _stat(lambda_stats, "min"),
                        _stat(lambda_stats, "mean"),
                        _stat(lambda_stats, "p95"),
                        _stat(lambda_stats, "max"),
                        _stat(mu_stats, "min"),
                        _stat(mu_stats, "mean"),
                        _stat(mu_stats, "p95"),
                        _stat(mu_stats, "max"),
                        _stat(atten_stats, "min"),
                        _stat(atten_stats, "mean"),
                        _stat(atten_stats, "p95"),
                        _stat(atten_stats, "max"),
                        float(atten_frac_gt20) if atten_frac_gt20 is not None else float("nan"),
                        float(atten_frac_clamp) if atten_frac_clamp is not None else float("nan"),
                        gain_log,
                        nonfinite_pred,
                        nonfinite_lambda,
                        nonfinite_atten,
                        grad_norm_global,
                        float(grad_norm_gen_pre),
                        int(clip_event),
                        z_enc_l2,
                        z_latent_l2,
                    ],
                )
            val_stats = None
            should_run_val = (
                val_interval > 0
                and (step % val_interval) == 0
                and (not args.no_val)
                and proj_loss_active
            )
            def _should_log_val_skip(reason_str):
                if debug_eval_flow:
                    return True
                if step == 1:
                    return True
                if val_interval > 0:
                    interval_reason = f"step%val_interval={step % val_interval}"
                    if reason_str == interval_reason:
                        return False
                return True

            if should_run_val:
                rays_eval = None if ray_split_enabled else rays_per_proj
                subsets = {
                    "test_all": ray_indices["pixel"]["test_idx_all"],
                }
                if ray_split_enabled:
                    subsets["test_fg"] = ray_indices["pixel"]["test_idx_fg"]
                    subsets["test_top10"] = ray_indices["pixel"]["test_idx_top10"]
                    subsets["test_bg"] = ray_indices["pixel"]["test_idx_bg"]
                gain_for_eval = None
                if hybrid_enabled and args.proj_target_source == "counts":
                    gain_for_eval = gain_val_used if gain_val_used is not None else gain_val
                    if val_loader is None or len(val_loader.dataset) == 0:
                        skip_reason = "val_loader missing or empty"
                        print("[eval][warn] val_loader missing or empty; skipping evaluation.", flush=True)
                        if _should_log_val_skip(skip_reason):
                            print(
                                f"[val][skip] reason={skip_reason} val_interval={val_interval} step={step} "
                                f"no_val={args.no_val} proj_loss_active={proj_loss_active}",
                                flush=True,
                            )
                            if debug_eval_flow:
                                print(
                                    f"[val][done] step={step} metrics_keys=None (reason: {skip_reason})",
                                    flush=True,
                                )
                    else:
                        n_val_batches = len(val_loader) if hasattr(val_loader, "__len__") else "?"
                        if debug_eval_flow:
                            print(
                                f"[val][enter] step={step} val_interval={val_interval}",
                                flush=True,
                            )
                            print(
                                f"[val][call] n_val_batches={n_val_batches}",
                                flush=True,
                            )
                        val_stats = evaluate_val_loader(
                            val_loader,
                            generator,
                            z_latent.detach(),
                            z_latent_base.detach(),
                            rays_cache,
                            subsets,
                            device,
                            args,
                            loss_fn,
                            proj_loss_active,
                            rays_eval,
                            args.bg_weight,
                            args.weight_threshold,
                            pa_xflip,
                            W,
                            gain_for_eval,
                            log_proj_metrics_physical,
                            encoder=encoder,
                            z_fuser=z_fuser,
                            z_enc_alpha=z_enc_alpha,
                        )
                        if debug_eval_flow:
                            if val_stats is None:
                                print(
                                    f"[val][done] step={step} metrics_keys=None (reason: evaluate_val_loader returned None)",
                                    flush=True,
                                )
                            else:
                                metrics_keys = _format_val_metrics_keys(sorted(val_stats.keys()))
                                print(
                                    f"[val][done] step={step} metrics_keys={metrics_keys}",
                                    flush=True,
                                )
            else:
                reason_parts = []
                if val_interval <= 0:
                    reason_parts.append(f"val_interval={val_interval}<=0")
                if args.no_val:
                    reason_parts.append("no_val=True")
                if not proj_loss_active:
                    reason_parts.append("proj_loss_active=False")
                if val_interval > 0 and (step % val_interval) != 0:
                    reason_parts.append(f"step%val_interval={step % val_interval}")
                if not reason_parts:
                    reason_parts.append("unknown")
                reason_str = "; ".join(reason_parts)
                if _should_log_val_skip(reason_str):
                    print(
                        f"[val][skip] reason={reason_str} val_interval={val_interval} step={step} "
                        f"no_val={args.no_val} proj_loss_active={proj_loss_active}",
                        flush=True,
                    )
                    if debug_eval_flow:
                        print(
                            f"[val][done] step={step} metrics_keys=None (reason: {reason_str})",
                            flush=True,
                        )
            val_all = val_stats.get("test_all") if isinstance(val_stats, dict) else None
            val_fg = val_stats.get("test_fg") if isinstance(val_stats, dict) else None
            val_top10 = val_stats.get("test_top10") if isinstance(val_stats, dict) else None
            val_bg = val_stats.get("test_bg") if isinstance(val_stats, dict) else None
    
            val_loss = val_all["loss"] if val_all is not None else None
            val_mae = val_all["mae"] if val_all is not None else None
            val_rmse = val_all["rmse"] if val_all is not None else None
            val_nll = val_all["nll"] if val_all is not None else None
            val_dev = val_all["dev"] if val_all is not None else None

            val_loss_fg = val_fg["loss"] if val_fg is not None else None
            val_mae_fg = val_fg["mae"] if val_fg is not None else None
            val_rmse_fg = val_fg["rmse"] if val_fg is not None else None
            val_nll_fg = val_fg["nll"] if val_fg is not None else None
            val_dev_fg = val_fg["dev"] if val_fg is not None else None
            val_loss_bg = val_bg["loss"] if val_bg is not None else None
            val_mae_bg = val_bg["mae"] if val_bg is not None else None
            val_rmse_bg = val_bg["rmse"] if val_bg is not None else None
            val_nll_bg = val_bg["nll"] if val_bg is not None else None
            val_dev_bg = val_bg["dev"] if val_bg is not None else None
            val_pred_mean_bg = val_bg.get("pred_mean") if val_bg is not None else None
            val_target_mean_bg = val_bg.get("target_mean") if val_bg is not None else None
            val_view_all = val_all.get("view") if val_all is not None else None
            val_phys_all = val_all.get("phys") if val_all is not None else None
            val_phys_fg = val_fg.get("phys") if val_fg is not None else None
            val_phys_bg = val_bg.get("phys") if val_bg is not None else None
            val_phys_top10 = val_top10.get("phys") if val_top10 is not None else None
            val_loss_ap = val_view_all["ap"]["loss"] if val_view_all is not None else None
            val_loss_pa = val_view_all["pa"]["loss"] if val_view_all is not None else None
            val_rmse_ap_val = val_view_all["ap"]["rmse"] if val_view_all is not None else None
            val_rmse_pa_val = val_view_all["pa"]["rmse"] if val_view_all is not None else None
            val_nll_ap_val = val_view_all["ap"]["nll"] if val_view_all is not None else None
            val_nll_pa_val = val_view_all["pa"]["nll"] if val_view_all is not None else None
            val_dev_ap_val = val_view_all["ap"]["dev"] if val_view_all is not None else None
            val_dev_pa_val = val_view_all["pa"]["dev"] if val_view_all is not None else None
            val_mae_ap_val = val_view_all["ap"]["mae"] if val_view_all is not None else None
            val_mae_pa_val = val_view_all["pa"]["mae"] if val_view_all is not None else None
            loss_test_all = val_loss
            loss_test_ap = val_loss_ap
            loss_test_pa = val_loss_pa
            mae_test_all = val_mae
            mae_test_ap = val_mae_ap_val
            mae_test_pa = val_mae_pa_val
            rmse_test_all = val_rmse
            rmse_test_ap = val_rmse_ap_val
            rmse_test_pa = val_rmse_pa_val
            nll_test_all = val_nll
            nll_test_ap = val_nll_ap_val
            nll_test_pa = val_nll_pa_val
            dev_test_all = val_dev
            dev_test_ap = val_dev_ap_val
            dev_test_pa = val_dev_pa_val
            val_top10 = val_stats.get("test_top10") if isinstance(val_stats, dict) else None

            val_loss_top10 = val_top10["loss"] if val_top10 is not None else None
            val_mae_top10 = val_top10["mae"] if val_top10 is not None else None
            val_rmse_top10 = val_top10["rmse"] if val_top10 is not None else None
            val_nll_top10 = val_top10["nll"] if val_top10 is not None else None
            val_dev_top10 = val_top10["dev"] if val_top10 is not None else None
    
            act_tv_loss_value = float(loss_act_tv.item())
            act_tv_contrib = float(args.act_tv_weight) * act_tv_loss_value
            act_tv_log_value = float(last_act_tv_value)
            msg = (
                f"[step {step:05d}] loss={loss.item():.6f} | act={loss_act.item():.6f} "
                f"| gain_reg={loss_gain.item():.6f} | ct={loss_ct.item():.6f} "
                f"| sparsity={loss_sparsity.item():.6f} | act_tv_loss={act_tv_loss_value:.6f} "
                f"|                 act_tv_last={act_tv_log_value:.6f} | act_tv_contrib={act_tv_contrib:.6f} "
                f"| ray_tv={loss_ray_tv.item():.6f} | ray_tv_w={loss_ray_tv_w.item():.6f} "
                f"| bg_depth_mass={bg_depth_mass.item():.6f} | bg_depth_mass_w={bg_depth_mass_w.item():.6f} | bg_depth_frac={bg_depth_frac:.4f} "
                f"| tv={loss_tv.item():.6f} "
            )
            if proj_metrics_enabled:
                msg += (
                    f"| ap={loss_ap.item():.6f} | pa={loss_pa.item():.6f} "
                    f"| mae_ap={mae_ap:.6f} | mae_pa={mae_pa:.6f} "
                    f"| rmse_ap={rmse_ap:.6f} | rmse_pa={rmse_pa:.6f} "
                    f"| dev_ap={dev_ap:.6f} | dev_pa={dev_pa:.6f} "
                    f"| predμ_raw=({pred_mean_raw[0]:.3e},{pred_mean_raw[1]:.3e}) predσ_raw=({pred_std_raw[0]:.3e},{pred_std_raw[1]:.3e}) "
                    f"| predμ=({pred_mean[0]:.3e},{pred_mean[1]:.3e}) predσ=({pred_std[0]:.3e},{pred_std[1]:.3e})"
                )
            if log_proj_metrics_physical and (metrics_phys_ap is not None and metrics_phys_pa is not None):
                msg += (
                    f" | phys_mae_ap={metrics_phys_ap['mae']:.6f} | phys_mae_pa={metrics_phys_pa['mae']:.6f} "
                    f"| phys_rmse_ap={metrics_phys_ap['rmse']:.6f} | phys_rmse_pa={metrics_phys_pa['rmse']:.6f} "
                    f"| phys_dev_ap={metrics_phys_ap['dev']:.6f} | phys_dev_pa={metrics_phys_pa['dev']:.6f}"
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
                    f" | val_all_loss={val_loss:.6f} | val_all_rmse={val_rmse:.6f} | val_all_mae={val_mae:.6f} "
                    f"| val_all_dev={val_dev:.6f}"
                )
            if val_fg is not None:
                msg += (
                    f" | val_fg_loss={val_loss_fg:.6f} | val_fg_rmse={val_rmse_fg:.6f} "
                    f"| val_fg_dev={val_dev_fg:.6f} | val_fg_mae={val_mae_fg:.6f}"
                )
            if val_top10 is not None:
                msg += (
                    f" | val_top10_loss={val_loss_top10:.6f} | val_top10_rmse={val_rmse_top10:.6f} "
                    f"| val_top10_dev={val_dev_top10:.6f} | val_top10_mae={val_mae_top10:.6f}"
                )
            if val_bg is not None:
                msg += (
                    f" | val_bg_loss={val_loss_bg:.6f} | val_bg_rmse={val_rmse_bg:.6f} "
                    f"| val_bg_dev={val_dev_bg:.6f} | val_bg_mae={val_mae_bg:.6f}"
                )
            if log_proj_metrics_physical and val_phys_all is not None:
                msg += (
                    f" | val_all_phys_rmse={val_phys_all['rmse']:.6f} | val_all_phys_mae={val_phys_all['mae']:.6f} "
                    f"| val_all_phys_dev={val_phys_all['dev']:.6f}"
                )
            if log_proj_metrics_physical and val_phys_fg is not None:
                msg += (
                    f" | val_fg_phys_rmse={val_phys_fg['rmse']:.6f} | val_fg_phys_mae={val_phys_fg['mae']:.6f} "
                    f"| val_fg_phys_dev={val_phys_fg['dev']:.6f}"
                )
            if log_proj_metrics_physical and val_phys_top10 is not None:
                msg += (
                    f" | val_top10_phys_rmse={val_phys_top10['rmse']:.6f} | val_top10_phys_mae={val_phys_top10['mae']:.6f} "
                    f"| val_top10_phys_dev={val_phys_top10['dev']:.6f}"
                )
            if log_proj_metrics_physical and val_phys_bg is not None:
                msg += (
                    f" | val_bg_phys_rmse={val_phys_bg['rmse']:.6f} | val_bg_phys_mae={val_phys_bg['mae']:.6f} "
                    f"| val_bg_phys_dev={val_phys_bg['dev']:.6f}"
                )
            if val_pred_mean_bg is not None and val_target_mean_bg is not None:
                print(
                    f"[ray-split-bg-check] mean target={val_target_mean_bg[0]:.3e}/{val_target_mean_bg[1]:.3e} "
                    f"mean pred={val_pred_mean_bg[0]:.3e}/{val_pred_mean_bg[1]:.3e}",
                    flush=True,
                )
            if val_view_all is not None:
                msg += (
                    f" | test_ap_loss={val_loss_ap:.6f} | test_ap_rmse={val_rmse_ap_val:.6f} | test_ap_dev={val_dev_ap_val:.6f}"
                    f" | test_pa_loss={val_loss_pa:.6f} | test_pa_rmse={val_rmse_pa_val:.6f} | test_pa_dev={val_dev_pa_val:.6f}"
                    f" | test_ap_mae={val_mae_ap_val:.6f} | test_pa_mae={val_mae_pa_val:.6f}"
                )
            print(msg, flush=True)
            lambda_mean = float("nan")
            lambda_p95 = float("nan")
            lambda_eff_mean = float("nan")
            lambda_eff_p95 = float("nan")
            do_quantiles = args.log_every > 0 and (step % args.log_every == 0 or step == 1)
            if (
                proj_loss_active
                and proj_loss_type == "poisson"
                and lambda_ap_used is not None
                and lambda_pa_used is not None
                and pred_ap is not None
                and pred_pa is not None
            ):
                lambda_all = torch.cat([lambda_ap_used.reshape(-1), lambda_pa_used.reshape(-1)], dim=0).detach()
                if lambda_all.numel() > 0:
                    lambda_mean = float(lambda_all.mean().item())
                    if do_quantiles:
                        lambda_p95 = float(torch.quantile(lambda_all, 0.95).item())
                lambda_eff_all = torch.cat([pred_ap.reshape(-1), pred_pa.reshape(-1)], dim=0).detach()
                if lambda_eff_all.numel() > 0:
                    lambda_eff_mean = float(lambda_eff_all.mean().item())
                    if do_quantiles:
                        lambda_eff_p95 = float(torch.quantile(lambda_eff_all, 0.95).item())
            lambda_floor_frac_log = float("nan")
            if proj_loss_active and proj_loss_type == "poisson" and float(args.poisson_rate_floor) > 0.0:
                lambda_floor_frac_log = float(lambda_floor_frac) if lambda_floor_frac is not None else 0.0
            lambda_ray_tv_log = float(loss_lambda_ray_tv.item()) if torch.is_tensor(loss_lambda_ray_tv) else float("nan")
            append_log(
                log_path,
                [
                    step,
                    loss.item(),
                    loss_ap.item(),
                    loss_pa.item(),
                    loss_act.item(),
                    loss_sparsity.item(),
                    loss_act_tv.item(),
                    loss_ct.item(),
                    loss_ray_tv.item(),
                    loss_ray_tv_w.item(),
                    lambda_ray_tv_log,
                    lambda_floor_frac_log,
                    lambda_mean,
                    lambda_p95,
                    lambda_eff_mean,
                    lambda_eff_p95,
                    bg_depth_mass.item(),
                    bg_depth_mass_w.item(),
                    bg_depth_frac,
                    loss_tv.item(),
                    mae_ap,
                    mae_pa,
                    rmse_ap,
                    rmse_pa,
                    nll_ap,
                    nll_pa,
                    dev_ap,
                    dev_pa,
                    pred_mean[0],
                    pred_mean[1],
                    pred_std[0],
                    pred_std[1],
                    val_loss,
                    val_loss_ap,
                    val_loss_pa,
                    val_mae,
                    val_rmse,
                    val_nll,
                    val_dev,
                    val_mae_ap_val,
                    val_mae_pa_val,
                    val_rmse_ap_val,
                    val_rmse_pa_val,
                    val_nll_ap_val,
                    val_nll_pa_val,
                    val_dev_ap_val,
                    val_dev_pa_val,
                    val_loss_fg,
                    val_mae_fg,
                    val_rmse_fg,
                    val_nll_fg,
                    val_dev_fg,
                    val_loss_top10,
                    val_mae_top10,
                    val_rmse_top10,
                    val_nll_top10,
                    val_dev_top10,
                    val_loss_bg,
                    val_mae_bg,
                    val_rmse_bg,
                    val_nll_bg,
                    val_dev_bg,
                    loss_test_all,
                    loss_test_ap,
                    loss_test_pa,
                    mae_test_all,
                    mae_test_ap,
                    mae_test_pa,
                    rmse_test_all,
                    rmse_test_ap,
                    rmse_test_pa,
                    nll_test_all,
                    nll_test_ap,
                    nll_test_pa,
                    dev_test_all,
                    dev_test_ap,
                    dev_test_pa,
                    iter_ms,
                    optimizer.param_groups[0]["lr"],
                    ray_tv_mode,
                    ray_tv_w_mean,
                ],
            )
            if args.save_every > 0 and (step % args.save_every == 0):
                save_checkpoint(
                    step,
                    generator,
                    optimizer,
                    scaler,
                    ckpt_dir,
                    encoder=encoder,
                    z_fuser=z_fuser,
                    gain_head=gain_head,
                    gain_param=gain_param,
                )
            maybe_render_preview(
                step,
                args,
                generator,
                z_latent.detach(),
                outdir,
                ct_vol,
                act_vol,
                ct_context,
                target_ap=ap,
                target_pa=pa,
                target_ap_counts=ap_counts,
                target_pa_counts=pa_counts,
                patient_id=safe_patient_id,
            )
            pred_path_step = None
            pred_vol_step = None
            if args.export_vol_every > 0 and (step % args.export_vol_every == 0):
                pred_path_step = outdir / f"activity_pred_step_{step:05d}.npy"
                pred_vol_step = export_activity_volume(
                    generator,
                    z_latent.detach(),
                    pred_path_step,
                    args.export_vol_res,
                    device,
                    log_world_range=args.debug_sanity_checks,
                )
            if bool(getattr(args, "final_act_compare", False)) and step >= 200 and (step % 200 == 0):
                if act_vol is not None and act_vol.numel() > 0:
                    if pred_path_step is None:
                        pred_vol_step = export_activity_volume(
                            generator,
                            z_latent.detach(),
                            None,
                            args.export_vol_res,
                            device,
                            log_world_range=args.debug_sanity_checks,
                        )
                    preview_dir = outdir / "preview"
                    step_label = f"step{step:06d}"
                    out_path = preview_dir / f"{step_label}_{safe_patient_id}_act_compare_axial.png"
                    save_final_act_compare_volume_slicing(
                        args,
                        act_vol,
                        outdir,
                        pred_path_step,
                        pred_vol_step,
                        out_path_override=out_path,
                        grid_radius=_radius_to_float(generator.radius),
                        radius_xyz=tuple(float(r) for r in generator.radius_xyz),
                    )
    
        proj_metrics_enabled_final = proj_loss_active and not (
            args.act_only or (args.hybrid and args.proj_loss_weight <= 0.0)
        )
        if proj_metrics_enabled_final:
            prev_flag = generator.use_test_kwargs
            generator.eval()
            generator.use_test_kwargs = True
            with torch.no_grad():
                proj_ap, _, _, _ = generator.render_from_pose(last_z_latent.detach(), generator.pose_ap, ct_context=ct_context)
                proj_pa, _, _, _ = generator.render_from_pose(last_z_latent.detach(), generator.pose_pa, ct_context=ct_context)
            generator.train()
            generator.use_test_kwargs = prev_flag or False
    
            use_counts_final = (
                (ap_counts is not None)
                and (pa_counts is not None)
                and (ap_counts.numel() > 0)
                and (pa_counts.numel() > 0)
            )
            if proj_loss_type == "poisson":
                if proj_loss_active:
                    lambda_ap_used = compute_poisson_rate(proj_ap, args.poisson_rate_mode, eps=1e-6)
                    lambda_pa_used = compute_poisson_rate(proj_pa, args.poisson_rate_mode, eps=1e-6)
                else:
                    lambda_ap_used = proj_ap
                    lambda_pa_used = proj_pa
            else:
                lambda_ap_used = proj_ap
                lambda_pa_used = proj_pa
            if use_counts_final and proj_loss_active:
                gain_val_final = None
                if gain_param is not None:
                    gain_val_final = F.softplus(gain_param)
                elif gain_head is not None:
                    # compute z_enc for final batch (encoder sees normalized AP/PA)
                    proj_scale_enc = compute_proj_scale(ap, pa, args.proj_scale_source, meta)
                    proj_scale_enc = torch.clamp(proj_scale_enc, min=1e-6)
                    enc_input = build_encoder_input(
                        ap_enc_input,
                        pa_enc_input,
                        ct_vol,
                        proj_scale_enc,
                        args.encoder_proj_transform,
                        args.encoder_use_ct,
                    )
                    z_enc_final = encoder(enc_input)
                    gain_val_final = F.softplus(gain_head(z_enc_final))
                if gain_val_final is not None:
                    g_min = float(args.gain_clamp_min) if args.gain_clamp_min is not None else None
                    g_max = args.gain_clamp_max
                    if g_min is not None or g_max is not None:
                        gmin = g_min if g_min is not None else -float("inf")
                        gmax = g_max if g_max is not None else float("inf")
                        gain_val_final = torch.clamp(gain_val_final, min=gmin, max=gmax)
                    lambda_ap_used = lambda_ap_used * gain_val_final
                    lambda_pa_used = lambda_pa_used * gain_val_final
    
            H, W = generator.H, generator.W
            ap_np = lambda_ap_used[0].reshape(H, W).detach().cpu().numpy()
            pa_np = lambda_pa_used[0].reshape(H, W).detach().cpu().numpy()
            fp = outdir / "preview"
            fp.mkdir(parents=True, exist_ok=True)
            if args.log_quantiles_final_only:
                if use_counts_final:
                    ap_t_np = ap_counts.detach().cpu().numpy()[0]
                    pa_t_np = pa_counts.detach().cpu().numpy()[0]
                else:
                    ap_t_np = ap.detach().cpu().numpy()[0] if ap is not None else None
                    pa_t_np = pa.detach().cpu().numpy()[0] if pa is not None else None
                log_projection_quantiles(lambda_ap_used, lambda_pa_used, ap_target=ap_t_np, pa_target=pa_t_np, tag="final")
                if log_proj_metrics_physical:
                    log_projection_quantiles_scaled(
                        lambda_ap_used,
                        lambda_pa_used,
                        ap_target=ap_t_np,
                        pa_target=pa_t_np,
                        tag="final_phys",
                        ap_scale=scale_ap_used,
                        pa_scale=scale_pa_used,
                    )
                if use_counts_final:
                    raw_ap_q = np.quantile(proj_ap.detach().cpu().numpy().ravel(), [0.5, 0.8, 0.95, 0.99])
                    raw_pa_q = np.quantile(proj_pa.detach().cpu().numpy().ravel(), [0.5, 0.8, 0.95, 0.99])
                    soft_ap_q = np.quantile(F.softplus(proj_ap).detach().cpu().numpy().ravel(), [0.5, 0.8, 0.95, 0.99])
                    soft_pa_q = np.quantile(F.softplus(proj_pa).detach().cpu().numpy().ravel(), [0.5, 0.8, 0.95, 0.99])
                    lam_ap_q = np.quantile(lambda_ap_used.detach().cpu().numpy().ravel(), [0.5, 0.8, 0.95, 0.99])
                    lam_pa_q = np.quantile(lambda_pa_used.detach().cpu().numpy().ravel(), [0.5, 0.8, 0.95, 0.99])
                    tgt_ap_q = np.quantile(ap_t_np.ravel(), [0.5, 0.8, 0.95, 0.99]) if ap_t_np is not None else None
                    tgt_pa_q = np.quantile(pa_t_np.ravel(), [0.5, 0.8, 0.95, 0.99]) if pa_t_np is not None else None
                    print(
                        "[quantiles][final][diag] "
                        f"pred_raw_ap p50={raw_ap_q[0]:.3e} p80={raw_ap_q[1]:.3e} p95={raw_ap_q[2]:.3e} p99={raw_ap_q[3]:.3e} | "
                        f"pred_raw_pa p50={raw_pa_q[0]:.3e} p80={raw_pa_q[1]:.3e} p95={raw_pa_q[2]:.3e} p99={raw_pa_q[3]:.3e}",
                        flush=True,
                    )
                    print(
                        "[quantiles][final][diag] "
                        f"softplus_ap p50={soft_ap_q[0]:.3e} p80={soft_ap_q[1]:.3e} p95={soft_ap_q[2]:.3e} p99={soft_ap_q[3]:.3e} | "
                        f"softplus_pa p50={soft_pa_q[0]:.3e} p80={soft_pa_q[1]:.3e} p95={soft_pa_q[2]:.3e} p99={soft_pa_q[3]:.3e}",
                        flush=True,
                    )
                    print(
                        "[quantiles][final][diag] "
                        f"lambda_ap p50={lam_ap_q[0]:.3e} p80={lam_ap_q[1]:.3e} p95={lam_ap_q[2]:.3e} p99={lam_ap_q[3]:.3e} | "
                        f"lambda_pa p50={lam_pa_q[0]:.3e} p80={lam_pa_q[1]:.3e} p95={lam_pa_q[2]:.3e} p99={lam_pa_q[3]:.3e}",
                        flush=True,
                    )
                    if tgt_ap_q is not None and tgt_pa_q is not None:
                        print(
                            "[quantiles][final][diag] "
                            f"target_ap p50={tgt_ap_q[0]:.3e} p80={tgt_ap_q[1]:.3e} p95={tgt_ap_q[2]:.3e} p99={tgt_ap_q[3]:.3e} | "
                            f"target_pa p50={tgt_pa_q[0]:.3e} p80={tgt_pa_q[1]:.3e} p95={tgt_pa_q[2]:.3e} p99={tgt_pa_q[3]:.3e}",
                            flush=True,
                        )
                    if lam_ap_q[2] < 5 or lam_pa_q[2] < 5:
                        print("[WARN] Final pred quantiles appear non-count-scaled; logging wrong tensor.", flush=True)
            save_img(ap_np, fp / "final_AP.png", "AP final")
            save_img(pa_np, fp / "final_PA.png", "PA final")
            print("🖼️ Finale Previews gespeichert.", flush=True)
            print("   ", (fp / "final_AP.png").resolve(), flush=True)
            print("   ", (fp / "final_PA.png").resolve(), flush=True)

        save_final_sagittal_depth_consistency(
            args,
            generator,
            last_z_latent.detach(),
            act_vol,
            ct_context,
            outdir,
        )

        pred_path_final = outdir / "activity_pred_final.npy"
        pred_vol_final = export_activity_volume(
            generator,
            last_z_latent.detach(),
            pred_path_final,
            args.export_vol_res,
            device,
            log_world_range=args.debug_sanity_checks,
        )
        final_label = f"final_step{last_step:06d}_{last_safe_patient_id}"
        final_preview_dir = outdir / "preview"
        final_out_path = final_preview_dir / f"{final_label}_act_compare_axial.png"
        save_final_act_compare_volume_slicing(
            args,
            act_vol,
            outdir,
            pred_path_final,
            pred_vol_final,
            out_path_override=final_out_path,
            grid_radius=_radius_to_float(generator.radius),
            radius_xyz=tuple(float(r) for r in generator.radius_xyz),
            force_save=True,
        )
        final_test_stats = None
        n_test_batches = 0
        test_metadata = None
        if run_final_test and test_loader is not None and len(test_loader.dataset) > 0:
            try:
                n_test_batches = len(test_loader)
            except Exception:
                n_test_batches = -1
            test_subsets = {
                "test_all": ray_indices["pixel"]["test_idx_all"],
            }
            if ray_split_enabled:
                test_subsets["test_fg"] = ray_indices["pixel"]["test_idx_fg"]
                test_subsets["test_top10"] = ray_indices["pixel"]["test_idx_top10"]
                test_subsets["test_bg"] = ray_indices["pixel"]["test_idx_bg"]
            rays_eval_test = None if ray_split_enabled else rays_per_proj
            prev_use_test_kwargs = generator.use_test_kwargs
            generator.eval()
            with torch.no_grad():
                final_test_stats = evaluate_val_loader(
                    test_loader,
                    generator,
                    last_z_latent.detach(),
                    z_latent_base.detach(),
                    rays_cache,
                    test_subsets,
                    device,
                    args,
                    loss_fn,
                    True,
                    rays_eval_test,
                    args.bg_weight,
                    args.weight_threshold,
                    pa_xflip,
                    W,
                    None,
                    log_proj_metrics_physical,
                    encoder=encoder,
                    z_fuser=z_fuser,
                    z_enc_alpha=z_enc_alpha,
                )
            if prev_use_test_kwargs:
                generator.eval()
            else:
                generator.train()
            metrics_keys = sorted(final_test_stats.keys()) if final_test_stats else []
            print(
                f"[test][final] n_batches={n_test_batches} metrics_keys={metrics_keys}",
                flush=True,
            )
            if final_test_stats is not None and final_test_stats.get("test_all") is not None:
                test_all = final_test_stats.get("test_all")
                print(
                    f"[test][final][summary] test_all_loss={test_all['loss']:.6f} "
                    f"test_all_rmse={test_all['rmse']:.6f} test_all_dev={test_all['dev']:.6f} test_all_mae={test_all['mae']:.6f}",
                    flush=True,
                )
            patient_ids = (
                dataset_split_stats.get("patient_ids_test") if dataset_split_stats is not None else None
            )
            test_metadata = {
                "n_test_batches": n_test_batches,
                "metrics": final_test_stats,
                "patient_ids": patient_ids,
            }
            test_metrics_path = outdir / "test_metrics.json"
            test_metrics_path.write_text(json.dumps(test_metadata, indent=2))
            save_test_volume_slices(
                args,
                generator,
                z_latent_base.detach(),
                test_loader,
                outdir,
                device,
                encoder=encoder,
                z_fuser=z_fuser,
                z_enc_alpha=z_enc_alpha,
            )
        else:
            reason = "run_final_test disabled or no test_loader"
            print(f"[test][final] skipped reason={reason}", flush=True)
        save_checkpoint(
            max_steps,
            generator,
            optimizer,
            scaler,
            ckpt_dir,
            encoder=encoder,
            z_fuser=z_fuser,
            gain_head=gain_head,
            gain_param=gain_param,
        )
        print("✅ Training run finished.", flush=True)
        if args.debug_z_sensitivity and debug_z_sample is not None:
            _log_debug_z_sensitivity(debug_z_sample, generator, args)
    
    
        print(f"[end] reached step={last_step} max_steps={max_steps} exit_reason={exit_reason}", flush=True)
    except Exception as exc:
        exit_reason = "exception"
        print(f"[exception] {exc.__class__.__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        print(f"[end] reached step={last_step} max_steps={max_steps} exit_reason={exit_reason}", flush=True)
def _apply_debug_projection_postprocessing(pred_ap_raw, pred_pa_raw, sample, args):
    if args.proj_loss_type == "poisson":
        pred_ap = compute_poisson_rate(pred_ap_raw, args.poisson_rate_mode, eps=1e-6)
        pred_pa = compute_poisson_rate(pred_pa_raw, args.poisson_rate_mode, eps=1e-6)
    else:
        pred_ap = pred_ap_raw
        pred_pa = pred_pa_raw

    if sample.get("use_counts"):
        gain_val = sample.get("gain_val")
        if gain_val is not None:
            pred_ap = pred_ap * gain_val
            pred_pa = pred_pa * gain_val
        else:
            gain_param = sample.get("gain_param")
            if gain_param is not None:
                gain_scale = F.softplus(gain_param)
                pred_ap = pred_ap * gain_scale
                pred_pa = pred_pa * gain_scale

    poisson_floor = float(sample.get("poisson_rate_floor", 0.0))
    floor_mode = sample.get("poisson_rate_floor_mode", args.poisson_rate_floor_mode)
    if sample.get("use_counts") and poisson_floor > 0.0 and args.proj_loss_type == "poisson":
        pred_ap, _ = apply_poisson_rate_floor(pred_ap, poisson_floor, floor_mode)
        pred_pa, _ = apply_poisson_rate_floor(pred_pa, poisson_floor, floor_mode)
    return pred_ap, pred_pa


def _evaluate_debug_projection(generator, sample, z_variant, args):
    ct_context = sample.get("ct_context")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=sample["amp_enabled"]):
            pred_ap_raw, _ = render_minibatch(generator, z_variant, sample["ray_batch_ap"], ct_context=ct_context)
            pred_pa_raw, _ = render_minibatch(generator, z_variant, sample["ray_batch_pa"], ct_context=ct_context)
    pred_ap, pred_pa = _apply_debug_projection_postprocessing(pred_ap_raw, pred_pa_raw, sample, args)
    metrics_ap = compute_projection_metrics(pred_ap, sample["target_ap"])
    metrics_pa = compute_projection_metrics(pred_pa, sample["target_pa"])
    return pred_ap, pred_pa, metrics_ap, metrics_pa


def _evaluate_debug_act(generator, sample, z_variant):
    coords = sample.get("act_coords")
    if coords is None:
        return None, None
    with torch.no_grad():
        pred_act, pred_act_raw = query_emission_at_points(generator, z_variant, coords, return_raw=True)
    return pred_act, pred_act_raw


def _log_debug_z_sensitivity(sample, generator, args):
    if sample is None:
        return

    base_ap = sample["pred_ap_normal"]
    base_pa = sample["pred_pa_normal"]
    base_abs_mean = base_ap.abs().mean() + base_pa.abs().mean()
    denom = 0.5 * base_abs_mean + 1e-8

    variants = [
        ("normal", sample["z_latent"]),
        ("zero", torch.zeros_like(sample["z_latent"])),
    ]
    z_latent = sample["z_latent"]
    if z_latent.shape[0] > 1:
        perm = torch.randperm(z_latent.shape[0], device=z_latent.device)
        variants.append(("swap", z_latent[perm].clone()))
    else:
        variants.append(("swap", z_latent + 0.01 * torch.randn_like(z_latent)))
    variants.append(("noise", z_latent + 0.01 * torch.randn_like(z_latent)))

    print("[debug-z] Starting sensitivity check", flush=True)
    for name, z_variant in variants:
        if name == "normal":
            pred_ap = base_ap
            pred_pa = base_pa
            metrics_ap = compute_projection_metrics(pred_ap, sample["target_ap"])
            metrics_pa = compute_projection_metrics(pred_pa, sample["target_pa"])
        else:
            pred_ap, pred_pa, metrics_ap, metrics_pa = _evaluate_debug_projection(
                generator,
                sample,
                z_variant.detach(),
                args,
            )

        diff_ap = (base_ap - pred_ap).abs().mean().item()
        diff_pa = (base_pa - pred_pa).abs().mean().item()
        z_effect_proj = 0.5 * (diff_ap + diff_pa) / denom

        def _fmt_metrics(metrics):
            return f"mae={metrics['mae']:.6f} rmse={metrics['rmse']:.6f} nll={metrics['nll']:.6f}"

        print(
            f"[debug-z] variant={name} proj_ap({_fmt_metrics(metrics_ap)}) "
            f"proj_pa({_fmt_metrics(metrics_pa)}) z_effect_proj={z_effect_proj:.3e}",
            flush=True,
        )

        if sample.get("act_coords") is not None and sample.get("act_pred_raw") is not None:
            act_base_raw = sample["act_pred_raw"]
            act_result, act_result_raw = _evaluate_debug_act(generator, sample, z_variant.detach())
            if act_result_raw is not None:
                diff_act = (act_base_raw - act_result_raw).abs().mean().item()
                stats = tensor_stats(act_result_raw)
                print(
                    f"[debug-z][act] variant={name} act_raw_stats={fmt_stats(stats)} act_diff_mean={diff_act:.6e}",
                    flush=True,
                )


def split_by_patient_id(
    dataset,
    seed: int,
    split_mode: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    train_count: int,
    val_count: int,
    test_count: int,
):
    """
    Teilt dataset deterministic nach patient_id auf train/val/test auf.
    Returns: (Subset, Subset, Subset, stats_dict)
    """
    if dataset is None:
        raise ValueError("dataset darf nicht None sein.")
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("Ratios müssen >0 sein.")
    train_ratio = train_ratio / total_ratio
    val_ratio = val_ratio / total_ratio
    test_ratio = test_ratio / total_ratio

    n_samples = len(dataset)
    patient_ids = []
    pid_to_indices = {}
    for idx in range(n_samples):
        pid = dataset.get_patient_id(idx) if hasattr(dataset, "get_patient_id") else None
        if pid is None:
            continue
        patient_ids.append(pid)
        pid_to_indices.setdefault(pid, []).append(idx)
    patient_ids = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    split_mode = str(split_mode or "ratios").lower()

    train_pids = []
    val_pids = []
    test_pids = []
    if split_mode == "counts":
        for count in (train_count, val_count, test_count):
            if count is None or count < 0:
                raise ValueError("Bei split_mode='counts' müssen alle split_*_count >= 0 gesetzt sein.")
        total_counts = train_count + val_count + test_count
        if total_counts != len(patient_ids):
            raise ValueError(
                f"split_mode='counts' erwartet sum(split_*_count)==#patienten ({len(patient_ids)}), "
                f"bekommen {total_counts}."
            )
        offset = 0
        train_pids = patient_ids[offset : offset + train_count]
        offset += train_count
        val_pids = patient_ids[offset : offset + val_count]
        offset += val_count
        test_pids = patient_ids[offset : offset + test_count]
    else:
        n_patients = len(patient_ids)
        if n_patients == 0:
            train_pids = val_pids = test_pids = []
        else:
            if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
                raise ValueError("split ratios müssen >= 0 sein.")
            total_ratio = train_ratio + val_ratio + test_ratio
            if total_ratio == 0:
                raise ValueError("split ratios dürfen nicht alle 0 sein.")
            train_ratio_norm = train_ratio / total_ratio
            val_ratio_norm = val_ratio / total_ratio
            n_train = int(math.floor(train_ratio_norm * n_patients))
            n_val = int(math.floor(val_ratio_norm * n_patients))
            n_test = max(n_patients - n_train - n_val, 0)
            train_pids = patient_ids[:n_train]
            val_pids = patient_ids[n_train : n_train + n_val]
            test_pids = patient_ids[n_train + n_val : n_train + n_val + n_test]

    def _collect(indices_pids):
        idxs = []
        for pid in indices_pids:
            idxs.extend(pid_to_indices.get(pid, []))
        return idxs

    train_indices = _collect(train_pids)
    val_indices = _collect(val_pids)
    test_indices = _collect(test_pids)

    def _summary(pid_list):
        if not pid_list:
            return (0, 0.0, 0)
        counts = [len(pid_to_indices[pid]) for pid in pid_list]
        return (int(np.min(counts)), float(np.median(counts)), int(np.max(counts)))

    stats = {
        "n_samples_train": len(train_indices),
        "n_samples_val": len(val_indices),
        "n_samples_test": len(test_indices),
        "n_patients_train": len(train_pids),
        "n_patients_val": len(val_pids),
        "n_patients_test": len(test_pids),
        "patient_ids_train": train_pids if len(train_pids) <= 30 else None,
        "patient_ids_val": val_pids if len(val_pids) <= 30 else None,
        "patient_ids_test": test_pids if len(test_pids) <= 30 else None,
        "samples_per_patient_train": _summary(train_pids),
        "samples_per_patient_val": _summary(val_pids),
        "samples_per_patient_test": _summary(test_pids),
        "patient_ids_train_full": list(train_pids),
        "patient_ids_val_full": list(val_pids),
        "patient_ids_test_full": list(test_pids),
        "split_mode": split_mode,
        "split_seed": int(seed),
    }

    train_subset = Subset(dataset, train_indices) if train_indices else Subset(dataset, [])
    val_subset = Subset(dataset, val_indices) if val_indices else Subset(dataset, [])
    test_subset = Subset(dataset, test_indices) if test_indices else Subset(dataset, [])

    return train_subset, val_subset, test_subset, stats


if __name__ == "__main__":
    train()
