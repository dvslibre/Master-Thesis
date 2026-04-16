#!/usr/bin/env python3
"""Postprocess run results: total-activity bias, voxel metrics, optional organ/projection metrics.

python postprocessing.py \
  --run-dir results_spect \
  --split-json results_spect/split.json \
  --manifest data/manifest_abs.csv \
  --out-dir results_spect/postproc \
  --save-proj-npy \
  --mask-path-pattern /home/mnguest12/projects/thesis/Data_Processing/{phantom}/out/mask.npy \
  --device cuda

"""
import argparse
import cProfile
import csv
import importlib
import importlib.util
import io
import json
import logging
import pstats
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - best effort
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_AVAILABLE = False

_LOG = logging.getLogger(__name__)

_PHYSICS_PROJECTOR_IMPORTS: Optional[tuple[type, Callable[..., Any]]] = None
_TRAIN_EMISSION_IMPORTS: Optional[dict[str, Any]] = None

METRICS_CSV_HEADER = [
    "phantom_id",
    "vol_mae_vol",
    "vol_rmse_vol",
    "activity_bias_vol",
    "activity_rel_abs_error_vol",
    "voxel_mae_fg",
    "voxel_rmse_fg",
    "voxel_n_fg",
    "fg_tau",
    "proj_mae_counts",
    "proj_poisson_dev_counts",
    "proj_mae_counts_clean",
    "proj_poisson_dev_counts_clean",
    "proj_mae_counts_noisy",
    "proj_poisson_dev_counts_noisy",
    "proj_metrics_target_mode_used",
    "organ_rel_error_total_activity_active_mean",
    "active_organ_fraction_mae",
    "label_averaged_organ_fraction_mae",
    "active_organ_fraction_n",
    "organ_active_n",
    "organ_inactive_n",
    "inactive_organs_pred_sum",
    "inactive_organs_pred_frac_of_pred",
    "inactive_organs_pred_frac_of_gt",
    "pred_total_sum_predgrid",
    "pred_in_body_sum_predgrid",
    "pred_outside_body_sum_predgrid",
    "pred_in_gt_active_sum_predgrid",
    "pred_not_gt_active_sum_predgrid",
    "pred_in_body_frac_predgrid",
    "pred_outside_body_frac_predgrid",
    "pred_in_gt_active_frac_predgrid",
    "pred_not_gt_active_frac_predgrid",
    "pred_total_sum_gtgrid",
    "pred_in_body_sum_gtgrid",
    "pred_outside_body_sum_gtgrid",
    "pred_in_gt_active_sum_gtgrid",
    "pred_not_gt_active_sum_gtgrid",
    "pred_in_body_frac_gtgrid",
    "pred_outside_body_frac_gtgrid",
    "pred_in_gt_active_frac_gtgrid",
    "pred_not_gt_active_frac_gtgrid",
    "pred_outside_mask_sum",
    "pred_outside_mask_frac",
    "A_gt_vol",
    "A_pred_vol",
    "A_pred_native",
    "timestamp",
    "git_hash",
    "config_path",
    "proj_status",
    "proj_domain",
    "proj_is_normalized",
    "proj_norm_factor",
    "calibration_scale",
    "calibrate_scale_enabled",
]

FG_THRESHOLD = 1e-6
ACTIVE_ORGAN_EPS = 1e-6
ORGAN_ACTIVE_TOLERANCE = 1e-12


class BlockTimer:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.blocks: dict[str, float] = {}

    @contextmanager
    def block(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.blocks[name] = time.perf_counter() - start


BLOCK_NAMES = (
    "load_gt",
    "load_pred",
    "resample",
    "voxel_metrics",
    "organ_metrics",
    "plots",
    "projections",
)
FLAG_ATTRS = (
    "timing",
    "profile",
    "skip_plots",
    "skip_organ_table",
    "skip_organ_metrics",
    "fast_resample_roi",
)

def _compute_proj_metrics_from_arrays(
    pred_ap: np.ndarray,
    pred_pa: np.ndarray,
    target_ap: np.ndarray,
    target_pa: np.ndarray,
    device: str,
) -> tuple[float, float]:
    tr = _get_train_emission_imports()
    compute_projection_metrics = tr["compute_projection_metrics"]
    # robust gegen non-contiguous Arrays (z.B. flip/slice)
    pred_ap_t   = torch.from_numpy(np.ascontiguousarray(pred_ap,   dtype=np.float32)).reshape(1, -1).to(device)
    pred_pa_t   = torch.from_numpy(np.ascontiguousarray(pred_pa,   dtype=np.float32)).reshape(1, -1).to(device)
    target_ap_t = torch.from_numpy(np.ascontiguousarray(target_ap, dtype=np.float32)).reshape(1, -1).to(device)
    target_pa_t = torch.from_numpy(np.ascontiguousarray(target_pa, dtype=np.float32)).reshape(1, -1).to(device)

    metrics_ap = compute_projection_metrics(pred_ap_t, target_ap_t)
    metrics_pa = compute_projection_metrics(pred_pa_t, target_pa_t)

    mae = 0.5 * (metrics_ap["mae"] + metrics_pa["mae"])
    dev = 0.5 * (metrics_ap["dev"] + metrics_pa["dev"])
    return float(mae), float(dev)

def parse_args():
    parser = argparse.ArgumentParser(description="Postprocess pieNeRF evaluation run")
    parser.add_argument("--run-dir", default="results_spect", help="path to inference run directory")
    parser.add_argument("--split-json", default="results_spect/split.json", help="run split json path")
    parser.add_argument("--manifest", default="data/manifest_abs.csv", help="dataset manifest csv")
    parser.add_argument("--config", default="configs/spect.yaml", help="config yaml (radius/voxel_mm)")
    parser.add_argument("--out-dir", default=None, help="postprocessing output directory (defaults to run-dir/postproc)")
    parser.add_argument("--mask-path-pattern", default="data/{phantom}/out/mask.npy", help="mask path template")
    parser.add_argument(
        "--pred-act-pattern",
        default="activity_pred_final.npy,activity_pred_step_*.npy",
        help="comma separated glob patterns (relative to run dir) to search for pred act npy",
    )
    parser.add_argument("--pred-act-path", help="explicit pred activity file (overrides pattern)")
    parser.add_argument(
        "--pred-act-per-phantom",
        action="store_true",
        help="expect run_dir/{phantom}/activity_pred_*.npy",
    )
    parser.add_argument("--checkpoint", help="checkpoint path (optional) to render AP/PA projections")
    parser.add_argument(
        "--proj-forward-model",
        choices=["physics", "train"],
        default="physics",
        help="projection forward model: physics projector or training renderer path",
    )
    parser.add_argument(
        "--force-use-attenuation",
        action="store_true",
        help="force use_attenuation=True for train-forward renderer, regardless of training config",
    )
    parser.add_argument("--pred-ap-path", help="explicit pred AP projection (.npy)")
    parser.add_argument("--pred-pa-path", help="explicit pred PA projection (.npy)")
    parser.add_argument(
        "--pred-proj-search",
        action="store_true",
        default=True,
        help="automatically search run_dir[/{phantom}] for pred_ap.npy/pred_pa.npy",
    )
    parser.add_argument("--save-proj-npy", action="store_true", help="store rendered pred AP/PA as .npy")
    parser.add_argument(
        "--save-proj-png",
        action="store_true",
        help="save pred/gt projection PNG previews in postproc/<phantom>/plots",
    )
    parser.add_argument(
        "--calibrate-scale",
        action="store_true",
        help="apply per-phantom joint AP/PA least-squares scale before metrics",
    )
    parser.add_argument(
        "--ls-calibrate-global",
        action="store_true",
        help="apply one joint AP+PA least-squares scale alpha to train-forward projections",
    )
    parser.add_argument(
        "--ls-calibrate-eps",
        type=float,
        default=1e-12,
        help="epsilon added to LS denominator for global AP+PA calibration",
    )
    parser.add_argument("--render-projections", action="store_true", help="render AP/PA projections from volumes")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="torch device")
    parser.add_argument(
        "--proj-domain",
        choices=["counts", "normalized"],
        default="counts",
        help="Choose counts or normalized projection domain",
    )
    parser.add_argument(
        "--proj-metrics-target",
        choices=["clean_counts", "noisy_counts"],
        default="clean_counts",
        help=(
            "Target for legacy projection metrics keys (proj_mae_counts/proj_poisson_dev_counts). "
            "clean_counts uses manifest counts; noisy_counts uses "
            "test_slices/<phantom>/noisy_{ap,pa}_counts.npy from test-noise export."
        ),
    )
    parser.add_argument(
        "--psf-sigma",
        type=float,
        default=None,
        help="override scatter sigma for the forward projector",
    )
    parser.add_argument(
        "--z0-slices",
        type=int,
        default=None,
        help="override number of slices before PSF takes effect",
    )
    parser.add_argument(
        "--proj-sensitivity-cps-per-mbq",
        type=float,
        default=None,
        help="override sensitivity (counts per MBq) for projection simulation",
    )
    parser.add_argument(
        "--proj-acq-time-s",
        type=float,
        default=None,
        help="override acquisition time (seconds) for projection simulation",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--timing", action="store_true", help="log per-phantom block durations")
    parser.add_argument("--profile", action="store_true", help="collect cProfile stats for the entire run")
    parser.add_argument("--profile-topk", type=int, default=30, help="number of top cProfile entries to display")
    parser.add_argument("--skip-plots", action="store_true", help="do not emit matplotlib organ plots")
    parser.add_argument("--skip-organ-table", action="store_true", help="do not write organ_table.csv")
    parser.add_argument("--skip-organ-metrics", action="store_true", help="skip organ stats/metrics completely")
    parser.add_argument("--fast-resample-roi", action="store_true", help="fast approximation: resample predictions only on ROI")
    parser.add_argument(
        "--save-active-organ-plots",
        action="store_true",
        help="save bar plots for GT active organs (absolute and fractional)",
    )
    parser.add_argument(
        "--save-act-compare-5slices",
        action="store_true",
        help="save 5-slice GT-vs-pred activity comparison in postproc/<phantom>/plots",
    )
    parser.add_argument(
        "--act-compare-axis",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="slice axis for 5-slice activity comparison",
    )
    parser.add_argument(
        "--act-compare-stride",
        type=int,
        default=1,
        help="only allow slice indices i where i %% stride == 0 for activity comparison",
    )
    parser.add_argument(
        "--act-compare-outname",
        type=str,
        default="act_compare_5slices_axis{axis}.png",
        help="output filename template (supports {axis}) for activity comparison",
    )
    parser.add_argument(
        "--debug-orientation-search",
        action="store_true",
        help="log all orientation candidates and scores for pred->GT-grid alignment debug (train path)",
    )
    parser.add_argument(
        "--save-orientation-debug-volumes",
        action="store_true",
        help="save gt_roi/pred_roi_gtgrid dumps (+meta) for orientation sanity checks",
    )
    return parser.parse_args()


def git_short_hash():
    try:
        proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return proc.stdout.strip()
    except Exception:  # pragma: no cover - best effort
        return "unknown"


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def read_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"manifest missing: {path}")
    result = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("patient_id")
            if not pid:
                continue
            result[pid] = row
    return result


def resolve_manifest_path(entry: dict, candidates: list[str]) -> Path | None:
    for key in candidates:
        val = entry.get(key)
        if not val:
            continue
        path = Path(val)
        if path.exists():
            return path
    return None


def read_split_ids(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"split json missing: {path}")
    data = json.loads(path.read_text())
    ids = data.get("test_ids")
    if ids is None:
        raise KeyError("split json missing 'test_ids'")
    return list(ids)


def find_pred_activity_paths(
    run_dir: Path,
    test_ids: list[str],
    args,
) -> dict[str, Path]:
    if args.pred_act_path:
        explicit = Path(args.pred_act_path)
        if not explicit.is_absolute():
            explicit = run_dir / explicit
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return {pid: explicit for pid in test_ids}

    if args.pred_act_per_phantom:
        if not args.pred_act_pattern:
            raise ValueError("--pred-act-pattern is required when --pred-act-per-phantom is set")
        mapping = {}
        for pid in test_ids:
            rel = args.pred_act_pattern.format(phantom=pid, pid=pid)
            candidate = run_dir / rel
            if not candidate.exists():
                raise FileNotFoundError(f"predicted activity for {pid} not found at {candidate}")
            mapping[pid] = candidate
            if getattr(args, "verbose", False):
                _LOG.info("[pred-act] %s -> %s", pid, candidate)
        return mapping

    patterns = [p.strip() for p in args.pred_act_pattern.split(",") if p.strip()]
    candidates = []
    for pattern in patterns:
        candidates.extend(sorted(run_dir.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(f"no pred act files found in {run_dir} with {patterns}")
    if len(candidates) == 1:
        return {pid: candidates[0] for pid in test_ids}
    # multiple files: pick final if exists, else highest step
    final = [p for p in candidates if p.name == "activity_pred_final.npy"]
    if final:
        return {pid: final[0] for pid in test_ids}
    highest = sorted_by_step(candidates)[-1]
    if len(test_ids) == 1:
        return {test_ids[0]: highest}
    raise RuntimeError(
        "multiple pred files found but cannot attribute to phantom_id; use --pred-act-path or --pred-act-per-phantom"
    )


def find_projection_candidates(args, run_dir: Path, out_dir: Path, phantom_id: str) -> dict:
    if args.pred_ap_path and args.pred_pa_path:
        return {"ap": Path(args.pred_ap_path), "pa": Path(args.pred_pa_path), "status": "explicit"}
    if not args.pred_proj_search:
        return {}
    candidates = []
    base_candidates = [
        (out_dir / phantom_id / "pred_ap.npy", out_dir / phantom_id / "pred_pa.npy"),
        (run_dir / phantom_id / "pred_ap.npy", run_dir / phantom_id / "pred_pa.npy"),
        (run_dir / "pred_ap.npy", run_dir / "pred_pa.npy"),
        (run_dir / "preview" / "pred_ap.npy", run_dir / "preview" / "pred_pa.npy"),
    ]
    for ap_path, pa_path in base_candidates:
        if ap_path.exists() and pa_path.exists():
            return {"ap": ap_path, "pa": pa_path, "status": "loaded_npy"}
    return {}


def load_projection_arrays(paths: dict[str, Path], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ap = np.load(paths["ap"]).astype(np.float32, copy=False)
    pa = np.load(paths["pa"]).astype(np.float32, copy=False)
    if ap.shape != shape or pa.shape != shape:
        raise ValueError("projection shape mismatch with GT")
    return ap, pa


def sorted_by_step(paths: list[Path]) -> list[Path]:
    def step_value(path: Path) -> int:
        parts = path.stem.split("_")
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0

    return sorted(paths, key=step_value)


def load_array(path: Path, *, mmap: bool = False) -> np.ndarray:
    load_kwargs = {"mmap_mode": "r"} if mmap else {}
    arr = np.load(path, **load_kwargs)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def spacing_from_meta(act_path: Path, fallback_mm: float) -> tuple[float, float, float]:
    meta_path = act_path.parent / "meta_simple.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        sd_mm = meta.get("sd_mm")
        if sd_mm is not None:
            sd = float(sd_mm)
            sd_cm = sd / 10.0
            return sd_cm, sd_cm, sd_cm
    spacing_cm = fallback_mm / 10.0
    return spacing_cm, spacing_cm, spacing_cm


def resample_pred_to_gt(pred: np.ndarray, target_shape: tuple[int, int, int], device: str) -> np.ndarray:
    tensor = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        resampled = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)
    return resampled.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def resample_mask_to_pred(mask_GT: np.ndarray, pred_shape: tuple[int, int, int]) -> np.ndarray:
    if mask_GT.shape == pred_shape:
        return mask_GT.astype(np.int32, copy=True)
    tensor = torch.from_numpy(mask_GT.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        resampled = F.interpolate(tensor, size=pred_shape, mode="nearest")
    return resampled.squeeze(0).squeeze(0).cpu().numpy().astype(np.int32)


def _ncc_global(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = x - np.mean(x)
    y = y - np.mean(y)
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den <= 1e-18:
        return float("nan")
    return float(np.dot(x, y) / den)


def _center_of_mass_vox(arr: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(arr, dtype=np.float64), 0.0, None)
    total = float(np.sum(w))
    if total <= 1e-18:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    coords = np.indices(w.shape, dtype=np.float64)
    return np.array([float(np.sum(coords[i] * w) / total) for i in range(w.ndim)], dtype=np.float64)


def _com_dist_vox(a: np.ndarray, b: np.ndarray) -> float:
    com_a = _center_of_mass_vox(a)
    com_b = _center_of_mass_vox(b)
    if not np.all(np.isfinite(com_a)) or not np.all(np.isfinite(com_b)):
        return float("inf")
    return float(np.linalg.norm(com_a - com_b))


def plane_axes_for_slice_axis(axis: int) -> tuple[int, int]:
    if axis == 2:
        return (0, 1)
    if axis == 1:
        return (0, 2)
    if axis == 0:
        return (1, 2)
    raise ValueError(f"invalid axis={axis}; expected one of 0,1,2")


def _apply_orientation_transform(
    vol: np.ndarray,
    k_rot90: int,
    do_fliplr: bool,
    do_flipud: bool,
    plane_axes: tuple[int, int],
) -> np.ndarray:
    a, b = plane_axes
    out = np.rot90(vol, k=int(k_rot90), axes=(a, b))
    if do_flipud:
        out = np.flip(out, axis=a)
    if do_fliplr:
        out = np.flip(out, axis=b)
    return out


def _orientation_transform_label(
    k_rot90: int,
    do_fliplr: bool,
    do_flipud: bool,
    plane_axes: tuple[int, int],
) -> str:
    a, b = plane_axes
    parts = [f"rot90(k={int(k_rot90)}, axes=({a},{b}))"]
    if do_fliplr:
        parts.append(f"fliplr(axis={b})")
    if do_flipud:
        parts.append(f"flipud(axis={a})")
    return " + ".join(parts)


def _pick_best_pred_orientation(
    gt_roi: np.ndarray,
    pred_roi: np.ndarray,
    phantom_id: str,
    debug: bool,
    plane_axes: tuple[int, int],
) -> dict[str, Any]:
    flip_modes = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    results: list[dict[str, Any]] = []
    for k in (0, 1, 2, 3):
        for do_fliplr, do_flipud in flip_modes:
            cand = _apply_orientation_transform(
                pred_roi,
                k_rot90=k,
                do_fliplr=do_fliplr,
                do_flipud=do_flipud,
                plane_axes=plane_axes,
            )
            if cand.shape != gt_roi.shape:
                continue
            ncc = _ncc_global(gt_roi, cand)
            com_dist = _com_dist_vox(gt_roi, cand)
            item = {
                "k_rot90": int(k),
                "fliplr": bool(do_fliplr),
                "flipud": bool(do_flipud),
                "label": _orientation_transform_label(k, do_fliplr, do_flipud, plane_axes),
                "ncc": float(ncc),
                "com_dist_vox": float(com_dist),
            }
            results.append(item)
            if debug:
                _LOG.info(
                    "[orient-debug][%s] candidate=%s ncc=%.6f com_dist_vox=%.6f",
                    phantom_id,
                    item["label"],
                    item["ncc"],
                    item["com_dist_vox"],
                )
    if not results:
        raise RuntimeError(f"[orient-fix][{phantom_id}] no valid orientation candidates")

    def _key(item: dict[str, Any]) -> tuple[float, float]:
        ncc = item["ncc"]
        com_dist = item["com_dist_vox"]
        ncc_rank = ncc if np.isfinite(ncc) else -np.inf
        com_rank = -com_dist if np.isfinite(com_dist) else -np.inf
        return (ncc_rank, com_rank)

    best = max(results, key=_key)
    identity = next(
        (item for item in results if item["k_rot90"] == 0 and not item["fliplr"] and not item["flipud"]),
        None,
    )
    if identity is None:
        raise RuntimeError(f"[orient-fix][{phantom_id}] identity candidate missing")
    return {
        "best": best,
        "identity": identity,
        "results": results,
    }


def _safe_divide(numer: float, denom: float) -> float:
    denom_val = float(denom)
    if denom_val == 0.0 or not np.isfinite(denom_val):
        return float("nan")
    return float(numer / denom_val)


def extract_active_organ_ids(
    mask_gt: np.ndarray | None,
    gt: np.ndarray | None,
    eps: float = ACTIVE_ORGAN_EPS,
) -> np.ndarray:
    if mask_gt is None or gt is None:
        return np.array([], dtype=np.int32)
    if mask_gt.shape != gt.shape:
        # shapes should match; fallback to empty
        return np.array([], dtype=np.int32)
    valid = (gt > eps) & (mask_gt != 0)
    if not valid.any():
        return np.array([], dtype=np.int32)
    ids = np.unique(mask_gt[valid])
    ids = ids[ids != 0]
    return ids.astype(np.int32)


def compute_pred_grid_activity_metrics(
    pred: np.ndarray,
    mask: np.ndarray | None,
    active_ids: np.ndarray,
    voxel_volume: float,
) -> dict[str, float]:
    total = float(np.nansum(pred)) * voxel_volume
    result = {
        "total_sum": total,
        "in_body_sum": float("nan"),
        "out_body_sum": float("nan"),
        "in_gt_active_sum": float("nan"),
        "not_gt_active_sum": float("nan"),
        "in_body_frac": float("nan"),
        "out_body_frac": float("nan"),
        "in_gt_active_frac": float("nan"),
        "not_gt_active_frac": float("nan"),
    }
    if mask is None:
        return result
    mask_arr = mask.astype(np.int32)
    body_mask = mask_arr != 0
    in_body_sum = float(np.nansum(pred[body_mask])) * voxel_volume
    out_body_sum = float(np.nansum(pred[~body_mask])) * voxel_volume
    if active_ids.size > 0:
        active_mask = np.isin(mask_arr, active_ids) & body_mask
    else:
        active_mask = np.zeros_like(body_mask, dtype=bool)
    in_active_sum = float(np.nansum(pred[active_mask])) * voxel_volume
    not_active_sum = max(in_body_sum - in_active_sum, 0.0)
    result.update(
        {
            "in_body_sum": in_body_sum,
            "out_body_sum": out_body_sum,
            "in_gt_active_sum": in_active_sum,
            "not_gt_active_sum": not_active_sum,
            "in_body_frac": _safe_divide(in_body_sum, total),
            "out_body_frac": _safe_divide(out_body_sum, total),
            "in_gt_active_frac": _safe_divide(in_active_sum, total),
            "not_gt_active_frac": _safe_divide(not_active_sum, total),
        }
    )
    return result


def _is_in_body(mask: np.ndarray) -> np.ndarray:
    return mask != 0


def compute_voxel_metrics(gt_roi: np.ndarray, pred_roi: np.ndarray) -> tuple[float, float]:
    diff = gt_roi - pred_roi
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return mae, rmse


def masked_voxel_metrics(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    if mask is None or not mask.any():
        return float("nan"), float("nan"), 0
    gt_vals = gt[mask]
    pred_vals = pred[mask]
    diff = gt_vals - pred_vals
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return mae, rmse, int(mask.sum())



def load_mask(mask_path: Path, gt_shape: tuple[int, int, int]) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    mask = np.load(mask_path)
    if mask.shape != gt_shape:
        tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            resized = F.interpolate(tensor, size=gt_shape, mode="nearest")
        mask = resized.squeeze(0).squeeze(0).cpu().numpy()
    mask_int = np.rint(mask).astype(np.int32)
    return mask_int


def read_organ_names(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    mapping = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            name, val = line.split("=", 1)
            name = name.strip()
            val = val.strip()
            try:
                mapping[int(val)] = name
            except ValueError:
                continue
    return mapping


def compute_projection_metrics_from_arrays(
    pred_ap: np.ndarray,
    pred_pa: np.ndarray,
    gt_ap: np.ndarray,
    gt_pa: np.ndarray,
    domain: str,
) -> dict[str, float]:
    def mae(target, pred):
        return float(np.mean(np.abs(pred - target)))

    def poisson_dev(target, pred):
        eps = 1e-6
        y = np.clip(target, 0.0, None)
        mu = np.clip(pred, eps, None)
        ratio_term = np.where(y == 0, 0.0, y * np.log((y + eps) / mu))
        term = 2.0 * (ratio_term - (y - mu))
        return float(np.mean(term))

    mae_ap = mae(gt_ap, pred_ap)
    mae_pa = mae(gt_pa, pred_pa)
    results = {"proj_mae_counts": 0.5 * (mae_ap + mae_pa), "proj_domain": domain, "proj_status": ""}
    if domain == "counts":
        dev_ap = poisson_dev(gt_ap, pred_ap)
        dev_pa = poisson_dev(gt_pa, pred_pa)
        results["proj_poisson_dev_counts"] = 0.5 * (dev_ap + dev_pa)
        results["proj_status"] = "loaded_npy"
    else:
        results["proj_poisson_dev_counts"] = float("nan")
        results["proj_status"] = "normalized_inputs"
    return results


def compute_projection_metrics_with_fallback(
    args,
    run_dir: Path,
    out_dir: Path,
    phantom_id: str,
    gt_ap: np.ndarray,
    gt_pa: np.ndarray,
    domain: str,
) -> dict[str, float]:
    if args.checkpoint:
        _LOG.warning("checkpoint render not implemented; projections require latent export")
        return {
            "proj_mae_counts": float("nan"),
            "proj_poisson_dev_counts": float("nan"),
            "proj_status": "render_missing",
            "proj_domain": domain,
        }
    candidates = find_projection_candidates(args, run_dir, out_dir, phantom_id)
    if candidates and "ap" in candidates and "pa" in candidates:
        ap, pa = load_projection_arrays(candidates, gt_ap.shape)
        metrics = compute_projection_metrics_from_arrays(ap, pa, gt_ap, gt_pa, domain)
        metrics["proj_status"] = candidates.get("status", metrics["proj_status"])
        metrics["proj_domain"] = domain
        metrics["pred_ap"] = ap
        metrics["pred_pa"] = pa
        _LOG.info("loaded pred projections from %s/%s for %s", candidates["ap"], candidates["pa"], phantom_id)
        return metrics
    return {
        "proj_mae_counts": float("nan"),
        "proj_poisson_dev_counts": float("nan"),
        "proj_status": "missing",
        "proj_domain": domain,
    }


def _load_meta_simple(act_path: Path) -> dict:
    meta_path = act_path.parent / "meta_simple.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception as exc:
        _LOG.warning("failed to load meta_simple.json (%s): %s", meta_path, exc)
        return {}


def _get_physics_projector_imports() -> tuple[type, Callable[..., Any]]:
    global _PHYSICS_PROJECTOR_IMPORTS
    if _PHYSICS_PROJECTOR_IMPORTS is not None:
        return _PHYSICS_PROJECTOR_IMPORTS
    try:
        from physics.projector import ProjectorConfig, project_activity
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Physics projector dependencies missing (e.g. Data_Processing). "
            "Either add the dependency to PYTHONPATH / install it, or run with "
            "--proj-forward-model train."
        ) from e
    _PHYSICS_PROJECTOR_IMPORTS = (ProjectorConfig, project_activity)
    return _PHYSICS_PROJECTOR_IMPORTS


def _get_train_emission_imports() -> dict[str, Any]:
    global _TRAIN_EMISSION_IMPORTS
    if _TRAIN_EMISSION_IMPORTS is not None:
        return _TRAIN_EMISSION_IMPORTS
    try:
        import train_emission as te
    except ModuleNotFoundError:
        te_path = _REPO_ROOT / "train_emission.py"
        spec = importlib.util.spec_from_file_location("train_emission", str(te_path))
        te = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(te)  # type: ignore[attr-defined]
    _TRAIN_EMISSION_IMPORTS = {
        "te": te,
        "compute_projection_metrics": te.compute_projection_metrics,
        "compute_proj_scale": te.compute_proj_scale,
        "build_hybrid_latent_from_batch": te.build_hybrid_latent_from_batch,
        "build_encoder_input": te.build_encoder_input if hasattr(te, "build_encoder_input") else None,
        "compute_poisson_rate": te.compute_poisson_rate,
        "apply_poisson_rate_floor": te.apply_poisson_rate_floor if hasattr(te, "apply_poisson_rate_floor") else None,
        "get_data": te.get_data if hasattr(te, "get_data") else None,
        "build_models": te.build_models if hasattr(te, "build_models") else None,
    }
    return _TRAIN_EMISSION_IMPORTS


def _build_projector_config(
    meta: dict,
    args,
    spacing_gt_cm: tuple[float, float, float],
    default_kernel: Path,
    projector_config_cls: type,
) -> object | None:
    kernel_path = meta.get("kernel_mat")
    if not kernel_path:
        if not default_kernel.exists():
            _LOG.warning("no LEAP kernel path found in metadata and default missing")
            return None
        kernel_path = default_kernel
    kernel_path = Path(kernel_path)
    if not kernel_path.exists():
        _LOG.warning("kernel file missing: %s", kernel_path)
        return None
    kernel_var = meta.get("kernel_var", "kernel_mat")
    step_len = float(meta.get("step_len", spacing_gt_cm[2]))
    psf_sigma = args.psf_sigma if args.psf_sigma is not None else float(meta.get("psf_sigma", 2.0))
    z0_slices = args.z0_slices if args.z0_slices is not None else int(meta.get("z0_slices", 29))
    sensitivity = (
        float(meta.get("sensitivity_cps_per_mbq", 65.0))
        if args.proj_sensitivity_cps_per_mbq is None
        else float(args.proj_sensitivity_cps_per_mbq)
    )
    acq_time = (
        float(meta.get("acq_time_s", 600.0))
        if args.proj_acq_time_s is None
        else float(args.proj_acq_time_s)
    )
    mu_unit_in = meta.get("mu_unit_in", "per_mm")
    mu_unit_out = meta.get("mu_unit_out", "per_cm")
    return projector_config_cls(
        kernel_mat=kernel_path,
        kernel_var=kernel_var,
        psf_sigma=psf_sigma,
        z0_slices=z0_slices,
        step_len=step_len,
        mu_unit_in=mu_unit_in,
        mu_unit_out=mu_unit_out,
        sensitivity_cps_per_mbq=sensitivity,
        acq_time_s=acq_time,
    )


def _load_manifest_projection(entry: dict, keys: list[str]) -> np.ndarray | None:
    path = resolve_manifest_path(entry, keys)
    if path is None or not path.exists():
        return None
    return load_array(path)


def _load_gt_projections(
    entry: dict,
    counts_keys_ap: list[str],
    counts_keys_pa: list[str],
    norm_keys_ap: list[str],
    norm_keys_pa: list[str],
    phantom_id: str,
    preferred_domain: str,
) -> tuple[str, np.ndarray, np.ndarray]:
    preferred = preferred_domain or "counts"
    if preferred == "normalized":
        ap_norm = _load_manifest_projection(entry, norm_keys_ap)
        pa_norm = _load_manifest_projection(entry, norm_keys_pa)
        if ap_norm is None or pa_norm is None:
            raise FileNotFoundError(f"missing normalized projections for {phantom_id}")
        return "normalized", ap_norm, pa_norm
    ap_counts = _load_manifest_projection(entry, counts_keys_ap)
    pa_counts = _load_manifest_projection(entry, counts_keys_pa)
    if ap_counts is not None and pa_counts is not None:
        return "counts", ap_counts, pa_counts
    ap_norm = _load_manifest_projection(entry, norm_keys_ap)
    pa_norm = _load_manifest_projection(entry, norm_keys_pa)
    if ap_norm is not None and pa_norm is not None:
        _LOG.warning("counts projections missing for %s, using normalized targets", phantom_id)
        return "normalized", ap_norm, pa_norm
    raise FileNotFoundError(f"missing projection paths for {phantom_id}")


def _load_noisy_projection_targets(
    run_dir: Path,
    phantom_id: str,
    *,
    expected_shape: tuple[int, int] | None,
    strict: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    ap_path = run_dir / "test_slices" / phantom_id / "noisy_ap_counts.npy"
    pa_path = run_dir / "test_slices" / phantom_id / "noisy_pa_counts.npy"
    if not ap_path.exists() or not pa_path.exists():
        if strict:
            missing = []
            if not ap_path.exists():
                missing.append(str(ap_path))
            if not pa_path.exists():
                missing.append(str(pa_path))
            raise FileNotFoundError(
                f"phantom={phantom_id}: --proj-metrics-target noisy_counts requires noisy projection files; "
                f"missing={missing}"
            )
        return None, None
    ap = load_array(ap_path)
    pa = load_array(pa_path)
    if ap.ndim != 2 or pa.ndim != 2:
        raise ValueError(
            f"phantom={phantom_id}: noisy targets must be 2D, got ap.shape={ap.shape}, pa.shape={pa.shape}"
        )
    if expected_shape is not None:
        if tuple(ap.shape) != tuple(expected_shape) or tuple(pa.shape) != tuple(expected_shape):
            raise ValueError(
                f"phantom={phantom_id}: noisy target shape mismatch, expected={expected_shape}, "
                f"ap.shape={ap.shape}, pa.shape={pa.shape}"
            )
    return np.asarray(ap, dtype=np.float32), np.asarray(pa, dtype=np.float32)


def _array_stats(arr: np.ndarray) -> dict[str, float]:
    flat = np.asarray(arr, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan"), "sum": float("nan")}
    return {
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
        "sum": float(np.sum(finite)),
    }


def _format_stat_snippet(prefix: str, stats: dict[str, float]) -> str:
    return (
        f"{prefix} min={stats['min']:.3e} mean={stats['mean']:.3e} "
        f"max={stats['max']:.3e} sum={stats['sum']:.3e}"
    )


def _compute_projection_scale_factors(
    pred_ap: np.ndarray,
    pred_pa: np.ndarray,
    gt_ap: np.ndarray,
    gt_pa: np.ndarray,
) -> tuple[float, float]:
    sum_pred = float(np.nansum(pred_ap)) + float(np.nansum(pred_pa))
    sum_gt = float(np.nansum(gt_ap)) + float(np.nansum(gt_pa))
    alpha_sum = float("nan")
    if sum_pred > 0:
        alpha_sum = sum_gt / sum_pred
    denom = float(np.nansum(pred_ap * pred_ap)) + float(np.nansum(pred_pa * pred_pa))
    numer = float(np.nansum(pred_ap * gt_ap)) + float(np.nansum(pred_pa * gt_pa))
    alpha_ls = numer / denom if denom > 0 else float("nan")
    return alpha_sum, alpha_ls


def _compute_joint_projection_ls_scale(
    pred_ap: np.ndarray,
    pred_pa: np.ndarray,
    gt_ap: np.ndarray,
    gt_pa: np.ndarray,
) -> float:
    # Joint AP+PA least-squares scalar:
    # s = ( <gt_ap,pred_ap> + <gt_pa,pred_pa> ) / ( <pred_ap,pred_ap> + <pred_pa,pred_pa> )
    denom = float(np.nansum(pred_ap * pred_ap)) + float(np.nansum(pred_pa * pred_pa))
    numer = float(np.nansum(gt_ap * pred_ap)) + float(np.nansum(gt_pa * pred_pa))
    return numer / denom if denom > 0 else float("nan")


def _ls_alpha_global(
    pred_ap: np.ndarray,
    pred_pa: np.ndarray,
    gt_ap: np.ndarray,
    gt_pa: np.ndarray,
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    if pred_ap.shape != gt_ap.shape:
        raise RuntimeError(f"LS calibration shape mismatch AP: pred={pred_ap.shape} gt={gt_ap.shape}")
    if pred_pa.shape != gt_pa.shape:
        raise RuntimeError(f"LS calibration shape mismatch PA: pred={pred_pa.shape} gt={gt_pa.shape}")
    p_ap = np.asarray(pred_ap, dtype=np.float64).ravel()
    p_pa = np.asarray(pred_pa, dtype=np.float64).ravel()
    g_ap = np.asarray(gt_ap, dtype=np.float64).ravel()
    g_pa = np.asarray(gt_pa, dtype=np.float64).ravel()
    num = float(np.dot(p_ap, g_ap) + np.dot(p_pa, g_pa))
    den_no_eps = float(np.dot(p_ap, p_ap) + np.dot(p_pa, p_pa))
    den = den_no_eps + float(eps)
    if den_no_eps <= float(eps):
        raise RuntimeError(
            f"LS calibration denominator too small: den_no_eps={den_no_eps:.6e} eps={float(eps):.6e}"
        )
    alpha = num / den
    if not np.isfinite(alpha):
        raise RuntimeError(
            f"LS calibration alpha not finite: alpha={alpha} num={num:.6e} den={den:.6e}"
        )
    return float(alpha), float(num), float(den)


def _log_projection_stats(
    phantom_id: str,
    pred_ap: np.ndarray,
    pred_pa: np.ndarray,
    gt_ap: np.ndarray,
    gt_pa: np.ndarray,
    alpha_sum: float,
    alpha_ls: float,
) -> None:
    pred_ap_stats = _array_stats(pred_ap)
    pred_pa_stats = _array_stats(pred_pa)
    gt_ap_stats = _array_stats(gt_ap)
    gt_pa_stats = _array_stats(gt_pa)
    _LOG.info(
        "[proj-stats][%s] pred_ap shape=%s %s | pred_pa shape=%s %s | gt_ap shape=%s %s | gt_pa shape=%s %s | alpha_sum=%.3e alpha_ls=%.3e",
        phantom_id,
        tuple(pred_ap.shape),
        _format_stat_snippet("pred_ap", pred_ap_stats),
        tuple(pred_pa.shape),
        _format_stat_snippet("pred_pa", pred_pa_stats),
        tuple(gt_ap.shape),
        _format_stat_snippet("gt_ap", gt_ap_stats),
        tuple(gt_pa.shape),
        _format_stat_snippet("gt_pa", gt_pa_stats),
        alpha_sum,
        alpha_ls,
    )


def _finite_percentile(arr: np.ndarray, q: float, default: float = 0.0) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return default
    return float(np.percentile(finite, q))


def _finite_abs_max(arr: np.ndarray, default: float = 0.0) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return default
    return float(np.max(np.abs(finite)))


def _log_transform_clip(arr: np.ndarray) -> tuple[np.ndarray, float]:
    arr_log = np.log1p(np.clip(np.asarray(arr, dtype=np.float32), 0.0, None))
    clip = _finite_percentile(arr_log, 99.5, default=1.0)
    if not np.isfinite(clip) or clip <= 0.0:
        clip = 1.0
    return np.clip(arr_log, 0.0, clip), clip


def _save_gray_image(path: Path, arr: np.ndarray, title: str, label: str) -> None:
    arr_clean = np.clip(np.asarray(arr, dtype=np.float32), 0.0, None)
    vmax = _finite_percentile(arr_clean, 99.5, default=1.0)
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = max(_finite_percentile(arr_clean, 99.5, default=1.0), 1.0)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr_clean, cmap="inferno", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_view_logpct(path: Path, gt: np.ndarray, pred: np.ndarray, view_label: str) -> None:
    gt_log, gt_clip = _log_transform_clip(gt)
    pred_log, pred_clip = _log_transform_clip(pred)
    diff = pred_log - gt_log
    log_clip = max(gt_clip, pred_clip)
    diff_clip = max(_finite_abs_max(diff, default=1e-3), 1e-3)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im = axes[0].imshow(gt_log, cmap="inferno", vmin=0.0, vmax=log_clip)
    axes[0].set_title(f"{view_label} GT log1p")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    im = axes[1].imshow(pred_log, cmap="inferno", vmin=0.0, vmax=log_clip)
    axes[1].set_title(f"{view_label} Pred log1p")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-diff_clip, vmax=diff_clip)
    axes[2].set_title(f"{view_label} Diff")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_projection_pngs(
    plot_dir: Path,
    pred_ap: np.ndarray,
    pred_pa: np.ndarray,
    gt_ap: np.ndarray,
    gt_pa: np.ndarray,
) -> None:
    if not _MATPLOTLIB_AVAILABLE:
        _LOG.warning("matplotlib unavailable; skipping PNG export")
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    _save_gray_image(plot_dir / "pred_ap.png", pred_ap, "Pred AP", "counts")
    _save_gray_image(plot_dir / "pred_pa.png", pred_pa, "Pred PA", "counts")
    _save_gray_image(plot_dir / "gt_ap.png", gt_ap, "GT AP", "counts")
    _save_gray_image(plot_dir / "gt_pa.png", gt_pa, "GT PA", "counts")
    _save_view_logpct(plot_dir / "ap_gt_vs_pred_logpct.png", gt_ap, pred_ap, "AP")
    _save_view_logpct(plot_dir / "pa_gt_vs_pred_logpct.png", gt_pa, pred_pa, "PA")


def _slice_mass(volume: np.ndarray, axis: int) -> np.ndarray:
    sum_axes = tuple(i for i in range(volume.ndim) if i != axis)
    return np.sum(volume, axis=sum_axes)


def _extract_slice(volume: np.ndarray, axis: int, idx: int) -> np.ndarray:
    return np.take(volume, idx, axis=axis)


def _dedupe_keep_order(values: list[int]) -> list[int]:
    seen = set()
    out: list[int] = []
    for val in values:
        if val not in seen:
            out.append(val)
            seen.add(val)
    return out


def _cdf_crossing_index(candidate_indices: np.ndarray, mass_gt: np.ndarray, frac: float) -> int:
    candidate_masses = np.asarray(mass_gt[candidate_indices], dtype=np.float64)
    cumulative = np.cumsum(candidate_masses)
    target = float(frac) * float(cumulative[-1])
    local = int(np.searchsorted(cumulative, target, side="left"))
    local = min(local, candidate_indices.size - 1)
    return int(candidate_indices[local])


def _select_act_compare_slices(
    mass_gt: np.ndarray,
    mass_pred: np.ndarray,
    stride: int,
    phantom_id: str,
) -> list[int]:
    if stride < 1:
        raise RuntimeError(
            f"[act-compare-5][{phantom_id}] invalid stride={stride}; --act-compare-stride must be >= 1"
        )
    n_slices = int(mass_gt.shape[0])
    candidate_indices = np.arange(0, n_slices, stride, dtype=int)
    if candidate_indices.size < 5:
        raise RuntimeError(
            f"[act-compare-5][{phantom_id}] only {candidate_indices.size} candidate slices for "
            f"axis-length={n_slices} stride={stride}; need at least 5"
        )

    idx_max_gt = int(candidate_indices[np.argmax(mass_gt[candidate_indices])])
    idx_max_pred = int(candidate_indices[np.argmax(mass_pred[candidate_indices])])
    idx_q25 = _cdf_crossing_index(candidate_indices, mass_gt, 0.25)
    idx_q50 = _cdf_crossing_index(candidate_indices, mass_gt, 0.50)
    idx_q75 = _cdf_crossing_index(candidate_indices, mass_gt, 0.75)

    selected = _dedupe_keep_order([idx_max_gt, idx_max_pred, idx_q25, idx_q50, idx_q75])
    if len(selected) < 5:
        ranked = candidate_indices[np.argsort(mass_gt[candidate_indices])[::-1]]
        for idx in ranked:
            ii = int(idx)
            if ii not in selected:
                selected.append(ii)
            if len(selected) == 5:
                break
    if len(selected) != 5:
        raise RuntimeError(f"[act-compare-5][{phantom_id}] failed to select 5 unique slices")
    return selected


def save_activity_compare_5slices(
    plot_dir: Path,
    phantom_id: str,
    gt: np.ndarray,
    pred_gtgrid: np.ndarray,
    axis: int,
    stride: int,
    outname_template: str,
) -> None:
    if not _MATPLOTLIB_AVAILABLE:
        _LOG.warning("[act-compare-5][%s] matplotlib unavailable; skipping", phantom_id)
        return
    if gt.shape != pred_gtgrid.shape:
        raise RuntimeError(
            f"[act-compare-5][{phantom_id}] shape mismatch gt={gt.shape} pred_gtgrid={pred_gtgrid.shape}"
        )

    mass_gt = _slice_mass(gt, axis=axis)
    mass_pred = _slice_mass(pred_gtgrid, axis=axis)
    total_gt_mass = float(np.sum(mass_gt))
    if total_gt_mass == 0.0:
        raise RuntimeError(f"[act-compare-5][{phantom_id}] total GT mass is 0")

    selected = _select_act_compare_slices(mass_gt, mass_pred, stride, phantom_id)
    _LOG.info(
        "[act-compare-5][%s] selected_slices=%s axis=%d stride=%d",
        phantom_id,
        selected,
        axis,
        stride,
    )
    for idx in selected:
        _LOG.info(
            "[act-compare-5][%s] slice=%d mass_gt=%.6g mass_pred=%.6g",
            phantom_id,
            idx,
            float(mass_gt[idx]),
            float(mass_pred[idx]),
        )

    # Display-only clamp for robust color scaling.
    gt_plot = np.clip(np.asarray(gt, dtype=np.float32), 0.0, None)
    pred_plot = np.clip(np.asarray(pred_gtgrid, dtype=np.float32), 0.0, None)
    vmax = 0.0
    for idx in selected:
        vmax = max(
            vmax,
            float(np.max(_extract_slice(gt_plot, axis, idx))),
            float(np.max(_extract_slice(pred_plot, axis, idx))),
        )
    vmin = 0.0
    _LOG.info("[act-compare-5][%s] color scale vmin=%.6g vmax=%.6g", phantom_id, vmin, vmax)

    try:
        outname = outname_template.format(axis=axis)
    except Exception as exc:
        raise RuntimeError(
            f"[act-compare-5][{phantom_id}] invalid --act-compare-outname template: {outname_template!r}"
        ) from exc
    out_path = plot_dir / outname
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8.5, 16), dpi=200)
    im = None
    for row, idx in enumerate(selected):
        gt_slice = _extract_slice(gt_plot, axis, idx)
        pred_slice = _extract_slice(pred_plot, axis, idx)
        ax_l = axes[row, 0]
        ax_r = axes[row, 1]
        im = ax_l.imshow(gt_slice, cmap="viridis", vmin=vmin, vmax=vmax)
        ax_r.imshow(pred_slice, cmap="viridis", vmin=vmin, vmax=vmax)
        ax_l.set_title(f"GT | z = {idx}")
        ax_r.set_title(f"Pred | z = {idx}")
        ax_l.set_xticks([])
        ax_l.set_yticks([])
        ax_r.set_xticks([])
        ax_r.set_yticks([])
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Activity")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    _LOG.info("[act-compare-5][%s] saved %s", phantom_id, out_path)



def _default_organ_aggregate_metrics() -> dict[str, Union[float, int]]:
    return {
        "organ_rel_error_total_activity_active_mean": float("nan"),
        "active_organ_fraction_mae": float("nan"),
        "label_averaged_organ_fraction_mae": float("nan"),
        "active_organ_fraction_n": 0,
        "organ_active_n": 0,
        "organ_inactive_n": 0,
        "inactive_organs_pred_sum": 0.0,
        "inactive_organs_pred_frac_of_pred": float("nan"),
        "inactive_organs_pred_frac_of_gt": float("nan"),
    }


def compute_organ_statistics(
    mask_roi: np.ndarray,
    gt_roi: np.ndarray,
    pred_roi: np.ndarray,
    spacing_gt: tuple[float, float, float],
    organ_map: dict[int, str],
) -> tuple[dict[int, dict], dict[str, Union[float, int]], float]:
    default_aggregate = _default_organ_aggregate_metrics()
    if mask_roi is None:
        return {}, default_aggregate, float("nan")
    mask_flat = mask_roi.reshape(-1).astype(np.int32)
    if mask_flat.size == 0:
        return {}, default_aggregate, float("nan")
    gt_flat = gt_roi.reshape(-1).astype(np.float32)
    pred_flat = pred_roi.reshape(-1).astype(np.float32)
    unique_ids, inverse = np.unique(mask_flat, return_inverse=True)
    if unique_ids.size == 0:
        return {}, default_aggregate, float("nan")
    counts = np.bincount(inverse)
    if counts.size == 0:
        return {}, default_aggregate, float("nan")
    sums_gt = np.bincount(inverse, weights=gt_flat, minlength=counts.size)
    sums_pred = np.bincount(inverse, weights=pred_flat, minlength=counts.size)
    vol_voxel = spacing_gt[0] * spacing_gt[1] * spacing_gt[2]
    gt_sums_activity = sums_gt * vol_voxel
    pred_sums_activity = sums_pred * vol_voxel
    means_gt = np.divide(sums_gt, counts, out=np.zeros_like(sums_gt), where=counts > 0)
    means_pred = np.divide(sums_pred, counts, out=np.zeros_like(sums_pred), where=counts > 0)

    stats = {}
    active_rel_errors = []
    inactive_pred_sum = 0.0
    sum_pred_all = 0.0
    sum_gt_all = 0.0
    active_count = 0
    inactive_count = 0
    for idx, organ_id in enumerate(unique_ids):
        if organ_id == 0 or counts[idx] == 0:
            continue
        gt_sum = float(gt_sums_activity[idx])
        pred_sum = float(pred_sums_activity[idx])
        sum_pred_all += pred_sum
        sum_gt_all += gt_sum
        is_active = gt_sum > ORGAN_ACTIVE_TOLERANCE
        if is_active:
            rel_error = abs(pred_sum - gt_sum) / gt_sum
            active_rel_errors.append(rel_error)
            active_count += 1
        else:
            rel_error = float("nan")
            inactive_pred_sum += pred_sum
            inactive_count += 1
        stats[int(organ_id)] = {
            "organ_name": organ_map.get(int(organ_id), f"organ_{organ_id}"),
            "gt_sum_activity": gt_sum,
            "pred_sum_activity": pred_sum,
            "gt_mean": float(means_gt[idx]),
            "pred_mean": float(means_pred[idx]),
            "rel_error_total_activity": rel_error,
            "is_active": is_active,
        }
    organ_rel_error_mean = float(np.nanmean(active_rel_errors)) if active_rel_errors else float("nan")
    inactive_pred_frac_of_pred = float("nan")
    inactive_pred_frac_of_gt = float("nan")
    if sum_pred_all > 0:
        inactive_pred_frac_of_pred = inactive_pred_sum / sum_pred_all
    if sum_gt_all > 0:
        inactive_pred_frac_of_gt = inactive_pred_sum / sum_gt_all
    aggregate_metrics = {
        "organ_rel_error_total_activity_active_mean": organ_rel_error_mean,
        "organ_active_n": active_count,
        "organ_inactive_n": inactive_count,
        "inactive_organs_pred_sum": inactive_pred_sum,
        "inactive_organs_pred_frac_of_pred": inactive_pred_frac_of_pred,
        "inactive_organs_pred_frac_of_gt": inactive_pred_frac_of_gt,
    }
    label_averaged_fraction_mae = compute_organ_fraction_error(stats)
    active_fraction_mae, active_fraction_n = compute_active_organ_fraction_mae(
        mask_roi=mask_roi,
        gt_roi=gt_roi,
        pred_roi=pred_roi,
        voxel_volume=vol_voxel,
    )
    aggregate_metrics["label_averaged_organ_fraction_mae"] = label_averaged_fraction_mae
    aggregate_metrics["active_organ_fraction_mae"] = active_fraction_mae
    aggregate_metrics["active_organ_fraction_n"] = active_fraction_n
    return stats, aggregate_metrics, active_fraction_mae


def compute_outside_mask_leakage(
    pred_roi: np.ndarray,
    mask_roi: np.ndarray,
    voxel_volume: float,
    total_pred_vol: float,
    phantom_id: str,
) -> tuple[float, float]:
    if mask_roi is None:
        return float("nan"), float("nan")
    mask_bool = _is_in_body(mask_roi)
    outside_mask = ~mask_bool
    pred_total_raw = float(pred_roi.sum())
    pred_outside_raw = float(pred_roi[outside_mask].sum())
    pred_outside_sum = pred_outside_raw * voxel_volume
    pred_outside_frac = float("nan")
    if total_pred_vol != 0:
        pred_outside_frac = pred_outside_sum / total_pred_vol
        if pred_outside_frac < 0.0 or pred_outside_frac > 1.0:
            _LOG.warning(
                "outside-mask fraction %.6g not in [0,1]; clamping for %s",
                pred_outside_frac,
                phantom_id,
            )
            pred_outside_frac = float(np.clip(pred_outside_frac, 0.0, 1.0))
    if not np.isnan(pred_outside_sum) and not np.isnan(total_pred_vol):
        if pred_outside_sum > total_pred_vol * (1.0 + 1e-8):
            _LOG.warning(
                "outside-mask sum %.6g exceeds total pred sum %.6g for %s",
                pred_outside_sum,
                total_pred_vol,
                phantom_id,
            )
            pred_outside_sum = total_pred_vol
    if total_pred_vol != 0 and pred_total_raw == 0.0:
        _LOG.debug(
            "mask leakage raw total %.6g equals zero before scaling for %s",
            pred_total_raw,
            phantom_id,
        )
    return pred_outside_sum, pred_outside_frac


def compute_organ_fraction_error(stats: dict[int, dict]) -> float:
    if not stats:
        return float("nan")
    gt_sums = np.array([row["gt_sum_activity"] for row in stats.values()], dtype=np.float64)
    pred_sums = np.array([row["pred_sum_activity"] for row in stats.values()], dtype=np.float64)
    total_gt = float(gt_sums.sum())
    total_pred = float(pred_sums.sum())
    if total_gt <= 1e-12 or total_pred <= 1e-12:
        return float("nan")
    gt_frac = gt_sums / total_gt
    pred_frac = pred_sums / total_pred
    return float(np.mean(np.abs(gt_frac - pred_frac)))


def compute_active_organ_fraction_mae(
    mask_roi: np.ndarray,
    gt_roi: np.ndarray,
    pred_roi: np.ndarray,
    voxel_volume: float,
) -> tuple[float, int]:
    """MAE of organ fractions using the same active-organ selection as save_active_organ_plots."""
    active_ids = extract_active_organ_ids(mask_roi, gt_roi)
    if active_ids.size == 0:
        return float("nan"), 0
    sorted_ids = np.sort(np.asarray(active_ids, dtype=np.int32))
    if np.any(sorted_ids == 1384):
        sorted_ids = sorted_ids[sorted_ids != 1384]
    organ_ids = np.array([int(oid) for oid in sorted_ids if int(oid) in ORGAN_LABEL_MAP], dtype=np.int32)
    if organ_ids.size == 0:
        return float("nan"), 0
    gt_sums = []
    pred_sums = []
    for organ_id in organ_ids:
        organ_mask = mask_roi == organ_id
        gt_sums.append(float(np.nansum(gt_roi[organ_mask])) * voxel_volume)
        pred_sums.append(float(np.nansum(pred_roi[organ_mask])) * voxel_volume)
    gt_arr = np.asarray(gt_sums, dtype=np.float64)
    pred_arr = np.asarray(pred_sums, dtype=np.float64)
    total_gt = float(np.sum(gt_arr))
    total_pred = float(np.sum(pred_arr))
    if total_gt <= 1e-12 or total_pred <= 1e-12:
        return float("nan"), int(organ_ids.size)
    gt_frac = gt_arr / total_gt
    pred_frac = pred_arr / total_pred
    mae = float(np.mean(np.abs(gt_frac - pred_frac)))
    return mae, int(organ_ids.size)

def render_projections_stub():
    # Projections currently not rendered (no checkpoint or latent).
    return {
        "proj_mae_counts": float("nan"),
        "proj_poisson_dev_counts": float("nan"),
        "proj_status": "missing",
    }


class TrainForwardProjector:
    def __init__(self, args, run_dir: Path, manifest: dict[str, dict], device: torch.device):
        self.args = args
        self.run_dir = run_dir
        self.manifest = manifest
        self.device = device
        self.repo_root = _REPO_ROOT
        tr = _get_train_emission_imports()
        self.train_mod = tr["te"]
        self.config_mod = importlib.import_module("graf.config")
        self.cli_tokens = self._extract_train_cli(run_dir / "command.sh")
        self.train_args = self._parse_train_args(self.cli_tokens)
        self.config = self._load_train_config(self.train_args.config)
        self.dataset, hwfr, _ = self.config_mod.get_data(self.config)
        self.config["data"]["hwfr"] = hwfr
        self.generator = self._build_models_flexible(self.config, device)
        self.generator.train()
        self.generator.use_test_kwargs = False
        self.generator.set_fixed_ap_pa(radius=hwfr[3])
        self.z_dim = int(self.config["z_dist"]["dim"])
        self.z_base = torch.zeros(1, self.z_dim, device=device)
        self.encoder = None
        self.z_fuser = None
        self.gain_head = None
        self.gain_param = None
        self._init_hybrid_modules()
        self.ckpt_path = self._select_checkpoint()
        self._load_checkpoint(self.ckpt_path)
        self.generator.eval()
        if self.encoder is not None:
            self.encoder.eval()
        if self.z_fuser is not None:
            self.z_fuser.eval()
        if self.gain_head is not None:
            self.gain_head.eval()
        self.pid_to_index = self._build_pid_to_test_index()

    def _extract_train_cli(self, command_path: Path) -> list[str]:
        tokens = shlex.split(command_path.read_text().strip())
        idx = None
        for i, tok in enumerate(tokens):
            if tok.endswith("train_emission.py"):
                idx = i
                break
        if idx is None:
            raise RuntimeError(f"train_emission.py missing in {command_path}")
        return tokens[idx + 1 :]

    def _parse_train_args(self, cli_tokens: list[str]) -> argparse.Namespace:
        old_argv = sys.argv[:]
        try:
            sys.argv = ["train_emission.py"] + cli_tokens
            return self.train_mod.parse_args()
        finally:
            sys.argv = old_argv

    def _load_train_config(self, config_path: str) -> dict:
        with Path(config_path).open("r") as f:
            cfg = yaml.safe_load(f)
        data_cfg = cfg.setdefault("data", {})
        radius_xyz = data_cfg.get("radius_xyz_cm")
        if bool(data_cfg.get("auto_near_far_from_radius", True)) and radius_xyz is not None:
            rz = float(radius_xyz[2])
            data_cfg["near"] = 0.0
            data_cfg["far"] = 2.0 * rz
        return cfg

    def _build_models_flexible(self, config: dict, device: torch.device):
        from graf.generator import Generator
        from graf.transforms import FlexGridRaySampler
        from nerf.run_nerf_mod import create_nerf

        cfg_nerf = argparse.Namespace(**config["nerf"])
        cfg_nerf.chunk = min(config["training"]["chunk"], 1024 * config["training"]["batch_size"])
        cfg_nerf.netchunk = config["training"]["netchunk"]
        cfg_nerf.white_bkgd = config["data"]["white_bkgd"]
        cfg_nerf.feat_dim = config["z_dist"]["dim"]
        cfg_nerf.feat_dim_appearance = config["z_dist"]["dim_appearance"]
        cfg_nerf.emission = True
        if not hasattr(cfg_nerf, "use_attenuation"):
            cfg_nerf.use_attenuation = False
        if not hasattr(cfg_nerf, "attenuation_debug"):
            cfg_nerf.attenuation_debug = False
        if not hasattr(cfg_nerf, "atten_scale"):
            cfg_nerf.atten_scale = 25.0
        if bool(getattr(self.args, "force_use_attenuation", False)):
            cfg_nerf.use_attenuation = True

        render_train, render_test, params, named_params = create_nerf(cfg_nerf)
        render_train["emission"] = True
        render_test["emission"] = True
        render_train["use_attenuation"] = bool(getattr(cfg_nerf, "use_attenuation", False))
        render_test["use_attenuation"] = bool(getattr(cfg_nerf, "use_attenuation", False))
        render_train["attenuation_debug"] = bool(getattr(cfg_nerf, "attenuation_debug", False))
        render_test["attenuation_debug"] = bool(getattr(cfg_nerf, "attenuation_debug", False))
        atten_scale = float(getattr(cfg_nerf, "atten_scale", 25.0))
        render_train["atten_scale"] = atten_scale
        render_test["atten_scale"] = atten_scale
        bds = {"near": config["data"]["near"], "far": config["data"]["far"]}
        render_train.update(bds)
        render_test.update(bds)

        ray_sampler = FlexGridRaySampler(
            N_samples=config["ray_sampler"]["N_samples"],
            min_scale=config["ray_sampler"]["min_scale"],
            max_scale=config["ray_sampler"]["max_scale"],
            scale_anneal=config["ray_sampler"]["scale_anneal"],
            orthographic=config["data"]["orthographic"],
        )
        H, W, f, r = config["data"]["hwfr"]
        generator = Generator(
            H,
            W,
            f,
            r,
            ray_sampler=ray_sampler,
            render_kwargs_train=render_train,
            render_kwargs_test=render_test,
            parameters=params,
            named_parameters=named_params,
            chunk=cfg_nerf.chunk,
            range_u=(float(config["data"]["umin"]), float(config["data"]["umax"])),
            range_v=(float(config["data"]["vmin"]), float(config["data"]["vmax"])),
            orthographic=config["data"]["orthographic"],
            radius_xyz_cm=tuple(config["data"]["radius_xyz_cm"]) if config["data"].get("radius_xyz_cm") is not None else None,
        )
        return generator.to(device)

    def _init_hybrid_modules(self) -> None:
        if not bool(self.train_args.hybrid):
            return
        enc_in_ch = 2 + (1 if self.train_args.encoder_use_ct else 0)
        self.encoder = self.train_mod.ProjectionEncoder(in_ch=enc_in_ch, z_dim=self.z_dim, base_ch=32).to(self.device)
        self.z_fuser = nn.Sequential(nn.Linear(self.z_dim, self.z_dim), nn.LayerNorm(self.z_dim)).to(self.device)
        if self.train_args.proj_target_source == "counts":
            if self.train_args.proj_gain_source == "z_enc":
                self.gain_head = nn.Linear(self.z_dim, 1).to(self.device)
            elif self.train_args.proj_gain_source == "scalar":
                self.gain_param = nn.Parameter(torch.zeros(1, device=self.device))

    def _select_checkpoint(self) -> Path:
        if self.args.checkpoint:
            cp = Path(self.args.checkpoint)
            return cp if cp.is_absolute() else (self.run_dir / cp)
        candidates = sorted((self.run_dir / "checkpoints").glob("checkpoint_step*.pt"))
        if candidates:
            return max(candidates, key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or "0"))
        fallback = self.run_dir / "checkpoints" / "checkpoint_step05000.pt"
        if fallback.exists():
            return fallback
        raise FileNotFoundError("no checkpoint found")

    def _load_checkpoint(self, ckpt_path: Path) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.generator.render_kwargs_train["network_fn"].load_state_dict(ckpt["generator_coarse"])
        if self.generator.render_kwargs_train["network_fine"] is not None and ckpt.get("generator_fine") is not None:
            self.generator.render_kwargs_train["network_fine"].load_state_dict(ckpt["generator_fine"])
        if self.encoder is not None and ckpt.get("encoder") is not None:
            self.encoder.load_state_dict(ckpt["encoder"])
        if self.z_fuser is not None and ckpt.get("z_fuser") is not None:
            self.z_fuser.load_state_dict(ckpt["z_fuser"])
        if self.gain_head is not None and ckpt.get("gain_head") is not None:
            self.gain_head.load_state_dict(ckpt["gain_head"])
        if self.gain_param is not None and ckpt.get("gain_param") is not None:
            self.gain_param.data.copy_(ckpt["gain_param"].to(self.device))

    def _build_pid_to_test_index(self) -> dict[str, int]:
        data_cfg = self.config["data"]
        _, _, test_subset, _ = self.train_mod.split_by_patient_id(
            dataset=self.dataset,
            seed=int(data_cfg.get("split_seed", 0)),
            split_mode=str(data_cfg.get("split_mode", "ratios")).lower(),
            train_ratio=float(data_cfg.get("split_train", 0.8)),
            val_ratio=float(data_cfg.get("split_val", 0.1)),
            test_ratio=float(data_cfg.get("split_test", 0.1)),
            train_count=int(data_cfg.get("split_train_count", -1)),
            val_count=int(data_cfg.get("split_val_count", -1)),
            test_count=int(data_cfg.get("split_test_count", -1)),
        )
        test_indices = set(getattr(test_subset, "indices", []))
        result: dict[str, int] = {}
        for idx in range(len(self.dataset)):
            pid = self.dataset.get_patient_id(idx)
            if pid in self.manifest and idx in test_indices:
                result[pid] = idx
        return result

    @staticmethod
    def _stats_tensor(tensor: torch.Tensor) -> dict[str, float]:
        flat = tensor.detach().reshape(-1).float().cpu().numpy()
        return {
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
        }

    def _assert_nonconstant(self, stage_label: str, tensor: torch.Tensor) -> None:
        stats = self._stats_tensor(tensor)
        if stats["std"] < 1e-12 or stats["min"] == stats["max"]:
            raise RuntimeError(stage_label)

    def render_patient(self, phantom_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
        idx = self.pid_to_index.get(phantom_id)
        if idx is None:
            raise KeyError(f"{phantom_id} missing in training test subset")
        sample = self.dataset[idx]
        batch = {}
        for k, v in sample.items():
            batch[k] = v.unsqueeze(0) if torch.is_tensor(v) else v

        ap = batch["ap"].to(self.device).float()
        pa = batch["pa"].to(self.device).float()
        ct_vol = batch.get("ct")
        if ct_vol is not None and torch.is_tensor(ct_vol) and ct_vol.numel() > 0:
            ct_vol = ct_vol.to(self.device).float()
        else:
            ct_vol = None
        ap_counts = batch.get("ap_counts")
        pa_counts = batch.get("pa_counts")
        use_counts = (
            ap_counts is not None and pa_counts is not None and torch.is_tensor(ap_counts)
            and torch.is_tensor(pa_counts) and ap_counts.numel() > 0 and pa_counts.numel() > 0
        )

        z_enc = None
        z_proj = None
        if bool(self.train_args.hybrid) and self.encoder is not None:
            z_final, z_enc, z_proj = self.train_mod.build_hybrid_latent_from_batch(
                args=self.train_args,
                batch=batch,
                device=self.device,
                z_latent_base=self.z_base,
                encoder=self.encoder,
                z_fuser=self.z_fuser,
                z_enc_alpha=float(self.train_args.z_enc_alpha),
            )
        else:
            z_final = self.z_base.detach()
        ct_context = self.generator.build_ct_context(ct_vol, padding_mode=self.train_args.ct_padding_mode) if ct_vol is not None else None
        prev_flag = self.generator.use_test_kwargs
        self.generator.eval()
        self.generator.use_test_kwargs = True
        with torch.no_grad():
            pred_ap_raw, _, _, _ = self.generator.render_from_pose(z_final, self.generator.pose_ap, ct_context=ct_context)
            pred_pa_raw, _, _, _ = self.generator.render_from_pose(z_final, self.generator.pose_pa, ct_context=ct_context)
        self.generator.use_test_kwargs = prev_flag
        self._assert_nonconstant("renderer_raw/AP", pred_ap_raw)
        self._assert_nonconstant("renderer_raw/PA", pred_pa_raw)

        if str(self.train_args.proj_loss_type) == "poisson":
            pred_ap = self.train_mod.compute_poisson_rate(pred_ap_raw, self.train_args.poisson_rate_mode, eps=1e-6)
            pred_pa = self.train_mod.compute_poisson_rate(pred_pa_raw, self.train_args.poisson_rate_mode, eps=1e-6)
        else:
            pred_ap = pred_ap_raw
            pred_pa = pred_pa_raw
        self._assert_nonconstant("post_rate/AP", pred_ap)
        self._assert_nonconstant("post_rate/PA", pred_pa)

        gain_tensor = None
        if use_counts and bool(self.train_args.use_gain):
            if self.gain_head is not None and z_enc is not None:
                gain_tensor = F.softplus(self.gain_head(z_enc))
            elif self.gain_param is not None:
                gain_tensor = F.softplus(self.gain_param)
            if gain_tensor is not None:
                g_min = float(self.train_args.gain_clamp_min) if self.train_args.gain_clamp_min is not None else None
                g_max = self.train_args.gain_clamp_max
                if g_min is not None or g_max is not None:
                    gain_tensor = torch.clamp(
                        gain_tensor,
                        min=(g_min if g_min is not None else -float("inf")),
                        max=(g_max if g_max is not None else float("inf")),
                    )
                pred_ap = pred_ap * gain_tensor
                pred_pa = pred_pa * gain_tensor

        if use_counts and str(self.train_args.proj_loss_type) == "poisson" and float(self.train_args.poisson_rate_floor) > 0.0:
            pred_ap, _ = self.train_mod.apply_poisson_rate_floor(
                pred_ap, float(self.train_args.poisson_rate_floor), self.train_args.poisson_rate_floor_mode
            )
            pred_pa, _ = self.train_mod.apply_poisson_rate_floor(
                pred_pa, float(self.train_args.poisson_rate_floor), self.train_args.poisson_rate_floor_mode
            )
        self._assert_nonconstant("post_gain/AP", pred_ap)
        self._assert_nonconstant("post_gain/PA", pred_pa)

        H, W = int(self.generator.H), int(self.generator.W)
        ap_np = pred_ap[0].reshape(H, W).detach().cpu().numpy().astype(np.float32)
        pa_np = pred_pa[0].reshape(H, W).detach().cpu().numpy().astype(np.float32)
        proj_scale = self.train_mod.compute_proj_scale(ap, pa, self.train_args.proj_scale_source, batch.get("meta"))
        proj_scale = torch.clamp(proj_scale, min=1e-6)
        meta = {
            "checkpoint_path": str(self.ckpt_path),
            "proj_forward_model": "train",
            "proj_scale_source": str(self.train_args.proj_scale_source),
            "proj_scale_value": [float(v) for v in proj_scale.detach().cpu().reshape(-1).tolist()],
            "poisson_mode": str(self.train_args.poisson_rate_mode),
            "gain_on": bool(gain_tensor is not None),
            "H": H,
            "W": W,
            "hybrid": bool(self.train_args.hybrid),
            "encoder_use_ct": bool(self.train_args.encoder_use_ct),
            "encoder_proj_transform": str(self.train_args.encoder_proj_transform),
            "z_enc_alpha": float(self.train_args.z_enc_alpha),
            "proj_target_source": str(self.train_args.proj_target_source),
            "pa_xflip_flag": bool(self.train_args.pa_xflip),
            "z_enc_stats": self._stats_tensor(z_enc) if z_enc is not None else None,
            "z_proj_stats": self._stats_tensor(z_proj) if z_proj is not None else None,
            "z_final_stats": self._stats_tensor(z_final),
            "raw_ap_stats": self._stats_tensor(pred_ap_raw),
            "raw_pa_stats": self._stats_tensor(pred_pa_raw),
            "post_rate_ap_stats": self._stats_tensor(pred_ap),
            "post_rate_pa_stats": self._stats_tensor(pred_pa),
        }
        return ap_np, pa_np, meta


def write_metrics_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def append_metrics_csv(path: Path, row: dict):
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(METRICS_CSV_HEADER)
        writer.writerow([row.get(col, "") for col in METRICS_CSV_HEADER])


def plot_organs(organ_stats: dict[int, dict], out_dir: Path, phantom_id: str):
    if not organ_stats:
        return
    organ_ids = list(organ_stats.keys())
    names = [organ_stats[oid]["organ_name"] for oid in organ_ids]
    gt_sums = [organ_stats[oid]["gt_sum_activity"] for oid in organ_ids]
    pred_sums = [organ_stats[oid]["pred_sum_activity"] for oid in organ_ids]
    bar_out = out_dir / f"{phantom_id}_organ_bar.png"
    width = max(8, len(names) * 0.6)
    height = max(4, len(names) * 0.35)
    fig, ax = plt.subplots(figsize=(width, height))
    bar_width = 0.35
    indices = np.arange(len(names))
    ax.bar(indices - bar_width / 2, gt_sums, bar_width, label="GT")
    ax.bar(indices + bar_width / 2, pred_sums, bar_width, label="Pred")
    ax.set_xticks(indices)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("sum activity (kBq/mL * cm^3)")
    ax.set_title(f"Organ sums {phantom_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(bar_out, dpi=150)
    plt.close(fig)
    scatter_out = out_dir / f"{phantom_id}_organ_scatter.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt_sums, pred_sums, alpha=0.7, s=40)
    max_val = max(max(gt_sums), max(pred_sums))
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1)
    ax.set_xlabel("GT sum activity")
    ax.set_ylabel("Pred sum activity")
    ax.set_title(f"Organ scatter {phantom_id}")
    fig.tight_layout()
    fig.savefig(scatter_out, dpi=150)
    plt.close(fig)


ORGAN_LABEL_MAP = {
    1017: "prostate",
    1228: "liver",
    1267: "spleen",
    1269: "kidney (right)",
    1270: "kidney (left)",
}


def _compute_active_organ_sums(
    ids: np.ndarray,
    mask_gt: np.ndarray,
    gt: np.ndarray,
    mask_predgrid: np.ndarray,
    pred_predgrid: np.ndarray,
    pred_gtgrid: np.ndarray,
    voxel_vol_gt: float,
    voxel_vol_pred: float,
) -> tuple[list[float], list[float], list[float]]:
    gt_sums = []
    pred_native_sums = []
    pred_gt_sums = []
    for organ_id in ids:
        gt_mask = mask_gt == organ_id
        if not gt_mask.any():
            gt_sums.append(0.0)
            pred_native_sums.append(0.0)
            pred_gt_sums.append(0.0)
            continue
        gt_sum = float(np.nansum(gt[gt_mask])) * voxel_vol_gt
        native_sum = float(np.nansum(pred_predgrid[mask_predgrid == organ_id])) * voxel_vol_pred
        gt_grid_sum = float(np.nansum(pred_gtgrid[gt_mask])) * voxel_vol_gt
        gt_sums.append(gt_sum)
        pred_native_sums.append(native_sum)
        pred_gt_sums.append(gt_grid_sum)
    return gt_sums, pred_native_sums, pred_gt_sums


def save_active_organ_plots(
    plot_dir: Path,
    phantom_id: str,
    active_ids: np.ndarray,
    mask_gt: np.ndarray | None,
    gt: np.ndarray,
    mask_predgrid: np.ndarray | None,
    pred_predgrid: np.ndarray,
    pred_gtgrid: np.ndarray,
    voxel_vol_gt: float,
    voxel_vol_pred: float,
) -> None:
    if not active_ids.size:
        return
    if mask_gt is None or mask_predgrid is None:
        return
    if not _MATPLOTLIB_AVAILABLE:
        _LOG.warning("matplotlib unavailable; skipping active organ plots for %s", phantom_id)
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    sorted_ids = np.sort(np.asarray(active_ids, dtype=np.int32))
    if np.any(sorted_ids == 1384):
        sorted_ids = sorted_ids[sorted_ids != 1384]
        _LOG.info("[organ-filter] removed unmapped organ id 1384 from active organ plots")
    # Keep original ordering (sorted IDs), but only keep organs with valid label mapping.
    organ_ids = np.array([int(oid) for oid in sorted_ids if int(oid) in ORGAN_LABEL_MAP], dtype=np.int32)
    if organ_ids.size == 0:
        return
    assert 1384 not in organ_ids
    names = [ORGAN_LABEL_MAP[int(oid)] for oid in organ_ids]
    gt_sums, _, pred_gt_sums = _compute_active_organ_sums(
        organ_ids,
        mask_gt,
        gt,
        mask_predgrid,
        pred_predgrid,
        pred_gtgrid,
        voxel_vol_gt,
        voxel_vol_pred,
    )
    indices = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.6), 4))
    width = 0.35
    ax.bar(indices - width / 2, gt_sums, width, label="GT")
    ax.bar(indices + width / 2, pred_gt_sums, width, label="Pred")
    ax.set_xticks(indices)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("sum activity")
    ax.set_title(f"Active organ sums {phantom_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "active_organs_abs.png", dpi=150)
    plt.close(fig)

    gt_total = sum(gt_sums)
    pred_gt_total = sum(pred_gt_sums)
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.6), 4))
    ax.bar(indices - width / 2, [_safe_divide(val, gt_total) for val in gt_sums], width, label="GT")
    ax.bar(indices + width / 2, [_safe_divide(val, pred_gt_total) for val in pred_gt_sums], width, label="Pred")
    ax.set_xticks(indices)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("fraction of active total")
    ax.set_title(f"Active organ fractions {phantom_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "active_organs_frac.png", dpi=150)
    plt.close(fig)


def run_postprocessing(args):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "postproc"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(Path(args.manifest))
    test_ids = read_split_ids(Path(args.split_json))
    for pid in test_ids:
        if pid not in manifest:
            raise KeyError(f"patient_id {pid} missing in manifest")
    pred_paths = find_pred_activity_paths(run_dir, test_ids, args)
    config = load_yaml(Path(args.config))
    data_cfg = config.get("data", {})
    radius_xyz_cm = config.get("data", {}).get("radius_xyz_cm")
    if radius_xyz_cm is None:
        raise KeyError("config missing data.radius_xyz_cm")
    radius_xyz = tuple(float(r) for r in radius_xyz_cm)
    voxel_mm = float(data_cfg.get("voxel_mm", 1.5))
    timestamp = datetime.utcnow().isoformat()
    git_hash = git_short_hash()
    config_path = str(Path(args.config).resolve())
    aggregated_path = out_dir / "metrics.csv"
    organ_name_map = read_organ_names(Path("data/organ_ids.txt"))
    physics_projector_config_cls = None
    physics_project_activity = None
    if args.proj_forward_model == "physics":
        physics_projector_config_cls, physics_project_activity = _get_physics_projector_imports()
    elif args.calibrate_scale:
        raise ValueError("--calibrate-scale is only supported with --proj-forward-model physics")
    if args.ls_calibrate_global and args.proj_forward_model != "train":
        raise ValueError("--ls-calibrate-global is only supported with --proj-forward-model train")
    train_forward_projector = None
    if args.proj_forward_model == "train":
        train_forward_projector = TrainForwardProjector(
            args=args,
            run_dir=run_dir,
            manifest=manifest,
            device=torch.device(args.device),
        )
        _LOG.info(
            "initialized train forward projector checkpoint=%s",
            train_forward_projector.ckpt_path,
        )

    counts_keys_ap = ["ap_counts_path", "ap_counts", "ap_counts_abs", "ap_counts_path_abs"]
    counts_keys_pa = ["pa_counts_path", "pa_counts", "pa_counts_abs", "pa_counts_path_abs"]
    norm_keys_ap = ["ap_path", "ap", "ap_abs", "ap_path_abs"]
    norm_keys_pa = ["pa_path", "pa", "pa_abs", "pa_path_abs"]

    block_totals: dict[str, list[float]] = defaultdict(list)
    proj_metrics_target_rows: list[dict[str, Union[str, float]]] = []
    run_proj_metrics_records: list[dict[str, Union[str, float]]] = []
    fast_warning_logged = False
    start_total = time.perf_counter()
    for pid in test_ids:
        entry = manifest[pid]
        patient_dir = out_dir / pid
        patient_dir.mkdir(parents=True, exist_ok=True)
        act_path = Path(entry["act_path"]) if entry.get("act_path") else None
        if not act_path or not act_path.exists():
            raise FileNotFoundError(f"missing act for {pid}")

        timer = BlockTimer(args.timing)
        with timer.block("load_gt"):
            gt_act = load_array(act_path, mmap=True)
        gt_shape = gt_act.shape
        spacing_gt = spacing_from_meta(act_path, voxel_mm)
        slices = (slice(None), slice(None), slice(None))
        gt_roi = gt_act[slices]
        idx_ranges = {
            "axis0": {"min": 0, "max": gt_shape[0] - 1},
            "axis1": {"min": 0, "max": gt_shape[1] - 1},
            "axis2": {"min": 0, "max": gt_shape[2] - 1},
        }
        phys_extent = {}
        for axis in range(3):
            size = gt_shape[axis]
            half_extent = ((size - 1) / 2.0) * spacing_gt[axis]
            phys_extent[f"axis{axis}"] = [-half_extent, half_extent]
        roi_shape = gt_shape
        ct_path = Path(entry["ct_path"]) if entry.get("ct_path") else None
        meta = _load_meta_simple(act_path)
        kernel_default = Path("Data_Processing/LEAP_Kernel.mat")
        projector_config = None
        if args.proj_forward_model == "physics":
            projector_config = _build_projector_config(
                meta,
                args,
                spacing_gt,
                kernel_default,
                physics_projector_config_cls,
            )
        proj_domain_requested = args.proj_domain
        domain_loaded, gt_ap_counts, gt_pa_counts = _load_gt_projections(
            entry,
            counts_keys_ap,
            counts_keys_pa,
            norm_keys_ap,
            norm_keys_pa,
            pid,
            proj_domain_requested,
        )
        proj_domain_loaded = domain_loaded
        gt_shape_2d = tuple(gt_ap_counts.shape) if gt_ap_counts is not None else None
        noisy_strict = args.proj_metrics_target == "noisy_counts"
        noisy_ap_counts, noisy_pa_counts = _load_noisy_projection_targets(
            run_dir,
            pid,
            expected_shape=gt_shape_2d,
            strict=noisy_strict,
        )
        clean_ap_path = resolve_manifest_path(entry, counts_keys_ap)
        clean_pa_path = resolve_manifest_path(entry, counts_keys_pa)
        _LOG.info(
            "[proj-metrics-target] phantom=%s mode=%s clean_paths=(%s,%s) noisy_paths=(%s,%s)",
            pid,
            args.proj_metrics_target,
            clean_ap_path,
            clean_pa_path,
            (run_dir / "test_slices" / pid / "noisy_ap_counts.npy"),
            (run_dir / "test_slices" / pid / "noisy_pa_counts.npy"),
        )
        if noisy_ap_counts is not None and noisy_pa_counts is not None:
            _LOG.info(
                "[proj-metrics-target][noisy-sum] phantom=%s sum(target_ap)=%.6e sum(target_pa)=%.6e",
                pid,
                float(np.nansum(noisy_ap_counts)),
                float(np.nansum(noisy_pa_counts)),
            )

        with timer.block("load_pred"):
            pred_path = pred_paths.get(pid)
            if pred_path is None:
                raise RuntimeError(f"no pred path for {pid}")
            pred_act = load_array(pred_path)

        with timer.block("resample"):
            if args.fast_resample_roi and not fast_warning_logged:
                _LOG.warning("--fast-resample-roi ignored: always resampling to full GT grid")
                fast_warning_logged = True
            pred_gt_full = resample_pred_to_gt(pred_act, gt_shape, args.device)
            pred_roi = pred_gt_full[slices]

        orientation_fix_margin = 0.02
        orientation_search_axis = int(args.act_compare_axis)
        orientation_search_axes = plane_axes_for_slice_axis(orientation_search_axis)
        orientation_fix_info = {
            "applied": False,
            "transform": "identity",
            "ncc_identity": float("nan"),
            "ncc_best": float("nan"),
            "com_dist_identity_vox": float("nan"),
            "com_dist_best_vox": float("nan"),
            "search_axis": orientation_search_axis,
            "search_axes": orientation_search_axes,
            "apply_margin_ncc": float(orientation_fix_margin),
        }
        if args.proj_forward_model == "train":
            _LOG.info(
                "[orient-fix][%s] orientation_search_axes=%s (from slice axis=%d)",
                pid,
                orientation_search_axes,
                orientation_search_axis,
            )
            orient_eval = _pick_best_pred_orientation(
                gt_roi=gt_roi,
                pred_roi=pred_roi,
                phantom_id=pid,
                debug=bool(args.debug_orientation_search),
                plane_axes=orientation_search_axes,
            )
            best = orient_eval["best"]
            identity = orient_eval["identity"]
            should_apply = bool(
                np.isfinite(best["ncc"])
                and np.isfinite(identity["ncc"])
                and float(best["ncc"]) > float(identity["ncc"]) + orientation_fix_margin
            )
            if should_apply:
                pred_gt_full = _apply_orientation_transform(
                    pred_gt_full,
                    k_rot90=best["k_rot90"],
                    do_fliplr=best["fliplr"],
                    do_flipud=best["flipud"],
                    plane_axes=orientation_search_axes,
                ).astype(np.float32, copy=False)
                pred_roi = pred_gt_full[slices]
            orientation_fix_info = {
                "applied": bool(should_apply),
                "transform": str(best["label"]),
                "ncc_identity": float(identity["ncc"]),
                "ncc_best": float(best["ncc"]),
                "com_dist_identity_vox": float(identity["com_dist_vox"]),
                "com_dist_best_vox": float(best["com_dist_vox"]),
                "search_axis": orientation_search_axis,
                "search_axes": orientation_search_axes,
                "apply_margin_ncc": float(orientation_fix_margin),
            }
            _LOG.info(
                "[orient-fix][%s] best transform: %s | ncc_identity=%.6f ncc_best=%.6f "
                "| com_identity=%.6f com_best=%.6f | apply_margin=%.3f | applied=%s | shapes gt=%s pred=%s",
                pid,
                best["label"],
                float(identity["ncc"]),
                float(best["ncc"]),
                float(identity["com_dist_vox"]),
                float(best["com_dist_vox"]),
                orientation_fix_margin,
                orientation_fix_info["applied"],
                tuple(gt_roi.shape),
                tuple(pred_roi.shape),
            )
        calibration_scale = 1.0
        if args.calibrate_scale:
            if ct_path is None or not ct_path.exists():
                raise FileNotFoundError(f"--calibrate-scale requires ct_path for {pid}")
            if projector_config is None:
                raise RuntimeError(f"--calibrate-scale requires valid projector configuration for {pid}")
            ct_vol_cal = load_array(ct_path).astype(np.float32)
            pred_proj_for_cal = physics_project_activity(pred_roi, ct_vol_cal, spacing_gt, projector_config)
            pred_ap_cal = (
                pred_proj_for_cal["ap_counts"] if proj_domain_loaded == "counts" else pred_proj_for_cal["ap_norm"]
            )
            pred_pa_cal = (
                pred_proj_for_cal["pa_counts"] if proj_domain_loaded == "counts" else pred_proj_for_cal["pa_norm"]
            )
            calibration_scale = _compute_joint_projection_ls_scale(
                pred_ap_cal,
                pred_pa_cal,
                gt_ap_counts,
                gt_pa_counts,
            )
            if not np.isfinite(calibration_scale):
                raise ValueError(f"invalid calibration scale for {pid}: {calibration_scale}")
            scale32 = np.float32(calibration_scale)
            pred_act = (pred_act * scale32).astype(np.float32, copy=False)
            pred_gt_full = (pred_gt_full * scale32).astype(np.float32, copy=False)
            pred_roi = (pred_roi * scale32).astype(np.float32, copy=False)
            _LOG.info("[%s] calibration scale=%.5f", pid, calibration_scale)

        if args.save_orientation_debug_volumes:
            orient_dir = patient_dir / "orientation_debug"
            orient_dir.mkdir(parents=True, exist_ok=True)
            np.save(orient_dir / "gt_roi.npy", np.asarray(gt_roi, dtype=np.float32))
            np.save(orient_dir / "pred_roi_gtgrid.npy", np.asarray(pred_roi, dtype=np.float32))
            (orient_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "phantom_id": pid,
                        "orientation_fix": orientation_fix_info,
                        "gt_shape": tuple(gt_roi.shape),
                        "pred_shape": tuple(pred_roi.shape),
                    },
                    indent=2,
                )
            )
        V_gt = spacing_gt[0] * spacing_gt[1] * spacing_gt[2]
        A_gt_vol = float(gt_roi.sum()) * V_gt
        A_pred_vol = float(pred_roi.sum()) * V_gt
        bias_vol = (A_pred_vol - A_gt_vol) / A_gt_vol if A_gt_vol != 0 else float("nan")
        rel_abs_vol = abs(A_pred_vol - A_gt_vol) / (A_gt_vol if A_gt_vol != 0 else 1e-6)

        with timer.block("voxel_metrics"):
            vol_mae_vol, vol_rmse_vol = compute_voxel_metrics(gt_roi, pred_roi)
            pred_res = pred_act.shape
            spacing_pred = tuple((2.0 * float(radius_xyz[i])) / (pred_res[i] - 1) for i in range(3))
            V_pred = spacing_pred[0] * spacing_pred[1] * spacing_pred[2]
            pred_act_phys = pred_act
            A_pred_native = float(pred_act_phys.sum()) * V_pred
            fg_mask = gt_roi > FG_THRESHOLD
            voxel_mae_fg, voxel_rmse_fg, voxel_n_fg = masked_voxel_metrics(gt_roi, pred_roi, fg_mask)

        organ_stats = {}
        active_organ_fraction_mae = float("nan")
        organ_aggregate_metrics = _default_organ_aggregate_metrics()
        mask = None
        mask_roi = None
        active_ids = np.array([], dtype=np.int32)
        mask_path = Path(args.mask_path_pattern.format(phantom=pid))
        with timer.block("organ_metrics"):
            mask = load_mask(mask_path, gt_shape)
            if mask is not None:
                mask_roi = mask[slices]
            active_ids = extract_active_organ_ids(mask, gt_act)
            if not args.skip_organ_metrics and mask_roi is not None:
                organ_stats, organ_aggregate_metrics, active_organ_fraction_mae = compute_organ_statistics(
                    mask_roi,
                    gt_roi,
                    pred_roi,
                    spacing_gt,
                    organ_name_map,
                )
                _LOG.info(
                    "[organ-fraction][%s] active_organ_fraction_mae=%.6g (plot-aligned, fraction of active mapped organs) "
                    "label_averaged_organ_fraction_mae=%.6g K_active=%d",
                    pid,
                    active_organ_fraction_mae,
                    organ_aggregate_metrics["label_averaged_organ_fraction_mae"],
                    int(organ_aggregate_metrics["active_organ_fraction_n"]),
                )
            else:
                organ_stats = {}

        mask_predgrid = resample_mask_to_pred(mask, pred_act.shape) if mask is not None else None
        pred_metrics_predgrid = compute_pred_grid_activity_metrics(
            pred_act,
            mask_predgrid,
            active_ids,
            V_pred,
        )
        pred_metrics_gtgrid = compute_pred_grid_activity_metrics(
            pred_gt_full,
            mask,
            active_ids,
            V_gt,
        )
        pred_mass_in_gt_active = pred_metrics_predgrid.get("in_gt_active_sum", float("nan"))

        if args.verbose:
            _LOG.info(
                "[pred_predgrid] total=%.6g in_body=%.6g in_gt_active=%.6g",
                pred_metrics_predgrid["total_sum"],
                pred_metrics_predgrid["in_body_sum"],
                pred_metrics_predgrid["in_gt_active_sum"],
            )
            _LOG.info(
                "[pred_gtgrid] total=%.6g in_body=%.6g in_gt_active=%.6g",
                pred_metrics_gtgrid["total_sum"],
                pred_metrics_gtgrid["in_body_sum"],
                pred_metrics_gtgrid["in_gt_active_sum"],
            )

        pred_outside_mask_sum, pred_outside_mask_frac = compute_outside_mask_leakage(
            pred_roi,
            mask_roi,
            V_gt,
            A_pred_vol,
            pid,
        )
        pred_in_body_frac = float("nan")
        if not np.isnan(pred_outside_mask_frac):
            pred_in_body_frac = 1.0 - pred_outside_mask_frac
        _LOG.info(
            "[body-leakage][%s] pred_outside_body_frac=%.6f pred_in_body_frac=%.6f",
            pid,
            pred_outside_mask_frac,
            pred_in_body_frac,
        )

        if organ_stats and not args.skip_organ_table:
            table_path = patient_dir / "organ_table.csv"
            with table_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "organ_id",
                        "organ_name",
                        "gt_sum_activity",
                        "pred_sum_activity",
                        "gt_mean",
                        "pred_mean",
                        "rel_error_total_activity",
                        "is_active",
                    ]
                )
                for oid in sorted(organ_stats):
                    row = organ_stats[oid]
                    if (
                        abs(row["gt_sum_activity"]) < 1e-12
                        and abs(row["pred_sum_activity"]) < 1e-12
                    ):
                        continue
                    writer.writerow(
                        [
                            oid,
                            row["organ_name"],
                            row["gt_sum_activity"],
                            row["pred_sum_activity"],
                            row["gt_mean"],
                            row["pred_mean"],
                            row["rel_error_total_activity"] if row["is_active"] else "",
                            int(bool(row["is_active"])),
                        ]
                    )

        with timer.block("plots"):
            plot_dir = patient_dir / "plots"
            if organ_stats and not args.skip_plots:
                plot_dir.mkdir(exist_ok=True)
                plot_organs(organ_stats, plot_dir, pid)
            if args.save_active_organ_plots:
                save_active_organ_plots(
                    plot_dir,
                    pid,
                    active_ids,
                    mask,
                    gt_act,
                    mask_predgrid,
                    pred_act,
                    pred_gt_full,
                    V_gt,
                    V_pred,
                )
            if args.save_act_compare_5slices:
                # Use the same GT-grid arrays as voxel metrics (gt_roi/pred_roi).
                gt = gt_roi
                pred_gtgrid = pred_roi
                _LOG.info(
                    "[act-compare-5][%s] plot-input-check gt.shape=%s pred_gtgrid.shape=%s gt.sum=%.6g pred_gtgrid.sum=%.6g",
                    pid,
                    tuple(gt.shape),
                    tuple(pred_gtgrid.shape),
                    float(np.sum(gt)),
                    float(np.sum(pred_gtgrid)),
                )
                save_activity_compare_5slices(
                    plot_dir=plot_dir,
                    phantom_id=pid,
                    gt=gt,
                    pred_gtgrid=pred_gtgrid,
                    axis=args.act_compare_axis,
                    stride=args.act_compare_stride,
                    outname_template=args.act_compare_outname,
                )

        with timer.block("projections"):
            proj_domain = proj_domain_loaded
            proj_metrics: dict[str, float] | None = None
            proj_status = "missing"
            proj_norm_factor = None
            pred_save_ap = None
            pred_save_pa = None
            gt_save_ap = gt_ap_counts
            gt_save_pa = gt_pa_counts
            proj_is_normalized = proj_domain == "normalized"

            should_render_projections = args.render_projections or args.calibrate_scale
            if args.proj_forward_model == "train":
                should_render_projections = True
            if should_render_projections:
                if args.proj_forward_model == "train":
                    if train_forward_projector is None:
                        raise RuntimeError("train forward projector not initialized")
                    try:
                        pred_save_ap, pred_save_pa, train_meta = train_forward_projector.render_patient(pid)
                    except Exception as exc:
                        _LOG.exception("failed to render train-path projections for %s: %s", pid, exc)
                        raise
                    pred_save_ap, pred_save_pa = pred_save_pa, pred_save_ap
                    _LOG.info("[ap/pa-fix] swapped train-forward AP/PA outputs before metrics/saving for %s", pid)
                    ls_alpha = None
                    ls_num = None
                    ls_den = None
                    pred_sum_before = float(np.nansum(pred_save_ap) + np.nansum(pred_save_pa))
                    gt_sum_ap_val = float(np.nansum(gt_save_ap)) if gt_save_ap is not None else float("nan")
                    gt_sum_pa_val = float(np.nansum(gt_save_pa)) if gt_save_pa is not None else float("nan")
                    if args.ls_calibrate_global:
                        if proj_domain_loaded != "counts":
                            raise RuntimeError(
                                f"--ls-calibrate-global requires counts GT, but loaded domain={proj_domain_loaded} for {pid}"
                            )
                        if gt_save_ap is None or gt_save_pa is None:
                            raise RuntimeError(
                                f"--ls-calibrate-global requires gt counts for {pid}, but GT projections are missing"
                            )
                        ls_alpha, ls_num, ls_den = _ls_alpha_global(
                            pred_save_ap,
                            pred_save_pa,
                            gt_save_ap,
                            gt_save_pa,
                            eps=float(args.ls_calibrate_eps),
                        )
                        pred_save_ap = (np.asarray(pred_save_ap, dtype=np.float32) * np.float32(ls_alpha)).astype(np.float32, copy=False)
                        pred_save_pa = (np.asarray(pred_save_pa, dtype=np.float32) * np.float32(ls_alpha)).astype(np.float32, copy=False)
                        _LOG.info(
                            "[ls-cal][%s] alpha_global=%.6e num=%.6e den=%.6e",
                            pid,
                            ls_alpha,
                            ls_num,
                            ls_den,
                        )
                    pred_sum_after = float(np.nansum(pred_save_ap) + np.nansum(pred_save_pa))
                    train_meta["ls_calibrate_global"] = bool(args.ls_calibrate_global)
                    train_meta["alpha_global"] = float(ls_alpha) if ls_alpha is not None else None
                    train_meta["alpha_num"] = float(ls_num) if ls_num is not None else None
                    train_meta["alpha_den"] = float(ls_den) if ls_den is not None else None
                    train_meta["pred_sum_before"] = pred_sum_before
                    train_meta["pred_sum_after"] = pred_sum_after
                    train_meta["gt_sum_ap"] = gt_sum_ap_val
                    train_meta["gt_sum_pa"] = gt_sum_pa_val
                    proj_status = "rendered_train"
                    proj_domain = "counts"
                    proj_is_normalized = False
                    train_proj_dir = patient_dir / "proj_train"
                    train_proj_dir.mkdir(parents=True, exist_ok=True)
                    np.save(train_proj_dir / "ap_pred.npy", pred_save_ap)
                    np.save(train_proj_dir / "pa_pred.npy", pred_save_pa)
                    (train_proj_dir / "meta.json").write_text(json.dumps(train_meta, indent=2))
                elif ct_path and ct_path.exists() and projector_config is not None:
                    try:
                        ct_vol = load_array(ct_path).astype(np.float32)
                        pred_proj = physics_project_activity(pred_roi, ct_vol, spacing_gt, projector_config)
                    except Exception as exc:
                        _LOG.exception("failed to render projections for %s: %s", pid, exc)
                        if args.calibrate_scale:
                            raise RuntimeError(
                                f"--calibrate-scale failed while rendering scaled projections for {pid}"
                            ) from exc
                        proj_status = "failed"
                    else:
                        pred_save_ap = (
                            pred_proj["ap_counts"] if proj_domain == "counts" else pred_proj["ap_norm"]
                        )
                        pred_save_pa = (
                            pred_proj["pa_counts"] if proj_domain == "counts" else pred_proj["pa_norm"]
                        )
                        proj_status = "rendered"
                        proj_norm_factor = pred_proj.get("norm_scale")

            if proj_metrics is None and (pred_save_ap is None or pred_save_pa is None):
                if args.calibrate_scale:
                    raise RuntimeError(
                        f"--calibrate-scale requires rendered scaled projections for metrics, but rendering was unavailable for {pid}"
                    )
                fallback_result = compute_projection_metrics_with_fallback(
                    args,
                    run_dir,
                    out_dir,
                    pid,
                    gt_save_ap,
                    gt_save_pa,
                    proj_domain,
                )
                if pred_save_ap is None:
                    pred_save_ap = fallback_result.pop("pred_ap", None)
                else:
                    fallback_result.pop("pred_ap", None)
                if pred_save_pa is None:
                    pred_save_pa = fallback_result.pop("pred_pa", None)
                else:
                    fallback_result.pop("pred_pa", None)
                proj_status = fallback_result.get("proj_status", proj_status)
                proj_metrics = fallback_result
                proj_domain = proj_metrics.get("proj_domain", proj_domain)
                proj_is_normalized = proj_domain == "normalized"
                if proj_is_normalized and proj_norm_factor is None:
                    proj_norm_factor = (
                        float(meta.get("proj_scale_joint_p99"))
                        if isinstance(meta.get("proj_scale_joint_p99"), (int, float))
                        else None
                    )

            # Always compute projection metrics against clean/noisy targets (if available) on the
            # exact same prediction arrays. Legacy keys are selected by --proj-metrics-target.
            proj_metrics_clean = {
                "proj_mae_counts": float("nan"),
                "proj_poisson_dev_counts": float("nan"),
            }
            proj_metrics_noisy = {
                "proj_mae_counts": float("nan"),
                "proj_poisson_dev_counts": float("nan"),
            }
            if pred_save_ap is not None and pred_save_pa is not None:
                if gt_save_ap is not None and gt_save_pa is not None:
                    mae_c, dev_c = _compute_proj_metrics_from_arrays(
                        pred_save_ap, pred_save_pa, gt_save_ap, gt_save_pa, args.device
                    )
                    proj_metrics_clean["proj_mae_counts"] = mae_c
                    proj_metrics_clean["proj_poisson_dev_counts"] = dev_c
                if noisy_ap_counts is not None and noisy_pa_counts is not None:
                    mae_n, dev_n = _compute_proj_metrics_from_arrays(
                        pred_save_ap, pred_save_pa, noisy_ap_counts, noisy_pa_counts, args.device
                    )
                    proj_metrics_noisy["proj_mae_counts"] = mae_n
                    proj_metrics_noisy["proj_poisson_dev_counts"] = dev_n
            elif args.proj_metrics_target == "noisy_counts":
                raise RuntimeError(
                    f"phantom={pid}: --proj-metrics-target noisy_counts requested, but prediction projections are unavailable"
                )

            if args.proj_metrics_target == "clean_counts":
                proj_metrics = {
                    "proj_mae_counts": proj_metrics_clean["proj_mae_counts"],
                    "proj_poisson_dev_counts": proj_metrics_clean["proj_poisson_dev_counts"],
                    "proj_status": proj_status,
                    "proj_domain": proj_domain,
                }
            elif args.proj_metrics_target == "noisy_counts":
                proj_metrics = {
                    "proj_mae_counts": proj_metrics_noisy["proj_mae_counts"],
                    "proj_poisson_dev_counts": proj_metrics_noisy["proj_poisson_dev_counts"],
                    "proj_status": proj_status,
                    "proj_domain": proj_domain,
                }
            else:
                raise ValueError(f"Unknown proj-metrics-target: {args.proj_metrics_target}")

        if (
            pred_save_ap is not None
            and pred_save_pa is not None
            and gt_save_ap is not None
            and gt_save_pa is not None
        ):
            alpha_sum, alpha_ls = _compute_projection_scale_factors(
                pred_save_ap, pred_save_pa, gt_save_ap, gt_save_pa
            )
            _log_projection_stats(pid, pred_save_ap, pred_save_pa, gt_save_ap, gt_save_pa, alpha_sum, alpha_ls)
            if args.save_proj_png:
                plot_dir = patient_dir / "plots"
                _save_projection_pngs(plot_dir, pred_save_ap, pred_save_pa, gt_save_ap, gt_save_pa)

        if args.save_proj_npy and pred_save_ap is not None and pred_save_pa is not None:
            proj_dir = patient_dir / "proj"
            proj_dir.mkdir(parents=True, exist_ok=True)
            try:
                np.save(proj_dir / "pred_ap.npy", pred_save_ap)
                np.save(proj_dir / "pred_pa.npy", pred_save_pa)
                if gt_save_ap is not None and gt_save_pa is not None:
                    np.save(proj_dir / "gt_ap.npy", gt_save_ap)
                    np.save(proj_dir / "gt_pa.npy", gt_save_pa)
                _LOG.info(
                    "saved projections for %s (domain=%s) to %s",
                    pid,
                    proj_domain,
                    proj_dir,
                )
            except Exception as exc:
                _LOG.warning("failed to save projection npys for %s: %s", pid, exc)

        proj_mae_counts = proj_metrics["proj_mae_counts"]
        proj_poisson_dev_counts = proj_metrics["proj_poisson_dev_counts"]
        proj_mae_counts_clean = proj_metrics_clean["proj_mae_counts"]
        proj_poisson_dev_counts_clean = proj_metrics_clean["proj_poisson_dev_counts"]
        proj_mae_counts_noisy = proj_metrics_noisy["proj_mae_counts"]
        proj_poisson_dev_counts_noisy = proj_metrics_noisy["proj_poisson_dev_counts"]
        proj_metrics_target_mode_used = str(args.proj_metrics_target)
        proj_status = proj_metrics["proj_status"]
        proj_domain = proj_metrics["proj_domain"]
        proj_is_normalized = proj_domain == "normalized"
        assumptions = {
            "evaluation_region": "full_gt_grid",
            "metric_region_suffix": "_vol",
        }
        if args.fast_resample_roi:
            assumptions["fast_resample_roi"] = "ignored in favor of full-grid resampling"
        organ_rel_error_active_mean = organ_aggregate_metrics[
            "organ_rel_error_total_activity_active_mean"
        ]
        metrics_data = {
            "phantom_id": pid,
            "run": {
                "git_hash": git_hash,
                "config_path": config_path,
                "args": vars(args),
                "timestamp": timestamp,
            },
            "grid": {
                "gt_shape": gt_shape,
                "pred_shape": pred_act.shape,
                "spacing_gt_cm": spacing_gt,
                "spacing_pred_cm": spacing_pred,
                "roi_slices": idx_ranges,
                "roi_shape": roi_shape,
                "physical_extent_cm": phys_extent,
                "resample": {
                    "direction": "pred_to_gt",
                    "mode": "trilinear",
                    "align_corners": False,
                    "device": args.device,
                },
            },
            "metrics": {
                "vol_mae_vol": vol_mae_vol,
                "vol_rmse_vol": vol_rmse_vol,
                "activity_bias_vol": bias_vol,
                "activity_rel_abs_error_vol": rel_abs_vol,
                "voxel_mae_fg": voxel_mae_fg,
                "voxel_rmse_fg": voxel_rmse_fg,
                "voxel_n_fg": voxel_n_fg,
                "fg_tau": FG_THRESHOLD,
                "proj_mae_counts": proj_mae_counts,
                "proj_poisson_dev_counts": proj_poisson_dev_counts,
                "proj_mae_counts_clean": proj_mae_counts_clean,
                "proj_poisson_dev_counts_clean": proj_poisson_dev_counts_clean,
                "proj_mae_counts_noisy": proj_mae_counts_noisy,
                "proj_poisson_dev_counts_noisy": proj_poisson_dev_counts_noisy,
                "proj_metrics_target_mode_used": proj_metrics_target_mode_used,
                "organ_rel_error_total_activity_active_mean": organ_rel_error_active_mean,
                "active_organ_fraction_mae": active_organ_fraction_mae,
                "label_averaged_organ_fraction_mae": organ_aggregate_metrics["label_averaged_organ_fraction_mae"],
                "active_organ_fraction_n": int(organ_aggregate_metrics["active_organ_fraction_n"]),
                "organ_active_n": organ_aggregate_metrics["organ_active_n"],
                "organ_inactive_n": organ_aggregate_metrics["organ_inactive_n"],
                "inactive_organs_pred_sum": organ_aggregate_metrics["inactive_organs_pred_sum"],
                "inactive_organs_pred_frac_of_pred": organ_aggregate_metrics["inactive_organs_pred_frac_of_pred"],
                "inactive_organs_pred_frac_of_gt": organ_aggregate_metrics["inactive_organs_pred_frac_of_gt"],
                "pred_outside_mask_sum": pred_outside_mask_sum,
                "pred_outside_mask_frac": pred_outside_mask_frac,
                "A_gt_vol": A_gt_vol,
                "A_pred_vol": A_pred_vol,
                "A_pred_native": A_pred_native,
                "pred_mass_in_gt_active": pred_mass_in_gt_active,
                "pred_total_sum_predgrid": pred_metrics_predgrid["total_sum"],
                "pred_in_body_sum_predgrid": pred_metrics_predgrid["in_body_sum"],
                "pred_outside_body_sum_predgrid": pred_metrics_predgrid["out_body_sum"],
                "pred_in_gt_active_sum_predgrid": pred_metrics_predgrid["in_gt_active_sum"],
                "pred_not_gt_active_sum_predgrid": pred_metrics_predgrid["not_gt_active_sum"],
                "pred_in_body_frac_predgrid": pred_metrics_predgrid["in_body_frac"],
                "pred_outside_body_frac_predgrid": pred_metrics_predgrid["out_body_frac"],
                "pred_in_gt_active_frac_predgrid": pred_metrics_predgrid["in_gt_active_frac"],
                "pred_not_gt_active_frac_predgrid": pred_metrics_predgrid["not_gt_active_frac"],
                "pred_total_sum_gtgrid": pred_metrics_gtgrid["total_sum"],
                "pred_in_body_sum_gtgrid": pred_metrics_gtgrid["in_body_sum"],
                "pred_outside_body_sum_gtgrid": pred_metrics_gtgrid["out_body_sum"],
                "pred_in_gt_active_sum_gtgrid": pred_metrics_gtgrid["in_gt_active_sum"],
                "pred_not_gt_active_sum_gtgrid": pred_metrics_gtgrid["not_gt_active_sum"],
                "pred_in_body_frac_gtgrid": pred_metrics_gtgrid["in_body_frac"],
                "pred_outside_body_frac_gtgrid": pred_metrics_gtgrid["out_body_frac"],
                "pred_in_gt_active_frac_gtgrid": pred_metrics_gtgrid["in_gt_active_frac"],
                "pred_not_gt_active_frac_gtgrid": pred_metrics_gtgrid["not_gt_active_frac"],
                "proj_status": proj_status,
                "proj_domain": proj_domain,
                "proj_is_normalized": proj_is_normalized,
                "proj_norm_factor": proj_norm_factor,
                "calibration_scale": calibration_scale,
                "calibrate_scale_enabled": bool(args.calibrate_scale),
                "orientation_fix_applied": bool(orientation_fix_info["applied"]),
                "orientation_fix_transform": orientation_fix_info["transform"],
                "orientation_fix_ncc_identity": orientation_fix_info["ncc_identity"],
                "orientation_fix_ncc_best": orientation_fix_info["ncc_best"],
                "orientation_fix_com_dist_identity_vox": orientation_fix_info["com_dist_identity_vox"],
                "orientation_fix_com_dist_best_vox": orientation_fix_info["com_dist_best_vox"],
            },
            "assumptions": assumptions,
        }
        write_metrics_json(patient_dir / "metrics.json", metrics_data)
        append_metrics_csv(
            aggregated_path,
            {
                "phantom_id": pid,
                "vol_mae_vol": vol_mae_vol,
                "vol_rmse_vol": vol_rmse_vol,
                "activity_bias_vol": bias_vol,
                "activity_rel_abs_error_vol": rel_abs_vol,
                "voxel_mae_fg": voxel_mae_fg,
                "voxel_rmse_fg": voxel_rmse_fg,
                "voxel_n_fg": voxel_n_fg,
                "fg_tau": FG_THRESHOLD,
                "proj_mae_counts": proj_mae_counts,
                "proj_poisson_dev_counts": proj_poisson_dev_counts,
                "proj_mae_counts_clean": proj_mae_counts_clean,
                "proj_poisson_dev_counts_clean": proj_poisson_dev_counts_clean,
                "proj_mae_counts_noisy": proj_mae_counts_noisy,
                "proj_poisson_dev_counts_noisy": proj_poisson_dev_counts_noisy,
                "proj_metrics_target_mode_used": proj_metrics_target_mode_used,
                "organ_rel_error_total_activity_active_mean": organ_rel_error_active_mean,
                "active_organ_fraction_mae": active_organ_fraction_mae,
                "label_averaged_organ_fraction_mae": organ_aggregate_metrics["label_averaged_organ_fraction_mae"],
                "active_organ_fraction_n": int(organ_aggregate_metrics["active_organ_fraction_n"]),
                "organ_active_n": organ_aggregate_metrics["organ_active_n"],
                "organ_inactive_n": organ_aggregate_metrics["organ_inactive_n"],
                "inactive_organs_pred_sum": organ_aggregate_metrics["inactive_organs_pred_sum"],
                "inactive_organs_pred_frac_of_pred": organ_aggregate_metrics["inactive_organs_pred_frac_of_pred"],
                "inactive_organs_pred_frac_of_gt": organ_aggregate_metrics["inactive_organs_pred_frac_of_gt"],
                "pred_total_sum_predgrid": pred_metrics_predgrid["total_sum"],
                "pred_in_body_sum_predgrid": pred_metrics_predgrid["in_body_sum"],
                "pred_outside_body_sum_predgrid": pred_metrics_predgrid["out_body_sum"],
                "pred_in_gt_active_sum_predgrid": pred_metrics_predgrid["in_gt_active_sum"],
                "pred_not_gt_active_sum_predgrid": pred_metrics_predgrid["not_gt_active_sum"],
                "pred_in_body_frac_predgrid": pred_metrics_predgrid["in_body_frac"],
                "pred_outside_body_frac_predgrid": pred_metrics_predgrid["out_body_frac"],
                "pred_in_gt_active_frac_predgrid": pred_metrics_predgrid["in_gt_active_frac"],
                "pred_not_gt_active_frac_predgrid": pred_metrics_predgrid["not_gt_active_frac"],
                "pred_total_sum_gtgrid": pred_metrics_gtgrid["total_sum"],
                "pred_in_body_sum_gtgrid": pred_metrics_gtgrid["in_body_sum"],
                "pred_outside_body_sum_gtgrid": pred_metrics_gtgrid["out_body_sum"],
                "pred_in_gt_active_sum_gtgrid": pred_metrics_gtgrid["in_gt_active_sum"],
                "pred_not_gt_active_sum_gtgrid": pred_metrics_gtgrid["not_gt_active_sum"],
                "pred_in_body_frac_gtgrid": pred_metrics_gtgrid["in_body_frac"],
                "pred_outside_body_frac_gtgrid": pred_metrics_gtgrid["out_body_frac"],
                "pred_in_gt_active_frac_gtgrid": pred_metrics_gtgrid["in_gt_active_frac"],
                "pred_not_gt_active_frac_gtgrid": pred_metrics_gtgrid["not_gt_active_frac"],
                "pred_outside_mask_sum": pred_outside_mask_sum,
                "pred_outside_mask_frac": pred_outside_mask_frac,
                "A_gt_vol": A_gt_vol,
                "A_pred_vol": A_pred_vol,
                "A_pred_native": A_pred_native,
                "timestamp": timestamp,
                "git_hash": git_hash,
                "config_path": config_path,
                "proj_status": proj_status,
                "proj_domain": proj_domain,
                "proj_is_normalized": proj_is_normalized,
                "proj_norm_factor": proj_norm_factor,
                "calibration_scale": calibration_scale,
                "calibrate_scale_enabled": bool(args.calibrate_scale),
            },
        )
        proj_metrics_target_rows.append(
            {
                "phantom": pid,
                "target_mode": "clean_counts",
                "proj_mae_counts": float(proj_mae_counts_clean),
                "proj_poisson_dev_counts": float(proj_poisson_dev_counts_clean),
                "proj_status": str(proj_status),
                "proj_domain": str(proj_domain),
            }
        )
        proj_metrics_target_rows.append(
            {
                "phantom": pid,
                "target_mode": "noisy_counts",
                "proj_mae_counts": float(proj_mae_counts_noisy),
                "proj_poisson_dev_counts": float(proj_poisson_dev_counts_noisy),
                "proj_status": str(proj_status),
                "proj_domain": str(proj_domain),
            }
        )
        run_proj_metrics_records.append(
            {
                "phantom_id": pid,
                "proj_mae_counts_clean": float(proj_mae_counts_clean),
                "proj_poisson_dev_counts_clean": float(proj_poisson_dev_counts_clean),
                "proj_mae_counts_noisy": float(proj_mae_counts_noisy),
                "proj_poisson_dev_counts_noisy": float(proj_poisson_dev_counts_noisy),
            }
        )
        _LOG.info(
            "processed phantom %s (metrics -> %s; projections status=%s domain=%s)",
            pid,
            patient_dir / "metrics.json",
            proj_status,
            proj_domain,
        )

        if args.timing:
            block_report = " ".join(
                f"{name}={timer.blocks.get(name, 0.0):.3f}s" for name in BLOCK_NAMES
            )
            print(f"[timing][phantom={pid}] {block_report}", flush=True)
            for name, duration in timer.blocks.items():
                block_totals[name].append(duration)

    proj_targets_csv = out_dir / "proj_metrics_targets.csv"
    with proj_targets_csv.open("w", newline="") as f:
        fieldnames = ["phantom", "target_mode", "proj_mae_counts", "proj_poisson_dev_counts", "proj_status", "proj_domain"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in proj_metrics_target_rows:
            writer.writerow(row)

    def _mean_std(values: list[float]) -> dict[str, float]:
        vals = [float(v) for v in values if np.isfinite(v)]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
            "n": int(arr.size),
        }

    clean_mae_vals = [float(r["proj_mae_counts_clean"]) for r in run_proj_metrics_records]
    clean_dev_vals = [float(r["proj_poisson_dev_counts_clean"]) for r in run_proj_metrics_records]
    noisy_mae_vals = [float(r["proj_mae_counts_noisy"]) for r in run_proj_metrics_records]
    noisy_dev_vals = [float(r["proj_poisson_dev_counts_noisy"]) for r in run_proj_metrics_records]

    test_metrics_payload = {
        "run": {
            "run_dir": str(run_dir),
            "out_dir": str(out_dir),
            "proj_metrics_target_mode": str(args.proj_metrics_target),
            "n_phantoms": int(len(run_proj_metrics_records)),
        },
        "per_phantom_projection_metrics": run_proj_metrics_records,
        "projection_metrics_aggregate": {
            "clean_counts": {
                "proj_mae_counts": _mean_std(clean_mae_vals),
                "proj_poisson_dev_counts": _mean_std(clean_dev_vals),
            },
            "noisy_counts": {
                "proj_mae_counts": _mean_std(noisy_mae_vals),
                "proj_poisson_dev_counts": _mean_std(noisy_dev_vals),
            },
        },
    }
    write_metrics_json(out_dir / "test_metrics.json", test_metrics_payload)

    total_runtime = time.perf_counter() - start_total
    log_summary(total_runtime, block_totals, args)


def log_summary(total_runtime: float, block_totals: dict[str, list[float]], args):
    if not (args.timing or args.profile):
        return
    print(f"[summary] total_runtime={total_runtime:.2f}s", flush=True)
    if args.timing:
        for name in BLOCK_NAMES:
            durations = block_totals.get(name)
            if not durations:
                continue
            mean = sum(durations) / len(durations)
            mx = max(durations)
            print(
                f"[summary][block={name}] mean={mean:.3f}s max={mx:.3f}s",
                flush=True,
            )
    active_flags = [name for name in FLAG_ATTRS if getattr(args, name)]
    if active_flags:
        print(f"[summary] active_flags={active_flags}", flush=True)
    else:
        print("[summary] active_flags=none", flush=True)


def main():
    args = parse_args()
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        run_postprocessing(args)
        profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
        stats.print_stats(args.profile_topk)
        print(stream.getvalue())
    else:
        run_postprocessing(args)


if __name__ == "__main__":
    main()
