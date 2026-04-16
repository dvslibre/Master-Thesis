#!/usr/bin/env python3
"""Baseline sanity checker for run outputs.

Usage:
  python tools/baseline_sanity_check.py --run_dir /path/to/run_661 --device cpu
  python tools/baseline_sanity_check.py --run_dir /path/to/run_661 --split_json /path/to/split.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import re
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic baseline sanity checks for pieNeRF run outputs.")
    p.add_argument("--run_dir", required=True, type=Path)
    p.add_argument("--split_json", default=None, type=Path)
    p.add_argument("--checkpoint", default=None, type=Path)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    m = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            m.update(b)
    return m.hexdigest()


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den <= 1e-18:
        return float("nan")
    return float(np.dot(x, y) / den)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(d * d))


def _stats(arr: np.ndarray) -> dict[str, float]:
    x = np.asarray(arr, dtype=np.float64)
    return {
        "min": float(np.min(x)),
        "mean": float(np.mean(x)),
        "max": float(np.max(x)),
        "std": float(np.std(x)),
        "sum": float(np.sum(x)),
        "pct_zeros": float(np.mean(x == 0.0) * 100.0),
    }


def _downsample_3d(arr: np.ndarray, stride: int = 4) -> np.ndarray:
    if arr.ndim != 3:
        return arr
    return arr[::stride, ::stride, ::stride]


def _apply_yx_orientation_transform(vol: np.ndarray, k_rot90: int, do_fliplr: bool, do_flipud: bool) -> np.ndarray:
    out = np.rot90(vol, k=int(k_rot90), axes=(-2, -1))
    if do_flipud:
        out = np.flip(out, axis=-2)
    if do_fliplr:
        out = np.flip(out, axis=-1)
    return out


def _orientation_transform_label(k_rot90: int, do_fliplr: bool, do_flipud: bool) -> str:
    parts = [f"rot90(k={int(k_rot90)}, axes=(Y,X))"]
    if do_fliplr:
        parts.append("fliplr")
    if do_flipud:
        parts.append("flipud")
    return " + ".join(parts)


def _center_of_mass_vox(arr: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(arr, dtype=np.float64), 0.0, None)
    total = float(np.sum(w))
    if total <= 1e-18:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    coords = np.indices(w.shape, dtype=np.float64)
    return np.array([float(np.sum(coords[i] * w) / total) for i in range(w.ndim)], dtype=np.float64)


def _com_dist_vox(a: np.ndarray, b: np.ndarray) -> float:
    ca = _center_of_mass_vox(a)
    cb = _center_of_mass_vox(b)
    if not np.all(np.isfinite(ca)) or not np.all(np.isfinite(cb)):
        return float("inf")
    return float(np.linalg.norm(ca - cb))


def _best_orientation_candidate(gt_roi: np.ndarray, pred_roi: np.ndarray) -> dict[str, Any]:
    flip_modes = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    results: list[dict[str, Any]] = []
    for k in (0, 1, 2, 3):
        for do_fliplr, do_flipud in flip_modes:
            cand = _apply_yx_orientation_transform(pred_roi, k, do_fliplr, do_flipud)
            if cand.shape != gt_roi.shape:
                continue
            ncc = _ncc(gt_roi, cand)
            com_dist = _com_dist_vox(gt_roi, cand)
            results.append(
                {
                    "k_rot90": int(k),
                    "fliplr": bool(do_fliplr),
                    "flipud": bool(do_flipud),
                    "label": _orientation_transform_label(k, do_fliplr, do_flipud),
                    "ncc": float(ncc),
                    "com_dist_vox": float(com_dist),
                }
            )
    if not results:
        raise RuntimeError("no valid orientation candidates for 3D orientation check")

    def _rank(item: dict[str, Any]) -> tuple[float, float]:
        ncc = item["ncc"]
        com_dist = item["com_dist_vox"]
        ncc_rank = ncc if np.isfinite(ncc) else -np.inf
        com_rank = -com_dist if np.isfinite(com_dist) else -np.inf
        return (ncc_rank, com_rank)

    best = max(results, key=_rank)
    identity = next(
        (item for item in results if item["k_rot90"] == 0 and not item["fliplr"] and not item["flipud"]),
        None,
    )
    if identity is None:
        raise RuntimeError("identity orientation candidate missing")
    return {"best": best, "identity": identity}


def _extract_cli_from_command(command_path: Path) -> tuple[list[str], dict[str, Any]]:
    raw = command_path.read_text().strip()
    toks = shlex.split(raw)
    i = None
    for j, tok in enumerate(toks):
        if tok.endswith("train_emission.py"):
            i = j
            break
    if i is None:
        raise RuntimeError(f"train_emission.py not found in {command_path}")
    cli = toks[i + 1 :]
    info: dict[str, Any] = {}
    keys = {
        "--config": "config",
        "--proj-target-source": "proj_target_source",
        "--proj-loss-type": "proj_loss_type",
        "--poisson-rate-mode": "poisson_rate_mode",
        "--proj-scale-source": "proj_scale_source",
        "--z-enc-alpha": "z_enc_alpha",
    }
    for k, outk in keys.items():
        if k in cli:
            idx = cli.index(k)
            if idx + 1 < len(cli):
                info[outk] = cli[idx + 1]
    info["hybrid"] = "--hybrid" in cli
    pa_flag = None
    for k in ["--pa-xflip", "--pa_xflip"]:
        if k in cli:
            idx = cli.index(k)
            pa_flag = cli[idx + 1] if idx + 1 < len(cli) and not cli[idx + 1].startswith("--") else "true"
            break
    info["pa_xflip"] = pa_flag
    return cli, info


def _choose_checkpoint(run_dir: Path, explicit: Path | None) -> tuple[Path, int]:
    if explicit is not None:
        cp = explicit if explicit.is_absolute() else (run_dir / explicit)
        if not cp.exists():
            raise FileNotFoundError(cp)
        m = re.search(r"step(\d+)", cp.name)
        step = int(m.group(1)) if m else -1
        return cp, step
    ckpts = sorted((run_dir / "checkpoints").glob("checkpoint_step*.pt"))
    if not ckpts:
        fallback = run_dir / "checkpoints" / "checkpoint_step05000.pt"
        if fallback.exists():
            return fallback, 5000
        raise FileNotFoundError(f"No checkpoints found in {(run_dir / 'checkpoints')}")
    def _step(p: Path) -> int:
        m = re.search(r"step(\d+)", p.name)
        return int(m.group(1)) if m else -1
    cp = max(ckpts, key=_step)
    return cp, _step(cp)


def _load_train_emission(repo_root: Path):
    te_path = repo_root / "train_emission.py"
    spec = importlib.util.spec_from_file_location("baseline_train_emission", str(te_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load train_emission from {te_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["baseline_train_emission"] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _parse_latent_alive(train_log_csv: Path, hybrid_enabled: bool) -> tuple[bool, list[str]]:
    warns: list[str] = []
    if not train_log_csv.exists():
        warns.append(f"train_log.csv not found at {train_log_csv}")
        return False, warns
    with train_log_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        std_cols = {
            "z_enc_std": None,
            "z_proj_std": None,
            "z_final_std": None,
        }
        for key in std_cols:
            for h in headers:
                if h is None:
                    continue
                h_low = h.lower()
                if key in h_low:
                    std_cols[key] = h
                    break
        if any(v is None for v in std_cols.values()):
            warns.append(
                "latent std columns not found in train_log.csv "
                f"(needed z_enc/z_proj/z_final std); found headers={headers[:20]}"
            )
            return False, warns
        alive = False
        for row in reader:
            try:
                z_enc_std = float(row[std_cols["z_enc_std"]])  # type: ignore[index]
                z_proj_std = float(row[std_cols["z_proj_std"]])  # type: ignore[index]
                z_final_std = float(row[std_cols["z_final_std"]])  # type: ignore[index]
            except Exception:
                continue
            if z_enc_std > 1e-6 and z_proj_std > 1e-6 and z_final_std > 1e-6:
                if hybrid_enabled:
                    ratio = z_final_std / max(z_enc_std, 1e-12)
                    if ratio > 1.1 or ratio < 0.9:
                        alive = True
                        break
                else:
                    alive = True
                    break
        return alive, warns


def _resolve_config_path(config_raw: str | None, run_dir: Path, repo_root: Path) -> Path | None:
    if not config_raw:
        return None
    p = Path(config_raw)
    if p.is_absolute():
        return p
    cand1 = run_dir / p
    if cand1.exists():
        return cand1
    cand2 = repo_root / p
    if cand2.exists():
        return cand2
    return cand1


def _load_gt_counts_for_pid(
    pid: str,
    run_dir: Path,
    dataset_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    # Prefer audit bundle
    audit_dirs = sorted(run_dir.glob("postproc_audit*"))
    for ad in audit_dirs:
        if pid in ad.name:
            ap_p = ad / "ap_gt_counts.npy"
            pa_p = ad / "pa_gt_counts.npy"
            if ap_p.exists() and pa_p.exists():
                return np.load(ap_p).astype(np.float32), np.load(pa_p).astype(np.float32)
    if dataset_cache is not None and pid in dataset_cache:
        return dataset_cache[pid]
    return None


def _collect_dataset_counts_by_pid(config_path: Path, repo_root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    te = _load_train_emission(repo_root)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.setdefault("data", {})
    radius_xyz = data_cfg.get("radius_xyz_cm")
    if bool(data_cfg.get("auto_near_far_from_radius", True)) and radius_xyz is not None:
        data_cfg["near"] = 0.0
        data_cfg["far"] = 2.0 * float(radius_xyz[2])
    dset, hwfr, _ = te.get_data(cfg)
    cfg["data"]["hwfr"] = hwfr
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i in range(len(dset)):
        pid = dset.get_patient_id(i) if hasattr(dset, "get_patient_id") else None
        if pid is None:
            continue
        sample = dset[i]
        ap_c = sample.get("ap_counts")
        pa_c = sample.get("pa_counts")
        if ap_c is None or pa_c is None or ap_c.numel() == 0 or pa_c.numel() == 0:
            continue
        out[str(pid)] = (
            ap_c.squeeze(0).cpu().numpy().astype(np.float32),
            pa_c.squeeze(0).cpu().numpy().astype(np.float32),
        )
    return out


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    split_json = args.split_json.resolve() if args.split_json else (run_dir / "split.json")
    repo_root = Path(__file__).resolve().parents[1]

    np.random.seed(0)
    torch.manual_seed(0)

    fail_reasons: list[str] = []
    warnings: list[str] = []

    print("=== Baseline Sanity Check ===")
    print(f"run_dir={run_dir}")
    print(f"split_json={split_json}")

    # 1) Split integrity
    split = _load_json(split_json)
    train_ids = set(split.get("train_ids", []))
    val_ids = set(split.get("val_ids", []))
    test_ids = set(split.get("test_ids", []))
    ov_tv = sorted(train_ids.intersection(val_ids))
    ov_tt = sorted(train_ids.intersection(test_ids))
    ov_vt = sorted(val_ids.intersection(test_ids))
    print("\n[1] Split integrity")
    print(f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    print(f"overlap(train,val)={ov_tv}")
    print(f"overlap(train,test)={ov_tt}")
    print(f"overlap(val,test)={ov_vt}")
    if ov_tv or ov_tt or ov_vt:
        fail_reasons.append("split overlap detected")

    # 2) Checkpoint consistency + command parse
    print("\n[2] Checkpoint consistency")
    command_path = run_dir / "command.sh"
    if not command_path.exists():
        fail_reasons.append(f"missing command.sh at {command_path}")
        cmd_info = {}
    else:
        _, cmd_info = _extract_cli_from_command(command_path)
        print(
            "parsed flags:",
            {
                "config": cmd_info.get("config"),
                "proj_target_source": cmd_info.get("proj_target_source"),
                "proj_loss_type": cmd_info.get("proj_loss_type"),
                "poisson_rate_mode": cmd_info.get("poisson_rate_mode"),
                "proj_scale_source": cmd_info.get("proj_scale_source"),
                "z_enc_alpha": cmd_info.get("z_enc_alpha"),
                "pa_xflip": cmd_info.get("pa_xflip"),
                "hybrid": cmd_info.get("hybrid"),
            },
        )
    try:
        ckpt, step = _choose_checkpoint(run_dir, args.checkpoint)
        print(f"checkpoint={ckpt} step={step}")
    except Exception as e:
        fail_reasons.append(f"checkpoint selection failed: {e}")

    # 3) No overwrite / identical predictions
    print("\n[3] Test-slice prediction uniqueness")
    pred_paths = sorted((run_dir / "test_slices").glob("phantom_*/activity_pred.npy"))
    if len(pred_paths) < 2:
        pred_paths = sorted(run_dir.glob("*_slices/phantom_*/activity_pred.npy"))
    print(f"found_activity_pred_files={len(pred_paths)}")
    if len(pred_paths) < 2:
        fail_reasons.append("fewer than 2 activity_pred.npy files found")
    pred_data: list[tuple[str, Path, str, np.ndarray]] = []
    for p in pred_paths:
        pid = p.parent.name
        h = _md5_file(p)
        arr = np.load(p).astype(np.float32)
        st = _stats(arr)
        print(
            f"{pid}: shape={arr.shape} min={st['min']:.3e} mean={st['mean']:.3e} max={st['max']:.3e} "
            f"std={st['std']:.3e} sum={st['sum']:.3e} pct_zeros={st['pct_zeros']:.2f}% md5={h}"
        )
        pred_data.append((pid, p, h, arr))
    for (pid_a, _, h_a, a), (pid_b, _, h_b, b) in itertools.combinations(pred_data, 2):
        max_abs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        ncc = _ncc(_downsample_3d(a), _downsample_3d(b))
        print(f"pair {pid_a} vs {pid_b}: L_inf={max_abs:.6e} NCC(ds4)={ncc:.6f} md5_equal={h_a == h_b}")
        dup = (h_a == h_b) or (max_abs == 0.0) or ((ncc > 0.99999) and (max_abs < 1e-6))
        if dup:
            fail_reasons.append(f"possible duplicate/overwrite detected: {pid_a} vs {pid_b}")

    # 4) Encoder/latent alive via logs
    print("\n[4] Encoder/latent alive")
    hybrid_enabled = bool(cmd_info.get("hybrid", False))
    alive, latent_warns = _parse_latent_alive(run_dir / "train_log.csv", hybrid_enabled)
    warnings.extend(latent_warns)
    if alive:
        print("latent_alive=PASS")
    else:
        if latent_warns:
            print("latent_alive=WARN (stats unavailable)")
        else:
            print("latent_alive=FAIL")
            fail_reasons.append("latent stats found but did not pass alive condition")

    # 5) Train-forward projection compatibility
    print("\n[5] Train-forward projection compatibility")
    proj_pred_paths = sorted(run_dir.glob("postproc*/phantom_*/proj_train/ap_pred.npy"))
    if not proj_pred_paths:
        warnings.append("no proj_train outputs found under run_dir/postproc*/phantom_*/proj_train")
        print("proj_train=WARN not found")
    else:
        config_path = _resolve_config_path(cmd_info.get("config"), run_dir, repo_root)
        dataset_counts: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        if config_path is not None and config_path.exists():
            try:
                dataset_counts = _collect_dataset_counts_by_pid(config_path, repo_root)
            except Exception as e:
                warnings.append(f"could not load dataset counts via train_emission.get_data: {e}")
        else:
            warnings.append(f"config path missing/unresolved for dataset counts: {config_path}")

        test_list = sorted(test_ids) if test_ids else None
        at_least_one_ncc_pass = False
        swap_margin = 0.02
        for ap_path in proj_pred_paths:
            pid = ap_path.parents[1].name
            if test_list is not None and pid not in test_ids:
                continue
            pa_path = ap_path.parent / "pa_pred.npy"
            if not pa_path.exists():
                warnings.append(f"missing {pa_path}")
                continue
            pred_ap = np.load(ap_path).astype(np.float32)
            pred_pa = np.load(pa_path).astype(np.float32)
            st_ap = _stats(pred_ap)
            st_pa = _stats(pred_pa)
            if st_ap["std"] < 1e-12:
                fail_reasons.append(f"{pid} proj_train AP constant/near-constant")
            if st_pa["std"] < 1e-12:
                fail_reasons.append(f"{pid} proj_train PA constant/near-constant")

            gt_ap_path = ap_path.parents[1] / "proj" / "gt_ap.npy"
            gt_pa_path = ap_path.parents[1] / "proj" / "gt_pa.npy"
            if gt_ap_path.exists() and gt_pa_path.exists():
                gt = (
                    np.load(gt_ap_path).astype(np.float32),
                    np.load(gt_pa_path).astype(np.float32),
                )
            else:
                gt = _load_gt_counts_for_pid(pid, run_dir, dataset_counts)
            if gt is None:
                warnings.append(f"missing GT counts for {pid}; skipping NCC/MSE")
                continue
            gt_ap, gt_pa = gt
            if pred_ap.shape != gt_ap.shape or pred_pa.shape != gt_pa.shape:
                fail_reasons.append(
                    f"{pid} shape mismatch pred/gt: pred_ap={pred_ap.shape} gt_ap={gt_ap.shape} "
                    f"pred_pa={pred_pa.shape} gt_pa={gt_pa.shape}"
                )
                continue
            ncc_ap = _ncc(pred_ap, gt_ap)
            ncc_pa = _ncc(pred_pa, gt_pa)
            ncc_ap_gt_pa = _ncc(pred_ap, gt_pa)
            ncc_pa_gt_ap = _ncc(pred_pa, gt_ap)
            score_same = ncc_ap + ncc_pa
            score_swap = ncc_ap_gt_pa + ncc_pa_gt_ap
            mse_ap = _mse(pred_ap, gt_ap)
            mse_pa = _mse(pred_pa, gt_pa)
            eps = 1e-12
            num = float(np.dot(pred_ap.astype(np.float64).ravel(), gt_ap.astype(np.float64).ravel()) +
                        np.dot(pred_pa.astype(np.float64).ravel(), gt_pa.astype(np.float64).ravel()))
            den = float(np.dot(pred_ap.astype(np.float64).ravel(), pred_ap.astype(np.float64).ravel()) +
                        np.dot(pred_pa.astype(np.float64).ravel(), pred_pa.astype(np.float64).ravel()) + eps)
            alpha = num / den
            pred_ap_s = pred_ap * np.float32(alpha)
            pred_pa_s = pred_pa * np.float32(alpha)
            mse_ap_s = _mse(pred_ap_s, gt_ap)
            mse_pa_s = _mse(pred_pa_s, gt_pa)
            print(
                f"{pid}: NCC(AP/PA)={ncc_ap:.4f}/{ncc_pa:.4f} "
                f"NCC_cross(AP->PA,PA->AP)={ncc_ap_gt_pa:.4f}/{ncc_pa_gt_ap:.4f} "
                f"score_same={score_same:.4f} score_swap={score_swap:.4f} "
                f"MSE(AP/PA)={mse_ap:.6e}/{mse_pa:.6e} "
                f"alpha_global={alpha:.6e} "
                f"MSE_after(AP/PA)={mse_ap_s:.6e}/{mse_pa_s:.6e}"
            )
            if math.isfinite(score_same) and math.isfinite(score_swap):
                if score_swap > score_same + swap_margin:
                    fail_reasons.append(
                        f"{pid} proj_train AP/PA appears swapped: score_swap={score_swap:.4f} "
                        f"> score_same={score_same:.4f} + {swap_margin:.2f}"
                    )
                elif score_swap > score_same:
                    warnings.append(
                        f"{pid} proj_train swap score slightly higher: score_swap={score_swap:.4f} "
                        f"vs score_same={score_same:.4f}"
                    )
            else:
                warnings.append(f"{pid} non-finite NCC score(s) for AP/PA swap check")

            # Optional 3D GT-grid orientation sanity check (if postprocessing dumps exist).
            patient_dir = ap_path.parents[1]
            metrics_path = patient_dir / "metrics.json"
            orient_applied = None
            orient_transform = "unknown"
            if metrics_path.exists():
                try:
                    metrics_payload = _load_json(metrics_path)
                    metrics_block = metrics_payload.get("metrics", {})
                    orient_applied = metrics_block.get("orientation_fix_applied")
                    orient_transform = str(metrics_block.get("orientation_fix_transform", "unknown"))
                except Exception as e:
                    warnings.append(f"{pid} failed to parse orientation metadata from {metrics_path}: {e}")
            if orient_applied is True:
                print(f"{pid}: orientation_fix=INFO applied transform={orient_transform}")

            orient_dir = patient_dir / "orientation_debug"
            gt_roi_path = orient_dir / "gt_roi.npy"
            pred_roi_path = orient_dir / "pred_roi_gtgrid.npy"
            if gt_roi_path.exists() and pred_roi_path.exists():
                try:
                    gt_roi = np.load(gt_roi_path).astype(np.float32)
                    pred_roi = np.load(pred_roi_path).astype(np.float32)
                    if gt_roi.shape != pred_roi.shape:
                        warnings.append(
                            f"{pid} orientation_debug shape mismatch gt_roi={gt_roi.shape} pred_roi={pred_roi.shape}"
                        )
                    else:
                        orient_eval = _best_orientation_candidate(gt_roi, pred_roi)
                        best = orient_eval["best"]
                        identity = orient_eval["identity"]
                        print(
                            f"{pid}: 3D-orient NCC(identity/best)={identity['ncc']:.4f}/{best['ncc']:.4f} "
                            f"CoMdist(identity/best)={identity['com_dist_vox']:.4f}/{best['com_dist_vox']:.4f} "
                            f"best={best['label']}"
                        )
                        if orient_applied is not True:
                            if math.isfinite(best["ncc"]) and math.isfinite(identity["ncc"]) and best["ncc"] > identity["ncc"] + 0.05:
                                warnings.append(
                                    f"{pid} orientation not applied, but transformed NCC is much better: "
                                    f"best={best['ncc']:.4f} identity={identity['ncc']:.4f} transform={best['label']}"
                                )
                except Exception as e:
                    warnings.append(f"{pid} orientation_debug evaluation failed: {e}")
            else:
                warnings.append(
                    f"{pid} no orientation_debug volumes found at {orient_dir} (run postprocessing with --save-orientation-debug-volumes)"
                )
            if ncc_ap > 0.2 and ncc_pa > 0.2:
                at_least_one_ncc_pass = True
        if not at_least_one_ncc_pass:
            fail_reasons.append("no phantom with NCC(AP)>0.2 and NCC(PA)>0.2 in proj_train")

    # Final summary
    passed = len(fail_reasons) == 0
    status = "PASS" if passed else "FAIL"
    print("\n=== Summary ===")
    print(f"BASELINE_SANITY: {status}")
    print(f"FAIL_REASONS: {fail_reasons}")
    print(f"WARNINGS: {warnings}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
