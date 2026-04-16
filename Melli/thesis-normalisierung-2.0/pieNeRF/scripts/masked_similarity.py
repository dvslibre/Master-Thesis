#!/usr/bin/env python3
"""Pairwise masked similarity for predicted/GT activity volumes.

Examples:
  python scripts/masked_similarity.py \
    --pred-root results_spect/test_slices \
    --pred-glob '*/activity_pred.npy' \
    --gt-path-pattern 'data/{phantom}/out/act.npy' \
    --mask-path-pattern 'data/{phantom}/out/mask.npy' \
    --gt-threshold 1e-6 \
    --include-gt \
    --out-csv results_spect/masked_similarity.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-8


@dataclass
class PhantomData:
    phantom_id: str
    pred: np.ndarray
    gt: Optional[np.ndarray]
    mask: Optional[np.ndarray]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise masked similarity for activity volumes.")
    p.add_argument("--pred-root", type=Path, required=True, help="Root folder containing predicted volumes.")
    p.add_argument(
        "--pred-glob",
        type=str,
        default="*/activity_pred.npy",
        help="Glob pattern under --pred-root for predicted volumes.",
    )
    p.add_argument(
        "--gt-path-pattern",
        type=str,
        default="data/{phantom}/out/act.npy",
        help="Path template for GT volumes. Uses {phantom}.",
    )
    p.add_argument(
        "--mask-path-pattern",
        type=str,
        default="data/{phantom}/out/mask.npy",
        help="Path template for body masks. Uses {phantom}.",
    )
    p.add_argument("--gt-threshold", type=float, default=1e-6, help="Threshold for GT-active mask (gt > tau).")
    p.add_argument(
        "--mask-modes",
        type=str,
        default="body,gt_active,union,intersection",
        help="Comma-separated mask modes: body, gt_active, union, intersection.",
    )
    p.add_argument("--include-gt", action="store_true", help="Also compute pairwise metrics on GT volumes.")
    p.add_argument(
        "--resample-space",
        type=str,
        default="pred",
        choices=["pred", "gt"],
        help=(
            "Grid used for pairwise comparison: "
            "'pred' resamples GT/mask to prediction grid; "
            "'gt' resamples predictions to GT grid (falls back to pred grid if GT missing)."
        ),
    )
    p.add_argument("--out-csv", type=Path, default=None, help="Optional CSV output path.")
    return p.parse_args()


def _resize_array(arr: np.ndarray, target_shape: tuple[int, int, int], mode: str) -> np.ndarray:
    if tuple(arr.shape) == tuple(target_shape):
        return arr
    t = torch.from_numpy(arr.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    if mode == "nearest":
        out = F.interpolate(t, size=target_shape, mode="nearest")
    else:
        out = F.interpolate(t, size=target_shape, mode="trilinear", align_corners=False)
    return out.squeeze(0).squeeze(0).cpu().numpy()


def _load_optional(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.load(path)


def _load_data(args: argparse.Namespace) -> list[PhantomData]:
    pred_paths = sorted(args.pred_root.glob(args.pred_glob))
    if not pred_paths:
        raise FileNotFoundError(f"No predicted volumes found: root={args.pred_root} glob={args.pred_glob}")

    rows: list[PhantomData] = []
    for p in pred_paths:
        phantom_id = p.parent.name
        pred = np.load(p).astype(np.float32, copy=False)
        gt = _load_optional(Path(args.gt_path_pattern.format(phantom=phantom_id)))
        if gt is not None:
            gt = gt.astype(np.float32, copy=False)
        mask = _load_optional(Path(args.mask_path_pattern.format(phantom=phantom_id)))
        if mask is not None:
            mask = np.rint(mask).astype(np.int32, copy=False)
        rows.append(PhantomData(phantom_id=phantom_id, pred=pred, gt=gt, mask=mask))
    n_missing_gt = sum(int(r.gt is None) for r in rows)
    n_missing_mask = sum(int(r.mask is None) for r in rows)
    if n_missing_gt > 0:
        print(f"[warn] Missing GT for {n_missing_gt}/{len(rows)} phantoms.")
    if n_missing_mask > 0:
        print(f"[warn] Missing body mask for {n_missing_mask}/{len(rows)} phantoms.")
    return rows


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return num / (den + EPS)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - float(a.mean())
    b0 = b - float(b.mean())
    den = float(a0.std() * b0.std())
    if den <= 0:
        return float("nan")
    return float(np.mean(a0 * b0) / (den + EPS))


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + EPS))


def _best_scale_rel_l2(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    denom = float(np.dot(a, a))
    alpha = float(np.dot(a, b) / (denom + EPS))
    rel = float(np.linalg.norm(alpha * a - b) / (np.linalg.norm(b) + EPS))
    return alpha, rel


def _build_pair_masks(
    a: PhantomData,
    b: PhantomData,
    target_shape: tuple[int, int, int],
    gt_thr: float,
) -> Dict[str, np.ndarray]:
    if a.mask is None:
        body_a = np.ones(target_shape, dtype=bool)
    else:
        body_a = _resize_array(a.mask.astype(np.float32), target_shape, mode="nearest") != 0

    if b.mask is None:
        body_b = np.ones(target_shape, dtype=bool)
    else:
        body_b = _resize_array(b.mask.astype(np.float32), target_shape, mode="nearest") != 0

    body = body_a & body_b

    active_a = np.zeros(target_shape, dtype=bool)
    if a.gt is not None:
        gt_a = _resize_array(a.gt, target_shape, mode="trilinear")
        active_a = gt_a > gt_thr

    active_b = np.zeros(target_shape, dtype=bool)
    if b.gt is not None:
        gt_b = _resize_array(b.gt, target_shape, mode="trilinear")
        active_b = gt_b > gt_thr

    gt_active = active_a | active_b
    union = body | gt_active
    intersection = body & gt_active

    return {
        "body": body,
        "gt_active": gt_active,
        "union": union,
        "intersection": intersection,
    }


def _iter_rows(
    items: list[PhantomData],
    include_gt: bool,
    mask_modes: Iterable[str],
    gt_thr: float,
    resample_space: str,
):
    for a, b in itertools.combinations(items, 2):
        if resample_space == "gt":
            if a.gt is not None:
                target_shape = tuple(a.gt.shape)
            elif b.gt is not None:
                target_shape = tuple(b.gt.shape)
            else:
                target_shape = tuple(a.pred.shape)
        else:
            target_shape = tuple(a.pred.shape)

        pred_a = _resize_array(a.pred, target_shape, mode="trilinear").astype(np.float32, copy=False)
        pred_b = _resize_array(b.pred, target_shape, mode="trilinear").astype(np.float32, copy=False)

        gt_a = _resize_array(a.gt, target_shape, mode="trilinear").astype(np.float32, copy=False) if a.gt is not None else None
        gt_b = _resize_array(b.gt, target_shape, mode="trilinear").astype(np.float32, copy=False) if b.gt is not None else None

        masks = _build_pair_masks(a, b, target_shape, gt_thr)

        for source, va, vb in (("pred", pred_a, pred_b), ("gt", gt_a, gt_b)):
            if source == "gt" and not include_gt:
                continue
            if va is None or vb is None:
                continue
            for mode in mask_modes:
                m = masks.get(mode)
                if m is None:
                    continue
                n = int(m.sum())
                if n < 2:
                    continue
                x = va[m].astype(np.float64, copy=False).reshape(-1)
                y = vb[m].astype(np.float64, copy=False).reshape(-1)
                alpha, best_rel = _best_scale_rel_l2(x, y)
                yield {
                    "source": source,
                    "mask_mode": mode,
                    "id_a": a.phantom_id,
                    "id_b": b.phantom_id,
                    "n_vox": n,
                    "cosine": _cosine(x, y),
                    "corr": _corr(x, y),
                    "rel_l2": _rel_l2(x, y),
                    "best_scale_rel_l2": best_rel,
                    "best_scale_alpha": alpha,
                }


def _print_summary(rows: list[dict]) -> None:
    if not rows:
        print("No rows.")
        return
    print("source mask_mode pairs cosine_mean corr_mean relL2_mean bestRelL2_mean")
    keys = sorted({(r["source"], r["mask_mode"]) for r in rows})
    for source, mode in keys:
        sub = [r for r in rows if r["source"] == source and r["mask_mode"] == mode]
        if not sub:
            continue
        c = np.array([r["cosine"] for r in sub], dtype=np.float64)
        k = np.array([r["corr"] for r in sub], dtype=np.float64)
        l2 = np.array([r["rel_l2"] for r in sub], dtype=np.float64)
        bl2 = np.array([r["best_scale_rel_l2"] for r in sub], dtype=np.float64)
        print(
            f"{source:>6s} {mode:>12s} {len(sub):5d} "
            f"{np.nanmean(c):.4f} {np.nanmean(k):.4f} {np.nanmean(l2):.4f} {np.nanmean(bl2):.4f}"
        )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "source",
        "mask_mode",
        "id_a",
        "id_b",
        "n_vox",
        "cosine",
        "corr",
        "rel_l2",
        "best_scale_rel_l2",
        "best_scale_alpha",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    args = _parse_args()
    mask_modes = [m.strip() for m in str(args.mask_modes).split(",") if m.strip()]
    valid = {"body", "gt_active", "union", "intersection"}
    unknown = [m for m in mask_modes if m not in valid]
    if unknown:
        raise ValueError(f"Unknown mask mode(s): {unknown}. Valid: {sorted(valid)}")

    items = _load_data(args)
    rows = list(_iter_rows(items, args.include_gt, mask_modes, float(args.gt_threshold), args.resample_space))

    print(f"Loaded {len(items)} phantoms, computed {len(rows)} pairwise rows.")
    _print_summary(rows)

    if args.out_csv is not None:
        _write_csv(args.out_csv, rows)
        print(f"Saved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
