#!/usr/bin/env python3
"""Offline AP/PA orientation checker from existing projection files."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


STEP_RE = re.compile(r"step[_-]?(\d+)", re.IGNORECASE)
PHANTOM_RE_TPL = r"(?:phantom[_-]?{pid}\b|\b{pid}\b)"
TOKEN_AP = re.compile(r"(^|[^a-z0-9])ap([^a-z0-9]|$)", re.IGNORECASE)
TOKEN_PA = re.compile(r"(^|[^a-z0-9])pa([^a-z0-9]|$)", re.IGNORECASE)
TOKEN_PRED = re.compile(r"(pred|prediction|recon|output)", re.IGNORECASE)
TOKEN_GT = re.compile(r"(gt|target|truth|reference|ref)\b", re.IGNORECASE)


@dataclass
class FileCandidate:
    path: Path
    view: str
    kind: str
    ext: str
    step: Optional[int]
    score: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check AP/PA swap/flip orientation hypotheses from existing result files.")
    p.add_argument("--results_dir", required=True, type=Path, help="Results root directory (recursive search).")
    p.add_argument("--phantom_id", required=True, help="Phantom id (e.g. 34 or phantom_34).")
    p.add_argument("--step", type=int, default=None, help="Optional training step to prefer (e.g. 2000).")
    p.add_argument("--metric", choices=["ncc", "mse"], default="ncc", help="Similarity metric used for ranking.")
    return p.parse_args()


def infer_step(path: Path) -> Optional[int]:
    m = STEP_RE.search(path.as_posix())
    if m:
        return int(m.group(1))
    return None


def infer_view(path: Path) -> Optional[str]:
    s = path.as_posix().lower()
    ap = TOKEN_AP.search(s) is not None
    pa = TOKEN_PA.search(s) is not None
    if ap and not pa:
        return "ap"
    if pa and not ap:
        return "pa"
    return None


def infer_kind(path: Path) -> Optional[str]:
    s = path.as_posix().lower()
    pred = TOKEN_PRED.search(s) is not None
    gt = TOKEN_GT.search(s) is not None
    if pred and not gt:
        return "pred"
    if gt and not pred:
        return "gt"
    return None


def _pid_regex(phantom_id: str) -> re.Pattern[str]:
    pid = re.sub(r"[^0-9]", "", str(phantom_id))
    pid = pid if pid else str(phantom_id)
    return re.compile(PHANTOM_RE_TPL.format(pid=re.escape(pid)), re.IGNORECASE)


def _candidate_score(path: Path, view: str, kind: str, phantom_re: re.Pattern[str], step: Optional[int]) -> float:
    p = path.as_posix().lower()
    base = 0.0
    if phantom_re.search(p):
        base += 200.0
    if path.suffix.lower() == ".npy":
        base += 80.0
    if "/proj/" in p:
        base += 40.0
    if "/plots/" in p:
        base += 20.0
    if "diff" in path.name.lower() or "logpct" in path.name.lower() or "compare" in path.name.lower():
        base -= 120.0
    if f"{kind}_{view}" in path.name.lower() or f"{view}_{kind}" in path.name.lower():
        base += 80.0
    fstep = infer_step(path)
    if step is not None:
        if fstep == step:
            base += 120.0
        elif fstep is not None:
            base -= 40.0
    if view == "ap" and TOKEN_AP.search(path.name.lower()):
        base += 10.0
    if view == "pa" and TOKEN_PA.search(path.name.lower()):
        base += 10.0
    if kind == "pred" and TOKEN_PRED.search(path.name.lower()):
        base += 10.0
    if kind == "gt" and TOKEN_GT.search(path.name.lower()):
        base += 10.0
    return base


def find_candidates(results_dir: Path, phantom_id: str, step: Optional[int]) -> list[FileCandidate]:
    phantom_re = _pid_regex(phantom_id)
    files = [p for p in results_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".npy", ".png"}]
    out: list[FileCandidate] = []
    for p in files:
        view = infer_view(p)
        kind = infer_kind(p)
        if view is None or kind is None:
            continue
        score = _candidate_score(p, view, kind, phantom_re, step)
        out.append(
            FileCandidate(
                path=p,
                view=view,
                kind=kind,
                ext=p.suffix.lower(),
                step=infer_step(p),
                score=score,
            )
        )
    return out


def pick_best(cands: Iterable[FileCandidate], kind: str, view: str) -> Optional[FileCandidate]:
    subset = [c for c in cands if c.kind == kind and c.view == view]
    if not subset:
        return None
    subset.sort(key=lambda c: (c.score, c.path.suffix.lower() == ".npy", str(c.path)), reverse=True)
    return subset[0]


def load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
        return arr.astype(np.float64)

    try:
        from PIL import Image

        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float64)
        return arr
    except Exception:
        import matplotlib.image as mpimg

        arr = np.asarray(mpimg.imread(path))
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=-1)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D image in {path}, got shape {arr.shape}")
        return arr.astype(np.float64)


def sanitize_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        raise ValueError("No finite overlap between arrays")
    return a[mask], b[mask]


def mse_score(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = sanitize_pair(a, b)
    return float(np.mean((aa - bb) ** 2))


def ncc_score(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = sanitize_pair(a, b)
    aa = aa - np.mean(aa)
    bb = bb - np.mean(bb)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def apply_flip(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return arr
    if mode == "x":
        return np.flip(arr, axis=1)
    if mode == "y":
        return np.flip(arr, axis=0)
    if mode == "xy":
        return np.flip(np.flip(arr, axis=0), axis=1)
    raise ValueError(f"Unknown flip mode: {mode}")


def score_pair(pred_ap: np.ndarray, pred_pa: np.ndarray, gt_ap: np.ndarray, gt_pa: np.ndarray, metric: str, swapped: bool, flip_mode: str) -> tuple[float, float, float]:
    pap = apply_flip(pred_ap, flip_mode)
    ppa = apply_flip(pred_pa, flip_mode)
    tap = gt_pa if swapped else gt_ap
    tpa = gt_ap if swapped else gt_pa

    if metric == "mse":
        s_ap = mse_score(pap, tap)
        s_pa = mse_score(ppa, tpa)
        total = 0.5 * (s_ap + s_pa)
    else:
        s_ap = ncc_score(pap, tap)
        s_pa = ncc_score(ppa, tpa)
        total = 0.5 * (s_ap + s_pa)
    return s_ap, s_pa, total


def hypothesis_name(swapped: bool, flip_mode: str) -> str:
    base = "swapped" if swapped else "baseline"
    if flip_mode == "none":
        return base
    return f"{base} + flip_{flip_mode}"


def format_score(v: float) -> str:
    if math.isnan(v):
        return "nan"
    return f"{v:.6g}"


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    cands = find_candidates(results_dir, args.phantom_id, args.step)
    if not cands:
        raise FileNotFoundError(
            "No AP/PA pred/gt candidates found. Expected patterns like *pred*AP*, *gt*PA* in *.npy or *.png"
        )

    selected = {
        "pred_ap": pick_best(cands, kind="pred", view="ap"),
        "pred_pa": pick_best(cands, kind="pred", view="pa"),
        "gt_ap": pick_best(cands, kind="gt", view="ap"),
        "gt_pa": pick_best(cands, kind="gt", view="pa"),
    }

    missing = [k for k, v in selected.items() if v is None]
    if missing:
        raise FileNotFoundError(
            f"Missing required projection files: {missing}. Found {len(cands)} partial candidates under {results_dir}"
        )

    pred_ap = load_image(selected["pred_ap"].path)
    pred_pa = load_image(selected["pred_pa"].path)
    gt_ap = load_image(selected["gt_ap"].path)
    gt_pa = load_image(selected["gt_pa"].path)

    # Quick shape check early for clearer error messages.
    _ = sanitize_pair(pred_ap, gt_ap)
    _ = sanitize_pair(pred_pa, gt_pa)

    print("[selected-files]")
    for key in ("pred_ap", "pred_pa", "gt_ap", "gt_pa"):
        c = selected[key]
        assert c is not None
        print(f"{key}: {c.path} (score={c.score:.1f}, step={c.step}, ext={c.ext})")

    print(f"\n[hypotheses] metric={args.metric}")
    rows = []
    for swapped in (False, True):
        for flip_mode in ("none", "x", "y", "xy"):
            s_ap, s_pa, s_tot = score_pair(
                pred_ap=pred_ap,
                pred_pa=pred_pa,
                gt_ap=gt_ap,
                gt_pa=gt_pa,
                metric=args.metric,
                swapped=swapped,
                flip_mode=flip_mode,
            )
            name = hypothesis_name(swapped, flip_mode)
            rows.append((name, s_ap, s_pa, s_tot))
            print(f"{name:20s} | AP={format_score(s_ap):>10s} | PA={format_score(s_pa):>10s} | total={format_score(s_tot):>10s}")

    if args.metric == "mse":
        best = min(rows, key=lambda r: (float("inf") if math.isnan(r[3]) else r[3]))
    else:
        best = max(rows, key=lambda r: (-float("inf") if math.isnan(r[3]) else r[3]))

    print(f"\nBEST: {best[0]} (score={format_score(best[3])}, metric={args.metric})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
