import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PixelSplit:
    H: int
    W: int
    train_idx_all: np.ndarray
    test_idx_all: np.ndarray
    train_idx_fg: np.ndarray
    train_idx_bg: np.ndarray
    test_idx_fg: np.ndarray
    test_idx_bg: np.ndarray
    test_idx_top10: Optional[np.ndarray]
    thr_used: float


def _resolve_threshold(score_img: np.ndarray, thr: float) -> float:
    if thr < 0.0:
        q = abs(float(thr))
        positives = score_img[score_img > 0]
        if positives.size == 0:
            return 0.0
        return float(np.quantile(positives, q))
    return float(thr)


def _sanity_check(split: PixelSplit):
    total_expected = split.H * split.W
    if split.train_idx_all.size + split.test_idx_all.size != total_expected:
        raise ValueError("PixelSplit size mismatch.")
    if np.intersect1d(split.train_idx_all, split.test_idx_all).size != 0:
        raise ValueError("Train/Test overlap detected.")
    for name, all_idx, fg, bg in [
        ("train", split.train_idx_all, split.train_idx_fg, split.train_idx_bg),
        ("test", split.test_idx_all, split.test_idx_fg, split.test_idx_bg),
    ]:
        merged = np.union1d(fg, bg)
        if merged.size != all_idx.size or np.setdiff1d(all_idx, merged).size != 0:
            raise ValueError(f"{name} FG/BG union mismatch.")
    if split.test_idx_top10 is not None:
        if np.setdiff1d(split.test_idx_top10, split.test_idx_all).size != 0:
            raise ValueError("top10 not subset of test_idx_all.")
    fg_ratio = split.train_idx_fg.size / float(total_expected) if total_expected > 0 else 0.0
    if split.test_idx_fg.size == 0:
        logging.warning("pixel-split: no FG rays found in test split (thr=%.3e)", split.thr_used)
    if fg_ratio < 0.01:
        logging.warning("pixel-split: FG fraction very low (%.3f%% of pixels, thr=%.3e)", fg_ratio * 100.0, split.thr_used)


def make_pixel_split_from_ap_pa(
    target_ap: np.ndarray,
    target_pa: np.ndarray,
    train_frac: float,
    tile: int,
    thr: float,
    seed: int,
    pa_xflip: bool,
    topk_frac: float = 0.10,
) -> PixelSplit:
    if target_ap.shape != target_pa.shape:
        raise ValueError(f"AP/PA shapes differ: {target_ap.shape} vs {target_pa.shape}")
    if target_ap.ndim != 2:
        raise ValueError(f"target images must be (H,W), got {target_ap.shape}")
    H, W = target_ap.shape
    tile_size = max(int(tile), 1)
    ratio = float(np.clip(train_frac, 0.0, 1.0))

    def map_pa(pa_img: np.ndarray) -> np.ndarray:
        if not pa_xflip:
            return pa_img
        return pa_img[:, ::-1]

    pa_mapped = map_pa(target_pa)
    score = np.maximum(target_ap, pa_mapped)
    thr_used = _resolve_threshold(score, thr)

    train_fg, train_bg, test_fg, test_bg = [], [], [], []
    tile_id = 0
    for y0 in range(0, H, tile_size):
        y1 = min(y0 + tile_size, H)
        for x0 in range(0, W, tile_size):
            x1 = min(x0 + tile_size, W)
            rng = np.random.default_rng(seed + tile_id)
            tile_id += 1
            tile_score = score[y0:y1, x0:x1]
            yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
            tile_idx = (yy * W + xx).reshape(-1)
            fg_mask = (tile_score > thr_used).reshape(-1)
            fg_idx = tile_idx[fg_mask]
            bg_idx = tile_idx[~fg_mask]
            if fg_idx.size > 0:
                perm_fg = rng.permutation(fg_idx.size)
                n_train_fg = int(np.floor(ratio * fg_idx.size))
                train_fg.append(fg_idx[perm_fg[:n_train_fg]])
                test_fg.append(fg_idx[perm_fg[n_train_fg:]])
            if bg_idx.size > 0:
                perm_bg = rng.permutation(bg_idx.size)
                n_train_bg = int(np.floor(ratio * bg_idx.size))
                train_bg.append(bg_idx[perm_bg[:n_train_bg]])
                test_bg.append(bg_idx[perm_bg[n_train_bg:]])

    def _concat(parts):
        return np.concatenate(parts) if parts else np.array([], dtype=np.int64)

    train_idx_fg = _concat(train_fg)
    test_idx_fg = _concat(test_fg)
    train_idx_bg = _concat(train_bg)
    test_idx_bg = _concat(test_bg)
    train_idx_all = np.concatenate([train_idx_fg, train_idx_bg]) if train_idx_fg.size + train_idx_bg.size > 0 else np.array([], dtype=np.int64)
    test_idx_all = np.concatenate([test_idx_fg, test_idx_bg]) if test_idx_fg.size + test_idx_bg.size > 0 else np.array([], dtype=np.int64)

    rng_global = np.random.default_rng(seed)
    for arr in (train_idx_fg, test_idx_fg, train_idx_bg, test_idx_bg, train_idx_all, test_idx_all):
        if arr.size > 0:
            rng_global.shuffle(arr)

    test_idx_top10 = None
    if test_idx_all.size > 0:
        test_values = score.reshape(-1)[test_idx_all]
        top_k = max(1, int(np.ceil(topk_frac * test_values.size)))
        top_order = np.argpartition(-test_values, top_k - 1)[:top_k]
        test_idx_top10 = test_idx_all[top_order]
        rng_global.shuffle(test_idx_top10)

    split = PixelSplit(
        H=H,
        W=W,
        train_idx_all=train_idx_all.astype(np.int64),
        test_idx_all=test_idx_all.astype(np.int64),
        train_idx_fg=train_idx_fg.astype(np.int64),
        train_idx_bg=train_idx_bg.astype(np.int64),
        test_idx_fg=test_idx_fg.astype(np.int64),
        test_idx_bg=test_idx_bg.astype(np.int64),
        test_idx_top10=test_idx_top10.astype(np.int64) if test_idx_top10 is not None else None,
        thr_used=float(thr_used),
    )
    _sanity_check(split)

    def _safe_stats(arr: np.ndarray):
        if arr.size == 0:
            return (np.nan, np.nan, np.nan)
        return (float(np.min(arr)), float(np.max(arr)), float(np.mean(arr)))

    fg_vals = score.reshape(-1)[split.train_idx_fg] if split.train_idx_fg.size > 0 else np.array([], dtype=score.dtype)
    bg_vals = score.reshape(-1)[split.train_idx_bg] if split.train_idx_bg.size > 0 else np.array([], dtype=score.dtype)
    fg_stats = _safe_stats(fg_vals)
    bg_stats = _safe_stats(bg_vals)
    logging.info(
        "pixel-split: thr=%.3e | FG min/max/mean=%.3e/%.3e/%.3e | BG min/max/mean=%.3e/%.3e/%.3e",
        split.thr_used,
        fg_stats[0], fg_stats[1], fg_stats[2],
        bg_stats[0], bg_stats[1], bg_stats[2],
    )
    return split


def sample_train_indices(split: PixelSplit, n: int, fg_frac: float, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.int64)
    fg_pool = split.train_idx_fg
    bg_pool = split.train_idx_bg
    all_pool = split.train_idx_all
    if fg_pool.size == 0 or bg_pool.size == 0 or fg_frac <= 0.0 or fg_frac >= 1.0:
        if all_pool.size == 0:
            return np.array([], dtype=np.int64)
        return rng.choice(all_pool, size=n, replace=True)
    n_fg = int(round(n * fg_frac))
    n_fg = max(0, min(n_fg, n))
    n_bg = n - n_fg
    fg_samples = rng.choice(fg_pool if fg_pool.size > 0 else bg_pool, size=n_fg, replace=True) if n_fg > 0 else np.array([], dtype=np.int64)
    bg_samples = rng.choice(bg_pool if bg_pool.size > 0 else fg_pool, size=n_bg, replace=True) if n_bg > 0 else np.array([], dtype=np.int64)
    samples = np.concatenate([fg_samples, bg_samples])
    rng.shuffle(samples)
    return samples
