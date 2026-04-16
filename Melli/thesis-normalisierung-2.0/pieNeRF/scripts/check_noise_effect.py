#!/usr/bin/env python3
"""Check whether test-time Poisson noise affects encoder input and predictions.

Outputs under BASE_DIR:
  - noise_check_summary.csv
  - noise_log_check.csv
  - noise_log_excerpt.txt
  - noise_sanity_images/*.png (optional)
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


EPS = 1e-12


@dataclass
class RunInfo:
    tag: str
    path: Path
    kappa: float
    seed: int


def parse_tag(tag: str) -> Tuple[float, int]:
    m = re.match(r"^noiseK_([0-9]*\.?[0-9]+)_seed_([0-9]+)$", tag)
    if not m:
        raise ValueError(f"Unsupported run tag format: {tag}")
    return float(m.group(1)), int(m.group(2))


def stats(arr: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return {"sum": math.nan, "mean": math.nan, "std": math.nan, "p99": math.nan}
    return {
        "sum": float(np.sum(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p99": float(np.quantile(flat, 0.99)),
    }


def load_manifest_counts(manifest_path: Path) -> Dict[str, Dict[str, str]]:
    if not manifest_path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("patient_id") or "").strip()
            if not pid:
                continue
            out[pid] = row
    return out


def list_runs(base_dir: Path) -> List[RunInfo]:
    runs: List[RunInfo] = []
    for d in sorted(base_dir.glob("noiseK_*_seed_*")):
        if not d.is_dir():
            continue
        try:
            kappa, seed = parse_tag(d.name)
        except ValueError:
            continue
        runs.append(RunInfo(tag=d.name, path=d, kappa=kappa, seed=seed))
    return runs


def find_slurm_log(base_dir: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit if explicit.exists() else None
    m = re.search(r"noiseK_sweep_run_(\d+)$", str(base_dir))
    if not m:
        return None
    job_id = m.group(1)
    candidate = Path(f"/home/mnguest12/slurm/sweep_noiseK_inf.{job_id}.out")
    return candidate if candidate.exists() else None


def parse_cfg_hybrid(line: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in ("encoder_proj_transform", "proj_scale_source", "z_enc_alpha"):
        m = re.search(rf"{key}=([^| ]+)", line)
        if m:
            out[key] = m.group(1).strip()
    return out


def parse_proj_input(line: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    m = re.search(r"source_requested=([^ ]+)", line)
    if m:
        out["proj_input_source"] = m.group(1).strip()
    return out


def parse_slurm_sections(slurm_log: Path, run_tags: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    tags = set(run_tags)
    sections: Dict[str, Dict[str, Any]] = {t: {} for t in tags}
    counts: Dict[str, Dict[str, int]] = {
        t: {
            "proj_input_lines": 0,
            "encoder_input_lines": 0,
            "test_noise_lines": 0,
            "latent_lines": 0,
            "pred_lines": 0,
        }
        for t in tags
    }
    excerpts: Dict[str, Dict[str, List[str]]] = {
        t: {
            "proj_input": [],
            "encoder_input": [],
            "test_noise": [],
            "latent": [],
            "pred": [],
        }
        for t in tags
    }

    current_tag: Optional[str] = None
    with slurm_log.open("r", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            m_tag = re.search(r"🔁\s+(noiseK_[0-9.]+_seed_[0-9]+)", line)
            if m_tag:
                t = m_tag.group(1)
                current_tag = t if t in tags else None
                continue
            if current_tag is None:
                continue
            if line.startswith("[cfg][hybrid]"):
                sections[current_tag].update(parse_cfg_hybrid(line))
            if "[DEBUG][proj-input]" in line:
                counts[current_tag]["proj_input_lines"] += 1
                sections[current_tag].update(parse_proj_input(line))
                if len(excerpts[current_tag]["proj_input"]) < 3:
                    excerpts[current_tag]["proj_input"].append(line)
            if "[DEBUG][encoder-input]" in line:
                counts[current_tag]["encoder_input_lines"] += 1
                if len(excerpts[current_tag]["encoder_input"]) < 3:
                    excerpts[current_tag]["encoder_input"].append(line)
            if "[test][noise]" in line:
                counts[current_tag]["test_noise_lines"] += 1
                if len(excerpts[current_tag]["test_noise"]) < 6:
                    excerpts[current_tag]["test_noise"].append(line)
            if "[test][slices][latent]" in line:
                counts[current_tag]["latent_lines"] += 1
                if len(excerpts[current_tag]["latent"]) < 6:
                    excerpts[current_tag]["latent"].append(line)
            if "[test][slices][pred]" in line:
                counts[current_tag]["pred_lines"] += 1
                if len(excerpts[current_tag]["pred"]) < 6:
                    excerpts[current_tag]["pred"].append(line)

    for tag in tags:
        sections[tag].update(counts[tag])
        sections[tag]["_excerpts"] = excerpts[tag]
    return sections


def _safe_rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.linalg.norm((a - b).ravel()))
    den = float(np.linalg.norm(b.ravel()))
    return num / (den + EPS)


def _load_optional(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.load(path)


def _load_first_existing(paths: Iterable[Path]) -> Optional[np.ndarray]:
    for p in paths:
        arr = _load_optional(p)
        if arr is not None:
            return arr
    return None


def make_sanity_image(
    out_path: Path,
    arr_ref: np.ndarray,
    arr_cmp: np.ndarray,
    title_ref: str,
    title_cmp: str,
) -> None:
    import matplotlib.pyplot as plt

    # Rotate only the image content by 90° clockwise + horizontal flip; keep full figure layout unchanged.
    ref = np.fliplr(np.rot90(np.asarray(arr_ref, dtype=np.float32), k=-1))
    cmp_ = np.fliplr(np.rot90(np.asarray(arr_cmp, dtype=np.float32), k=-1))
    vmax = float(np.quantile(np.concatenate([ref.reshape(-1), cmp_.reshape(-1)]), 0.995))
    vmax = max(vmax, 1e-6)
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(12, 4),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.02},
    )
    im0 = axs[0].imshow(ref, cmap="viridis", vmin=0.0, vmax=vmax)
    axs[0].set_title(title_ref)
    axs[1].imshow(cmp_, cmap="viridis", vmin=0.0, vmax=vmax)
    axs[1].set_title(title_cmp)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(im0, ax=axs.tolist(), shrink=0.9)
    cbar.set_label("counts")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_base = Path("/home/mnguest12/projects/thesis/pieNeRF/results_spect_test_1/noiseK_sweep_run_700")
    default_manifest = repo_root / "data" / "manifest_abs.csv"

    ap = argparse.ArgumentParser(description="Check test-time noise effects across noiseK runs.")
    ap.add_argument("--base-dir", type=Path, default=default_base)
    ap.add_argument("--manifest", type=Path, default=default_manifest)
    ap.add_argument("--slurm-log", type=Path, default=None)
    ap.add_argument("--ref-kappa", type=float, default=1.0)
    ap.add_argument("--cmp-kappa", type=float, default=0.1, help="For optional sanity images.")
    ap.add_argument("--make-images", action="store_true")
    args = ap.parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    if not base_dir.exists():
        parent = base_dir.parent
        candidates = sorted([p for p in parent.glob("noiseK_sweep_run_*") if p.is_dir()])
        if candidates:
            fallback = candidates[-1]
            print(f"[warn] BASE_DIR not found: {base_dir}")
            print(f"[warn] Using latest available sweep dir instead: {fallback}")
            base_dir = fallback
        else:
            raise FileNotFoundError(
                f"BASE_DIR not found: {base_dir} (and no noiseK_sweep_run_* directories found under {parent})"
            )

    runs = list_runs(base_dir)
    if not runs:
        raise RuntimeError(f"No noiseK_*_seed_* runs found in {base_dir}")

    manifest_map = load_manifest_counts(args.manifest.expanduser().resolve())

    slurm_log = find_slurm_log(base_dir, args.slurm_log.expanduser().resolve() if args.slurm_log else None)
    log_sections = parse_slurm_sections(slurm_log, [r.tag for r in runs]) if slurm_log else {}

    # Build reference prediction map by (seed, phantom)
    ref_preds: Dict[Tuple[int, str], np.ndarray] = {}
    ref_run_tags: Dict[Tuple[int, str], str] = {}
    for r in runs:
        if abs(r.kappa - float(args.ref_kappa)) > 1e-12:
            continue
        ts = r.path / "test_slices"
        if ts.exists():
            for pdir in sorted(ts.iterdir()):
                if not pdir.is_dir():
                    continue
                pred = _load_optional(pdir / "activity_pred.npy")
                if pred is not None:
                    ref_preds[(r.seed, pdir.name)] = np.asarray(pred, dtype=np.float32)
                    ref_run_tags[(r.seed, pdir.name)] = r.tag
        pred_final = _load_optional(r.path / "activity_pred_final.npy")
        if pred_final is not None:
            ref_preds[(r.seed, "__run__")] = np.asarray(pred_final, dtype=np.float32)
            ref_run_tags[(r.seed, "__run__")] = r.tag

    rows: List[Dict[str, Any]] = []
    noisy_between_kappa: Dict[Tuple[str, str], np.ndarray] = {}

    for r in runs:
        ts = r.path / "test_slices"
        phantoms: List[str] = []
        if ts.exists():
            phantoms = sorted([d.name for d in ts.iterdir() if d.is_dir()])
        # run-level fallback row
        if not phantoms:
            phantoms = ["__run__"]

        for phantom in phantoms:
            prow: Dict[str, Any] = {
                "tag": r.tag,
                "kappa": r.kappa,
                "seed": r.seed,
                "phantom": phantom,
            }

            noisy_ap = None
            noisy_pa = None
            pred = None
            if phantom == "__run__":
                pred = _load_optional(r.path / "activity_pred_final.npy")
            else:
                pdir = ts / phantom
                noisy_ap = _load_optional(pdir / "noisy_ap_counts.npy")
                noisy_pa = _load_optional(pdir / "noisy_pa_counts.npy")
                pred = _load_optional(pdir / "activity_pred.npy")

            # Noisy stats
            if noisy_ap is not None:
                s = stats(noisy_ap)
                for k, v in s.items():
                    prow[f"noisy_ap_{k}"] = v
                noisy_between_kappa[(r.tag, phantom, "ap")] = np.asarray(noisy_ap, dtype=np.float32)
            else:
                for k in ("sum", "mean", "std", "p99"):
                    prow[f"noisy_ap_{k}"] = math.nan

            if noisy_pa is not None:
                s = stats(noisy_pa)
                for k, v in s.items():
                    prow[f"noisy_pa_{k}"] = v
                noisy_between_kappa[(r.tag, phantom, "pa")] = np.asarray(noisy_pa, dtype=np.float32)
            else:
                for k in ("sum", "mean", "std", "p99"):
                    prow[f"noisy_pa_{k}"] = math.nan

            # Original counts (from manifest) if available
            if phantom in manifest_map:
                mrow = manifest_map[phantom]
                ap_path = Path(mrow.get("ap_counts_path", ""))
                pa_path = Path(mrow.get("pa_counts_path", ""))
                ap_orig = _load_optional(ap_path) if ap_path.exists() else None
                pa_orig = _load_optional(pa_path) if pa_path.exists() else None
            else:
                ap_orig = None
                pa_orig = None

            if ap_orig is not None and noisy_ap is not None:
                ap_orig_2d = np.asarray(ap_orig).squeeze()
                noisy_ap_2d = np.asarray(noisy_ap).squeeze()
                diff = noisy_ap_2d - ap_orig_2d
                prow["ap_diff_mean_abs_to_orig"] = float(np.mean(np.abs(diff)))
                prow["ap_diff_max_abs_to_orig"] = float(np.max(np.abs(diff)))
            else:
                prow["ap_diff_mean_abs_to_orig"] = math.nan
                prow["ap_diff_max_abs_to_orig"] = math.nan

            if pa_orig is not None and noisy_pa is not None:
                pa_orig_2d = np.asarray(pa_orig).squeeze()
                noisy_pa_2d = np.asarray(noisy_pa).squeeze()
                diff = noisy_pa_2d - pa_orig_2d
                prow["pa_diff_mean_abs_to_orig"] = float(np.mean(np.abs(diff)))
                prow["pa_diff_max_abs_to_orig"] = float(np.max(np.abs(diff)))
            else:
                prow["pa_diff_mean_abs_to_orig"] = math.nan
                prow["pa_diff_max_abs_to_orig"] = math.nan

            # Prediction differences vs ref kappa
            key = (r.seed, phantom)
            if pred is not None and key in ref_preds:
                pred_a = np.asarray(pred, dtype=np.float32)
                pred_ref = ref_preds[key]
                if pred_a.shape == pred_ref.shape:
                    d = np.abs(pred_a - pred_ref)
                    prow["pred_rel_l2_vs_ref"] = _safe_rel_l2(pred_a, pred_ref)
                    prow["pred_max_abs_diff_vs_ref"] = float(np.max(d))
                    prow["pred_mean_abs_diff_vs_ref"] = float(np.mean(d))
                    prow["pred_ref_tag"] = ref_run_tags.get(key, "")
                else:
                    prow["pred_rel_l2_vs_ref"] = math.nan
                    prow["pred_max_abs_diff_vs_ref"] = math.nan
                    prow["pred_mean_abs_diff_vs_ref"] = math.nan
                    prow["pred_ref_tag"] = ref_run_tags.get(key, "")
            else:
                prow["pred_rel_l2_vs_ref"] = math.nan
                prow["pred_max_abs_diff_vs_ref"] = math.nan
                prow["pred_mean_abs_diff_vs_ref"] = math.nan
                prow["pred_ref_tag"] = ""

            # Add parsed log/config facts per run
            sec = log_sections.get(r.tag, {})
            prow["encoder_proj_transform_effective"] = sec.get("encoder_proj_transform", "")
            prow["proj_scale_source_effective"] = sec.get("proj_scale_source", "")
            prow["z_enc_alpha_effective"] = sec.get("z_enc_alpha", "")
            prow["proj_input_source_effective"] = sec.get("proj_input_source", "")
            for c in ("proj_input_lines", "encoder_input_lines", "test_noise_lines", "latent_lines", "pred_lines"):
                prow[c] = int(sec.get(c, 0) or 0)

            rows.append(prow)

    # Cross-kappa noisy diffs when orig is unavailable
    # compare against reference kappa, same seed+phantom
    index_rows: Dict[Tuple[str, int, str], Dict[str, Any]] = {(r["tag"], int(r["seed"]), str(r["phantom"])): r for r in rows}
    ref_runs = [r for r in runs if abs(r.kappa - float(args.ref_kappa)) <= 1e-12]
    for rr in ref_runs:
        for phantom in sorted({str(r["phantom"]) for r in rows if int(r["seed"]) == rr.seed}):
            for view in ("ap", "pa"):
                ref_arr = noisy_between_kappa.get((rr.tag, phantom, view))
                if ref_arr is None:
                    continue
                for r in runs:
                    if r.seed != rr.seed:
                        continue
                    arr = noisy_between_kappa.get((r.tag, phantom, view))
                    if arr is None or arr.shape != ref_arr.shape:
                        continue
                    row = index_rows.get((r.tag, r.seed, phantom))
                    if row is None:
                        continue
                    d = np.abs(arr - ref_arr)
                    row[f"{view}_diff_mean_abs_vs_k{args.ref_kappa}"] = float(np.mean(d))
                    row[f"{view}_diff_max_abs_vs_k{args.ref_kappa}"] = float(np.max(d))

    # Fill missing cross-kappa columns
    for row in rows:
        for view in ("ap", "pa"):
            for metric in ("mean_abs", "max_abs"):
                col = f"{view}_diff_{metric}_vs_k{args.ref_kappa}"
                row.setdefault(col, math.nan)

    summary_csv = base_dir / "noise_check_summary.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (int(x["seed"]), float(x["kappa"]), str(x["phantom"]))):
            w.writerow(r)

    # Log check table
    log_rows: List[Dict[str, Any]] = []
    for r in runs:
        sec = log_sections.get(r.tag, {})
        log_rows.append(
            {
                "tag": r.tag,
                "kappa": r.kappa,
                "seed": r.seed,
                "encoder_proj_transform": sec.get("encoder_proj_transform", ""),
                "proj_scale_source": sec.get("proj_scale_source", ""),
                "z_enc_alpha": sec.get("z_enc_alpha", ""),
                "proj_input_source": sec.get("proj_input_source", ""),
                "proj_input_lines": int(sec.get("proj_input_lines", 0) or 0),
                "encoder_input_lines": int(sec.get("encoder_input_lines", 0) or 0),
                "test_noise_lines": int(sec.get("test_noise_lines", 0) or 0),
                "latent_lines": int(sec.get("latent_lines", 0) or 0),
                "pred_lines": int(sec.get("pred_lines", 0) or 0),
            }
        )

    log_csv = base_dir / "noise_log_check.csv"
    with log_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()) if log_rows else ["tag"])
        w.writeheader()
        for r in sorted(log_rows, key=lambda x: (int(x["seed"]), float(x["kappa"]))):
            w.writerow(r)

    excerpt_txt = base_dir / "noise_log_excerpt.txt"
    with excerpt_txt.open("w") as f:
        if slurm_log is None:
            f.write("No slurm log found/provided.\n")
        else:
            f.write(f"slurm_log={slurm_log}\n\n")
            for r in sorted(runs, key=lambda x: (x.seed, x.kappa)):
                f.write(f"## {r.tag}\n")
                ex = log_sections.get(r.tag, {}).get("_excerpts", {})
                for key in ("proj_input", "encoder_input", "test_noise", "latent", "pred"):
                    f.write(f"[{key}]\n")
                    for line in ex.get(key, []):
                        f.write(line + "\n")
                    if not ex.get(key):
                        f.write("<none>\n")
                f.write("\n")

    # Optional sanity images
    if args.make_images:
        image_dir = base_dir / "noise_sanity_images"
        runs_by_seed_kappa = {(r.seed, r.kappa): r for r in runs}
        for seed in sorted({r.seed for r in runs}):
            r_ref = runs_by_seed_kappa.get((seed, float(args.ref_kappa)))
            r_cmp = runs_by_seed_kappa.get((seed, float(args.cmp_kappa)))
            if r_ref is None or r_cmp is None:
                continue
            ref_ts = r_ref.path / "test_slices"
            cmp_ts = r_cmp.path / "test_slices"
            common = sorted({d.name for d in ref_ts.iterdir() if d.is_dir()} & {d.name for d in cmp_ts.iterdir() if d.is_dir()})
            for phantom in common:
                arr_ref = _load_optional(ref_ts / phantom / "noisy_ap_counts.npy")
                arr_cmp = _load_optional(cmp_ts / phantom / "noisy_ap_counts.npy")
                if arr_ref is None or arr_cmp is None:
                    continue
                out_path = image_dir / f"seed{seed}_{phantom}_ap_k{args.ref_kappa}_vs_k{args.cmp_kappa}.png"
                make_sanity_image(
                    out_path,
                    np.asarray(arr_ref).squeeze(),
                    np.asarray(arr_cmp).squeeze(),
                    f"{phantom} noisy AP k={args.ref_kappa}",
                    f"{phantom} noisy AP k={args.cmp_kappa}",
                )

    # Verdicts
    noise_line_ok = any(int(r.get("test_noise_lines", 0)) > 0 for r in log_rows)
    noisy_shift_vals = [float(r.get(f"ap_diff_mean_abs_vs_k{args.ref_kappa}", math.nan)) for r in rows]
    noisy_shift_vals += [float(r.get(f"pa_diff_mean_abs_vs_k{args.ref_kappa}", math.nan)) for r in rows]
    noisy_shift_vals = [v for v in noisy_shift_vals if np.isfinite(v)]
    noise_affects = bool(noise_line_ok and noisy_shift_vals and max(noisy_shift_vals) > 0.0)

    rel_l2_vals = [float(r.get("pred_rel_l2_vs_ref", math.nan)) for r in rows]
    rel_l2_vals = [v for v in rel_l2_vals if np.isfinite(v)]
    volume_change = bool(rel_l2_vals and max(rel_l2_vals) > 0.0)

    print(f"BASE_DIR={base_dir}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Log CSV: {log_csv}")
    print(f"Log excerpts: {excerpt_txt}")
    if args.make_images:
        print(f"Sanity images: {base_dir / 'noise_sanity_images'}")
    print(f"Noise affects encoder input: {'YES' if noise_affects else 'NO'}")
    if rel_l2_vals:
        print(
            f"Volumes change: {'YES' if volume_change else 'NO'} "
            f"(rel_L2 min/mean/max={min(rel_l2_vals):.6e}/{(sum(rel_l2_vals)/len(rel_l2_vals)):.6e}/{max(rel_l2_vals):.6e})"
        )
    else:
        print("Volumes change: NO (rel_L2 unavailable)")


if __name__ == "__main__":
    main()
