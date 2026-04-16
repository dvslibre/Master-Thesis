#!/usr/bin/env python3
"""
Aggregate QPlanar organ quantification results across phantoms.

Scans recursively for JSON files matching the provided pattern, normalizes the
per-organ results into a single pandas DataFrame, computes summary statistics,
saves CSV/LaTeX artifacts, and draws diagnostics (GT vs REC scatter, boxplots,
and a Bland-Altman plot). The boxplot visualizes the distribution of
`rel_error_percent` per organ.


python3 eval_qplanar_results.py \
  --root /home/mnguest12/projects/thesis/PhantomGenerator \
  --out /home/mnguest12/projects/thesis/PhantomGenerator/qplanar_eval

"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def discover_results(root: Path, pattern: str) -> List[Path]:
    return sorted(root.rglob(pattern))


def load_phantom_results(
    json_path: Path,
    only_organs: Optional[Sequence[str]],
    min_gt: float,
) -> List[Dict]:
    phantom_name = json_path.parent.name
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[WARN] Could not read {json_path}: {exc}")
        return []

    rows: List[Dict] = []
    organ_chars = {org.lower() for org in only_organs} if only_organs else None

    for organ, entry in data.items():
        if organ_chars and organ.lower() not in organ_chars:
            continue

        missing = [k for k in ("organ_id", "gt_kBq_per_ml", "rec_kBq_per_ml", "rel_error_percent") if k not in entry]
        if missing:
            print(f"[WARN] Skipping organ '{organ}' in {json_path} (missing {missing})")
            continue

        try:
            gt = float(entry["gt_kBq_per_ml"])
            rec = float(entry["rec_kBq_per_ml"])
            rel = float(entry["rel_error_percent"])
            organ_id = int(entry["organ_id"])
        except Exception as exc:
            print(f"[WARN] Bad values for organ '{organ}' in {json_path}: {exc}")
            continue

        signed_error = rec - gt
        abs_error = abs(signed_error)
        if min_gt <= 0:
            signed_rel = (
                100.0 * signed_error / gt if gt != 0 else np.nan
            )
        else:
            signed_rel = 100.0 * signed_error / gt if abs(gt) >= min_gt and gt != 0 else np.nan

        rows.append(
            {
                "phantom": phantom_name,
                "organ": organ,
                "organ_id": organ_id,
                "gt_kBq_per_ml": gt,
                "rec_kBq_per_ml": rec,
                "rel_error_percent": rel,
                "abs_error_kBq_per_ml": abs_error,
                "signed_error_kBq_per_ml": signed_error,
                "signed_rel_error_percent": signed_rel,
            }
        )
    return rows


def summarize_by_organ(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("organ", sort=True).agg(
        N=("organ", "count"),
        mean_rel_error_percent=("rel_error_percent", "mean"),
        std_rel_error_percent=("rel_error_percent", "std"),
        median_rel_error_percent=("rel_error_percent", "median"),
        max_rel_error_percent=("rel_error_percent", "max"),
        mean_abs_error_kBq_per_ml=("abs_error_kBq_per_ml", "mean"),
        std_abs_error_kBq_per_ml=("abs_error_kBq_per_ml", "std"),
        mean_signed_error_kBq_per_ml=("signed_error_kBq_per_ml", "mean"),
    )
    return agg


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    lines = []
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Organ & N & Mean RelErr (\%) & Std (\%) & Max (\%) & Bias (kBq/mL) \\")
    lines.append(r"\midrule")
    for organ, row in summary.iterrows():
        organ_label = organ.replace("_", r"\_")
        lines.append(
            f"{organ_label} & "
            f"{int(row['N'])} & "
            f"{row['mean_rel_error_percent']:.3f} & "
            f"{row['std_rel_error_percent']:.3f} & "
            f"{row['max_rel_error_percent']:.3f} & "
            f"{row['mean_signed_error_kBq_per_ml']:.3f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_gt_vs_rec(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    organs = sorted(df["organ"].unique())
    cmap = plt.get_cmap("tab20")
    colors = {organ: cmap(i % cmap.N) for i, organ in enumerate(organs)}

    for organ, group in df.groupby("organ"):
        ax.scatter(
            group["gt_kBq_per_ml"],
            group["rec_kBq_per_ml"],
            label=organ if len(organs) <= 20 else None,
            color=colors[organ],
            alpha=0.7,
            edgecolors="none",
        )

    lims = [
        min(df["gt_kBq_per_ml"].min(), df["rec_kBq_per_ml"].min()),
        max(df["gt_kBq_per_ml"].max(), df["rec_kBq_per_ml"].max()),
    ]
    ax.plot(lims, lims, color="black", linestyle="--", label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("GT (kBq/mL)")
    ax.set_ylabel("Rec (kBq/mL)")
    ax.set_title("GT vs REC per organ")
    if len(organs) <= 20:
        ax.legend(loc="best", fontsize="small")
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column=["rel_error_percent"], by="organ", ax=ax, rot=45, fontsize=8)
    ax.set_title("Relative error per organ (rel_error_percent)")
    ax.set_xlabel("")
    ax.set_ylabel("Relative error (%)")
    ax.tick_params(axis="x", labelrotation=45)
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bland_altman(df: pd.DataFrame, out_path: Path) -> None:
    mean_vals = (df["gt_kBq_per_ml"] + df["rec_kBq_per_ml"]) / 2.0
    diff_vals = df["rec_kBq_per_ml"] - df["gt_kBq_per_ml"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_vals, diff_vals, alpha=0.6, s=30)
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("Mean (kBq/mL)")
    ax.set_ylabel("Rec - GT (kBq/mL)")
    ax.set_title("Bland-Altman plot")
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate QPlanar quantification results.")
    parser.add_argument("--root", type=Path, required=True, help="Root directory to scan for JSON results.")
    parser.add_argument("--out", type=Path, default=Path("eval_out"), help="Directory for aggregated output.")
    parser.add_argument("--pattern", type=str, default="quantification_results.json", help="Filename pattern to look for.")
    parser.add_argument("--min_gt", type=float, default=0.0, help="Minimum GT value to compute signed relative errors.")
    parser.add_argument(
        "--only_organs",
        type=str,
        default=None,
        help="Comma-separated whitelist of organ names to include (case-insensitive).",
    )

    args = parser.parse_args()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    only_organs = [s.strip() for s in args.only_organs.split(",")] if args.only_organs else None

    if not args.root.is_dir():
        raise SystemExit(f"Root directory does not exist: {args.root}")

    paths = discover_results(args.root, args.pattern)
    if not paths:
        raise SystemExit(f"No files matching '{args.pattern}' under {args.root}")

    all_rows = []
    for path in paths:
        rows = load_phantom_results(path, only_organs, args.min_gt)
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No valid results loaded.")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "all_results.csv", index=False)

    summary = summarize_by_organ(df)
    summary.to_csv(out_dir / "summary_per_organ.csv")
    write_latex_table(summary, out_dir / "summary_per_organ.tex")

    plot_gt_vs_rec(df, out_dir / "gt_vs_rec.png")
    plot_boxplot(df, out_dir / "rel_error_boxplot.png")
    plot_bland_altman(df, out_dir / "bland_altman.png")

    print(f"[INFO] Aggregated {len(paths)} files from {df['phantom'].nunique()} phantoms.")
    organ_list = sorted(df["organ"].unique())
    print(f"[INFO] Organs encountered: {', '.join(organ_list)}")

    top3 = summary.sort_values("mean_rel_error_percent", ascending=False).head(3)
    print("[INFO] Top 3 organs by mean relative error:")
    for organ, row in top3.iterrows():
        print(f"  {organ}: {row['mean_rel_error_percent']:.3f}% (N={int(row['N'])})")


if __name__ == "__main__":
    main()

# Example:
# python3 eval_qplanar_results.py --root /home/mnguest12/projects/thesis/PhantomGenerator --out /home/mnguest12/projects/thesis/PhantomGenerator/qplanar_eval
