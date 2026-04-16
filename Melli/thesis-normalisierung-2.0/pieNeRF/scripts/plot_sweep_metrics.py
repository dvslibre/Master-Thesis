#!/usr/bin/env python3
"""Summaries sweep metrics and produces comparison plots."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_PHANTOMS = ["phantom_16", "phantom_24", "phantom_30"]
VOLUME_ORGAN_METRICS = [
    "vol_mae_vol",
    "activity_rel_abs_error_vol",
    "organ_rel_error_total_activity_active_mean",
]
PROJECTION_LEAK_METRICS = [
    "proj_poisson_dev_counts",
    "pred_outside_mask_frac",
]
ALL_METRICS = VOLUME_ORGAN_METRICS + PROJECTION_LEAK_METRICS
PLOT_GROUP_PRIMARY = [
    "vol_mae_vol",
    "proj_poisson_dev_counts",
]
PLOT_GROUP_SECONDARY = [
    "activity_rel_abs_error_vol",
    "organ_rel_error_total_activity_active_mean",
    "pred_outside_mask_frac",
]

METRIC_COLORS = {
    "proj_poisson_dev_counts": "red",
    "pred_outside_mask_frac": "pink",
    "vol_mae_vol": "purple",
    "activity_rel_abs_error_vol": "lightgreen",
    "organ_rel_error_total_activity_active_mean": "cornflowerblue",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize sweep metrics and plot selected statistics."
    )
    parser.add_argument(
        "--tag-prefix",
        default="",
        help="Only process tag directories whose name starts with this prefix (e.g., actW_).",
    )
    parser.add_argument(
        "--tag-regex",
        default="",
        help="Optional regex to filter tag directories (takes precedence over --tag-prefix).",
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("results_spect_sweep"),
        help="Root directory that contains sweep TAG folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit directory for the CSV/PNG outputs; default is calculated under the sweep root.",
    )
    parser.add_argument(
        "--summary-name",
        default="sweep_summary.csv",
        help="Filename for the tabular summary in the output directory.",
    )
    parser.add_argument(
        "--plot-vol",
        default="sweep_volume_metrics.png",
        help="Filename for the combined volume/organ metrics plot.",
    )
    parser.add_argument(
        "--plot-proj",
        default="sweep_projection_metrics.png",
        help="Filename for the combined projection/leakage metrics plot.",
    )
    return parser.parse_args()


def resolve_output_dir(
    sweep_root: Path, tag_prefix: str, tag_regex: str
) -> Path:
    if tag_regex:
        suffix = "regex_sweep"
    elif tag_prefix:
        cleaned = tag_prefix.rstrip("_").strip("_")
        suffix = f"{cleaned or 'tag'}_sweep"
    else:
        suffix = "full_sweep"
    return sweep_root / suffix


def parse_sweep_param(tag: str) -> tuple[str, Optional[float]]:
    """Return the raw sweep parameter string and an optional numeric value for sorting."""

    raw = tag.split("_", 1)[1] if "_" in tag else tag
    normalized = raw.replace("p", ".")
    try:
        numeric = float(normalized)
    except ValueError:
        numeric = None
    return raw, numeric


def resolve_metrics_path(tag_dir: Path) -> Optional[Path]:
    candidates = [
        tag_dir / "postproc_trainfwd_test_slices" / "metrics.csv",
        tag_dir / "results_spect" / "postproc_baseline_calib" / "metrics.csv",
        tag_dir / "results_spect" / "postproc" / "metrics.csv",
        tag_dir / "postproc" / "metrics.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def collect_tag_metrics(tag_dir: Path) -> Optional[dict[str, float]]:
    csv_path = resolve_metrics_path(tag_dir)
    if csv_path is None:
        logging.warning("Missing metrics.csv in %s", tag_dir)
        return None

    df = pd.read_csv(csv_path)
    missing_columns = [m for m in ALL_METRICS + ["phantom_id"] if m not in df.columns]
    if missing_columns:
        logging.warning(
            "Skipping %s because missing columns: %s",
            tag_dir.name,
            ", ".join(missing_columns),
        )
        return None

    subset = df[df["phantom_id"].isin(REQUIRED_PHANTOMS)]
    missing_phantoms = set(REQUIRED_PHANTOMS) - set(subset["phantom_id"].unique())
    if missing_phantoms:
        logging.warning(
            "Skipping %s because missing phantom rows: %s",
            tag_dir.name,
            ", ".join(sorted(missing_phantoms)),
        )
        return None

    stats: dict[str, float] = {}
    for metric in ALL_METRICS:
        series = subset[metric].astype(float)
        stats[f"mean_{metric}"] = series.mean()
        stats[f"std_{metric}"] = series.std(ddof=0)
    return stats


def collect_sweep_summary(
    sweep_root: Path, tag_prefix: str, tag_regex: str
) -> pd.DataFrame:
    rows = []
    tag_dirs = [
        d
        for d in sorted(sweep_root.iterdir())
        if d.is_dir() and not d.name.endswith("_sweep")
    ]
    if tag_regex:
        matcher = re.compile(tag_regex)
        tag_dirs = [d for d in tag_dirs if matcher.search(d.name)]
    elif tag_prefix:
        tag_dirs = [d for d in tag_dirs if d.name.startswith(tag_prefix)]
    for tag_dir in tag_dirs:
        raw_param, numeric_param = parse_sweep_param(tag_dir.name)
        stats = collect_tag_metrics(tag_dir)
        if stats is None:
            continue

        row = {
            "tag": tag_dir.name,
            "sweep_param": raw_param,
            "sweep_param_numeric": numeric_param,
        }
        row.update(stats)
        rows.append(row)

    if not rows:
        raise SystemExit("No valid sweep runs were collected. Check the sweep root path.")

    df = pd.DataFrame(rows)
    if df["sweep_param_numeric"].notna().all():
        df = df.sort_values("sweep_param_numeric")
    else:
        df = df.sort_values("sweep_param")
    df = df.reset_index(drop=True)
    return df


def plot_metrics(
    df: pd.DataFrame,
    metrics: Iterable[str],
    title: Optional[str],
    filepath: Path,
    x_label: str = "Sweep Parameter",
) -> None:
    numeric = df["sweep_param_numeric"].notna().all()
    fig, ax = plt.subplots(figsize=(8, 5))

    if numeric:
        x = df["sweep_param_numeric"].to_numpy()
        ax.set_xlabel(x_label)
    else:
        x = df.index.to_numpy()
        ax.set_xticks(x)
        ax.set_xticklabels(df["sweep_param"], rotation=45, ha="right")
        ax.set_xlabel("Sweep Tag")

    for metric in metrics:
        mean_col = f"mean_{metric}"
        std_col = f"std_{metric}"
        if mean_col not in df.columns or std_col not in df.columns:
            logging.warning("Plot metric %s missing from summary table", metric)
            continue

        ax.errorbar(
            x,
            df[mean_col],
            yerr=df[std_col],
            marker="o",
            capsize=3,
            label=metric,
            color=METRIC_COLORS.get(metric),
        )

    if title:
        ax.set_title(title)
    ax.set_ylabel("Metric value")
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    logging.info("Saved plot %s", filepath)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_dir = args.output_dir or resolve_output_dir(
        args.sweep_root, args.tag_prefix, args.tag_regex
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Scanning sweep root %s", args.sweep_root)
    summary_df = collect_sweep_summary(
        args.sweep_root, args.tag_prefix, args.tag_regex
    )

    summary_path = output_dir / args.summary_name
    summary_df.to_csv(summary_path, index=False)
    logging.info("Summary CSV written to %s", summary_path)

    plot_metrics(
        summary_df,
        PLOT_GROUP_PRIMARY,
        None,
        output_dir / args.plot_vol,
    )
    plot_metrics(
        summary_df,
        PLOT_GROUP_SECONDARY,
        None,
        output_dir / args.plot_proj,
    )


if __name__ == "__main__":
    main()
