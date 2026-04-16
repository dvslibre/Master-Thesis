#!/usr/bin/env python3
import os, glob, re, csv
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "results_sweep"
OUT_SUM = os.path.join(ROOT, "_summary")
os.makedirs(OUT_SUM, exist_ok=True)

runs = sorted(glob.glob(os.path.join(ROOT, "actw_*")))
rows = []

for run in runs:
    m = re.search(r"actw_([0-9.]+)$", run)
    if not m:
        continue
    actw = float(m.group(1))

    metrics_csv = os.path.join(run, "postproc", "metrics.csv")
    if not os.path.exists(metrics_csv):
        print(f"[WARN] missing {metrics_csv}")
        continue

    df = pd.read_csv(metrics_csv)

    # aggregate over phantoms
    agg = {"act_loss_weight": actw, "run_dir": run, "n_phantoms": len(df)}
    for col in [
        "proj_mae_counts",
        "proj_poisson_dev_counts",
        "active_organ_fraction_mae",
        "inactive_organs_pred_frac_of_pred",
        "pred_outside_mask_frac",
    ]:
        if col in df.columns:
            agg[f"{col}_mean"] = float(df[col].mean())
            agg[f"{col}_std"]  = float(df[col].std(ddof=0))
    rows.append(agg)

out_df = pd.DataFrame(rows).sort_values("act_loss_weight")
out_path = os.path.join(OUT_SUM, "summary.csv")
out_df.to_csv(out_path, index=False)
print("[OK] wrote", out_path)

# simple plots (if columns exist)
def plot_metric(ycol, ylabel, fname):
    if ycol not in out_df.columns:
        print("[SKIP] missing", ycol)
        return
    plt.figure()
    plt.plot(out_df["act_loss_weight"], out_df[ycol], marker="o")
    plt.xlabel("act_loss_weight")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUT_SUM, fname), dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] wrote", fname)

plot_metric(
    "active_organ_fraction_mae_mean",
    "Active Organ Fraction MAE (mean over test phantoms)",
    "metric_vs_actw_active_organ_fraction_mae.png",
)
plot_metric("inactive_organs_pred_frac_of_pred_mean", "inactive_organs_pred_frac_of_pred (mean)", "metric_vs_actw_inactive_frac.png")
plot_metric("pred_outside_mask_frac_mean", "pred_outside_mask_frac (mean)", "metric_vs_actw_outside_body_frac.png")
plot_metric("proj_mae_counts_mean", "proj_mae_counts (mean)", "metric_vs_actw_proj_mae.png")
plot_metric("proj_poisson_dev_counts_mean", "proj_poisson_dev_counts (mean)", "metric_vs_actw_proj_dev.png")
