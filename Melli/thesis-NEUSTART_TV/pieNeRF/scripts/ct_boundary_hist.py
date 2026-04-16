#!/usr/bin/env python3
"""CT boundary depth histogram along AP rays."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from graf.config import get_data, build_models
from nerf.run_nerf_helpers_mod import get_rays_ortho
from nerf.run_nerf_mod import sample_ct_volume


def parse_args():
    parser = argparse.ArgumentParser(description="CT boundary depth histogram along AP rays.")
    parser.add_argument("--config", type=str, default="configs/spect.yaml", help="Config YAML path.")
    parser.add_argument("--num-rays", type=int, default=4096, help="Number of random AP rays to sample.")
    parser.add_argument("--mu-threshold", type=float, default=1e-3, help="CT mu threshold for boundary.")
    parser.add_argument("--padding-mode", type=str, default="border", choices=["border", "zeros"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save-hist", type=str, default="", help="Optional output PNG for histogram.")
    parser.add_argument("--save-npz", type=str, default="", help="Optional output NPZ for depths.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset, hwfr, _ = get_data(config)
    config["data"]["hwfr"] = hwfr
    generator = build_models(config)
    generator.set_fixed_ap_pa(radius=hwfr[3])
    generator.to(args.device)
    generator.eval()

    sample = dataset[0]
    ct_vol = sample.get("ct")
    if ct_vol is None or ct_vol.numel() == 0:
        raise RuntimeError("CT volume missing in dataset sample.")
    ct_vol = ct_vol.to(args.device, non_blocking=True).float()
    ct_context = generator.build_ct_context(ct_vol, padding_mode=args.padding_mode)

    H, W = generator.H, generator.W
    size_h, size_w = generator.ortho_size
    rays_o, rays_d = get_rays_ortho(H, W, generator.pose_ap.to(args.device), size_h, size_w)
    rays_o = rays_o.view(-1, 3)
    rays_d = rays_d.view(-1, 3)

    num_pixels = H * W
    num_rays = min(args.num_rays, num_pixels)
    idx = torch.randperm(num_pixels, device=args.device)[:num_rays]
    rays_o = rays_o[idx]
    rays_d = rays_d[idx]

    near = float(config["data"].get("near", 0.0))
    far = float(config["data"].get("far", 1.0))
    N_samples = int(config["nerf"].get("N_samples", 96))

    t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=args.device)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand(num_rays, N_samples)

    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]
    mu_vals = sample_ct_volume(pts, ct_context)

    if mu_vals is None:
        raise RuntimeError("CT sampling failed; mu_vals is None.")

    mask = mu_vals > float(args.mu_threshold)
    valid = mask.any(dim=-1)
    if not valid.any():
        raise RuntimeError("No boundaries found with given mu_threshold.")

    first_idx = mask.float().argmax(dim=-1)
    depths = first_idx[valid].float() / max(1.0, float(N_samples - 1))
    depths_cpu = depths.detach().cpu().numpy()

    stats = {
        "count": int(depths_cpu.size),
        "mean": float(np.mean(depths_cpu)),
        "median": float(np.median(depths_cpu)),
        "p10": float(np.quantile(depths_cpu, 0.10)),
        "p90": float(np.quantile(depths_cpu, 0.90)),
    }

    print("[ct-boundary] samples=%d" % stats["count"])
    print("[ct-boundary] mean=%.4f median=%.4f p10=%.4f p90=%.4f" % (stats["mean"], stats["median"], stats["p10"], stats["p90"]))

    hist, edges = np.histogram(depths_cpu, bins=40, range=(0.0, 1.0))
    peak_bin = int(np.argmax(hist))
    peak_center = 0.5 * (edges[peak_bin] + edges[peak_bin + 1])
    print("[ct-boundary] peak_bin_center=%.4f count=%d" % (peak_center, int(hist[peak_bin])))

    if args.save_npz:
        out = Path(args.save_npz)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, depths=depths_cpu, hist=hist, edges=edges)

    if args.save_hist:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("[ct-boundary] matplotlib not available; skipping PNG.")
        else:
            plt.figure(figsize=(6, 4))
            plt.hist(depths_cpu, bins=40, range=(0.0, 1.0))
            plt.xlabel("normalized depth")
            plt.ylabel("count")
            plt.title("CT boundary depth histogram")
            out = Path(args.save_hist)
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(out, dpi=150)
            plt.close()


if __name__ == "__main__":
    main()
