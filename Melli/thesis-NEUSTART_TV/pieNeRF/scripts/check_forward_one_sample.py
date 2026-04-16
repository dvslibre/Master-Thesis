#!/usr/bin/env python3
"""Forward-only sanity check for one sample."""

import argparse
import sys

import numpy as np
import torch
import yaml

from graf.config import get_data, build_models


def parse_args():
    parser = argparse.ArgumentParser(description="Forward-only sanity check for one sample.")
    parser.add_argument("--config", type=str, default="configs/spect.yaml", help="Path to YAML config.")
    parser.add_argument("--index", type=int, default=0, help="Sample index.")
    parser.add_argument("--outdir", type=str, default=None, help="Optional output directory for debug images.")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset, hwfr, _ = get_data(config)
    generator = build_models(config)
    generator.eval()
    generator.use_test_kwargs = True

    sample = dataset[args.index]
    ap = sample["ap"].squeeze(0).numpy()
    pa = sample["pa"].squeeze(0).numpy()
    ct = sample["ct"]
    meta = sample.get("meta", {})
    mu_scale = float(meta.get("mu_scale_p99_9", meta.get("ct_scale_p99_9", 1.0)))
    voxel_size_mm = float(meta.get("voxel_size_mm", 1.5))

    device = generator.device
    z_dim = config["z_dist"]["dim"]
    z = torch.zeros(1, z_dim, device=device)

    ct_context = None
    if ct is not None and ct.numel() > 0:
        ct_context = generator.build_ct_context(
            ct.to(device),
            padding_mode="border",
            voxel_size_mm=voxel_size_mm,
            mu_scale_p99_9=mu_scale,
        )

    # Build full rays for AP pose
    focal_or_size = generator.ortho_size if generator.orthographic else generator.focal
    rays, _, _ = generator.val_ray_sampler(generator.H, generator.W, focal_or_size, generator.pose_ap)
    rays = rays.to(device, non_blocking=True)

    render_kwargs = dict(generator.render_kwargs_test)
    render_kwargs["features"] = z
    render_kwargs["retraw"] = True
    if ct_context is not None:
        render_kwargs["ct_context"] = ct_context

    with torch.no_grad():
        proj, _, _, extras = generator.render(rays=rays, **render_kwargs)

    if not torch.isfinite(proj).all():
        print("NaN/Inf in projection.", flush=True)
        return 1

    if isinstance(extras, dict):
        mu = extras.get("mu")
        z_vals = extras.get("z_vals")
        dists = extras.get("dists")
        if mu is not None and z_vals is not None and ct_context is not None:
            if dists is None:
                dists = z_vals[..., 1:] - z_vals[..., :-1]
                dists = torch.cat([dists, dists[..., -1:].clone()], dim=-1)

            rays_d = rays[1]
            ray_norm = torch.norm(rays_d[..., None, :], dim=-1)
            dir_norm = rays_d / (ray_norm + 1e-8)
            dir_abs = torch.abs(dir_norm)
            sx, sy, sz = ct_context["world_scale_cm_xyz"]
            scale = dir_abs[..., 0] * sx + dir_abs[..., 1] * sy + dir_abs[..., 2] * sz
            dists_cm = dists * scale[..., None]
            mu_phys = mu * mu_scale
            attenuation = torch.cumsum(mu_phys * dists_cm, dim=-1)
            attenuation = torch.nn.functional.pad(attenuation[..., :-1], (1, 0), mode="constant", value=0.0)
            transmission = torch.exp(-attenuation)
            t_mean = float(transmission.mean().detach().cpu().item())
            if t_mean < 1e-6:
                print(f"Transmission collapsed (mean={t_mean:.3e}).", flush=True)
                return 1

    if args.outdir:
        import os
        import matplotlib.pyplot as plt

        os.makedirs(args.outdir, exist_ok=True)
        H, W = generator.H, generator.W
        pred_ap = proj[0].reshape(H, W).detach().cpu().numpy()
        plt.imsave(os.path.join(args.outdir, "pred_ap_norm.png"), pred_ap, cmap="gray")
        plt.imsave(os.path.join(args.outdir, "target_ap_norm.png"), ap, cmap="gray")

    print("OK: forward-only check passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
