"""
Minimal training script for SpectPixelPieNeRF.

Loads SpectDataset, trains with MSE(AP/PA), logs to stdout, saves visuals and checkpoints.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure repository root and pixel-nerf sources are importable before any local imports.
_ROOT = Path(__file__).resolve().parents[1]
_PIXELNERF_ROOT = _ROOT / "pixel-nerf"
_PIXELNERF_SRC = _PIXELNERF_ROOT / "src"

# Ensure our project root has priority, then append pixel-nerf paths.
if (p_str := str(_ROOT)) not in sys.path:
    sys.path.insert(0, p_str)
for p in (_PIXELNERF_ROOT, _PIXELNERF_SRC):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.append(p_str)

import matplotlib
import numpy as np

# Headless backend for PNG saving.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch import nn, optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.SpectDataset import SpectDataset  # noqa: E402
from model.spect_pixel_pienerf import SpectPixelPieNeRF  # noqa: E402


def _resolve_data_root(data_root: str) -> Path:
    """Resolve data_root with a few fallbacks for convenience."""
    root = Path(__file__).resolve().parents[1]
    candidates = [
        Path(data_root),
        root / data_root,
        root.parent / "pieNeRF" / "data",  # common sibling path
    ]
    for c in candidates:
        if c.exists():
            return c
    tried = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find data_root. Tried:\n{tried}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SpectPixelPieNeRF (simple loop).")
    parser.add_argument("--data_root", type=str, default="thesis_med/pieNeRF/data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--vis_interval", type=int, default=100)
    parser.add_argument(
        "--vis_index",
        type=int,
        default=0,
        help="Visualize/depth-profile only this batch index to avoid looping over full batches.",
    )
    parser.add_argument("--ckpt_interval", type=int, default=200)
    parser.add_argument("--run_name", type=str, default="spect_pienerf_pixelnerf")
    parser.add_argument(
        "--target_hw",
        type=int,
        nargs=2,
        default=[64, 128],
        metavar=("H", "W"),
        help="Resize AP/PA to (H,W) to control memory. Defaults to 64x128 for lighter training.",
    )
    parser.add_argument(
        "--target_depth",
        type=int,
        default=64,
        help="Resize CT/ACT depth to control memory. Default 64 for lighter training.",
    )
    parser.add_argument(
        "--depth_profile_interval",
        type=int,
        default=0,
        help="If >0, save depth profile (GT act/ct vs pred sigma) every N steps at max-intensity pixel.",
    )
    parser.add_argument("--use_attenuation", action="store_true", help="Enable attenuation in forward projection.")
    parser.add_argument(
        "--mu_scale",
        type=float,
        default=0.01,
        help="Scale factor applied to positive CT values to form mu_volume when attenuation is enabled.",
    )
    parser.add_argument(
        "--step_len",
        type=float,
        default=1.0,
        help="Step length for attenuation integration inside forward_spect.",
    )
    parser.add_argument(
        "--w_act",
        type=float,
        default=0.0,
        help="Weight for optional ACT L1 loss between predicted sigma_volume and GT activity.",
    )
    parser.add_argument(
        "--w_reg_sigma",
        type=float,
        default=0.0,
        help="Weight for L2 regularization on predicted sigma_volume.",
    )
    return parser.parse_args()


def ensure_dirs(run_name: str):
    log_dir = Path("logs") / run_name
    ckpt_dir = Path("checkpoints") / run_name
    vis_dir = Path("visuals") / run_name
    for d in (log_dir, ckpt_dir, vis_dir):
        d.mkdir(parents=True, exist_ok=True)
    return log_dir, ckpt_dir, vis_dir


def get_dataloader(data_root: str, batch_size: int, target_hw: tuple[int, int], target_depth: int):
    resolved_root = _resolve_data_root(data_root)
    print(f"Using data_root: {resolved_root}")
    dataset = SpectDataset(
        datadir=str(resolved_root),
        stage="train",
        target_hw=target_hw,
        target_depth=target_depth,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    first = next(iter(loader))
    print(f"SpectDataset batch keys: {list(first.keys())}")
    print(f"Batch shapes: ct {tuple(first['ct'].shape)}, ap {tuple(first['ap'].shape)}, pa {tuple(first['pa'].shape)}")
    return dataset, loader, first


def make_model(first_batch, device: torch.device, mu_scale: float, step_len: float):
    ct = first_batch["ct"]
    # ct comes as (B, D, H, W); extract volume shape (D, H, W).
    if ct.dim() == 4:
        volume_shape = tuple(ct.shape[1:])
    elif ct.dim() == 5:
        volume_shape = tuple(ct.shape[2:])
    else:
        raise ValueError(f"Unexpected CT shape: {ct.shape}")

    model = SpectPixelPieNeRF(
        volume_shape=volume_shape,
        voxel_size=1.0,
        num_samples=volume_shape[1],
        latent_dim=512,
        mlp_width=256,
        mlp_depth=8,
        multires_xyz=6,
        mu_scale=mu_scale,
        step_len=step_len,
    ).to(device)
    return model


def _rotate_for_vis(img_np, rotate: bool):
    if not rotate:
        return img_np
    return np.rot90(np.flipud(img_np), k=-1)


def save_visual(ap, pa, ap_pred, pa_pred, vis_path: Path, rotate: bool):
    # Normalize predictions before visualization.
    ap_pred = (ap_pred - ap_pred.min()) / (ap_pred.max() - ap_pred.min() + 1e-8)
    pa_pred = (pa_pred - pa_pred.min()) / (pa_pred.max() - pa_pred.min() + 1e-8)

    def _to_np(x):
        x = x.detach().cpu().squeeze()
        return x.numpy()

    ap_np = _rotate_for_vis(_to_np(ap), rotate)
    pa_np = _rotate_for_vis(_to_np(pa), rotate)
    ap_pred_np = _rotate_for_vis(_to_np(ap_pred), rotate)
    pa_pred_np = _rotate_for_vis(_to_np(pa_pred), rotate)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(ap_np, cmap="gray")
    axes[0, 0].set_title("AP GT")
    axes[0, 1].imshow(pa_np, cmap="gray")
    axes[0, 1].set_title("PA GT")
    axes[1, 0].imshow(ap_pred_np, cmap="gray")
    axes[1, 0].set_title("AP Pred")
    axes[1, 1].imshow(pa_pred_np, cmap="gray")
    axes[1, 1].set_title("PA Pred")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(vis_path, dpi=150)
    plt.close(fig)


def save_depth_profile(ap, pa, sigma_volume, act, ct, vis_path: Path, batch_index: int = 0):
    """
    Save depth profile at max-intensity pixel (sum of AP+PA GT).
    Plots GT activity (if available), predicted sigma, and CT (if available) along AP depth.
    Always writes a file (with a note if no curves available). Returns True if curves were found.
    """
    def _pick_batch(tensor):
        if tensor is None or not torch.is_tensor(tensor):
            return tensor
        if tensor.dim() >= 4 and tensor.shape[0] > 1:
            return tensor[batch_index : batch_index + 1]
        return tensor

    ap = _pick_batch(ap)
    pa = _pick_batch(pa)
    sigma_volume = _pick_batch(sigma_volume)
    act = _pick_batch(act)
    ct = _pick_batch(ct)

    curves_found = False
    with torch.no_grad():
        weight_proj = (ap + pa).detach().cpu()  # (1,1,SI,LR)
        w2d = weight_proj[0, 0]

        # Prefer a pixel where GT activity is present; fallback to AP+PA max.
        act_map = None
        if act is not None and act.numel() > 0:
            act_cpu = act.detach().cpu()
            while act_cpu.dim() > 3:
                act_cpu = act_cpu.squeeze(0)
            if act_cpu.dim() == 3:
                # Collapse AP for activity map (SI, LR)
                act_map = act_cpu.sum(dim=1)

        if act_map is not None and act_map.numel() > 0 and act_map.max() > 0:
            flat_idx = torch.argmax(act_map)
            si_idx = int(flat_idx // act_map.shape[1])
            lr_idx = int(flat_idx % act_map.shape[1])
            picked_from = "act"
        else:
            flat_idx = torch.argmax(w2d)
            si_idx = int(flat_idx // w2d.shape[1])
            lr_idx = int(flat_idx % w2d.shape[1])
            picked_from = "ap+pa"

        depth_len = sigma_volume.shape[3]  # AP dimension
        depth_axis = np.linspace(0.0, 1.0, depth_len)

    si_target, lr_target = w2d.shape

    def _stats(t):
        if t is None or t.numel() == 0:
            return None
        t_cpu = t.detach().cpu()
        return (float(t_cpu.min()), float(t_cpu.max()))

    debug_info = {
        "ap_shape": tuple(ap.shape),
        "pa_shape": tuple(pa.shape),
        "sigma_shape": tuple(sigma_volume.shape) if sigma_volume is not None else None,
        "act_shape": tuple(act.shape) if act is not None else None,
        "ct_shape": tuple(ct.shape) if ct is not None else None,
        "ap_minmax": _stats(ap),
        "pa_minmax": _stats(pa),
        "sigma_minmax": _stats(sigma_volume),
        "act_minmax": _stats(act),
        "ct_minmax": _stats(ct),
        "si_idx": si_idx,
        "lr_idx": lr_idx,
        "depth_len": depth_len,
        "si_target": si_target,
        "lr_target": lr_target,
        "picked_from": picked_from,
    }

    def _curve_from(vol, name):
        if vol is None or vol.numel() == 0:
            debug_info[f"{name}_reason"] = "empty"
            return None
        v = vol.detach().cpu()
        # Squeeze batch/channel dims until 3D or fewer.
        while v.dim() > 3:
            v = v.squeeze(0)
        if v.dim() != 3:
            debug_info[f"{name}_reason"] = f"dim{v.dim()}"
            return None
        # Direct guess: (SI, AP, LR)
        if v.shape[0] == si_target and v.shape[2] == lr_target and v.shape[1] == depth_len:
            if si_idx < v.shape[0] and lr_idx < v.shape[2]:
                c = v[si_idx, :, lr_idx].numpy()
                goto_norm = True
            else:
                goto_norm = False
        # Alt guess: (AP, SI, LR)
        elif v.shape[1] == si_target and v.shape[2] == lr_target and v.shape[0] == depth_len:
            if si_idx < v.shape[1] and lr_idx < v.shape[2]:
                c = v[:, si_idx, lr_idx].numpy()
                goto_norm = True
            else:
                goto_norm = False
        else:
            # Fallback: try permutations
            c = None
            goto_norm = False
            for si_axis, ap_axis, lr_axis in [
                (0, 1, 2),
                (0, 2, 1),
                (1, 0, 2),
                (1, 2, 0),
                (2, 0, 1),
                (2, 1, 0),
            ]:
                if v.shape[ap_axis] != depth_len or v.shape[lr_axis] != lr_target:
                    continue
                if si_idx >= v.shape[si_axis] or lr_idx >= v.shape[lr_axis]:
                    continue
                c = v.permute(si_axis, ap_axis, lr_axis)[si_idx, :, lr_idx].numpy()
                goto_norm = True
                break
        if not goto_norm or c is None:
            debug_info[f"{name}_reason"] = "no_axis_match"
            return None
        c = c - c.min()
        denom = c.max() + 1e-8
        debug_info[f"{name}_reason"] = "ok"
        return c / denom

    curves = []
    labels = []
    act_curve = _curve_from(act, "act")
    if act_curve is not None:
        curves.append(act_curve)
        labels.append("Aktivität (GT)")
    sigma_curve = _curve_from(sigma_volume, "sigma")
    if sigma_curve is not None:
        curves.append(sigma_curve)
        labels.append("Sigma (Pred)")
    ct_curve = _curve_from(ct, "ct")
    if ct_curve is not None:
        curves.append(ct_curve)
        labels.append("μ (CT)")

    vis_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    if curves:
        curves_found = True
        for c, lbl in zip(curves, labels):
            plt.plot(depth_axis, c, label=lbl)
        plt.title(f"Depth profile @ (SI={si_idx}, LR={lr_idx})")
        plt.xlabel("Depth (AP, normalized)")
        plt.ylabel("Normalized intensity")
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.2)
        plt.legend(loc="upper right", fontsize=8)
    else:
        plt.text(0.5, 0.5, "No curves available", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(vis_path, dpi=150)
    plt.close()
    # Always write debug log for inspection.
    log_path = vis_path.with_suffix(".log")
    with open(log_path, "w") as f:
        for k, v in debug_info.items():
            f.write(f"{k}: {v}\n")
        f.write(f"curves_found: {curves_found}\n")
    return curves_found


def train():
    args = parse_args()
    log_dir, ckpt_dir, vis_dir = ensure_dirs(args.run_name)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_hw = (args.target_hw[0], args.target_hw[1])
    dataset, loader, first_batch = get_dataloader(args.data_root, args.batch_size, target_hw, args.target_depth)
    model = make_model(first_batch, device, mu_scale=args.mu_scale, step_len=args.step_len)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    rotate_for_vis = getattr(dataset, "rotate_projections", False)
    use_attenuation = args.use_attenuation

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            ct = batch["ct"]
            ap = batch["ap"]
            pa = batch["pa"]
            act = batch.get("act")

            # Ensure channel dims exist.
            if ct.dim() == 4:
                ct = ct.unsqueeze(1)
            if ap.dim() == 3:
                ap = ap.unsqueeze(1)
            if pa.dim() == 3:
                pa = pa.unsqueeze(1)
            if act is not None and torch.is_tensor(act) and act.numel() > 0:
                if act.dim() == 3:
                    act = act.unsqueeze(0)

            ct = ct.to(device, non_blocking=True)
            ap = ap.to(device, non_blocking=True)
            pa = pa.to(device, non_blocking=True)
            if act is not None and torch.is_tensor(act) and act.numel() > 0:
                act = act.to(device, non_blocking=True)
            else:
                act = None

            optimizer.zero_grad(set_to_none=True)
            out = model(ct, ap, pa, use_attenuation=use_attenuation, step_len=args.step_len)

            ap_pred = out["ap_pred"]
            pa_pred = out["pa_pred"]

            loss_ap = criterion(ap_pred, ap)
            loss_pa = criterion(pa_pred, pa)
            loss = loss_ap + loss_pa

            loss_act = None
            if args.w_act > 0.0 and act is not None and act.numel() > 0:
                act_t = act
                if act_t.dim() == 3:
                    act_t = act_t.unsqueeze(0)
                if act_t.dim() == 4:
                    act_t = act_t.unsqueeze(1)
                loss_act = torch.mean(torch.abs(out["sigma_volume"] - act_t))
                loss = loss + args.w_act * loss_act

            loss_reg = None
            if args.w_reg_sigma > 0.0:
                loss_reg = torch.mean(out["sigma_volume"] ** 2)
                loss = loss + args.w_reg_sigma * loss_reg

            loss.backward()
            optimizer.step()

            if global_step % args.log_interval == 0:
                msg = (
                    f"[epoch {epoch} step {global_step}] "
                    f"loss={loss.item():.6f} ap={loss_ap.item():.6f} pa={loss_pa.item():.6f}"
                )
                if loss_act is not None:
                    msg += f" act={loss_act.item():.6f} (w={args.w_act})"
                if loss_reg is not None:
                    msg += f" reg_sigma={loss_reg.item():.6f} (w={args.w_reg_sigma})"
                print(msg)

            if global_step % args.vis_interval == 0:
                vis_path = vis_dir / f"step_{global_step:06d}.png"
                vis_idx = min(max(args.vis_index, 0), ap.shape[0] - 1)
                save_visual(ap[vis_idx], pa[vis_idx], ap_pred[vis_idx], pa_pred[vis_idx], vis_path, rotate_for_vis)
                print(f"Saved visualization to {vis_path}")

            if args.depth_profile_interval > 0 and global_step % args.depth_profile_interval == 0:
                dp_path = vis_dir / f"depth_profile_{global_step:06d}.png"
                vis_idx = min(max(args.vis_index, 0), ap.shape[0] - 1)
                saved = save_depth_profile(
                    ap,
                    pa,
                    out.get("sigma_volume"),
                    batch.get("act"),
                    ct,
                    dp_path,
                    batch_index=vis_idx,
                )
                if saved:
                    print(f"Saved depth profile to {dp_path}")
                else:
                    print(f"Skipped depth profile at step {global_step} (no curves)")

            if global_step % args.ckpt_interval == 0 and global_step > 0:
                ckpt_path = ckpt_dir / f"step_{global_step:06d}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint to {ckpt_path}")

            global_step += 1

    print("Training completed.")


def main():
    train()


if __name__ == "__main__":
    main()
