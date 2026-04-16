# NeRF-style training for SPECT projections (AP/PA)

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
import torch.nn.functional as F
import numpy as np
import trainlib
import util
from data import get_split_dataset
from model import make_model
from render import NeRFRenderer


def extra_args(parser):
    parser.add_argument("--batch_size", "-B", type=int, default=1, help="Batch size")
    parser.add_argument("--n_samples", type=int, default=8192, help="Ray batch size for rendering")
    parser.add_argument("--vis_interval", type=int, default=50, help="Visualization interval")
    return parser


args, conf = util.args.parse_args(
    extra_args,
    training=True,
    default_conf="conf/exp/spect_nerf.conf",
    default_expname="spect_nerf",
    default_data_format="spect",
    default_ray_batch_size=8192,
)
device = util.get_cuda(args.gpu_id[0])

dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir)

net = make_model(conf["model"]).to(device=device)
renderer = NeRFRenderer.from_conf(conf["renderer"], lindisp=False).to(device=device)
render_par = renderer.bind_parallel(net, args.gpu_id).eval()


def make_ortho_rays(H, W, direction="ap", near=0.0, far=1.0, device="cpu"):
    """
    Build orthographic rays.
    Convention: volume axes (SI, AP, LR) map to (y, z, x) here:
    - AP axis corresponds to z
    - LR axis corresponds to x
    - SI axis corresponds to y
    AP view: direction +AP (+z), PA view: direction -AP (-z).
    """
    y, x = torch.meshgrid(
        torch.linspace(-0.5, 0.5, H, device=device),
        torch.linspace(-0.5, 0.5, W, device=device),
        indexing="ij",
    )
    if direction == "ap":
        dirs = torch.stack([torch.zeros_like(x), torch.zeros_like(x), torch.ones_like(x)], dim=-1)
    else:
        dirs = torch.stack([torch.zeros_like(x), torch.zeros_like(x), -torch.ones_like(x)], dim=-1)
    origins = torch.stack([x, y, torch.zeros_like(x)], dim=-1)
    rays = torch.cat(
        [origins.reshape(-1, 3), dirs.reshape(-1, 3), torch.full((H * W, 1), near, device=device), torch.full((H * W, 1), far, device=device)],
        dim=-1,
    )
    return rays


def vis_step_spect_nerf(global_step, ap_pred, pa_pred, ap_gt, pa_gt, out_dir="visuals/spect_nerf"):
    os.makedirs(out_dir, exist_ok=True)

    def norm_img(x):
        x = x.astype(np.float32)
        x = x - x.min()
        m = x.max()
        return x / m if m > 0 else x

    def to_vis(img):
        img = norm_img(img)
        img = np.rot90(np.flipud(img), k=-1)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        return np.stack([img] * 3, axis=-1)

    ap_gt_v = to_vis(ap_gt)
    pa_gt_v = to_vis(pa_gt)
    ap_pr_v = to_vis(ap_pred)
    pa_pr_v = to_vis(pa_pred)

    H = max(ap_gt_v.shape[0], pa_gt_v.shape[0], ap_pr_v.shape[0], pa_pr_v.shape[0])
    def pad_h(img):
        if img.shape[0] == H:
            return img
        pad = H - img.shape[0]
        top = pad // 2
        bottom = pad - top
        return np.pad(img, ((top, bottom), (0, 0), (0, 0)), mode="constant")

    ap_gt_v, pa_gt_v, ap_pr_v, pa_pr_v = map(pad_h, [ap_gt_v, pa_gt_v, ap_pr_v, pa_pr_v])
    row_gt = np.concatenate([ap_gt_v, pa_gt_v], axis=1)
    row_pr = np.concatenate([ap_pr_v, pa_pr_v], axis=1)
    target_w = max(row_gt.shape[1], row_pr.shape[1])
    def pad_w(img):
        if img.shape[1] == target_w:
            return img
        pad = target_w - img.shape[1]
        left = pad // 2
        right = pad - left
        return np.pad(img, ((0, 0), (left, right), (0, 0)), mode="constant")
    row_gt, row_pr = pad_w(row_gt), pad_w(row_pr)

    vis = np.concatenate([row_gt, row_pr], axis=0)
    out_path = os.path.join(out_dir, f"step_{global_step}.png")
    import imageio
    imageio.imwrite(out_path, vis)
    return out_path


class SpectNeRFTrainer(trainlib.Trainer):
    def __init__(self):
        super().__init__(net, dset, val_dset, args, conf["train"], device=device)
        self.crit = torch.nn.MSELoss()

    def _prep_batch(self, data):
        ap = data["ap"].to(device)  # (B,1,H,W)
        pa = data["pa"].to(device)
        B, _, H, W = ap.shape
        # Build 3-channel inputs
        ap_rgb = ap.repeat(1, 3, 1, 1)
        pa_rgb = pa.repeat(1, 3, 1, 1)
        src_images = torch.stack([ap_rgb, pa_rgb], dim=1)  # (B,2,3,H,W)
        return src_images, ap, pa, H, W

    def _render_views(self, H, W):
        # AP axis is dim=1 (SI, AP, LR); use +AP for AP view, -AP for PA view
        rays_ap = make_ortho_rays(H, W, "ap", device=device)
        rays_pa = make_ortho_rays(H, W, "pa", device=device)
        rays = torch.stack([rays_ap, rays_pa], dim=0)  # (2, HW, 8)
        return rays

    def train_step(self, data, global_step):
        src_images, ap, pa, H, W = self._prep_batch(data)
        # print shapes once
        if global_step == 0:
            print("CT shape:", data["ct"].shape, "ACT shape:", data["act"].shape)
        net.encode(src_images)
        rays = self._render_views(H, W)
        # sample subset for efficiency
        if args.n_samples < rays.shape[1]:
            idx = torch.randint(0, rays.shape[1], (args.n_samples,), device=device)
            rays = rays[:, idx]
            gt_ap = ap.reshape(ap.shape[0], -1)[:, idx]
            gt_pa = pa.reshape(pa.shape[0], -1)[:, idx]
        else:
            gt_ap = ap.reshape(ap.shape[0], -1)
            gt_pa = pa.reshape(pa.shape[0], -1)
        gt = torch.stack([gt_ap, gt_pa], dim=1)  # (B,2,N)

        render_dict = render_par(rays, want_weights=False)
        rgb = render_dict["coarse"]["rgb"]  # (2, N, 3)
        # compare mean rgb to gt
        pred_int = rgb.mean(dim=-1)
        loss = self.crit(pred_int.unsqueeze(0), gt)
        loss.backward()

        if global_step % args.vis_interval == 0:
            # full renders for vis
            rays_full = self._render_views(H, W)
            render_full = render_par(rays_full, want_weights=False)
            rgb_full = render_full["coarse"]["rgb"]  # (2, HW, 3)
            pred_ap_full = rgb_full[0].mean(dim=-1).reshape(H, W).detach().cpu().numpy()
            pred_pa_full = rgb_full[1].mean(dim=-1).reshape(H, W).detach().cpu().numpy()
            ap_gt_full = ap[0, 0].detach().cpu().numpy()
            pa_gt_full = pa[0, 0].detach().cpu().numpy()
            print(
                "AP pred:",
                pred_ap_full.mean(),
                pred_ap_full.std(),
                "AP gt:",
                ap_gt_full.mean(),
                ap_gt_full.std(),
            )
            vis_step_spect_nerf(global_step, pred_ap_full, pred_pa_full, ap_gt_full, pa_gt_full)
        return {"loss": loss.item()}

    def eval_step(self, data, global_step):
        net.eval()
        with torch.no_grad():
            src_images, ap, pa, H, W = self._prep_batch(data)
            net.encode(src_images)
            rays = self._render_views(H, W)
            gt_ap = ap.reshape(ap.shape[0], -1)
            gt_pa = pa.reshape(pa.shape[0], -1)
            gt = torch.stack([gt_ap, gt_pa], dim=1)
            render_dict = render_par(rays, want_weights=False)
            rgb = render_dict["coarse"]["rgb"]
            pred_int = rgb.mean(dim=-1)
            loss = self.crit(pred_int.unsqueeze(0), gt)
        net.train()
        return {"loss": loss.item()}


trainer = SpectNeRFTrainer()
trainer.start()
