# Training loop for SPECT-like volumetric reconstruction (CT + AP/PA -> activity volume)

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
import torch.nn.functional as F
import trainlib
import util
import numpy as np
from data import get_split_dataset
from model import make_model


def extra_args(parser):
    parser.add_argument(
        "--batch_size", "-B", type=int, default=1, help="Batch size (volumes per step)"
    )
    parser.add_argument(
        "--loss", type=str, default="mse", choices=["mse", "l1"], help="Reconstruction loss"
    )
    return parser


args, conf = util.args.parse_args(
    extra_args,
    training=True,
    default_conf="conf/exp/spect.conf",
    default_expname="spect",
    default_data_format="spect",
    default_ray_batch_size=1,
    default_num_epochs=1000,
)
device = util.get_cuda(args.gpu_id[0])

# Load datasets
dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir)

net = make_model(conf["model"]).to(device=device)


class SpectTrainer(trainlib.Trainer):
    def __init__(self):
        super().__init__(net, dset, val_dset, args, conf["train"], device=device)
        self.criterion = torch.nn.L1Loss() if args.loss == "l1" else torch.nn.MSELoss()

    def train_step(self, data, global_step):
        ap = data["ap"].to(device)
        pa = data["pa"].to(device)
        ct = data["ct"].to(device)
        act = data["act"].to(device)

        pred = net(ap, pa, ct)
        loss = self.criterion(pred, act)
        loss.backward()
        return {"loss": loss.item()}

    def eval_step(self, data, global_step):
        net.eval()
        with torch.no_grad():
            ap = data["ap"].to(device)
            pa = data["pa"].to(device)
            ct = data["ct"].to(device)
            act = data["act"].to(device)
            pred = net(ap, pa, ct)
            loss = self.criterion(pred, act)
        net.train()
        return {"loss": loss.item()}

    def vis_step(self, data, global_step, idx=None):
        # Show only GT vs predicted projections (AP, PA)
        ap = data["ap"].to(device)
        pa = data["pa"].to(device)
        ct = data["ct"].to(device)
        with torch.no_grad():
            pred = net(ap, pa, ct)

        def norm_img(x):
            x = x.astype("float32")
            x = x - x.min()
            m = x.max()
            if m <= 1e-8:
                return x  # flat image; keep constant value for visualization
            return x / m

        # GT projections (coronal orientation, same as helper script)
        ap_np = norm_img(ap[0, 0].cpu().numpy())
        pa_np = norm_img(pa[0, 0].cpu().numpy())
        ap_np = np.rot90(np.flipud(ap_np), k=-1)
        pa_np = np.rot90(np.flipud(pa_np), k=-1)

        # Predicted projections: SUM over AP axis (dim=2 for (B, SI, AP, LR))
        pred_vol = pred.detach().cpu()
        proj_ap_pred = pred_vol.sum(dim=2)[0].numpy()  # (SI, LR)
        proj_pa_pred = pred_vol.flip(dims=[2]).sum(dim=2)[0].numpy()  # flip AP, then sum
        # Quick range check
        print(
            "Pred AP range:",
            proj_ap_pred.min(),
            proj_ap_pred.max(),
            "nonzero:",
            np.count_nonzero(proj_ap_pred),
        )
        ap_pred_np = norm_img(proj_ap_pred)
        pa_pred_np = norm_img(proj_pa_pred)
        # Rotate like GT; no extra flip to avoid inversion
        ap_pred_np = np.rot90(ap_pred_np, k=-1)
        pa_pred_np = np.rot90(pa_pred_np, k=-1)

        def to_rgb(x):
            return np.stack([x, x, x], axis=-1)

        ap_rgb = to_rgb(ap_np)
        pa_rgb = to_rgb(pa_np)
        ap_pred_rgb = to_rgb(ap_pred_np)
        pa_pred_rgb = to_rgb(pa_pred_np)

        # Pad rows to same height/width without scaling
        max_h = max(ap_rgb.shape[0], pa_rgb.shape[0], ap_pred_rgb.shape[0], pa_pred_rgb.shape[0])
        max_w_half = max(ap_rgb.shape[1], pa_rgb.shape[1], ap_pred_rgb.shape[1], pa_pred_rgb.shape[1])

        def pad_to(img, target_h, target_w):
            h, w = img.shape[:2]
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            return np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="constant")

        ap_rgb = pad_to(ap_rgb, max_h, max_w_half)
        pa_rgb = pad_to(pa_rgb, max_h, max_w_half)
        ap_pred_rgb = pad_to(ap_pred_rgb, max_h, max_w_half)
        pa_pred_rgb = pad_to(pa_pred_rgb, max_h, max_w_half)

        row_gt = np.concatenate([ap_rgb, pa_rgb], axis=1)
        row_pred = np.concatenate([ap_pred_rgb, pa_pred_rgb], axis=1)
        target_w = max(row_gt.shape[1], row_pred.shape[1])
        row_gt = pad_to(row_gt, max_h, target_w)
        row_pred = pad_to(row_pred, max_h, target_w)

        vis = np.concatenate([row_gt, row_pred], axis=0)
        return vis, {}


trainer = SpectTrainer()
trainer.start()
