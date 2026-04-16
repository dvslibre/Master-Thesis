"""
Hybrid orientation check:
- Loads one sample via SpectDataset.
- Runs RayNet-aligned forward_spect on ACT.
- Compares AP/PA against dataset GT (same orientation expected).
Saves visuals/orient_hybrid_vs_gt.png and prints MSE.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PIXELNERF_ROOT = ROOT / "pixel-nerf"
PIXELNERF_SRC = PIXELNERF_ROOT / "src"
RAYNET_CODE = Path("/home/mnguest12/projects/thesis/RayNet/code")

for p in (ROOT, PIXELNERF_ROOT, PIXELNERF_SRC, RAYNET_CODE):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from src.data.SpectDataset import SpectDataset  # noqa: E402
from forward.spect_operator_wrapper import forward_spect  # noqa: E402


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x - x.min()
    return x / (x.max() + 1e-8)


def main():
    data_root = "/home/mnguest12/projects/thesis/pieNeRF/data"
    ds = SpectDataset(datadir=data_root, stage="train")
    sample = ds[0]
    ap_gt = sample["ap"][0].numpy()
    pa_gt = sample["pa"][0].numpy()
    act = sample["act"]  # (SI, AP, LR) torch

    sigma_volume = act.unsqueeze(0).unsqueeze(0)  # (1,1,SI,AP,LR)
    with torch.no_grad():
        ap_fwd, pa_fwd = forward_spect(sigma_volume)

    ap_fwd = ap_fwd[0].cpu().numpy()
    pa_fwd = pa_fwd[0].cpu().numpy()

    ap_gt_n = _normalize(ap_gt)
    pa_gt_n = _normalize(pa_gt)
    ap_fwd_n = _normalize(ap_fwd)
    pa_fwd_n = _normalize(pa_fwd)

    mse_ap = float(np.mean((ap_gt_n - ap_fwd_n) ** 2))
    mse_pa = float(np.mean((pa_gt_n - pa_fwd_n) ** 2))
    print(f"MSE AP={mse_ap:.6e}, PA={mse_pa:.6e}")

    # Visual diff
    diff_ap = np.abs(ap_gt_n - ap_fwd_n)
    diff_pa = np.abs(pa_gt_n - pa_fwd_n)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0, 0].imshow(ap_gt_n, cmap="gray"); axs[0, 0].set_title("AP GT"); axs[0, 0].axis("off")
    axs[0, 1].imshow(ap_fwd_n, cmap="gray"); axs[0, 1].set_title("AP forward_spect"); axs[0, 1].axis("off")
    axs[0, 2].imshow(diff_ap, cmap="magma"); axs[0, 2].set_title("|AP diff|"); axs[0, 2].axis("off")

    axs[1, 0].imshow(pa_gt_n, cmap="gray"); axs[1, 0].set_title("PA GT"); axs[1, 0].axis("off")
    axs[1, 1].imshow(pa_fwd_n, cmap="gray"); axs[1, 1].set_title("PA forward_spect"); axs[1, 1].axis("off")
    axs[1, 2].imshow(diff_pa, cmap="magma"); axs[1, 2].set_title("|PA diff|"); axs[1, 2].axis("off")

    plt.tight_layout()
    out_dir = ROOT / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "orient_hybrid_vs_gt.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
