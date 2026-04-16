"""
Orientation reference using RayNet gamma_camera_core (source of truth for ACT→AP/PA).

Volumes:
    - RayNet expects act/atn as (nx, ny, nz) = (LR, AP, SI) before internal rotations.
    - gamma_camera_core internally applies:
        * act_ap = transpose(0,2,1) → rot90 → transpose(1,0,2)
        * act_pa = flip(act_ap, axis=2)
        * proj_AP/PA = _process_view_phys(...)
        * orient_patch: rot90 → flipud → transpose
    - The saved ap.npy/pa.npy from RayNet therefore already include these rotations.

This script loads a sample ACT from SpectDataset, runs gamma_camera_core,
and saves the reference AP/PA projections for comparison.
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
from preprocessing import gamma_camera_core  # noqa: E402


def main():
    data_root = "/home/mnguest12/projects/thesis/pieNeRF/data"
    ds = SpectDataset(datadir=data_root, stage="train")
    sample = ds[0]
    act = sample["act"].numpy()  # (SI, AP, LR)

    # RayNet expects (LR, AP, SI).
    act_xyz = act.transpose(2, 1, 0).astype(np.float32)
    nx, ny, nz = act_xyz.shape
    dummy_mu = np.zeros_like(act_xyz, dtype=np.float32)
    kernel_dummy = np.zeros((1, 1, 1), dtype=np.float32)

    ap_np, pa_np = gamma_camera_core(
        act_xyz,
        dummy_mu,
        view="frontal",
        comp_scatter=False,
        atn_on=False,
        coll_on=False,
        kernel_mat=kernel_dummy,
        nx=nx,
        ny=ny,
        nz=nz,
        sigma=0.0,
        z0_slices=0,
        step_len=1.0,
    )

    # Save reference visuals.
    out_dir = ROOT / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_dir / "orient_ref_ap.png", ap_np, cmap="gray")
    plt.imsave(out_dir / "orient_ref_pa.png", pa_np, cmap="gray")
    print("Saved reference projections to", out_dir)


if __name__ == "__main__":
    main()
