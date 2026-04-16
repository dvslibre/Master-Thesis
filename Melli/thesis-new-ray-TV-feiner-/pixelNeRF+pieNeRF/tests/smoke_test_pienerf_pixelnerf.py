import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.spect_pixel_pienerf import SpectPixelPieNeRF


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    volume_shape = (64, 64, 64)  # (D, H, W) = (SI, AP, LR)
    voxel_size = 1.0
    num_samples = volume_shape[1]  # AP dimension
    latent_dim = 512
    mlp_width = 256
    mlp_depth = 8
    multires_xyz = 6

    # Dummy inputs
    ct = torch.rand(1, 1, *volume_shape)
    ap = torch.rand(1, 1, volume_shape[1], volume_shape[2])
    pa = torch.rand(1, 1, volume_shape[1], volume_shape[2])

    model = SpectPixelPieNeRF(
        volume_shape=volume_shape,
        voxel_size=voxel_size,
        num_samples=num_samples,
        latent_dim=latent_dim,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
        multires_xyz=multires_xyz,
    )

    print(f"Params: {count_parameters(model):,}")
    print(f"volume_shape={volume_shape}, num_samples={num_samples}, embed_dim={multires_xyz}")

    try:
        start = time.time()
        out = model(ct, ap, pa)
        end = time.time()

        print("ap_pred shape:", tuple(out["ap_pred"].shape))
        print("pa_pred shape:", tuple(out["pa_pred"].shape))
        print("sigma_volume shape:", tuple(out["sigma_volume"].shape))

        print("ap_pred min/max:", out["ap_pred"].min().item(), out["ap_pred"].max().item())
        print("pa_pred min/max:", out["pa_pred"].min().item(), out["pa_pred"].max().item())

        print(f"Forward OK in {(end - start)*1000:.2f} ms")

        loss = out["ap_pred"].mean() + out["pa_pred"].mean()
        loss.backward()

        # Gradient sanity check
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                raise RuntimeError(f"NaN/Inf gradient in parameter {name}")

        print("Backward OK")
    except Exception as e:
        print("ERROR in smoke-test:", e)


if __name__ == "__main__":
    main()
