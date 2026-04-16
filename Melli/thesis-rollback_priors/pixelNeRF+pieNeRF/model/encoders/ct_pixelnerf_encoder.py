"""CT-aware PixelNeRF encoder."""

import sys
from pathlib import Path
from typing import Optional
import importlib.util

import torch
from torch import nn
import torch.nn.functional as F

# Ensure pixel-nerf/src is importable to reuse the existing SpatialEncoder backbone.
_PIXEL_NERF_SRC = Path(__file__).resolve().parents[2] / "pixel-nerf" / "src"
if _PIXEL_NERF_SRC.exists():
    src_str = str(_PIXEL_NERF_SRC)
    if src_str not in sys.path:
        # Prepend to prioritize pixel-nerf modules over similarly named local packages.
        sys.path.insert(0, src_str)

try:
    from model.encoder import SpatialEncoder  # type: ignore
except ImportError:
    # Fallback: load encoder directly from pixel-nerf source path to avoid package name clash.
    encoder_path = _PIXEL_NERF_SRC / "model" / "encoder.py"
    if not encoder_path.exists():
        raise ImportError(
            "Could not import pixel-nerf SpatialEncoder. "
            "Ensure pixel-nerf/src is available."
        )
    spec = importlib.util.spec_from_file_location("pixelnerf_encoder", encoder_path)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load pixel-nerf encoder module from path.")
    pixelnerf_encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pixelnerf_encoder)
    SpatialEncoder = pixelnerf_encoder.SpatialEncoder  # type: ignore


class CTPixelNeRFEncoder(nn.Module):
    """
    PixelNeRF-style CNN encoder that ingests CT + AP + PA and returns a global latent.

    Expects:
        ct: Tensor (B, 1, D, H, W)   # CT volume ordered (SI, AP, LR)
        ap: Tensor (B, 1, H, W)      # AP projection
        pa: Tensor (B, 1, H, W)      # PA projection

    Returns:
        z_feat: Tensor (B, C_latent) after global average pooling over the backbone output.
    """

    def __init__(self, backbone: str = "resnet18", num_layers: int = 4, latent_channels: Optional[int] = None):
        """
        Args:
            backbone: torchvision resnet backbone name to use via PixelNeRF SpatialEncoder.
            num_layers: number of resnet layers to aggregate in SpatialEncoder.
            latent_channels: optional override for expected latent dim; if None, derived from backbone.
        """
        super().__init__()
        self.backbone = SpatialEncoder(
            backbone=backbone,
            pretrained=False,
            num_layers=num_layers,
            index_interp="bilinear",
            index_padding="border",
            upsample_interp="bilinear",
            feature_scale=1.0,
            use_first_pool=True,
            norm_type="batch",
        )

        # Adjust first conv to accept 5-channel input (3 CT slices + AP + PA).
        outplanes = self.backbone.model.conv1.out_channels
        if self.backbone.model.conv1.in_channels != 5:
            self.backbone.model.conv1 = nn.Conv2d(
                in_channels=5,
                out_channels=outplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # Reported latent size from SpatialEncoder; actual C is determined at runtime after concat.
        self.latent_channels = latent_channels

    def forward(self, ct: torch.Tensor, ap: torch.Tensor, pa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ct: Tensor (B, 1, D, H, W).
            ap: Tensor (B, 1, H, W).
            pa: Tensor (B, 1, H, W).

        Returns:
            z_feat: Tensor (B, C_latent) global latent after pooling.
        """
        _, _, d, h, w = ct.shape
        # axial slice: center along SI axis (keep HxW footprint).
        ct_ax = ct[:, :, d // 2, :, :]  # (B, 1, H, W)
        # coronal slice: center along AP axis → reshape D x W into H x W.
        ct_cor = ct[:, :, :, h // 2, :]  # (B, 1, D, W)
        ct_cor = F.interpolate(ct_cor, size=(h, w), mode="bilinear", align_corners=False)
        # sagittal slice: center along LR axis → reshape D x H into H x W.
        ct_sag = ct[:, :, :, :, w // 2]  # (B, 1, D, H)
        ct_sag = ct_sag.permute(0, 1, 3, 2)  # (B, 1, H, D)
        ct_sag = F.interpolate(ct_sag, size=(h, w), mode="bilinear", align_corners=False)

        ct_slices = torch.cat([ct_ax, ct_cor, ct_sag], dim=1)  # (B, 3, H, W)
        input_2d = torch.cat([ct_slices, ap, pa], dim=1)  # (B, 5, H, W)

        feat_map = self.backbone(input_2d)  # (B, C, H', W')
        z_feat = feat_map.mean(dim=[2, 3])  # (B, C)
        return z_feat
