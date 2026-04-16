"""Top-level SpectPixelPieNeRF model."""

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

from model.encoders.ct_pixelnerf_encoder import CTPixelNeRFEncoder
from model.nerf.pienerf_mlp_cond import PieNeRFConditionalMLP
from utils.geometry.ray_sampler_spect import sample_spect_rays
from forward.spect_operator_wrapper import forward_spect

# Import positional encoder from pieNeRF helpers.
_PIENERF_ROOT = Path(__file__).resolve().parents[1] / "pieNeRF"
if _PIENERF_ROOT.exists():
    sys.path.append(str(_PIENERF_ROOT))
try:
    from nerf.run_nerf_helpers_mod import get_embedder  # type: ignore
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "Could not import get_embedder from pieNeRF. Ensure pieNeRF is on PYTHONPATH."
    ) from exc


class SpectPixelPieNeRF(nn.Module):
    """
    Pipeline:
        ct, ap, pa
          → CTPixelNeRFEncoder → z_feat (B, latent_dim)
          → sample_spect_rays (AP/PA) → xyz
          → positional encoding xyz
          → PieNeRFConditionalMLP → sigma
          → sigma_volume (D, H, W)
          → forward_spect → AP/PA images (optional attenuation via CT-scaled mu_volume)
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        voxel_size: float | Tuple[float, float, float],
        num_samples: int,
        latent_dim: int = 512,
        mlp_width: int = 256,
        mlp_depth: int = 8,
        skips: Optional[list[int]] = None,
        multires_xyz: int = 6,
        step_len: float = 1.0,
        mu_scale: float = 0.01,
    ):
        super().__init__()
        self.volume_shape = volume_shape
        self.voxel_size = voxel_size
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.step_len = step_len
        self.mu_scale = mu_scale

        self.encoder = CTPixelNeRFEncoder(backbone="resnet18", num_layers=4, latent_channels=latent_dim)
        self.embed_xyz, embed_xyz_out_dim = get_embedder(multires_xyz, i=0)
        self.embed_xyz_out_dim = embed_xyz_out_dim
        self.mlp = PieNeRFConditionalMLP(
            input_ch_xyz=embed_xyz_out_dim,
            latent_dim=latent_dim,
            D=mlp_depth,
            W=mlp_width,
            skips=skips or [4],
        )

    def forward(
        self,
        ct: torch.Tensor,   # (B,1,D,H,W)
        ap: torch.Tensor,   # (B,1,H,W)
        pa: torch.Tensor,   # (B,1,H,W)
        mu_volume: Optional[torch.Tensor] = None,  # (B,D,H,W) or None
        use_attenuation: bool = False,
        step_len: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            {
              "ap_pred": (B,1,H,W),
              "pa_pred": (B,1,H,W),
              "sigma_volume": (B,1,D,H,W)
            }
        """
        B, _, D, H, W = ct.shape
        if self.num_samples != H:
            raise AssertionError("Assumption num_samples == H (AP dimension) not satisfied.")

        device = ct.device
        dtype = ct.dtype

        # Encoder: z_feat (B, latent_dim)
        z_feat = self.encoder(ct, ap, pa)

        # Ray sampling AP/PA
        xyz_ap, _ = sample_spect_rays(
            self.volume_shape,
            self.voxel_size,
            "AP",
            self.num_samples,
            batch_size=B,
            device=device,
            dtype=dtype,
        )
        xyz_pa, _ = sample_spect_rays(
            self.volume_shape,
            self.voxel_size,
            "PA",
            self.num_samples,
            batch_size=B,
            device=device,
            dtype=dtype,
        )

        # Positional encoding
        N = xyz_ap.shape[1]
        xyz_ap_pe_flat = self.embed_xyz(xyz_ap.reshape(-1, 3))
        xyz_pa_pe_flat = self.embed_xyz(xyz_pa.reshape(-1, 3))
        xyz_ap_pe = xyz_ap_pe_flat.view(B, N, self.embed_xyz_out_dim)
        xyz_pa_pe = xyz_pa_pe_flat.view(B, N, self.embed_xyz_out_dim)

        # MLP to sigma
        sigma_ap = self.mlp(xyz_ap_pe, z_feat)  # (B,N,1)
        sigma_pa = self.mlp(xyz_pa_pe, z_feat)  # (B,N,1)

        # Average AP/PA predictions.
        sigma = 0.5 * (sigma_ap + sigma_pa)  # (B,N,1)
        # sample_spect_rays flattens in order (SI, LR, AP), so reshape accordingly then permute to (SI, AP, LR).
        sigma_lr_ap = sigma.view(B, D, W, H)               # (B, SI, LR, AP)
        sigma_volume = sigma_lr_ap.permute(0, 1, 3, 2)     # (B, SI, AP, LR)
        sigma_volume = sigma_volume.unsqueeze(1)           # (B,1,SI,AP,LR)

        # Simple CT-derived attenuation map if requested and not provided explicitly.
        mu_for_forward = None
        if use_attenuation:
            if mu_volume is not None:
                mu_for_forward = mu_volume
            else:
                # Lightweight heuristic: positive CT values scaled to approximate mu.
                mu_for_forward = torch.relu(ct) * self.mu_scale  # (B,1,D,H,W)

        # Forward operator (placeholder) consumes full volume.
        ap_pred, pa_pred = forward_spect(
            sigma_volume,
            mu_volume=mu_for_forward,
            use_attenuation=use_attenuation,
            step_len=step_len if step_len is not None else self.step_len,
        )
        ap_pred = ap_pred.unsqueeze(1)  # (B,1,SI,LR)
        pa_pred = pa_pred.unsqueeze(1)  # (B,1,SI,LR)

        return {
            "ap_pred": ap_pred,
            "pa_pred": pa_pred,
            "sigma_volume": sigma_volume,
        }
