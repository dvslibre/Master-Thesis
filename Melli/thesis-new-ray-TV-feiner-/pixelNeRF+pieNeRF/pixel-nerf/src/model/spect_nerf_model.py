import torch
import torch.nn as nn

from .encoder import SpatialEncoder
from .code import PositionalEncoding
from .model_util import make_mlp


class SpectNeRFNet(nn.Module):
    """
    NeRF-like network for SPECT density field.
    - Encodes AP/PA (and optional CT stacked as extra channels) via SpatialEncoder.
    - Outputs rgb (used as intensity) + sigma; rgb channels are identical.
    """

    def __init__(self, conf):
        super().__init__()
        self.use_code = conf.get_bool("use_code", True)
        self.encoder = SpatialEncoder.from_conf(conf["encoder"])
        d_latent = self.encoder.latent_size

        d_in = 3  # xyz
        if self.use_code:
            self.code = PositionalEncoding.from_conf(conf["code"], d_in=d_in)
            d_in = self.code.d_out

        d_out = 4  # rgb (3) + sigma (1)
        self.mlp = make_mlp(conf["mlp"], d_in, d_latent, d_out=d_out)

        # buffers for global latent
        self.register_buffer("global_latent", torch.empty(1, d_latent), persistent=False)

    def encode(self, images):
        """
        :param images: (B, N, 3, H, W) or (B, 3, H, W) if single view
        Stores global latent (average pooled over spatial + views)
        """
        if images.dim() == 5:
            B, N = images.shape[:2]
            images = images.reshape(B * N, *images.shape[2:])
        else:
            B = images.shape[0]
        feats = self.encoder(images)  # (B*N, C, Hf, Wf)
        pooled = feats.mean(dim=[2, 3])  # (B*N, C)
        pooled = pooled.reshape(B, -1, pooled.shape[-1]).mean(dim=1)  # (B, C)
        self.global_latent = pooled
        return pooled

    def forward(self, xyz, coarse=True, viewdirs=None, far=False):
        """
        :param xyz: (B, R, 3) points
        """
        B, R, _ = xyz.shape
        x = xyz.reshape(-1, 3)
        if self.use_code:
            x = self.code(x)
        # repeat global latent per point
        gl = self.global_latent
        if gl.shape[0] == 1 and B > 1:
            gl = gl.expand(B, -1)
        gl = gl[:, None, :].expand(-1, R, -1).reshape(-1, gl.shape[-1])
        mlp_in = (gl, x)
        mlp_out = self.mlp(torch.cat(mlp_in, dim=-1))
        mlp_out = mlp_out.reshape(B, R, -1)
        rgb = torch.sigmoid(mlp_out[..., :3])
        sigma = torch.relu(mlp_out[..., 3:])
        return torch.cat([rgb, sigma], dim=-1)
