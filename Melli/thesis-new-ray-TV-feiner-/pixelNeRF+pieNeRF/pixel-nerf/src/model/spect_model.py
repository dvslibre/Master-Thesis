import torch
import torch.nn as nn


class SpectReconNet(nn.Module):
    """
    Simple 3D convolutional network for volumetric activity reconstruction.
    Inputs:
        ap: (B, 1, H, W)
        pa: (B, 1, H, W)
        ct: (B, D, H, W)
    Output:
        activity volume: (B, D, H, W)
    """

    def __init__(self, base_channels: int = 32):
        super().__init__()
        # Keep spatial resolution; lightweight stack of 3D convs.
        self.net = nn.Sequential(
            nn.Conv3d(3, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, 1, kernel_size=1),
        )

    def forward(self, ap, pa, ct):
        """
        :param ap: (B, 1, H, W)
        :param pa: (B, 1, H, W)
        :param ct: (B, D, H, W)
        """
        # Lift 2D projections to volume by repeating along depth.
        ct_vol = ct.unsqueeze(1)  # (B, 1, D, H, W)
        depth = ct_vol.shape[2]
        ap_vol = ap.unsqueeze(2).repeat(1, 1, depth, 1, 1)  # (B, 1, D, H, W)
        pa_vol = pa.unsqueeze(2).repeat(1, 1, depth, 1, 1)  # (B, 1, D, H, W)
        x = torch.cat([ct_vol, ap_vol, pa_vol], dim=1)  # (B, 3, D, H, W)
        out = self.net(x)  # (B, 1, D, H, W)
        return torch.sigmoid(out).squeeze(1)
