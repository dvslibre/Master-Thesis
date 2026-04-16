"""Encoders for projection-based conditioning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionEncoder(nn.Module):
    """Simple 2D CNN encoder for AP/PA (and optional CT) projections."""

    def __init__(self, in_ch: int, z_dim: int, base_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, base_ch, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(8, base_ch)
        self.norm2 = nn.GroupNorm(8, base_ch * 2)
        self.norm3 = nn.GroupNorm(8, base_ch * 4)
        self.norm4 = nn.GroupNorm(8, base_ch * 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_ch * 4, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.conv1(x)))
        h = F.relu(self.norm2(self.conv2(h)))
        h = F.relu(self.norm3(self.conv3(h)))
        h = F.relu(self.norm4(self.conv4(h)))
        h = self.pool(h).flatten(1)
        return self.fc(h)
