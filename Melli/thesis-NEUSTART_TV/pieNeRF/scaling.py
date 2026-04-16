"""Scaling utilities for normalized inputs and physical units."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ScalingContext:
    proj_scale_joint_p99: float
    mu_scale_p99_9: float
    act_scale_p99_9: float
    voxel_size_mm: float = 1.5
    mu_unit: str = "per_cm"
    radius: float = 1.0
    world_scale_cm: float = 1.0

    @staticmethod
    def compute_world_scale_cm(
        volume_shape_dhw: Tuple[int, int, int],
        radius: float,
        voxel_size_mm: float = 1.5,
    ) -> float:
        if radius <= 0:
            raise ValueError("radius must be > 0 for world_scale_cm.")
        if voxel_size_mm <= 0:
            raise ValueError("voxel_size_mm must be > 0.")
        d, h, w = volume_shape_dhw
        voxel_size_cm = voxel_size_mm / 10.0
        lx_cm = float(w) * voxel_size_cm
        ly_cm = float(h) * voxel_size_cm
        lz_cm = float(d) * voxel_size_cm
        world_scale_cm = max(lx_cm, ly_cm, lz_cm) / (2.0 * float(radius))
        if world_scale_cm <= 0:
            raise ValueError("world_scale_cm must be > 0.")
        return world_scale_cm

    def __post_init__(self):
        if self.proj_scale_joint_p99 <= 0:
            raise ValueError("proj_scale_joint_p99 must be > 0.")
        if self.mu_scale_p99_9 <= 0:
            raise ValueError("mu_scale_p99_9 must be > 0.")
        if self.act_scale_p99_9 <= 0:
            raise ValueError("act_scale_p99_9 must be > 0.")
        if self.radius <= 0:
            raise ValueError("radius must be > 0.")
        if self.world_scale_cm <= 0:
            raise ValueError("world_scale_cm must be > 0.")

    def denorm_proj(self, x):
        return x * float(self.proj_scale_joint_p99)

    def denorm_mu(self, x):
        return x * float(self.mu_scale_p99_9)

    def denorm_act(self, x):
        return x * float(self.act_scale_p99_9)
