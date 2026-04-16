from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
from Data_Processing.preprocessing import (
    convert_mu_units,
    gamma_camera_core,
    normalize_projections,
)

try:
    import scipy.io as sio  # type: ignore[import]
except ImportError:  # pragma: no cover - best effort
    sio = None


@dataclass
class ProjectorConfig:
    kernel_mat: Path
    kernel_var: str = "kernel_mat"
    psf_sigma: float = 2.0
    z0_slices: int = 29
    step_len: float = 0.15
    mu_unit_in: str = "per_mm"
    mu_unit_out: str = "per_cm"
    sensitivity_cps_per_mbq: float = 65.0
    acq_time_s: float = 300.0
    comp_scatter: bool = True
    atn_on: bool = True
    coll_on: bool = True


@lru_cache(maxsize=4)
def _load_kernel(path: Path, var: str) -> np.ndarray:
    if sio is None:
        raise ModuleNotFoundError("scipy.io is required to load the LEAP kernel")
    if not path.exists():
        raise FileNotFoundError(f"LEAP kernel missing: {path}")
    mat = sio.loadmat(path)
    if var not in mat:
        raise KeyError(f"Kernel variable '{var}' not found in {path}")
    kernel = mat[var]
    if not isinstance(kernel, np.ndarray):
        raise TypeError(f"Kernel '{var}' is not an ndarray ({type(kernel)})")
    return kernel.astype(np.float32, copy=False)


def project_activity(
    act_kbq_per_ml: np.ndarray,
    ct_atn: np.ndarray,
    spacing_cm: Tuple[float, float, float],
    config: ProjectorConfig,
) -> dict[str, object]:
    """Simulate AP/PA projections (MBq/Counts/Normalized) using the gamma-camera model."""
    if act_kbq_per_ml.shape != ct_atn.shape:
        raise ValueError("Activity and CT attenuation volumes must share shape.")

    voxel_volume_ml = float(spacing_cm[0] * spacing_cm[1] * spacing_cm[2])
    act_mbq = np.clip(act_kbq_per_ml, 0.0, None) * 1e-3 * voxel_volume_ml
    ct_target = convert_mu_units(ct_atn.astype(np.float32), config.mu_unit_in, config.mu_unit_out)

    kernel = _load_kernel(config.kernel_mat, config.kernel_var)
    ap_mbq, pa_mbq = gamma_camera_core(
        act_mbq.astype(np.float32),
        ct_target,
        kernel,
        config.psf_sigma,
        config.z0_slices,
        step_len=config.step_len,
        comp_scatter=config.comp_scatter,
        atn_on=config.atn_on,
        coll_on=config.coll_on,
    )

    counts_scale = float(config.sensitivity_cps_per_mbq) * float(config.acq_time_s)
    ap_counts = np.clip(ap_mbq, 0.0, None) * counts_scale
    pa_counts = np.clip(pa_mbq, 0.0, None) * counts_scale

    ap_norm, pa_norm, scale = normalize_projections(ap_counts, pa_counts)

    return {
        "ap_counts": ap_counts.astype(np.float32),
        "pa_counts": pa_counts.astype(np.float32),
        "ap_norm": ap_norm.astype(np.float32),
        "pa_norm": pa_norm.astype(np.float32),
        "norm_scale": float(scale),
        "ap_mbq": ap_mbq.astype(np.float32),
        "pa_mbq": pa_mbq.astype(np.float32),
    }
