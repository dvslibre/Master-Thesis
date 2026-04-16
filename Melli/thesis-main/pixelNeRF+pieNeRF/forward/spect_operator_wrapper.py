"""Wrapper that aligns hybrid forward projection with RayNet orientation.

Volume convention: (B, 1, SI, AP, LR) or (B, SI, AP, LR). Only attenuation is
implemented; scatter and collimator are TODO/not implemented on purpose.
"""

from typing import Optional

import torch


def forward_spect(
    sigma_volume: torch.Tensor,
    mu_volume: Optional[torch.Tensor] = None,
    use_attenuation: bool = False,
    step_len: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Replicates pieNeRF's AP/PA forward chain:
    - Volumes are interpreted as (SI, AP, LR) and reordered to (AP, SI, LR) like pieNeRF.
    - CT-based attenuation is flipped along LR before integration (build_ct_context).
    - AP integrates along +z→-z (reverse AP axis), PA along -z→+z.
    - Attenuation uses the same cumulative/shift logic as raw2outputs_emission.
    """

    def _strip_channel(vol: torch.Tensor) -> torch.Tensor:
        if vol.dim() == 5 and vol.shape[1] == 1:
            return vol[:, 0]
        if vol.dim() == 4:
            return vol
        raise ValueError(f"Expected volume with shape (B,1,SI,AP,LR) or (B,SI,AP,LR), got {tuple(vol.shape)}")

    act = _strip_channel(sigma_volume)  # (B, SI, AP, LR)
    # pieNeRF internal order: (AP=z, SI=y, LR=x)
    act_aslr = act.permute(0, 2, 1, 3).contiguous()

    mu_aslr = None
    atten_enabled = bool(use_attenuation)
    if mu_volume is not None and atten_enabled:
        mu = _strip_channel(mu_volume)
        mu_aslr = mu.permute(0, 2, 1, 3).contiguous()
        # build_ct_context flips LR to align CT with AP/PA rays
        mu_aslr = torch.flip(mu_aslr, dims=[3])
        mu_aslr = torch.clamp(mu_aslr, min=0.0)
    else:
        atten_enabled = False  # mirror pieNeRF: disable attenuation if no CT is provided

    step = torch.as_tensor(step_len, device=act_aslr.device, dtype=act_aslr.dtype)

    def _project(act_vol: torch.Tensor, mu_vol: Optional[torch.Tensor], reverse_ap_axis: bool) -> torch.Tensor:
        """Integrate along AP axis with optional attenuation; reverse controls +z→-z vs -z→+z."""
        a_dir = torch.flip(act_vol, dims=[1]) if reverse_ap_axis else act_vol
        m_dir = torch.flip(mu_vol, dims=[1]) if (mu_vol is not None and reverse_ap_axis) else mu_vol

        weights = a_dir * step
        if atten_enabled and m_dir is not None:
            mu_scaled = m_dir * step
            # raw2outputs_emission: cumulative μ·Δs, shifted by one step, clamped
            atten = torch.cumsum(mu_scaled, dim=1)
            atten = torch.cat([torch.zeros_like(atten[:, :1]), atten[:, :-1]], dim=1)
            atten = torch.clamp(atten, min=0.0, max=60.0)
            trans = torch.exp(-atten)
            weights = a_dir * trans * step

        return weights.sum(dim=1)  # (B, SI, LR)

    # AP = camera at +z, integrates toward -z ⇒ reverse AP axis
    proj_ap = _project(act_aslr, mu_aslr, reverse_ap_axis=True)
    # PA = camera at -z, integrates toward +z ⇒ natural order
    proj_pa = _project(act_aslr, mu_aslr, reverse_ap_axis=False)

    # Align LR orientation with dataset projections (mirror LR once).
    proj_ap = torch.flip(proj_ap, dims=[2])
    proj_pa = torch.flip(proj_pa, dims=[2])

    return proj_ap, proj_pa
