import os
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class SpectDataset(torch.utils.data.Dataset):
    """
    Volumes are returned in (SI, AP, LR) order.
    Dataset for CT + AP/PA projections + 3D activity volume.
    Structure is driven by a manifest CSV with columns:
    patient_id, ap_path, pa_path, ct_path, act_path (act_path optional).
    AP/PA are stored as npy arrays of shape (H, W).
    CT and ACT are loaded and explicitly reordered to (SI, AP, LR) via permute if needed.
    """

    def __init__(
        self,
        datadir: str,
        stage: str = "train",
        manifest: Optional[str] = None,
        scale_ct: float = 10.0,
        target_hw: Optional[tuple] = (128, 320),
        target_depth: Optional[int] = 128,
        rotate_projections: bool = False,
    ):
        super().__init__()
        self.base_path = Path(datadir).expanduser().resolve()
        if manifest is None:
            manifest_candidate = self.base_path / f"manifest_{stage}.csv"
            if manifest_candidate.exists():
                manifest = manifest_candidate
            else:
                manifest = self.base_path / "manifest.csv"
        self.manifest_path = Path(manifest).expanduser().resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.scale_ct = scale_ct
        self.target_hw = target_hw
        self.target_depth = target_depth
        self.rotate_projections = rotate_projections
        self.entries = []
        with open(self.manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ap_path = self._resolve_path(Path(row["ap_path"]))
                pa_path = self._resolve_path(Path(row["pa_path"]))
                ct_path = self._resolve_path(Path(row["ct_path"]))

                act_path = row.get("act_path")
                if act_path:
                    act_path = self._resolve_path(Path(act_path))
                else:
                    candidate = ap_path.with_name("act.npy")
                    act_path = candidate if candidate.exists() else None
                self.entries.append(
                    {
                        "patient_id": row.get("patient_id", ""),
                        "ap_path": ap_path,
                        "pa_path": pa_path,
                        "ct_path": ct_path,
                        "act_path": act_path,
                    }
                )

        # Dummy values to keep parity with other datasets if accessed
        self.z_near = 0.0
        self.z_far = 1.0
        self.lindisp = False

        print(f"Loaded SpectDataset from {self.manifest_path} with {len(self.entries)} entries")

    def __len__(self):
        return len(self.entries)

    def _resolve_path(self, path: Path) -> Path:
        """Resolve relative paths against dataset base path."""
        if path.is_absolute():
            return path
        candidate = self.base_path / path
        if candidate.exists():
            return candidate
        # Sometimes manifest paths are relative to parent (e.g., start with "data/").
        alt = self.base_path.parent / path
        if alt.exists():
            return alt
        return candidate

    def _load_proj(self, path: Path):
        arr = np.load(path).astype(np.float32)
        if self.rotate_projections:
            # Match pieNeRF visualization: flip vertically then rotate 90 CW
            arr = np.rot90(np.flipud(arr), k=-1)
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        maxv = tensor.max()
        if maxv > 0:
            tensor = tensor / maxv
        return tensor

    def _reorder_to_siaplr(self, vol: np.ndarray, name: str) -> np.ndarray:
        """
        Ensure volume is in (SI, AP, LR) order.
        If first dim >> last dim (e.g., 651,256,256), assume (LR, AP, SI) and permute.
        If already (SI, AP, LR) (e.g., 256,256,651), leave as is.
        """
        if vol.shape[0] > vol.shape[2]:
            vol = np.transpose(vol, (2, 1, 0))  # (LR, AP, SI) -> (SI, AP, LR)
        return vol

    def _load_ct(self, path: Path):
        vol = np.load(path).astype(np.float32)
        vol = self._reorder_to_siaplr(vol, "ct")
        # Now vol is (SI, AP, LR)
        vol *= self.scale_ct
        return torch.from_numpy(vol)

    def _load_act(self, path: Optional[Path]):
        if path is None or (not path.exists()):
            return torch.empty(0)
        vol = np.load(path).astype(np.float32)
        vol = self._reorder_to_siaplr(vol, "act")
        # Now vol is (SI, AP, LR)
        maxv = vol.max()
        if maxv > 0:
            vol = vol / maxv
        return torch.from_numpy(vol)

    def __getitem__(self, idx):
        e = self.entries[idx]
        ap = self._load_proj(e["ap_path"])
        pa = self._load_proj(e["pa_path"])
        ct = self._load_ct(e["ct_path"])
        act = self._load_act(e["act_path"])

        # Optional downsampling to keep memory manageable
        if self.target_hw is not None:
            import torch.nn.functional as F

            ap = F.interpolate(ap.unsqueeze(0), size=self.target_hw, mode="bilinear", align_corners=False).squeeze(0)
            pa = F.interpolate(pa.unsqueeze(0), size=self.target_hw, mode="bilinear", align_corners=False).squeeze(0)
            if ct.numel() > 0:
                new_d = ct.shape[0] if self.target_depth is None else self.target_depth
                ct = F.interpolate(
                    ct.unsqueeze(0).unsqueeze(0),
                    size=(new_d, *self.target_hw),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            if act.numel() > 0:
                new_d = act.shape[0] if self.target_depth is None else self.target_depth
                act = F.interpolate(
                    act.unsqueeze(0).unsqueeze(0),
                    size=(new_d, *self.target_hw),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

        return {
            "ap": ap,
            "pa": pa,
            "ct": ct,
            "act": act,
            "meta": {"patient_id": e["patient_id"]},
        }
