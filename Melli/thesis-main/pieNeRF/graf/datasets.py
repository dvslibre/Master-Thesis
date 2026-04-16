"""Dataset utilities for SPECT data: load AP/PA projections, CT/act volumes from manifest CSV."""

import glob
import numpy as np
from PIL import Image
from pathlib import Path
import csv
import torch

from torchvision.datasets.vision import VisionDataset



class SpectDataset(torch.utils.data.Dataset):
    """
    Dataset für SPECT-ähnliche Rekonstruktion:
    Lädt pro Fall AP, PA und CT, opt. ACT (alles als .npy), basierend auf einem Manifest (CSV).
    """
    def __init__(
        self,
        manifest_path,
        imsize=None,
        transform_img=None,
        transform_ct=None,
        act_scale: float = 1.0,
        use_organ_mask: bool = False,
        act_flip_lr: bool = False,
        act_flip_si: bool = False,
        act_debug_marker: bool = False,
    ):  
        super().__init__()
        self.manifest_path = Path(manifest_path)                                # manifest_path = Pfad zur csv mit Spalten phantom_id, ap_path, pa_path, ct_path
        self.imsize = imsize                                                    # momentan nicht genutzt
        self.transform_img = transform_img                                      # optionale Transformationsfunktionen für AP/PA und CT
        self.transform_ct = transform_ct                                        # "
        self.act_scale = float(act_scale)                                       # globaler Faktor für ACT/λ (keine Normierung)
        self.use_organ_mask = bool(use_organ_mask)
        self.act_flip_lr = bool(act_flip_lr)                                    # optionaler LR-Flip für ACT, um Koordinatensysteme abzugleichen
        self.act_flip_si = bool(act_flip_si)                                    # optionaler SI-Flip für ACT
        self.act_debug_marker = bool(act_debug_marker)                          # Marker + Logs für ACT-Debug
        if self.act_flip_lr and self.act_flip_si:
            raise ValueError("act_flip_lr and act_flip_si cannot both be True – choose one.")

        self.entries = []                                                       # Liste in der für jeden Fall ein kleines Dict mit Pfaden & ID steht
        with open(self.manifest_path, newline="") as f:                         # CSV öffnen
            reader = csv.DictReader(f)                                          # liest jede Zeile als dict
            for row in reader:
                act_path = row.get("act_path")
                if act_path is None:
                    # optional automatisch aus dem Ordner ableiten
                    candidate = Path(row["ap_path"]).with_name("act.npy")
                    act_path = candidate if candidate.exists() else None
                else:
                    act_path = Path(act_path)

                self.entries.append({                                           # jeweils als Path-Objekte speichern: phantom_id, ap_path, pa_path, ct_path
                    "patient_id": row["patient_id"],
                    "ap_path": Path(row["ap_path"]),
                    "pa_path": Path(row["pa_path"]),
                    "ct_path": Path(row["ct_path"]),
                    "act_path": act_path,
                    "mask_path": Path(row["ct_path"]).with_name("mask.npy"),
                    "spect_att_path": Path(row["ct_path"]).with_name("spect_att.npy"),
                })

    def __len__(self):
        return len(self.entries)                                                # Anzahl der Einträge = Anzahl Zeilen im Manifest = Anzahl Phantome/Patienten


    def _load_npy_image(self, path):                                            # lädt .npy Array, z.B. [H,W] mit Counts
        arr = np.load(path).astype(np.float32)                                  # stellt sicher, dass float32
        
        tensor = torch.from_numpy(arr).unsqueeze(0)                             # macht einen Tensor draus [H,W] -> [1,H,W]

        return tensor


    def _load_npy_ct(self, path):
        if path is None:
            return torch.empty(0)
        path = Path(path)
        if not path.exists():
            return torch.empty(0)

        vol = np.load(path).astype(np.float32)                                  # Original gespeichert als (LR, AP, SI)
        vol = np.transpose(vol, (1, 0, 2))                                      # neu: (AP, LR, SI) -> depth=AP, H=LR, W=SI

        vol *= 10.0                                                             # optionaler scale-factor

        return torch.from_numpy(vol)


    def _load_npy_act(self, path):
        if path is None:
            return torch.empty(0)

        vol = np.load(path).astype(np.float32)
        vol = np.transpose(vol, (1, 0, 2))                                      # (AP, LR, SI); axis=1 ist LR, axis=2 ist SI
        if self.act_flip_lr:
            vol = np.flip(vol, axis=1)                                          # LR-Flip entlang der LR-Achse
        if self.act_flip_si:
            vol = np.flip(vol, axis=2)                                          # SI-Flip entlang der SI-Achse

        # Kein Min/Max-Scaling mehr – optional nur globaler Faktor, damit λ im festen Maßstab bleibt.
        vol = vol * self.act_scale
        if self.act_debug_marker and (self.act_flip_lr or self.act_flip_si):
            vol = vol * 123.456                                                 # Marker zum Nachverfolgen der Pipeline
            print(f"[ACT DEBUG][dataset_after_flip] min/max/shape {vol.min():.3e}/{vol.max():.3e} {vol.shape}", flush=True)

        return torch.from_numpy(vol)

    def _load_npy_att(self, path):
        """Lädt Attenuations-Volumes (spect_att) mit identischer Permutation/Scale wie CT."""
        if path is None:
            return torch.empty(0)
        path = Path(path)
        if not path.exists():
            return torch.empty(0)
        vol = np.load(path).astype(np.float32)
        vol = np.transpose(vol, (1, 0, 2))                                      # (AP, LR, SI)
        vol *= 10.0
        return torch.from_numpy(vol)

    def _load_npy_mask(self, path):
        if path is None or not self.use_organ_mask:
            return torch.empty(0)
        path = Path(path)
        if not path.exists():
            return torch.empty(0)
        vol = np.load(path).astype(np.float32)
        vol = np.transpose(vol, (1, 0, 2))                                      # (AP, LR, SI)
        vol = np.clip(vol, 0.0, 1.0)
        return torch.from_numpy(vol)


    def __getitem__(self, idx):
        e = self.entries[idx]                                                   # holt das idx-te Manifest-Dict
        if "_cache" in e:
            cached = e["_cache"]
        else:
            ap = self._load_npy_image(e["ap_path"])                             # lädt AP als [1,H,W]-Tensor (roh, keine Min/Max-Norm)
            pa = self._load_npy_image(e["pa_path"])                             # lädt PA als [1,H,W]-Tensor (roh, keine Min/Max-Norm)
            ct = self._load_npy_ct(e["ct_path"])                                # lädt CT Volumen (gescaled)  
            act = self._load_npy_act(e["act_path"])                             # lädt ACT Volumen (ohne Normierung, nur globaler Faktor)
            mask = self._load_npy_mask(e.get("mask_path"))
            spect_att = self._load_npy_att(e.get("spect_att_path"))
            if ct.numel() == 0 and spect_att.numel() > 0:
                ct = spect_att                                                  # nutze spect_att als CT-Basis, falls ct.npy fehlt

            # Maske auf AP/PA-Auflösung bringen (nearest), damit Mask-Loss denselben Pixelraum nutzt.
            if mask.numel() > 0:
                import torch.nn.functional as F
                d, h, w = mask.shape
                target_h = ap.shape[-2]
                target_w = ap.shape[-1]
                if (h, w) != (target_h, target_w):
                    mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
                    mask = F.interpolate(mask, size=(d, target_h, target_w), mode="nearest")
                    mask = mask.squeeze(0).squeeze(0)
                print(f"[MASK DEBUG][dataset] shape={tuple(mask.shape)} target=({target_h},{target_w})", flush=True)

            ap = self._normalize_projection(ap)
            pa = self._normalize_projection(pa)

            if self.transform_img is not None:                                  # optionale zusätzliche Schritte (z.B. Resize, Cropping, ...)
                ap = self.transform_img(ap)
                pa = self.transform_img(pa)
            if self.transform_ct is not None:
                ct = self.transform_ct(ct)

            cached = {                                                         # Rückgabe ist ein Dictionary   
                "ap": ap,
                "pa": pa,
                "ct": ct,
                "act": act,
                "spect_att": spect_att,
                "mask": mask,
                "meta": {
                    "patient_id": e["patient_id"],
                    "act_scale": self.act_scale,
                },
            }
            if self.act_debug_marker and act is not None and act.numel() > 0:
                act_min = act.min().item()
                act_max = act.max().item()
                print(f"[ACT DEBUG][dataset_getitem] min/max/shape {act_min:.3e}/{act_max:.3e} {tuple(act.shape)}", flush=True)
            e["_cache"] = cached

        # Debug-Ausgabe nur einmal global, um Skalen zu prüfen (kein Einfluss auf Verhalten).
        if not hasattr(self, "_debug_printed") or not self._debug_printed:
            ct_min = cached["ct"].min().item() if cached["ct"].numel() > 0 else float("nan")
            ct_max = cached["ct"].max().item() if cached["ct"].numel() > 0 else float("nan")
            print(
                f"[DEBUG][datasets] AP min/max: {cached['ap'].min().item():.3e}/{cached['ap'].max().item():.3e} | "
                f"PA min/max: {cached['pa'].min().item():.3e}/{cached['pa'].max().item():.3e} | "
                f"CT min/max: {ct_min:.3e}/{ct_max:.3e} | "
                f"ACT min/max: {(cached['act'].min().item() if cached['act'].numel()>0 else float('nan')):.3e}/"
                f"{(cached['act'].max().item() if cached['act'].numel()>0 else float('nan')):.3e} | "
                f"MASK min/max: {(cached['mask'].min().item() if cached['mask'].numel()>0 else float('nan')):.3e}/"
                f"{(cached['mask'].max().item() if cached['mask'].numel()>0 else float('nan')):.3e} | "
                f"SPECT_att min/max: {(cached['spect_att'].min().item() if cached['spect_att'].numel()>0 else float('nan')):.3e}/"
                f"{(cached['spect_att'].max().item() if cached['spect_att'].numel()>0 else float('nan')):.3e}",
                flush=True,
            )
            self._debug_printed = True

        return cached

    def _normalize_projection(self, tensor: torch.Tensor) -> torch.Tensor:
        """Einfache per-Projektion-Normierung auf [0,1], wie im ursprünglichen Code."""
        maxv = tensor.max()
        if maxv > 0:
            tensor = tensor / maxv
        return tensor
