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
    ):  
        super().__init__()
        self.manifest_path = Path(manifest_path)                                # manifest_path = Pfad zur csv mit Spalten phantom_id, ap_path, pa_path, ct_path
        self.imsize = imsize                                                    # momentan nicht genutzt
        self.transform_img = transform_img                                      # optionale Transformationsfunktionen für AP/PA und CT
        self.transform_ct = transform_ct                                        # "
        self.act_scale = float(act_scale)                                       # globaler Faktor für ACT/λ (keine Normierung)
        self._logged_debug = False                                               # sorgt dafür, dass Debug-Ausgabe nur einmal erfolgt

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
        
        vol = np.load(path).astype(np.float32)                                  # Original gespeichert als (LR, AP/Depth, SI)
        vol = np.transpose(vol, (1, 0, 2))                                      # Layout: (AP, LR, SI) = (D,H,W) für Renderer

        vol *= 10.0                                                             # optionaler scale-factor

        return torch.from_numpy(vol)


    def _load_npy_act(self, path):
        if path is None:
            return torch.empty(0)

        vol = np.load(path).astype(np.float32)
        vol = np.transpose(vol, (1, 0, 2))                                      # Layout-Anpassung wie CT: (AP, LR, SI) = (D,H,W)

        # Kein Min/Max-Scaling mehr – optional nur globaler Faktor, damit λ im festen Maßstab bleibt.
        vol = vol * self.act_scale

        return torch.from_numpy(vol)


    def __getitem__(self, idx):
        e = self.entries[idx]                                                   # holt das idx-te Manifest-Dict
        ap = self._load_npy_image(e["ap_path"])                                 # lädt AP als [1,H,W]-Tensor (roh, keine Min/Max-Norm)
        pa = self._load_npy_image(e["pa_path"])                                 # lädt PA als [1,H,W]-Tensor (roh, keine Min/Max-Norm)
        ct = self._load_npy_ct(e["ct_path"])                                    # lädt CT Volumen (gescaled)  
        act = self._load_npy_act(e["act_path"])                                 # lädt ACT Volumen (ohne Normierung, nur globaler Faktor)

        ap = self._normalize_projection(ap)
        pa = self._normalize_projection(pa)

        if self.transform_img is not None:                                      # optionale zusätzliche Schritte (z.B. Resize, Cropping, ...)
            ap = self.transform_img(ap)
            pa = self.transform_img(pa)
        if self.transform_ct is not None:
            ct = self.transform_ct(ct)

        # Debug-Ausgabe nur einmal pro Dataset-Instanz, um Skalen zu prüfen (kein Einfluss auf Verhalten).
        if not self._logged_debug:
            print(
                f"[DEBUG][datasets] AP min/max: {ap.min().item():.3e}/{ap.max().item():.3e} | "
                f"PA min/max: {pa.min().item():.3e}/{pa.max().item():.3e} | "
                f"CT min/max: {ct.min().item():.3e}/{ct.max().item():.3e} | "
                f"ACT min/max: {(act.min().item() if act.numel()>0 else float('nan')):.3e}/"
                f"{(act.max().item() if act.numel()>0 else float('nan')):.3e}",
                flush=True,
            )
            print(
                f"[DEBUG][datasets] Shapes after layout permute: AP {tuple(ap.shape)}, PA {tuple(pa.shape)}, "
                f"CT {tuple(ct.shape)} (depth axis=0), ACT {tuple(act.shape)}",
                flush=True,
            )
            self._logged_debug = True

        return {                                                                # Rückgabe ist ein Dictionary   
            "ap": ap,
            "pa": pa,
            "ct": ct,
            "act": act,
            "meta": {
                "patient_id": e["patient_id"],
                "act_scale": self.act_scale,
                "ap_path": str(e["ap_path"]),
                "pa_path": str(e["pa_path"]),
                "ct_path": str(e["ct_path"]),
                "act_path": str(e["act_path"]) if e["act_path"] is not None else None,
            },
        }

    def _normalize_projection(self, tensor: torch.Tensor) -> torch.Tensor:
        """Einfache per-Projektion-Normierung auf [0,1], wie im ursprünglichen Code."""
        maxv = tensor.max()
        if maxv > 0:
            tensor = tensor / maxv
        return tensor
