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
    Lädt pro Fall AP, PA und CT (alles als .npy), basierend auf einem Manifest (CSV).
    """
    def __init__(self, manifest_path, imsize=None, transform_img=None, transform_ct=None):  
        super().__init__()
        self.manifest_path = Path(manifest_path)    # manifest_path = Pfad zur csv mit Spalten phantom_id, ap_path, pa_path, ct_path
        self.imsize = imsize                        # momentan nicht genutzt
        self.transform_img = transform_img          # optionale Transformationsfunktionen für AP/PA und CT
        self.transform_ct = transform_ct            # "

        self.entries = []                                   # Liste in der für jeden Fall ein kleines Dict mit Pfaden & ID steht
        with open(self.manifest_path, newline="") as f:     # CSV öffnen
            reader = csv.DictReader(f)                      # liest jede Zeile als dict
            for row in reader:
                act_path = row.get("act_path")
                if act_path is None:
                    # optional automatisch aus dem Ordner ableiten
                    candidate = Path(row["ap_path"]).with_name("act.npy")
                    act_path = candidate if candidate.exists() else None
                else:
                    act_path = Path(act_path)

                self.entries.append({                       # jeweils als Path-Objekte speichern: phantom_id, ap_path, pa_path, ct_path
                    "patient_id": row["patient_id"],
                    "ap_path": Path(row["ap_path"]),
                    "pa_path": Path(row["pa_path"]),
                    "ct_path": Path(row["ct_path"]),
                    "act_path": act_path,
                })

    def __len__(self):
        return len(self.entries)                    # Anzahl der Einträge = Anzahl Zeilen im Manifest = Anzahl Phantome/Patienten

    def _load_npy_image(self, path):                # lädt .npy Array, z.B. [H,W] mit Counts
        arr = np.load(path).astype(np.float32)      # stellt sicher, dass float32
        # [H,W] -> [1,H,W]
        tensor = torch.from_numpy(arr).unsqueeze(0) # macht einen Tensor draus [H,W]

        # einfache Normierung auf [0,1]
        maxv = tensor.max()                         # Maximalwert suchen
        if maxv > 0:
            tensor = tensor / maxv

        # optionales Resize kann mann später einbauen, wenn nötig
        return tensor

    def _load_npy_ct(self, path):
        if path is None:
            return torch.empty(0)

        # Original gespeichert als (LR, AP, SI)
        vol = np.load(path).astype(np.float32)

        # 👉 korrekt in (AP, SI, LR) = (D,H,W) permutieren
        vol = np.transpose(vol, (1, 2, 0))

        # optionaler scale-factor (falls du den drin hattest)
        vol *= 10.0

        return torch.from_numpy(vol)

    def _load_npy_act(self, path):
        if path is None:
            return torch.empty(0)

        vol = np.load(path).astype(np.float32)

        # selbe Permutation wie CT!
        vol = np.transpose(vol, (1, 2, 0))

        maxv = vol.max()
        if maxv > 0:
            vol = vol / maxv

        return torch.from_numpy(vol)

    def __getitem__(self, idx):
        e = self.entries[idx]                           # holt das idx-te Manifest-Dict
        ap = self._load_npy_image(e["ap_path"])         # lädt AP als [1,H,W]-Tensor mit Normierung [0,1]
        pa = self._load_npy_image(e["pa_path"])         # lädt PA als [1,H,W]-Tensor mit Normierung [0,1]
        ct = self._load_npy_ct(e["ct_path"])            # lädt CT Volumen, normiert    
        act = self._load_npy_act(e["act_path"])

        if self.transform_img is not None:              # optionale zusätzliche Schritte (z.B. Resize, Cropping, ...)
            ap = self.transform_img(ap)
            pa = self.transform_img(pa)
        if self.transform_ct is not None:
            ct = self.transform_ct(ct)

        return {                                         # Rückgabe ist ein Dictionary   
            "ap": ap,
            "pa": pa,
            "ct": ct,
            "act": act,
            "meta": {
                "patient_id": e["patient_id"],
            },
        }
