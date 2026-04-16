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
        debug_proj_stats: bool = False,
        act_scale: float = 1.0,
        ct_prefer_raw: bool = False,
    ):  
        super().__init__()
        self.manifest_path = Path(manifest_path)                                # manifest_path = Pfad zur csv mit Spalten phantom_id, ap_path, pa_path, ct_path
        self._manifest_dir = self.manifest_path.parent
        self.imsize = imsize                                                    # momentan nicht genutzt
        self.transform_img = transform_img                                      # optionale Transformationsfunktionen für AP/PA und CT
        self.transform_ct = transform_ct                                        # "
        self.act_scale = float(act_scale)                                       # globaler Faktor für ACT/λ (keine Normierung)
        self.ct_prefer_raw = bool(ct_prefer_raw)
        self._logged_debug = False                                               # sorgt dafür, dass Debug-Ausgabe nur einmal erfolgt
        self._debug_proj_stats = bool(debug_proj_stats)
        self._logged_debug_proj_stats = False
        self._warned_scale_mismatch = False

        self.entries = []                                                       # Liste in der für jeden Fall ein kleines Dict mit Pfaden & ID steht
        with open(self.manifest_path, newline="") as f:                         # CSV öffnen
            reader = csv.DictReader(f)                                          # liest jede Zeile als dict
            for row in reader:
                act_path = row.get("act_path")
                if act_path is None or act_path == "":
                    # optional automatisch aus dem Ordner ableiten
                    candidate = self._resolve_path(row["ap_path"]).with_name("act.npy")
                    act_path = candidate if candidate.exists() else None
                else:
                    act_path = self._resolve_path(act_path)

                proj_scale = row.get("proj_scale_joint_p99")
                proj_scale_val = None
                if proj_scale is not None and proj_scale != "":
                    try:
                        proj_scale_val = float(proj_scale)
                    except Exception:
                        proj_scale_val = None
                proj_scale_missing = proj_scale_val is None

                self.entries.append({                                           # jeweils als Path-Objekte speichern: phantom_id, ap_path, pa_path, ct_path
                    "patient_id": row["patient_id"],
                    "ap_path": self._resolve_path(row["ap_path"]),
                    "pa_path": self._resolve_path(row["pa_path"]),
                    "ct_path": self._resolve_path(row["ct_path"]),
                    "act_path": act_path,
                    "ap_counts_path": None,
                    "pa_counts_path": None,
                    "proj_scale_joint_p99": proj_scale_val,
                    "proj_scale_joint_p99_missing": proj_scale_missing,
                })

        # optionale Counts-Targets automatisch aus Nachbarschaft ableiten
        for e in self.entries:
            ap_counts = e["ap_path"].with_name("ap_counts.npy")
            pa_counts = e["pa_path"].with_name("pa_counts.npy")
            if ap_counts.exists():
                e["ap_counts_path"] = ap_counts
            if pa_counts.exists():
                e["pa_counts_path"] = pa_counts

    def __len__(self):
        return len(self.entries)                                                # Anzahl der Einträge = Anzahl Zeilen im Manifest = Anzahl Phantome/Patienten


    @staticmethod
    def _tensor_stats(t: torch.Tensor):
        if t.numel() == 0:
            return float("nan"), float("nan"), float("nan")
        t = t.detach()
        return (
            float(t.min().item()),
            float(t.max().item()),
            float(torch.quantile(t.reshape(-1), 0.999).item()),
        )


    def _load_npy_image(self, path):                                            # lädt .npy Array, z.B. [H,W] mit Counts
        arr = np.load(path).astype(np.float32)                                  # stellt sicher, dass float32
        
        tensor = torch.from_numpy(arr).unsqueeze(0)                             # macht einen Tensor draus [H,W] -> [1,H,W]

        return tensor


    def _load_npy_ct(self, path):
        if path is None:
            return torch.empty(0)
        ct_path = self._resolve_ct_path(path)
        vol = np.load(ct_path).astype(np.float32)                               # Original gespeichert als (LR, AP/Depth, SI)
        vol = np.transpose(vol, (1, 0, 2))                                      # Layout: (AP, LR, SI) = (D,H,W) für Renderer

        return torch.from_numpy(vol)

    def _resolve_ct_path(self, path: Path) -> Path:
        if path is None:
            return path
        ct_path = path
        if self.ct_prefer_raw:
            name = ct_path.name
            if name.endswith("_norm.npy"):
                raw_name = name.replace("_norm.npy", ".npy")
                raw_path = ct_path.with_name(raw_name)
                if raw_path.exists():
                    ct_path = raw_path
        return ct_path


    def _load_npy_act(self, path):
        if path is None:
            return torch.empty(0)

        vol = np.load(path).astype(np.float32)
        vol = np.transpose(vol, (1, 0, 2))                                      # Layout-Anpassung wie CT: (AP, LR, SI) = (D,H,W)

        # Kein Min/Max-Scaling mehr – optional nur globaler Faktor, damit λ im festen Maßstab bleibt.
        vol = vol * self.act_scale

        return torch.from_numpy(vol)

    def _load_npy_counts(self, path):
        if path is None:
            return torch.empty(0)
        arr = np.load(path).astype(np.float32)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor


    def __getitem__(self, idx):
        e = self.entries[idx]                                                   # holt das idx-te Manifest-Dict
        ap = self._load_npy_image(e["ap_path"])                                 # lädt AP als [1,H,W]-Tensor (roh, keine Min/Max-Norm)
        pa = self._load_npy_image(e["pa_path"])                                 # lädt PA als [1,H,W]-Tensor (roh, keine Min/Max-Norm)
        ct_path_used = self._resolve_ct_path(e["ct_path"])
        ct = self._load_npy_ct(e["ct_path"])                                    # lädt CT Volumen (gescaled)  
        act = self._load_npy_act(e["act_path"])                                 # lädt ACT Volumen (ohne Normierung, nur globaler Faktor)
        ap_counts = self._load_npy_counts(e.get("ap_counts_path"))
        pa_counts = self._load_npy_counts(e.get("pa_counts_path"))

        ap_pre_stats = self._tensor_stats(ap)
        pa_pre_stats = self._tensor_stats(pa)

        if self.transform_img is not None:                                      # optionale zusätzliche Schritte (z.B. Resize, Cropping, ...)
            ap = self.transform_img(ap)
            pa = self.transform_img(pa)
        if self.transform_ct is not None:
            ct = self.transform_ct(ct)

        # Debug-Ausgabe nur einmal pro Dataset-Instanz, um Skalen zu prüfen (kein Einfluss auf Verhalten).
        if self._debug_proj_stats and not self._logged_debug_proj_stats:
            print(
                f"[DEBUG][proj] AP min/max/p99.9={ap_pre_stats[0]:.3e}/{ap_pre_stats[1]:.3e}/{ap_pre_stats[2]:.3e} | "
                f"PA min/max/p99.9={pa_pre_stats[0]:.3e}/{pa_pre_stats[1]:.3e}/{pa_pre_stats[2]:.3e} | "
                f"proj_scale_joint_p99={e.get('proj_scale_joint_p99')} "
                f"(missing={e.get('proj_scale_joint_p99_missing')})",
                flush=True,
            )
            self._logged_debug_proj_stats = True

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

        ap_p999 = ap_pre_stats[2]
        pa_p999 = pa_pre_stats[2]
        hi = max(ap_p999, pa_p999)
        lo = min(ap_p999, pa_p999)
        if hi > 0.9 and lo < 0.3 and (hi / max(lo, 1e-8)) > 3.0 and not self._warned_scale_mismatch:
            print(
                "[WARN] AP/PA p99.9 stark unterschiedlich (moegliche per-view Normierung). "
                "Bitte Preprocessing mit joint p99.9 erneut ausfuehren.",
                flush=True,
            )
            self._warned_scale_mismatch = True

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
                "ct_path_used": str(ct_path_used) if ct_path_used is not None else "",
                "act_path": str(e["act_path"]) if e["act_path"] is not None else "",
                "act_path_missing": e["act_path"] is None,
                "ap_counts_path": str(e["ap_counts_path"]) if e.get("ap_counts_path") is not None else "",
                "pa_counts_path": str(e["pa_counts_path"]) if e.get("pa_counts_path") is not None else "",
                "proj_scale_joint_p99": float(e["proj_scale_joint_p99"]) if e.get("proj_scale_joint_p99") is not None else float("nan"),
                "proj_scale_joint_p99_missing": bool(e.get("proj_scale_joint_p99_missing")),
            },
            "ap_counts": ap_counts,
            "pa_counts": pa_counts,
        }
    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return (self._manifest_dir / path).resolve()
