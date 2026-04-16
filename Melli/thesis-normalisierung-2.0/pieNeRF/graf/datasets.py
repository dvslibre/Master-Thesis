"""Dataset utilities for SPECT data: load AP/PA projections, CT/act volumes from manifest CSV."""

import glob
import numpy as np
from PIL import Image
from pathlib import Path
import csv
import torch

from torchvision.datasets.vision import VisionDataset



_CT_ACT_STATS_LOGGED = False

def _quantile_sample(flat: torch.Tensor, max_samples: int = 200_000) -> torch.Tensor:
    sample_size = min(flat.numel(), max_samples)
    if sample_size < flat.numel():
        indices = torch.randint(flat.numel(), (sample_size,), device=flat.device)
        flat = flat[indices]
    if flat.device.type == "cuda":
        flat = flat.cpu()
    return flat

def _safe_quantile(tensor: torch.Tensor, quantile: float) -> float:
    sample = _quantile_sample(tensor.reshape(-1))
    return float(torch.quantile(sample, quantile).item())

def _tensor_stats_tuple(tensor: torch.Tensor):
    if tensor.numel() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    flat = tensor.reshape(-1)
    return (
        float(flat.min().item()),
        float(flat.max().item()),
        float(flat.mean().item()),
        _safe_quantile(flat, 0.01),
        _safe_quantile(flat, 0.5),
        _safe_quantile(flat, 0.99),
    )

def _axis_sums(vol: torch.Tensor):
    sum_over_x = vol.sum(dim=(0, 2))
    sum_over_y = vol.sum(dim=(0, 1))
    sum_over_z = vol.sum(dim=(1, 2))
    return sum_over_x, sum_over_y, sum_over_z

def _center_of_mass(vol: torch.Tensor, mask: torch.Tensor):
    mask_float = mask.to(vol.dtype)
    masked = vol * mask_float
    mass = masked.sum()
    if mass <= 0:
        return torch.tensor([float("nan"), float("nan"), float("nan")], device=vol.device)
    D, H, W = vol.shape
    z_coords = torch.arange(D, device=vol.device, dtype=vol.dtype)
    y_coords = torch.arange(H, device=vol.device, dtype=vol.dtype)
    x_coords = torch.arange(W, device=vol.device, dtype=vol.dtype)
    com_z = (masked.sum(dim=(1, 2)) * z_coords).sum() / mass
    com_y = (masked.sum(dim=(0, 2)) * y_coords).sum() / mass
    com_x = (masked.sum(dim=(0, 1)) * x_coords).sum() / mass
    return torch.stack([com_z, com_y, com_x])

def _log_ct_act_once(ct: torch.Tensor, act: torch.Tensor):
    global _CT_ACT_STATS_LOGGED
    if _CT_ACT_STATS_LOGGED:
        return
    _CT_ACT_STATS_LOGGED = True
    def describe(name: str, vol: torch.Tensor):
        shape = tuple(vol.shape)
        stats = _tensor_stats_tuple(vol)
        sum_x, sum_y, sum_z = _axis_sums(vol)
        sum_stats = (
            float(sum_x.min().item()),
            float(sum_x.max().item()),
            float(sum_y.min().item()),
            float(sum_y.max().item()),
            float(sum_z.min().item()),
            float(sum_z.max().item()),
        )
        return shape, stats, sum_stats

    def compute_com(vol: torch.Tensor, thresh: float, name: str):
        if vol.numel() == 0:
            return torch.tensor([float("nan")] * 3)
        active = vol > thresh
        if not active.any():
            active = vol > 0.0
        if not active.any():
            active = vol >= 0.0
        return _center_of_mass(vol, active)

    if ct.numel() > 0:
        shape, stats, sum_stats = describe("CT", ct)
        print(
            f"[debug][ct] shape={shape} stats(min/max/mean/p1/p50/p99)=({stats[0]:.3e},{stats[1]:.3e},"
            f"{stats[2]:.3e},{stats[3]:.3e},{stats[4]:.3e},{stats[5]:.3e}) "
            f"axis_sums(min/max)=({sum_stats[0]:.3e}/{sum_stats[1]:.3e}->{sum_stats[2]:.3e}/{sum_stats[3]:.3e}->"
            f"{sum_stats[4]:.3e}/{sum_stats[5]:.3e})",
            flush=True,
        )
        thresh_ct = max(_safe_quantile(ct, 0.8), 0.1)
        ct_com = compute_com(ct, thresh_ct, "CT").cpu().tolist()
        print(f"[debug][ct] CoM(z,y,x) above thresh {thresh_ct:.3e}: {ct_com}", flush=True)
    if act.numel() > 0:
        shape, stats, sum_stats = describe("ACT", act)
        print(
            f"[debug][act] shape={shape} stats(min/max/mean/p1/p50/p99)=({stats[0]:.3e},{stats[1]:.3e},"
            f"{stats[2]:.3e},{stats[3]:.3e},{stats[4]:.3e},{stats[5]:.3e}) "
            f"axis_sums(min/max)=({sum_stats[0]:.3e}/{sum_stats[1]:.3e}->{sum_stats[2]:.3e}/{sum_stats[3]:.3e}->"
            f"{sum_stats[4]:.3e}/{sum_stats[5]:.3e})",
            flush=True,
        )
        thresh_act = max(_safe_quantile(act, 0.5), 1e-6)
        act_com = compute_com(act, thresh_act, "ACT").cpu().tolist()
        print(f"[debug][act] CoM(z,y,x) above thresh {thresh_act:.3e}: {act_com}", flush=True)


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
        proj_input_source: str = "counts",
    ):  
        super().__init__()
        self.manifest_path = Path(manifest_path)                                # manifest_path = Pfad zur csv mit Spalten phantom_id, ap_path, pa_path, ct_path
        self._manifest_dir = self.manifest_path.parent
        self.imsize = imsize                                                    # momentan nicht genutzt
        self.transform_img = transform_img                                      # optionale Transformationsfunktionen für AP/PA und CT
        self.transform_ct = transform_ct                                        # "
        self.act_scale = float(act_scale)                                       # globaler Faktor für ACT/λ (keine Normierung)
        self.ct_prefer_raw = bool(ct_prefer_raw)
        self.proj_input_source = str(proj_input_source or "counts").lower()
        if self.proj_input_source not in {"counts", "normalized"}:
            raise ValueError(f"Unknown proj_input_source: {proj_input_source}")
        self._logged_debug = False                                               # sorgt dafür, dass Debug-Ausgabe nur einmal erfolgt
        self._debug_proj_stats = bool(debug_proj_stats)
        self._logged_debug_proj_stats = False
        self._warned_scale_mismatch = False
        self._logged_proj_input_stats = False

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

    def get_patient_id(self, idx):
        """Liefert patient_id ohne Bild-/Volumen-Loading (nur Manifest-Metadaten)."""
        if idx < 0 or idx >= len(self.entries):
            raise IndexError(f"Index out of range: {idx}")
        return self.entries[idx].get("patient_id")

    def get_meta_light(self, idx):
        """Leichtgewichtige Meta-Infos ohne Bild-/Volumen-Loading."""
        if idx < 0 or idx >= len(self.entries):
            raise IndexError(f"Index out of range: {idx}")
        e = self.entries[idx]
        return {
            "patient_id": e.get("patient_id"),
            "ap_path": str(e.get("ap_path")) if e.get("ap_path") is not None else "",
            "pa_path": str(e.get("pa_path")) if e.get("pa_path") is not None else "",
            "ct_path": str(e.get("ct_path")) if e.get("ct_path") is not None else "",
            "act_path": str(e.get("act_path")) if e.get("act_path") is not None else "",
            "ap_counts_path": str(e.get("ap_counts_path")) if e.get("ap_counts_path") is not None else "",
            "pa_counts_path": str(e.get("pa_counts_path")) if e.get("pa_counts_path") is not None else "",
            "proj_scale_joint_p99": float(e.get("proj_scale_joint_p99")) if e.get("proj_scale_joint_p99") is not None else float("nan"),
            "proj_scale_joint_p99_missing": bool(e.get("proj_scale_joint_p99_missing")),
        }

    def capture_orientation_summary(self, sample):
        """Loggt einfache Orientierungs- und Shape-Infos eines Samples (AP/PA/CT/ACT)."""
        if not isinstance(sample, dict):
            print("[datasets][orientation] sample missing; skipping orientation summary", flush=True)
            return
        patient_id = None
        meta = sample.get("meta")
        if isinstance(meta, dict):
            patient_id = meta.get("patient_id")
            if torch.is_tensor(patient_id):
                patient_id = patient_id.view(-1)[0].item() if patient_id.numel() > 0 else None
        ap = sample.get("ap")
        pa = sample.get("pa")
        ct = sample.get("ct")
        act = sample.get("act")
        def _shape(tensor):
            if tensor is None:
                return "None"
            try:
                return tuple(int(x) for x in tensor.shape)
            except Exception:
                return "?"
        print(
            "[datasets][orientation] "
            f"patient={patient_id or 'unknown'} "
            f"ap_shape={_shape(ap)} pa_shape={_shape(pa)} ct_shape={_shape(ct)} act_shape={_shape(act)}",
            flush=True,
        )


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
        _log_ct_act_once(ct, act)
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

        counts_available = (ap_counts.numel() > 0) and (pa_counts.numel() > 0)
        use_counts_for_input = self.proj_input_source == "counts" and counts_available
        proj_input_source_used = "counts" if use_counts_for_input else "normalized"
        proj_input_ap = ap_counts if use_counts_for_input else ap
        proj_input_pa = pa_counts if use_counts_for_input else pa

        if not self._logged_proj_input_stats:
            self._logged_proj_input_stats = True
            def _min_max_mean(tensor: torch.Tensor) -> tuple[float, float, float]:
                if tensor.numel() == 0:
                    return float("nan"), float("nan"), float("nan")
                flat = tensor.reshape(-1)
                return float(flat.min().item()), float(flat.mean().item()), float(flat.max().item())

            ap_counts_stats = _min_max_mean(ap_counts)
            pa_counts_stats = _min_max_mean(pa_counts)
            proj_ap_stats = _min_max_mean(proj_input_ap)
            proj_pa_stats = _min_max_mean(proj_input_pa)
            counts_status = "available" if counts_available else "missing"
            print(
                f"[DEBUG][proj-input] source_requested={self.proj_input_source} counts={counts_status} | "
                f"ap_counts(min/mean/max)={ap_counts_stats[0]:.3e}/{ap_counts_stats[1]:.3e}/{ap_counts_stats[2]:.3e} | "
                f"pa_counts(min/mean/max)={pa_counts_stats[0]:.3e}/{pa_counts_stats[1]:.3e}/{pa_counts_stats[2]:.3e} | "
                f"encoder_input(min/mean/max) AP={proj_ap_stats[0]:.3e}/{proj_ap_stats[1]:.3e}/{proj_ap_stats[2]:.3e} "
                f"PA={proj_pa_stats[0]:.3e}/{proj_pa_stats[1]:.3e}/{proj_pa_stats[2]:.3e}",
                flush=True,
            )

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
            "proj_input_ap": proj_input_ap,
            "proj_input_pa": proj_input_pa,
            "proj_input_source_used": proj_input_source_used,
            "ap_counts": ap_counts,
            "pa_counts": pa_counts,
        }

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return (self._manifest_dir / path).resolve()
