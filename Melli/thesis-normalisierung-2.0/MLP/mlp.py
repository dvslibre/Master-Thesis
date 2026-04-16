#!/usr/bin/env python3
"""
mlp_raynet_baseline.py — Einfaches MLP-Baseline-Netz für RayNet

Idee:
- Gleiches NPZ-Handling wie raynet_test.py (NPZRays, NPZRaysSubset)
- Gleicher Train/Test-Split über num_rays und seed
- Gleiche Physik-Aggregation für I_ap_hat / I_pa_hat
- Gleiche Metriken (RMSE, RRMSE, Sparsity, corr(a,µ))
- ABER: statt Segment-RayNet wird ein simples MLP benutzt,
  das pro Strahl einen Vektor a[0..N-1] vorhersagt.

Aufruf (Beispiel):

python mlp.py \
  --data /home/mnguest12/projects/thesis/RayNet/phantom_04/out/rays_train.npz \
  --device cuda \
  --epochs 40 \
  --batch_size 32 \
  --lr 1e-3 \
  --loss loghuber --huber_delta 100 \
  --alpha 1.0 --beta 0.3 \
  --lambda_l1 0 \
  --lambda_bg 0 --bg_eps 5e-4 \
  --lambda_off 0 \
  --num_rays 3000 \
  --hidden 256 --depth 3 --dropout 0.1 \
  --out /home/mnguest12/projects/thesis/MLP/results \
  --use_I --use_xy

"""
from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os


# ---------------- Dataset ----------------
class NPZRays(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)

        self.mu      = torch.from_numpy(data["mu_seq"]).float()      # [M,N]
        self.T_ap    = torch.from_numpy(data["T_ap_seq"]).float()    # [M,N]
        self.T_pa    = torch.from_numpy(data["T_pa_seq"]).float()    # [M,N]
        self.I_pairs = torch.from_numpy(data["I_pairs"]).float()     # [M,2]

        self.a_gt = torch.from_numpy(data["a_seq"]).float() \
            if "a_seq" in data.files else None

        self.mask_npz = torch.from_numpy(data["mask_meas"]).float() \
            if "mask_meas" in data.files else None

        self.ds = float(data["ds"]) if "ds" in data.files else 1.0

        # Größe merken
        self.M, self.N = self.mu.shape

        # optionale Strahlkoordinaten (Pixel-Indizes) -> auf [-1,1] normieren
        if "xy_pairs" in data.files:
            xy = data["xy_pairs"].astype(np.float32)  # [M,2], (x,y) Pixel
            x_max = max(float(xy[:, 0].max()), 1.0)
            y_max = max(float(xy[:, 1].max()), 1.0)
            x_norm = (xy[:, 0] / x_max) * 2.0 - 1.0
            y_norm = (xy[:, 1] / y_max) * 2.0 - 1.0
            xy_norm = np.stack([x_norm, y_norm], axis=1)  # [M,2]
            self.xy_norm = torch.from_numpy(xy_norm).float()
        else:
            self.xy_norm = None

        # Default-Messmaske: Strahlen, bei denen mind. eine der beiden
        # Intensitäten > 0 ist
        with torch.no_grad():
            eps = 1e-13  # numerische Toleranz für "aktive" Strahlen
            nz = (self.I_pairs[:, 0].abs() > eps) | (self.I_pairs[:, 1].abs() > eps)
            self.mask_default = nz.float()

            frac = self.mask_default.mean().item() * 100.0
            print(f"[INFO] Active rays (|I|>{eps:g}): {int(nz.sum())} / {len(nz)} ({frac:.2f}%)")

    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        d = {
            "mu":       self.mu[idx],          # [N]
            "T_ap":     self.T_ap[idx],        # [N]
            "T_pa":     self.T_pa[idx],        # [N]
            "I_ap":     self.I_pairs[idx, 0],  # []
            "I_pa":     self.I_pairs[idx, 1],  # []
            "ds":       torch.tensor(self.ds, dtype=torch.float32),
            "mask_def": self.mask_default[idx],
        }
        if self.xy_norm is not None:
            d["xy"] = self.xy_norm[idx]        # [2], normierte Koords
        if self.a_gt is not None:
            d["a_gt"] = self.a_gt[idx]
        if self.mask_npz is not None:
            d["mask_npz"] = self.mask_npz[idx]
        return d


class NPZRaysSubset(Dataset):
    """Subset-Variante von NPZRays für Train/Test-Splits."""
    def __init__(self, base: NPZRays, indices):
        idx = torch.as_tensor(indices, dtype=torch.long)

        self.mu      = base.mu[idx]
        self.T_ap    = base.T_ap[idx]
        self.T_pa    = base.T_pa[idx]
        self.I_pairs = base.I_pairs[idx]
        self.ds      = base.ds

        self.xy_norm = base.xy_norm[idx] if getattr(base, "xy_norm", None) is not None else None
        self.a_gt     = base.a_gt[idx] if base.a_gt is not None else None
        self.mask_npz = base.mask_npz[idx] if base.mask_npz is not None else None

        self.M, self.N = self.mu.shape

        with torch.no_grad():
            nz = (self.I_pairs[:, 0] > 0) | (self.I_pairs[:, 1] > 0)
            self.mask_default = nz.float()

    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        d = {
            "mu":       self.mu[idx],
            "T_ap":     self.T_ap[idx],
            "T_pa":     self.T_pa[idx],
            "I_ap":     self.I_pairs[idx, 0],
            "I_pa":     self.I_pairs[idx, 1],
            "ds":       torch.tensor(self.ds, dtype=torch.float32),
            "mask_def": self.mask_default[idx],
        }
        if self.a_gt is not None:
            d["a_gt"] = self.a_gt[idx]
        if self.mask_npz is not None:
            d["mask_npz"] = self.mask_npz[idx]
        if self.xy_norm is not None:
            d["xy"] = self.xy_norm[idx]
        return d


# ---------------- Physics aggregator ----------------
class PhysicsAggregator(nn.Module):
    def forward(self, a: torch.Tensor, T_ap: torch.Tensor, T_pa: torch.Tensor, ds: torch.Tensor | float):
        if not torch.is_tensor(ds):
            ds = torch.tensor(ds, dtype=a.dtype, device=a.device)
        ds = ds.reshape(-1, 1)  # [B,1]
        T_ap = torch.nan_to_num(T_ap)
        T_pa = torch.nan_to_num(T_pa)
        I_ap_hat = torch.sum((a * T_ap) * ds, dim=1)
        I_pa_hat = torch.sum((a * T_pa) * ds, dim=1)
        return torch.nan_to_num(I_ap_hat), torch.nan_to_num(I_pa_hat)


# ---------------- Einfaches MLP-Modell ----------------
class MLPActivityNet(nn.Module):
    """Einfaches MLP, das pro Strahl a[0..N-1] vorhersagt.

    Eingabe-Features:
      - Sequenzteil (pro Bin): mu, T_ap, T_pa  →  3N Dimensionen
      - Skalarteil (Ray-Level, *nicht* pro Bin repliziert):
          * ds (immer als Feature)
          * optional I_ap, I_pa (wenn use_I=True)
          * optional (x,y) (wenn use_xy=True)

    Alles wird pro Strahl zu einem Vektor geflattet und durch ein MLP
    geschickt. Ausgabe ist a_raw[0..N-1], das über Softplus >=0 gemacht wird.
    """

    def __init__(self, N: int, hidden: int = 256, depth: int = 3,
                 dropout: float = 0.0, use_I: bool = True, use_xy: bool = False):
        super().__init__()
        self.N = N
        self.use_I = use_I
        self.use_xy = use_xy
        self.phys = PhysicsAggregator()

        # Sequenz-Teil: mu, T_ap, T_pa → 3N
        seq_dim = 3 * N
        # Skalar-Teil: ds (immer), optional I_ap/I_pa, optional (x,y)
        scalar_dim = 1  # ds
        if use_I:
            scalar_dim += 2  # I_ap, I_pa
        if use_xy:
            scalar_dim += 2  # x, y
        in_dim = seq_dim + scalar_dim

        layers = []
        last_dim = in_dim
        for i in range(depth - 1):
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            last_dim = hidden
        layers.append(nn.Linear(last_dim, N))  # direkt N-Bins Ausgabe
        self.mlp = nn.Sequential(*layers)

    def forward(self, mu, T_ap, T_pa, I_ap, I_pa, ds,
                xy: Optional[torch.Tensor] = None):
        # mu, T_ap, T_pa: [B,N]
        B, N = mu.shape
        assert N == self.N, "N in Daten und Modell muss übereinstimmen"

        # Sequenz-Teil: mu/T_ap/T_pa hintereinander
        seq = torch.cat([mu, T_ap, T_pa], dim=1)  # [B, 3N]

        # Skalar-Teil: Ray-Level-Features
        scalars = []
        # ds ist immer Feature (Ray-Länge/Voxelschrittweite)
        scalars.append(ds.view(B, 1))            # [B,1]

        if self.use_I:
            scalars.append(I_ap.view(B, 1))      # [B,1]
            scalars.append(I_pa.view(B, 1))      # [B,1]

        if self.use_xy:
            if xy is None:
                raise ValueError("Model configured with use_xy=True but xy not provided.")
            # xy: [B,2] (bereits Ray-Level), unverändert anhängen
            scalars.append(xy.view(B, 2))        # [B,2]

        if scalars:
            scal = torch.cat(scalars, dim=1)     # [B, scalar_dim]
            feat = torch.cat([seq, scal], dim=1) # [B, 3N + scalar_dim]
        else:
            feat = seq

        a_raw = self.mlp(feat)                   # [B,N]
        a_eff = F.softplus(a_raw)                # >=0

        I_ap_hat, I_pa_hat = self.phys(a_eff, T_ap, T_pa, ds)
        return a_eff, I_ap_hat, I_pa_hat


# ---------------- Losses & Metriken ----------------

def meas_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "mse",
              mask: Optional[torch.Tensor] = None, huber_delta: float = 100.0) -> torch.Tensor:
    if kind in ("logmae", "loghuber"):
        pred = torch.log1p(torch.clamp(pred, min=0.0))
        target = torch.log1p(torch.clamp(target, min=0.0))
    if kind in ("mae", "logmae"):
        err = torch.abs(pred - target)
    elif kind == "mse":
        err = (pred - target) ** 2
    elif kind == "loghuber":
        err = F.huber_loss(pred, target, delta=huber_delta, reduction="none")
    else:
        raise ValueError(f"Unknown loss kind: {kind}")
    if mask is None:
        return err.mean()
    w = mask.view(-1)
    err = err.view(-1)
    return (err * w).sum() / (w.sum() + 1e-8)


@torch.no_grad()
def pearson_corr_per_ray(a: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Pearson a vs. µ pro Ray, Mittel über Batch; a,mu: [B,N]."""
    a = a.float()
    mu = mu.float()
    a_m = a.mean(dim=1, keepdim=True)
    mu_m = mu.mean(dim=1, keepdim=True)
    a_c = a - a_m
    mu_c = mu - mu_m
    num = (a_c * mu_c).sum(dim=1)
    den = torch.sqrt((a_c.pow(2).sum(dim=1) + 1e-12) * (mu_c.pow(2).sum(dim=1) + 1e-12))
    r = num / (den + 1e-12)
    return r.mean()


# ---------------- Training ----------------
@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 16
    lr: float = 1e-3
    alpha: float = 1.0       # Gewicht Mess-Loss
    beta: float = 1e-2       # Gewicht supervised (a)
    device: str = "cpu"
    use_supervised: bool = True
    loss_kind: str = "logmae"
    huber_delta: float = 100.0
    mask_nonzero: bool = True
    seed: int = 0
    # Regularisierung
    lambda_l1: float = 0.0
    lambda_bg: float = 0.0
    bg_eps: float = 5e-4
    lambda_off: float = 0.0


def train_one_epoch(model: MLPActivityNet, loader: DataLoader,
                    opt: torch.optim.Optimizer, cfg: TrainConfig) -> Dict[str, float]:
    model.train()
    stats = {
        "loss": 0.0, "loss_I": 0.0, "loss_sup": 0.0,
        "loss_l1": 0.0, "loss_bg": 0.0, "loss_off": 0.0,
        "rmse_I_ap": 0.0, "rmse_I_pa": 0.0, "rmse_a": 0.0,
        "mae_I_ap": 0.0, "mae_I_pa": 0.0,
        "logrmse_ap": 0.0, "logrmse_pa": 0.0,
        "spars": 0.0, "corr_am": 0.0,
    }
    n = 0

    for batch in loader:
        mu = batch["mu"].to(cfg.device)
        T_ap = batch["T_ap"].to(cfg.device)
        T_pa = batch["T_pa"].to(cfg.device)
        I_ap = batch["I_ap"].to(cfg.device)
        I_pa = batch["I_pa"].to(cfg.device)
        ds = batch["ds"].to(cfg.device)
        xy = batch.get("xy")
        xy = xy.to(cfg.device) if xy is not None else None

        a_gt = batch.get("a_gt")
        a_gt = a_gt.to(cfg.device) if a_gt is not None else None

        m_npz = batch.get("mask_npz")
        m_def = batch.get("mask_def")
        mask = m_npz.to(cfg.device) if m_npz is not None else (
            m_def.to(cfg.device) if cfg.mask_nonzero and m_def is not None else None
        )

        opt.zero_grad(set_to_none=True)
        a_eff, I_ap_hat, I_pa_hat = model(mu, T_ap, T_pa, I_ap, I_pa, ds, xy=xy)

        # Kein per-Batch-Rescaling: physikalische Skala der Intensitäten
        I_ap_hat_n, I_pa_hat_n = I_ap_hat, I_pa_hat
        I_ap_n, I_pa_n         = I_ap, I_pa

        l_ap = meas_loss(I_ap_hat_n, I_ap_n, kind=cfg.loss_kind,
                         mask=mask, huber_delta=cfg.huber_delta)
        l_pa = meas_loss(I_pa_hat_n, I_pa_n, kind=cfg.loss_kind,
                         mask=mask, huber_delta=cfg.huber_delta)
        loss_I = l_ap + l_pa

        if cfg.use_supervised and a_gt is not None:
            w_bg = (a_gt < cfg.bg_eps).float()
            w_pos = 1.0 - w_bg
            w = 3.0 * w_bg + 1.0 * w_pos
            loss_sup = ((a_eff - a_gt) ** 2 * w).mean()
        else:
            loss_sup = a_eff.new_tensor(0.0)

        loss_l1 = cfg.lambda_l1 * a_eff.mean()
        loss_bg = a_eff.new_tensor(0.0)
        if a_gt is not None and cfg.lambda_bg > 0:
            w_bg = (a_gt < cfg.bg_eps).float()
            loss_bg = cfg.lambda_bg * (F.relu(a_eff - cfg.bg_eps) * w_bg).mean()

        loss_off = a_eff.new_tensor(0.0)  # Platzhalter, falls globale Offsets o.ä.

        loss = cfg.alpha * loss_I + cfg.beta * loss_sup + loss_l1 + loss_bg + loss_off
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        with torch.no_grad():
            bsz = mu.shape[0]

            rmse_I_ap = torch.sqrt(F.mse_loss(I_ap_hat, I_ap))
            rmse_I_pa = torch.sqrt(F.mse_loss(I_pa_hat, I_pa))

            rmse_I_ap_item = rmse_I_ap.item()
            rmse_I_pa_item = rmse_I_pa.item()

            mae_I_ap = F.l1_loss(I_ap_hat, I_ap).item()
            mae_I_pa = F.l1_loss(I_pa_hat, I_pa).item()

            logI_ap_hat = torch.log1p(torch.clamp(I_ap_hat, min=0.0))
            logI_ap = torch.log1p(torch.clamp(I_ap, min=0.0))
            logI_pa_hat = torch.log1p(torch.clamp(I_pa_hat, min=0.0))
            logI_pa = torch.log1p(torch.clamp(I_pa, min=0.0))
            logrmse_ap = torch.sqrt(F.mse_loss(logI_ap_hat, logI_ap)).item()
            logrmse_pa = torch.sqrt(F.mse_loss(logI_pa_hat, logI_pa)).item()

            rmse_a = torch.sqrt(F.mse_loss(a_eff, a_gt)).item() if a_gt is not None else 0.0
            spars = (a_eff < cfg.bg_eps).float().mean().item()
            corr_am = pearson_corr_per_ray(a_eff.detach(), mu.detach()).item()

            stats["loss"] += loss.item() * bsz
            stats["loss_I"] += loss_I.item() * bsz
            stats["loss_sup"] += loss_sup.item() * bsz
            stats["loss_l1"] += (loss_l1.item() if torch.is_tensor(loss_l1) else loss_l1) * bsz
            stats["loss_bg"] += (loss_bg.item() if torch.is_tensor(loss_bg) else loss_bg) * bsz
            stats["loss_off"] += (loss_off.item() if torch.is_tensor(loss_off) else loss_off) * bsz

            stats["rmse_I_ap"] += rmse_I_ap_item * bsz
            stats["rmse_I_pa"] += rmse_I_pa_item * bsz
            stats["rmse_a"] += rmse_a * bsz
            stats["mae_I_ap"] += mae_I_ap * bsz
            stats["mae_I_pa"] += mae_I_pa * bsz
            stats["logrmse_ap"] += logrmse_ap * bsz
            stats["logrmse_pa"] += logrmse_pa * bsz
            stats["spars"] += spars * bsz
            stats["corr_am"] += corr_am * bsz
            n += bsz

    for k in stats.keys():
        stats[k] = stats[k] / max(1, n)
    return stats


@torch.no_grad()
def evaluate_on_loader(model: MLPActivityNet, loader: DataLoader, cfg: TrainConfig) -> Dict[str, float]:
    model.eval()
    stats = {
        "rmse_I_ap": 0.0, "rmse_I_pa": 0.0, "rmse_a": 0.0,
        "mae_I_ap": 0.0, "mae_I_pa": 0.0,
        "logrmse_ap": 0.0, "logrmse_pa": 0.0,
        "spars": 0.0, "corr_am": 0.0,
    }
    n = 0

    for batch in loader:
        mu = batch["mu"].to(cfg.device)
        T_ap = batch["T_ap"].to(cfg.device)
        T_pa = batch["T_pa"].to(cfg.device)
        I_ap = batch["I_ap"].to(cfg.device)
        I_pa = batch["I_pa"].to(cfg.device)
        ds = batch["ds"].to(cfg.device)
        xy = batch.get("xy")
        xy = xy.to(cfg.device) if xy is not None else None

        a_gt = batch.get("a_gt")
        a_gt = a_gt.to(cfg.device) if a_gt is not None else None

        a_eff, I_ap_hat, I_pa_hat = model(mu, T_ap, T_pa, I_ap, I_pa, ds, xy=xy)

        bsz = mu.shape[0]

        rmse_I_ap = torch.sqrt(F.mse_loss(I_ap_hat, I_ap))
        rmse_I_pa = torch.sqrt(F.mse_loss(I_pa_hat, I_pa))

        rmse_I_ap_item = rmse_I_ap.item()
        rmse_I_pa_item = rmse_I_pa.item()

        mae_I_ap = F.l1_loss(I_ap_hat, I_ap).item()
        mae_I_pa = F.l1_loss(I_pa_hat, I_pa).item()

        logI_ap_hat = torch.log1p(torch.clamp(I_ap_hat, min=0.0))
        logI_ap = torch.log1p(torch.clamp(I_ap, min=0.0))
        logI_pa_hat = torch.log1p(torch.clamp(I_pa_hat, min=0.0))
        logI_pa = torch.log1p(torch.clamp(I_pa, min=0.0))
        logrmse_ap = torch.sqrt(F.mse_loss(logI_ap_hat, logI_ap)).item()
        logrmse_pa = torch.sqrt(F.mse_loss(logI_pa_hat, logI_pa)).item()

        rmse_a = torch.sqrt(F.mse_loss(a_eff, a_gt)).item() if a_gt is not None else 0.0
        spars = (a_eff < cfg.bg_eps).float().mean().item()
        corr_am = pearson_corr_per_ray(a_eff.detach(), mu.detach()).item()

        stats["rmse_I_ap"] += rmse_I_ap_item * bsz
        stats["rmse_I_pa"] += rmse_I_pa_item * bsz
        stats["rmse_a"] += rmse_a * bsz
        stats["mae_I_ap"] += mae_I_ap * bsz
        stats["mae_I_pa"] += mae_I_pa * bsz
        stats["logrmse_ap"] += logrmse_ap * bsz
        stats["logrmse_pa"] += logrmse_pa * bsz
        stats["spars"] += spars * bsz
        stats["corr_am"] += corr_am * bsz
        n += bsz

    for k in stats.keys():
        stats[k] = stats[k] / max(1, n)
    return stats


# ---------------- CLI + main ----------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1e-2)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--loss", type=str, default="logmae",
                    choices=["mse", "mae", "logmae", "loghuber"])
    ap.add_argument("--huber_delta", type=float, default=100.0)
    ap.add_argument("--mask_nonzero", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None,
                help="Optionaler Ausgabeordner für Ergebnisse (Standard: Ordner der NPZ-Datei)")
    # Regularisierung
    ap.add_argument("--lambda_l1", type=float, default=0.0)
    ap.add_argument("--lambda_bg", type=float, default=0.0)
    ap.add_argument("--bg_eps", type=float, default=5e-4)
    ap.add_argument("--lambda_off", type=float, default=0.0)
    # Modell-Hyperparameter
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--no_supervised", action="store_true")
    ap.add_argument("--no_use_I", dest="use_I", action="store_false",
                    help="I_ap/I_pa nicht als Features benutzen")
    ap.add_argument("--use_I", dest="use_I", action="store_true",
                    help="I_ap/I_pa als Features benutzen (Default)")
    ap.set_defaults(use_I=True)
    ap.add_argument("--use_xy", action="store_true",
                    help="xy-Koordinaten als Features benutzen")
    # Anzahl Strahlen
    ap.add_argument("--num_rays", type=int, default=1000,
                    help="Anzahl Strahlen K, die für Train/Test verwendet werden (wird auf M gekappt)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # ---------------- Voller Datensatz ----------------
    ds_full = NPZRays(args.data)

    # ---------------- Ausgabepfad ----------------
    if args.out is not None:
        out_dir = args.out
    else:
        out_dir = os.path.dirname(args.data)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output directory set to: {out_dir}")

    # K Strahlen auswählen (oder alles, falls num_rays >= M)
    M = ds_full.M
    K = min(args.num_rays, M)

    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed if args.seed is not None else 0)
    perm = torch.randperm(M, generator=g)[:K]

    n_train = max(1, int(0.8 * K))
    train_idx = perm[:n_train]
    test_idx  = perm[n_train:]

    ds_train = NPZRaysSubset(ds_full, train_idx)
    ds_test  = NPZRaysSubset(ds_full, test_idx) if len(test_idx) > 0 else NPZRaysSubset(ds_full, train_idx)

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        alpha=args.alpha, beta=args.beta, device=args.device,
        use_supervised=not args.no_supervised and (ds_full.a_gt is not None),
        loss_kind=args.loss, huber_delta=args.huber_delta,
        mask_nonzero=args.mask_nonzero, seed=args.seed,
        lambda_l1=args.lambda_l1, lambda_bg=args.lambda_bg, bg_eps=args.bg_eps,
        lambda_off=args.lambda_off,
    )

    device = torch.device(cfg.device)
    loader_train = DataLoader(ds_train, batch_size=cfg.batch_size,
                              shuffle=True, drop_last=False)
    loader_test  = DataLoader(ds_test, batch_size=cfg.batch_size,
                              shuffle=False, drop_last=False)

    model = MLPActivityNet(N=ds_full.N, hidden=args.hidden,
                           depth=args.depth, dropout=args.dropout,
                           use_I=args.use_I, use_xy=args.use_xy).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f"Training MLPActivityNet (Train/Test-Split) on {args.data} | "
          f"M_full={ds_full.M}, K_used={K} (train={ds_train.M}, test={ds_test.M}) | "
          f"N={ds_full.N} bins | device={device}")

    for epoch in range(1, cfg.epochs + 1):
        stats_train = train_one_epoch(model, loader_train, opt, cfg)
        stats_test  = evaluate_on_loader(model, loader_test, cfg)

        print(
            f"Epoch {epoch:02d} | "
            f"train_total={stats_train['loss']:.3e} | "
            f"train_meas={stats_train['loss_I']:.3e} | "
            f"train_sup={stats_train['loss_sup']:.3e} | "
            f"train_rmse_I_ap={stats_train['rmse_I_ap']:.3e} | "
            f"train_rmse_I_pa={stats_train['rmse_I_pa']:.3e} | "
            f"train_mae_I_ap={stats_train['mae_I_ap']:.3e} | "
            f"train_mae_I_pa={stats_train['mae_I_pa']:.3e} | "
            f"train_logrmse_ap={stats_train['logrmse_ap']:.3e} | "
            f"train_logrmse_pa={stats_train['logrmse_pa']:.3e} | "
            f"train_rmse_a={stats_train['rmse_a']:.3e} | "
            f"test_rmse_I_ap={stats_test['rmse_I_ap']:.3e} | "
            f"test_rmse_I_pa={stats_test['rmse_I_pa']:.3e} | "
            f"test_mae_I_ap={stats_test['mae_I_ap']:.3e} | "
            f"test_mae_I_pa={stats_test['mae_I_pa']:.3e} | "
            f"test_logrmse_ap={stats_test['logrmse_ap']:.3e} | "
            f"test_logrmse_pa={stats_test['logrmse_pa']:.3e} | "
            f"test_rmse_a={stats_test['rmse_a']:.3e} | "
            f"spars={stats_train['spars']:.3f} | "
            f"corr(a,mu)={stats_train['corr_am']:.3f}"
        )

    # -------------------------------------------------
    # Nach dem Training: finale Train/Test-Auswertung
    # -------------------------------------------------
    final_train_stats = evaluate_on_loader(model, loader_train, cfg)
    final_test_stats  = evaluate_on_loader(model, loader_test, cfg)

    def fmt(x: float) -> str:
        return f"{x:.3e}"

    print("\n======================")
    print("Zusammenfassung (letztes Modell)")
    print("======================")
    header = (
        "Datensatz | RMSE(I_AP) | RMSE(I_PA) | MAE(I_AP) | MAE(I_PA) | "
        "LogRMSE(I_AP) | LogRMSE(I_PA) | RMSE(a) | Sparsity | corr(a,µ)"
    )
    print(header)
    print("-" * len(header))

    def print_row(name: str, s: dict):
        print(
            f"{name:8s} | "
            f"{fmt(s['rmse_I_ap'])} | "
            f"{fmt(s['rmse_I_pa'])} | "
            f"{fmt(s['mae_I_ap'])} | "
            f"{fmt(s['mae_I_pa'])} | "
            f"{fmt(s['logrmse_ap'])} | "
            f"{fmt(s['logrmse_pa'])} | "
            f"{fmt(s['rmse_a'])} | "
            f"{s['spars']:.3f} | "
            f"{s['corr_am']:.3f}"
        )

    print_row("Train", final_train_stats)
    print_row("Test",  final_test_stats)

    # ---------------- Checkpoint speichern ----------------
    ckpt_path = os.path.join(out_dir, "mlp_raynet_baseline.pt")
    torch.save(
        {"state_dict": model.state_dict(),
         "N": ds_full.N,
         "config": cfg.__dict__},
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")

    # ---------------- Visuals (Train & Test) ----------------
    import matplotlib.pyplot as plt

    def _norm(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().float().cpu()
        mn, mx = float(x.min()), float(x.max())
        if mx - mn < 1e-12:
            return torch.full_like(x, 0.2)
        return (x - mn) / (mx - mn)


    model.eval()

    # ---- Einzelstrahl-Profil (Train) ----
    with torch.no_grad():
        I_train = ds_train.I_pairs
        nz = (I_train[:, 0] > 0) | (I_train[:, 1] > 0)
        idx_train = int(torch.argmax((I_train[:, 0] + I_train[:, 1]) * nz.float()).item()) if nz.any() else 0

        mu = ds_train.mu[idx_train].unsqueeze(0).to(device)
        T_ap = ds_train.T_ap[idx_train].unsqueeze(0).to(device)
        T_pa = ds_train.T_pa[idx_train].unsqueeze(0).to(device)
        I_ap = ds_train.I_pairs[idx_train, 0].unsqueeze(0).to(device)
        I_pa = ds_train.I_pairs[idx_train, 1].unsqueeze(0).to(device)
        ds_t = torch.tensor(ds_train.ds, dtype=torch.float32, device=device).view(1)
        xy = ds_train.xy_norm[idx_train].unsqueeze(0).to(device) if ds_train.xy_norm is not None else None

        a_pred, _, _ = model(mu, T_ap, T_pa, I_ap, I_pa, ds_t, xy=xy)
        a_true = (
            ds_train.a_gt[idx_train].to(device).unsqueeze(0)
            if ds_train.a_gt is not None else None
        )

        x = torch.arange(ds_train.N)

        plt.figure(figsize=(10, 5))
        plt.plot(x.numpy(), _norm(mu.squeeze(0)).numpy(),
                 color="black", linestyle=":", label="mu", linewidth=2)
        plt.plot(x.numpy(), _norm(a_pred.squeeze(0)).numpy(), color="red", label="a_pred", linestyle="-", linewidth=2)
        if a_true is not None:
            plt.plot(x.numpy(), _norm(a_true.squeeze(0)).numpy(), color="green", label="a_true", linestyle="-", linewidth=2)
        plt.title(f"Ray-Profil TRAIN (Index {idx_train}) — MLP-Baseline")
        plt.xlabel("Bin")
        plt.ylabel("norm.")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ray_profile_train_mlp.png"), dpi=160)
        plt.close()
        print("Ray-Profil Train gespeichert: ray_profile_train_mlp.png")

    # ---- Mehrstrahl-Panel (Train/Test) ----
    def visualize_multiple_rays(model, ds, cfg,
                                num=10, save_prefix="ray_profiles_mlp"):
        model.eval()
        I_sum = ds.I_pairs[:, 0] + ds.I_pairs[:, 1]
        vals, idx_sorted = torch.sort(I_sum)  # von dunkel nach hell
        m = idx_sorted.numel()

        num = min(num, m)
        positions = torch.linspace(0, m - 1, steps=num).long()
        top_idx = idx_sorted[positions].tolist()

        plt.figure(figsize=(14, num * 2))
        for i, idx in enumerate(top_idx):
            mu = ds.mu[idx].unsqueeze(0).to(cfg.device)
            T_ap = ds.T_ap[idx].unsqueeze(0).to(cfg.device)
            T_pa = ds.T_pa[idx].unsqueeze(0).to(cfg.device)
            I_ap = ds.I_pairs[idx, 0].unsqueeze(0).to(cfg.device)
            I_pa = ds.I_pairs[idx, 1].unsqueeze(0).to(cfg.device)
            ds_val = torch.tensor(ds.ds, dtype=torch.float32,
                                  device=cfg.device).view(1)
            xy = ds.xy_norm[idx].unsqueeze(0).to(cfg.device) if ds.xy_norm is not None else None

            a_pred, _, _ = model(mu, T_ap, T_pa, I_ap, I_pa, ds_val, xy=xy)
            a_true = (
                ds.a_gt[idx].unsqueeze(0).to(cfg.device)
                if ds.a_gt is not None else None
            )
            x = torch.arange(ds.N).cpu().numpy()

            mu_np = mu.squeeze().detach().cpu().numpy()
            a_pred_np = a_pred.squeeze().detach().cpu().numpy()
            if a_true is not None:
                a_true_np = a_true.squeeze().detach().cpu().numpy()

            mu_np = (mu_np - mu_np.min()) / (mu_np.max() - mu_np.min() + 1e-8)
            a_pred_np = (a_pred_np - a_pred_np.min()) / (a_pred_np.max() - a_pred_np.min() + 1e-8)
            if a_true is not None:
                a_true_np = (a_true_np - a_true_np.min()) / (a_true_np.max() - a_true_np.min() + 1e-8)

            ax = plt.subplot(num, 1, i + 1)
            ax.plot(x, mu_np, label="mu", color="black", linestyle=":", lw=1)
            ax.plot(x, a_pred_np, label="a_pred", color="red", linestyle="-")
            if a_true is not None:
                ax.plot(x, a_true_np, label="a_true", color="green", linestyle="-")
            ax.set_title(f"Ray {idx} — MLP-Baseline")
            ax.set_ylabel("norm.")
            if i == num - 1:
                ax.set_xlabel("Bin")
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        out = f"{save_prefix}.png"
        plt.savefig(os.path.join(out_dir, out), dpi=160)
        plt.close()
        print(f"Mehrstrahl-Profil gespeichert: {out}")

    # Train-Batchplot
    visualize_multiple_rays(model, ds_train, cfg, num=10, save_prefix="ray_profiles_train_mlp")
    # Test-Batchplot
    visualize_multiple_rays(model, ds_test,  cfg, num=10, save_prefix="ray_profiles_test_mlp")

    # ---- Heatmap I vs I_hat (Train) ----
    def heatmap_I_compare(ds, model, cfg, max_rays=256,
                          save="I_compare_train_mlp.png"):
        model.eval()
        with torch.no_grad():
            m = min(ds.M, max_rays)
            idx = torch.arange(m)
            mu = ds.mu[idx].to(cfg.device)
            T_ap = ds.T_ap[idx].to(cfg.device)
            T_pa = ds.T_pa[idx].to(cfg.device)
            I_ap = ds.I_pairs[idx, 0].to(cfg.device)
            I_pa = ds.I_pairs[idx, 1].to(cfg.device)
            ds_t = torch.full((m,), ds.ds,
                              dtype=torch.float32,
                              device=cfg.device)
            xy = ds.xy_norm[idx].to(cfg.device) if ds.xy_norm is not None else None

            a_pred, I_ap_hat, I_pa_hat = model(
                mu, T_ap, T_pa, I_ap, I_pa, ds_t, xy=xy
            )

            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            im0 = axs[0, 0].imshow(a_pred.detach().cpu().numpy(),
                                   aspect="auto")
            axs[0, 0].set_title("a_pred (MLP)")
            fig.colorbar(im0, ax=axs[0, 0])

            if ds.a_gt is not None:
                im1 = axs[0, 1].imshow(ds.a_gt[idx].cpu().numpy(),
                                       aspect="auto")
                axs[0, 1].set_title("a_true")
                fig.colorbar(im1, ax=axs[0, 1])

            I_meas = torch.stack([I_ap, I_pa], dim=1).cpu().numpy()
            im2 = axs[1, 0].imshow(I_meas, aspect="auto")
            axs[1, 0].set_title("I (AP/PA)")
            fig.colorbar(im2, ax=axs[1, 0])

            I_hat = torch.stack([I_ap_hat, I_pa_hat], dim=1).cpu().numpy()
            im3 = axs[1, 1].imshow(I_hat, aspect="auto")
            axs[1, 1].set_title("I_hat (MLP)")
            fig.colorbar(im3, ax=axs[1, 1])

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, save), dpi=160)
            plt.close()
            print(f"I-Heatmaps gespeichert: {save}")

    heatmap_I_compare(ds_train, model, cfg, save="I_compare_train_mlp.png")


if __name__ == "__main__":
    main()
