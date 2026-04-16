#!/usr/bin/env python3
"""
raynet_test.py — Segment-RayNet mit explizitem Train/Test-Split auf K Strahlen

Beispielaufruf:

python raynet_test.py \
  --data /home/mnguest12/projects/thesis/RayNet/phantom_06/out/rays_train.npz \
  --device cuda \
  --epochs 40 \
  --batch_size 32 \
  --lr 1e-3 \
  --loss loghuber --huber_delta 100 \
  --alpha 1.0 --beta 0.3 \
  --edge_tau 0.15 --min_seg 3 --K_max 8 \
  --dropout 0.1 \
  --lambda_l1 0 \
  --lambda_bg 0 --bg_eps 5e-4 \
  --lambda_off 1e-2 \
  --lambda_nb 0 \
  --lambda_corr 0.0 \
  --meas_quantile 0.999 \
  --num_rays 3000 \
  --num_fourier 4 \
  --mask_nonzero
"""
from __future__ import annotations
import argparse, math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------- Dataset ----------------
class NPZRays(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)

        # Ray-Features aus dem .npz (vom preprocessing.py gebaut)
        self.mu      = torch.from_numpy(data["mu_seq"]).float()      # [M,N]
        self.T_ap    = torch.from_numpy(data["T_ap_seq"]).float()    # [M,N]
        self.T_pa    = torch.from_numpy(data["T_pa_seq"]).float()    # [M,N]
        self.I_pairs = torch.from_numpy(data["I_pairs"]).float()     # [M,2]
        self.a_gt = torch.from_numpy(data["a_seq"]).float() \
            if "a_seq" in data.files else None
        self.mask_npz = torch.from_numpy(data["mask_meas"]).float() \
            if "mask_meas" in data.files else None
        self.ds = float(data["ds"]) if "ds" in data.files else 1.0

        # Anzahl Rays (M) und Samples pro Ray (N)
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
        # Basis-Output pro Ray
        d = {
            "mu":       self.mu[idx],          # [N]
            "T_ap":     self.T_ap[idx],        # [N]
            "T_pa":     self.T_pa[idx],        # [N]
            "I_ap":     self.I_pairs[idx, 0],  # Skalar
            "I_pa":     self.I_pairs[idx, 1],  # Skalar
            "ds":       torch.tensor(self.ds, dtype=torch.float32),
            "mask_def": self.mask_default[idx],# 0/1: aktiver Ray?
        }
        if self.xy_norm is not None:
            d["xy"] = self.xy_norm[idx]        
        if self.a_gt is not None:
            d["a_gt"] = self.a_gt[idx]
        if self.mask_npz is not None:
            d["mask_npz"] = self.mask_npz[idx]
        return d


class NPZRaysSubset(Dataset):
    """Subset-Variante von NPZRays für Train/Test-Splits.

    Schneidet alle Tensoren einmalig auf die gegebenen Indizes zu,
    damit Visualisierungen (I_pairs, a_gt, ...) weiter funktionieren.

    Erklärung: 
        - torch.utils.data.Subset liefert nur Indizes, aber die Daten
        bleiben im Original-Datenset
        - hier wird stattdessen ein "echtes" Sub-Dataset gebaut: alle
        Tensoren werden einmal zugeschnitten und gespeichert
    """
    def __init__(self, base: NPZRays, indices):
        idx = torch.as_tensor(indices, dtype=torch.long)
        # nehme aus dem Original-Dataset base nur die gewünschten Zeilen
        self.mu      = base.mu[idx]
        self.T_ap    = base.T_ap[idx]
        self.T_pa    = base.T_pa[idx]
        self.I_pairs = base.I_pairs[idx]
        self.ds      = base.ds
        self.xy_norm = base.xy_norm[idx] if getattr(base, "xy_norm", None) is not None else None
        self.a_gt     = base.a_gt[idx] if base.a_gt is not None else None
        self.mask_npz = base.mask_npz[idx] if base.mask_npz is not None else None
        self.M, self.N = self.mu.shape

        # Neue Default-Maske: Strahlen, bei denen I_ap oder I_pa > 0 ist
        # (hier kleiner als NPZRays: kein eps, sondern "größer null")
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


# ---------------- Helpers ----------------
@torch.no_grad()
# einfache Normalisierung
def zscore_1d(x: torch.Tensor) -> torch.Tensor:
    m = x.mean()                    # Mittelwert
    s = x.std()                     # Standardabweichung
    return (x - m) / (s + 1e-6)     # z-Normierung, kleiner Offset gegen Div/0

def fourier_encode_xy(xy: torch.Tensor, num_freqs: int = 4) -> torch.Tensor:
    """
    Fourier-Positional-Encoding für normierte Strahlkoordinaten.
    xy: [B,2] mit Werten in [-1,1]
    Rückgabe: [B, 4*num_freqs] (sin/cos für x und y über num_freqs Frequenzen)
    """
    if num_freqs <= 0:
        return torch.empty(xy.shape[0], 0, device=xy.device, dtype=xy.dtype)
    B = xy.shape[0]
    # Frequenzen: 1,2,4,... * pi --> Deckung versch. Skalen
    freqs = 2.0 ** torch.arange(num_freqs, device=xy.device, dtype=xy.dtype) * math.pi  # [F]
    freqs = freqs.view(1, num_freqs, 1)          # [1,F,1]
    xy = xy.view(B, 1, 2)                        # [B,1,2]
    ang = xy * freqs                             # [B,F,2]
    sin = torch.sin(ang)                         # [B,F,2]
    cos = torch.cos(ang)                         # [B,F,2]
    enc = torch.cat([sin, cos], dim=-1)          # [B,F,4]
    return enc.view(B, -1)                       # [B, 4*num_freqs]

@torch.no_grad()
def segment_from_mu(mu: torch.Tensor, edge_tau: float, min_seg: int, K_max: int) -> Tuple[torch.Tensor, int]:
    """Segmentmasken aus µ-Kanten."""
    N = mu.numel()
    # µ z-normalisieren → Kanten unabhängig von Absolutwerten
    mu_n = (mu - mu.mean()) / (mu.std() + 1e-6)
    # Kantenstärke als Betrag der 1D-Differenz
    e = (mu_n[1:] - mu_n[:-1]).abs()  # [N-1]
    # Kante, wenn Differenz > edge_tau
    cuts = e > edge_tau               # bool

    # Indexliste der Segmentgrenzen (inkl. Anfang und Ende)
    idx = [0]
    if cuts.any():
        # Positionen der Kanten → idx ==> 0: Start, +1: linke Grenze vom neuen Segment, N = Ende
        pos = torch.nonzero(cuts, as_tuple=False).squeeze(-1) + 1
        idx += pos.tolist()
    idx.append(N)

    # Greedy rechts, dann Reparatur
    merged = [idx[0]]
    for j in range(1, len(idx)):
        if (idx[j] - merged[-1]) >= min_seg:
            merged.append(idx[j])
    if merged[-1] != N:
        merged.append(N)

    segs = [[merged[k], merged[k + 1]] for k in range(len(merged) - 1) if merged[k + 1] > merged[k]]
    if not segs:
        segs = [[0, N]]

    # Reparatur: mergen, bis keine kurzen Segmente übrig
    while True:
        lengths = torch.tensor([b - a for a, b in segs], device=mu.device)
        short = torch.nonzero(lengths < min_seg, as_tuple=False).flatten()
        if short.numel() == 0:
            break
        i = int(short[0])
        if len(segs) == 1:                          # Nur ein Segment → alles [0,N]
            segs = [[0, N]]
            break
        if i == 0:                                  # Am Rand: mit Nachbar rechts/links mergen
            segs[0][1] = segs[1][1]
            del segs[1]
        elif i == len(segs) - 1:
            segs[-2][1] = segs[-1][1]
            del segs[-1]
        else:                                       # In der Mitte: mit längerem Nachbarsegment mergen
            Lm = segs[i - 1][1] - segs[i - 1][0]
            Rm = segs[i + 1][1] - segs[i + 1][0]
            if Lm >= Rm:
                segs[i - 1][1] = segs[i][1]
                del segs[i]
            else:
                segs[i][1] = segs[i + 1][1]
                del segs[i + 1]

    # Maximal K_max Segmente erlauben → kleinste Segmente iterativ mergen
    while len(segs) > K_max:
        lengths = torch.tensor([b - a for a, b in segs], device=mu.device)
        i = int(torch.argmin(lengths).item())
        if i == 0:
            segs[0][1] = segs[1][1]
            del segs[1]
        elif i == len(segs) - 1:
            segs[-2][1] = segs[-1][1]
            del segs[-1]
        else:
            Lm = segs[i - 1][1] - segs[i - 1][0]
            Rm = segs[i + 1][1] - segs[i + 1][0]
            if Lm >= Rm:
                segs[i - 1][1] = segs[i][1]
                del segs[i]
            else:
                segs[i][1] = segs[i + 1][1]
                del segs[i + 1]

    # Segmentmasken S[k,n] = 1, wenn n in Segment k liegt
    K = len(segs)
    S = torch.zeros((K, N), dtype=torch.float32, device=mu.device)
    for k, (a, b) in enumerate(segs):
        S[k, a:b] = 1.0
    return S, K

# ---------------- Physics aggregator ----------------
class PhysicsAggregator(nn.Module):
    # Berechnet die vorhergesagten Messungen I_AP und I_PA (aus Aktivitäten, Transmissionen und Schrittweite)
    def forward(self, a: torch.Tensor, T_ap: torch.Tensor, T_pa: torch.Tensor, ds: torch.Tensor | float):
        if not torch.is_tensor(ds):
            ds = torch.tensor(ds, dtype=a.dtype, device=a.device)
        ds = ds.reshape(-1, 1)  # [B,1]
        T_ap = torch.nan_to_num(T_ap)
        T_pa = torch.nan_to_num(T_pa)
        # Physikalisches Modell: Summe a[n] * T[n] * ds  (über n = Ray-Samples)
        I_ap_hat = torch.sum((a * T_ap) * ds, dim=1)
        I_pa_hat = torch.sum((a * T_pa) * ds, dim=1)
        return torch.nan_to_num(I_ap_hat), torch.nan_to_num(I_pa_hat)

# ---------------- Segment-RayNet ----------------      Encoder --> FiLM --> Segmentierung --> Segment-Pooling --> a_k --> e_eff --> Physik
class SegmentRayNet(nn.Module):
    """Vorhersage stückweise-konstanter Aktivität je µ-Segment."""
    def __init__(self, in_ch: int = 3, hidden: int = 64,                                # Input-Kanäle: mu, T_ap, T_pa; hidden = Anz. Feature-Channels H
                 kernel_size: int = 5, dropout: float = 0.0,                            
                 num_fourier: int = 0):
        super().__init__()
        pad = kernel_size // 2
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size, padding=pad), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size, padding=pad), nn.GELU(),
            nn.Dropout(p=dropout),
        )                                                                               # Encoder produziert h mit Shape [B,H,N]
        # FiLM (Feature-wise Linear Modulation) nimmt globale Infos (I_ap, I_pa, Fourier-Encoding von xy) --> berechnet gamma & beta, die dann h modulieren
        self.num_fourier = num_fourier
        film_in_dim = 2 + (4 * num_fourier if num_fourier > 0 else 0)
        self.film = nn.Sequential(
            nn.Linear(film_in_dim, 32), nn.GELU(), nn.Linear(32, 2 * hidden)
        )
        self.seg_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))       # aus Segment-Feature (H-dim) wird eine Aktivität pro Segment (Skalar) berechnet
        self.softplus = nn.Softplus()                                                                   # a_k ≥ 0 (keine negative Aktivität)
        self.log_gain = nn.Parameter(torch.tensor(math.log(1.0)))                                       # start ~1
        self.b_global = nn.Parameter(torch.tensor(0.0))                                                 # stark L2-regularisieren
        self.disable_film = False
        self.disable_gain = False
        self.phys = PhysicsAggregator()

    @staticmethod
    def _robust_unit(x: torch.Tensor) -> torch.Tensor:                                  # normiert skalare Größen wie I_ap und I_pa --> stabileres Training
        x_pos = x[x > 0]
        if x_pos.numel() > 16:
            s = torch.quantile(x_pos.detach(), 0.99).clamp(min=1e-6)
        else:
            s = x.detach().abs().max().clamp(min=1e-6)
        return x / s

    def forward(self, mu, T_ap, T_pa, I_ap, I_pa, ds,                                   # mu, T_ap, T_pa: [B,N] --> stapeln --> [B,3,N]
                edge_tau: float, min_seg: int, K_max: int,
                xy: Optional[torch.Tensor] = None):
        B, N = mu.shape[0], mu.shape[1]
        x = torch.stack([mu, T_ap, T_pa], dim=1)                                        # [B,3,N]
        h = self.enc(x)                                                                 # [B,H,N]

        # FiLM (optional) – jetzt mit optionalen Fourier-Koordinaten
        if not self.disable_film:
            I_ap_n = self._robust_unit(I_ap)                                            # [B]
            I_pa_n = self._robust_unit(I_pa)                                            # [B]
            film_in = [torch.stack([I_ap_n, I_pa_n], dim=1)]                            # [B,2]

            # Wenn num_fourier>0: immer 4F Zusatzeinträge liefern
            if self.num_fourier > 0:
                if xy is not None:
                    enc_xy = fourier_encode_xy(xy, num_freqs=self.num_fourier)          # [B,4F]
                else:
                    # Fallback: Dummy-Features, damit Dimension stimmt
                    enc_xy = I_ap.new_zeros(B, 4 * self.num_fourier)
                film_in.append(enc_xy)
            film_in = torch.cat(film_in, dim=1)                                         # [B,2+4F]
            ab = self.film(film_in)                                                     # [B,2H]
            gamma, beta = ab.chunk(2, dim=1)                                            # teile Tensor in zwei Teile entlang der Feature-Achse (wenn ab[B,128] in 64 Werte gamma und 64 beta)
            gamma = torch.tanh(gamma)                                                   # tanh: begrenzt beta und gamma auf [-1,1] --> keine extremen Skalierungen
            beta = torch.tanh(beta)
            h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)                            # [B,H,N] Feature-Tensor, der aus den Eingangssequenzen extrahiert wird

        # Segmentierung & Segment-Pooling (pro Ray)
        a_hat_list = []
        Ks = []
        for b in range(B):
            S_b, K_eff = segment_from_mu(mu[b], edge_tau, min_seg, K_max)               # S_b: [K,N], Binärmasken; K_eff: tatsächliche Segmentzahl
            Ks.append(K_eff)
            h_b = h[b]                                                                  # [H,N]
            denom = S_b.sum(dim=1, keepdim=True).clamp(min=1.0)                         # [K,1] Anzahl Punkte pro Segment (Länge des Segments)
            seg_mean = (h_b @ S_b.t()) / denom.t()                                      # [H,K] Mittelwert der Features pro Segment
            seg_mean = seg_mean.transpose(0, 1)                                         # [K,H] Transpose --> pro Segment ein H-dimensionsloser Feature-Vektor
            a_k = self.softplus(self.seg_head(seg_mean)).squeeze(-1)                    # [K] MLP von H --> 1 & softplus: a_k ≥ 0
            a_hat_b = (a_k.unsqueeze(1) * S_b).sum(dim=0)                               # [N] Summe über k --> an jeder Position n steht a_k-Wert seines Segments
            a_hat_list.append(a_hat_b)

        a_hat = torch.stack(a_hat_list, dim=0)                                          # [B,N] alle Rays stapeln
        if self.disable_gain:                                                           # globaler Gain: korrigiert globale Fehlskalierung (z.B. falsche ds, Phantom-Scaling, ...)    
            gain = a_hat.new_tensor(1.0)
            b_global = a_hat.new_tensor(0.0)
        else:
            gain = F.softplus(self.log_gain) + 1e-6
            b_global = self.b_global

        a_eff = a_hat * gain + b_global                                                 # finale effektive Aktivität, die in die Physik eingeht
        I_ap_hat, I_pa_hat = self.phys(a_eff, T_ap, T_pa, ds)
        return a_eff, I_ap_hat, I_pa_hat, gain, Ks                                      # a_eff [B,N]: vorhergesagte Aktivitätsprofile, I_ap_hat / I_pa_hat [B]: vorhergesagte Messungen, gain: Skalar, Ks: Liste der Segmentzahlen pro Ray

# ---------------- Losses & Regularizers ----------------
def meas_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "mse",              # Messfehler zwischen Vorhersage und Messung
              mask: Optional[torch.Tensor] = None, huber_delta: float = 100.0) -> torch.Tensor:
    # Optional: erst in den Log-Raum gehen
    if kind in ("logmae", "loghuber"):                  # Huber-Loss im Log-Raum --> kombiniert L1- und L2-Verhalten (huber_delta als Schwelle)
        pred = torch.log1p(torch.clamp(pred, min=0.0))
        target = torch.log1p(torch.clamp(target, min=0.0))
    # Je nach 'kind' unterschiedliche Fehlerfunktion
    if kind in ("mae", "logmae"):
        err = torch.abs(pred - target)                  # L1 / MAE  (absoluter Fehler: robust gegen Ausreißer)
    elif kind == "mse":
        err = (pred - target) ** 2                      # L2 / MSE  (quadratischer Fehler: empfindlich für Ausreißer)
    elif kind == "loghuber":
        err = F.huber_loss(pred, target, delta=huber_delta, reduction="none")
    else:
        raise ValueError(f"Unknown loss kind: {kind}")
    # ohne Maske: einfacher Mittelwert über alle Einträge
    if mask is None:
        return err.mean()
    # mit Maske (jeder Messpunkt bekommt ein Gewicht w[i]): gewichteter Mittelwert
    w = mask.view(-1)
    err = err.view(-1)
    return (err * w).sum() / (w.sum() + 1e-8)

@torch.no_grad()
def pearson_corr_per_ray(a: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:           
    """Klassische Pearson-Korrelation zwischen a und µ pro Ray, dann Mittelwert über Batch; a,mu: [B,N]."""
    a = a.float()
    mu = mu.float()
    a_m = a.mean(dim=1, keepdim=True)
    mu_m = mu.mean(dim=1, keepdim=True)
    a_c = a - a_m                                                                       # zentriere pro Ray
    mu_c = mu - mu_m
    num = (a_c * mu_c).sum(dim=1)                                                       # Numerator = Kovarianz-Summe
    den = torch.sqrt((a_c.pow(2).sum(dim=1) + 1e-12) * (mu_c.pow(2).sum(dim=1) + 1e-12))# Denominator = Produkt der Standardabweichungen
    r = num / (den + 1e-12)                                                             # r E [-1,1]: nahe +1 --> a und µ stark positiv korreliert (nur zur Auswertung, nicht für Training)
    return r.mean()

def pearson_corr_per_ray_grad(a: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Differenzierbare Pearson-Korrelation pro Ray (Mittel über Batch); a,mu: [B,N].
    Gleiche Formel etc. wie oben, aber mit Gradienten --> geeignet, in einem Loss zu erscheinen..."""
    a_m = a.mean(dim=1, keepdim=True)
    mu_m = mu.mean(dim=1, keepdim=True)
    a_c = a - a_m
    mu_c = mu - mu_m
    num = (a_c * mu_c).sum(dim=1)
    den = torch.sqrt((a_c.pow(2).sum(dim=1) + 1e-12) * (mu_c.pow(2).sum(dim=1) + 1e-12))
    r = num / (den + 1e-12)
    return r.mean()

def neighbor_coherence(a_batch: torch.Tensor, mu_batch: torch.Tensor,
                       sigma_mu: float = 0.2, radius: int = 2) -> torch.Tensor:
    """Einfacher Nachbarverlust über Batch-Indizes (lokales Fenster)."""
    B, N = a_batch.shape
    loss = a_batch.new_tensor(0.0)
    cnt = 0
    for i in range(B):
        i0 = max(0, i - radius)
        i1 = min(B, i + radius + 1)
        for j in range(i0, i1):
            if j == i:
                continue
            w = torch.exp(-((mu_batch[i] - mu_batch[j]) ** 2).mean() /
                          (2 * sigma_mu ** 2 + 1e-12))
            loss = loss + w * (a_batch[i] - a_batch[j]).abs().mean()
            cnt += 1
    return loss / max(1, cnt)

# ---------------- Training ----------------
@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 16
    lr: float = 1e-3
    alpha: float = 1.0              # Gewicht Mess-Loss (I_ap/I_pa)
    beta: float = 1e-2              # Gewicht supervised (a_eff vs. a_gt)
    gamma: float = 0.0              # TV (unused)
    device: str = "cpu"
    use_supervised: bool = True
    loss_kind: str = "logmae"
    huber_delta: float = 100.0
    mask_nonzero: bool = True
    seed: int = 0
    edge_tau: float = 0.15
    min_seg: int = 3
    K_max: int = 8
    # Regularisierung
    lambda_l1: float = 0.0          # L1 auf a_eff (Sparsity)
    lambda_bg: float = 0.0          # bestraft zu hohe Aktivität im Hintergrund
    bg_eps: float = 5e-4            # Schwelle "Hintergrund"
    lambda_off: float = 0.0         # Bestrafung für globalen Offset b_global
    lambda_nb: float = 0.0          # Gewicht für neighbor_coherence
    sigma_mu: float = 0.2
    nb_radius: int = 2
    # Ablationen
    no_film: bool = False
    no_gain: bool = False
    K_one: bool = False             # wenn True: nur 1 Segment pro Ray (kein Segment-Modell)
    dropout: float = 0.0
    # Adaptives Beta
    beta_adapt: bool = False
    corr_tau: float = 0.5
    corr_kappa: float = 0.5
    # Anti-Korrelation + Mess-Skalierung
    lambda_corr: float = 0.0
    meas_quantile: float = 0.999
    # Anzahl Fourier-Frequenzen für (x,y)-Koordinaten (0 = keine Koordinatenfeatures)
    num_fourier: int = 0


def train_one_epoch(model: SegmentRayNet, loader: DataLoader,
                    opt: torch.optim.Optimizer, cfg: TrainConfig) -> Dict[str, float]:
    model.train()
    stats = {   
        "loss": 0.0, "loss_I": 0.0, "loss_sup": 0.0,
        "loss_l1": 0.0, "loss_bg": 0.0, "loss_off": 0.0, "loss_nb": 0.0,
        "rmse_I_ap": 0.0, "rmse_I_pa": 0.0, "rmse_a": 0.0,
        "mae_I_ap": 0.0, "mae_I_pa": 0.0,
        "logrmse_ap": 0.0, "logrmse_pa": 0.0,
        "spars": 0.0, "corr_am": 0.0,
    }
    n = 0                                                                                       # Anz. der Rays über die gemittelt wird

    for batch in loader:                                                                        # Batch aus DataLoader holen
        mu = batch["mu"].to(cfg.device)                                                         # Inputs...
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

        opt.zero_grad(set_to_none=True)                                             # alle Gradienten der Parameter vor jeder neuen Batch auf None setzen
        a_eff, I_ap_hat, I_pa_hat, gain, Ks = model(
            mu, T_ap, T_pa, I_ap, I_pa, ds,
            cfg.edge_tau, cfg.min_seg, 1 if cfg.K_one else cfg.K_max,
            xy=xy,
        )

        # Mess-Loss (Data-Fit)
        I_ap_hat_n, I_pa_hat_n = I_ap_hat, I_pa_hat
        I_ap_n, I_pa_n         = I_ap, I_pa

        l_ap = meas_loss(I_ap_hat_n, I_ap_n, kind=cfg.loss_kind,
                         mask=mask, huber_delta=cfg.huber_delta)
        l_pa = meas_loss(I_pa_hat_n, I_pa_n, kind=cfg.loss_kind,
                         mask=mask, huber_delta=cfg.huber_delta)
        loss_I = l_ap + l_pa

        # Supervised-Loss auf a (wenn a_gt vorhanden)
        if cfg.use_supervised and a_gt is not None:
            w_bg = (a_gt < cfg.bg_eps).float()                                      # bg_eps: Schwellenwert, ab dem als Hintergrund betrachtet wird
            w_pos = 1.0 - w_bg
            w = 3.0 * w_bg + 1.0 * w_pos
            loss_sup = ((a_eff - a_gt) ** 2 * w).mean()
        else:
            loss_sup = a_eff.new_tensor(0.0)

        # Zusatzregularisierungen
        loss_l1 = cfg.lambda_l1 * a_eff.mean()                                      # einfacher Sparsity-Anteil (L1-artig, weil mean(a_eff) klein gehalten wird)
        
        loss_bg = a_eff.new_tensor(0.0)                                             # bestraft zu hohe Aktivitäten im Hintergrund
        if a_gt is not None and cfg.lambda_bg > 0:
            w_bg = (a_gt < cfg.bg_eps).float()
            loss_bg = cfg.lambda_bg * (F.relu(a_eff - cfg.bg_eps) * w_bg).mean()

        loss_off = a_eff.new_tensor(0.0)                                            # quadratische Strafe auf globalen Bias b_global --> soll nahe 0 bleiben
        if hasattr(model, "b_global") and cfg.lambda_off > 0:
            loss_off = cfg.lambda_off * model.b_global.pow(2)

        loss_nb = a_eff.new_tensor(0.0)                                             # neighbor_coherence: koppel a-Effekte zwischen Rays mit ähnlichen mu-Profilen
        if cfg.lambda_nb > 0:
            loss_nb = cfg.lambda_nb * neighbor_coherence(
                a_eff, mu, sigma_mu=cfg.sigma_mu, radius=cfg.nb_radius
            )

        # Anti-Korrelation a vs. µ (differenzierbar)
        loss_corr = a_eff.new_tensor(0.0)
        if cfg.lambda_corr > 0.0:
            corr_for_loss = pearson_corr_per_ray_grad(a_eff, mu)
            loss_corr = cfg.lambda_corr * (corr_for_loss ** 2)

        loss = (                                                                    # Gesamtverlust: gewichtete Summe aller Losses...
            cfg.alpha * loss_I +
            cfg.beta * loss_sup +
            loss_l1 + loss_bg + loss_off + loss_nb + loss_corr
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)                  # begrenzt Gradienten-Norm --> stabilisiert Training
        opt.step()

        # Metriken
        with torch.no_grad():
            bsz = mu.shape[0]

            # RMSE (physische Skala)
            rmse_I_ap = torch.sqrt(F.mse_loss(I_ap_hat, I_ap))
            rmse_I_pa = torch.sqrt(F.mse_loss(I_pa_hat, I_pa))

            rmse_I_ap_item = rmse_I_ap.item()
            rmse_I_pa_item = rmse_I_pa.item()

            # MAE und Log-RMSE (Lograum robust gegen winzige Werte)
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

            stats["loss"] += loss.item() * bsz                                      # Berechnung Durchschnitt über alle Rays der Epoche (bsz=batchsize)
            stats["loss_I"] += loss_I.item() * bsz
            stats["loss_sup"] += loss_sup.item() * bsz
            stats["loss_l1"] += (loss_l1.item() if torch.is_tensor(loss_l1) else loss_l1) * bsz
            stats["loss_bg"] += (loss_bg.item() if torch.is_tensor(loss_bg) else loss_bg) * bsz
            stats["loss_off"] += (loss_off.item() if torch.is_tensor(loss_off) else loss_off) * bsz
            stats["loss_nb"] += (loss_nb.item() if torch.is_tensor(loss_nb) else loss_nb) * bsz

            stats["rmse_I_ap"] += rmse_I_ap_item * bsz
            stats["rmse_I_pa"] += rmse_I_pa_item * bsz
            stats["mae_I_ap"] += mae_I_ap * bsz
            stats["mae_I_pa"] += mae_I_pa * bsz
            stats["logrmse_ap"] += logrmse_ap * bsz
            stats["logrmse_pa"] += logrmse_pa * bsz
            stats["rmse_a"] += rmse_a * bsz
            stats["spars"] += spars * bsz
            stats["corr_am"] += corr_am * bsz
            n += bsz

        # Adaptives Beta (optional): erhöhe β, wenn corr(a,µ) > τ
        if cfg.beta_adapt and (n > 0):
            mean_corr = stats["corr_am"] / n
            if mean_corr > cfg.corr_tau:
                cfg.beta *= (1.0 + cfg.corr_kappa * (mean_corr - cfg.corr_tau))
                cfg.beta = float(min(cfg.beta, 1.0))  # Deckel

    for k in stats.keys():
        stats[k] = stats[k] / max(1, n)
    return stats


@torch.no_grad()
def evaluate_on_loader(model: SegmentRayNet, loader: DataLoader, cfg: TrainConfig) -> Dict[str, float]:
    """wie train_one_epoch, aber kein Training, nur Auswertung (wird für Val-Loader nach jeder Epoche und Test-Loader am Ende genutzt)
    man will sehen, wie gut das Modell auf ungesehenen Daten generalisiert --> dafür darf während der Auswertung nicht geupdated werden ==> reine Metrikberechnung"""
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

        a_eff, I_ap_hat, I_pa_hat, _, _ = model(
            mu, T_ap, T_pa, I_ap, I_pa, ds,
            cfg.edge_tau, cfg.min_seg, 1 if cfg.K_one else cfg.K_max,
            xy=xy,
        )

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
        stats["mae_I_ap"] += mae_I_ap * bsz
        stats["mae_I_pa"] += mae_I_pa * bsz
        stats["logrmse_ap"] += logrmse_ap * bsz
        stats["logrmse_pa"] += logrmse_pa * bsz
        stats["rmse_a"] += rmse_a * bsz
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
    ap.add_argument("--edge_tau", type=float, default=0.15)
    ap.add_argument("--min_seg", type=int, default=3)
    ap.add_argument("--K_max", type=int, default=8)
    # Regs
    ap.add_argument("--lambda_l1", type=float, default=0.0)
    ap.add_argument("--lambda_bg", type=float, default=0.0)
    ap.add_argument("--bg_eps", type=float, default=5e-4)
    ap.add_argument("--lambda_off", type=float, default=0.0)
    ap.add_argument("--lambda_nb", type=float, default=0.0)
    ap.add_argument("--sigma_mu", type=float, default=0.2)
    ap.add_argument("--nb_radius", type=int, default=2)
    # Anti-Korrelation + Mess-Skalierung
    ap.add_argument("--lambda_corr", type=float, default=0.0)
    ap.add_argument("--meas_quantile", type=float, default=0.999)
    # Ablations
    ap.add_argument("--no_supervised", action="store_true")
    ap.add_argument("--no_film", action="store_true")
    ap.add_argument("--no_gain", action="store_true")
    ap.add_argument("--K_one", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)
    # Adaptive beta
    ap.add_argument("--beta_adapt", action="store_true")
    ap.add_argument("--corr_tau", type=float, default=0.5)
    ap.add_argument("--corr_kappa", type=float, default=0.5)
    # Koordinaten-Fourier-Encoding
    ap.add_argument("--num_fourier", type=int, default=0,
                    help="Anzahl Fourier-Frequenzen für (x,y)-Strahlkoordinaten (0 = deaktiviert)")
    # Neue Option: Anzahl Strahlen K
    ap.add_argument("--num_rays", type=int, default=1000,
                    help="Anzahl Strahlen K, die für Train/Test verwendet werden (wird auf M gekappt)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:                                       # seed setzen --> reproduzierbare Zufälle (z.B. bei Ray-Auswahl)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Voller Datensatz
    ds_full = NPZRays(args.data)

    # Ausgabepfad (gleicher Ordner wie NPZ-Datei)
    import os
    out_dir = os.path.dirname(args.data)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output directory set to: {out_dir}")

    M = ds_full.M                                                   # Anzahl Rays im vollen Dataset
    K = min(args.num_rays, M)                                       # so viele Rays wollen wir max. nutzen

    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed if args.seed is not None else 0)
    perm = torch.randperm(M, generator=g)[:K]

    n_train = max(1, int(0.8 * K))                                  # 80 % Trainingsdaten, Rest für Test
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
        edge_tau=args.edge_tau, min_seg=args.min_seg,
        K_max=(1 if args.K_one else args.K_max),
        lambda_l1=args.lambda_l1, lambda_bg=args.lambda_bg, bg_eps=args.bg_eps,
        lambda_off=args.lambda_off, lambda_nb=args.lambda_nb,
        sigma_mu=args.sigma_mu, nb_radius=args.nb_radius,
        no_film=args.no_film, no_gain=args.no_gain,
        K_one=args.K_one, dropout=args.dropout,
        beta_adapt=args.beta_adapt, corr_tau=args.corr_tau,
        corr_kappa=args.corr_kappa,
        lambda_corr=args.lambda_corr, meas_quantile=args.meas_quantile,
        num_fourier=args.num_fourier,
    )

    device = torch.device(cfg.device)
    loader_train = DataLoader(ds_train, batch_size=cfg.batch_size,
                              shuffle=True, drop_last=False)
    loader_test  = DataLoader(ds_test, batch_size=cfg.batch_size,
                              shuffle=False, drop_last=False)

    model = SegmentRayNet(in_ch=3, hidden=64,
                          kernel_size=5, dropout=cfg.dropout,
                          num_fourier=cfg.num_fourier).to(device)
    model.disable_film = cfg.no_film
    model.disable_gain = cfg.no_gain
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f"Training Segment-RayNet (Train/Test-Split) on {args.data} | "
          f"M_full={ds_full.M}, K_used={K} (train={ds_train.M}, test={ds_test.M}) | "
          f"N={ds_full.N} bins | device={device}")

    for epoch in range(1, cfg.epochs + 1):
        stats_train = train_one_epoch(model, loader_train, opt, cfg)            # für jede Epoche trainieren auf Train-Set und evaluieren auf Test-Set
        stats_test  = evaluate_on_loader(model, loader_test, cfg)

        # Ein Batch für Gain/Ks und Logging
        with torch.no_grad():
            batch = next(iter(loader_train))
            I_ap = batch["I_ap"].to(device)
            I_pa = batch["I_pa"].to(device)
            mu   = batch["mu"].to(device)
            T_ap = batch["T_ap"].to(device)
            T_pa = batch["T_pa"].to(device)
            ds_t = batch["ds"].to(device)
            _, I_ap_hat, I_pa_hat, gain_val, Ks = model(
                mu, T_ap, T_pa, I_ap, I_pa, ds_t,
                cfg.edge_tau, cfg.min_seg, cfg.K_max
            )


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
            f"test_rmse_a={stats_test['rmse_a']:.3e} | "
            f"test_mae_I_ap={stats_test['mae_I_ap']:.3e} | "
            f"test_mae_I_pa={stats_test['mae_I_pa']:.3e} | "
            f"test_logrmse_ap={stats_test['logrmse_ap']:.3e} | "
            f"test_logrmse_pa={stats_test['logrmse_pa']:.3e} | "
            f"spars={stats_train['spars']:.3f} | "
            f"corr(a,mu)={stats_train['corr_am']:.3f} | "
            f"gain={gain_val.item():.3e} | "
            f"meanK={np.mean(Ks):.2f} | beta={cfg.beta:.2e}"
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
    torch.save(
        {"state_dict": model.state_dict(),                          # Gewichte des Modells
         "N": ds_full.N,                                            # Anzahl Bins pro Ray
         "config": cfg.__dict__},                                   # Konfiguration
        "segment_raynet_test.pt",
    )
    print("Saved checkpoint: segment_raynet_test.pt")

    # ---------------- Visuals (Train & Test) ----------------
    import matplotlib.pyplot as plt

    def _norm(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().float().cpu()
        mn, mx = float(x.min()), float(x.max())
        if mx - mn < 1e-12:
            return torch.full_like(x, 0)
        return (x - mn) / (mx - mn)

    model.eval()

    # ---- Einzelstrahl-Profil (Train) ----
    with torch.no_grad():
        I_train = ds_train.I_pairs
        nz = (I_train[:, 0] > 0) | (I_train[:, 1] > 0)
        idx_train = int(torch.argmax((I_train[:, 0] + I_train[:, 1]) * nz.float()).item()) if nz.any() else 0   # sucht Index des Rays mit größter Summe I_ap+I_pa und lässt Modell damit laufen

        mu = ds_train.mu[idx_train].unsqueeze(0).to(device)
        T_ap = ds_train.T_ap[idx_train].unsqueeze(0).to(device)
        T_pa = ds_train.T_pa[idx_train].unsqueeze(0).to(device)
        I_ap = ds_train.I_pairs[idx_train, 0].unsqueeze(0).to(device)
        I_pa = ds_train.I_pairs[idx_train, 1].unsqueeze(0).to(device)
        ds_t = torch.tensor(ds_train.ds, dtype=torch.float32, device=device).view(1)

        a_pred, _, _, _, Ks = model(
            mu, T_ap, T_pa, I_ap, I_pa, ds_t,
            cfg.edge_tau, cfg.min_seg, cfg.K_max
        )
        a_true = (
            ds_train.a_gt[idx_train].to(device).unsqueeze(0)
            if ds_train.a_gt is not None else None
        )

        x = torch.arange(ds_train.N)

        S_vis, K_vis = segment_from_mu(mu.squeeze(0),
                                       cfg.edge_tau,
                                       cfg.min_seg,
                                       cfg.K_max)
        cuts = torch.nonzero(
            (S_vis.sum(0)[:-1] > 0) &
            (S_vis.sum(0)[1:] > 0) &
            (S_vis.argmax(0)[:-1] != S_vis.argmax(0)[1:]),
            as_tuple=False
        ).squeeze(-1)

        plt.figure(figsize=(10, 5))
        plt.plot(x.numpy(), _norm(mu.squeeze(0)).numpy(),
                 label="mu", linewidth=2, color="black", linestyle=":")
        plt.plot(x.numpy(), _norm(a_pred.squeeze(0)).numpy(),
                 label="a_pred", color="red", linestyle="-", linewidth=2)
        if a_true is not None:
            plt.plot(x.numpy(), _norm(a_true.squeeze(0)).numpy(),
                     label="a_true", color="green", linestyle="-", linewidth=2)
        for c in cuts.tolist():
            plt.axvline(c + 0.5, linestyle=":", alpha=0.5)
        plt.title(f"Ray-Profil TRAIN (Index {idx_train}) — Segmente≈{Ks[0]}")
        plt.xlabel("Bin")
        plt.ylabel("norm.")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ray_profile_train.png"), dpi=160)
        plt.close()
        print("Ray-Profil Train gespeichert: ray_profile_train.png")

    # ---- Mehrstrahl-Panel (Train/Test) ----
    def visualize_multiple_rays(model, ds, cfg,
                                num=10, save_prefix="ray_profiles"):                # soritert alle Rays nach Gesamtintensität --> wählt z.B. 10 Rays gleichmäßig aus dem Bereich (von schwach bis stark)
        model.eval()
        I_sum = ds.I_pairs[:, 0] + ds.I_pairs[:, 1]
        vals, idx_sorted = torch.sort(I_sum)  # von dunkel nach hell
        m = idx_sorted.numel()

        # z.B. num Rays gleichmäßig über den Bereich verteilen
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

            a_pred, _, _, _, Ks = model(
                mu, T_ap, T_pa, I_ap, I_pa, ds_val,
                cfg.edge_tau, cfg.min_seg, cfg.K_max
            )
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
            ax.plot(x, mu_np, label="mu", lw=1, color="black", linestyle=":")
            ax.plot(x, a_pred_np, color="red", linestyle="-", label="a_pred")
            if a_true is not None:
                ax.plot(x, a_true_np, color="green", linestyle="-", label="a_true")
            ax.set_title(f"Ray {idx} — Seg≈{Ks[0]}")
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
    visualize_multiple_rays(model, ds_train, cfg, num=10, save_prefix="ray_profiles_train")
    # Test-Batchplot
    visualize_multiple_rays(model, ds_test,  cfg, num=10, save_prefix="ray_profiles_test")

    # ---- Heatmap I vs I_hat (Train) ----
    def heatmap_I_compare(ds, model, cfg, max_rays=256,
                          save="I_compare_train.png"):
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
            a_pred, I_ap_hat, I_pa_hat, _, _ = model(
                mu, T_ap, T_pa, I_ap, I_pa, ds_t,
                cfg.edge_tau, cfg.min_seg, cfg.K_max
            )

            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            im0 = axs[0, 0].imshow(a_pred.detach().cpu().numpy(),
                                   aspect="auto")
            axs[0, 0].set_title("a_pred")
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
            axs[1, 1].set_title("I_hat")
            fig.colorbar(im3, ax=axs[1, 1])

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, save), dpi=160)
            plt.close()
            print(f"I-Heatmaps gespeichert: {save}")

    heatmap_I_compare(ds_train, model, cfg, save="I_compare_train.png")


if __name__ == "__main__":
    main()
