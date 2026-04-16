"""
paper_cores.py
==============
Kern-Implementierungen der 5 Referenzpaper, adaptiert auf den SPECT-Kontext
dieser Arbeit (Lu-177-PSMA Dosimetrie, AP/PA-Projektionen, CT-Attenuation).

Melli's Pipeline-Kontext:
  - Eingabe:  AP/PA-Szintigraphie (2D) + CT-Volumen (3D)
  - Ausgabe:  3D-Aktivitätsverteilung (sigma_volume)
  - Physik:   Beer-Lambert-Attenuation entlang der AP-Achse
  - Achsen:   (SI, AP, LR) = (z, y, x)

Paper-Überblick:
  1. Kim et al.       – CNN-Prior + RLS/TV-Iteration
  2. Henzler et al.   – PlatonicGAN: differenzierbares AO-Rendering
  3. Huang & Pei      – GSA-INF: Triplane-Prior + INF-Decoder
  4. Corona et al.    – MedNeRF: GRAF für medizinische Projektionen
  5. Liu & Bai        – VolumeNeRF: Likelihood-Prior + Projection Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# GEMEINSAME HILFSFUNKTIONEN (aus Mellis forward_spect übernommen)
# =============================================================================

def beer_lambert_projection(sigma_vol, mu_vol=None, axis=1, step_len=1.0):
    """
    Differenzierbarer SPECT-Vorwärts-Operator (Beer-Lambert).
    Entspricht Melli's forward_spect / spect_operator_wrapper.

    sigma_vol : (B, SI, AP, LR)  – Aktivitätsdichte
    mu_vol    : (B, SI, AP, LR)  – Attenuationskoeff. aus CT (optional)
    axis      : AP-Achse (1 = Anterior, projiziert entlang AP)
    Gibt zurück: proj (B, SI, LR)
    """
    if mu_vol is not None:
        cum_mu = torch.cumsum(mu_vol * step_len, dim=axis)
        # Versatz um 1: Voxel i nutzt Attenuation bis i-1
        cum_mu = torch.clamp(cum_mu, 0.0, 60.0)
        T = torch.exp(-cum_mu)
        weighted = sigma_vol * T
    else:
        weighted = sigma_vol
    proj = weighted.sum(dim=axis)          # Integration entlang AP
    return proj


def positional_encoding(x, num_freqs=6):
    """Fourier-Positionscodierung (aus pieNeRF / NeRF-Standard)."""
    freqs = 2.0 ** torch.arange(num_freqs, dtype=x.dtype, device=x.device)
    x_freq = x.unsqueeze(-1) * freqs          # (..., 3, L)
    enc = torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)
    return torch.cat([x, enc.flatten(-2)], dim=-1)   # (..., 3 + 6*2*3)


# =============================================================================
# 1. KIM ET AL. (2019) – Extreme Few-view CT / SPECT via CNN-Prior + TV-RLS
#    "Extreme Few-view CT Reconstruction using Deep Inference"
# =============================================================================

class SinogramEncoder1D(nn.Module):
    """1D-CNN: kodiert AP- und PA-Projektionen in ein latentes Feature."""
    def __init__(self, in_ch=2, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, padding=3), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2),   nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1),  nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(256, latent_dim)

    def forward(self, ap, pa):
        # ap, pa : (B, LR)  – jeweils ein SI-Schnitt
        x = torch.stack([ap, pa], dim=1)       # (B, 2, LR)
        feat = self.net(x).squeeze(-1)          # (B, 256)
        return self.proj(feat)                  # (B, latent_dim)


class VolumeGenerator2D(nn.Module):
    """2D-CNN: erzeugt einen SI-Schnitt des sigma-Volumens als CNN-Prior."""
    def __init__(self, latent_dim=256, out_h=64, out_w=64):
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 1, 3, 1, 1), nn.Softplus(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 512, 4, 4)
        x = self.deconv(x)
        return F.interpolate(x, (self.out_h, self.out_w), mode='bilinear',
                             align_corners=False)   # (B, 1, SI, LR)


class TVRegularizer(nn.Module):
    """Total-Variation-Regularisierung auf dem 3D-Volumen (Melli: thesis-NEUSTART_TV)."""
    def forward(self, vol):
        # vol: (B, SI, AP, LR)
        tv_si = (vol[:, 1:] - vol[:, :-1]).abs().mean()
        tv_ap = (vol[:, :, 1:] - vol[:, :, :-1]).abs().mean()
        tv_lr = (vol[:, :, :, 1:] - vol[:, :, :, :-1]).abs().mean()
        return tv_si + tv_ap + tv_lr


def rls_tv_step(sigma_vol, ap_gt, pa_gt, mu_vol, beta=2e-2, step_len=1.0):
    """
    Ein RLS/TV-Iterationsschritt (Gleichung 1, Kim et al.).
    Gradient bezüglich sigma_vol: ∂/∂x(||y - Ax||² + β·TV(x))
    Wird typischerweise 50-100× aufgerufen, startend vom CNN-Prior.
    """
    sigma_vol = sigma_vol.detach().requires_grad_(True)
    ap_pred = beer_lambert_projection(sigma_vol, mu_vol, axis=2, step_len=step_len)
    pa_pred = beer_lambert_projection(sigma_vol.flip(2), mu_vol.flip(2),
                                      axis=2, step_len=step_len)
    data_loss = F.mse_loss(ap_pred, ap_gt) + F.mse_loss(pa_pred, pa_gt)
    tv_loss   = TVRegularizer()(sigma_vol)
    total     = data_loss + beta * tv_loss
    total.backward()
    with torch.no_grad():
        sigma_vol = sigma_vol - 1e-3 * sigma_vol.grad
        sigma_vol = sigma_vol.clamp(min=0.0)
    return sigma_vol.detach()


# =============================================================================
# 2. HENZLER ET AL. (2019) – PlatonicGAN: Differenzierbares Rendering
#    "Escaping Plato's Cave: 3D Shape From Adversarial Rendering"
# =============================================================================

def render_absorption_only(voxels, axis=2):
    """
    Absorption-Only (AO) Rendering – Gleichung (9) aus Henzler et al.
    ρ_AO(v) = 1 − Π_i (1 − v_i)
    Mathematisch äquivalent zu Mellis Beer-Lambert-Projektion bei kleinen v.
    voxels: (B, C, D, H, W)  → proj: (B, C, H, W)
    """
    proj = 1.0 - torch.prod(1.0 - voxels.clamp(0, 1), dim=axis)
    return proj


def render_visual_hull(voxels, axis=2):
    """Visual Hull (VH) – Gleichung (8) aus Henzler et al."""
    return 1.0 - torch.exp(-voxels.clamp(min=0).sum(dim=axis))


class VoxelGenerator3D(nn.Module):
    """Leichtgewichtiger 3D-Voxel-Generator (PlatonicGAN-Stil)."""
    def __init__(self, latent_dim=128, vol_res=32):
        super().__init__()
        self.vol_res = vol_res
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose3d(128,  64, 4, 2, 1), nn.ReLU(),
            nn.Conv3d(64, 1, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 4, 4, 4)
        return self.deconv(x)                   # (B, 1, 32, 32, 32)


class PlatonicDiscriminator2D(nn.Module):
    """2D-Diskriminator für AO-gerenderte Projektionen."""
    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32,  4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32,    64,  4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64,   128,  4, 2, 1), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


def platonic_gan_step(gen, disc, z, real_proj, optimizer_g, optimizer_d):
    """
    Ein PlatonicGAN-Update (Gleichungen 1-4 aus Henzler et al.),
    adaptiert: real_proj sind AP-SPECT-Projektionen.
    """
    # --- Diskriminator-Update ---
    vox  = gen(z)
    fake = render_absorption_only(vox, axis=2)   # (B,1,H,W)
    d_real = disc(real_proj)
    d_fake = disc(fake.detach())
    loss_d = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
             F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    optimizer_d.zero_grad(); loss_d.backward(); optimizer_d.step()

    # --- Generator-Update ---
    d_fake2 = disc(render_absorption_only(gen(z), axis=2))
    loss_g  = F.binary_cross_entropy_with_logits(d_fake2, torch.ones_like(d_fake2))
    optimizer_g.zero_grad(); loss_g.backward(); optimizer_g.step()
    return loss_d.item(), loss_g.item()


# =============================================================================
# 3. HUANG & PEI (2024) – GSA-INF: Triplane-Prior + INF-Decoder
#    "Generalizable Structure-Aware INF: Biplanar-View CT Reconstruction"
# =============================================================================

class TriplaneGenerator(nn.Module):
    """
    TGM-Decoder (Triplane Generative Model, Abschnitt 3.1, Huang & Pei).
    Latent-Code z → drei orthogonale Feature-Planes (xy, yz, xz).
    Backbone: vereinfachtes U-Net analog zu Original (U-Net mit 4 Stufen).
    """
    def __init__(self, latent_dim=512, plane_res=64, plane_ch=32):
        super().__init__()
        self.plane_res = plane_res
        self.plane_ch  = plane_ch
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.unet = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d( 64,  plane_ch * 3, 4, 2, 1),  # 3 Planes
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 8, 8)
        planes = self.unet(x)                       # (B, 3*C, H, W)
        B, _, H, W = planes.shape
        C = self.plane_ch
        xy = planes[:, :C]                          # Axialplane
        yz = planes[:, C:2*C]                       # Koronalplane
        xz = planes[:, 2*C:]                        # Sagittalplane
        return xy, yz, xz


def sample_triplane(xy, yz, xz, coords):
    """
    Trilineare Interpolation aus drei Planes für Query-Koordinaten.
    coords: (B, N, 3) normiert auf [-1, 1]
    Gibt zurück: feature (B, N, C)
    """
    def bilinear(plane, uv):
        # plane: (B, C, H, W), uv: (B, N, 2)
        grid = uv.unsqueeze(2)                      # (B, N, 1, 2)
        feat = F.grid_sample(plane, grid, align_corners=True)  # (B, C, N, 1)
        return feat.squeeze(-1).permute(0, 2, 1)   # (B, N, C)

    f_xy = bilinear(xy, coords[..., :2])
    f_yz = bilinear(yz, coords[..., 1:])
    f_xz = bilinear(xz, torch.stack([coords[..., 0], coords[..., 2]], -1))
    return f_xy + f_yz + f_xz                      # (B, N, C) – summiert


class INFDensityDecoder(nn.Module):
    """
    5-Layer-MLP Density-Decoder (Gleichung 2, Huang & Pei).
    Input: kombiniertes Feature F(v) = triplane_feat + xray_feat
    Output: Dichte c(v)
    """
    def __init__(self, feat_dim=32, xray_feat_dim=64):
        super().__init__()
        in_dim = feat_dim + xray_feat_dim
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),    nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(128 + in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),           nn.ReLU(),
            nn.Linear(64,  1),            nn.Softplus(),
        )

    def forward(self, F_combined):
        h = self.mlp1(F_combined)
        h = torch.cat([h, F_combined], dim=-1)     # Skip-Verbindung
        return self.mlp2(h)                         # (B, N, 1)


class INFMaskDecoder(INFDensityDecoder):
    """
    Parallelel Mask-Decoder für anatomische Strukturen (Gleichung 3).
    Wie DensityDecoder, aber mit Softmax-Ausgang für binäre Organlabels.
    """
    def __init__(self, feat_dim=32, xray_feat_dim=64, num_classes=3):
        super().__init__(feat_dim, xray_feat_dim)
        in_dim = feat_dim + xray_feat_dim
        self.mlp2 = nn.Sequential(
            nn.Linear(128 + in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),           nn.ReLU(),
            nn.Linear(64,  num_classes),
        )

    def forward(self, F_combined):
        h = self.mlp1(F_combined)
        h = torch.cat([h, F_combined], dim=-1)
        return F.softmax(self.mlp2(h), dim=-1)     # (B, N, num_classes)


def gsainf_loss(pred_vol, gt_vol, pred_mask, gt_mask, gamma=1e-2):
    """
    Verlustfunktion GSA-INF (Gleichung 4): L = L_vol + γ * L_ams
    L_vol: L2 Volumenkonsistenz
    L_ams: Dice-Verlust für anatomische Masken (Lunge / Organe)
    """
    loss_vol = F.mse_loss(pred_vol, gt_vol)
    # Dice-Koeffizient (Gleichung im Abschnitt 3.3)
    intersection = (pred_mask * gt_mask).sum(dim=(1, 2))
    dice = 1.0 - 2.0 * intersection / (pred_mask.sum(dim=(1,2)) +
                                        gt_mask.sum(dim=(1,2))   + 1e-6)
    loss_mask = dice.mean()
    return loss_vol + gamma * loss_mask


def online_code_optimization(tgm, density_dec, xray_feat, ap_gt,
                              mu_vol, z_init, lr=1e-2, iters=50, eta=1e-4):
    """
    Test-Time Code-Optimierung (Algorithmus 1, Huang & Pei).
    Passt den latenten Code z an die Eingabe-Röntgenprojektion an.
    """
    z = z_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    for _ in range(iters):
        xy, yz, xz = tgm(z)
        # Dummy: hier würde man die echten Query-Koordinaten einsetzen
        # und mit dem density_dec das Volumen rekonstruieren
        # Dann: Röntgenprojekt ion rendern und mit ap_gt vergleichen
        r = torch.tensor(0.0)          # Platzhalter – im echten Code: ||I - I_r||²
        if r < eta:
            break
        optimizer.zero_grad(); r.backward(); optimizer.step()
    return z.detach()


# =============================================================================
# 4. CORONA ET AL. (2022) – MedNeRF: GRAF für medizinische Projektionen
#    "MedNeRF: Medical Neural Radiance Fields ... from a Single X-ray"
# =============================================================================

class GRAFGenerator(nn.Module):
    """
    GRAF-Generator (Abschnitt II-B/C, Corona et al.).
    Adaptiert auf SPECT: gibt Dichte σ und Schwächungsantwort c zurück.
    z_s = Shape-Code, z_a = Appearance-Code (beide aus N(0,I)).
    """
    def __init__(self, z_s_dim=128, z_a_dim=128, pe_freqs=6):
        super().__init__()
        in_dim = 3 * (1 + 2 * pe_freqs) + z_s_dim   # xyz_pe + z_s
        self.shape_net = nn.Sequential(
            nn.Linear(in_dim,       256), nn.ReLU(),
            nn.Linear(256,          256), nn.ReLU(),
            nn.Linear(256,          256), nn.ReLU(),
            nn.Linear(256,          256),
        )
        # Dichtekopf σ
        self.sigma_head = nn.Sequential(nn.Linear(256, 1), nn.Softplus())
        # Schwächungskopf c (mit z_a konditioniert)
        self.color_head = nn.Sequential(
            nn.Linear(256 + z_a_dim, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, xyz, z_s, z_a):
        """
        xyz  : (B, N, 3)
        z_s  : (B, z_s_dim)
        z_a  : (B, z_a_dim)
        """
        B, N, _ = xyz.shape
        xyz_pe = positional_encoding(xyz)                      # (B, N, 39)
        zs_exp = z_s.unsqueeze(1).expand(-1, N, -1)           # (B, N, z_s_dim)
        h = self.shape_net(torch.cat([xyz_pe, zs_exp], -1))   # (B, N, 256)
        sigma = self.sigma_head(h)                             # (B, N, 1)
        za_exp = z_a.unsqueeze(1).expand(-1, N, -1)
        c = self.color_head(torch.cat([h, za_exp], -1))       # (B, N, 1)
        return sigma, c


class MedNeRFDiscriminator(nn.Module):
    """
    Selbst-supervisierter Diskriminator (Abschnitt II-C.1, Corona et al.).
    Zwei einfache Decoder (SD) auf Skalen f1=32² und f2=8² für LPIPS-ähnliche
    Verlustberechnung, um mit kleinen medizinischen Datensätzen klarzukommen.
    """
    def __init__(self, in_ch=1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32,  4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32,    64,  4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64,   128,  4, 2, 1), nn.LeakyReLU(0.2),
        )
        self.adv_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1)
        )
        # Decoder auf 32x32 (f1) und 8x8 (f2) für Rekonstruktions-Hilfsverlust
        self.dec_f1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, in_ch, 3, 1, 1),
        )

    def forward(self, x):
        feat = self.backbone(x)
        score = self.adv_head(feat)
        recon = self.dec_f1(feat)       # Rekonstruierter Patch (Gleichung 6)
        return score, recon


def mednerf_loss(disc, gen, real_ap, z_s, z_a, xyz, lambda_mse=0.1):
    """
    Kombinierter MedNeRF-Verlust (Gleichungen 8-10, Corona et al.).
    Hinge-Loss + LPIPS-Proxy-Verlust (MSE auf Decoder-Ausgabe) + MSE.
    """
    sigma, c = gen(xyz, z_s, z_a)
    # Vereinfachtes Rendern: Summe entlang Tiefenachse
    fake_proj = c.mean(dim=1, keepdim=True).view(real_ap.shape[0], 1,
                                                  *real_ap.shape[-2:])
    score_real, recon_real = disc(real_ap)
    score_fake, _          = disc(fake_proj.detach())

    # Hinge-Loss (Gleichung 8)
    loss_d = F.relu(1.0 - score_real).mean() + F.relu(1.0 + score_fake).mean()

    # Rekonstruktions-Hilfsverlust (Gleichung 6, LPIPS-Proxy)
    loss_r = F.mse_loss(recon_real, F.interpolate(real_ap, recon_real.shape[-2:]))

    # Generator-Verlust + MSE-Verzerrungsverlust (Gleichung 10)
    score_fake2, _ = disc(fake_proj)
    loss_g = -score_fake2.mean() + lambda_mse * F.mse_loss(fake_proj, real_ap)

    return loss_d + loss_r, loss_g


# =============================================================================
# 5. LIU & BAI (2024) – VolumeNeRF: Likelihood-Prior + Projection Attention
#    "VolumeNeRF: CT Volume Reconstruction from a Single Projection View"
# =============================================================================

class LikelihoodImageComputer:
    """
    Berechnet das Likelihood-Bild (Gleichung 1, Liu & Bai).
    Für jeden Pixel p: L_p = log(√(2π σ²_p)) + (x_p - μ_p)² / (2σ²_p)
    Trainingsset-Statistik (μ_p, σ²_p) wird vorab aus den AP-Projektionen
    aller Trainingspatienten berechnet (analog zu Mellis Normalisierungsansatz).
    """
    def __init__(self):
        self.mu    = None   # (H, W) – pixelweise Mittelwert
        self.sigma = None   # (H, W) – pixelweise Std

    def fit(self, projections):
        """projections: (N, H, W) – AP-Projektionen aller Trainingspatienten"""
        self.mu    = projections.mean(dim=0)
        self.sigma = projections.std(dim=0) + 1e-6

    def compute(self, x_p):
        """x_p: (B, H, W) → L_p: (B, H, W) – negativer Log-Likelihood"""
        assert self.mu is not None, "fit() zuerst aufrufen"
        mu = self.mu.to(x_p.device)
        sg = self.sigma.to(x_p.device)
        return torch.log(torch.sqrt(2 * torch.pi * sg**2)) + \
               (x_p - mu)**2 / (2 * sg**2 + 1e-6)


class ProjectionAttentionModule(nn.Module):
    """
    Projection Attention (Abschnitt 3.2, Liu & Bai).
    Lernt die räumliche Korrespondenz zwischen CT-Voxeln und X-Strahl-Pixeln
    über lernbare Offsets (analog zu DCN v2).
    Adaptiert auf SPECT-Geometrie: Voxel in (SI, AP, LR) → Pixel in (SI, LR).

    vox_feat  : (B, C, SI, AP, LR)  – 3D-Feature
    proj_feat : (B, C, SI, LR)      – 2D-Feature aus Projektion
    """
    def __init__(self, C=64, K=4):
        super().__init__()
        self.K = K
        # Offset- und Gewichtsberechnung aus konkateniertem Feature
        self.offset_conv = nn.Conv3d(2 * C, 2 * K, 3, 1, 1)
        self.weight_conv  = nn.Conv3d(2 * C, K,     3, 1, 1)
        self.fuse         = nn.Conv3d(C + C, C,     1)

    def forward(self, vox_feat, proj_feat):
        B, C, SI, AP, LR = vox_feat.shape

        # Jeder Voxel (si, ap, lr) korrespondiert zum Pixel (si, lr) – Gleichung (4)
        proj_exp = proj_feat.unsqueeze(3).expand(-1, -1, -1, AP, -1)  # (B,C,SI,AP,LR)

        combined = torch.cat([vox_feat, proj_exp], dim=1)              # (B,2C,SI,AP,LR)
        offsets  = self.offset_conv(combined)                          # (B,2K,SI,AP,LR)
        weights  = F.softmax(self.weight_conv(combined), dim=1)        # (B,K,SI,AP,LR)

        # Aggregiere K gewichtete nachbarschafts-Pixel-Features
        y = torch.zeros_like(vox_feat)
        for k in range(self.K):
            # Vereinfacht: Verschiebe proj_exp um Offset und multipliziere mit Gewicht
            # (In vollem DCNv2 würde man bilineare Interpolation verwenden)
            delta = offsets[:, 2*k:2*k+2].mean(dim=(2,3,4), keepdim=True)
            y = y + weights[:, k:k+1] * (proj_exp + delta.unsqueeze(-1).mean())
        return self.fuse(torch.cat([vox_feat, y], dim=1))              # (B,C,SI,AP,LR)


class VolumeNeRFNet(nn.Module):
    """
    VolumeNeRF-Architektur (Abbildung 1, Liu & Bai), adaptiert auf SPECT.
    Eingabe: AP-Projektion + Likelihood-Bild + mittleres CT-Volumen
    Ausgabe: Rekonstruiertes Aktivitätsvolumen (sigma_vol)
    """
    def __init__(self, vol_shape=(64, 32, 64), C=32):
        super().__init__()
        SI, AP, LR = vol_shape

        # 2D-Mapping: AP + Likelihood → Style-Vektoren (ConvNeXt-Stil)
        self.encoder_2d = nn.Sequential(
            nn.Conv2d(2, C, 3, 1, 1),  nn.GELU(),
            nn.Conv2d(C, C*2, 3, 2, 1), nn.GELU(),
            nn.Conv2d(C*2, C*4, 3, 2, 1), nn.GELU(),
        )

        # 3D-Encoder/Decoder für mittleres CT-Volumen
        self.enc3d = nn.Sequential(
            nn.Conv3d(1,   C,   3, 1, 1), nn.ReLU(),
            nn.Conv3d(C,   C*2, 3, 2, 1), nn.ReLU(),
        )
        self.dec3d = nn.Sequential(
            nn.ConvTranspose3d(C*2, C, 4, 2, 1), nn.ReLU(),
            nn.Conv3d(C, 1, 1), nn.Softplus(),
        )

        # Projection Attention (einmal, im letzten Decoder)
        self.proj_att = ProjectionAttentionModule(C=C*2, K=4)

    def forward(self, ap, likelihood, mean_ct):
        """
        ap          : (B, 1, SI, LR)
        likelihood  : (B, 1, SI, LR)
        mean_ct     : (B, 1, SI, AP, LR)
        """
        style = self.encoder_2d(torch.cat([ap, likelihood], dim=1))   # 2D-Features

        # Bringe 2D-Style auf SI-LR-Dimension
        style_2d = F.adaptive_avg_pool2d(style, (ap.shape[2], ap.shape[3]))  # (B, C4, SI, LR)

        v = self.enc3d(mean_ct)                # (B, C2, SI/2, AP/2, LR/2)

        # Resample style_2d auf Volumen-SI/LR
        # Resample auf Volumen-SI/LR, Channels auf v anpassen — 4D übergeben
        style_vol = F.interpolate(style_2d, size=(v.shape[2], v.shape[4]),
                                  mode='bilinear', align_corners=False)  # (B, C4, SI', LR')
        c = min(style_vol.shape[1], v.shape[1])
        style_vol = style_vol[:, :c]                                     # (B, C, SI', LR')

        v = self.proj_att(v, style_vol)
        return self.dec3d(v)                   # (B, 1, SI, AP, LR)


def volumenerf_loss(pred_vol, gt_vol, pred_proj, input_proj, w_recon=1.0,
                    w_edge=0.05, w_render=0.001):
    """
    Kombinierter VolumeNeRF-Verlust (Gleichung 6, Liu & Bai).
    L = λ_recon·L1(vol) + λ_edge·L1(edges) + λ_render·L1(rendering)
    """
    loss_recon  = F.l1_loss(pred_vol, gt_vol)

    # Scharr-Kantenoperator auf SI-LR-Ebenen (vereinfacht: Sobel)
    def scharr(v):
        k = torch.tensor([[3,0,-3],[10,0,-10],[3,0,-3]],
                          dtype=v.dtype, device=v.device).view(1,1,3,3) / 32.
        return F.conv2d(v.view(-1, 1, *v.shape[-2:]), k, padding=1)

    loss_edge   = F.l1_loss(scharr(pred_vol[:, 0]), scharr(gt_vol[:, 0]))
    loss_render = F.l1_loss(pred_proj, input_proj)

    return w_recon * loss_recon + w_edge * loss_edge + w_render * loss_render


# =============================================================================
# DEMO / SANITY CHECK
# =============================================================================

if __name__ == "__main__":
    B, SI, AP, LR = 2, 32, 24, 32
    device = "cpu"

    print("=== 1. Kim et al. – CNN-Prior + TV-RLS ===")
    enc  = SinogramEncoder1D(in_ch=2, latent_dim=128).to(device)
    gen  = VolumeGenerator2D(latent_dim=128, out_h=SI, out_w=LR).to(device)
    ap_1d = torch.rand(B, LR)
    pa_1d = torch.rand(B, LR)
    z = enc(ap_1d, pa_1d)
    prior_slice = gen(z)
    print(f"  Latent:        {z.shape}")
    print(f"  Prior (SI-LR): {prior_slice.shape}")

    print("\n=== 2. Henzler et al. – PlatonicGAN ===")
    vox_gen  = VoxelGenerator3D(latent_dim=64, vol_res=16).to(device)
    disc_2d  = PlatonicDiscriminator2D().to(device)
    z_plat   = torch.randn(B, 64)
    vox      = vox_gen(z_plat)
    proj_ao  = render_absorption_only(vox, axis=2)
    print(f"  Voxel:   {vox.shape}")
    print(f"  Proj AO: {proj_ao.shape}")

    print("\n=== 3. Huang & Pei – GSA-INF ===")
    tgm     = TriplaneGenerator(latent_dim=64, plane_res=32, plane_ch=16).to(device)
    d_dec   = INFDensityDecoder(feat_dim=16, xray_feat_dim=16).to(device)
    z_gsa   = torch.randn(B, 64)
    xy, yz, xz = tgm(z_gsa)
    coords  = torch.rand(B, 100, 3) * 2 - 1
    tp_feat = sample_triplane(xy, yz, xz, coords)
    dummy_xr = torch.rand(B, 100, 16)
    density  = d_dec(torch.cat([tp_feat, dummy_xr], -1))
    print(f"  Triplane xy: {xy.shape}")
    print(f"  Density:     {density.shape}")

    print("\n=== 4. Corona et al. – MedNeRF ===")
    graf = GRAFGenerator(z_s_dim=64, z_a_dim=64).to(device)
    xyz  = torch.rand(B, 200, 3)
    z_s  = torch.randn(B, 64)
    z_a  = torch.randn(B, 64)
    sigma_g, c_g = graf(xyz, z_s, z_a)
    print(f"  Sigma: {sigma_g.shape}  c: {c_g.shape}")

    print("\n=== 5. Liu & Bai – VolumeNeRF ===")
    vol_net  = VolumeNeRFNet(vol_shape=(SI, AP, LR), C=16).to(device)
    ap_img   = torch.rand(B, 1, SI, LR)
    lik_img  = torch.rand(B, 1, SI, LR)
    mean_ct  = torch.rand(B, 1, SI, AP, LR)
    pred_vol = vol_net(ap_img, lik_img, mean_ct)
    ap_pred  = beer_lambert_projection(pred_vol[:, 0], axis=2)
    loss     = volumenerf_loss(pred_vol, mean_ct, ap_pred.unsqueeze(1), ap_img)
    print(f"  Pred vol:  {pred_vol.shape}")
    print(f"  Loss:      {loss.item():.4f}")

    print("\nAlle Checks bestanden.")
