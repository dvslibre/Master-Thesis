"""Helper utilities for the modified NeRF (positional encoding, MLP core, ray helpers)."""

DEBUG_ORIENTATION_CHECKS = False
_ORIENTATION_CHECKS_LOGGED = False

import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

_ORTHO_RAYS_DEBUGGED = False

# Positional Encoding (Fourier Features) - wichtig für NeRF
class Embedder:
    """Baut mehrere Frequenz-Basisfunktionen sin(kx), cos(kx)
    für positionsbezogene Encodings (wie im NeRF Paper)"""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:                        # Originalwerte behalten (x selbst)
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']                 # Frequenzbänder bestimmen (0, ..., 2^max_freq)
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:                                 # sin(freq*x), cos(freq*x)
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        """Alle Embeddings konkatenieren"""
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    """Hilfsfunktion: Erzeugt ein Embedder-Objekt + Lambda-Funktion für den Encoder"""
    if i == -1:
        return nn.Identity(), 3                                 # kein positional encoding
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# NeRF-MLP (hier werden Punkte -> Werte gemappt)
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ Standard-NeRF-MLP mit optionaler Viewdir-Konditionierung"""
        super(NeRF, self).__init__()
        self.D = D                                  # Tiefe: Anz. Punkt-Layer
        self.W = W                                  # Breite: Anz. Neuronen pro Layer
        self.input_ch = input_ch                    # Dim der Positions-Eingabe (nach PosEnc)
        self.input_ch_views = input_ch_views        # Dim der Viewdir-Eingabe (nach PosEnc)    
        self.skips = skips                          # Layer-Indizes mit Skip-Connection    
        self.use_viewdirs = use_viewdirs            # falls True: RGB abh. von Blickrichtung
        
        # MLP für 3D-Punkte (mit opt. Skip-Connections)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        # View-MLP (offizielle NeRF Implementierung: ein Layer)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        # Falls Blickrichtungen genutzt: Feature/Alpha/RGB getrennt
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:               # sonst direkt Gesamt-Output (z.B. [sigma, rgb] oder Emission, ...)
            self.output_linear = nn.Linear(W, output_ch)
        
        # --- Emission-NeRF Init Fix: leichte positive Startwerte ---
        with torch.no_grad():
            self.output_linear.bias.fill_(1e-3)    # kleiner positiver Bias
            self.output_linear.weight.mul_(0.01)   # kleine Gewichtsamplitude (nahe 0)

    def forward(self, x):
        """x: [..., input_ch + input_ch_views]
           = (positional-encodete Positionen || positional-encodete Viewdirs)"""
        # Aufteilung: Punkt- und Viewdir-Anteil
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        relu = partial(F.relu, inplace=True)

        # Punkt-MLP durchlaufen
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = relu(h)
            if i in self.skips:
                # Skip-Connection: ursprüngliche Eingabe wieder anhängen
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # Dichte / Alpha direkt aus Punkt-Features
            alpha = self.alpha_linear(h)
            # Feature-Vektor für weitere Verarbeitung
            feature = self.feature_linear(h)
            # Feature + Viewdir als Input für RGB-MLP
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = relu(h)

            rgb = self.rgb_linear(h)                            # Farbausgabe
            outputs = torch.cat([rgb, alpha], -1)               # Output: [rgb, alpha]
        else:
            outputs = self.output_linear(h)

        return outputs    


# Ray helpers - Funktionen, um aus Kamera-Parametern Rays zu bauen
def get_rays(H, W, focal, c2w):
    """NICHT GENUTZT: Erzeugt für eine pinhole-Kamera alle Rays (Ursprung + Richtung) eines HxW-Bildes"""
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))         # Pixelgitter ni Bildkoordinaten
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)      # Richtungen im Kameraraum (z zeigt nach -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)                       # in Welt-Raum rotieren
    rays_o = c2w[:3,-1].expand(rays_d.shape)                                            # Kameraursprung ins Weltkoordinatensystem (identisch für alle Rays)
    return rays_o, rays_d


def set_debug_orientation_checks(enabled: bool) -> None:
    """Erlaubt zusätzliche Orientierungsausgaben für orthografische Rays."""
    global DEBUG_ORIENTATION_CHECKS, _ORIENTATION_CHECKS_LOGGED
    DEBUG_ORIENTATION_CHECKS = bool(enabled)
    _ORIENTATION_CHECKS_LOGGED = False


def get_rays_ortho(H, W, c2w, size_h, size_w):
    """Erzeugt orthografische Rays: parallele Richtungen, Ursprünge liegen auf einer Ebene 
    mit physikalischer Größe size_w x size_h in Weltkoordinaten.
    Wird für SPECT-AP/PA-Projektionen genutzt"""
    device = c2w.device
    dtype = c2w.dtype

    # Parallele Strahlen entlang der -Z-Achse des Kameraraums
    rays_d = -c2w[:3, 2].view(1, 1, 3).expand(H, W, -1)

    # Erzeuge ein kartesisches Gitter in der Bildebene (Weltmaße = size_w/size_h)
    xs = torch.linspace(-0.5 * size_w, 0.5 * size_w, W, device=device, dtype=dtype)
    ys = torch.linspace(-0.5 * size_h, 0.5 * size_h, H, device=device, dtype=dtype)
    try:
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:  # older torch without indexing kwarg
        grid_y, grid_x = torch.meshgrid(ys, xs)
    # Invertiere die Detektor-x-Achse, damit Bildkoordinaten (links→rechts) mit Weltkoordinaten übereinstimmen.
    grid_x = -grid_x

    # Kamerarotationsachsen: Spalten 0/1 entsprechen Kamera-x/y im Welt-Raum.
    x_axis = c2w[:3, 0].view(1, 1, 3)
    y_axis = c2w[:3, 1].view(1, 1, 3)
    origin = c2w[:3, -1].view(1, 1, 3)

    # Setze die Ursprungsebene der Rays anhand der Rasterkoordinaten zusammen.
    rays_o = grid_x.unsqueeze(-1) * x_axis + grid_y.unsqueeze(-1) * y_axis
    rays_o = rays_o + origin

    global _ORTHO_RAYS_DEBUGGED, _ORIENTATION_CHECKS_LOGGED
    if not _ORTHO_RAYS_DEBUGGED:
        _ORTHO_RAYS_DEBUGGED = True
        xs_range = (float(xs.min().item()), float(xs.max().item()))
        ys_range = (float(ys.min().item()), float(ys.max().item()))
        rays_min = torch.stack([rays_o[..., 0].amin(), rays_o[..., 1].amin(), rays_o[..., 2].amin()])
        rays_max = torch.stack([rays_o[..., 0].amax(), rays_o[..., 1].amax(), rays_o[..., 2].amax()])
        print(
            "[debug][get_rays_ortho] size_h={:.3f}cm size_w={:.3f}cm "
            "xs_range={:.3f}/{:.3f}cm ys_range={:.3f}/{:.3f}cm "
            "rays_o_x={:.3f}/{:.3f}cm rays_o_y={:.3f}/{:.3f}cm rays_o_z={:.3f}/{:.3f}cm".format(
                size_h,
                size_w,
                xs_range[0],
                xs_range[1],
                ys_range[0],
                ys_range[1],
                rays_min[0].item(),
                rays_max[0].item(),
                rays_min[1].item(),
                rays_max[1].item(),
                rays_min[2].item(),
                rays_max[2].item(),
            ),
            flush=True,
        )
    if DEBUG_ORIENTATION_CHECKS and not _ORIENTATION_CHECKS_LOGGED:
        _ORIENTATION_CHECKS_LOGGED = True
        grid_x_range = (float(grid_x.amin().item()), float(grid_x.amax().item()))
        grid_y_range = (float(grid_y.amin().item()), float(grid_y.amax().item()))
        rays_min = torch.stack([rays_o[..., 0].amin(), rays_o[..., 1].amin(), rays_o[..., 2].amin()])
        rays_max = torch.stack([rays_o[..., 0].amax(), rays_o[..., 1].amax(), rays_o[..., 2].amax()])
        print(
            "[orientation-check][get_rays_ortho] size_h={:.3f}cm (image axis0 -> world Y) "
            "size_w={:.3f}cm (image axis1 -> world X); "
            "xs_range(before invert)={:.3f}/{:.3f}cm ys_range={:.3f}/{:.3f}cm; "
            "grid_x_range(after invert)={:.3f}/{:.3f}cm grid_y_range={:.3f}/{:.3f}cm; "
            "world rays_o_x={:.3f}/{:.3f}cm rays_o_y={:.3f}/{:.3f}cm rays_o_z={:.3f}/{:.3f}cm; "
            "grid_x is flipped to keep image left->right aligned with world X.",
            flush=True,
        )

    return rays_o, rays_d
