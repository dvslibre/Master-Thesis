import numpy as np
import torch
from graf.utils import sample_on_sphere, look_at, to_sphere
from graf.transforms import FullRaySampler
from nerf.run_nerf_mod import render, run_network  # conditional render
from functools import partial


def _pose_from_loc(loc_np: np.ndarray, up=np.array([0.0, 1.0, 0.0], dtype=np.float32)) -> torch.Tensor:
    """Erzeugt eine c2w-Pose (3x4) für eine Weltposition `loc` (Kamera blickt zum Ursprung)."""
    R = look_at(loc_np, up=np.array(up, dtype=np.float32))[0]  # 3x3
    RT = np.concatenate([R, loc_np.reshape(3, 1)], axis=1)  # 3x4
    return torch.tensor(RT, dtype=torch.float32)


class Generator(object):
    """
    Wrapper um das NeRF-Rendering.

    In unserer Emissions-Variante:
      - NeRF gibt pro Punkt e(x) (Emission) aus.
      - raw2outputs integriert entlang Rays: I = Sum_i e_i * Δs_i.
      - 'proj_rgb' aus render(...) ist eine Grauwert-Projektion (3 gleiche Kanäle).
      - __call__ gibt flache Projektionskarten zurück.
    """

    def __init__(
        self,
        H,
        W,
        focal,
        radius,
        ray_sampler,
        render_kwargs_train,
        render_kwargs_test,
        parameters,
        named_parameters,
        range_u=(0, 1),
        range_v=(0.01, 0.49),
        chunk=None,
        device="cuda",
        orthographic=False,
    ):
        self.device = device
        self.H = int(H)
        self.W = int(W)
        self.focal = focal
        self.radius = radius
        self.range_u = range_u
        self.range_v = range_v
        self.chunk = chunk
        self.orthographic = orthographic  # wichtig für FullRaySampler
        self.fixed_poses_enabled = False
        self.pose_ap = None
        self.pose_pa = None
        self.ortho_size = None  # (height, width) in Weltkoordinaten für Orthographie
        self._fixed_pose_toggle = 0  # 0=AP, 1=PA

        # Pixel-Gitter
        coords = torch.from_numpy(
            np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing="ij"), -1)
        )
        self.coords = coords.view(-1, 2)

        self.ray_sampler = ray_sampler
        self.val_ray_sampler = FullRaySampler(orthographic=orthographic)

        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test
        self.initial_raw_noise_std = self.render_kwargs_train["raw_noise_std"]

        self._parameters = parameters
        self._named_parameters = named_parameters
        self.module_dict = {"generator": self.render_kwargs_train["network_fn"]}

        for name, module in [("generator_fine", self.render_kwargs_train["network_fine"])]:
            if module is not None:
                self.module_dict[name] = module

        # ggf. Zusatzmodule aufnehmen
        for k, v in self.module_dict.items():
            if k in ["generator", "generator_fine"]:
                continue
            self._parameters += list(v.parameters())
            self._named_parameters += list(v.named_parameters())

        # kompatibel zu nn.Module-API
        self.parameters = lambda: self._parameters
        self.named_parameters = lambda: self._named_parameters

        # use_test_kwargs = True → eval-Rendering
        self.use_test_kwargs = False

        # render-Funktion mit festen H, W, focal, chunk
        self.render = partial(render, H=self.H, W=self.W, focal=self.focal, chunk=self.chunk)

    def __call__(self, z, y=None, rays=None):
        """
        z:    [B, feat_dim] – latenter Code
        rays: optional; wenn None → sample_rays() (zufällige Posen)

        Rückgabe:
          - im Train-Mode: proj_flat [B, H*W]
          - im Eval-Mode: (proj_flat, disp_flat, acc_flat, extras)
        """
        bs = z.shape[0]
        # Alle Features strikt auf das Generator-Device legen, sonst kollidieren CPU/GPU-Tensors.
        z = z.to(self.device, non_blocking=True)

        # Falls keine Rays gegeben sind: standardmäßige GRAF-Pose-Samples
        if rays is None:
            rays = torch.cat([self.sample_rays() for _ in range(bs)], dim=1)

        # Sicherstellen, dass die Rays auf demselben Device liegen wie das Netz.
        rays = rays.to(self.device, non_blocking=True)

        # Render-Settings wählen
        render_kwargs = self.render_kwargs_test if self.use_test_kwargs else self.render_kwargs_train
        render_kwargs = dict(render_kwargs)  # copy

        # variable radius → near/far anpassen
        if isinstance(self.radius, tuple):
            assert (
                self.radius[1] - self.radius[0] <= render_kwargs["near"]
            ), "Your smallest radius lies behind your near plane!"

            # rays: [2, N_rays, 3] → wir nehmen Ursprung (Index 0)
            rays_radius = rays[0].norm(dim=-1)
            shift = (self.radius[1] - rays_radius).view(-1, 1).float()
            render_kwargs["near"] = render_kwargs["near"] - shift
            render_kwargs["far"] = render_kwargs["far"] - shift
            assert (render_kwargs["near"] >= 0).all() and (render_kwargs["far"] >= 0).all(), (
                rays_radius.min(),
                rays_radius.max(),
                shift.min(),
                shift.max(),
            )

        # z als Features an NeRF durchreichen
        render_kwargs["features"] = z

        if render_kwargs.get("use_attenuation") and "ct_context" not in render_kwargs:
            render_kwargs["use_attenuation"] = False

        # Emissions-NeRF rendern
        proj_rgb, disp, acc, extras = self.render(rays=rays, **render_kwargs)
        # proj_rgb: [N_rays, 3]
        # disp:     [N_rays]
        # acc:      [N_rays]

        def rays_to_output(x):
            """
            Formt Ray-Ausgaben zu flachen Batch-Outputs um.
            Bei Emission keine [-1, 1]-Skalierung (wir wollen positive Intensitäten).
            """
            x = x.view(len(x), -1)
            return x

        if self.use_test_kwargs:
            proj_flat = rays_to_output(proj_rgb)
            disp_flat = rays_to_output(disp)
            acc_flat = rays_to_output(acc)
            return proj_flat, disp_flat, acc_flat, extras

        proj_flat = rays_to_output(proj_rgb)
        return proj_flat

    def decrease_nerf_noise(self, it):
        end_it = 5000
        if it < end_it:
            noise_std = self.initial_raw_noise_std - self.initial_raw_noise_std / end_it * it
            self.render_kwargs_train["raw_noise_std"] = noise_std

    def sample_pose(self):
        if self.fixed_poses_enabled and (self.pose_ap is not None) and (self.pose_pa is not None):
            pose = self.pose_ap if self._fixed_pose_toggle == 0 else self.pose_pa
            self._fixed_pose_toggle = 1 - self._fixed_pose_toggle  # toggle AP<->PA
            return pose

        # fallback: zufällige Pose auf Kugel
        loc = sample_on_sphere(self.range_u, self.range_v)
        radius = self.radius
        if isinstance(radius, tuple):
            radius = np.random.uniform(*radius)
        loc = loc * radius
        R = look_at(loc)[0]
        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        RT = torch.Tensor(RT.astype(np.float32))
        return RT

    def sample_rays(self):
        pose = self.sample_pose()
        # Orthographische SPECT-Setups brauchen stets das volle Raster → FullRaySampler nutzen.
        if self.orthographic:
            sampler = self.val_ray_sampler
            focal_or_size = self.ortho_size or (self.H, self.W)
        else:
            sampler = self.val_ray_sampler if self.use_test_kwargs else self.ray_sampler
            focal_or_size = self.focal

        batch_rays, _, _ = sampler(self.H, self.W, focal_or_size, pose)
        return batch_rays

    def to(self, device):
        self.render_kwargs_train["network_fn"].to(device)
        if self.render_kwargs_train["network_fine"] is not None:
            self.render_kwargs_train["network_fine"].to(device)
        self.device = device
        return self

    def train(self):
        self.use_test_kwargs = False
        self.render_kwargs_train["network_fn"].train()
        if self.render_kwargs_train["network_fine"] is not None:
            self.render_kwargs_train["network_fine"].train()

    def eval(self):
        self.use_test_kwargs = True
        self.render_kwargs_train["network_fn"].eval()
        if self.render_kwargs_train["network_fine"] is not None:
            self.render_kwargs_train["network_fine"].eval()


    def set_fixed_ap_pa(self, radius=None, up=(0, 1, 0)):
        """
        Definiert zwei feste Posen:
        - AP: Kamera auf +Z, blickt zum Ursprung
        - PA: Kamera auf -Z, blickt zum Ursprung
        `radius`: nutzt self.radius, wenn None. `up` ist die Welt-„oben“-Richtung.
        """
        if radius is None:
            radius = self.radius if not isinstance(self.radius, tuple) else self.radius[1]

        # AP = +z, PA = -z
        loc_ap = np.array([0.0, 0.0, +float(radius)], dtype=np.float32)
        loc_pa = np.array([0.0, 0.0, -float(radius)], dtype=np.float32)

        self.pose_ap = _pose_from_loc(loc_ap, up=up)
        self.pose_pa = _pose_from_loc(loc_pa, up=up)
        # look_at(..., -z) kehrt die x-Achse um (diag[-1,1,-1]); dadurch wäre PA spiegelverkehrt
        # zu den PA-GT-Projektionen. Durch Rückspiegeln der x-Achse stimmen die Pixelrichtungen wieder.
        self.pose_pa[:, 0] *= -1.0

        self.fixed_poses_enabled = True
        self._fixed_pose_toggle = 0  # starte mit AP

        # Orthographische Größe an den CT-Würfel anpassen: Breite/Höhe = 2*radius
        if self.orthographic:
            size = 2.0 * float(radius)
            self.ortho_size = (size, size)

    def render_from_pose(self, z, pose, ct_context=None):
        """
        Render eine komplette Projektion aus einer festen Kamerapose.

        Rückgabe:
            proj_flat: [bs, H*W]  (synthetische Projektion, flach)
            disp_flat: [bs, H*W]
            acc_flat:  [bs, H*W]
            extras:    dict (alles Weitere aus NeRF-Rendering)
        """
        bs = z.shape[0]
        device = self.device
        # Alle Inputs direkt auf das Generator-Device legen, damit Ray-Sampling/render konsistent bleiben.
        z = z.to(device, non_blocking=True)
        pose = pose.to(device, non_blocking=True)

        # 1) Feste Rays für diese Pose aufbauen
        #    Orthographisch: (H, W) als Größe, sonst "focal"
        focal_or_size = self.ortho_size if getattr(self, "orthographic", False) else self.focal

        # FullRaySampler erzeugt alle Rays für das Bild-FOV
        batch_rays, _, _ = self.val_ray_sampler(self.H, self.W, focal_or_size, pose)
        batch_rays = batch_rays.to(device, non_blocking=True)

        # 2) Render-Argumente wählen (Train/Test)
        render_kwargs = self.render_kwargs_test if self.use_test_kwargs else self.render_kwargs_train
        render_kwargs = dict(render_kwargs)  # kopieren

        # Latente Codes als "features" übergeben
        render_kwargs["features"] = z

        # 3) Falls Radius als Intervall gegeben ist: near/far pro Ray anpassen
        if isinstance(self.radius, tuple):
            rays_radius = batch_rays[0].norm(dim=-1)  # [N_rays]
            shift = (self.radius[1] - rays_radius).view(-1, 1).float().to(device)

            # near/far sind im render_kwargs evtl. Skalare oder Tensoren -> in Tensoren gießen
            near = render_kwargs["near"]
            far = render_kwargs["far"]

            if not torch.is_tensor(near):
                near = torch.tensor(near, device=device, dtype=torch.float32)
            if not torch.is_tensor(far):
                far = torch.tensor(far, device=device, dtype=torch.float32)

            # auf Form [N_rays, 1] bringen
            near = near.expand_as(shift) - shift
            far = far.expand_as(shift) - shift

            render_kwargs["near"] = near
            render_kwargs["far"] = far

        # 4) Zusatz-Kontext (z. B. CT-Volumen) übergeben, falls vorhanden
        if ct_context is not None:
            render_kwargs["ct_context"] = ct_context
        elif render_kwargs.get("use_attenuation"):
            render_kwargs["use_attenuation"] = False

        # 5) NeRF-Rendering aufrufen
        #    Achtung: in unserer Emissions-Variante gibt "render" proj_map statt rgb_map zurück.
        proj_map, disp, acc, extras = self.render(rays=batch_rays, **render_kwargs)

        # 6) proj_map auf einen Skalar pro Ray reduzieren und auf [bs, H*W] bringen
        #    Mögliche Formen:
        #      - [N_rays]         (unser Emissionsfall)
        #      - [N_rays, 1]      (ein Kanal)
        #      - [N_rays, 3]      (RGB, im alten Setup)
        #      - [H, W, 3]        (altes Vollbild)
        if proj_map.dim() == 3:
            # z.B. [H, W, 3] -> Grauwert = Kanal 0
            if proj_map.shape[-1] > 1:
                proj_scalar = proj_map[..., 0]
            else:
                proj_scalar = proj_map[..., 0]
            proj_scalar = proj_scalar.reshape(-1)  # [H*W]
        elif proj_map.dim() == 2:
            # [N_rays, C]
            if proj_map.shape[1] > 1:
                proj_scalar = proj_map[:, 0]
            else:
                proj_scalar = proj_map[:, 0]
        elif proj_map.dim() == 1:
            # [N_rays]
            proj_scalar = proj_map
        else:
            raise RuntimeError(f"Unerwartete proj_map-Shape: {proj_map.shape}")

        # Jetzt sollte proj_scalar die Länge N_rays = H*W haben
        proj_flat = proj_scalar.view(bs, -1)  # [bs, H*W]

        # 7) disp und acc analog in [bs, H*W] bringen (falls vorhanden)
        disp_flat = None
        acc_flat = None

        if disp is not None:
            disp_flat = disp.reshape(-1).view(bs, -1)
        if acc is not None:
            acc_flat = acc.reshape(-1).view(bs, -1)

        return proj_flat, disp_flat, acc_flat, extras

    def build_ct_context(self, ct_volume):
        """
        Bereitet ein CT-Volumen für das Attenuation-Rendering vor.
        Annahme: Das Volumen ist zentriert und teilt sich den Bounding Cube [-radius, radius]^3 mit dem NeRF.
        """
        if ct_volume is None:
            return None
        if not torch.is_tensor(ct_volume):
            raise TypeError("ct_volume must be a torch.Tensor")
        if ct_volume.numel() == 0:
            return None

        vol = ct_volume.float()
        if vol.dim() == 3:
            vol = vol.unsqueeze(0)
        if vol.dim() == 4:
            vol = vol.unsqueeze(0)
        if vol.dim() != 5:
            raise ValueError(f"Expected CT data with D/H/W axes, got shape {tuple(ct_volume.shape)}")

        # Flip entlang der x-Achse, damit CT-Seitenlage zur AP/PA-Darstellung passt
        vol = torch.flip(vol, dims=[-1])

        radius = self.radius
        if isinstance(radius, tuple):
            radius = radius[1]
        radius = float(radius)
        if radius <= 0:
            raise ValueError("Generator radius must be > 0 to map world coords to CT grid.")

        vmin = float(vol.min().item())
        vmax = float(vol.max().item())
        vol = vol.contiguous().to(self.device, non_blocking=True)
        return {
            "volume": vol,
            "grid_radius": radius,
            "value_range": (vmin, vmax),
        }
