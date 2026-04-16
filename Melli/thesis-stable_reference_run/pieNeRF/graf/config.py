"""Config helpers: load/update YAML, build SPECT dataset and emission NeRF generator."""

import numpy as np
import torch

from .datasets import SpectDataset
from .transforms import FlexGridRaySampler
from nerf.run_nerf_mod import create_nerf


def save_config(outpath, config):
    """Speichert eine Config als YAML-Datei – nützlich für Reproduzierbarkeit."""
    from yaml import safe_dump
    with open(outpath, "w") as f:
        safe_dump(config, f)


def update_config(config, unknown):
    """Ermöglicht CLI-Overrides, z.B. --data:imsize 128 überschreibt YAML."""
    for idx, arg in enumerate(unknown):
        if not arg.startswith("--"):
            continue
        if ":" in arg:
            k1, k2 = arg.replace("--", "").split(":")
            argtype = type(config[k1][k2])
            if argtype == bool:
                v = unknown[idx + 1].lower() == "true"
            else:
                v = type(config[k1][k2])(unknown[idx + 1]) if config[k1][k2] is not None else unknown[idx + 1]
            print(f"Changing {k1}:{k2} ---- {config[k1][k2]} to {v}")
            config[k1][k2] = v
        else:
            k = arg.replace("--", "")
            v = unknown[idx + 1]
            argtype = type(config[k])
            print(f"Changing {k} ---- {config[k]} to {v}")
            config[k] = v
    return config


def get_data(config):
    """Lädt das SPECT-Dataset (AP, PA, CT) basierend auf manifest.csv
    und leitet H, W direkt aus den echten AP-Bildern ab.
    """
    dset_type = config["data"]["type"]
    if dset_type != "spect":
        raise ValueError(f"Dieser Build unterstützt nur 'spect', nicht '{dset_type}'.")

    # 1) Dataset bauen (ohne Resize-Transforms)
    dset = SpectDataset(
        manifest_path=config["data"]["manifest"],
        imsize=config["data"]["imsize"],                            # imsize ist hier nur noch „Meta“, nicht verbindlich
        transform_img=None,
        transform_ct=None,
        act_scale=float(config["data"].get("act_scale", 1.0)),
    )

    # 2) H und W aus einem Beispiel-AP-Bild ableiten
    sample0 = dset[0]
    ap0 = sample0["ap"]                                             # Shape: [1, H, W]
    _, H, W = ap0.shape                                             # H und W stammen aus echten Daten (nicht aus config)

    # 3) FOV aus Config nehmen und formale "focal" berechnen (wird für orthografisches Rendern nicht genutzt)
    fov = config["data"]["fov"]
    dset.H = H
    dset.W = W
    dset.focal = W / 2 * 1 / np.tan(0.5 * fov * np.pi / 180.0)

    # 4) Radius wie gehabt aus Config (definiert den Bounding-Box-Radius in Weltkoordinaten)
    radius = config["data"]["radius"]
    render_radius = radius
    if isinstance(radius, str):
        radius = tuple(float(r) for r in radius.split(","))
        render_radius = max(radius)
    dset.radius = radius

    # 5) Keine Render-Posen (AP/PA sind fix), also None
    render_poses = None

    # 6) Debug-Ausgabe
    datainfo = config["data"].get("manifest", "n/a")
    print(
        f"Loaded {dset_type}: H={H}, W={W}, samples={len(dset)}, "
        f"radius={dset.radius}, data={datainfo}"
    )

    # 7) hwfr jetzt mit echten H, W
    hwfr = [H, W, dset.focal, dset.radius]
    return dset, hwfr, render_poses


def build_models(config):
    """Baut ausschließlich den Generator (NeRF) für die Emissionsrekonstruktion."""
    from argparse import Namespace
    from graf.generator import Generator

    config_nerf = Namespace(**config["nerf"])
    config_nerf.chunk = min(config["training"]["chunk"], 1024 * config["training"]["batch_size"])           # wie viele Rays gleichzeitig durch das Netz laufen
    config_nerf.netchunk = config["training"]["netchunk"]                                                   # wie viele Punkte pro forward pass durch das MLP laufen dürfen
    config_nerf.white_bkgd = config["data"]["white_bkgd"]                                                   # Hintergrundfarbe bei Volumen-Rendering (True: weiß, False: schwarz)
    config_nerf.feat_dim = config["z_dist"]["dim"]                                                          # Dimension des latent Codes z (Geometrie)
    config_nerf.feat_dim_appearance = config["z_dist"]["dim_appearance"]                                    # Dimension des latent Codes z_appearance (Appearance)
    config_nerf.emission = True                                                                             # Flag für Emission-Rendering (keine klassische Absorption)
    if not hasattr(config_nerf, "use_attenuation"):
        config_nerf.use_attenuation = False                                                                 # Wenn True: Emission wird mit CT-basiertem μ moduliert 
    if not hasattr(config_nerf, "attenuation_debug"):
        config_nerf.attenuation_debug = False
    if not hasattr(config_nerf, "atten_scale"):
        config_nerf.atten_scale = 25.0

    render_kwargs_train, render_kwargs_test, params, named_parameters = create_nerf(config_nerf)            # Erstellung des eigentlichen NeRFs (alles in render_kwargs_*, die später vom Generator benutzt werden)
    render_kwargs_train['emission'] = True
    render_kwargs_test['emission']  = True
    render_kwargs_train['use_attenuation'] = bool(getattr(config_nerf, "use_attenuation", False))
    render_kwargs_test['use_attenuation'] = bool(getattr(config_nerf, "use_attenuation", False))
    debug_flag = bool(getattr(config_nerf, "attenuation_debug", False))
    render_kwargs_train['attenuation_debug'] = debug_flag
    render_kwargs_test['attenuation_debug'] = debug_flag
    atten_scale = float(getattr(config_nerf, "atten_scale", 25.0))
    render_kwargs_train['atten_scale'] = atten_scale
    render_kwargs_test['atten_scale'] = atten_scale

    bds_dict = {"near": config["data"]["near"], "far": config["data"]["far"]}                               # near / far als z-Grenzen entlang der Rays, in denen gesampelt wird (near: Start, far: Ende)
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    ray_sampler = FlexGridRaySampler(                                                                       # definiert, wie entlang der Rays gesampelt wird
        N_samples=config["ray_sampler"]["N_samples"],                                                       # Anzahl Punkt-Samples pro Ray    
        min_scale=config["ray_sampler"]["min_scale"],                                                       # steuert wie groß der Bildausschnitt ist, aus dem rays gesampelt werden        
        max_scale=config["ray_sampler"]["max_scale"],
        scale_anneal=config["ray_sampler"]["scale_anneal"],
        orthographic=config["data"]["orthographic"],                                                        # parallele Rays?
    )

    H, W, f, r = config["data"]["hwfr"]                                                                     # Paket aus get_data(): H --> Bildhöhe, W --> Bildbreite, f --> formale focal length, r --> radius
    generator = Generator(
        H,
        W,
        f,
        r,
        ray_sampler=ray_sampler,
        render_kwargs_train=render_kwargs_train,
        render_kwargs_test=render_kwargs_test,
        parameters=params,
        named_parameters=named_parameters,
        chunk=config_nerf.chunk,
        range_u=(float(config["data"]["umin"]), float(config["data"]["umax"])),                             # normalisierte Koordinatenbereiche im Bild (für Auswahl von Pixelbereichen)
        range_v=(float(config["data"]["vmin"]), float(config["data"]["vmax"])),
        orthographic=config["data"]["orthographic"],
    )
    generator = generator.to("cuda")                                                                        # verschiebt alle Modelle auf die GPU

    return generator


def build_lr_scheduler(optimizer, config, last_epoch=-1):                                                   # nimmt optimizer (z.B. Adam) und config, gibt PyTorch LR-Scheduler zurück (steuert, wie aggressiv die Lernrate während des Trainings abgesenkt wird)
    """Lernratenplanung: Step- oder MultiStep-Scheduler."""
    import torch.optim as optim

    step_size = config["training"]["lr_anneal_every"]
    if isinstance(step_size, str):                                                                          # wenn lr_anneal_every string --> multistep scheduler (zu angegebenen steps wird lr = lr * 0,5)
        milestones = [int(m) for m in step_size.split(",")]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=config["training"]["lr_anneal"],
            last_epoch=last_epoch,
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(                                                              # wenn lr_anneal_every int --> step scheduler (z.B. alle 10000 steps: lr = lr * 0,5)
            optimizer,
            step_size=step_size,
            gamma=config["training"]["lr_anneal"],
            last_epoch=last_epoch,
        )
    return scheduler
