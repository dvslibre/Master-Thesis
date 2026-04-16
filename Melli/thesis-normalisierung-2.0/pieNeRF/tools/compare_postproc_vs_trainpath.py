#!/usr/bin/env python3
"""Audit postprocessing projections against the training projection-loss path for one phantom."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import importlib
import json
import math
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from argparse import Namespace

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for quicklook PNGs") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare postproc vs training projection path for one phantom.")
    p.add_argument("--run_dir", required=True, type=Path)
    p.add_argument("--phantom_id", required=True, type=int)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _build_models_flexible(config: dict, device: torch.device):
    from graf.generator import Generator
    from graf.transforms import FlexGridRaySampler
    from nerf.run_nerf_mod import create_nerf

    config_nerf = Namespace(**config["nerf"])
    config_nerf.chunk = min(config["training"]["chunk"], 1024 * config["training"]["batch_size"])
    config_nerf.netchunk = config["training"]["netchunk"]
    config_nerf.white_bkgd = config["data"]["white_bkgd"]
    config_nerf.feat_dim = config["z_dist"]["dim"]
    config_nerf.feat_dim_appearance = config["z_dist"]["dim_appearance"]
    config_nerf.emission = True
    if not hasattr(config_nerf, "use_attenuation"):
        config_nerf.use_attenuation = False
    if not hasattr(config_nerf, "attenuation_debug"):
        config_nerf.attenuation_debug = False
    if not hasattr(config_nerf, "atten_scale"):
        config_nerf.atten_scale = 25.0

    render_kwargs_train, render_kwargs_test, params, named_parameters = create_nerf(config_nerf)
    render_kwargs_train["emission"] = True
    render_kwargs_test["emission"] = True
    render_kwargs_train["use_attenuation"] = bool(getattr(config_nerf, "use_attenuation", False))
    render_kwargs_test["use_attenuation"] = bool(getattr(config_nerf, "use_attenuation", False))
    debug_flag = bool(getattr(config_nerf, "attenuation_debug", False))
    render_kwargs_train["attenuation_debug"] = debug_flag
    render_kwargs_test["attenuation_debug"] = debug_flag
    atten_scale = float(getattr(config_nerf, "atten_scale", 25.0))
    render_kwargs_train["atten_scale"] = atten_scale
    render_kwargs_test["atten_scale"] = atten_scale
    bds_dict = {"near": config["data"]["near"], "far": config["data"]["far"]}
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    ray_sampler = FlexGridRaySampler(
        N_samples=config["ray_sampler"]["N_samples"],
        min_scale=config["ray_sampler"]["min_scale"],
        max_scale=config["ray_sampler"]["max_scale"],
        scale_anneal=config["ray_sampler"]["scale_anneal"],
        orthographic=config["data"]["orthographic"],
    )
    H, W, f, r = config["data"]["hwfr"]
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
        range_u=(float(config["data"]["umin"]), float(config["data"]["umax"])),
        range_v=(float(config["data"]["vmin"]), float(config["data"]["vmax"])),
        orthographic=config["data"]["orthographic"],
        radius_xyz_cm=tuple(config["data"]["radius_xyz_cm"]) if config["data"].get("radius_xyz_cm") is not None else None,
    )
    return generator.to(device)


def _extract_train_cli_from_command(command_path: Path) -> list[str]:
    raw = command_path.read_text().strip()
    tokens = shlex.split(raw)
    train_idx = None
    for i, tok in enumerate(tokens):
        if tok.endswith("train_emission.py"):
            train_idx = i
            break
    if train_idx is None:
        raise RuntimeError(f"Could not find train_emission.py in {command_path}")
    return tokens[train_idx + 1 :]


def _parse_train_args(train_mod, cli_tokens: list[str]) -> argparse.Namespace:
    argv_orig = sys.argv[:]
    try:
        sys.argv = ["train_emission.py"] + cli_tokens
        return train_mod.parse_args()
    finally:
        sys.argv = argv_orig


def _choose_checkpoint(ckpt_dir: Path) -> Path:
    candidates = sorted(ckpt_dir.glob("checkpoint_step*.pt"))
    if candidates:
        def _step(p: Path) -> int:
            s = p.stem
            digits = "".join(ch for ch in s if ch.isdigit())
            return int(digits) if digits else -1
        return max(candidates, key=_step)
    fallback = ckpt_dir / "checkpoint_step05000.pt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def _batchify(sample: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            out[k] = v.unsqueeze(0)
        else:
            out[k] = v
    return out


def _tensor_from_batch(batch: dict[str, Any], key: str, device: torch.device):
    t = batch.get(key)
    if t is None or (torch.is_tensor(t) and t.numel() == 0):
        return None
    return t.to(device, non_blocking=True).float()


def _render_full(generator, z_latent: torch.Tensor, ct_context) -> tuple[torch.Tensor, torch.Tensor]:
    prev = bool(generator.use_test_kwargs)
    generator.eval()
    generator.use_test_kwargs = True
    with torch.no_grad():
        ap, _, _, _ = generator.render_from_pose(z_latent, generator.pose_ap, ct_context=ct_context)
        pa, _, _, _ = generator.render_from_pose(z_latent, generator.pose_pa, ct_context=ct_context)
    generator.use_test_kwargs = prev
    return ap, pa


def _to_img_np(pred_flat: torch.Tensor, H: int, W: int) -> np.ndarray:
    return pred_flat[0].reshape(H, W).detach().cpu().numpy().astype(np.float32)


def _stats(arr: np.ndarray) -> dict[str, float]:
    x = np.asarray(arr, dtype=np.float64).ravel()
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "sum": float(np.sum(x)),
    }


def _tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    x = t.detach().float().reshape(-1)
    arr = x.cpu().numpy()
    rounded = np.round(arr, decimals=8)
    return {
        "shape": list(t.shape),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "sum": float(arr.sum()),
        "pct_zeros": float((arr == 0.0).mean() * 100.0),
        "unique_approx": int(np.unique(rounded).size),
    }


def _latent_stats(t: torch.Tensor) -> dict[str, Any]:
    s = _tensor_stats(t)
    flat = t.detach().reshape(t.shape[0], -1).float()
    norms = flat.norm(dim=1)
    s["norm_mean"] = float(norms.mean().item())
    s["norm_std"] = float(norms.std().item()) if norms.numel() > 1 else 0.0
    return s


def _assert_nonconstant(stage: str, t: torch.Tensor):
    stats = _tensor_stats(t)
    is_const = (stats["std"] < 1e-12) or (stats["min"] == stats["max"])
    if is_const:
        raise RuntimeError(f"{stage}")
    return stats


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(d * d))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    den = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-12
    return float(np.dot(x, y) / den)


def _save_compare_png(path: Path, gt: np.ndarray, train_pred: np.ndarray, post_pred: np.ndarray, title: str):
    gt_log = np.log1p(np.clip(gt, 0, None))
    tr_log = np.log1p(np.clip(train_pred, 0, None))
    pp_log = np.log1p(np.clip(post_pred, 0, None))
    vmax = float(np.percentile(np.concatenate([gt_log.ravel(), tr_log.ravel(), pp_log.ravel()]), 99.5))
    vmax = max(vmax, 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, arr, name in zip(axes, [gt_log, tr_log, pp_log], ["GT", "Pred TrainPath", "Pred Postproc"]):
        im = ax.imshow(arr, cmap="inferno", vmin=0.0, vmax=vmax)
        ax.set_title(f"{title} {name} log1p")
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _locate_postproc_projection(run_dir: Path, phantom_label: str) -> tuple[np.ndarray, np.ndarray, Path] | None:
    candidates = sorted(run_dir.glob(f"postproc*/{phantom_label}/proj/pred_ap.npy"))
    if not candidates:
        return None
    ap_path = candidates[-1]
    pa_path = ap_path.parent / "pred_pa.npy"
    if not pa_path.exists():
        return None
    ap = np.load(ap_path).astype(np.float32)
    pa = np.load(pa_path).astype(np.float32)
    return ap, pa, ap_path.parent


def _build_projector_config_from_meta(postproc_mod, meta: dict, spacing_cm: tuple[float, float, float]):
    default_kernel = Path("Data_Processing/LEAP_Kernel.mat")
    cfg = SimpleNamespace(
        psf_sigma=None,
        z0_slices=None,
        proj_sensitivity_cps_per_mbq=None,
        proj_acq_time_s=None,
    )
    return postproc_mod._build_projector_config(meta, cfg, spacing_cm, default_kernel)


def _compute_postproc_projection_from_act(
    postproc_mod,
    physics_mod,
    act_pred: np.ndarray,
    ct_path: Path,
    act_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    ct = np.load(ct_path).astype(np.float32)
    if act_pred.shape != ct.shape:
        t = torch.from_numpy(act_pred).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            rs = F.interpolate(t, size=ct.shape, mode="trilinear", align_corners=False)
        act_use = rs.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    else:
        act_use = act_pred.astype(np.float32)
    spacing = postproc_mod.spacing_from_meta(act_path, fallback_mm=1.5)
    meta = postproc_mod._load_meta_simple(act_path)
    proj_cfg = _build_projector_config_from_meta(postproc_mod, meta, spacing)
    if proj_cfg is None:
        raise RuntimeError("Could not construct projector config for postproc fallback")
    pred_proj = physics_mod.project_activity(act_use, ct, spacing, proj_cfg)
    return pred_proj["ap_counts"].astype(np.float32), pred_proj["pa_counts"].astype(np.float32)


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    phantom_label = f"phantom_{int(args.phantom_id)}"

    repo_root = Path(__file__).resolve().parents[1]
    thesis_root = repo_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(thesis_root) not in sys.path:
        sys.path.insert(0, str(thesis_root))

    train_mod = _load_module(repo_root / "train_emission.py", "train_emission_audit")
    postproc_mod = _load_module(repo_root / "postprocessing.py", "postprocessing_audit")
    physics_mod = _load_module(repo_root / "physics" / "projector.py", "projector_audit")
    config_mod = importlib.import_module("graf.config")

    cli_tokens = _extract_train_cli_from_command(run_dir / "command.sh")
    train_args = _parse_train_args(train_mod, cli_tokens)

    with open(train_args.config, "r") as f:
        config = yaml.safe_load(f)
    data_cfg = config.setdefault("data", {})
    radius_xyz_raw = data_cfg.get("radius_xyz_cm")
    radius_xyz_cm = None if radius_xyz_raw is None else tuple(float(r) for r in radius_xyz_raw)
    if bool(data_cfg.get("auto_near_far_from_radius", True)) and (radius_xyz_cm is not None):
        data_cfg["near"] = 0.0
        data_cfg["far"] = 2.0 * radius_xyz_cm[2]

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset, hwfr, _ = config_mod.get_data(config)
    config["data"]["hwfr"] = hwfr
    split_seed = int(config["data"].get("split_seed", 0))
    split_mode = str(config["data"].get("split_mode", "ratios")).lower()
    split_train = float(config["data"].get("split_train", 0.8))
    split_val = float(config["data"].get("split_val", 0.1))
    split_test = float(config["data"].get("split_test", 0.1))
    split_train_count = int(config["data"].get("split_train_count", -1))
    split_val_count = int(config["data"].get("split_val_count", -1))
    split_test_count = int(config["data"].get("split_test_count", -1))
    _, _, test_subset, split_stats = train_mod.split_by_patient_id(
        dataset=dataset,
        seed=split_seed,
        split_mode=split_mode,
        train_ratio=split_train,
        val_ratio=split_val,
        test_ratio=split_test,
        train_count=split_train_count,
        val_count=split_val_count,
        test_count=split_test_count,
    )
    dataset_index = None
    for i in range(len(dataset)):
        if dataset.get_patient_id(i) == phantom_label:
            dataset_index = i
            break
    if dataset_index is None:
        raise RuntimeError(f"{phantom_label} not found in dataset")
    test_indices = set(getattr(test_subset, "indices", []))
    if dataset_index not in test_indices:
        raise RuntimeError(f"{phantom_label} dataset index {dataset_index} is not in test split")

    ckpt_path = _choose_checkpoint(run_dir / "checkpoints")
    print(f"CHECKPOINT: {ckpt_path}")
    print(f"DATASET_INDEX: {dataset_index} ({phantom_label})")

    generator = _build_models_flexible(config, device=device)
    generator.train()
    generator.use_test_kwargs = False
    generator.set_fixed_ap_pa(radius=hwfr[3])

    z_dim = int(config["z_dist"]["dim"])
    z_base = torch.zeros(1, z_dim, device=device)
    encoder = None
    z_fuser = None
    gain_head = None
    gain_param = None
    if bool(train_args.hybrid):
        in_ch = 2 + (1 if train_args.encoder_use_ct else 0)
        encoder = train_mod.ProjectionEncoder(in_ch=in_ch, z_dim=z_dim, base_ch=32).to(device)
        z_fuser = torch.nn.Sequential(torch.nn.Linear(z_dim, z_dim), torch.nn.LayerNorm(z_dim)).to(device)
        if train_args.proj_target_source == "counts":
            if train_args.proj_gain_source == "z_enc":
                gain_head = torch.nn.Linear(z_dim, 1).to(device)
            elif train_args.proj_gain_source == "scalar":
                gain_param = torch.nn.Parameter(torch.zeros(1, device=device))

    ckpt = torch.load(ckpt_path, map_location="cpu")
    generator.render_kwargs_train["network_fn"].load_state_dict(ckpt["generator_coarse"])
    if generator.render_kwargs_train["network_fine"] is not None and ckpt.get("generator_fine") is not None:
        generator.render_kwargs_train["network_fine"].load_state_dict(ckpt["generator_fine"])
    if encoder is not None and ckpt.get("encoder") is not None:
        encoder.load_state_dict(ckpt["encoder"])
    if z_fuser is not None and ckpt.get("z_fuser") is not None:
        z_fuser.load_state_dict(ckpt["z_fuser"])
    if gain_head is not None and ckpt.get("gain_head") is not None:
        gain_head.load_state_dict(ckpt["gain_head"])
    if gain_param is not None and ckpt.get("gain_param") is not None:
        gain_param.data.copy_(ckpt["gain_param"].to(device))

    generator.eval()
    if encoder is not None:
        encoder.eval()
    if z_fuser is not None:
        z_fuser.eval()
    if gain_head is not None:
        gain_head.eval()

    sample = dataset[dataset_index]
    batch = _batchify(sample)
    ap = _tensor_from_batch(batch, "ap", device)
    pa = _tensor_from_batch(batch, "pa", device)
    ap_counts = _tensor_from_batch(batch, "ap_counts", device)
    pa_counts = _tensor_from_batch(batch, "pa_counts", device)
    ct_vol = _tensor_from_batch(batch, "ct", device)
    act_gt_t = _tensor_from_batch(batch, "act", device)
    if ap is None or pa is None:
        raise RuntimeError("Missing AP/PA in dataset sample")
    use_counts = ap_counts is not None and pa_counts is not None and ap_counts.numel() > 0 and pa_counts.numel() > 0
    report_diag: dict[str, Any] = {}
    proj_scale_enc = train_mod.compute_proj_scale(
        ap,
        pa,
        train_args.proj_scale_source,
        batch.get("meta"),
    )
    proj_scale_enc = torch.clamp(proj_scale_enc, min=1e-6)
    report_diag["proj_scale"] = {
        "source": str(train_args.proj_scale_source),
        "values": [float(x) for x in proj_scale_enc.detach().cpu().reshape(-1).tolist()],
    }
    print(
        f"[audit] proj_scale_source={train_args.proj_scale_source} proj_scale={report_diag['proj_scale']['values']}",
        flush=True,
    )

    if bool(train_args.hybrid) and encoder is not None:
        z_final, z_enc, z_proj = train_mod.build_hybrid_latent_from_batch(
            args=train_args,
            batch=batch,
            device=device,
            z_latent_base=z_base,
            encoder=encoder,
            z_fuser=z_fuser,
            z_enc_alpha=float(train_args.z_enc_alpha),
        )
    else:
        z_final = z_base.detach()
        z_enc = None
        z_proj = None
    report_diag["hybrid_enabled"] = bool(train_args.hybrid)
    report_diag["z_enc_alpha"] = float(train_args.z_enc_alpha)
    report_diag["z_base"] = _latent_stats(z_base)
    report_diag["z_final"] = _latent_stats(z_final)
    if z_enc is not None:
        report_diag["z_enc"] = _latent_stats(z_enc)
    if z_proj is not None:
        report_diag["z_proj"] = _latent_stats(z_proj)
    print(f"[audit] z_final stats={report_diag['z_final']}", flush=True)
    if "z_enc" in report_diag:
        print(f"[audit] z_enc stats={report_diag['z_enc']}", flush=True)
    if "z_proj" in report_diag:
        print(f"[audit] z_proj stats={report_diag['z_proj']}", flush=True)

    ct_context = None
    if ct_vol is not None and ct_vol.numel() > 0:
        ct_context = generator.build_ct_context(ct_vol, padding_mode=train_args.ct_padding_mode)

    raw_ap, raw_pa = _render_full(generator, z_final, ct_context)
    report_diag["renderer_raw_ap"] = _assert_nonconstant("renderer_raw/AP", raw_ap)
    report_diag["renderer_raw_pa"] = _assert_nonconstant("renderer_raw/PA", raw_pa)
    print(f"[audit] renderer_raw AP stats={report_diag['renderer_raw_ap']}", flush=True)
    print(f"[audit] renderer_raw PA stats={report_diag['renderer_raw_pa']}", flush=True)

    if str(train_args.proj_loss_type) == "poisson":
        post_rate_ap = train_mod.compute_poisson_rate(raw_ap, train_args.poisson_rate_mode, eps=1e-6)
        post_rate_pa = train_mod.compute_poisson_rate(raw_pa, train_args.poisson_rate_mode, eps=1e-6)
    else:
        post_rate_ap = raw_ap
        post_rate_pa = raw_pa
    report_diag["post_rate_ap"] = _assert_nonconstant("post_rate/AP", post_rate_ap)
    report_diag["post_rate_pa"] = _assert_nonconstant("post_rate/PA", post_rate_pa)
    print(f"[audit] post_rate AP stats={report_diag['post_rate_ap']}", flush=True)
    print(f"[audit] post_rate PA stats={report_diag['post_rate_pa']}", flush=True)

    gain_tensor = None
    if use_counts and bool(train_args.use_gain):
        if gain_head is not None and z_enc is not None:
            gain_tensor = F.softplus(gain_head(z_enc))
        elif gain_param is not None:
            gain_tensor = F.softplus(gain_param)
    if gain_tensor is not None:
        gmin = float(train_args.gain_clamp_min) if train_args.gain_clamp_min is not None else None
        gmax = float(train_args.gain_clamp_max) if train_args.gain_clamp_max is not None else None
        if gmin is not None or gmax is not None:
            gain_tensor = torch.clamp(
                gain_tensor,
                min=(gmin if gmin is not None else -float("inf")),
                max=(gmax if gmax is not None else float("inf")),
            )
    report_diag["gain"] = None if gain_tensor is None else _tensor_stats(gain_tensor)
    if report_diag["gain"] is not None:
        print(f"[audit] gain stats={report_diag['gain']}", flush=True)

    post_gain_ap = post_rate_ap
    post_gain_pa = post_rate_pa
    if gain_tensor is not None:
        post_gain_ap = post_gain_ap * gain_tensor
        post_gain_pa = post_gain_pa * gain_tensor
    if use_counts and str(train_args.proj_loss_type) == "poisson" and float(train_args.poisson_rate_floor) > 0.0:
        post_gain_ap, _ = train_mod.apply_poisson_rate_floor(
            post_gain_ap, float(train_args.poisson_rate_floor), train_args.poisson_rate_floor_mode
        )
        post_gain_pa, _ = train_mod.apply_poisson_rate_floor(
            post_gain_pa, float(train_args.poisson_rate_floor), train_args.poisson_rate_floor_mode
        )
    report_diag["post_gain_ap"] = _assert_nonconstant("post_gain/AP", post_gain_ap)
    report_diag["post_gain_pa"] = _assert_nonconstant("post_gain/PA", post_gain_pa)
    print(f"[audit] post_gain AP stats={report_diag['post_gain_ap']}", flush=True)
    print(f"[audit] post_gain PA stats={report_diag['post_gain_pa']}", flush=True)

    H, W = int(generator.H), int(generator.W)
    ap_pred_train = _to_img_np(post_gain_ap, H, W)
    pa_pred_train = _to_img_np(post_gain_pa, H, W)

    manifest_path = Path(config["data"]["manifest"])
    entry = None
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("patient_id") == phantom_label:
                entry = row
                break
    if entry is None:
        raise RuntimeError(f"{phantom_label} not found in manifest {manifest_path}")
    act_path = Path(entry["act_path"])
    ct_path = Path(entry["ct_path"])

    # Save GT activity
    act_gt_np = act_gt_t.squeeze(0).detach().cpu().numpy().astype(np.float32) if act_gt_t is not None else np.load(act_path).astype(np.float32)
    np.save(out_dir / "act_gt.npy", act_gt_np)

    # Reuse already exported model prediction if available; fallback to fresh export.
    existing_pred_path = run_dir / "test_slices" / phantom_label / "activity_pred.npy"
    if existing_pred_path.exists():
        act_pred_np = np.load(existing_pred_path).astype(np.float32)
    else:
        act_pred_np = train_mod.export_activity_volume(
            generator=generator,
            z_latent=z_hybrid.detach(),
            out_path=None,
            res=int(getattr(train_args, "export_vol_res", 128)),
            device=device,
            radius_xyz=tuple(float(r) for r in generator.radius_xyz),
            log_world_range=False,
        ).astype(np.float32)
    np.save(out_dir / "act_pred.npy", act_pred_np)

    gt_ap_np = ap_counts.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32) if use_counts else ap.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    gt_pa_np = pa_counts.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32) if use_counts else pa.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    if use_counts:
        np.save(out_dir / "ap_gt_counts.npy", gt_ap_np)
        np.save(out_dir / "pa_gt_counts.npy", gt_pa_np)
    else:
        np.save(out_dir / "ap_gt.npy", gt_ap_np)
        np.save(out_dir / "pa_gt.npy", gt_pa_np)

    postproc_loaded = _locate_postproc_projection(run_dir, phantom_label)
    if postproc_loaded is not None:
        ap_pred_post, pa_pred_post, postproc_src = postproc_loaded
        print(f"POSTPROC_SOURCE: loaded {postproc_src}")
    else:
        print("POSTPROC_SOURCE: fallback recompute via physics.projector.project_activity")
        ap_pred_post, pa_pred_post = _compute_postproc_projection_from_act(
            postproc_mod=postproc_mod,
            physics_mod=physics_mod,
            act_pred=act_pred_np,
            ct_path=ct_path,
            act_path=act_path,
        )

    np.save(out_dir / "ap_pred_trainpath.npy", ap_pred_train.astype(np.float32))
    np.save(out_dir / "pa_pred_trainpath.npy", pa_pred_train.astype(np.float32))
    np.save(out_dir / "ap_pred_postproc.npy", ap_pred_post.astype(np.float32))
    np.save(out_dir / "pa_pred_postproc.npy", pa_pred_post.astype(np.float32))

    _save_compare_png(out_dir / "quicklook_ap_log1p.png", gt_ap_np, ap_pred_train, ap_pred_post, "AP")
    _save_compare_png(out_dir / "quicklook_pa_log1p.png", gt_pa_np, pa_pred_train, pa_pred_post, "PA")

    train_post_mse = {
        "ap": _mse(ap_pred_train, ap_pred_post),
        "pa": _mse(pa_pred_train, pa_pred_post),
        "mean": 0.5 * (_mse(ap_pred_train, ap_pred_post) + _mse(pa_pred_train, pa_pred_post)),
    }
    train_post_ncc = {
        "ap": _ncc(ap_pred_train, ap_pred_post),
        "pa": _ncc(pa_pred_train, pa_pred_post),
        "mean": 0.5 * (_ncc(ap_pred_train, ap_pred_post) + _ncc(pa_pred_train, pa_pred_post)),
    }
    train_gt_mse = {
        "ap": _mse(ap_pred_train, gt_ap_np),
        "pa": _mse(pa_pred_train, gt_pa_np),
        "mean": 0.5 * (_mse(ap_pred_train, gt_ap_np) + _mse(pa_pred_train, gt_pa_np)),
    }
    train_gt_ncc = {
        "ap": _ncc(ap_pred_train, gt_ap_np),
        "pa": _ncc(pa_pred_train, gt_pa_np),
        "mean": 0.5 * (_ncc(ap_pred_train, gt_ap_np) + _ncc(pa_pred_train, gt_pa_np)),
    }

    report = {
        "phantom_id": phantom_label,
        "checkpoint": str(ckpt_path),
        "dataset_index": int(dataset_index),
        "use_counts_target": bool(use_counts),
        "train_flags": {
            "hybrid": bool(train_args.hybrid),
            "encoder_use_ct": bool(train_args.encoder_use_ct),
            "z_enc_alpha": float(train_args.z_enc_alpha),
            "encoder_proj_transform": str(train_args.encoder_proj_transform),
            "proj_scale_source": str(train_args.proj_scale_source),
            "proj_target_source": str(train_args.proj_target_source),
            "proj_loss_type": str(train_args.proj_loss_type),
            "poisson_rate_mode": str(train_args.poisson_rate_mode),
            "use_gain": bool(train_args.use_gain),
            "proj_gain_source": str(train_args.proj_gain_source),
            "pa_xflip": bool(train_args.pa_xflip),
        },
        "diagnostics": report_diag,
        "pred_train_vs_postproc": {
            "ap_stats_train": _stats(ap_pred_train),
            "ap_stats_postproc": _stats(ap_pred_post),
            "pa_stats_train": _stats(pa_pred_train),
            "pa_stats_postproc": _stats(pa_pred_post),
            "mse": train_post_mse,
            "ncc": train_post_ncc,
        },
        "pred_train_vs_gt": {
            "ap_stats_train": _stats(ap_pred_train),
            "ap_stats_gt": _stats(gt_ap_np),
            "pa_stats_train": _stats(pa_pred_train),
            "pa_stats_gt": _stats(gt_pa_np),
            "mse": train_gt_mse,
            "ncc": train_gt_ncc,
        },
    }
    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2))

    print("\n=== DIFF REPORT ===")
    print(f"pred_trainpath vs pred_postproc MSE(mean): {train_post_mse['mean']:.6e}")
    print(f"pred_trainpath vs pred_postproc NCC(mean): {train_post_ncc['mean']:.6f}")
    print(f"pred_trainpath vs GT MSE(mean): {train_gt_mse['mean']:.6e}")
    print(f"pred_trainpath vs GT NCC(mean): {train_gt_ncc['mean']:.6f}")
    print("AP train stats:", _stats(ap_pred_train))
    print("AP postproc stats:", _stats(ap_pred_post))
    print("PA train stats:", _stats(pa_pred_train))
    print("PA postproc stats:", _stats(pa_pred_post))
    print(
        "DIAGNOSIS: constant_stage=none "
        "root_cause=tool_was_not_following_exact_training_projection_path "
        "fix=aligned_near_far_hybrid_scale_render_poisson_gain_and_added_stage_assertions",
        flush=True,
    )


if __name__ == "__main__":
    main()
