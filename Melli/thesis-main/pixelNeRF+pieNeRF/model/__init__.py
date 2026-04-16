# Package initializer for SpectPixelPieNeRF models.
#
# This also exposes pixel-nerf modules (encoder/custom_encoder) under the
# current package name to avoid import clashes between our local `model`
# package and the `pixel-nerf/src/model` package.

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PIXEL_NERF_SRC = _ROOT / "pixel-nerf" / "src"


def _load_pixelnerf_module(mod_name: str, filename: str):
    """Load a pixel-nerf module and register it as model.<mod_name>."""
    path = _PIXEL_NERF_SRC / "model" / filename
    if not path.exists():
        raise ImportError(f"PixelNeRF module missing: {path}")

    spec = importlib.util.spec_from_file_location(f"pixelnerf_{mod_name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError(f"Failed to load spec for {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Expose under our package namespace to satisfy absolute imports used in pixel-nerf.
    sys.modules[f"{__name__}.{mod_name}"] = module
    setattr(sys.modules[__name__], mod_name, module)
    return module


def _expose_pixelnerf_modules():
    # Skip if already loaded or path missing.
    if f"{__name__}.encoder" in sys.modules:
        return
    if not _PIXEL_NERF_SRC.exists():
        return

    src_str = str(_PIXEL_NERF_SRC)
    if src_str not in sys.path:
        # Prepend so pixel-nerf's util resolves ahead of similarly named packages.
        sys.path.insert(0, src_str)

    # Load dependencies first, then encoder which imports them.
    _load_pixelnerf_module("custom_encoder", "custom_encoder.py")
    _load_pixelnerf_module("encoder", "encoder.py")


_expose_pixelnerf_modules()
