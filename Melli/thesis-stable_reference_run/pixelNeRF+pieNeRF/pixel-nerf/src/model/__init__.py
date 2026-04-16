from .models import PixelNeRFNet
from .spect_model import SpectReconNet
from .spect_nerf_model import SpectNeRFNet


def make_model(conf, *args, **kwargs):
    """ Placeholder to allow more model types """
    model_type = conf.get_string("type", "pixelnerf")  # single
    if model_type == "pixelnerf":
        net = PixelNeRFNet(conf, *args, **kwargs)
    elif model_type == "spect":
        base_ch = conf.get_int("base_channels", 32)
        net = SpectReconNet(base_channels=base_ch)
    elif model_type == "spect_nerf":
        net = SpectNeRFNet(conf, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net
