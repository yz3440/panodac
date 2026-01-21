"""
panodac - Plug-and-play metric depth estimation for any camera.

Based on Depth Any Camera (DAC) - CVPR 2025.
Original paper: https://arxiv.org/abs/2501.02464
Original repo: https://github.com/yuliangguo/depth_any_camera
"""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from .predictor import DepthPredictor
from .utils import get_device

__version__ = "0.1.0"
__all__ = ["predict", "list_models", "get_device", "DepthPredictor"]

# Available models
AVAILABLE_MODELS = [
    "outdoor-resnet101",
    "outdoor-swinl",
    "indoor-resnet101",
    "indoor-swinl",
]

# Global predictor cache (lazy-loaded)
_predictor_cache: dict[str, DepthPredictor] = {}


def list_models() -> list[str]:
    """List all available pretrained models.

    Returns:
        List of model names that can be passed to predict()
    """
    return AVAILABLE_MODELS.copy()


def predict(
    image: Union[str, Path, np.ndarray, Image.Image],
    model: str = "outdoor-resnet101",
    device: Union[str, None] = None,
    fix_panorama_seam: bool = True,
) -> np.ndarray:
    """Predict metric depth from any camera image.

    Supports perspective, fisheye, and 360° panorama images.
    Models are automatically downloaded from HuggingFace on first use.

    Args:
        image: Input image. Can be:
            - Path to image file (str or Path)
            - numpy array (H, W, 3) in RGB format, values 0-255
            - PIL Image
        model: Model to use. One of:
            - 'outdoor-resnet101' (default): Fast outdoor model
            - 'outdoor-swinl': High-quality outdoor model
            - 'indoor-resnet101': Fast indoor model
            - 'indoor-swinl': High-quality indoor model
        device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
        fix_panorama_seam: If True (default), apply Poisson blending to correct
            left-right seam artifacts in ERP panorama depth outputs.

    Returns:
        Depth map as numpy array (H, W) with metric depth in meters.

    Example:
        >>> import panodac
        >>> depth = panodac.predict("photo.jpg")
        >>> depth = panodac.predict("panorama.jpg", model="outdoor-swinl")
    """
    if model not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model}'. Available models: {AVAILABLE_MODELS}"
        )

    # Get or create cached predictor
    cache_key = f"{model}:{device}:{fix_panorama_seam}"
    if cache_key not in _predictor_cache:
        _predictor_cache[cache_key] = DepthPredictor(
            model=model, device=device, fix_panorama_seam=fix_panorama_seam
        )

    predictor = _predictor_cache[cache_key]
    return predictor(image)
