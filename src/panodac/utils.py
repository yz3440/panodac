"""Utility functions for panodac."""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image


def get_device(device: Union[str, None] = None) -> torch.device:
    """Get the best available device for inference.
    
    Args:
        device: Specific device to use ('cuda', 'mps', 'cpu'), 
                or None for auto-detection.
    
    Returns:
        torch.device for inference
    """
    if device is not None:
        return torch.device(device)
    
    # Auto-detect best device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_image(image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """Load and normalize an image to numpy array.
    
    Args:
        image: Input image (path, numpy array, or PIL Image)
    
    Returns:
        numpy array (H, W, 3) in RGB format, uint8
    """
    if isinstance(image, (str, Path)):
        image = np.asarray(Image.open(image))
    elif isinstance(image, Image.Image):
        image = np.asarray(image)
    elif isinstance(image, np.ndarray):
        image = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    # Handle different channel formats
    if len(image.shape) == 2:  # Grayscale
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]
    
    return image


def is_panorama(image: np.ndarray, threshold: float = 1.8) -> bool:
    """Check if an image is likely a 360° equirectangular panorama.
    
    Args:
        image: Input image array (H, W, 3)
        threshold: Aspect ratio threshold (width/height)
    
    Returns:
        True if the image appears to be a panorama
    """
    h, w = image.shape[:2]
    return (w / h) >= threshold
