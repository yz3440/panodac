"""Basic tests for panodac."""

import numpy as np
import pytest


def test_list_models():
    """Test that list_models returns expected models."""
    import panodac
    
    models = panodac.list_models()
    assert isinstance(models, list)
    assert "outdoor-resnet101" in models
    assert "outdoor-swinl" in models
    assert "indoor-resnet101" in models
    assert "indoor-swinl" in models


def test_get_device():
    """Test that get_device returns a valid device."""
    import torch
    import panodac
    
    device = panodac.get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda", "mps")


def test_get_device_override():
    """Test device override."""
    import torch
    import panodac
    
    device = panodac.get_device("cpu")
    assert device.type == "cpu"


def test_utils_load_image():
    """Test image loading utilities."""
    from pathlib import Path
    from panodac.utils import load_image
    
    # Test loading from numpy array
    img_np = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    loaded = load_image(img_np)
    assert loaded.shape == (100, 200, 3)
    
    # Test grayscale conversion
    gray = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
    loaded = load_image(gray)
    assert loaded.shape == (100, 200, 3)


def test_utils_is_panorama():
    """Test panorama detection."""
    from panodac.utils import is_panorama
    
    # 2:1 aspect ratio = panorama
    pano = np.zeros((500, 1000, 3), dtype=np.uint8)
    assert is_panorama(pano) is True
    
    # 4:3 aspect ratio = not panorama
    photo = np.zeros((768, 1024, 3), dtype=np.uint8)
    assert is_panorama(photo) is False
    
    # 16:9 aspect ratio = borderline (should be not panorama)
    wide = np.zeros((540, 960, 3), dtype=np.uint8)
    assert is_panorama(wide) is False


# Skip slow tests that require model downloads unless explicitly requested
@pytest.mark.slow
def test_predict_dummy():
    """Test prediction with a dummy image (requires model download)."""
    import panodac
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # This will download the model on first run
    depth = panodac.predict(dummy_image, model="outdoor-resnet101")
    
    assert isinstance(depth, np.ndarray)
    assert depth.shape == (480, 640)
    assert depth.dtype == np.float32 or depth.dtype == np.float64
