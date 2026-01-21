# panodac

Metric depth estimation for any camera. Perspective, fisheye, 360° panorama.

Based on Depth Any Camera (CVPR 2025).

## Installation

```bash
pip install panodac
```

Or install from source:

```bash
pip install "panodac @ git+https://github.com/yz3440/panodac.git"
```

## Quick Start

```python
import panodac

# Predict depth from any image
depth = panodac.predict("photo.jpg")
# depth is a numpy array (H, W) with metric depth in meters

# Use a specific model
depth = panodac.predict("panorama.jpg", model="outdoor-swinl")

# List available models
print(panodac.list_models())
# ['outdoor-resnet101', 'outdoor-swinl', 'indoor-resnet101', 'indoor-swinl']
```

## Models

| Model               | Use Case | Speed | Quality |
| ------------------- | -------- | ----- | ------- |
| `outdoor-resnet101` | Outdoor  | Fast  | Good    |
| `outdoor-swinl`     | Outdoor  | Slow  | Best    |
| `indoor-resnet101`  | Indoor   | Fast  | Good    |
| `indoor-swinl`      | Indoor   | Slow  | Best    |

Models auto-download from HuggingFace on first use (~500MB each).

## Device Selection

panodac automatically uses the best available device:

1. **Apple Silicon (MPS)** — Used by default on M1/M2/M3 Macs
2. **CUDA** — Used when NVIDIA GPU is available
3. **CPU** — Fallback when no GPU is available

```python
# Check current device
print(panodac.get_device())  # 'mps', 'cuda', or 'cpu'

# Force specific device
depth = panodac.predict("image.jpg", device="cpu")
```

## Input Formats

The `predict()` function accepts multiple input types:

```python
import panodac
from PIL import Image
import numpy as np

# File path (string or Path)
depth = panodac.predict("photo.jpg")

# PIL Image
img = Image.open("photo.jpg")
depth = panodac.predict(img)

# NumPy array (H, W, 3) RGB
img_np = np.array(Image.open("photo.jpg"))
depth = panodac.predict(img_np)
```

## Panorama Detection

360° equirectangular panoramas (2:1 aspect ratio) are automatically detected and processed with appropriate spherical coordinates.

```python
# Panoramas are auto-detected
depth = panodac.predict("panorama_360.jpg")
```

## Next

- [Examples](examples.md) - Working scripts
- [API Reference](api/index.md) - Full documentation
