# panodac

<p align="center">
  <b>Plug-and-play metric depth estimation for any camera</b>
</p>

<p align="center">
  Perspective • Fisheye • 360° Panorama
</p>

---

`panodac` is a lightweight Python package for zero-shot metric depth estimation that works with any camera type—perspective, fisheye, or 360° equirectangular panoramas.

Based on **Depth Any Camera (DAC)** from CVPR 2025.

## Features

- 🎯 **One-line API** — `panodac.predict(image)` returns metric depth in meters
- 📷 **Any camera** — Works with perspective, fisheye, and 360° panoramas
- 🚀 **Backend-agnostic** — Runs on CUDA, Apple Silicon (MPS), or CPU
- 📦 **Zero config** — Models auto-download from HuggingFace Hub
- 🔧 **Pure PyTorch** — No CUDA compilation required

## Installation

```bash
pip install panodac
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add panodac
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

| Model | Use Case | Speed | Quality |
|-------|----------|-------|---------|
| `outdoor-resnet101` | Outdoor scenes (default) | Fast | Good |
| `outdoor-swinl` | Outdoor scenes | Slow | Best |
| `indoor-resnet101` | Indoor scenes | Fast | Good |
| `indoor-swinl` | Indoor scenes | Slow | Best |

Models are automatically downloaded on first use (~500MB each).

## Examples

### Basic Depth Prediction

```python
import panodac
import matplotlib.pyplot as plt

# Predict depth
depth = panodac.predict("your_image.jpg")

# Visualize
plt.imshow(depth, cmap="turbo")
plt.colorbar(label="Depth (m)")
plt.show()
```

### 360° Panorama with Point Cloud Export

```python
import panodac
import numpy as np

# Predict depth from panorama
depth = panodac.predict("panorama_360.jpg")

# The package automatically detects panoramas (2:1 aspect ratio)
# and uses appropriate spherical coordinates for processing
```

See [`examples/`](./examples/) for complete examples including point cloud generation.

### Batch Processing

```bash
python examples/batch_inference.py ./input_images ./output_depths --model outdoor-swinl
```

## Device Selection

panodac automatically uses the best available device:

1. **Apple Silicon (MPS)** — Used by default on M1/M2/M3 Macs
2. **CUDA** — Used when NVIDIA GPU is available
3. **CPU** — Fallback when no GPU is available

Override with:

```python
depth = panodac.predict("image.jpg", device="cpu")  # Force CPU
```

Check current device:

```python
print(panodac.get_device())  # 'mps', 'cuda', or 'cpu'
```

## API Reference

### `panodac.predict(image, model="outdoor-resnet101", device=None)`

Predict metric depth from an image.

**Args:**
- `image`: Path to image file, numpy array (H, W, 3), or PIL Image
- `model`: Model name (see table above)
- `device`: `'cuda'`, `'mps'`, `'cpu'`, or `None` for auto-detect

**Returns:**
- `np.ndarray`: Depth map (H, W) with metric depth in meters

### `panodac.list_models()`

Returns list of available model names.

### `panodac.get_device(device=None)`

Returns the torch device that will be used for inference.

## Credits

This package is based on **Depth Any Camera (DAC)** by Yuliang Guo et al.

- **Paper**: [Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](https://arxiv.org/abs/2501.02464)
- **Original Repository**: [github.com/yuliangguo/depth_any_camera](https://github.com/yuliangguo/depth_any_camera)
- **Project Page**: [depth-any-camera.github.io](https://depth-any-camera.github.io/)

If you use this work in your research, please cite:

```bibtex
@article{guo2025depthany,
  title={Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera},
  author={Guo, Yuliang and Garg, Sparsh and Ren, Xuan and ElSayed, Mohamed and Guizilini, Vitor},
  journal={CVPR},
  year={2025},
}
```

## License

MIT License. See [LICENSE](./LICENSE) for details.

The pretrained models are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
