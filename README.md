# panodac

Metric depth estimation for any camera. Perspective, fisheye, 360° panorama.

Based on Depth Any Camera (CVPR 2025).

## Installation

```bash
pip install panodac
```

## Usage

```python
import panodac

depth = panodac.predict("photo.jpg")
# depth is a numpy array (H, W) with metric depth in meters
```

## Panorama Seam Blending

ERP panoramas wrap horizontally, but CNN padding can introduce a visible seam at the left/right boundary. By default, `panodac` applies a Poisson-based seam correction when a panorama is detected.

```python
import panodac

# Default: seam correction enabled for panoramas
depth = panodac.predict("panorama.jpg")

# Disable seam correction if you want raw output
depth_raw = panodac.predict("panorama.jpg", fix_panorama_seam=False)
```

## Models

| Model               | Use Case | Speed | Quality |
| ------------------- | -------- | ----- | ------- |
| `outdoor-resnet101` | Outdoor  | Fast  | Good    |
| `outdoor-swinl`     | Outdoor  | Slow  | Best    |
| `indoor-resnet101`  | Indoor   | Fast  | Good    |
| `indoor-swinl`      | Indoor   | Slow  | Best    |

Models auto-download from HuggingFace on first use (~500MB each).

```python
# Use a specific model
depth = panodac.predict("panorama.jpg", model="outdoor-swinl")

# List available models
print(panodac.list_models())
# ['outdoor-resnet101', 'outdoor-swinl', 'indoor-resnet101', 'indoor-swinl']
```

## Documentation

See [yz3440.github.io/panodac](https://yz3440.github.io/panodac/) for full API reference and examples.

## Credits

Based on **Depth Any Camera (DAC)** by Yuliang Guo et al.

- [Paper](https://arxiv.org/abs/2501.02464)
- [Original Repository](https://github.com/yuliangguo/depth_any_camera)
- [Project Page](https://depth-any-camera.github.io/)

```bibtex
@article{guo2025depthany,
  title={Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera},
  author={Guo, Yuliang and Garg, Sparsh and Ren, Xuan and ElSayed, Mohamed and Guizilini, Vitor},
  journal={CVPR},
  year={2025},
}
```

## License

MIT
