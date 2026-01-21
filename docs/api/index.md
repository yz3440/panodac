# API Reference

## Top-Level API

The main entry points are in the `panodac` module.

::: panodac.predict
    options:
      show_root_heading: true

::: panodac.list_models
    options:
      show_root_heading: true

::: panodac.get_device
    options:
      show_root_heading: true

## Panorama Seam Correction

Panoramic (ERP) depth outputs can show a visible seam at the left/right boundary. `panodac` applies a Poisson-based seam correction by default when a panorama is detected. You can also call the seam blender directly.

::: panodac.seam_blending.fix_panorama_seam
    options:
      show_root_heading: true

## Module Structure

```text
panodac/
├── __init__.py       # Top-level API (predict, list_models, get_device)
├── predictor.py      # DepthPredictor class
├── seam_blending.py  # Poisson seam correction for ERP panoramas
├── hub.py            # HuggingFace model download
├── utils.py          # Device detection, image loading
└── models/           # Neural network architectures
    ├── idisc.py      # IDisc model (perspective)
    ├── idisc_erp.py  # IDiscERP model (panorama)
    ├── encoder.py    # Image encoder
    └── backbones/    # ResNet, Swin Transformer
```

## Submodules

- [Predictor](predictor.md) - `DepthPredictor` class for advanced usage
- [Seam Blending](seam_blending.md) - Poisson seam correction utilities
- [Hub](hub.md) - Model download utilities
