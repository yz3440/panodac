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

## Module Structure

```text
panodac/
├── __init__.py       # Top-level API (predict, list_models, get_device)
├── predictor.py      # DepthPredictor class
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
- [Hub](hub.md) - Model download utilities
