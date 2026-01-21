# Predictor

The `DepthPredictor` class provides fine-grained control over depth prediction.

For most use cases, the top-level `panodac.predict()` function is sufficient. Use `DepthPredictor` directly when you need to:

- Reuse a loaded model across multiple predictions
- Access the underlying PyTorch model
- Customize preprocessing behavior

## DepthPredictor

::: panodac.predictor.DepthPredictor
    options:
      show_root_heading: true
      members:
        - __init__
        - __call__

## Usage

```python
from panodac import DepthPredictor

# Create predictor (model loads once)
predictor = DepthPredictor(model="outdoor-swinl", device="cuda")

# Reuse for multiple images
for image_path in image_paths:
    depth = predictor(image_path)
```

The top-level `panodac.predict()` function caches predictors internally, so this is mainly useful when you need direct access to the predictor instance.
