# Seam Blending

ERP panoramas wrap horizontally, so the **left and right image boundaries are adjacent in 3D**. Standard CNN padding can cause a visible seam in predicted depth at that wrap boundary.

`panodac` includes a Poisson-based seam correction that operates in the **gradient domain** and solves a narrow band around the seam.

## Usage

### Use the built-in panorama seam correction (default)

```python
import panodac

# Enabled by default for panoramas
depth = panodac.predict("panorama.jpg")

# Disable if you want raw output
depth_raw = panodac.predict("panorama.jpg", fix_panorama_seam=False)
```

### Apply seam correction as a post-process

```python
from panodac.seam_blending import fix_panorama_seam

depth_fixed = fix_panorama_seam(depth_raw, blend_width=32)
```

### Tuning: prevent DC/mean drift with anchoring

Poisson blending can drift in absolute scale (DC offset). The seam solver uses a **screened Poisson** anchoring term controlled by `anchor_strength`:

- Higher `anchor_strength` → less drift / less change to the original depth in the seam band
- Lower `anchor_strength` → more freedom to match gradients (can improve seam smoothness)

```python
from panodac.seam_blending import fix_panorama_seam

depth_fixed = fix_panorama_seam(
    depth_raw,
    blend_width=32,
    anchor_strength=1e-3,  # default
)
```

## API

::: panodac.seam_blending.fix_panorama_seam
    options:
      show_root_heading: true

::: panodac.seam_blending.PoissonSeamBlender
    options:
      show_root_heading: true
