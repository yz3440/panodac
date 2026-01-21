import numpy as np
import pytest


def _make_periodicish_depth(H: int, W: int) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 2.0 * np.pi, W, dtype=np.float32)[None, :]
    depth = 2.0 + 0.5 * np.sin(xx) + 0.2 * np.cos(2.0 * np.pi * yy)
    return depth.astype(np.float32)


def test_fix_panorama_seam_reduces_boundary_discontinuity():
    from panodac.seam_blending import fix_panorama_seam, validate_seam_quality

    H, W = 64, 512
    depth = _make_periodicish_depth(H, W)

    # Inject an artificial seam artifact near the left boundary.
    depth_bad = depth.copy()
    depth_bad[:, :8] += 5.0

    before = validate_seam_quality(depth_bad, depth_bad)["seam_diff_before"]
    depth_fixed = fix_panorama_seam(depth_bad, blend_width=32)
    after = validate_seam_quality(depth_bad, depth_fixed)["seam_diff_after"]

    assert depth_fixed.shape == depth_bad.shape
    assert np.isfinite(depth_fixed).all()
    # Expect a meaningful reduction (not necessarily perfect elimination).
    assert after < before * 0.7


def test_fix_panorama_seam_preserves_dtype_float32():
    from panodac.seam_blending import fix_panorama_seam

    depth = _make_periodicish_depth(32, 256).astype(np.float32)
    out = fix_panorama_seam(depth, blend_width=16)
    assert out.dtype == np.float32


def test_blend_leaves_interior_unchanged_bitwise():
    """
    The solver only overwrites the seam band which maps to the left/right edges
    after rolling back. The interior should remain exactly unchanged.
    """
    from panodac.seam_blending import PoissonSeamBlender

    rng = np.random.default_rng(0)
    H, W = 48, 256
    bw = 16
    depth = rng.normal(size=(H, W)).astype(np.float32)

    out = PoissonSeamBlender(blend_width=bw, anchor_strength=1e-3).blend(depth)

    assert np.array_equal(out[:, bw : W - bw], depth[:, bw : W - bw])


def test_anchor_strength_reduces_band_drift():
    """
    Screened Poisson anchoring should keep the solved band closer to the original
    background values (reducing DC/mean drift and general deviation).
    """
    from panodac.seam_blending import PoissonSeamBlender

    rng = np.random.default_rng(1)
    H, W = 64, 256
    bw = 16
    depth = rng.normal(loc=10.0, scale=3.0, size=(H, W)).astype(np.float32)

    out_no_anchor = PoissonSeamBlender(blend_width=bw, anchor_strength=0.0).blend(depth)
    out_anchor = PoissonSeamBlender(blend_width=bw, anchor_strength=1e-1).blend(depth)

    # Only the edge bands can change. Compare mean absolute change within those bands.
    band_no_anchor = np.concatenate([out_no_anchor[:, :bw], out_no_anchor[:, -bw:]], axis=1)
    band_anchor = np.concatenate([out_anchor[:, :bw], out_anchor[:, -bw:]], axis=1)
    band_src = np.concatenate([depth[:, :bw], depth[:, -bw:]], axis=1)

    mae_no_anchor = float(np.mean(np.abs(band_no_anchor - band_src)))
    mae_anchor = float(np.mean(np.abs(band_anchor - band_src)))

    assert mae_anchor < mae_no_anchor


def test_poisson_blender_rejects_too_large_blend_width():
    from panodac.seam_blending import PoissonSeamBlender

    depth = _make_periodicish_depth(32, 64)
    blender = PoissonSeamBlender(blend_width=32)
    with pytest.raises(ValueError):
        blender.blend(depth)

