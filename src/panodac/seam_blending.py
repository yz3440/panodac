"""
Poisson blending utilities for ERP panorama seam correction.

This module fixes left-right boundary discontinuities in equirectangular (ERP)
depth maps produced by planar CNN models.

Core idea:
1) roll the depth map by W//2 so the problematic seam moves to the image center
2) solve a Poisson equation in a narrow vertical band around the center seam
3) roll back to restore original alignment with a seamless boundary
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _as_float32_depth(depth: np.ndarray) -> np.ndarray:
    if not isinstance(depth, np.ndarray):
        raise TypeError(f"depth must be a numpy array, got {type(depth)}")
    if depth.ndim != 2:
        raise ValueError(f"depth must have shape (H, W), got {depth.shape}")
    if depth.dtype == np.float32:
        out = depth
    else:
        out = depth.astype(np.float32, copy=False)

    if not np.isfinite(out).all():
        raise ValueError("depth contains NaN or Inf values")
    return out


def _recommended_blend_width(width: int) -> int:
    # Rule-of-thumb from brief: W/32 .. W/16
    return max(16, min(128, width // 32))


def _build_laplacian_matrix(height: int, region_width: int) -> sparse.csr_matrix:
    """
    Build sparse Laplacian for an (H x region_width) unknown region using a 5-point stencil.
    Dirichlet boundary handling is done by adding known neighbor values to the RHS.
    """
    n = height * region_width

    # 5-point stencil (unknown region only). Default interior diagonal is 4.
    # For the top/bottom image boundaries we use zero-Neumann (replicated neighbor),
    # which reduces the effective neighbor count by 1 (missing vertical neighbor is self):
    # 4*f - (self + other_neighbors) = ...  =>  3*f - other_neighbors = ...
    main_diag = 4.0 * np.ones(n, dtype=np.float32)
    if height >= 2:
        main_diag[:region_width] -= 1.0  # top row
        main_diag[-region_width:] -= 1.0  # bottom row

    # Horizontal neighbors (within rows only)
    side_diag = -1.0 * np.ones(n - 1, dtype=np.float32)
    # Remove connections across row boundaries
    side_diag[region_width - 1 :: region_width] = 0.0

    # Vertical neighbors (between rows)
    vert_diag = -1.0 * np.ones(n - region_width, dtype=np.float32)

    A = sparse.diags(
        diagonals=[main_diag, side_diag, side_diag, vert_diag, vert_diag],
        offsets=[0, 1, -1, region_width, -region_width],
        format="csr",
        dtype=np.float32,
    )
    return A


def _compute_guidance_gradients(
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    left: int,
    right: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a mixed gradient field (guidance) by blending gradients from depth_a and depth_b
    with a smooth ramp across [left, right).
    """
    H, W = depth_a.shape
    region_w = right - left
    if region_w <= 0:
        raise ValueError("Invalid blend region width")

    # Horizontal gradients: use circular diff to respect ERP wrap (only really matters near edges).
    gx_a = np.diff(depth_a, axis=1, append=depth_a[:, :1])
    gx_b = np.diff(depth_b, axis=1, append=depth_b[:, :1])

    # Vertical gradients: non-circular (poles are not periodic). Use forward diff with edge replication.
    gy_a = np.diff(depth_a, axis=0, append=depth_a[-1:, :])
    gy_b = np.diff(depth_b, axis=0, append=depth_b[-1:, :])

    # Weight ramp: 0 -> 1 across the band; constant 0 left of band, 1 right of band.
    weights = np.zeros((H, W), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, num=region_w, dtype=np.float32)
    weights[:, left:right] = ramp[np.newaxis, :]
    weights[:, right:] = 1.0

    gx = (1.0 - weights) * gx_a + weights * gx_b
    gy = (1.0 - weights) * gy_a + weights * gy_b
    return gx, gy


def _divergence(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    Compute divergence of a guidance field.
    div(g) = d(gx)/dx + d(gy)/dy using backward differences.
    """
    H, W = gx.shape

    dgx_dx = np.zeros((H, W), dtype=np.float32)
    dgx_dx[:, 1:] = gx[:, 1:] - gx[:, :-1]

    dgy_dy = np.zeros((H, W), dtype=np.float32)
    dgy_dy[1:, :] = gy[1:, :] - gy[:-1, :]

    return dgx_dx + dgy_dy


def _solve_poisson_band(
    depth_bg: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    left: int,
    right: int,
    A: sparse.csr_matrix | None = None,
    *,
    anchor_strength: float = 0.0,
) -> np.ndarray:
    """
    Solve Poisson equation in the vertical band [left, right) across full image height.

    Unknowns: depth values within the band. Outside the band, depth_bg provides Dirichlet values.
    """
    H, W = depth_bg.shape
    region_w = right - left
    if A is None:
        A = _build_laplacian_matrix(H, region_w)

    div = _divergence(gx, gy)
    b = div[:, left:right].reshape(-1).astype(np.float32, copy=False)

    # Add Dirichlet boundary contributions for neighbors outside the band:
    #
    # For each unknown pixel p, Laplacian stencil is:
    # 4*f(p) - sum(f(neighbors)) = div(g)(p)
    # If a neighbor is outside the band, its value is known from depth_bg, so move it to RHS:
    # b += known_neighbor_value
    for y in range(H):
        row0 = y * region_w
        row_last = row0 + (region_w - 1)
        b[row0] += depth_bg[y, left - 1]  # left neighbor outside band
        b[row_last] += depth_bg[y, right]  # right neighbor outside band

    # Top/bottom edges: handled by reducing Laplacian diagonal in _build_laplacian_matrix()
    # (zero-Neumann via replicated neighbor). Do NOT add any absolute-value terms to b here.

    # Screened Poisson anchoring to prevent DC (additive constant) drift:
    # (Δ + λI)f = div(v) + λ f0, where f0 is the background band.
    lam = float(anchor_strength)
    if lam < 0.0:
        raise ValueError("anchor_strength must be >= 0")
    if lam > 0.0:
        n = H * region_w
        A = A + (lam * sparse.identity(n, format="csr", dtype=np.float32))
        b = b + (lam * depth_bg[:, left:right].reshape(-1).astype(np.float32, copy=False))

    sol = spsolve(A, b).astype(np.float32, copy=False)
    return sol.reshape(H, region_w)


@dataclass
class PoissonSeamBlender:
    """
    Poisson blending seam fixer for ERP panoramic depth maps.

    Args:
        blend_width: Half-width (in pixels) of the solve band around the seam after rolling.
                    Total solved band width = 2 * blend_width.
    """

    blend_width: int = 32
    anchor_strength: float = 1e-3

    # Cache Laplacian matrix for reuse (same H and region_w)
    _cached_shape: tuple[int, int] | None = None
    _cached_A: sparse.csr_matrix | None = None

    def blend(self, depth: np.ndarray) -> np.ndarray:
        depth = _as_float32_depth(depth)
        H, W = depth.shape
        bw = int(self.blend_width)
        if bw <= 0:
            raise ValueError("blend_width must be > 0")
        if W < 4 * bw:
            raise ValueError(f"Image too narrow for blend_width={bw} (W={W})")

        # 1) Roll to move seam to center
        shift = W // 2
        depth_rolled = np.roll(depth, shift=shift, axis=1)
        center = W // 2
        left = center - bw
        right = center + bw

        # Need valid external neighbors for Dirichlet boundaries
        if left <= 0 or right >= W - 1:
            raise ValueError(
                f"blend_width={bw} too large for width={W} (requires 1px margins)"
            )

        # 2) Mixed guidance gradients
        gx, gy = _compute_guidance_gradients(depth, depth_rolled, left=left, right=right)

        # 3) Solve Poisson equation within the band
        region_w = right - left
        if self._cached_shape != (H, region_w) or self._cached_A is None:
            self._cached_A = _build_laplacian_matrix(H, region_w)
            self._cached_shape = (H, region_w)

        solved_band = _solve_poisson_band(
            depth_bg=depth_rolled,
            gx=gx,
            gy=gy,
            left=left,
            right=right,
            A=self._cached_A,
            anchor_strength=self.anchor_strength,
        )

        # 4) Reconstruct and roll back
        out_rolled = depth_rolled.copy()
        out_rolled[:, left:right] = solved_band
        out = np.roll(out_rolled, shift=-shift, axis=1)
        return out


def fix_panorama_seam(
    depth: np.ndarray,
    blend_width: int | None = None,
    *,
    anchor_strength: float = 1e-3,
) -> np.ndarray:
    """
    Convenience wrapper to fix ERP panorama depth seam artifacts.

    Args:
        depth: (H, W) depth map.
        blend_width: Half-width of the solve band. If None, auto-pick from width.
    """
    depth_f = _as_float32_depth(depth)
    H, W = depth_f.shape
    bw = _recommended_blend_width(W) if blend_width is None else int(blend_width)
    blender = PoissonSeamBlender(blend_width=bw, anchor_strength=float(anchor_strength))
    return blender.blend(depth_f)


def validate_seam_quality(depth_original: np.ndarray, depth_fixed: np.ndarray) -> dict[str, float]:
    """
    Simple seam metrics for validation.

    Returns:
        dict with seam discontinuity (mean abs diff between col0 and col-1) before/after.
    """
    d0 = _as_float32_depth(depth_original)
    d1 = _as_float32_depth(depth_fixed)
    if d0.shape != d1.shape:
        raise ValueError("depth_original and depth_fixed must have same shape")

    seam_diff_before = float(np.abs(d0[:, 0] - d0[:, -1]).mean())
    seam_diff_after = float(np.abs(d1[:, 0] - d1[:, -1]).mean())

    return {
        "seam_diff_before": seam_diff_before,
        "seam_diff_after": seam_diff_after,
        "improvement_pct": (1.0 - seam_diff_after / max(1e-12, seam_diff_before)) * 100.0,
    }

