#!/usr/bin/env python3
"""
Panorama depth prediction: compare seam correction on/off.

This example runs the model once (without seam correction), then applies the
Poisson seam blender as a post-process to produce a blended result. It saves:

- panorama_depth_no_blend.jpg
- panorama_depth_blend.jpg

and prints simple seam metrics.
"""

import numpy as np
import panodac
from PIL import Image


def main():
    # Path to your panorama image (should have 2:1 aspect ratio)
    pano_path = "../assets/test-pano.jpg"

    print("Predicting raw depth (no blending)...")
    depth_raw = panodac.predict(
        pano_path,
        model="outdoor-resnet101",
        fix_panorama_seam=False,
    )

    print(f"Raw depth shape: {depth_raw.shape}")
    print(f"Raw depth range: {depth_raw.min():.2f}m - {depth_raw.max():.2f}m")

    print("Applying Poisson seam blending...")
    from panodac.seam_blending import fix_panorama_seam, validate_seam_quality

    H, W = depth_raw.shape
    blend_width = max(8, W // 32)
    depth_blend = fix_panorama_seam(depth_raw, blend_width=blend_width)

    metrics = validate_seam_quality(depth_raw, depth_blend)
    print(
        "Seam discontinuity (mean |col0-colLast|): "
        f"{metrics['seam_diff_before']:.4f} -> {metrics['seam_diff_after']:.4f} "
        f"({metrics['improvement_pct']:.1f}%)"
    )

    save_depth_visualization(depth_raw, "panorama_depth_no_blend.jpg")
    save_depth_visualization(depth_blend, "panorama_depth_blend.jpg")


def save_depth_visualization(depth: np.ndarray, output_path: str):
    """Save a colorized depth visualization."""
    import cv2

    depth = depth.astype(np.float32, copy=False)
    dmin, dmax = float(depth.min()), float(depth.max())
    denom = max(1e-6, dmax - dmin)

    # Normalize depth to 0-255 for visualization
    depth_norm = (depth - dmin) / denom
    depth_vis = (depth_norm * 255).astype(np.uint8)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    Image.fromarray(depth_colored).save(output_path)
    print(f"Saved depth visualization to {output_path}")


if __name__ == "__main__":
    main()
