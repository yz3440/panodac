#!/usr/bin/env python3
"""
Panorama depth prediction with point cloud export.

This example shows how to process 360° equirectangular panoramas
and export the result as a 3D point cloud.
"""

import numpy as np
import panodac
from PIL import Image


def main():
    # Path to your panorama image (should have 2:1 aspect ratio)
    pano_path = "../assets/test-pano.jpg"

    # Predict depth
    print("Predicting depth...")
    depth = panodac.predict(pano_path, model="outdoor-resnet101")

    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: {depth.min():.2f}m - {depth.max():.2f}m")

    # Save depth visualization
    save_depth_visualization(depth, "panorama_depth.jpg")

    # Export as point cloud
    print("Generating point cloud...")
    pano_img = np.array(Image.open(pano_path))
    points, colors = erp_to_pointcloud(pano_img, depth)
    save_ply("panorama_pointcloud.ply", points, colors)
    print(f"Saved point cloud with {len(points)} points")


def save_depth_visualization(depth: np.ndarray, output_path: str):
    """Save a colorized depth visualization."""
    import cv2

    # Normalize depth to 0-255 for visualization
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_vis = (depth_norm * 255).astype(np.uint8)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    Image.fromarray(depth_colored).save(output_path)
    print(f"Saved depth visualization to {output_path}")


def erp_to_pointcloud(
    image: np.ndarray,
    depth: np.ndarray,
    max_points: int = 500000,
    min_depth: float = 0.1,
    max_depth: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert equirectangular panorama with depth to 3D point cloud.

    Args:
        image: RGB image (H, W, 3)
        depth: Depth map (H, W) in meters
        max_points: Maximum number of points to generate
        min_depth: Minimum depth threshold in meters
        max_depth: Maximum depth threshold in meters

    Returns:
        points: (N, 3) xyz coordinates
        colors: (N, 3) RGB colors (0-255)
    """
    H, W = depth.shape

    # Create coordinate grids
    u = np.linspace(0, 1, W, dtype=np.float32)
    v = np.linspace(0, 1, H, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    # Convert to spherical coordinates
    # u: 0->1 maps to longitude -π->π
    # v: 0->1 maps to latitude π/2->-π/2
    longitude = (u - 0.5) * 2 * np.pi
    latitude = (0.5 - v) * np.pi

    # Convert to Cartesian coordinates (Y-up convention)
    # x: right, y: up, z: forward
    x = depth * np.cos(latitude) * np.sin(longitude)
    y = depth * np.sin(latitude)
    z = depth * np.cos(latitude) * np.cos(longitude)

    # Flatten all arrays
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    colors = image.reshape(-1, 3)
    depth_flat = depth.ravel()

    # Filter by actual depth (radial distance), not Cartesian z-coordinate
    # This is critical for 360° panoramas where z can be negative (behind viewer)
    valid = (depth_flat > min_depth) & (depth_flat < max_depth)
    points = points[valid]
    colors = colors[valid]

    # Subsample if too many points (after filtering to preserve valid point ratio)
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    return points, colors


def save_ply(filename: str, points: np.ndarray, colors: np.ndarray):
    """Save point cloud to PLY file."""
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")


if __name__ == "__main__":
    main()
