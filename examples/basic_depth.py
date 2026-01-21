#!/usr/bin/env python3
"""
Basic depth prediction example.

This demonstrates the simplest usage of panodac: predicting depth from a single image.
"""

import panodac
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    # Predict depth - it's this simple!
    depth = panodac.predict("path/to/your/image.jpg")
    
    # depth is a numpy array (H, W) with metric depth in meters
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: {depth.min():.2f}m - {depth.max():.2f}m")
    
    # Visualize the depth map
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(Image.open("path/to/your/image.jpg"))
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Predicted Depth")
    plt.imshow(depth, cmap="turbo")
    plt.colorbar(label="Depth (m)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("depth_result.jpg")
    print("Saved visualization to depth_result.jpg")


def example_with_options():
    """Example showing all available options."""
    
    # List available models
    print("Available models:", panodac.list_models())
    # Output: ['outdoor-resnet101', 'outdoor-swinl', 'indoor-resnet101', 'indoor-swinl']
    
    # Check current device
    print("Using device:", panodac.get_device())
    # Output: cpu, cuda, or mps (depending on your system)
    
    # Predict with a specific model
    depth = panodac.predict(
        "photo.jpg",
        model="outdoor-swinl",  # Higher quality, slower
        device="mps",           # Force specific device
    )
    
    # The predict function accepts multiple input types:
    
    # 1. File path (string or Path)
    depth = panodac.predict("photo.jpg")
    
    # 2. PIL Image
    from PIL import Image
    img = Image.open("photo.jpg")
    depth = panodac.predict(img)
    
    # 3. NumPy array (H, W, 3) RGB
    img_np = np.array(Image.open("photo.jpg"))
    depth = panodac.predict(img_np)


if __name__ == "__main__":
    main()
