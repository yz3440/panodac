#!/usr/bin/env python3
"""
Batch processing example.

Process multiple images from a directory and save depth maps.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import panodac


def main():
    parser = argparse.ArgumentParser(description="Batch depth prediction")
    parser.add_argument("input_dir", type=str, help="Directory with input images")
    parser.add_argument("output_dir", type=str, help="Directory for output depth maps")
    parser.add_argument(
        "--model", type=str, default="outdoor-resnet101",
        choices=panodac.list_models(),
        help="Model to use"
    )
    parser.add_argument(
        "--format", type=str, default="png",
        choices=["png", "npy", "exr"],
        help="Output format for depth maps"
    )
    args = parser.parse_args()
    
    # Create output directory
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Using model: {args.model}")
    print(f"Using device: {panodac.get_device()}")
    
    # Process images
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            # Predict depth
            depth = panodac.predict(str(img_path), model=args.model)
            
            # Save output
            output_name = img_path.stem + "_depth"
            
            if args.format == "png":
                # Save as 16-bit PNG (depth in mm)
                depth_mm = (depth * 1000).astype(np.uint16)
                output_path = output_dir / f"{output_name}.png"
                cv2.imwrite(str(output_path), depth_mm)
                
            elif args.format == "npy":
                # Save as numpy array (original float values in meters)
                output_path = output_dir / f"{output_name}.npy"
                np.save(output_path, depth)
                
            elif args.format == "exr":
                # Save as OpenEXR (requires opencv-python-headless[contrib])
                output_path = output_dir / f"{output_name}.exr"
                cv2.imwrite(str(output_path), depth.astype(np.float32))
            
            # Also save a visualization
            save_visualization(depth, output_dir / f"{output_name}_vis.jpg")
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
    
    print(f"\nResults saved to {output_dir}")


def save_visualization(depth: np.ndarray, output_path: Path):
    """Save a colorized depth visualization."""
    # Normalize depth
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_vis = (depth_norm * 255).astype(np.uint8)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(output_path), depth_colored)


if __name__ == "__main__":
    main()
