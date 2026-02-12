#!/usr/bin/env python3
"""
Precompute monocular depth and normal priors for a dataset.

This script processes all images in a dataset directory and generates
depth and normal priors that will be used during 2D Gaussian Splatting training.

Usage:
    python precompute_priors.py --source_path <dataset_path> [--images <images_folder>]
    
Example:
    python precompute_priors.py --source_path ./data/scene1 --images images

The script will create:
    <source_path>/mono_priors/mono_depth/<image_name>.npy
    <source_path>/mono_priors/mono_normal/<image_name>.npy
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.mono_prior import estimate_depth, estimate_normal


def find_images(images_dir):
    """Find all image files in directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    images = []
    
    for file in sorted(os.listdir(images_dir)):
        if Path(file).suffix.lower() in image_extensions:
            images.append(os.path.join(images_dir, file))
    
    return images


def main():
    parser = argparse.ArgumentParser(description="Precompute monocular depth and normal priors")
    parser.add_argument("--source_path", "-s", required=True, type=str,
                        help="Path to dataset directory")
    parser.add_argument("--images", "-i", default="images", type=str,
                        help="Name of images subdirectory (default: 'images')")
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device to use for inference (default: 'cuda')")
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation even if priors exist")
    parser.add_argument("--depth_only", action="store_true",
                        help="Only compute depth priors")
    parser.add_argument("--normal_only", action="store_true",
                        help="Only compute normal priors")
    
    args = parser.parse_args()
    
    # Resolve paths
    source_path = Path(args.source_path).resolve()
    images_dir = source_path / args.images
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)
    
    # Create output directories
    output_dir = source_path / "mono_priors"
    depth_dir = output_dir / "mono_depth"
    normal_dir = output_dir / "mono_normal"
    
    if not args.normal_only:
        depth_dir.mkdir(parents=True, exist_ok=True)
    if not args.depth_only:
        normal_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_paths = find_images(images_dir)
    print(f"Found {len(image_paths)} images in {images_dir}")
    
    if len(image_paths) == 0:
        print("No images found!")
        sys.exit(1)
    
    # Process images
    print(f"\nComputing monocular priors...")
    print(f"Device: {args.device}")
    print(f"Output directory: {output_dir}")
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        img_name = Path(img_path).stem
        
        depth_path = depth_dir / f"{img_name}.npy"
        normal_path = normal_dir / f"{img_name}.npy"
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        scale=image.width/1600.
        image = image.resize((int(image.width/scale), int(image.height/scale)))  # Resize for faster processing
        # Compute depth
        if not args.normal_only:
            if not depth_path.exists() or args.force:
                try:
                    depth = estimate_depth(image, device=args.device)
                    np.save(depth_path, depth.cpu().numpy().astype(np.float32))
                except Exception as e:
                    print(f"\nError computing depth for {img_name}: {e}")
                    continue
        
        # Compute normal
        if not args.depth_only:
            if not normal_path.exists() or args.force:
                try:
                    normal = estimate_normal(image, device=args.device)
                    np.save(normal_path, normal.cpu().numpy().astype(np.float32))
                except Exception as e:
                    print(f"\nError computing normal for {img_name}: {e}")
                    continue
    
    print(f"\nDone! Priors saved to: {output_dir}")
    print("\nTo use priors during training, run:")
    print(f"  python train.py -s {source_path} --lambda_mono_depth 0.1 --lambda_mono_normal 0.05")


if __name__ == "__main__":
    main()
