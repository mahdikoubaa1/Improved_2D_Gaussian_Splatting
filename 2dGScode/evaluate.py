"""
Evaluation script for 2D Gaussian Splatting models.
Computes PSNR, SSIM, LPIPS, L1, and Chamfer Distance metrics.

Usage:
    python evaluate.py -m output/model1 output/model2 \\
                      --names "Baseline" "Improved" \\
                      -s /path/to/dataset \\
                      --output results.csv
"""

import torch
import os
import sys
from pathlib import Path
from tqdm import tqdm
import csv
import argparse
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from utils.metrics import compute_all_metrics


def load_model(model_path, iteration=-1):
    """Load a trained Gaussian Splatting model."""
    # Create dummy args
    class DummyArgs:
        def __init__(self, model_path):
            self.model_path = model_path
            self.source_path = model_path  # Will be updated from cfg_args
            self.sh_degree = 3
            self.images = "images"
            self.resolution = -1
            self.white_background = False
            self.data_device = "cuda"
            self.eval = True
            self.render_items = ['RGB']
    
    # Load config args if available
    cfg_path = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            cfg_str = f.read()
            cfg_args = eval(cfg_str)
            # Extract source path from config
            args = DummyArgs(model_path)
            args.source_path = cfg_args.source_path
            args.images = getattr(cfg_args, 'images', 'images')
    else:
        args = DummyArgs(model_path)
    
    # Load scene and gaussians
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)
    
    return scene, gaussians


def evaluate_model(model_path, model_name, dataset_path=None, iteration=-1):
    """
    Evaluate a single model on test views.
    
    Returns:
        Dictionary of average metrics across all test views
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}\n")
    
    # Load model
    scene, gaussians = load_model(model_path, iteration)
    
    # Get test cameras
    test_cameras = scene.getTestCameras()
    
    if len(test_cameras) == 0:
        print(f"Warning: No test cameras found for {model_name}")
        return None
    
    print(f"Found {len(test_cameras)} test views")
    
    # Setup rendering
    bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    
    class PipeArgs:
        convert_SHs_python = False
        compute_cov3D_python = False
        depth_ratio = 0.0
        debug = False
    
    pipe = PipeArgs()
    
    # Accumulate metrics
    all_metrics = {
        'psnr': [],
        'ssim': [],
        'lpips': [],
        'l1': [],
        'chamfer': []
    }
    
    # Evaluate on each test view
    for idx, camera in enumerate(tqdm(test_cameras, desc="Rendering test views")):
        with torch.no_grad():
            # Render
            render_pkg = render(camera, gaussians, pipe, bg_color)
            pred_image = render_pkg["render"]  # [3, H, W]
            
            # Ground truth
            gt_image = camera.original_image.cuda()  # [3, H, W]
            
            # Compute metrics
            metrics = compute_all_metrics(pred_image, gt_image)
            
            # Accumulate
            for key in all_metrics.keys():
                if metrics[key] is not None:
                    all_metrics[key].append(metrics[key])
    
    # Average metrics
    avg_metrics = {}
    for key, values in all_metrics.items():
        if len(values) > 0:
            avg_metrics[key] = np.mean(values)
        else:
            avg_metrics[key] = None
    
    # Print results
    print(f"\nResults for {model_name}:")
    print(f"  PSNR:    {avg_metrics['psnr']:.2f} dB")
    print(f"  SSIM:    {avg_metrics['ssim']:.4f}")
    print(f"  LPIPS:   {avg_metrics['lpips']:.4f}")
    print(f"  L1:      {avg_metrics['l1']:.4f}")
    if avg_metrics['chamfer'] is not None:
        print(f"  Chamfer: {avg_metrics['chamfer']:.6f}")
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gaussian Splatting models")
    parser.add_argument('-m', '--models', nargs='+', required=True,
                       help='Paths to model output directories')
    parser.add_argument('--names', nargs='+',
                       help='Names for each model (optional)')
    parser.add_argument('-s', '--source', type=str,
                       help='Dataset source path (optional, will use from cfg_args if available)')
    parser.add_argument('--iteration', type=int, default=-1,
                       help='Iteration to load (-1 for latest)')
    parser.add_argument('--output', type=str, default='evaluation_results.csv',
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.names and len(args.names) != len(args.models):
        print("Error: Number of names must match number of models")
        return
    
    # Use default names if not provided
    if not args.names:
        args.names = [f"Model{i+1}" for i in range(len(args.models))]
    
    # Evaluate each model
    results = []
    for model_path, model_name in zip(args.models, args.names):
        if not os.path.exists(model_path):
            print(f"Warning: Model path not found: {model_path}")
            continue
        
        metrics = evaluate_model(model_path, model_name, args.source, args.iteration)
        
        if metrics is not None:
            results.append({
                'name': model_name,
                **metrics
            })
    
    # Write results to CSV
    if results:
        print(f"\n{'='*60}")
        print(f"Writing results to: {args.output}")
        print(f"{'='*60}\n")
        
        with open(args.output, 'w', newline='') as f:
            fieldnames = ['name', 'psnr', 'ssim', 'lpips', 'l1', 'chamfer']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        # Also print as table
        print("\n" + "="*80)
        print(f"{'Model':<20} {'PSNR ↑':<10} {'SSIM ↑':<10} {'LPIPS ↓':<10} {'L1 ↓':<10} {'Chamfer ↓':<12}")
        print("="*80)
        for result in results:
            chamfer_str = f"{result['chamfer']:.6f}" if result['chamfer'] is not None else "N/A"
            print(f"{result['name']:<20} {result['psnr']:<10.2f} {result['ssim']:<10.4f} "
                  f"{result['lpips']:<10.4f} {result['l1']:<10.4f} {chamfer_str:<12}")
        print("="*80)
        print("↑ = higher is better, ↓ = lower is better")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
