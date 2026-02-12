"""
Evaluation Metrics for Gaussian Splatting
Implements PSNR, SSIM, LPIPS, L1, and Chamfer Distance
"""

import torch
import torch.nn.functional as F
import numpy as np
from math import log10
import lpips


class MetricsCalculator:
    """Singleton class for computing evaluation metrics."""
    
    _lpips_fn = None
    
    @classmethod
    def get_lpips_fn(cls):
        """Lazy initialization of LPIPS network."""
        if cls._lpips_fn is None:
            cls._lpips_fn = lpips.LPIPS(net='vgg').cuda()
        return cls._lpips_fn
    
    @staticmethod
    def compute_psnr(pred, gt, mask=None):
        """
        Compute Peak Signal-to-Noise Ratio.
        
        Args:
            pred: [3, H, W] or [H, W, 3] predicted image in [0, 1]
            gt: [3, H, W] or [H, W, 3] ground truth image in [0, 1]
            mask: Optional [H, W] valid region mask
            
        Returns:
            PSNR value in dB (higher is better)
        """
        if pred.dim() == 3 and pred.shape[0] == 3:
            pred = pred.permute(1, 2, 0)
        if gt.dim() == 3 and gt.shape[0] == 3:
            gt = gt.permute(1, 2, 0)
        
        if mask is not None:
            pred = pred[mask]
            gt = gt[mask]
        
        mse = ((pred - gt) ** 2).mean()
        
        if mse < 1e-10:
            return 100.0
        
        return 20 * log10(1.0 / torch.sqrt(mse).item())
    
    @staticmethod
    def compute_ssim(pred, gt, window_size=11):
        """
        Compute Structural Similarity Index.
        
        Args:
            pred: [3, H, W] predicted image in [0, 1]
            gt: [3, H, W] ground truth image in [0, 1]
            
        Returns:
            SSIM value in [0, 1] (higher is better)
        """
        from utils.loss_utils import ssim
        return ssim(pred.unsqueeze(0), gt.unsqueeze(0)).item()
    
    @staticmethod
    def compute_lpips(pred, gt):
        """
        Compute Learned Perceptual Image Patch Similarity.
        
        Args:
            pred: [3, H, W] predicted image in [0, 1]
            gt: [3, H, W] ground truth image in [0, 1]
            
        Returns:
            LPIPS distance in [0, 1] (lower is better)
        """
        lpips_fn = MetricsCalculator.get_lpips_fn()
        
        # Convert from [0, 1] to [-1, 1]
        pred_norm = pred * 2 - 1
        gt_norm = gt * 2 - 1
        
        # Add batch dimension
        pred_norm = pred_norm.unsqueeze(0)
        gt_norm = gt_norm.unsqueeze(0)
        
        with torch.no_grad():
            dist = lpips_fn(pred_norm, gt_norm)
        
        return dist.item()
    
    @staticmethod
    def compute_l1(pred, gt, mask=None):
        """
        Compute L1 (Mean Absolute Error).
        
        Args:
            pred: [3, H, W] predicted image in [0, 1]
            gt: [3, H, W] ground truth image in [0, 1]
            mask: Optional [H, W] valid region mask
            
        Returns:
            L1 error (lower is better)
        """
        if mask is not None:
            mask_3d = mask.unsqueeze(0).expand_as(pred)
            diff = torch.abs(pred - gt) * mask_3d
            return (diff.sum() / mask_3d.sum()).item()
        else:
            return torch.abs(pred - gt).mean().item()
    
    @staticmethod
    def compute_chamfer_distance(pred_xyz, gt_xyz, subsample=10000):
        """
        Compute Chamfer Distance between two point clouds.
        
        Args:
            pred_xyz: [N, 3] predicted point cloud
            gt_xyz: [M, 3] ground truth point cloud
            subsample: Max points to use (for memory efficiency)
            
        Returns:
            Chamfer distance (lower is better)
        """
        # Subsample if too many points
        if pred_xyz.shape[0] > subsample:
            indices = torch.randperm(pred_xyz.shape[0])[:subsample]
            pred_xyz = pred_xyz[indices]
        
        if gt_xyz.shape[0] > subsample:
            indices = torch.randperm(gt_xyz.shape[0])[:subsample]
            gt_xyz = gt_xyz[indices]
        
        # Ensure on GPU
        pred_xyz = pred_xyz.cuda()
        gt_xyz = gt_xyz.cuda()
        
        # Forward: min distance from pred to GT
        dists_pred_to_gt = torch.cdist(pred_xyz, gt_xyz)
        min_dists_forward = dists_pred_to_gt.min(dim=1)[0]
        
        # Backward: min distance from GT to pred
        min_dists_backward = dists_pred_to_gt.min(dim=0)[0]
        
        # Chamfer distance: average of both directions
        chamfer = (min_dists_forward.mean() + min_dists_backward.mean()) / 2
        
        return chamfer.item()


def compute_all_metrics(pred_image, gt_image, pred_xyz=None, gt_xyz=None, mask=None):
    """
    Compute all evaluation metrics.
    
    Args:
        pred_image: [3, H, W] rendered image
        gt_image: [3, H, W] ground truth image
        pred_xyz: Optional [N, 3] predicted point cloud
        gt_xyz: Optional [M, 3] ground truth point cloud
        mask: Optional [H, W] valid region mask
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Image quality metrics
    metrics['psnr'] = MetricsCalculator.compute_psnr(pred_image, gt_image, mask)
    metrics['ssim'] = MetricsCalculator.compute_ssim(pred_image, gt_image)
    metrics['lpips'] = MetricsCalculator.compute_lpips(pred_image, gt_image)
    metrics['l1'] = MetricsCalculator.compute_l1(pred_image, gt_image, mask)
    
    # Geometry metric (if point clouds provided)
    if pred_xyz is not None and gt_xyz is not None:
        metrics['chamfer'] = MetricsCalculator.compute_chamfer_distance(pred_xyz, gt_xyz)
    else:
        metrics['chamfer'] = None
    
    return metrics
