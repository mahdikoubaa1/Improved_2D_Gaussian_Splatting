#
# Depth-based Gaussian Reinitialization Utilities
# Implements Mini-Splatting strategy for 2D Gaussian Splatting
#

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


def unproject_depth_to_points(
    depth_map: torch.Tensor,
    camera,
    rgb_image: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Unproject a depth map to 3D world coordinates.
    
    Args:
        depth_map: [H, W] or [1, H, W] depth tensor
        camera: Camera object with intrinsics and world_view_transform
        rgb_image: Optional [3, H, W] for point colors
        
    Returns:
        points: [N, 3] world coordinates
        colors: [N, 3] RGB colors or None
    """
    if depth_map.dim() == 3:
        depth_map = depth_map.squeeze(0)
    
    H, W = depth_map.shape
    device = depth_map.device
    
    # Create pixel grid
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Get camera parameters
    fx = W / (2 * np.tan(camera.FoVx / 2))
    fy = H / (2 * np.tan(camera.FoVy / 2))
    cx = W / 2
    cy = H / 2
    
    # Unproject to camera coordinates
    z = depth_map
    x_cam = (x - cx) * z / fx
    y_cam = (y - cy) * z / fy
    
    # Stack to [H, W, 3]
    points_cam = torch.stack([x_cam, y_cam, z], dim=-1)
    
    # Transform to world coordinates
    # world_view_transform is world-to-camera, we need inverse
    cam_to_world = torch.inverse(camera.world_view_transform)
    
    # Reshape to [H*W, 3]
    points_cam_flat = points_cam.reshape(-1, 3)
    
    # Apply rotation and translation
    # NOTE: world_view_transform is column-major, so translation is in row 3
    R = cam_to_world[:3, :3]
    t = cam_to_world[3, :3]
    
    points_world = points_cam_flat @ R + t
    
    # Get colors if provided
    colors = None
    if rgb_image is not None:
        colors = rgb_image.permute(1, 2, 0).reshape(-1, 3)
    
    return points_world, colors


def sample_points_from_depth(
    depth_map: torch.Tensor,
    camera,
    rgb_image: torch.Tensor,
    num_samples: int = 10000,
    min_depth: float = 0.1,
    max_depth: float = 100.0,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample 3D points from a depth map.
    
    Args:
        depth_map: [H, W] or [1, H, W] rendered depth
        camera: Camera object
        rgb_image: [3, H, W] for colors
        num_samples: Number of points to sample
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
        mask: Optional [H, W] valid region mask
        
    Returns:
        points: [N, 3] sampled world coordinates
        colors: [N, 3] RGB values
    """
    if depth_map.dim() == 3:
        depth_map = depth_map.squeeze(0)
    
    H, W = depth_map.shape
    device = depth_map.device
    
    # Create validity mask
    valid_mask = (depth_map > min_depth) & (depth_map < max_depth) & ~torch.isnan(depth_map)
    if mask is not None:
        valid_mask = valid_mask & mask
    
    valid_indices = torch.where(valid_mask.flatten())[0]
    
    if len(valid_indices) == 0:
        return torch.zeros((0, 3), device=device), torch.zeros((0, 3), device=device)
    
    # Sample indices
    if len(valid_indices) > num_samples:
        perm = torch.randperm(len(valid_indices), device=device)[:num_samples]
        sampled_indices = valid_indices[perm]
    else:
        sampled_indices = valid_indices
    
    # Convert flat indices to 2D
    y_indices = sampled_indices // W
    x_indices = sampled_indices % W
    
    # Get depth and color values
    depth_values = depth_map[y_indices, x_indices]
    rgb_values = rgb_image[:, y_indices, x_indices].T  # [N, 3]
    
    # Unproject sampled points
    fx = W / (2 * np.tan(camera.FoVx / 2))
    fy = H / (2 * np.tan(camera.FoVy / 2))
    cx = W / 2
    cy = H / 2
    
    x_cam = (x_indices.float() - cx) * depth_values / fx
    y_cam = (y_indices.float() - cy) * depth_values / fy
    
    points_cam = torch.stack([x_cam, y_cam, depth_values], dim=-1)
    
    # Transform to world
    # NOTE: world_view_transform is column-major, so translation is in row 3
    cam_to_world = torch.inverse(camera.world_view_transform)
    R = cam_to_world[:3, :3]
    t = cam_to_world[3, :3]
    
    points_world = points_cam @ R + t
    
    return points_world, rgb_values


def aggregate_depth_points(
    cameras: List,
    gaussians,
    pipe,
    background: torch.Tensor,
    render_fn,
    target_total_points: int = 3_500_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate depth-sampled points from multiple camera views using importance sampling.
    
    Implements Mini-Splatting strategy: samples based on 1 - accum_alpha to prioritize
    under-reconstructed regions (floaters, gaps).
    
    Args:
        cameras: List of Camera objects
        gaussians: GaussianModel
        pipe: Pipeline parameters
        background: Background color tensor
        render_fn: Render function
        target_total_points: Target number of total points to sample (default: 3.5M)
        
    Returns:
        all_points: [N, 3] aggregated world coordinates
        all_colors: [N, 3] aggregated RGB colors
    """
    all_points = []
    all_colors = []
    
    num_views = len(cameras)
    
    with torch.no_grad():
        for view_idx, camera in enumerate(cameras):
            # Render depth and alpha
            render_pkg = render_fn(camera, gaussians, pipe, background)
            depth_map = render_pkg.get('surf_depth', render_pkg.get('depth'))
            if depth_map is None:
                print(f"[Warning] No depth map found for camera {view_idx}, skipping")
                continue
            gt_image = camera.original_image.cuda()
            # Get accumulated alpha (opacity)
            accum_alpha = render_pkg.get('rend_alpha', None)
            if accum_alpha is None:
                # Fallback: assume full opacity
                accum_alpha = torch.ones_like(depth_map)

            if accum_alpha.dim() == 3:
                accum_alpha = accum_alpha.squeeze(0)
            if depth_map.dim() == 3:
                depth_map = depth_map.squeeze(0)
            
            H,W = depth_map.shape
            # Importance sampling: prioritize opaque regions (reliable depth)
            prob = accum_alpha  # Higher probability where depth is meaningful
            prob = prob.flatten()
            
            # Filter out invalid depths
            valid_mask = (depth_map > 0.1) & (depth_map < 100.0) & ~torch.isnan(depth_map)
            valid_mask = valid_mask.flatten()
            
            prob = prob * valid_mask.float()
            
            prob_sum = prob.sum()
            if prob_sum < 1e-6:
                continue  # Skip if no valid points
            
            prob = prob / prob_sum  # Normalize to probability distribution
            
            # Adaptive sampling: distribute target points across views
            factor = target_total_points / (H * W * num_views)
            num_samples = int(H * W * factor)
            num_samples = min(num_samples, valid_mask.sum().item())
            
            if num_samples == 0:
                continue
            
            # Probabilistic sampling
            prob_np = prob.cpu().numpy()
            indices = np.random.choice(H * W, size=num_samples, replace=False, p=prob_np)
            indices = np.unique(indices)  # Remove any duplicates
            
            # Convert flat indices to 2D
            y_indices = indices // W
            x_indices = indices % W
            
            # Get depth and color values
            depth_values = depth_map.flatten()[indices]
            rgb_flat = gt_image.permute(1, 2, 0).reshape(-1, 3)
            rgb_values = rgb_flat[indices]
            
            # Unproject to world coordinates
            fx = W / (2 * np.tan(camera.FoVx / 2))
            fy = H / (2 * np.tan(camera.FoVy / 2))
            cx = W / 2
            cy = H / 2
            
            x_indices_t = torch.from_numpy(x_indices).float().to(depth_map.device)
            y_indices_t = torch.from_numpy(y_indices).float().to(depth_map.device)
            
            x_cam = (x_indices_t - cx) * depth_values / fx
            y_cam = (y_indices_t - cy) * depth_values / fy
            
            points_cam = torch.stack([x_cam, y_cam, depth_values], dim=-1)
            
            # Transform to world
            cam_to_world = torch.inverse(camera.world_view_transform)
            R = cam_to_world[:3, :3]
            t = cam_to_world[3, :3]
            
            points_world = points_cam @ R + t
            
            all_points.append(points_world)
            all_colors.append(rgb_values)
    
    if len(all_points) == 0:
        print("[Warning] No valid depth points sampled!")
        return torch.zeros((0, 3), device="cuda"), torch.zeros((0, 3), device="cuda")
    
    # Concatenate
    all_points = torch.cat(all_points, dim=0)
    all_colors = torch.cat(all_colors, dim=0)
    
    print(f"[Depth Sampling] Sampled {all_points.shape[0]} points from {num_views} views")
    
    return all_points, all_colors


def filter_duplicate_points(
    points: torch.Tensor,
    colors: torch.Tensor,
    voxel_size: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove duplicate points using voxel downsampling.
    
    Args:
        points: [N, 3] point coordinates
        colors: [N, 3] point colors
        voxel_size: Size of voxel grid cells
        
    Returns:
        Filtered points and colors
    """
    if len(points) == 0:
        return points, colors
    
    # Quantize to voxel grid
    voxel_indices = torch.floor(points / voxel_size).long()
    
    # Create unique key for each voxel
    mins = voxel_indices.min(dim=0).values
    voxel_indices = voxel_indices - mins
    maxs = voxel_indices.max(dim=0).values + 1
    
    # Flatten to single index
    flat_indices = (
        voxel_indices[:, 0] * maxs[1] * maxs[2] +
        voxel_indices[:, 1] * maxs[2] +
        voxel_indices[:, 2]
    )
    
    # Get unique voxels (keep first occurrence)
    unique_flat, inverse_indices = torch.unique(flat_indices, return_inverse=True)
    
    # For each unique voxel, take the first point
    first_occurrence = torch.zeros(len(unique_flat), dtype=torch.long, device=points.device)
    for i in range(len(inverse_indices) - 1, -1, -1):
        first_occurrence[inverse_indices[i]] = i
    
    filtered_points = points[first_occurrence]
    filtered_colors = colors[first_occurrence]
    
    return filtered_points, filtered_colors
