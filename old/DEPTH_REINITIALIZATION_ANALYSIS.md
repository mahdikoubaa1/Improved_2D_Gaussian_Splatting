# Depth Gaussian Reinitialization Strategy from Mini-Splatting

## Summary
The depth Gaussian reinitialization strategy in Mini-Splatting periodically reinitializes Gaussian splats using depth-rendered 3D points from camera views. This addresses the problem of poorly positioned or redundant Gaussians by sampling new points from areas with low alpha accumulation (under-represented regions).

## Key Files

### 1. **Training Script with Reinitialization Logic**
**File:** `/mini-splatting/ms_d/train.py`

**Key Section:** Lines 144-188

The reinitialization happens every 5000 iterations during the densification phase:

```python
if iteration % 5000 == 0:
    out_pts_list = []
    gt_list = []
    views = scene.getTrainCameras()
    
    # For each training view
    for view in views:
        gt = view.original_image[0:3, :, :]
        
        # Render depth to get 3D points
        render_depth_pkg = render_depth(view, gaussians, pipe, background)
        out_pts = render_depth_pkg["out_pts"]
        accum_alpha = render_depth_pkg["accum_alpha"]

        # Calculate probability based on low alpha accumulation
        # (areas that are not well represented)
        prob = 1 - accum_alpha
        prob = prob / prob.sum()
        prob = prob.reshape(-1).cpu().numpy()

        # Sample based on number of desired points
        factor = 1 / (image.shape[1] * image.shape[2] * len(views) / args.num_depth)
        N_xyz = prob.shape[0]
        num_sampled = int(N_xyz * factor)

        # Sample indices based on probability
        indices = np.random.choice(N_xyz, size=num_sampled, p=prob)
        indices = np.unique(indices)
        
        # Collect sampled points and colors
        out_pts = out_pts.permute(1, 2, 0).reshape(-1, 3)
        gt = gt.permute(1, 2, 0).reshape(-1, 3)

        out_pts_list.append(out_pts[indices])
        gt_list.append(gt[indices])       

    # Merge all sampled points
    out_pts_merged = torch.cat(out_pts_list)
    gt_merged = torch.cat(gt_list)

    # Reinitialize Gaussians with new points
    gaussians.reinitial_pts(out_pts_merged, gt_merged)
    gaussians.training_setup(opt)
    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
    torch.cuda.empty_cache()
    viewpoint_stack = scene.getTrainCameras().copy()
```

**Command Line Argument:**
- `--num_depth`: Controls the target number of points to sample (default: 3,500,000)

### 2. **Gaussian Model Reinitialization Method**
**File:** `/mini-splatting/scene/gaussian_model.py`

**Method:** `GaussianModel.reinitial_pts(pts, rgb)` (Lines 460-483)

```python
def reinitial_pts(self, pts, rgb):
    fused_point_cloud = pts
    fused_color = RGB2SH(rgb)
    
    # Initialize features with spherical harmonics
    features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0

    # Initialize scales based on nearest neighbor distance
    dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    
    # Initialize rotations (identity quaternion)
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1

    # Initialize opacities (0.1)
    opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), 
                                                  dtype=torch.float, device="cuda"))

    # Replace all Gaussian parameters
    self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
    self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
    self._scaling = nn.Parameter(scales.requires_grad_(True))
    self._rotation = nn.Parameter(rots.requires_grad_(True))
    self._opacity = nn.Parameter(opacities.requires_grad_(True))
    self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
```

### 3. **Depth Rendering Function**
**File:** `/mini-splatting/gaussian_renderer/__init__.py`

**Function:** `render_depth(viewpoint_camera, pc, pipe, bg_color, ...)` (Lines 203-281)

This function renders the scene and returns:
- `out_pts`: 3D points in world space [H, W, 3]
- `accum_alpha`: Accumulated alpha values [H, W] (how well each pixel is covered)
- `rendered_depth`: Depth map
- Other diagnostic outputs

### 4. **Custom CUDA Rasterizer**
**File:** `/mini-splatting/submodules/diff-gaussian-rasterization_ms/diff_gaussian_rasterization_ms/__init__.py`

**Method:** `_RasterizeGaussians.render_depth()` (Lines 130-176)

Returns a dictionary with:
```python
res = {
    "render": color,
    "out_pts": out_pts,          # 3D points in world coordinates
    "rendered_depth": depth,
    "discriminants": discriminants,
    "gidx": gidx,
    "accum_alpha": accum_alpha,  # Alpha accumulation for coverage
}
```

## How It Works

### 1. **Depth Rendering**
- Each training view is rendered using a special `render_depth` function
- This outputs 3D world-space points (`out_pts`) for each pixel
- Also outputs accumulated alpha (`accum_alpha`) showing coverage

### 2. **Under-represented Region Detection**
- Probability is calculated as `prob = 1 - accum_alpha`
- High probability = low coverage = needs more Gaussians
- Low probability = high coverage = well represented

### 3. **Importance Sampling**
- Points are sampled using the probability distribution
- Target number controlled by `num_depth` parameter
- Ensures more samples from poorly covered regions

### 4. **Gaussian Reinitialization**
- All existing Gaussians are replaced with new ones
- New positions come from depth-rendered points
- New colors come from ground truth images
- All other properties (scales, rotations, opacities) are reinitialized

### 5. **Timing**
- Happens every 5000 iterations during densification phase
- Also happens once at the end of densification (iteration `densify_until_iter`)

## Key Differences from Standard 3DGS

1. **Periodic reset**: Unlike 3DGS which only adds/splits/clones Gaussians, Mini-Splatting completely replaces them
2. **Depth-guided**: Uses rendered depth to find 3D positions rather than using gradients
3. **Coverage-aware**: Samples more from poorly covered regions
4. **Aggressive**: Throws away all existing Gaussians and starts fresh

## Parameters

- `--num_depth`: Target number of points to sample (default: 3,500,000)
- Reinitialization interval: 5000 iterations (hardcoded)

## Benefits

1. **Prevents Gaussian drift**: Periodic resets prevent Gaussians from drifting to suboptimal positions
2. **Better coverage**: Samples from under-represented areas
3. **Shape quality**: Improves geometric accuracy by using depth information
4. **Compact representation**: Can maintain quality with fewer Gaussians

## Integration Points for 2DGS

To integrate this into 2D Gaussian Splatting, you would need to:

1. **Add depth rendering capability** to your 2DGS rasterizer
2. **Implement the reinitialization logic** in your training loop
3. **Modify the Gaussian model** to support `reinitial_pts()`
4. **Adapt for 2D Gaussians**: The original uses 3D Gaussians with 3D scales. For 2DGS, you'll need to adapt the scale initialization for 2D splats

## Related Files

- `/mini-splatting/ms/train.py`: Similar implementation for Mini-Splatting with simplification
- `/mini-splatting/submodules/diff-gaussian-rasterization_ms/cuda_rasterizer/forward.cu`: CUDA implementation of depth rendering
