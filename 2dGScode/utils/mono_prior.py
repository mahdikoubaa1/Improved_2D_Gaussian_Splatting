#
# Monocular Depth and Normal Prior Estimation
# For 2D Gaussian Splatting enhancement
#
import math
import sys

import logging

# 1. Force sys.stdout to behave like a terminal-friendly object
if not hasattr(sys.stdout, 'isatty'):
    sys.stdout.isatty = lambda: False
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import os
from utils.graphics_utils import fov2focal
from utils.loss_utils import compute_scale_and_shift
from transformers import pipeline
from PIL import Image

# Global model instances (lazy-loaded)
_depth_model = None
_normal_model = None
_depth_processor = None


def get_depth_model(device="cuda"):
    """
    Load Depth-Anything-V2 model for monocular depth estimation.
    Uses HuggingFace transformers.
    """
    global _depth_model, _depth_processor
    
    if _depth_model is None:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            _depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            _depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device)


            
        except ImportError:
            print("Warning: transformers not installed. Install with: pip install transformers")
    
    return _depth_model, _depth_processor




def get_normal_model(device="cuda"):
    """
    Load Omnidata normal estimation model.
    
    Omnidata models should be downloaded from:
    https://github.com/EPFL-VILAB/omnidata
    
    Place weights in: ./pretrained_models/omnidata_dpt_normal_v2.ckpt
    """
    global _normal_model
    
    if _normal_model is None:
        _normal_model = torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)
    return _normal_model


def estimate_depth(viewpoint_camera, device="cuda"):
    """
    Estimate monocular depth from an image.
    
    Args:
        viewpoint_camera: Camera object containing the original image tensor
        device: torch device
        
    Returns:
        depth: numpy array [H, W] with relative depth values
    """
    model, processor = get_depth_model(device)
    image = viewpoint_camera.original_image.permute(1, 2, 0).to(device)*255
    H, W = image.shape[0], image.shape[1]
    image = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**image)
    # Interpolate to original size
    post_processed_output = processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(H, W)],
    )
    
    depth = post_processed_output[0]['predicted_depth']
    depth = ((depth - depth.min()) / (depth.max() - depth.min()+ 1e-8)).unsqueeze(0)  # Normalize to [0, 1] and add batch dimension
    
    return depth

def get_decay_factor(opt, iteration,start_iteration=4000):
    
    if opt.mono_prior_decay_end > 0:
        decay = math.exp(-(iteration-start_iteration) / (opt.iterations-start_iteration) * 10.)
    else:
        decay = 1.0
    return decay


def estimate_normal(viewpoint_camera, device="cuda"):
    """
    Estimate surface normals from an image.
    
    Args:
        image: PIL Image or numpy array [H, W, 3] (0-255)
        device: torch device
        
    Returns:
        normal: numpy array [H, W, 3] with xyz normal components in [-1, 1]
    """
    model = get_normal_model(device)
    img= viewpoint_camera.original_image.unsqueeze(0).to(device)
    H, W = img.shape[2], img.shape[3]
    fl_x= fov2focal(viewpoint_camera.FoVx, W)
    fl_y= fov2focal(viewpoint_camera.FoVy, H)
    cx = W / 2.0
    cy = H / 2.0
    intrins = torch.tensor([[fl_x, 0.0, cx], [0.0, fl_y, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device="cuda").unsqueeze(0)
    normal = model.infer_tensor(img,intrins)[0]
    normal = -(normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    return normal


