#
# Pure PyTorch implementation of Gaussian relocation for MCMC training.
# Based on "3D Gaussian Splatting as Markov Chain Monte Carlo"
#

import torch
import math

N_max = 51
binoms = None  # Lazy initialization to avoid CUDA issues at import time


def _init_binoms():
    """Initialize binomial coefficients tensor on first use."""
    global binoms
    if binoms is None:
        binoms = torch.zeros((N_max, N_max), dtype=torch.float32, device="cuda")
        for n in range(N_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)


def compute_relocation(opacity_old, scale_old, N):
    """
    Compute new opacity and scale when Gaussians are split/relocated.
    
    When a Gaussian is cloned N times, each copy should have opacity and scale
    adjusted so the total contribution remains approximately constant.
    
    Args:
        opacity_old: (M,) old opacity values (after sigmoid, in [0,1])
        scale_old: (M, 2) old 2D scale values (after exp activation)
        N: (M,) number of copies for each Gaussian
        
    Returns:
        new_opacity: (M,) updated opacity
        new_scale: (M, 2) updated scale
    """
    _init_binoms()
    N = N.clamp(min=1, max=N_max - 1).long()
    
    # Compute new opacity: o_new such that 1 - (1 - o_new)^N ≈ o_old
    # This ensures combined opacity of N copies approximates original
    N_float = N.float()
    new_opacity = 1.0 - torch.pow(1.0 - opacity_old.clamp(min=1e-6, max=1.0 - 1e-6), 1.0 / N_float)
    new_opacity = torch.clamp(new_opacity, min=1e-6, max=1.0 - 1e-6)
    
    # Scale: divide by sqrt(N) for 2D surfels to maintain area coverage
    # (In 3D this would be N^(1/3) for volume preservation)
    scale_factor = torch.sqrt(N_float).unsqueeze(-1)
    new_scale = scale_old / scale_factor
    
    return new_opacity, new_scale
