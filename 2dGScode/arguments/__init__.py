#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list:
                    # Handle list arguments - allow multiple values
                    group.add_argument("--" + key, default=value, nargs='+', type=int)
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.source_path = ""
        self.test_transforms_file = "transforms_train.json"
        self.train_transforms_file = "transforms_train.json"
        self.use_exposure_optimization = False
        self.use_colmap = False
        self.colmap_folder = "colmap"
        self.test_frame_entry="test_frames"
        self.train_frame_entry="frames"
        self.model_path = ""
        self.train_max_samples = 1000
        self.test_max_samples = 200
        self.images = "images"
        self.test_images = "images"
        self.resolution = -1.0
        self.white_background = False
        self.cap_max = -1  # Maximum number of Gaussians (required for MCMC)
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.001
        self.exposure_lr_final = 0.0001
        self.exposure_lr_delay_steps = 5000
        self.exposure_lr_delay_mult = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 10
        self.lambda_normal = 0.05
        self.opacity_cull = 0.05
        self.use_exposure_optimization = False

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        
        # Monocular prior losses (Task 4) - MonoSDF style
        self.lambda_mono_depth = 0.0  # Weight for monocular depth loss (MonoSDF default)
        self.lambda_mono_normal_l1 = 0.0  # Weight for normal L1 loss (MonoSDF default)
        self.lambda_mono_normal_cos = 0.0  # Weight for normal cosine loss (MonoSDF default)
        self.mono_prior_decay_end = 30000  # End step for exponential decay (MonoSDF uses decay)
        
        # Depth reinitialization (Task 2) - Mini-Splatting strategy
        self.depth_reinit_every = -1   # frequency of depth-based reinitialization (e.g., every 5000 iterations)
        self.reinit_target_points = 550000 # Target total points for reinitialization (3.5M default for m360)
        
        # MCMC parameters
        self.noise_lr = 5e5        # SGLD noise learning rate
        self.scale_reg = 0.01      # L1 regularization on scale
        self.opacity_reg = 0.01    # L1 regularization on opacity
        self.mcmc = False          # Toggle for MCMC training
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
