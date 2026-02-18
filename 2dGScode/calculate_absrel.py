import json
import cv2
import torch
import trimesh
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh, setup_renderer, get_depth_map
from utils.render_utils import generate_path, create_videos
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graphics_utils import fov2focal
import cv2
import numpy as np
import os
import glob
from skimage.morphology import dilation, disk
import argparse
from utils.mono_prior import estimate_depth, estimate_normal
import trimesh
from pathlib import Path
import subprocess
from utils.loss_utils import ScaleAndShiftInvariantLoss, get_normal_loss

import sys
from tqdm import tqdm
import open3d as o3d
import open3d.core as o3c

import torch
import pickle





if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    parser.add_argument('--masks', type=str, help='path to the masks for culling')

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    normal_predictor = torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)

# Convert legacy meshes to Tensor meshes
    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
#
    ##t_mesh2 = trimesh.load(os.path.join(args.source_path, '../scans', 'mesh_aligned_0.05.ply'))

    #
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    denom=0
    absrel=0.0
    geo_path= os.path.join(dataset.source_path, pipe.geo_name)
    renderer = None
    for viewpoint_camera in tqdm(scene.getTestCameras(), desc="Testing progress"):
        sampled_masks = []
        with torch.inference_mode():
            render_pkg = render(viewpoint_camera, gaussians, pipe, background)
            if renderer is None:
                renderer = setup_renderer(geo_path, viewpoint_camera.image_width, viewpoint_camera.image_height,geotype=pipe.geo_type)
            
            gt_depth = get_depth_map(renderer, viewpoint_camera, device="cuda")
            rendered_depth = render_pkg["surf_depth"]
            

            mask = gt_depth!= float('inf')
            gt_depth = gt_depth[mask]
            rendered_depth = rendered_depth[mask]


            absrel+=torch.sum(torch.abs(gt_depth - rendered_depth) / gt_depth).item()
            
            denom += torch.sum(mask).item()
    absrel /= denom
    results_dir = os.path.join(args.model_path, "point_cloud", "iteration_{}".format(iteration))
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'metrics.json'), 'r') as f:
        metrics = json.load(f)
    metrics.update({"DepthAbsRel": absrel})
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    print (f"Overall AbsRel Error: {absrel:.4f}")

    #print(f"Overall AbsRel Error: {absrel:.4f}")
    # Load a .pt file

# Sample to point clouds

# Calculate mean distances


# Use the compute_metrics tool
# This returns a dictionary containing Chamfer, Hausdorff, etc.

    
    
    #bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    #background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    #
    #train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    #test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    #gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    
    #
    #if not args.skip_train:
    #    print("export training images ...")
    #    os.makedirs(train_dir, exist_ok=True)
    #    gaussExtractor.reconstruction(scene.getTrainCameras())
    #    gaussExtractor.export_image(train_dir)
    #    
    #
    #if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
    #    print("export rendered testing images ...")
    #    os.makedirs(test_dir, exist_ok=True)
    #    gaussExtractor.reconstruction(scene.getTestCameras())
    #    gaussExtractor.export_image(test_dir)
    #
    #
    #if args.render_path:
    #    print("render videos ...")
    #    traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
    #    os.makedirs(traj_dir, exist_ok=True)
    #    n_fames = 240
    #    cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
    #    gaussExtractor.reconstruction(cam_traj)
    #    gaussExtractor.export_image(traj_dir)
    #    create_videos(base_dir=traj_dir,
    #                input_dir=traj_dir, 
    #                out_name='render_traj', 
    #                num_frames=n_fames)
#
    #if not args.skip_mesh:
    #    print("export mesh ...")
    #    os.makedirs(train_dir, exist_ok=True)
    #    # set the active_sh to 0 to export only diffuse texture
    #    gaussExtractor.gaussians.active_sh_degree = 0
    #    gaussExtractor.reconstruction(scene.getTrainCameras())
    #    # extract the mesh and save
    #    if args.unbounded:
    #        name = 'fuse_unbounded.ply'
    #        mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
    #    else:
    #        name = 'fuse.ply'
    #        depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
    #        voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
    #        sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
    #        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
    #    print("mesh extracted")
    #    o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
    #    print("mesh saved at {}".format(os.path.join(train_dir, name)))
    #    # post-process the mesh and save, saving the largest N clusters
    #    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    #    o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
    #    print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))