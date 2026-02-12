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
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
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

    t_mesh2 = trimesh.load(os.path.join(args.source_path, '../scans', 'mesh_aligned_0.05.ply'))
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    depth_loss= ScaleAndShiftInvariantLoss(alpha=0.5, scales=1, reduction='image-based')

    for viewpoint_camera in tqdm(scene.getTestCameras(), desc="Culling progress"):
        sampled_masks = []
        with torch.inference_mode():
            render_pkg = render(viewpoint_camera, gaussians, pipe, background)
            #depth_rendered = render_pkg["surf_depth"]
            #depth_rendered = 1/depth_rendered
            #depth_rendered = (depth_rendered - depth_rendered.min()) / (depth_rendered.max() - depth_rendered.min())
            rendered_normal = render_pkg["rend_normal"]
            #print(depth_rendered.shape)
            #torchvision.utils.save_image(depth_rendered, f"render_{viewpoint_camera.image_name}.png")
            normal= estimate_normal (viewpoint_camera)
            #normal = (normal + 1.0) / 2.0
            #depth = estimate_depth(viewpoint_camera).unsqueeze(0)
            
            #loss = depth_loss(depth_rendered, depth, mask= torch.ones_like(depth_rendered))
            
            normal_l1, normal_cos = get_normal_loss(rendered_normal.permute(1, 2, 0), normal.permute(1, 2, 0))
            #print("Depth loss: ", loss.item())
            print("Normal L1 loss: ", normal_l1.item())
            print("Normal Cosine loss: ", normal_cos.item())
            #torchvision.utils.save_image(depth, f"render_{viewpoint_camera.image_name}_depth.png")
        
            

            
            
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