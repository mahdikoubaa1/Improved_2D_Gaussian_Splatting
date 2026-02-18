import json
import sys
sys.path.append("2dGScode")
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
import open3d as o3d
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh, cull_mesh, compute_chamfer_distance

from skimage.morphology import binary_dilation, disk
import argparse

import trimesh
from pathlib import Path
import subprocess

import sys
from tqdm import tqdm



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the mesh.'
    )

    parser.add_argument('--input_mesh', type=str,  help='path to the mesh to be evaluated')
    parser.add_argument('--gt_mesh', type=str,  help='path to the ground truth mesh')
    parser.add_argument('--num_points', default=100000, type=int, help='number of points to sample for chamfer distance computation')
    args = parser.parse_args()

    print("computing chamfer distance ...")
    # load ground truth mesh
    gt_mesh = o3d.io.read_triangle_mesh(args.gt_mesh)
    input_mesh = o3d.io.read_triangle_mesh(args.input_mesh)
    chamfer_dist = compute_chamfer_distance(input_mesh, gt_mesh, n_points=args.num_points)
    results_dir = os.path.join(os.path.dirname(args.input_mesh), "../..", "point_cloud", "iteration_{}".format(30000))
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'metrics.json'), 'r') as f:
        metrics = json.load(f)
    metrics.update({"CD": chamfer_dist})
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)