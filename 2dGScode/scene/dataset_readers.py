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

import os
import sys
if not hasattr(sys.stdout, 'isatty'):
    sys.stdout.isatty = lambda: False
from time import sleep
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mono_depth: np.array = None  # Monocular depth prior [H, W]
    mono_normal: np.array = None  # Monocular normal prior [H, W, 3]


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    
    # Check for monocular priors directory
    base_path = os.path.dirname(images_folder)
    mono_depth_dir = os.path.join(base_path, "mono_priors", "mono_depth")
    mono_normal_dir = os.path.join(base_path, "mono_priors", "mono_normal")
    has_mono_priors = os.path.exists(mono_depth_dir) and os.path.exists(mono_normal_dir)
    
    if has_mono_priors:
        print("Found monocular priors directory, loading depth and normal priors...")
    
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        else: #elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
#        else:
#            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # Load monocular priors if available
        mono_depth = None
        mono_normal = None
        if has_mono_priors:
            depth_path = os.path.join(mono_depth_dir, f"{image_name}.npy")
            normal_path = os.path.join(mono_normal_dir, f"{image_name}.npy")
            if os.path.exists(depth_path):
                mono_depth = np.load(depth_path)
            if os.path.exists(normal_path):
                mono_normal = np.load(normal_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              mono_depth=mono_depth, mono_normal=mono_normal)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, colmap_folder="colmap"):
    try:
        cameras_extrinsic_file = os.path.join(path, colmap_folder, "images.bin")
        cameras_intrinsic_file = os.path.join(path, colmap_folder, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, colmap_folder, "images.txt")
        cameras_intrinsic_file = os.path.join(path, colmap_folder, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, colmap_folder, "points3D.ply")
    bin_path = os.path.join(path, colmap_folder, "points3D.bin")
    txt_path = os.path.join(path, colmap_folder, "points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", images_folder=None):
    cam_infos = []
    
    # Determine where to look for images
    if images_folder is not None:
        image_base_path = os.path.join(path, images_folder)
    else:
        image_base_path = path

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        
        # Handle both NeRF Synthetic and Nerfstudio formats
        if "camera_angle_x" in contents:
            # Standard NeRF format
            fovx = contents["camera_angle_x"]
            fovy = None  # Will be computed per image
            use_nerf_format = True
        elif "fl_x" in contents:
            # Nerfstudio format (ScanNet++, etc.)
            import math
            fl_x = contents["fl_x"]
            fl_y = contents["fl_y"]
            w = contents["w"]
            h = contents["h"]
            # Convert focal length to FOV
            fovx = 2 * math.atan(w / (2 * fl_x))
            fovy = 2 * math.atan(h / (2 * fl_y))
            use_nerf_format = False
        else:
            raise ValueError("Unsupported transforms format: missing camera parameters (need either 'camera_angle_x' or 'fl_x')")

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # Handle different file_path formats
            file_path = frame["file_path"]
            # Check if file already has an extension (for Nerfstudio format)
            has_extension = os.path.splitext(file_path)[1] != ''
            if has_extension:
                # File path already includes extension (e.g., "DSC01471.JPG")
                cam_name = file_path
            else:
                # File path needs extension added (e.g., "r_0" -> "r_0.png")
                cam_name = file_path + extension
            
            # Construct full path - use image_base_path which may be a subdirectory
            # For Nerfstudio format, file_path is just the filename (e.g., "DSC01471.JPG")
            image_filename = os.path.basename(cam_name)
            full_image_path = os.path.join(image_base_path, image_filename)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = full_image_path
            image_name = Path(cam_name).stem
            image = Image.open(full_image_path)

            # Handle alpha channel if present
            if image.mode == "RGBA":
                im_data = np.array(image.convert("RGBA"))
                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            elif image.mode != "RGB":
                image = image.convert("RGB")

            # Compute FOV
            if use_nerf_format:
                # Standard NeRF: compute fovy from fovx and image dimensions
                fovy_computed = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy_computed
                FovX = fovx
            else:
                # Nerfstudio: use pre-computed FOV values
                FovX = fovx
                FovY = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos
def readCamerasFromTransforms1(path, transforms_file, white_background, image_subdir, max_samples=1000, frame_entry="frames", test=False,colmap_folder="colmap"):
    cam_infos = []
    colmap_folder= Path(image_subdir).parent / colmap_folder
    with open(os.path.join(path, transforms_file)) as json_file:
        mono_depth_dir = os.path.join(path, "mono_priors", "mono_depth")
        mono_normal_dir = os.path.join(path, "mono_priors", "mono_normal")
        has_mono_priors = os.path.exists(mono_depth_dir) and os.path.exists(mono_normal_dir)

        contents = json.load(json_file)
        fl_x = contents["fl_x"]
        fl_y = contents["fl_y"]
        frames = contents[frame_entry]
        n= len(frames)
        if n> max_samples:
            print(f"Too many frames ({n}), sampling {max_samples} frames for {'testing' if test else 'training'}...")
            indices = np.linspace(0, n - 1, max_samples).astype(int)
            frames = [frames[i] for i in indices]
        try:
            cameras_extrinsic_file = os.path.join(path, colmap_folder, "images.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(path, colmap_folder, "images.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        print("Number of {} frames: {}".format("test" if test else "train", len(frames)))
        frames = [str(i['file_path']) for i in frames]
        print("Frames to read: ", len(frames))

        for idx, key in enumerate(cam_extrinsics):
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
            sys.stdout.flush()
            extr = cam_extrinsics[key]
        
            if not(extr.name in frames):
                continue
            
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

#            else:
#                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            image_path = os.path.join(path,image_subdir, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)
            if image.mode == "RGBA":
                im_data = np.array(image.convert("RGBA"))
                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image.close()  # Close the original image to free resources
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            elif image.mode != "RGB":
                image = image.convert("RGB")
            image.load()  # Force loading the image data into memory
        # Load monocular priors if available
            mono_depth = None
            mono_normal = None
            #if has_mono_priors:
            #    depth_path = os.path.join(mono_depth_dir, f"{image_name}.npy")
            #    normal_path = os.path.join(mono_normal_dir, f"{image_name}.npy")
            #    if os.path.exists(depth_path):
            #        mono_depth = np.load(depth_path)
            #    if os.path.exists(normal_path):
            #        mono_normal = np.load(normal_path)
            fovy = focal2fov(fl_y, image.size[1])
            fovx = focal2fov(fl_x, image.size[0])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], mono_depth=mono_depth, mono_normal=mono_normal))
        sys.stdout.write('\n')
        print(f"Finished reading {len(cam_infos)} cameras for {'test' if test else 'train'} set.")
    return cam_infos
def readNerfSyntheticInfo(args, extension=".png"):
    print("Reading Training Transforms")
    try: 
        train_cam_infos = readCamerasFromTransforms(args.source_path, "transforms_train.json", args.white_background, extension)
    except:
        train_cam_infos = readCamerasFromTransforms1(args.source_path, args.train_transforms_file, args.white_background, args.images, max_samples=args.train_max_samples, frame_entry=args.train_frame_entry, test=False, colmap_folder=args.colmap_folder)
    print("Reading Test Transforms")
    try:
        test_cam_infos = readCamerasFromTransforms(args.source_path, "transforms_test.json", args.white_background, extension)
    except:
        test_cam_infos = readCamerasFromTransforms1(args.source_path, args.test_transforms_file, args.white_background, args.test_images, max_samples=args.test_max_samples, frame_entry=args.test_frame_entry, test=True, colmap_folder=args.colmap_folder)
    
    if not args.eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if os.path.exists(os.path.join(args.source_path, args.colmap_folder)):
        ply_path = os.path.join(args.source_path, args.colmap_folder, "points3D.ply")
        bin_path = os.path.join(args.source_path, args.colmap_folder, "points3D.bin")
        txt_path = os.path.join(args.source_path, args.colmap_folder, "points3D.txt")
    else:
        ply_path = os.path.join(args.source_path, "sparse/0/points3D.ply")
        bin_path = os.path.join(args.source_path, "sparse/0/points3D.bin")
        txt_path = os.path.join(args.source_path, "sparse/0/points3D.txt")

    if not os.path.exists(ply_path) and not os.path.exists(bin_path) and not os.path.exists(txt_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    elif not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    
}