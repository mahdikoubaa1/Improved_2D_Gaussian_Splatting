from glob import glob
import os
import shutil
from pathlib import Path
import re
import numpy as np
import open3d as o3d
from scipy.io import loadmat


# Source and destination directories
source_dir = "/cluster/51/koubaa/data/DTU_Official/Points/stl/"
dest_base_dir = "/cluster/51/koubaa/data/DTU/"

# Find all files matching pattern stl[NUMBER]_total.ply
for filename in os.listdir(source_dir):
    match = re.match(r'stl(\d+)_total\.ply', filename)
    if match:
        number = match.group(1)
        number = str(int(match.group(1)))

        source_file = os.path.join(source_dir, filename)
        dest_dir = os.path.join(dest_base_dir, f"scan{number}")
        dest_file = os.path.join(dest_dir, "pcd.ply")
        print(f"Copying {source_file} to {dest_file}")
        # Create destination directory if it doesn't exist
        if not os.path.exists(dest_dir):
            continue
        pcd = o3d.io.read_point_cloud(source_file)
        stl= np.asarray(pcd.points)
        stl_rgb = np.asarray(pcd.colors)
        # Copy file
        image_dir = '{0}/images'.format(dest_dir)
        image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
        n_images = len(image_paths)
        cam_file = '{0}/cameras.npz'.format(dest_dir)
        camera_dict = np.load(cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        scale_mat = scale_mats[0]
        ground_plane = loadmat(f'{source_dir}../../ObsMask/Plane{number}.mat')['P']
    
        stl_hom = np.concatenate([stl, np.ones_like(stl[:,:1])], -1)
        above = (ground_plane.reshape((1,4)) * stl_hom).sum(-1) > 0
        stl_above = stl[above]
        
        stl_rgb_above = stl_rgb[above]

        
        
        
        pcd.points = o3d.utility.Vector3dVector((stl_above - scale_mat[:3, 3][None])/ scale_mat[0, 0])
        pcd.colors = o3d.utility.Vector3dVector(stl_rgb_above)
        #shutil.copy2(source_file, dest_file)
        o3d.io.write_point_cloud(dest_file, pcd)
