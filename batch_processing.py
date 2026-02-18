import argparse
import os
from itertools import combinations
import json
from time import sleep

def read_file_to_list(file_path):
    """
    Read a text file and return its content as a list of lines.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        list: List of lines from the file
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines


# Alternative: Remove newline characters
def read_file_to_list_clean(file_path):
    """
    Read a text file and return content as a list without newline characters.
    """
    with open(file_path, 'r') as file:
        lines = [line.rstrip('\n') for line in file.readlines()]        
    return lines

# Example usage
if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="/cluster/51/koubaa/data/scannet++")
    parser.add_argument('--scene', type=str, default='', help='scene to process')
    parser.add_argument('--output_path', type=str, default="/cluster/51/koubaa/data/output/scannet++/")
    parser.add_argument('--subscene', type=str, default="dslr", choices=['iphone', 'dslr','dtu'], help='which subscene to process')
    parser.add_argument('--port', type=int, default=6009, help='port for visualization')
    args = parser.parse_args()
    dataset_path = args.dataset_path
    subscene = args.subscene
    scene = args.scene
    modifications= {'iphone': ['exposure_optimization'], 'dslr': ['MCMC','depth Gaussian reinitialization','normal_depth_prior'], 'dtu': ['MCMC','depth Gaussian reinitialization','normal_depth_prior']}
    mod_list = modifications[subscene]
    all_combinations = []
    for r in range(len(mod_list) + 1):
        all_combinations.extend(combinations(mod_list, r))
    #all_combinations = all_combinations[-3:] # process combinations with more modifications first
    lambda_dist = {'7b6477cb95': 10, 'c50d2d1d42': 10, 'cc5237fd77': 1, '0b031f3119': 100} if subscene == 'dslr' else {'7b6477cb95': 10, 'c50d2d1d42': 10, 'cc5237fd77': 1, '0b031f3119': 10}
    subscene__options = {
        'iphone':  {'train':'--depth_ratio 1 --geometric_test --images rgb  --test_images ../dslr/resized_undistorted_images --train_transforms_file nerfstudio/transforms.json --test_transforms_file ../dslr/nerfstudio/transforms_undistorted.json --eval  ',
                    'render':'--depth_ratio 1 --images ../dslr/resized_undistorted_images --test_images ../dslr/resized_undistorted_images --train_transforms_file ../dslr/nerfstudio/transforms_undistorted.json --test_transforms_file ../dslr/nerfstudio/transforms_undistorted.json --eval --skip_train --skip_test --voxel_size 0.02 --depth_trunc 7 --sdf_trunc 0.1 --iteration 30000'},
        'dslr': {'train':'--depth_ratio 1 --geometric_test --images resized_undistorted_images --test_images resized_undistorted_images --train_transforms_file nerfstudio/transforms_undistorted.json --test_transforms_file nerfstudio/transforms_undistorted.json --eval',
                 'render':'--depth_ratio 1 --images ../dslr/resized_undistorted_images --test_images ../dslr/resized_undistorted_images --train_transforms_file ../dslr/nerfstudio/transforms_undistorted.json --test_transforms_file ../dslr/nerfstudio/transforms_undistorted.json --eval --skip_train --skip_test --voxel_size 0.02 --depth_trunc 7 --sdf_trunc 0.1 --iteration 30000'},
        'dtu': {'train':'--use_colmap --resolution 2 --colmap_folder \'sparse/0\' --depth_ratio 1 --eval --geo_type pcd --geo_name pcd.ply',
                'render':'--use_colmap --colmap_folder \'sparse/0\' --depth_ratio 1 --skip_train --skip_test --eval'}
    }
    
    print("Total combinations to test: ", all_combinations)
    print("Total number of combinations: ", len(all_combinations))
    modification_opts = {'exposure_optimization':'--use_exposure_optimization', 'MCMC':'--mcmc', 'depth Gaussian reinitialization':f'--depth_reinit_every 5000 --reinit_target_points 400000', 'normal_depth_prior':'  --lambda_mono_depth 0.1 --lambda_mono_normal_l1 0.05 --lambda_mono_normal_cos 0.05 --mono_prior_decay_end 15000'}
    #file_list = read_file_to_list_clean(os.path.join(dataset_path, "to_download.txt"))
    #val_list = read_file_to_list_clean(os.path.join(dataset_path, "splits/nvs_sem_val.txt"))
    #downloaded_val_list = [line for line in file_list if line in val_list]
    #all_combinations= [i for i in all_combinations if 'depth Gaussian reinitialization' in i] # Filter combinations to only those that include at least one of the modifications
    print("Filtered combinations to test: ", len(all_combinations))
    for comb in all_combinations:
        for k in modification_opts.keys():
            if k in comb:
                print(f'\033[92m{k} enabled\033[0m')
            else:
                print(f'\033[91m{k} disabled\033[0m')
        if subscene == 'dtu':
            source_path = os.path.join(dataset_path,scene)
            model_path = os.path.join(args.output_path,scene, '-'.join([opt.replace(' ','_') for opt in comb]) if len(comb)>0 else "base_model")
        else:
            source_path = os.path.join(dataset_path,'data',scene, subscene)
            model_path = os.path.join(args.output_path,scene, subscene, '-'.join([opt.replace(' ','_') for opt in comb]) if len(comb)>0 else "base_model")
        #if os.path.exists(os.path.join(model_path, 'point_cloud', 'iteration_30000', 'point_cloud.ply')) and os.path.exists(os.path.join(model_path, 'train','ours_30000', 'fuse_post.ply')):
        #    print(f"Model and render already exist for combination {comb}. Skipping...")
        #    continue
        if 'MCMC' in comb or 'depth Gaussian reinitialization' in comb:
            os.makedirs(model_path, exist_ok=True)
            with open(os.path.join(model_path, '../base_model/point_cloud/iteration_30000/metrics.json'), 'r') as log_file:
                log_data = json.load(log_file)
                cap_max = log_data['Points']
                if subscene == 'dtu':
                    reinit_points = log_data['Points']
                else:
                    reinit_points = ((log_data['Points']//100000)-1)*100000
                modification_opts['MCMC'] = f'--mcmc --cap_max {cap_max}'
                modification_opts['depth Gaussian reinitialization'] = f'--depth_reinit_every 5000 --reinit_target_points {reinit_points}'
        train_cmd = f'python 2dGScode/train.py --source_path {source_path} --model_path {model_path}'+ ' ' + ' '.join([modification_opts[opt] for opt in comb]) + ' ' + subscene__options[subscene]['train']+ f' --lambda_dist {lambda_dist[scene] if scene in lambda_dist else 1000 if subscene == "dtu" else 10}'+ f' --port {args.port}'
        
        render_cmd = f'python 2dGScode/render.py --source_path {source_path} --model_path {model_path}' + ' ' + subscene__options[subscene]['render']
        try:
            attempt = 0
            print("Executing training command: ", train_cmd)

            while ( not os.path.exists(os.path.join(model_path, 'point_cloud', 'iteration_30000', 'point_cloud.ply'))):
                os.system(train_cmd)
                attempt += 1
                if attempt >= 5:
                    print(f"Training failed after {attempt} attempts. Skipping to next combination.")
                    break
                sleep(1)
            attempt = 0

            print("Executing rendering command: ", render_cmd)
            while (attempt==0 or not os.path.exists(os.path.join(model_path, 'train','ours_30000', 'fuse_post.ply'))):
                #os.system(render_cmd)
                
                if subscene == 'dtu':
                    ply_file = os.path.join(model_path, 'train','ours_30000', 'fuse_post.ply')
                    scan_id = scene[4:]
                    string = f"python 2dGScode/scripts/eval_dtu/evaluate_single_scene.py " + \
                        f"--input_mesh {ply_file} " + \
                        f"--scan_id {scan_id} --output_dir {os.path.join(model_path, 'point_cloud', 'iteration_30000')} " + \
                        f"--mask_dir {args.dataset_path} " + \
                        f"--DTU {args.dataset_path}_Official"
                    print("Executing evaluation command: ", string)
                    os.system(string)
                    #string = f"python 2dGScode/calculate_absrel.py --source_path {source_path} --model_path {model_path} --depth_ratio 1 --eval --skip_train --skip_test --voxel_size 0.02 --depth_trunc 7 --sdf_trunc 0.10  --iteration 30000 --use_colmap --colmap_folder 'sparse/0' --geo_type pcd --geo_name pcd.ply --resolution 2"
                    #print (string)
                    #os.system(string)
                #elif subscene == 'dslr' or subscene == 'iphone':
                #    string = f"python 2dGScode/scripts/eval_dslr/evaluate_single_scene.py " +
                    
                attempt += 1
                if attempt >= 5:
                    print(f"Rendering failed after {attempt} attempts. Skipping to next combination.")
                    break
                sleep(1)

        except KeyboardInterrupt:
            print("Training interrupted by user.")
    #print('scenes to process: ', downloaded_val_list)