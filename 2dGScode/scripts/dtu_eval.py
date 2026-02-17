from itertools import combinations
import os
from argparse import ArgumentParser

dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']
dtu_scenes = ['scan55']

mod_list = ['MCMC', 'depth_Gaussian_reinitialization', 'normal_depth_prior']
all_combinations = []
for r in range(len(mod_list) + 1):
        all_combinations.extend(combinations(mod_list, r))
all_combinations = ['-'.join(comb) if comb else 'base_model' for comb in all_combinations]
        
parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval/dtu")
parser.add_argument('--dtu', "-dtu", required=True, type=str)
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(dtu_scenes)

if not args.skip_metrics:
    parser.add_argument('--DTU_Official', "-DTU", required=True, type=str)
    args = parser.parse_args()


if not args.skip_training:
    common_args = " --quiet --test_iterations -1 --depth_ratio 1.0 -r 2 --lambda_dist 1000"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)


if not args.skip_rendering:
    all_sources = []
    common_args = " --quiet --skip_train --depth_ratio 1.0 --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        print("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)


if not args.skip_metrics:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for scene in dtu_scenes:
        scan_id = scene[4:]
        iteration = 30000
        for comb in all_combinations:
            print (f"Evaluating combination: {comb} for scene: {scene}")
            ply_file = f"{args.output_path}/{scene}/{comb}/train/ours_30000/"
            string = f"python {script_dir}/eval_dtu/evaluate_single_scene.py " + \
                f"--input_mesh {ply_file}fuse_post.ply " + \
                f"--scan_id {scan_id} --output_dir {ply_file}../../point_cloud/iteration_{iteration} " + \
                f"--mask_dir {args.dtu} " + \
            f"--DTU {args.DTU_Official}"
            print("Executing evaluation command: ", string)
            os.system(string)