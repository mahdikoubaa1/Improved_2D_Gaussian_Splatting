import argparse
import os
from itertools import combinations
import json
from time import sleep
import pandas as pd

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
    parser.add_argument('--output_path', type=str, default="/cluster/51/koubaa/data/output/scannet++/")
    parser.add_argument('--subscene', type=str, default="dslr", choices=['iphone', 'dslr'], help='which subscene to process')
    parser.add_argument('--port', type=int, default=6009, help='port for visualization')
    args = parser.parse_args()
    dataset_path = args.dataset_path
    subscene = args.subscene
    modifications= {'iphone': ['exposure_optimization'], 'dslr': ['MCMC','depth Gaussian reinitialization','normal_depth_prior']}
    mod_list = modifications[subscene]
    all_combinations = []
    
    for r in range(len(mod_list) + 1):
        all_combinations.extend(combinations(mod_list, r))

    print("Total combinations to test: ", all_combinations)
    print("Total number of combinations: ", len(all_combinations))
    modification_opts = {'exposure_optimization':'--use_exposure_optimization', 'MCMC':'--mcmc', 'depth Gaussian reinitialization':f'--depth_reinit_every 5000 --reinit_target_points_ratio 1.4', 'normal_depth_prior':'  --lambda_mono_depth 0.1 --lambda_mono_normal_l1 0.05 --lambda_mono_normal_cos 0.05 --mono_prior_decay_end 15000'}
    #file_list = read_file_to_list_clean(os.path.join(dataset_path, "to_download.txt"))
    #val_list = read_file_to_list_clean(os.path.join(dataset_path, "splits/nvs_sem_val.txt"))
    #downloaded_val_list = [line for line in file_list if line in val_list]
    #all_combinations= [i for i in all_combinations if 'MCMC' in i and len(i)==1] # Filter combinations to only those that include at least one of the modifications
    print("Filtered combinations to test: ", len(all_combinations))
    results_dict = {}
    for scene in os.listdir(os.path.join(args.output_path)):
        for comb in all_combinations:

            model_path = os.path.join(args.output_path,scene, subscene, '-'.join([opt.replace(' ','_') for opt in comb]) if len(comb)>0 else "base_model")
            metrics_path = os.path.join(model_path, "point_cloud", 'iteration_30000', "metrics.json")
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path, 'r') as f:
                results = json.load(f)
                
                #entry = '+'.join([opt.replace('_',' ') for opt in comb]) if len(comb)>0 else "base model"
                #if entry not in results_dict:
                #    results_dict[entry] = [results]
                #else:
                #    results_dict[entry].append(results)
            output_dir=os.path.join(args.output_path,'new',scene, subscene, '-'.join([opt.replace(' ','_') for opt in comb]) if len(comb)>0 else "base_model", "point_cloud", 'iteration_30000')
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
                json.dump(results, f)
                
    #results = {}
    #for key in results_dict.keys():
    #    results[key] = {}
    #    for metric in results_dict[key][0].keys():
    #        results[key][metric] = sum([result[metric] for result in results_dict[key]])/len(results_dict[key])
    #
    #combination_names = [ '+'.join([opt.replace('_',' ') for opt in comb]) if len(comb)>0 else "base model" for comb in all_combinations]
    #results = {combination: results.get(combination, {}) for combination in combination_names}
    #results_final = {}  
    #for key in results.keys():
    #    value= results[key]
    #    key = key.replace('MCMC','1').replace('depth Gaussian reinitialization','2').replace('normal depth prior','3')
    #    print(key)
    #    results_final[key]= value
    #df = pd.DataFrame.from_dict(results_final, orient='index')
    #latex_table = df.to_latex(
    #    float_format="%.4f"
    #)
    #print(latex_table)
    #with open("my_table.tex", "w") as f:
    #    f.write(latex_table)


    #print('scenes to process: ', downloaded_val_list)