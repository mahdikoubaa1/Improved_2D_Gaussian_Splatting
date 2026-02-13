import argparse
import os
from itertools import combinations
import json
from time import sleep
from narwhals import col
import pandas as pd



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

    modification_opts = {'exposure_optimization':'--use_exposure_optimization', 'MCMC':'--mcmc', 'depth Gaussian reinitialization':f'--depth_reinit_every 5000 --reinit_target_points_ratio 1.4', 'normal_depth_prior':'  --lambda_mono_depth 0.1 --lambda_mono_normal_l1 0.05 --lambda_mono_normal_cos 0.05 --mono_prior_decay_end 15000'}

    results_dict = {}
    for scene in os.listdir(os.path.join(args.output_path)):
        for comb in all_combinations:

            model_path = os.path.join(args.output_path,scene, subscene, '-'.join([opt.replace(' ','_') for opt in comb]) if len(comb)>0 else "base_model")
            metrics_path = os.path.join(model_path, "point_cloud", 'iteration_30000', "metrics.json")
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path, 'r') as f:
                results = json.load(f)
                
                entry = '+'.join([opt.replace('_',' ') for opt in comb]) if len(comb)>0 else "base model"
                if entry not in results_dict:
                    results_dict[entry] = [results]
                else:
                    results_dict[entry].append(results)

            
    results = {}
    for key in results_dict.keys():
        results[key] = {}
        for metric in results_dict[key][0].keys():
            results[key][metric] = sum([result[metric] for result in results_dict[key]])/len(results_dict[key])
    
    combination_names = [ '+'.join([opt.replace('_',' ') for opt in comb]) if len(comb)>0 else "base model" for comb in all_combinations]
    results = {combination: results.get(combination, {}) for combination in combination_names}
    results_final = {}  
    for key in results.keys():
        value= results[key]
        key = key.replace('MCMC','1').replace('depth Gaussian reinitialization','2').replace('normal depth prior','3')
        results_final[key]= value
    
    df = pd.DataFrame.from_dict(results_final, orient='index')
    to_min = ['L1', 'LPIPS', 'CD']
    to_max = ['PSNR', 'SSIM']
    def bold_best(col):
        if col.name in to_min:
            is_best = col == col.min()
        elif col.name in to_max:
            is_best = col == col.max()
        elif col.name == 'Points':
            return col.astype(int)  # Return original values for metrics that are neither minimized nor maximized
        return ['\\textbf{' + f'{v:.3f}' + '}' if b else f'{v:.3f}' for v, b in zip(col, is_best)]
    
        # Reorder columns - specify your desired column order
    desired_order = ['PSNR', 'SSIM', 'L1', 'LPIPS', 'CD','Points']  # Adjust as needed
    df = df[desired_order].apply(bold_best)
    header_map = {
        'PSNR': 'PSNR $\\uparrow$',
        'SSIM': 'SSIM $\\uparrow$',
        'L1': 'L1 $\\downarrow$',
        'LPIPS': 'LPIPS $\\downarrow$',
        'CD': 'CD $\\downarrow$',
        'Points': 'Points $\\uparrow$'
    }
    latex_table = df.rename(columns=header_map).to_latex(escape=False)
    document='''\\documentclass[varwidth]{standalone}
\\usepackage{graphicx}
\\usepackage{amsmath}
\\usepackage{booktabs}
\\begin{document}
\\small
''' + latex_table + '''\\end{document}
'''
    with open(os.path.join( f'{subscene}_results_table.tex'), 'w') as f:
        f.write(document)
    
    os.system(f"tectonic  {f'{subscene}_results_table.tex'}")
    
    

    #print('scenes to process: ', downloaded_val_list)