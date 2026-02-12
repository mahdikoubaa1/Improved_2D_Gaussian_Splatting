#!/bin/bash
#
# Priority Experiments Runner (7 key experiments)
# For when you don't have time to run all 16
#

set -e

BASE_PATH="/cluster/51/koubaa/data/scannet++/data/0b031f3119"
OUTPUT_BASE="/cluster/51/koubaa/data/output/scannet++/0b031f3119"
RESULTS_FILE="results_priority.txt"
PYTHON_CMD="python 2dGScode/train.py"

COMMON_ARGS="--source_path ${BASE_PATH}/dslr/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000"

MCMC_FLAGS="--mcmc --cap_max 300000"
DEPTH_FLAGS="--depth_reinit_iters 2000 5000 10000 --reinit_target_points 3500000"
MONO_FLAGS="--lambda_mono_depth 0.1 --lambda_mono_normal_l1 0.05 --lambda_mono_normal_cos 0.05 --mono_prior_decay_end 15000"

# Initialize results file
echo "=====================================" > $RESULTS_FILE
echo "Priority Experiments Results" >> $RESULTS_FILE
echo "Started: $(date)" >> $RESULTS_FILE
echo "=====================================" >> $RESULTS_FILE

run_experiment() {
    local exp_name=$1
    local model_path=$2
    local extra_flags=$3
    
    echo ""
    echo "=========================================="
    echo "Starting: $exp_name"
    echo "=========================================="
    
    echo "----------------------------------------" >> $RESULTS_FILE
    echo "Experiment: $exp_name" >> $RESULTS_FILE
    echo "Started: $(date)" >> $RESULTS_FILE
    
    local full_cmd="$PYTHON_CMD --model_path ${OUTPUT_BASE}/${model_path}/ $COMMON_ARGS $extra_flags"
    
    start_time=$(date +%s)
    
    if eval $full_cmd; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        hours=$((duration / 3600))
        minutes=$(((duration % 3600) / 60))
        
        echo "✓ SUCCESS: ${hours}h ${minutes}m"
        echo "Status: SUCCESS - ${hours}h ${minutes}m" >> $RESULTS_FILE
    else
        echo "✗ FAILED"
        echo "Status: FAILED" >> $RESULTS_FILE
    fi
    
    echo "" >> $RESULTS_FILE
}

echo "Running 7 priority experiments..."

# Core 7 experiments
run_experiment "E0-Baseline" "baseline" ""
run_experiment "E1-MCMC" "e1_mcmc" "$MCMC_FLAGS"
run_experiment "E2-Depth" "e2_depth" "$DEPTH_FLAGS"
run_experiment "E4-Mono" "e4_mono" "$MONO_FLAGS"
run_experiment "E9-Depth+Mono" "e9_depth_mono" "$DEPTH_FLAGS $MONO_FLAGS"
run_experiment "E12-MCMC+Depth+Mono" "e12_mcmc_depth_mono" "$MCMC_FLAGS $DEPTH_FLAGS $MONO_FLAGS"
run_experiment "E15-ALL" "e15_all_combined" "$MCMC_FLAGS $DEPTH_FLAGS --use_exposure_optimization $MONO_FLAGS"

echo ""
echo "======================================"
echo "Priority experiments completed!"
echo "Results: $RESULTS_FILE"
echo "======================================"
