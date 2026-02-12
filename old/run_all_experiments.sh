#!/bin/bash
#
# Sequential Experiment Runner for 2DGS
# Runs all 16 experiments one after another and logs results
#
# Usage: 
#   chmod +x run_all_experiments.sh
#   nohup ./run_all_experiments.sh > experiments.log 2>&1 &
#

set -e  # Exit on error

# Configuration
BASE_PATH="/cluster/51/koubaa/data/scannet++/data/0b031f3119"
OUTPUT_BASE="/cluster/51/koubaa/data/output/scannet++/0b031f3119"
RESULTS_FILE="results.txt"
PYTHON_CMD="python 2dGScode/train.py"

# Common arguments
COMMON_ARGS="--source_path ${BASE_PATH}/dslr/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000"

# Task flags
MCMC_FLAGS="--mcmc --cap_max 300000"
DEPTH_FLAGS="--depth_reinit_iters 2000 5000 10000 --reinit_target_points 3500000"
EXPOSURE_FLAGS="--use_exposure_optimization"
MONO_FLAGS="--lambda_mono_depth 0.1 --lambda_mono_normal_l1 0.05 --lambda_mono_normal_cos 0.05 --mono_prior_decay_end 15000"

# Initialize results file
echo "=====================================" > $RESULTS_FILE
echo "2D Gaussian Splatting - Experiment Results" >> $RESULTS_FILE
echo "Started: $(date)" >> $RESULTS_FILE
echo "=====================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Function to run experiment
run_experiment() {
    local exp_name=$1
    local model_path=$2
    local extra_flags=$3
    
    echo ""
    echo "=========================================="
    echo "Starting: $exp_name"
    echo "Time: $(date)"
    echo "=========================================="
    
    # Log to results file
    echo "----------------------------------------" >> $RESULTS_FILE
    echo "Experiment: $exp_name" >> $RESULTS_FILE
    echo "Started: $(date)" >> $RESULTS_FILE
    
    # Full command
    local full_cmd="$PYTHON_CMD --model_path ${OUTPUT_BASE}/${model_path}/ $COMMON_ARGS $extra_flags"
    
    echo "Command: $full_cmd"
    echo "Command: $full_cmd" >> $RESULTS_FILE
    
    # Run training
    start_time=$(date +%s)
    
    if eval $full_cmd; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        hours=$((duration / 3600))
        minutes=$(((duration % 3600) / 60))
        
        echo "✓ SUCCESS: $exp_name completed in ${hours}h ${minutes}m"
        echo "Status: SUCCESS" >> $RESULTS_FILE
        echo "Duration: ${hours}h ${minutes}m" >> $RESULTS_FILE
        echo "Completed: $(date)" >> $RESULTS_FILE
        
        # Extract final metrics if available
        if [ -f "${OUTPUT_BASE}/${model_path}/train.log" ]; then
            echo "Final metrics:" >> $RESULTS_FILE
            tail -20 "${OUTPUT_BASE}/${model_path}/train.log" | grep -E "PSNR|L1" >> $RESULTS_FILE || true
        fi
    else
        echo "✗ FAILED: $exp_name"
        echo "Status: FAILED" >> $RESULTS_FILE
        echo "Failed: $(date)" >> $RESULTS_FILE
        # Continue to next experiment instead of exiting
    fi
    
    echo "" >> $RESULTS_FILE
}

# Main execution
echo "Starting all experiments..."
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# E0: Baseline
run_experiment "E0-Baseline" "baseline" ""

# E1: MCMC Only
run_experiment "E1-MCMC" "e1_mcmc" "$MCMC_FLAGS"

# E2: Depth Reinitialization Only
run_experiment "E2-Depth" "e2_depth" "$DEPTH_FLAGS"

# E3: Exposure Only
run_experiment "E3-Exposure" "e3_exposure" "$EXPOSURE_FLAGS"

# E4: Monocular Priors Only
run_experiment "E4-Mono" "e4_mono" "$MONO_FLAGS"

# E5: MCMC + Depth
run_experiment "E5-MCMC+Depth" "e5_mcmc_depth" "$MCMC_FLAGS $DEPTH_FLAGS"

# E6: MCMC + Exposure
run_experiment "E6-MCMC+Exposure" "e6_mcmc_exposure" "$MCMC_FLAGS $EXPOSURE_FLAGS"

# E7: MCMC + Mono
run_experiment "E7-MCMC+Mono" "e7_mcmc_mono" "$MCMC_FLAGS $MONO_FLAGS"

# E8: Depth + Exposure
run_experiment "E8-Depth+Exposure" "e8_depth_exposure" "$DEPTH_FLAGS $EXPOSURE_FLAGS"

# E9: Depth + Mono
run_experiment "E9-Depth+Mono" "e9_depth_mono" "$DEPTH_FLAGS $MONO_FLAGS"

# E10: Exposure + Mono
run_experiment "E10-Exposure+Mono" "e10_exposure_mono" "$EXPOSURE_FLAGS $MONO_FLAGS"

# E11: MCMC + Depth + Exposure
run_experiment "E11-MCMC+Depth+Exposure" "e11_mcmc_depth_exposure" "$MCMC_FLAGS $DEPTH_FLAGS $EXPOSURE_FLAGS"

# E12: MCMC + Depth + Mono
run_experiment "E12-MCMC+Depth+Mono" "e12_mcmc_depth_mono" "$MCMC_FLAGS $DEPTH_FLAGS $MONO_FLAGS"

# E13: MCMC + Exposure + Mono
run_experiment "E13-MCMC+Exposure+Mono" "e13_mcmc_exposure_mono" "$MCMC_FLAGS $EXPOSURE_FLAGS $MONO_FLAGS"

# E14: Depth + Exposure + Mono
run_experiment "E14-Depth+Exposure+Mono" "e14_depth_exposure_mono" "$DEPTH_FLAGS $EXPOSURE_FLAGS $MONO_FLAGS"

# E15: ALL Combined
run_experiment "E15-ALL" "e15_all_combined" "$MCMC_FLAGS $DEPTH_FLAGS $EXPOSURE_FLAGS $MONO_FLAGS"

# Final summary
echo ""
echo "======================================"
echo "All experiments completed!"
echo "Time: $(date)"
echo "======================================"

echo "" >> $RESULTS_FILE
echo "=====================================" >> $RESULTS_FILE
echo "All experiments completed: $(date)" >> $RESULTS_FILE
echo "=====================================" >> $RESULTS_FILE

# Count successes and failures
success_count=$(grep -c "Status: SUCCESS" $RESULTS_FILE || echo "0")
failed_count=$(grep -c "Status: FAILED" $RESULTS_FILE || echo "0")

echo "Summary:" >> $RESULTS_FILE
echo "  Successful: $success_count" >> $RESULTS_FILE
echo "  Failed: $failed_count" >> $RESULTS_FILE

echo ""
echo "Results saved to: $RESULTS_FILE"
echo "Successful experiments: $success_count"
echo "Failed experiments: $failed_count"
