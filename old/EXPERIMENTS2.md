# Comprehensive Experimental Guide - All 4 Tasks
## 2D Gaussian Splatting: Complete Implementation Testing

---

## Implementation Summary

✅ **Task 1: MCMC Training** - 3DGS-MCMC strategy for better exploration  
✅ **Task 2: Depth Reinitialization** - Mini-Splatting geometric initialization  
✅ **Task 3: Exposure Optimization** - Per-image exposure compensation  
✅ **Task 4: Monocular Priors** - MonoSDF-style depth/normal supervision

---

## Experiment Matrix (16 Combinations)

| ID | MCMC | Depth Reinit | Exposure | Mono Priors | Purpose |
|----|------|--------------|----------|-------------|---------|
| E0 | ❌ | ❌ | ❌ | ❌ | **Baseline** - Reference |
| E1 | ✅ | ❌ | ❌ | ❌ | MCMC only |
| E2 | ❌ | ✅ | ❌ | ❌ | Depth reinit only |
| E3 | ❌ | ❌ | ✅ | ❌ | Exposure only |
| E4 | ❌ | ❌ | ❌ | ✅ | Mono priors only |
| E5 | ✅ | ✅ | ❌ | ❌ | MCMC + Depth |
| E6 | ✅ | ❌ | ✅ | ❌ | MCMC + Exposure |
| E7 | ✅ | ❌ | ❌ | ✅ | MCMC + Mono |
| E8 | ❌ | ✅ | ✅ | ❌ | Depth + Exposure |
| E9 | ❌ | ✅ | ❌ | ✅ | Depth + Mono |
| E10 | ❌ | ❌ | ✅ | ✅ | Exposure + Mono |
| E11 | ✅ | ✅ | ✅ | ❌ | MCMC + Depth + Exposure |
| E12 | ✅ | ✅ | ❌ | ✅ | MCMC + Depth + Mono |
| E13 | ✅ | ❌ | ✅ | ✅ | MCMC + Exposure + Mono |
| E14 | ❌ | ✅ | ✅ | ✅ | Depth + Exposure + Mono |
| **E15** | ✅ | ✅ | ✅ | ✅ | **ALL TASKS COMBINED** |

---

## Prerequisites

### 1. Precompute Monocular Priors (for E4, E7, E9, E10, E12-E15)

```bash
python 2dGScode/precompute_priors.py \
  -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr \
  --images resized_undistorted_images
```

---

## Base Command Template

All experiments share this base structure:

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/[EXPERIMENT_NAME]/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000 \
  [TASK_FLAGS]
```

---

## Individual Experiments

### E0: Baseline (No Modifications)

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/baseline/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000
```

---

### E1: MCMC Only

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/e1_mcmc/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000 \
  --mcmc \
  --cap_max 300000
```

**What it does:**
- MCMC densification (relocation instead of split/clone)
- SGLD noise injection for exploration
- Scale/opacity regularization
- Cap at 300K Gaussians

---

### E2: Depth Reinitialization Only

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/e2_depth/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000 \
  --depth_reinit_iters 2000 5000 10000 \
  --reinit_target_points 3500000
```

**What it does:**
- Reinitialize ALL Gaussians at iterations 2K, 5K, 10K
- Sample from depth maps with 3.5M target points
- Recover from scratch after each reinit

---

### E3: Exposure Optimization Only

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/e3_exposure/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000 \
  --use_exposure_optimization
```

**What it does:**
- Per-image affine color transformation [3x4]
- Compensates for exposure variations
- Especially useful for iPhone video data

---

### E4: Monocular Priors Only

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/e4_mono/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000 \
  --lambda_mono_depth 0.1 \
  --lambda_mono_normal_l1 0.05 \
  --lambda_mono_normal_cos 0.05 \
  --mono_prior_decay_end 15000
```

**What it does:**
- Scale-invariant depth loss with 4-scale gradients
- Normal L1 + cosine similarity losses
- Alpha-based validity masking
- Exponential decay over 15K iterations

---

### E5: MCMC + Depth Reinitialization

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/e5_mcmc_depth/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000 \
  --mcmc \
  --cap_max 300000 \
  --depth_reinit_iters 2000 5000 10000 \
  --reinit_target_points 3500000
```

**Synergy:** MCMC exploration + periodic geometric reset

---

### E6: MCMC + Exposure

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/e6_mcmc_exposure/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000 \
  --mcmc \
  --cap_max 300000 \
  --use_exposure_optimization
```

---

### E7: MCMC + Monocular Priors

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/e7_mcmc_mono/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000 \
  --mcmc \
  --cap_max 300000 \
  --lambda_mono_depth 0.1 \
  --lambda_mono_normal_l1 0.05 \
  --lambda_mono_normal_cos 0.05 \
  --mono_prior_decay_end 15000
```

---

### E8: Depth Reinitialization + Exposure

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/e8_depth_exposure/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000 \
  --depth_reinit_iters 2000 5000 10000 \
  --reinit_target_points 3500000 \
  --use_exposure_optimization
```

---

### E9: Depth Reinitialization + Monocular Priors

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/e9_depth_mono/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000 \
  --depth_reinit_iters 2000 5000 10000 \
  --reinit_target_points 3500000 \
  --lambda_mono_depth 0.1 \
  --lambda_mono_normal_l1 0.05 \
  --lambda_mono_normal_cos 0.05 \
  --mono_prior_decay_end 15000
```

**Synergy:** Geometric reset + continuous shape guidance

---

### E10: Exposure + Monocular Priors

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/e10_exposure_mono/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000 \
  --use_exposure_optimization \
  --lambda_mono_depth 0.1 \
  --lambda_mono_normal_l1 0.05 \
  --lambda_mono_normal_cos 0.05 \
  --mono_prior_decay_end 15000
```

---

### E11: MCMC + Depth + Exposure

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/e11_mcmc_depth_exposure/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000 \
  --mcmc \
  --cap_max 300000 \
  --depth_reinit_iters 2000 5000 10000 \
  --reinit_target_points 3500000 \
  --use_exposure_optimization
```

---

### E12: MCMC + Depth + Monocular Priors

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/e12_mcmc_depth_mono/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000 \
  --mcmc \
  --cap_max 300000 \
  --depth_reinit_iters 2000 5000 10000 \
  --reinit_target_points 3500000 \
  --lambda_mono_depth 0.1 \
  --lambda_mono_normal_l1 0.05 \
  --lambda_mono_normal_cos 0.05 \
  --mono_prior_decay_end 15000
```

---

### E13: MCMC + Exposure + Monocular Priors

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/e13_mcmc_exposure_mono/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000 \
  --mcmc \
  --cap_max 300000 \
  --use_exposure_optimization \
  --lambda_mono_depth 0.1 \
  --lambda_mono_normal_l1 0.05 \
  --lambda_mono_normal_cos 0.05 \
  --mono_prior_decay_end 15000
```

---

### E14: Depth + Exposure + Monocular Priors

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/e14_depth_exposure_mono/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000 \
  --depth_reinit_iters 2000 5000 10000 \
  --reinit_target_points 3500000 \
  --use_exposure_optimization \
  --lambda_mono_depth 0.1 \
  --lambda_mono_normal_l1 0.05 \
  --lambda_mono_normal_cos 0.05 \
  --mono_prior_decay_end 15000
```

---

### E15: ALL TASKS COMBINED 🔥

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/e15_all_combined/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --iterations 30000 \
  --mcmc \
  --cap_max 300000 \
  --depth_reinit_iters 2000 5000 10000 \
  --reinit_target_points 3500000 \
  --use_exposure_optimization \
  --lambda_mono_depth 0.1 \
  --lambda_mono_normal_l1 0.05 \
  --lambda_mono_normal_cos 0.05 \
  --mono_prior_decay_end 15000
```

**Everything at once!**

---

## Batch Evaluation Script

Save as `evaluate_all_experiments.sh`:

```bash
#!/bin/bash
# Comprehensive evaluation of all 16 experiments

OUTPUT_BASE="/cluster/51/koubaa/data/output/scannet++/0b031f3119"

python 2dGScode/evaluate.py \
  -m "${OUTPUT_BASE}/baseline" \
     "${OUTPUT_BASE}/e1_mcmc" \
     "${OUTPUT_BASE}/e2_depth" \
     "${OUTPUT_BASE}/e3_exposure" \
     "${OUTPUT_BASE}/e4_mono" \
     "${OUTPUT_BASE}/e5_mcmc_depth" \
     "${OUTPUT_BASE}/e6_mcmc_exposure" \
     "${OUTPUT_BASE}/e7_mcmc_mono" \
     "${OUTPUT_BASE}/e8_depth_exposure" \
     "${OUTPUT_BASE}/e9_depth_mono" \
     "${OUTPUT_BASE}/e10_exposure_mono" \
     "${OUTPUT_BASE}/e11_mcmc_depth_exposure" \
     "${OUTPUT_BASE}/e12_mcmc_depth_mono" \
     "${OUTPUT_BASE}/e13_mcmc_exposure_mono" \
     "${OUTPUT_BASE}/e14_depth_exposure_mono" \
     "${OUTPUT_BASE}/e15_all_combined" \
  --names "E0-Baseline" "E1-MCMC" "E2-Depth" "E3-Exposure" "E4-Mono" \
          "E5-MCMC+Depth" "E6-MCMC+Exp" "E7-MCMC+Mono" "E8-Depth+Exp" \
          "E9-Depth+Mono" "E10-Exp+Mono" "E11-M+D+E" "E12-M+D+Mo" \
          "E13-M+E+Mo" "E14-D+E+Mo" "E15-ALL" \
  --output evaluation_all_16.csv
```

---

## Recommended Experiment Subset (If Time-Limited)

If you can't run all 16, prioritize these **7 experiments**:

1. **E0** - Baseline (required)
2. **E1** - MCMC only
3. **E2** - Depth only
4. **E4** - Mono only
5. **E9** - Depth + Mono (your original Task 2+4)
6. **E12** - MCMC + Depth + Mono
7. **E15** - All combined

---

## Timeline Estimate

| Experiment Count | Training Time | Evaluation | Total |
|-----------------|---------------|------------|-------|
| Core 7 | ~18 hours | ~1.5 hours | ~20 hours |
| All 16 | ~45 hours | ~3 hours | ~48 hours |

*Based on ~2.5-3 hours per experiment*

---

## Monitoring & Debugging

### Check MCMC Activation
```bash
# Should see relocation messages
grep "relocate" train.log
```

### Check Depth Reinitialization
```bash
grep "Depth Reinitialization" train.log
```

### Check Exposure Optimization
```bash
# Should see exposure parameter group
grep "exposure" train.log
```

### Check Monocular Priors
```bash
# Should see non-zero mono losses
grep "mono_depth_loss" train.log
```

---

## Quick Reference: Flag Combinations

```bash
# Task 1: MCMC
--mcmc --cap_max 300000

# Task 2: Depth Reinit
--depth_reinit_iters 2000 5000 10000 --reinit_target_points 3500000

# Task 3: Exposure
--use_exposure_optimization

# Task 4: Monocular Priors
--lambda_mono_depth 0.1 --lambda_mono_normal_l1 0.05 --lambda_mono_normal_cos 0.05 --mono_prior_decay_end 15000
```

---

## Expected Results Analysis

### Image Quality (PSNR, SSIM, LPIPS, L1)
- **E0 (Baseline)**: Reference values
- **E1-E4**: Incremental improvements
- **E9**: Best 2-task combo (Depth + Mono)
- **E15**: Potential best overall

### Geometry Quality (Chamfer Distance)
- **E2, E5, E8, E9, E11, E12, E14, E15**: Should show improvements
- Depth reinitialization directly improves geometric accuracy

### Training Stability
- **MCMC**: More stable, fewer artifacts
- **Depth Reinit**: Temporary PSNR drops after reinit
- **Exposure**: Better convergence on varied exposure data
- **Mono Priors**: Smoother depth/normal surfaces

---

## Final Deliverables

✅ 16 trained models  
✅ Comprehensive evaluation CSV with all metrics  
✅ TensorBoard logs for all experiments  
✅ Analysis of which task combinations work best  
✅ Recommendations for production deployment
