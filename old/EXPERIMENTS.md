# Comprehensive Experimental Testing Guide
## 2D Gaussian Splatting: Task 2 & Task 4 Implementation

---

## Overview

This guide provides systematic commands to test all combinations of:
- **Task 2**: Depth Gaussian Reinitialization (Mini-Splatting)
- **Task 4**: MonoSDF-Style Monocular Priors

### Test Matrix

| Experiment | Depth Reinit | Mono Priors | Purpose |
|------------|--------------|-------------|---------|
| Baseline | ❌ | ❌ | Reference performance |
| Task 2 | ✅ | ❌ | Test depth reinitialization alone |
| Task 4 | ❌ | ✅ | Test monocular priors alone |
| Combined | ✅ | ✅ | Test synergy of both methods |

---

## Prerequisites

### 1. Precompute Monocular Priors (for Task 4)

**Only needed for experiments with monocular priors**

```bash
# On server
cd /cluster/51/koubaa/abdullah/2DGaussianSplatting

python 2dGScode/precompute_priors.py \
  -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr \
  --images resized_undistorted_images
```

**Expected output:**
- Creates `mono_priors/mono_depth/*.npy` files
- Creates `mono_priors/mono_normal/*.npy` files
- Should process all training images (~271 images)

---

## Experiment 1: Baseline (No Modifications)

**Purpose:** Establish reference metrics

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
  --mcmc \
  --cap_max 300000 \
  --iterations 30000
```

**Key features:**
- Standard 2DGS training
- MCMC densification
- No depth reinitialization
- No monocular priors

**Expected training time:** ~2-3 hours

---

## Experiment 2: Task 2 - Depth Reinitialization Only

**Purpose:** Test Mini-Splatting depth reinitialization strategy

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/task2_depth_reinit/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --mcmc \
  --cap_max 300000 \
  --iterations 30000 \
  --depth_reinit_iters 2000 5000 10000 \
  --reinit_target_points 3500000
```

**Key parameters:**
- `--depth_reinit_iters 2000 5000 10000`: Reinitialize at iterations 2K, 5K, 10K
- `--reinit_target_points 3500000`: Target 3.5M Gaussians after sampling

**What happens:**
1. Training proceeds normally until iteration 2000
2. At iter 2000: Replace all Gaussians with depth-sampled points
3. Continue training until iter 5000
4. At iter 5000: Another reinitialization
5. Continue training until iter 10000
6. At iter 10000: Final reinitialization
7. Train until completion (30K iterations)

**Expected behavior:**
- PSNR may drop temporarily after each reinitialization
- Should recover and potentially exceed baseline
- Improved geometric quality (less floaters)

---

## Experiment 3: Task 4 - MonoSDF Priors Only

**Purpose:** Test monocular depth and normal supervision

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/task4_monosdf/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --mcmc \
  --cap_max 300000 \
  --iterations 30000 \
  --lambda_mono_depth 0.1 \
  --lambda_mono_normal_l1 0.05 \
  --lambda_mono_normal_cos 0.05 \
  --mono_prior_decay_end 15000
```

**Key parameters:**
- `--lambda_mono_depth 0.1`: Scale-invariant depth loss weight
- `--lambda_mono_normal_l1 0.05`: Normal L1 loss weight
- `--lambda_mono_normal_cos 0.05`: Normal cosine similarity weight
- `--mono_prior_decay_end 15000`: Exponential decay ends at 15K iterations

**What happens:**
- Multi-scale (4-level) gradient loss for depth
- Alpha-based validity masking (only supervise high-confidence pixels)
- Exponential decay: full weight at start, ~0 at iteration 15K
- TensorBoard logs: `mono_depth_loss`, `mono_normal_l1_loss`, `mono_normal_cos_loss`

---

## Experiment 4: Combined (Task 2 + Task 4)

**Purpose:** Test synergy of both methods

```bash
python 2dGScode/train.py \
  --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ \
  --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/combined/ \
  --depth_ratio 1 \
  --images '../dslr/resized_undistorted_images' \
  --test_images '../dslr/resized_undistorted_images' \
  --train_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' \
  --eval \
  --mcmc \
  --cap_max 300000 \
  --iterations 30000 \
  --depth_reinit_iters 2000 5000 10000 \
  --reinit_target_points 3500000 \
  --lambda_mono_depth 0.1 \
  --lambda_mono_normal_l1 0.05 \
  --lambda_mono_normal_cos 0.05 \
  --mono_prior_decay_end 15000
```

**Expected behavior:**
- Depth reinitialization provides better geometric structure
- Monocular priors guide shape recovery after reinitialization
- Potential for best overall performance

---

## Ablation Studies (Optional)

### A1: Different Reinitialization Schedules

**Early reinit:**
```bash
--depth_reinit_iters 1000 3000 7000
```

**Late reinit:**
```bash
--depth_reinit_iters 5000 10000 15000
```

### A2: Different Prior Weights

**Strong priors:**
```bash
--lambda_mono_depth 0.2 \
--lambda_mono_normal_l1 0.1 \
--lambda_mono_normal_cos 0.1
```

**Weak priors:**
```bash
--lambda_mono_depth 0.05 \
--lambda_mono_normal_l1 0.025 \
--lambda_mono_normal_cos 0.025
```

### A3: Different Target Points

**More Gaussians:**
```bash
--reinit_target_points 5000000
```

**Fewer Gaussians:**
```bash
--reinit_target_points 2000000
```

---

## Evaluation

### Step 1: Install Evaluation Dependencies

```bash
# On server
conda activate ml3d
pip install lpips
```

### Step 2: Run Comprehensive Evaluation

```bash
python 2dGScode/evaluate.py \
  -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/baseline \
     /cluster/51/koubaa/data/output/scannet++/0b031f3119/task2_depth_reinit \
     /cluster/51/koubaa/data/output/scannet++/0b031f3119/task4_monosdf \
     /cluster/51/koubaa/data/output/scannet++/0b031f3119/combined \
  --names "Baseline" "Task2-Reinit" "Task4-MonoSDF" "Combined" \
  --output evaluation_results.csv
```

### Step 3: Download Results

```bash
# On local machine
rsync -avz --progress \
  -e "ssh -i key" \
  abdullah@ml3d.vc.in.tum.de:/cluster/51/koubaa/data/output/scannet++/0b031f3119/evaluation_results.csv \
  ./results/
```

---

## Expected Output Format

### Training Logs

Monitor TensorBoard:
```bash
tensorboard --logdir /cluster/51/koubaa/data/output/scannet++/0b031f3119/
```

**Key metrics to watch:**
- `train_loss_patches/total_loss`
- `train_loss_patches/mono_depth_loss` (Task 4)
- `train_loss_patches/mono_normal_l1_loss` (Task 4)
- `test/loss_viewpoint - psnr`

### Evaluation Results (CSV)

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | L1 ↓ | Chamfer ↓ |
|-------|--------|--------|---------|------|-----------|
| Baseline | 24.5 | 0.850 | 0.150 | 0.050 | 0.012 |
| Task2-Reinit | 24.8 | 0.860 | 0.140 | 0.048 | 0.010 |
| Task4-MonoSDF | 25.1 | 0.870 | 0.130 | 0.045 | 0.009 |
| Combined | **25.3** | **0.880** | **0.120** | **0.042** | **0.008** |

*(Values are hypothetical examples)*

---

## Monitoring Training

### Check Progress

```bash
# See latest PSNR
tail -f /cluster/51/koubaa/data/output/scannet++/0b031f3119/[model_name]/train.log | grep PSNR

# Count Gaussians
ls /cluster/51/koubaa/data/output/scannet++/0b031f3119/[model_name]/point_cloud/
```

### Check for Depth Reinitialization (Task 2)

Look for these messages in logs:
```
[ITER 2000] Depth Reinitialization (Mini-Splatting)
[Depth Sampling] Sampled X points from 271 views
[Depth Reinitialization] Complete. Total Gaussians: Y
```

### Check for Monocular Priors (Task 4)

Look for non-zero mono losses in TensorBoard or logs:
```
mono_depth_loss: 0.0023
mono_normal_l1_loss: 0.0015
mono_normal_cos_loss: 0.0012
```

---

## Troubleshooting

### Issue: NaN losses

**Solution:** Remove `--detect_anomaly` flag (it's overly strict)

### Issue: No monocular priors loaded

**Check:**
```bash
ls /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/mono_priors/
```

Should contain `mono_depth/` and `mono_normal/` directories

### Issue: Training too slow

**Reduce resolution:**
```bash
--resolution 1  # Force 1x resolution
```

### Issue: Out of memory

**Reduce cap_max:**
```bash
--cap_max 200000  # Reduce from 300K
```

---

## Quick Reference: Command Generator

**Just depth reinit:**
```bash
+ --depth_reinit_iters 2000 5000 10000 --reinit_target_points 3500000
```

**Just monocular priors:**
```bash
+ --lambda_mono_depth 0.1 --lambda_mono_normal_l1 0.05 --lambda_mono_normal_cos 0.05 --mono_prior_decay_end 15000
```

**Both:**
```bash
+ --depth_reinit_iters 2000 5000 10000 --reinit_target_points 3500000 \
  --lambda_mono_depth 0.1 --lambda_mono_normal_l1 0.05 --lambda_mono_normal_cos 0.05 --mono_prior_decay_end 15000
```

---

## Timeline Estimate

| Experiment | Training Time | Evaluation Time | Total |
|------------|--------------|-----------------|-------|
| Baseline | ~2.5 hours | 10 min | 2.6 hours |
| Task 2 | ~3 hours | 10 min | 3.1 hours |
| Task 4 | ~2.5 hours | 10 min | 2.6 hours |
| Combined | ~3 hours | 10 min | 3.1 hours |
| **Total** | | | **~11.4 hours** |

*Plus ~30min for precomputing priors*

---

## Final Deliverables

1. ✅ 4 trained models (baseline, task2, task4, combined)
2. ✅ Evaluation CSV with all metrics
3. ✅ TensorBoard logs showing training curves
4. ✅ Visual comparisons (optional: render test views)
