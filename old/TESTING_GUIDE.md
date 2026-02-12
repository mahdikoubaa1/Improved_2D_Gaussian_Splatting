# Testing Guide: 2DGS with Depth Reinitialization

## Prerequisites

**IMPORTANT:** Before testing, ensure these code fixes are in place:

1. ✅ **Fixed `depth_reinit.py`**: Corrected 3 critical transformation matrix bugs (extracting translation vector)
2. ✅ **Fixed `dataset_readers.py`**: Added Nerfstudio format support (ScanNet++ uses `fl_x`, `fl_y` instead of `camera_angle_x`)
3. ✅ **Fixed `scene/__init__.py`**: Pass `images_folder` argument to Nerfstudio loader

**Commit these changes before proceeding!**

---

## Server Setup

**Your working directory:** `/cluster/51/koubaa/abdullah/2DGaussianSplatting`  
**Shared dataset:** `/cluster/51/koubaa/data/scannet++/data/`  
**Output directory:** `/cluster/51/koubaa/data/output/`  
**SSH server:** `koubaa@ml3d.vc.in.tum.de`  
**SSH key:** `key` (in project root)

---

## Step 1: Upload Fixed Code to Server

From your **local machine**, upload the fixed code to the server:

```bash
# Make sure you're in the local project directory
cd /Users/abdullah/Desktop/TUMstudy/ml3d/project/2DGaussianSplatting

# Commit your changes locally first
git add 2dGScode/utils/depth_reinit.py 2dGScode/scene/dataset_readers.py 2dGScode/scene/__init__.py
git commit -m "Fix depth reinitialization and add Nerfstudio format support"
git push

# Upload via rsync (recommended)
rsync -avz --progress \
    -e "ssh -i key" \
    --exclude='output/' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    2dGScode/ \
    koubaa@ml3d.vc.in.tum.de:/cluster/51/koubaa/abdullah/2DGaussianSplatting/2dGScode/
```

**OR pull from git on the server:**

```bash
# SSH into server
ssh -i key koubaa@ml3d.vc.in.tum.de
cd /cluster/51/koubaa/abdullah/2DGaussianSplatting
git pull
```

---

## Step 2: SSH into Server and Setup Environment

```bash
# SSH into the cluster
ssh <your_username>@<server>

# Navigate to your project directory
cd /cluster/51/koubaa/abdullah/2DGaussianSplatting

# Activate conda environment (assuming it's named ml3d like your teammate's)
conda activate ml3d

# Verify you have the required packages
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Step 3: Allocate GPU Resources

```bash
# Request a GPU (adjust based on your cluster's SLURM settings)
salloc --gpus=1 --time=4:00:00

# OR if using screen for long-running jobs
screen -S gaussian_test
# Then allocate GPU inside screen
```

---

## Step 4: Test 1 - Baseline 2DGS (WITHOUT Depth Reinitialization)

First, verify the baseline works correctly:

```bash
# Choose a scene for testing - using DSLR for better quality
SCENE_ID="0b031f3119"
DATA_PATH="/cluster/51/koubaa/data/scannet++/data/${SCENE_ID}/dslr"
OUTPUT_PATH="/cluster/51/koubaa/data/output/abdullah/test_baseline/${SCENE_ID}"

# Run baseline training (10K iterations, no depth reinit)
python 2dGScode/train.py \
  -s ${DATA_PATH} \
  -m ${OUTPUT_PATH} \
  --images resized_undistorted_images \
  --depth_ratio 1 \
  --iterations 10000 \
  --eval
```

**For iPhone data (alternative):**
```bash
SCENE_ID="fb5a96b1a2"
DATA_PATH="/cluster/51/koubaa/data/scannet++/data/${SCENE_ID}/iphone"
OUTPUT_PATH="/cluster/51/koubaa/data/output/abdullah/test_baseline_iphone/${SCENE_ID}"

python 2dGScode/train.py \
  -s ${DATA_PATH} \
  -m ${OUTPUT_PATH} \
  --images rgb \
  --depth_ratio 1 \
  --iterations 10000 \
  --eval
```

**Expected output:**
```
Reading Training Transforms
Loading Training Cameras
Loading Test Cameras
Number of points at initialisation : XXXXX
Training progress: [====>    ] XXXX/10000
...
```

**Monitor:** Watch for any errors, especially import errors or CUDA issues.

---

## Step 5: Test 2 - WITH Depth Reinitialization (SMALL TEST)

Now test the fixed depth reinitialization on a short run:

```bash
SCENE_ID="0b031f3119"
DATA_PATH="/cluster/51/koubaa/data/scannet++/data/${SCENE_ID}/dslr"
OUTPUT_PATH="/cluster/51/koubaa/data/output/abdullah/test_depth_reinit/${SCENE_ID}"

# CRITICAL: Test with early reinitialization to see it quickly
python 2dGScode/train.py \
  -s ${DATA_PATH} \
  -m ${OUTPUT_PATH} \
  --images resized_undistorted_images \
  --depth_ratio 1 \
  --iterations 2000 \
  --depth_reinit_iters 1000 \
  --reinit_target_points 500000 \
  --eval
```

**Note:** This short 2000-iteration test with reinit at 1000 will complete in ~10-15 minutes and clearly show if the feature works.

**CRITICAL: Watch for this message at iteration 1000:**

```
[ITER 1000] Depth Reinitialization (Mini-Splatting)
[Depth Sampling] Sampled XXXXX points from N views
[Depth Reinitialization] Complete. Total Gaussians: XXXXX
```

**Note:** The progress bar may hide this message. If you don't see it, check that:
1. You included `--depth_reinit_iters 1000` in the command
2. Training didn't crash at iteration 1000

**✅ SUCCESS if you see:**
- No errors about transformation matrices
- Point counts are reasonable (100K - 2M points)
- Training continues smoothly after reinit
- No NaN losses

**❌ FAILURE if you see:**
- `IndexError` or `RuntimeError` related to matrix indexing
- All points at (0,0,0)
- NaN or Inf in losses
- Training crashes after reinit

---

## Step 6: Test 3 - Full Training Run (DSLR Data)

If Test 2 succeeds, run a full training with DSLR data (higher quality):

```bash
SCENE_ID="0b031f3119"
DATA_PATH="/cluster/51/koubaa/data/scannet++/data/${SCENE_ID}/dslr"
OUTPUT_PATH="/cluster/51/koubaa/data/output/abdullah/full_test/${SCENE_ID}_dslr"

python 2dGScode/train.py \
  -s ${DATA_PATH} \
  -m ${OUTPUT_PATH} \
  --images resized_undistorted_images \
  --depth_ratio 1 \
  --iterations 30000 \
  --depth_reinit_iters 5000 10000 15000 \
  --reinit_target_points 2000000 \
  --eval
```

**Note:** Reduced `reinit_target_points` to 2M to avoid potential CUDA OOM errors.

**Training time:** ~2-4 hours depending on GPU

---

## Step 7: Monitor Training Progress

### Option A: TensorBoard (Recommended)

```bash
# In a separate terminal/screen
conda activate ml3d
tensorboard --logdir=/cluster/51/koubaa/data/output/abdullah/ --port=6006 --bind_all

# Then access via browser (if port forwarding is set up)
# Or SSH tunnel: ssh -L 6006:localhost:6006 <user>@<server>
```

### Option B: Check Output Directory

```bash
# Monitor output files
watch -n 10 'ls -lh /cluster/51/koubaa/data/output/abdullah/test_depth_reinit/fb5a96b1a2/'

# Check latest checkpoint
ls -lht /cluster/51/koubaa/data/output/abdullah/test_depth_reinit/fb5a96b1a2/point_cloud/
```

### Option C: Tail the Log (if you redirected output)

```bash
# If you ran with: python train.py ... > train.log 2>&1
tail -f train.log | grep -E "ITER|Depth|Loss"
```

---

## Step 8: Verify Results

After training completes, check the outputs:

```bash
OUTPUT_DIR="/cluster/51/koubaa/data/output/abdullah/test_depth_reinit/fb5a96b1a2"

# Check final Gaussian count
ls -lh ${OUTPUT_DIR}/point_cloud/iteration_*/point_cloud.ply

# Check if test images were rendered
ls ${OUTPUT_DIR}/test/ours_*/renders/

# Check metrics (if --eval was used)
cat ${OUTPUT_DIR}/results.json
```

---

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory

```bash
# Reduce target points
--reinit_target_points 1000000  # Instead of 3500000

# OR reduce batch size (if configurable)
```

### Issue 2: Import Error for depth_reinit

```bash
# Verify file is uploaded correctly
ls -lh /cluster/51/koubaa/abdullah/2DGaussianSplatting/2dGScode/utils/depth_reinit.py

# Check Python can import it
python -c "from utils.depth_reinit import aggregate_depth_points; print('OK')"
```

### Issue 3: Scene Type Not Recognized

```bash
# Error: "Could not recognize scene type!"

# Check that transforms_train.json exists (for Nerfstudio format)
ls /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/transforms_train.json

# OR check for Colmap sparse directory
ls /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/sparse/0/

# Note: The code auto-detects the format. You may need to create a symlink:
cd /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr
ln -s nerfstudio/transforms_undistorted.json transforms_train.json
```

### Issue 4: Training Hangs at Reinitialization

```bash
# Check GPU memory
nvidia-smi

# If stuck, reduce number of views sampled (modify depth_reinit.py if needed)
```

---

## Full Test Script (Copy-Paste Ready)

Save this as `test_depth_reinit.sh`:

```bash
#!/bin/bash

# Configuration
SCENE_ID="0b031f3119"
DATA_TYPE="dslr"  # or "iphone"
DATA_PATH="/cluster/51/koubaa/data/scannet++/data/${SCENE_ID}/${DATA_TYPE}"
OUTPUT_PATH="/cluster/51/koubaa/data/output/abdullah/test_$(date +%Y%m%d_%H%M)/${SCENE_ID}"

# Set images directory based on data type
if [ "$DATA_TYPE" = "dslr" ]; then
    IMAGES_DIR="resized_undistorted_images"
else
    IMAGES_DIR="rgb"
fi

# Activate environment  
conda activate ml3d

# Create output directory
mkdir -p ${OUTPUT_PATH}

# Run training
python 2dGScode/train.py \
  -s ${DATA_PATH} \
  -m ${OUTPUT_PATH} \
  --images ${IMAGES_DIR} \
  --depth_ratio 1 \
  --iterations 2000 \
  --depth_reinit_iters 1000 \
  --reinit_target_points 500000 \
  --eval \
  2>&1 | tee ${OUTPUT_PATH}/train.log

echo "Training complete! Output: ${OUTPUT_PATH}"
```

Run it:

```bash
chmod +x test_depth_reinit.sh
./test_depth_reinit.sh
```

---

## Quick Reference: Available Scenes

Based on your data structure, these scenes are available:

**ScanNet++ Small Scenes (Recommended for testing):**
- `fb5a96b1a2` - Has DSLR + iPhone + Panocam
- `c4c04e6d6c` - Has DSLR + iPhone + Panocam
- `09bced689e` - Has DSLR + iPhone + Panocam

**Data types per scene:**
- `iphone/` - iPhone RGB images (in `rgb/` directory)
- `dslr/` - DSLR undistorted images (in `resized_undistorted_images/`)
- `panocam/` - Panoramic camera images

---

## Expected Timeline

| Test | Duration | Purpose |
|------|----------|---------|
| Test 1: Baseline (10K iter) | ~30-45 min | Verify setup works |
| Test 2: Depth Reinit (8K iter) | ~30-45 min | Verify fixes work |
| Test 3: Full Run (30K iter) | ~2-4 hours | Production test |

---

## Success Criteria

✅ **Depth reinitialization is working correctly if:**

1. Training starts without errors
2. At reinit iterations (2000, 5000), you see depth sampling messages
3. Gaussian count changes dramatically at those iterations
4. Training continues smoothly (no crashes)
5. Final PSNR is comparable or better than baseline
6. No NaN/Inf values in losses

---

## Next Steps After Successful Test

1. **Compare quality:** Run baseline vs depth-reinit on same scene
2. **Visualize:** Use a viewer to inspect the Gaussian point clouds
3. **Benchmark:** Test on multiple scenes to validate consistency
4. **Tune parameters:** Adjust `reinit_target_points` and iteration schedule

---

## Questions?

If you encounter issues:
1. Check the error message carefully
2. Verify the dataset paths exist
3. Check GPU memory with `nvidia-smi`
4. Review the training log for the reinitialization messages

---

## Step 9: Download and Visualize Results

Once training completes, download the results to your local machine for visualization.

### **Download Results**

```bash
# On your local machine
cd /Users/abdullah/Desktop/TUMstudy/ml3d/project/2DGaussianSplatting

# Download entire output directory
rsync -avz --progress \
    -e "ssh -i key" \
    koubaa@ml3d.vc.in.tum.de:/cluster/51/koubaa/data/output/abdullah/test_depth_reinit/ \
    ./results/
```

### **View Rendered Images**

```bash
# Open renders folder
open results/0b031f3119/train/ours_2000/renders/

# Compare with ground truth
open results/0b031f3119/train/ours_2000/gt/

# View depth maps
open results/0b031f3119/train/ours_2000/vis/
```

### **View Point Cloud**

**Option 1: CloudCompare (Recommended)**
```bash
# Install
brew install --cask cloudcompare

# Open point cloud
open -a CloudCompare results/0b031f3119/point_cloud/iteration_2000/point_cloud.ply
```

**Option 2: Web Viewer (No Install)**
- Visit: https://antimatter15.com/splat/
- Drag and drop your `point_cloud.ply` file
- Interact with mouse to rotate/zoom

### **SIBR Viewer (Official Gaussian Splatting Viewer)**

For interactive real-time viewing:

```bash
# Check if SIBR is built in the project
ls SIBR_viewers/

# If available, run:
./SIBR_viewers/bin/SIBR_gaussianViewer_app -m ./results/0b031f3119/
```

---
