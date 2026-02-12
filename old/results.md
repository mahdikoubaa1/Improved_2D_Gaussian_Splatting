Starting: E0-Baseline
==========================================
Optimizing /cluster/51/koubaa/data/output/scannet++/0b031f3119/baseline/
Output folder: /cluster/51/koubaa/data/output/scannet++/0b031f3119/baseline/ [06/02 17:45:02]
Found transforms_train.json file, assuming Blender data set! [06/02 17:45:02]
Reading Training Transforms [06/02 17:45:02]
Number of train frames: 271 [06/02 17:45:02]
Reading Test Transforms [06/02 17:45:43]
Number of test frames: 18 [06/02 17:45:43]
Loading Training Cameras [06/02 17:45:45]
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
 If this is not desired, please explicitly specify '--resolution/-r' as 1 [06/02 17:45:45]
Loading Test Cameras [06/02 17:45:56]
Number of points at initialisation :  44379 [06/02 17:45:57]
initialized exposure for 271 images [06/02 17:45:57]
Training progress:  50%|███████████▌           | 15000/30000 [17:17<18:10, 13.76it/s, Loss=0.03107, distort=0.00021, normal=0.00785, Points=519925]
[ITER 15000] Evaluating test: L1 0.029183627209729616 PSNR 25.575460963779022 LPIPS 0.26746127588881385 SSIM 0.8836404747433132 [06/02 18:03:39]

[ITER 15000] Evaluating train: L1 0.01873668860644102 PSNR 28.362118911743167 LPIPS 0.21251595616340638 SSIM 0.9244833111763001 [06/02 18:03:53]
Training progress: 100%|███████████████████████| 30000/30000 [34:22<00:00, 14.55it/s, Loss=0.02768, distort=0.00014, normal=0.00700, Points=519925]

[ITER 30000] Evaluating test: L1 0.030387309897277087 PSNR 25.34724818335639 LPIPS 0.26824382444222766 SSIM 0.8796670370631747 [06/02 18:20:40]

[ITER 30000] Evaluating train: L1 0.0158357959240675 PSNR 29.58637161254883 LPIPS 0.19660514891147615 SSIM 0.9349522352218629 [06/02 18:20:52]

[ITER 30000] Saving Gaussians [06/02 18:20:52]

Training complete. [06/02 18:20:57]
✓ SUCCESS: 0h 36m

==========================================
Starting: E1-MCMC
==========================================
Optimizing /cluster/51/koubaa/data/output/scannet++/0b031f3119/e1_mcmc/
Output folder: /cluster/51/koubaa/data/output/scannet++/0b031f3119/e1_mcmc/ [06/02 18:21:06]
Found transforms_train.json file, assuming Blender data set! [06/02 18:21:06]
Reading Training Transforms [06/02 18:21:06]
Number of train frames: 271 [06/02 18:21:06]
Reading Test Transforms [06/02 18:21:37]
Number of test frames: 18 [06/02 18:21:37]
Loading Training Cameras [06/02 18:21:39]
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
 If this is not desired, please explicitly specify '--resolution/-r' as 1 [06/02 18:21:39]
Loading Test Cameras [06/02 18:21:49]
Number of points at initialisation :  44379 [06/02 18:21:49]
initialized exposure for 271 images [06/02 18:21:49]
Training progress:  50%|███████████▌           | 15000/30000 [16:30<16:37, 15.03it/s, Loss=0.04082, distort=0.00309, normal=0.00859, Points=300000]
[ITER 15000] Evaluating test: L1 0.03298359892020623 PSNR 24.859545283847385 LPIPS 0.2854314313994513 SSIM 0.8767072690857781 [06/02 18:38:31]

[ITER 15000] Evaluating train: L1 0.024734730273485186 PSNR 26.528350830078125 LPIPS 0.24019038677215576 SSIM 0.9062423825263978 [06/02 18:38:44]
Training progress: 100%|███████████████████████| 30000/30000 [32:03<00:00, 15.60it/s, Loss=0.03037, distort=0.00032, normal=0.00698, Points=300000]

[ITER 30000] Evaluating test: L1 0.032803966146376394 PSNR 24.957599639892578 LPIPS 0.2778348128000895 SSIM 0.8787778913974762 [06/02 18:54:13]

[ITER 30000] Evaluating train: L1 0.01916825119405985 PSNR 27.895216751098634 LPIPS 0.22090822756290437 SSIM 0.9190314531326295 [06/02 18:54:24]

[ITER 30000] Saving Gaussians [06/02 18:54:24]

Training complete. [06/02 18:54:27]
✓ SUCCESS: 0h 33m

==========================================
Starting: E2-Depth
==========================================
Optimizing /cluster/51/koubaa/data/output/scannet++/0b031f3119/e2_depth/
Output folder: /cluster/51/koubaa/data/output/scannet++/0b031f3119/e2_depth/ [06/02 18:54:35]
Found transforms_train.json file, assuming Blender data set! [06/02 18:54:35]
Reading Training Transforms [06/02 18:54:35]
Number of train frames: 271 [06/02 18:54:35]
Reading Test Transforms [06/02 18:55:07]
Number of test frames: 18 [06/02 18:55:07]
Loading Training Cameras [06/02 18:55:09]
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
 If this is not desired, please explicitly specify '--resolution/-r' as 1 [06/02 18:55:09]
Loading Test Cameras [06/02 18:55:18]
Number of points at initialisation :  44379 [06/02 18:55:19]
initialized exposure for 271 images [06/02 18:55:19]
Training progress:   7%|█▌                      | 2000/30000 [02:09<28:45, 16.23it/s, Loss=0.03326, distort=0.00000, normal=0.00000, Points=192637]
[ITER 2000] Depth Reinitialization (Mini-Splatting) [06/02 18:57:28]
[Depth Sampling] Sampled 3499965 points from 271 views [06/02 18:57:39]
Training progress:   7%|█▌                      | 2000/30000 [02:20<28:45, 16.23it/s, Loss=0.03326, distort=0.00000, normal=0.00000, Points=192637][Depth Reinitialization] Replacing ALL Gaussians with 2569424 depth-sampled points [06/02 18:58:38]
[Depth Reinitialization] Complete. Total Gaussians: 2569424 [06/02 18:58:39]
Training progress:  17%|████                    | 5000/30000 [06:20<23:04, 18.05it/s, Loss=0.15654, distort=0.00820, normal=0.00000, Points=457440]
[ITER 5000] Depth Reinitialization (Mini-Splatting) [06/02 19:01:39]
[Depth Sampling] Sampled 3499965 points from 271 views [06/02 19:01:50]
Training progress:  17%|████                    | 5000/30000 [06:40<23:04, 18.05it/s, Loss=0.15654, distort=0.00820, normal=0.00000, Points=457440][Depth Reinitialization] Replacing ALL Gaussians with 2925412 depth-sampled points [06/02 19:02:50]
[Depth Reinitialization] Complete. Total Gaussians: 2925412 [06/02 19:02:50]
Training progress:  33%|███████▋               | 10000/30000 [12:39<25:55, 12.85it/s, Loss=0.23999, distort=0.00011, normal=0.01641, Points=403376]
[ITER 10000] Depth Reinitialization (Mini-Splatting) [06/02 19:07:58]
Training progress:  33%|███████▋               | 10000/30000 [12:50<25:55, 12.85it/s, Loss=0.23999, distort=0.00011, normal=0.01641, Points=403376][Depth Sampling] Sampled 3499965 points from 271 views [06/02 19:08:12]
[Depth Reinitialization] Replacing ALL Gaussians with 3209179 depth-sampled points [06/02 19:09:11]
[Depth Reinitialization] Complete. Total Gaussians: 3209179 [06/02 19:09:12]
Training progress:  50%|███████████▌           | 15000/30000 [19:40<17:49, 14.02it/s, Loss=0.11398, distort=0.00299, normal=0.00914, Points=725710]
[ITER 15000] Evaluating test: L1 0.16235226558314428 PSNR 13.15152793460422 LPIPS 0.4498245699538125 SSIM 0.7269076059261957 [06/02 19:15:10]

[ITER 15000] Evaluating train: L1 0.08143989145755769 PSNR 16.957634735107423 LPIPS 0.39079899787902833 SSIM 0.8029143571853639 [06/02 19:15:23]
Training progress: 100%|███████████████████████| 30000/30000 [39:06<00:00, 12.78it/s, Loss=0.08212, distort=0.00616, normal=0.00870, Points=725710]

[ITER 30000] Evaluating test: L1 0.14955006746782196 PSNR 13.521271652645535 LPIPS 0.4428338259458542 SSIM 0.7418237030506134 [06/02 19:34:46]

[ITER 30000] Evaluating train: L1 0.049213545769453054 PSNR 20.58648147583008 LPIPS 0.3499054551124573 SSIM 0.8476544499397278 [06/02 19:34:58]

[ITER 30000] Saving Gaussians [06/02 19:34:58]

Training complete. [06/02 19:35:04]
✓ SUCCESS: 0h 40m

==========================================
Starting: E4-Mono
==========================================
Optimizing /cluster/51/koubaa/data/output/scannet++/0b031f3119/e4_mono/
Output folder: /cluster/51/koubaa/data/output/scannet++/0b031f3119/e4_mono/ [06/02 19:35:13]
Found transforms_train.json file, assuming Blender data set! [06/02 19:35:13]
Reading Training Transforms [06/02 19:35:13]
Number of train frames: 271 [06/02 19:35:13]
Reading Test Transforms [06/02 19:35:44]
Number of test frames: 18 [06/02 19:35:44]
Loading Training Cameras [06/02 19:35:46]
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
 If this is not desired, please explicitly specify '--resolution/-r' as 1 [06/02 19:35:46]
Loading Test Cameras [06/02 19:35:57]
Number of points at initialisation :  44379 [06/02 19:35:58]
initialized exposure for 271 images [06/02 19:35:58]
Training progress:  50%|███████████▌           | 15000/30000 [17:20<18:13, 13.71it/s, Loss=0.03032, distort=0.00028, normal=0.00797, Points=523914]
[ITER 15000] Evaluating test: L1 0.02903448697179556 PSNR 25.562868224249943 LPIPS 0.26771678609980476 SSIM 0.883667164378696 [06/02 19:53:22]

[ITER 15000] Evaluating train: L1 0.01856757067143917 PSNR 28.486439132690432 LPIPS 0.21226907670497897 SSIM 0.924940288066864 [06/02 19:53:36]
Training progress: 100%|███████████████████████| 30000/30000 [34:07<00:00, 14.65it/s, Loss=0.02749, distort=0.00015, normal=0.00698, Points=523914]

[ITER 30000] Evaluating test: L1 0.030435699141687814 PSNR 25.32126373714871 LPIPS 0.2685718172126346 SSIM 0.879738128847546 [06/02 20:10:26]

[ITER 30000] Evaluating train: L1 0.015584901347756386 PSNR 29.759564208984376 LPIPS 0.195835143327713 SSIM 0.9357229828834535 [06/02 20:10:39]

[ITER 30000] Saving Gaussians [06/02 20:10:39]

Training complete. [06/02 20:10:43]
✓ SUCCESS: 0h 35m

==========================================
Starting: E9-Depth+Mono
==========================================
Optimizing /cluster/51/koubaa/data/output/scannet++/0b031f3119/e9_depth_mono/
Output folder: /cluster/51/koubaa/data/output/scannet++/0b031f3119/e9_depth_mono/ [06/02 20:10:53]
Found transforms_train.json file, assuming Blender data set! [06/02 20:10:53]
Reading Training Transforms [06/02 20:10:53]
Number of train frames: 271 [06/02 20:10:53]
Reading Test Transforms [06/02 20:11:28]
Number of test frames: 18 [06/02 20:11:28]
Loading Training Cameras [06/02 20:11:30]
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
 If this is not desired, please explicitly specify '--resolution/-r' as 1 [06/02 20:11:30]
Loading Test Cameras [06/02 20:11:41]
Number of points at initialisation :  44379 [06/02 20:11:41]
initialized exposure for 271 images [06/02 20:11:41]
Training progress:   7%|█▌                      | 2000/30000 [02:09<28:53, 16.15it/s, Loss=0.03323, distort=0.00000, normal=0.00000, Points=194208]
[ITER 2000] Depth Reinitialization (Mini-Splatting) [06/02 20:13:51]
Training progress:   7%|█▌                      | 2000/30000 [02:20<28:53, 16.15it/s, Loss=0.03323, distort=0.00000, normal=0.00000, Points=194208][Depth Sampling] Sampled 3499965 points from 271 views [06/02 20:14:02]
[Depth Reinitialization] Replacing ALL Gaussians with 2645428 depth-sampled points [06/02 20:15:01]
[Depth Reinitialization] Complete. Total Gaussians: 2645428 [06/02 20:15:02]
Training progress:  17%|████                    | 5000/30000 [06:21<22:23, 18.60it/s, Loss=0.27155, distort=0.00012, normal=0.00000, Points=494123]
[ITER 5000] Depth Reinitialization (Mini-Splatting) [06/02 20:18:03]
[Depth Sampling] Sampled 3499965 points from 271 views [06/02 20:18:15]
Training progress:  17%|████                    | 5000/30000 [06:40<22:23, 18.60it/s, Loss=0.27155, distort=0.00012, normal=0.00000, Points=494123][Depth Reinitialization] Replacing ALL Gaussians with 3226314 depth-sampled points [06/02 20:19:14]
[Depth Reinitialization] Complete. Total Gaussians: 3226314 [06/02 20:19:15]
Training progress:  33%|███████▋               | 10000/30000 [13:51<30:25, 10.95it/s, Loss=0.25038, distort=0.00041, normal=0.01673, Points=590756]
[ITER 10000] Depth Reinitialization (Mini-Splatting) [06/02 20:25:33]
[Depth Sampling] Sampled 3499965 points from 271 views [06/02 20:25:48]
Training progress:  33%|███████▋               | 10000/30000 [14:10<30:25, 10.95it/s, Loss=0.25038, distort=0.00041, normal=0.01673, Points=590756][Depth Reinitialization] Replacing ALL Gaussians with 2903033 depth-sampled points [06/02 20:26:47]
[Depth Reinitialization] Complete. Total Gaussians: 2903033 [06/02 20:26:48]
Training progress:  50%|███████████▌           | 15000/30000 [21:20<19:38, 12.73it/s, Loss=0.12202, distort=0.01087, normal=0.00849, Points=712238]
[ITER 15000] Evaluating test: L1 0.15366211243801645 PSNR 13.4675202899509 LPIPS 0.44076853328280974 SSIM 0.735116723510954 [06/02 20:33:11]

[ITER 15000] Evaluating train: L1 0.111182801425457 PSNR 15.310690116882325 LPIPS 0.3957822024822235 SSIM 0.7807743072509766 [06/02 20:33:25]
Training progress: 100%|███████████████████████| 30000/30000 [43:12<00:00, 11.57it/s, Loss=0.08312, distort=0.00538, normal=0.00956, Points=712238]

[ITER 30000] Evaluating test: L1 0.13579700638850528 PSNR 14.320219251844618 LPIPS 0.4291244265105989 SSIM 0.7507918808195325 [06/02 20:55:14]

[ITER 30000] Evaluating train: L1 0.060866402089595796 PSNR 19.092239379882812 LPIPS 0.36292046904563907 SSIM 0.8307249784469605 [06/02 20:55:26]

[ITER 30000] Saving Gaussians [06/02 20:55:26]

Training complete. [06/02 20:55:32]
✓ SUCCESS: 0h 44m

==========================================
Starting: E12-MCMC+Depth+Mono
==========================================
Optimizing /cluster/51/koubaa/data/output/scannet++/0b031f3119/e12_mcmc_depth_mono/
Output folder: /cluster/51/koubaa/data/output/scannet++/0b031f3119/e12_mcmc_depth_mono/ [06/02 20:55:41]
Found transforms_train.json file, assuming Blender data set! [06/02 20:55:41]
Reading Training Transforms [06/02 20:55:41]
Number of train frames: 271 [06/02 20:55:41]
Reading Test Transforms [06/02 20:56:12]
Number of test frames: 18 [06/02 20:56:12]
Loading Training Cameras [06/02 20:56:14]
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
 If this is not desired, please explicitly specify '--resolution/-r' as 1 [06/02 20:56:14]
Loading Test Cameras [06/02 20:56:23]
Number of points at initialisation :  44379 [06/02 20:56:24]
initialized exposure for 271 images [06/02 20:56:24]
Training progress:   7%|█▋                       | 2000/30000 [02:15<30:49, 15.14it/s, Loss=0.04088, distort=0.00000, normal=0.00000, Points=87858]
[ITER 2000] Depth Reinitialization (Mini-Splatting) [06/02 20:58:39]
[Depth Sampling] Sampled 3499965 points from 271 views [06/02 20:58:50]
Training progress:   7%|█▋                       | 2000/30000 [02:30<30:49, 15.14it/s, Loss=0.04088, distort=0.00000, normal=0.00000, Points=87858][Depth Reinitialization] Replacing ALL Gaussians with 2914305 depth-sampled points [06/02 20:59:49]
[Depth Reinitialization] Complete. Total Gaussians: 2914305 [06/02 20:59:50]
Training progress:  17%|███▊                   | 5000/30000 [09:34<45:48,  9.10it/s, Loss=0.04355, distort=0.00426, normal=0.00000, Points=2914305]
[ITER 5000] Depth Reinitialization (Mini-Splatting) [06/02 21:05:59]
[Depth Sampling] Sampled 3499965 points from 271 views [06/02 21:06:11]
[Depth Reinitialization] Replacing ALL Gaussians with 2990660 depth-sampled points [06/02 21:07:11]
[Depth Reinitialization] Complete. Total Gaussians: 2990660 [06/02 21:07:12]
Training progress:  33%|███████▎              | 10000/30000 [21:53<37:10,  8.97it/s, Loss=0.04882, distort=0.00377, normal=0.01305, Points=2990660]
[ITER 10000] Depth Reinitialization (Mini-Splatting) [06/02 21:18:18]
[Depth Sampling] Sampled 3499965 points from 271 views [06/02 21:18:31]
[Depth Reinitialization] Replacing ALL Gaussians with 2656720 depth-sampled points [06/02 21:19:30]
[Depth Reinitialization] Complete. Total Gaussians: 2656720 [06/02 21:19:31]
Training progress:  50%|███████████           | 15000/30000 [34:24<28:05,  8.90it/s, Loss=0.03766, distort=0.00624, normal=0.01274, Points=2656720]
[ITER 15000] Evaluating test: L1 0.03910771612491872 PSNR 23.685729026794434 LPIPS 0.3153699363271395 SSIM 0.8621930215093824 [06/02 21:31:12]

[ITER 15000] Evaluating train: L1 0.027834712341427804 PSNR 25.509709930419923 LPIPS 0.2645799100399017 SSIM 0.8942586779594421 [06/02 21:31:27]
Training progress: 100%|████████████████████| 30000/30000 [1:00:23<00:00,  8.28it/s, Loss=0.02741, distort=0.00042, normal=0.00940, Points=2656720]

[ITER 30000] Evaluating test: L1 0.03431690318716897 PSNR 24.65020540025499 LPIPS 0.28530709197123844 SSIM 0.8729314274258083 [06/02 21:57:09]

[ITER 30000] Evaluating train: L1 0.017846088856458664 PSNR 28.707637786865234 LPIPS 0.21524638235569002 SSIM 0.9245960116386414 [06/02 21:57:22]

[ITER 30000] Saving Gaussians [06/02 21:57:22]

Training complete. [06/02 21:57:44]



| Experiment | Status  | Test L1 | Test PSNR | Test LPIPS | Test SSIM | Train L1 | Train PSNR | Train LPIPS | Train SSIM |
|------------|---------|---------|-----------|------------|-----------|----------|------------|-------------|------------|
| E0 Baseline | SUCCESS | 0.03039 | 25.35 | 0.26824 | 0.87967 | 0.01584 | 29.59 | 0.19661 | 0.93495 |
| E1 MCMC | SUCCESS | 0.03280 | 24.96 | 0.27783 | 0.87878 | 0.01917 | 27.90 | 0.22091 | 0.91903 |
| E2 Depth | SUCCESS | 0.14955 | 13.52 | 0.44283 | 0.74182 | 0.04921 | 20.59 | 0.34991 | 0.84765 |
| E4 Mono | SUCCESS | 0.03044 | 25.32 | 0.26857 | 0.87974 | 0.01558 | 29.76 | 0.19584 | 0.93572 |
| E9 Depth+Mono | SUCCESS | 0.13580 | 14.32 | 0.42912 | 0.75079 | 0.06087 | 19.09 | 0.36292 | 0.83072 |
| E12 MCMC+Depth+Mono | SUCCESS | 0.03432 | 24.65 | 0.28531 | 0.87293 | 0.01785 | 28.71 | 0.21525 | 0.92460 |