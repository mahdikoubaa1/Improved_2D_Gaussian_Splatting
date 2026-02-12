

Paper: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields
Dataset:
● ScanNet++
Modifications:
● Add MCMC (3D Gaussian Splatting as Markov Chain Monte Carlo) training
strategy to improve the Gaussian splats.
● Add the depth Gaussian reinitialization strategy from Mini-Splatting /
Mini-Splatting2 to improve the Gaussian splats
● Add exposure optimization from the paper: A Hierarchical 3D Gaussian
Representation for Real-Time Rendering of Very Large Datasets to handle
exposure changes in the training images. Although the DSLR images are
fixed exposure, the iPhone video data from ScanNet++ has exposure
changes.
● Add monocular normal and depth prior similar to the paper MonoSDF to
improve the reconstruction quality. You can use the newer monocular depth
estimator than the one used in MonoSDF.

My tasks: (Task 2 & 4)
Focus: Shape quality.
Tasks: Implement Depth Reinitialization (Task 2) and Monocular Priors (Task 4).
Rationale: These two are highly related. Both use depth maps (one for initialization, one for loss supervision). Combining them avoids redundant work in processing depth maps.
