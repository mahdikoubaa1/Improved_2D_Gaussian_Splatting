# Mini-Splatting2: Building 360 Scenes within Minutes  
## via Aggressive Gaussian Densification

**Guangchi Fang, Bing Wang**  
The Hong Kong Polytechnic University  
📧 guangchi.fang@gmail.com, bingwang@polyu.edu.hk

---

**Mini-Splatting2**  
- **2min48s**, **0.7M**, **25.2 dB**

**GT | 3DGS-accel**  
- **11min47s**, **4.0M**, **25.2 dB**  
- 8K iter (79s), 0.7M  
- 3K iter (36s), 5.1M

---

## Figure 1
Through aggressive Gaussian densification, **Mini-Splatting2** efficiently reconstructs dense 3D Gaussian representations early in training. With subsequent simplification and optimization, high-quality Gaussian models are produced in just a few minutes.

Gaussian centers of the bicycle model are visualized as:
- Dense point clouds (3K iterations, 5.1M Gaussians)
- High-quality point clouds (8K iterations, 0.7M Gaussians)

Rendering results are compared between **Mini-Splatting2 (3 min)** and **3DGS-accel (12 min)**.

---

## Abstract

Fast scene optimization remains a core challenge for Gaussian Splatting. By analyzing geometry modeling, we observe that dense point clouds can be reconstructed early through Gaussian representations.  

This insight leads to **aggressive Gaussian densification**, significantly increasing critical Gaussians to capture dense geometry early. Integrated with Mini-Splatting’s densification and simplification framework, this enables rapid convergence without quality loss.

We further introduce **visibility culling**, leveraging per-view Gaussian importance to accelerate optimization.  
**Mini-Splatting2** achieves a strong balance between optimization time, number of Gaussians, and rendering quality. Code will be released regardless of acceptance.

---

## 1. Introduction

3D Gaussian Splatting (3DGS) has emerged as a breakthrough for novel view synthesis, offering fast optimization and real-time rendering. Compared to NeRF-based methods, 3DGS benefits from Gaussian representations and efficient rasterization.

Despite these advances, typical 360° scenes still require **10+ minutes** to optimize. This work focuses on accelerating Gaussian Splatting while maintaining rendering quality.

Key observations:
- Geometry modeling dominates optimization time
- Progressive densification is inefficient for fast convergence
- Early Gaussian representations already encode usable geometry

---

## Contributions

- Analysis of 3DGS geometry modeling revealing early dense point reconstruction
- **Aggressive Gaussian densification** to accelerate convergence
- **Visibility Gaussian culling** using per-view importance
- Extensive experiments demonstrating improved speed–quality trade-offs

---

## 2. Related Work

### Gaussian Splatting
3DGS replaces implicit NeRFs with explicit Gaussian primitives, enabling efficient rasterization and real-time rendering.

Applications include:
- Surface reconstruction
- Content generation
- SLAM and robotics
- Industrial and commercial platforms

### Gaussian Densification
Vanilla 3DGS relies on adaptive densification via gradient-based cloning and splitting. Subsequent works improve selection strategies, multi-view constraints, and optimization techniques—but often increase training time.

### Accelerated Gaussian Optimization
Recent approaches include:
- Learning-based priors
- Pipeline optimizations (e.g., 3DGS-accel)
- Specialized optimizers (e.g., 3DGS-LM)

Our approach instead focuses on **geometry-driven densification efficiency**.

---

## 3. Bridging 3DGS and Point Reconstruction

### Gaussian Representation

A scene is represented by Gaussians:

- Center: \( p_i \in \mathbb{R}^3 \)
- Opacity: \( \alpha_i \in [0,1] \)
- Covariance: \( \Sigma_i \in \mathbb{R}^{3 \times 3} \)
- Color: spherical harmonics coefficients

Rendering equation:

\[
c(x) = \sum_{i=1}^{N} w_i c_i,\quad
w_i = T_i \alpha_i G_i^{2D}(x)
\]

---

### Geometry Modeling

Starting from sparse SfM points, Gaussian densification gradually captures geometry. However, progressive densification produces irregular distributions and slow convergence.

Mini-Splatting improves this with:
- Better spatial reorganization
- Aggressive simplification

Yet densification before 15K iterations remains costly.

---

### Visual Analysis of Gaussians

Gaussian centers correlate strongly with scene geometry.

Observations:
- Early iterations (1K–3K): incomplete geometry, many floaters
- Later iterations: convergence toward surfaces
- Early-stage Gaussians still allow usable point reconstruction

---

### Point Cloud Reconstruction via Gaussians

Depth is estimated using:

\[
d(x) = d^{mid}_{i_{max}}(x), \quad i_{max} = \arg\max_i w_i
\]

Depth maps are reprojected into world space to form point clouds.

**Key insight:**  
Dense point clouds can be reconstructed **early**, even from imperfect Gaussians.

---

## 4. Aggressive Gaussian Densification

### Progressive vs. Aggressive Densification

- **Progressive**: slow, restrained growth over many iterations
- **Aggressive**: rapid densification in early stages

Goal: shorten densification period and accelerate convergence.

---

### Critical Gaussian Identification

Approximate alpha blending:

\[
c(x) \approx w_{i_{max}} c_{i_{max}}
\]

Critical Gaussians are selected based on maximum blending weight \( w_{i_{max}} \), filtered using a **0.99 quantile threshold**.

---

### Aggressive Gaussian Clone

To avoid optimization bias:
- Clone only (no split)
- Preserve opacity via:

\[
\alpha_{new} = 1 - \sqrt{1 - \alpha_{old}}
\]

- Covariance updated following state-transition rules

---

### Overall Pipeline

- Progressive densification retained
- Aggressive densification every 250 iterations (from 500 iters)
- Depth reinitialization at 2K iterations
- Densification ends at 3K iterations
- Total optimization: 18K iterations

---

## 5. Visibility Gaussian Culling

### Gaussian Visibility

For view \( k \):

\[
I_i^k = \sum_j w_{ij}^k
\]

Visibility mask:

\[
V_i^k = \mathbb{I}(I_i^k > \tau)
\]

where \( \tau \) is the 0.99 quantile.

---

### Precomputed Visibility

- Visibility updated during densification and simplification
- Non-visible Gaussians excluded from rasterization
- Active between iterations 500–13K

Result: **significant reduction in computation**

---

## 6. Experiments

### Datasets
- Mip-NeRF360
- Tanks & Temples
- Deep Blending

Metrics:
- PSNR
- SSIM
- LPIPS

---

### Quantitative Results

Mini-Splatting2:
- Comparable quality to 3DGS / 3DGS-accel
- **Up to 7.6× faster** than 3DGS
- **2.8× faster** than 3DGS-accel

---

### Qualitative Results

- Sharper foregrounds
- Fewer floaters
- Better geometry consistency
- Background trade-offs in sparse variants

---

### Resource Consumption

- Slightly higher peak memory due to visibility
- Manageable on low-cost GPUs
- SH disabling reduces memory pressure

---

## 7. Limitations and Future Work

- Memory constraints not explicitly optimized
- Potential for efficient dense point cloud reconstruction
- Future integration with multi-view geometry techniques

---

## 8. Conclusion

**Mini-Splatting2** introduces:
- Aggressive Gaussian densification
- Visibility Gaussian culling

Result:
- Dramatically faster optimization
- High rendering quality
- Strong baseline for future Gaussian Splatting research

---

## References

*(References [1]–[44] preserved as-is from the original text)*

---

## Appendix

### A. Additional Ablation Studies

#### A.1 Critical Gaussian Identification
Maximum blending weight \( w_{i_{max}} \) performs best compared to random, opacity, or accumulated weights.

#### A.2 Aggressive Gaussian Clone
Modified clone operations outperform vanilla split/clone by avoiding opacity bias and optimization instability.
