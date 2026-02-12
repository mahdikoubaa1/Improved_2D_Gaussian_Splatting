# 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

[cite_start]**Authors:** Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao [cite: 3, 4, 5, 24, 25, 26]
[cite_start]**Conference:** SIGGRAPH Conference Papers '24 [cite: 29]

---

## Abstract

[cite_start]3D Gaussian Splatting (3DGS) has recently revolutionized radiance field reconstruction, achieving high quality novel view synthesis and fast rendering speed[cite: 19]. [cite_start]However, 3DGS fails to accurately represent surfaces due to the multi-view inconsistent nature of 3D Gaussians[cite: 20]. [cite_start]We present 2D Gaussian Splatting (2DGS), a novel approach to model and reconstruct geometrically accurate radiance fields from multi-view images[cite: 21]. [cite_start]Our key idea is to collapse the 3D volume into a set of 2D oriented planar Gaussian disks[cite: 22]. [cite_start]Unlike 3D Gaussians, 2D Gaussians provide view-consistent geometry while modeling surfaces intrinsically[cite: 23]. [cite_start]To accurately recover thin surfaces and achieve stable optimization, we introduce a perspective-accurate 2D splatting process utilizing ray-splat intersection and rasterization[cite: 23, 31]. [cite_start]Additionally, we incorporate depth distortion and normal consistency terms to further enhance the quality of the reconstructions[cite: 32]. [cite_start]We demonstrate that our differentiable renderer allows for noise-free and detailed geometry reconstruction while maintaining competitive appearance quality, fast training speed, and real-time rendering[cite: 33].

---

## 1. Introduction

[cite_start]Photorealistic novel view synthesis (NVS) and accurate geometry reconstruction stand as pivotal long-term objectives in computer graphics and vision[cite: 41]. [cite_start]Recently, 3D Gaussian Splatting (3DGS) has emerged as an appealing alternative to implicit and feature grid-based representations in NVS, due to its real-time photorealistic NVS results at high resolutions[cite: 42, 46, 48]. [cite_start]Rapidly evolving, 3DGS has been quickly extended with respect to multiple domains, including anti-aliasing rendering, material modeling, dynamic scene reconstruction, and animatable avatar creation[cite: 49, 50, 51]. [cite_start]Nevertheless, it falls short in capturing intricate geometry since the volumetric 3D Gaussian, which models the complete angular radiance, conflicts with the thin nature of surfaces[cite: 52].

[cite_start]On the other hand, earlier works have shown surfels (surface elements) to be an effective representation of complex geometry[cite: 53, 54]. [cite_start]Surfels approximate the object surface locally with shape and shade attributes and can be derived from known geometry[cite: 55]. [cite_start]They are widely used in SLAM and other robotics tasks as an efficient geometry representation[cite: 56]. [cite_start]Subsequent advancements have incorporated surfels into a differentiable framework[cite: 57]. [cite_start]However, these methods typically require ground truth (GT) geometry, depth sensor data, or operate under constrained scenarios with known lighting[cite: 58].

[cite_start]Inspired by these works, we propose 2D Gaussian Splatting for 3D scene reconstruction and novel view synthesis that combines the benefits of both worlds, while overcoming their limitations[cite: 59]. [cite_start]Unlike 3DGS, our approach represents a 3D scene with 2D Gaussian primitives, each defining an oriented elliptical disk[cite: 60]. [cite_start]The significant advantage of 2D Gaussian over its 3D counterpart lies in the accurate geometry representation during rendering[cite: 61]. [cite_start]Specifically, 3DGS evaluates a Gaussian's value at the intersection between a pixel ray and a 3D Gaussian, which leads to inconsistency depth when rendered from different viewpoints[cite: 62]. [cite_start]In contrast, our method utilizes explicit ray-splat intersection, resulting in a perspective correct splatting, as illustrated in Figure 2, which in turn significantly improves reconstruction quality[cite: 63]. [cite_start]Furthermore, the inherent surface normals in 2D Gaussian primitives enable direct surface regularization through normal constraints[cite: 64]. [cite_start]In contrast with surfels-based models, our 2D Gaussians can be recovered from unknown geometry with gradient-based optimization[cite: 65, 66].

[cite_start]While our 2D Gaussian approach excels in geometric modeling, optimizing solely with photometric losses can lead to noisy reconstructions, due to the inherently unconstrained nature of 3D reconstruction tasks[cite: 67]. [cite_start]To enhance reconstructions and achieve smoother surfaces, we introduce two regularization terms: depth distortion and normal consistency[cite: 68]. [cite_start]The depth distortion term concentrates 2D primitives distributed within a tight range along the ray, addressing the rendering process's limitation where the distance between Gaussians is ignored[cite: 69]. [cite_start]The normal consistency term minimizes discrepancies between the rendered normal map and the gradient of the rendered depth, ensuring alignment between the geometries defined by depth and normals[cite: 70]. [cite_start]Employing these regularizations in combination with our 2D Gaussian model enables us to extract highly accurate surface meshes, as demonstrated in Figure 1[cite: 71].

**In summary, we make the following contributions:**
* [cite_start]We present a highly efficient differentiable 2D Gaussian renderer, enabling perspective-correct splatting by leveraging 2D surface modeling, ray-splat intersection, and volumetric integration[cite: 79].
* [cite_start]We introduce two regularization losses for improved and noise-free surface reconstruction[cite: 80].
* [cite_start]Our approach achieves state-of-the-art geometry reconstruction and NVS results compared to other explicit representations[cite: 81].

> [cite_start]**Figure 1:** Our method, 2DGS, (a) optimizes a set of 2D oriented disks to represent and reconstruct a complex real-world scene from multi-view RGB images[cite: 15]. [cite_start]These optimized 2D disks are tightly aligned to the surfaces[cite: 16]. (b) [cite_start]With 2D Gaussian splatting, we allow real-time rendering of high quality novel view images with view consistent normals and depth maps[cite: 17]. (c) [cite_start]Finally, our method provides detailed and noise-free triangle mesh reconstruction from the optimized 2D disks[cite: 18].

---

## 2. Related Work

### 2.1 Novel view synthesis
[cite_start]Significant advancements have been achieved in NVS, particularly since the introduction of Neural Radiance Fields (NeRF)[cite: 84]. [cite_start]NeRF employs a multi-layer perceptron (MLP) to represent geometry and view-dependent appearance, optimized via volume rendering to deliver exceptional rendering quality[cite: 85]. [cite_start]Post-NeRF developments have further enhanced its capabilities, tackling aliasing issues and improving rendering efficiency through distillation and baking[cite: 86, 88, 89]. [cite_start]Moreover, the training and representational power of NeRF have been enhanced using feature-grid based scene representations[cite: 90].

[cite_start]Recently, 3D Gaussian Splatting (3DGS) has emerged, demonstrating impressive real-time NVS results[cite: 92]. [cite_start]This method has been quickly extended to multiple domains[cite: 93]. [cite_start]In this work, we propose to "flatten" 3D Gaussians to 2D Gaussian primitives to better align their shape with the object surface[cite: 94]. [cite_start]Combined with two novel regularization losses, our approach reconstructs surfaces more accurately than 3DGS while preserving its high-quality and real-time rendering capabilities[cite: 95].

### 2.2 3D reconstruction

[cite_start]3D Reconstruction from multi-view images has been a long-standing goal in computer vision[cite: 97]. [cite_start]Multi-view stereo based methods rely on a modular pipeline that involves feature matching, depth prediction, and fusion[cite: 98, 100]. [cite_start]In contrast, recent neural approaches represent surface implicitly via an MLP, extracting surfaces post-training via the Marching Cube algorithm[cite: 100, 101, 102]. [cite_start]Further advancements integrated implicit surfaces with volume rendering, achieving detailed surface reconstructions from RGB images[cite: 102, 103]. [cite_start]These methods have been extended to large-scale reconstructions via additional regularization[cite: 104].

Despite these impressive developments, efficient large-scale scene reconstruction remains a challenge. [cite_start]For instance, Neuralangelo requires 128 GPU hours for reconstructing a single scene from the Tanks and Temples Dataset[cite: 106, 107]. [cite_start]In this work, we introduce 2D Gaussian splatting, a method that significantly accelerates the reconstruction process[cite: 108]. [cite_start]It achieves similar or slightly better results compared to previous implicit neural surface representations, while being an order of magnitude faster[cite: 109].

### 2.3 Differentiable Point-based Graphics
[cite_start]Differentiable point-based rendering has been explored extensively due to its efficiency and flexibility in representing intricate structures[cite: 110]. [cite_start]Notably, NPBG rasterizes point cloud features onto an image plane[cite: 112]. [cite_start]DSS focuses on optimizing oriented point clouds from multi-view images under known lighting conditions[cite: 113]. [cite_start]Pulsar introduces a tile-based acceleration structure for more efficient rasterization[cite: 114]. [cite_start]More recently, 3DGS optimizes anisotropic 3D Gaussian primitives, demonstrating real-time photorealistic NVS results[cite: 115]. [cite_start]Despite these advances, using point-based representations from unconstrained multi-view images remains challenging[cite: 116]. [cite_start]In this paper, we demonstrate detailed surface reconstruction using 2D Gaussian primitives[cite: 117]. [cite_start]We also highlight the critical role of additional regularization losses in optimization[cite: 118].

### 2.4 Concurrent works
[cite_start]Since 3DGS was introduced, it has been rapidly adapted across multiple domains[cite: 120]. We review the closest work in inverse rendering. [cite_start]These works extend 3DGS by modeling normals as additional attributes of 3D Gaussian primitives[cite: 121, 122]. [cite_start]Our approach, in contrast, inherently defines normals by representing the tangent space of the 3D surface using 2D Gaussian primitives, aligning them more closely with the underlying geometry[cite: 123]. [cite_start]Additionally, none of these works specifically target surface reconstruction, the primary focus of our work[cite: 125].

[cite_start]We also highlight the distinctions between our method and concurrent works SuGaR and NeuSG[cite: 126]. [cite_start]Unlike SuGaR, which approximates 2D Gaussians with 3D Gaussians, our method directly employs 2D Gaussians, simplifying the process and enhancing the resulting geometry without additional mesh refinement[cite: 127, 130]. [cite_start]NeuSG optimizes 3D Gaussian primitives and an implicit SDF network jointly, while our approach leverages 2D Gaussian primitives for surface approximation, offering a faster and conceptually simpler solution[cite: 131].

> **Figure 2:** Comparison of 3DGS and 2DGS. [cite_start]3DGS utilizes different intersection planes for value evaluation when viewing from different viewpoints, resulting in inconsistency[cite: 76]. [cite_start]Our 2DGS provides multi-view consistent value evaluations[cite: 77].

---

## 3. 3D Gaussian Splatting

[cite_start]Kerbl et al. propose to represent 3D scenes with 3D Gaussian primitives and render images using differentiable volume splatting[cite: 133]. [cite_start]Specifically, 3DGS explicitly parameterizes Gaussian primitives via 3D covariance matrix $\Sigma$ and their location $p_k$[cite: 134]:

[cite_start]$$G(p)=exp(-\frac{1}{2}(p-p_{k})^{\top}\Sigma^{-1}(p-p_{k}))$$ [cite: 135]

[cite_start]where the covariance matrix $\Sigma=RSS^{T}R^{T}$ is factorized into a scaling matrix $S$ and a rotation matrix $R$[cite: 137]. To render an image, the 3D Gaussian is transformed into the camera coordinates with world-to-camera transform matrix $W$ and projected to image plane via a local affine transformation $J$:

[cite_start]$$\Sigma^{\prime}=JW\Sigma W^{\top}J^{\top}$$ [cite: 139]

[cite_start]By skipping the third row and column of $\Sigma^{\prime}$, we obtain a 2D Gaussian $G^{2D}$ with covariance matrix $\Sigma^{2D}$[cite: 141]. Next, 3DGS employs volumetric alpha blending to integrate alpha-weighted appearance from front to back:

[cite_start]$$C(x) = \sum_{k=1} c_{k} \alpha_{k} G^{2D}(x) \prod_{j=1}^{k-1} (1-\alpha_{j} G^{2D}(x))$$ [cite: 143, 146]

[cite_start]where $k$ is the index of the Gaussian primitives, $\alpha_{k}$ denotes the alpha values and $c_{k}$ is the view-dependent appearance[cite: 148]. [cite_start]The attributes of 3D Gaussian primitives are optimized using a photometric loss[cite: 149].

[cite_start]**Challenges in Surface Reconstruction.** Reconstructing surfaces using 3D Gaussian modeling and splatting faces several challenges[cite: 150]. [cite_start]First, the volumetric radiance representation of 3D Gaussians conflicts with the thin nature of surfaces[cite: 151]. [cite_start]Second, 3DGS does not natively model surface normals, essential for high-quality surface reconstruction[cite: 152]. [cite_start]Third, the rasterization process in 3DGS lacks multi-view consistency, leading to varied 2D intersection planes for different viewpoints, as illustrated in Figure 2 (a)[cite: 153]. [cite_start]Additionally, using an affine matrix for transforming a 3D Gaussian into ray space only yields accurate projections near the center, compromising on perspective accuracy around surrounding regions, often resulting in noisy reconstructions[cite: 154, 155].

---

## 4. 2D Gaussian Splatting

[cite_start]To accurately reconstruct geometry while maintaining high-quality novel view synthesis, we present differentiable 2D Gaussian splatting (2DGS)[cite: 157].

> **Figure 3:** Illustration of 2D Gaussian Splatting. [cite_start]2D Gaussian Splats are elliptical disks characterized by a center point $p_{k}$ tangential vectors $t_{u}$ and $t_{v}$, and two scaling factors ($s_{u}$ and $s_{v}$) control the variance[cite: 170]. [cite_start]Their elliptical projections are sampled through the ray-splat intersection (Section 4.2) and accumulated via alpha-blending in image space[cite: 171]. [cite_start]2DGS reconstructs surface attributes such as colors, depths, and normals through gradient descent[cite: 172].

### 4.1 Modeling

[cite_start]Unlike 3DGS, which models the entire angular radiance in a blob, we simplify the 3-dimensional modeling by adopting "flat" 2D Gaussians embedded in 3D space[cite: 174]. [cite_start]With 2D Gaussian modeling, the primitive distributes densities within a planar disk, defining the normal as the direction of the steepest change of density[cite: 175]. [cite_start]This feature enables better alignment with thin surfaces[cite: 176].

[cite_start]As illustrated in Figure 3, our 2D splat is characterized by its central point $p_{k}$, two principal tangential vectors $t_{u}$ and $t_{v}$, and a scaling vector $S=(s_{u},s_{v})$ that controls the variances of the 2D Gaussian[cite: 179]. [cite_start]Notice that the primitive normal is defined by two orthogonal tangential vectors $t_{w}=t_{u}\times t_{v}$[cite: 180]. [cite_start]We can arrange the orientation into a $3\times3$ rotation matrix $R=[t_{u},t_{v},t_{w}]$ and the scaling factors into a $3\times3$ diagonal matrix $S$ whose last entry is zero[cite: 181]. A 2D Gaussian is therefore defined in a local tangent plane in world space, which is parameterized:

[cite_start]$$P(u,v)=p_{k}+s_{u}t_{u}u+s_{v}t_{v}v=H(u,v,1,1)^{T}$$ [cite: 183]

[cite_start]$$H = \begin{bmatrix} s_u t_u & s_v t_v & 0 & p_k \\ 0 & 0 & 0 & 1 \end{bmatrix}$$ [cite: 184, 185, 186, 187]

[cite_start]where $H\in4\times4$ is a homogeneous transformation matrix representing the geometry of the 2D Gaussian[cite: 190]. For the point $u=(u,v)$ in uv space, its 2D Gaussian value can then be evaluated by standard Gaussian:

[cite_start]$$G(u) = exp(-\frac{u^2+v^2}{2})$$ [cite: 192, 193]

[cite_start]The center $p_{k}$, scaling $(s_{u},s_{v})$, and the rotation $(t_{u},t_{v})$ are learnable parameters[cite: 195]. [cite_start]Following 3DGS, each 2D Gaussian primitive has opacity $\alpha$ and view-dependent appearance $c$ parameterized with spherical harmonics[cite: 196].

### 4.2 Splatting

[cite_start]One common strategy for rendering 2D Gaussians is to project the 2D Gaussian primitives onto the image space using the affine approximation of the perspective projection[cite: 199]. [cite_start]However, this projection is only accurate at the center of the Gaussian and has increasing approximation error with increased distance to the center[cite: 200]. To address this problem, we utilize an explicit ray-splat intersection inspired by [Sigg et al. [cite_start]2006][cite: 210].

[cite_start]**Ray-splat Intersection.** We efficiently locate the ray-splat intersections by finding the intersection of three non-parallel planes[cite: 211]. [cite_start]Given an image coordinate $x=(x,y)$, we parameterize the ray of a pixel in the projective space as the intersection of two orthogonal planes: the x-plane and the y-plane[cite: 212]. The x-plane is defined by a normal vector (-1,0,0) and an offset x. The x-plane can be represented as a 4D homogeneous plane $h_{x}=(-1,0,0,x)^{T}$. Similarly, the y-plane is $h_{y}=(0,-1,0,y)^{T}$. [cite_start]Thus, the ray $x=(x,y)$ is determined by the intersection of the two planes[cite: 213, 214, 215].

[cite_start]Next, we transform both planes into the local coordinates of the 2D Gaussian primitives, the uv-coordinate system[cite: 216]. Applying $M=(WH)^{-1}$ is equivalent to $(WH)^{T}$, eliminating explicit matrix inversion and yielding:

[cite_start]$$h_{u}=(WH)^{T}h_{x} \quad h_{v}=(WH)^{T}h_{y}$$ [cite: 219]

[cite_start]As introduced in Section 4.1, points on the 2D Gaussian plane are represented as $(u, v, 1, 1)$[cite: 221]. At the same time, the intersection point should fall in the transformed x-plane and y-plane. [cite_start]Thus, $h_{u}\cdot(u,v,1,1)^{T}=h_{v}\cdot(u,v,1,1)^{T}=0$[cite: 222]. [cite_start]This leads to an efficient solution for the intersection point $u(x)$[cite: 223].

[cite_start]$$u(x) = \frac{1}{h_{u}^1 h_{v}^2 - h_{u}^2 h_{v}^1} \begin{bmatrix} h_{u}^2 h_{v}^4 - h_{u}^4 h_{v}^2 \\ h_{u}^4 h_{v}^1 - h_{u}^1 h_{v}^4 \end{bmatrix}$$ [cite: 225, 226, 227] (Approximated from text context)

[cite_start]where $h^i$ are the i-th parameter of the 4D plane[cite: 229]. [cite_start]Once we obtain the local coordinates $(u, v)$, we can calculate the depth $z$ of the intersected points and evaluate the Gaussian value[cite: 231].

[cite_start]**Degenerate Solutions.** When a 2D Gaussian is observed from a slanted viewpoint, it degenerates to a line in screen space[cite: 233]. To deal with these cases and stabilize optimization, we employ the object-space low-pass filter introduced in [Botsch et al. 2005]:

[cite_start]$$\hat{G}(x)=max\{G(u(x)),G(\frac{x-c}{\sigma})\}$$ [cite: 236]

where $u(x)$ is given by Eq. [cite_start]10 and $c$ is the projection of center $p_{k}$[cite: 238].

[cite_start]**Rasterization.** We follow a similar rasterization process as in 3DGS[cite: 240]. Finally, volumetric alpha blending is used to integrate alpha-weighted appearance from front to back:

[cite_start]$$c(x)=\sum_{i=1}c_{i}\alpha_{i}\hat{G}_{i}(u(x))\prod_{j=1}^{i-1}(1-\alpha_{j}\hat{G}_{j}(u(x)))$$ [cite: 244]

---

## 5. Training

[cite_start]Our 2D Gaussian method, while effective in geometric modeling, can result in noisy reconstructions when optimized only with photometric losses[cite: 248]. [cite_start]To mitigate this issue and improve the geometry reconstruction, we introduce two regularization terms: depth distortion and normal consistency[cite: 249].

**Depth Distortion.** We take inspiration from Mip-NeRF360 and propose a depth distortion loss to concentrate the weight distribution along the rays by minimizing the distance between the ray-splat intersections:

[cite_start]$$\mathcal{L}_{d}=\sum_{i,j}\omega_{i}\omega_{j}|z_{i}-z_{j}|$$ [cite: 255]

[cite_start]where $\omega_{i}=\alpha_{i}\hat{G}_{i}(u(x))\prod_{j=1}^{i-1}(1-\alpha_{j}\hat{G}_{j}(u(x)))$ is the blending weight of the i-th intersection and $z_{i}$ is the depth of the intersection points[cite: 257].

**Normal Consistency.** We align the splats' normal with the gradients of the depth maps as follows:

[cite_start]$$\mathcal{L}_{n} = \sum_{i} \omega_{i} (1-n_{i}^{\top}N)$$ [cite: 265]

[cite_start]where $i$ indexes over intersected splats along the ray, $\omega_i$ denotes the blending weight, $n_{i}$ represents the normal of the splat, and $N$ is the normal estimated by the gradient of the depth map[cite: 267]. [cite_start]Specifically, $N$ is computed with finite differences from nearby depth points[cite: 268]:

[cite_start]$$N(x,y) = \frac{\nabla_x P_s \times \nabla_y P_s}{||\nabla_x P_s \times \nabla_y P_s||}$$ [cite: 269, 270] (Approximated from text context)

**Final Loss.** We minimize the following loss function:

[cite_start]$$\mathcal{L}=\mathcal{L}_{c}+\alpha\mathcal{L}_{d}+\beta\mathcal{L}_{n}$$ [cite: 275]

[cite_start]where $\mathcal{L}_{c}$ is an RGB reconstruction loss combining $\mathcal{L}_{1}$ with the D-SSIM term[cite: 276]. [cite_start]We set $\alpha=1000$ for bounded scenes, $\alpha=100$ for unbounded scenes, and $\beta=0.05$ for all scenes[cite: 277].

---

## 6. Experiments

[cite_start]We evaluate the performance of our method on various datasets, including DTU, Tanks and Temples, and Mip-NeRF360[cite: 313, 314].

### 6.1 Implementation
[cite_start]We implement our 2D Gaussian Splatting with custom CUDA kernels, building upon the framework of 3DGS[cite: 282]. [cite_start]We extend the renderer to output depth distortion maps, depth maps and normal maps for regularizations[cite: 283]. [cite_start]We use the same training process for 3DGS and SuGaR for a comparison[cite: 318, 319]. [cite_start]For Mesh Extraction, we utilize truncated signed distance fusion (TSDF) to fuse the reconstruction depth maps[cite: 288].

### 6.2 Comparison

[cite_start]**Geometry Reconstruction.** Our 2DGS achieves the highest reconstruction accuracy among other methods and provides 100x speed up compared to the SDF based baselines[cite: 303].

[cite_start]**Table 1: Quantitative comparison on the DTU Dataset** [cite: 302]

| Method | Mean | Time |
| :--- | :--- | :--- |
| **Implicit** | | |
| NeRF | 1.49 | >12h |
| VolSDF | 0.86 | >12h |
| NeuS | 0.84 | >12h |
| **Explicit** | | |
| 3DGS | 1.96 | 11.2 m |
| SuGaR | 1.33 | ~1h |
| **2DGS-15k (Ours)** | **0.83** | **5.5 m** |
| **2DGS-30k (Ours)** | **0.80** | **10.9 m** |

[cite_start][cite: 304] (Data summarized from table content)

[cite_start]**Table 2: Quantitative results on the Tanks and Temples Dataset** [cite: 305]

| Method | Mean F1 | Time |
| :--- | :--- | :--- |
| NeuS | 0.38 | >24h |
| Geo-Neus | 0.35 | >24h |
| Neurlangelo | 0.50 | >24h |
| SuGaR | 0.19 | >1h |
| 3DGS | 0.09 | 14.3 m |
| **Ours** | **0.32** | **15.5 m** |

[cite_start][cite: 307]

[cite_start]**Table 3: Performance comparison on DTU dataset** [cite: 308]

| Method | CD | PSNR | Time | MB (Storage) |
| :--- | :--- | :--- | :--- | :--- |
| 3DGS | 1.96 | 35.76 | 11.2 m | 113 |
| SuGaR | 1.33 | 34.57 | ~1h | 1247 |
| 2DGS-15k (Ours) | 0.83 | 33.42 | 5.5 m | 52 |
| 2DGS-30k (Ours) | 0.80 | 34.52 | 10.9 m | 52 |

[cite_start][cite: 310]

[cite_start]**Appearance Reconstruction.** Our method consistently achieves competitive NVS results across state-of-the-art techniques while providing geometrically accurate surface reconstruction[cite: 365, 374].

[cite_start]**Table 4: Quantitative results on Mip-NeRF 360 dataset** [cite: 353]

| Method | Outdoor (PSNR / SSIM / LPIPS) | Indoor (PSNR / SSIM / LPIPS) |
| :--- | :--- | :--- |
| SuGaR | 22.93 / 0.629 / 0.356 | 29.43 / 0.906 / 0.225 |
| 3DGS | 24.64 / 0.731 / 0.234 | 30.41 / 0.920 / 0.189 |
| **2DGS (Ours)** | **24.34 / 0.717 / 0.246** | **30.40 / 0.916 / 0.195** |

[cite_start][cite: 356] (Data summarized)

### 6.3 Ablations

We isolate the design choices and measure their effect on reconstruction quality.

[cite_start]**Table 5: Quantitative studies for regularization terms and mesh extraction (DTU)** [cite: 328]

| Configuration | Accuracy | Completion | Average |
| :--- | :--- | :--- | :--- |
| A. w/o normal consistency | 1.35 | 1.13 | 1.24 |
| B. w/o depth distortion | 0.89 | 0.87 | 0.88 |
| C. w/ expected depth | 0.88 | 1.01 | 0.94 |
| D. w/ SPSR | 1.25 | 0.89 | 1.07 |
| **E. Full Model** | **0.79** | **0.86** | **0.83** |

[cite_start][cite: 329-351]

[cite_start]**Regularization.** Disabling the normal consistency loss leads to noisy surface orientations; conversely, omitting depth distortion regularization results in blurred surface normals[cite: 371, 372]. [cite_start]The complete model, employing both regularizations, successfully captures sharp and flat features[cite: 373].

[cite_start]**Mesh Extraction.** Our full model utilizes TSDF fusion for mesh extraction with median depth[cite: 385]. One alternative option is to use the expected depth instead of the median depth. [cite_start]However, it yields worse reconstructions as it is more sensitive to outliers[cite: 386, 387].

> **Figure 4:** Visual comparisons between our method, 3DGS, and SuGaR. [cite_start]Our method excels at synthesizing geometrically accurate radiance fields and surface reconstruction[cite: 301].
>
> **Figure 5:** Qualitative comparison on the DTU benchmark. [cite_start]Our 2DGS produces detailed and noise-free surfaces[cite: 352].
>
> [cite_start]**Figure 6:** Qualitative studies for the regularization effects.[cite: 370].

---

## 7. Conclusion

[cite_start]We presented 2D Gaussian splatting, a novel approach for geometrically accurate radiance field reconstruction[cite: 393]. [cite_start]We utilized 2D Gaussian primitives for 3D scene representation, facilitating accurate and view consistent geometry modeling and rendering[cite: 394]. [cite_start]We proposed two regularization techniques to further enhance the reconstructed geometry[cite: 395]. [cite_start]Extensive experiments on several challenging datasets verify the effectiveness and efficiency of our method[cite: 396].

[cite_start]**Limitations.** Our method struggles with the accurate reconstruction of semi-transparent surfaces, such as glass[cite: 398, 622]. [cite_start]Moreover, our current densification strategy favors texture-rich over geometry-rich areas, occasionally leading to less accurate representations of fine geometric structures[cite: 398].

---

## References

[cite_start][cite: 404] Aliev et al. 2020. Neural point-based graphics.
[cite_start][cite: 406] Barron et al. 2021. Mip-NeRF.
[cite_start][cite: 408] Barron et al. 2022a. Mip-nerf 360.
[cite_start][cite: 410] Barron et al. 2022b. Mip-NeRF 360 (CVPR).
[cite_start][cite: 412] Barron et al. 2023. Zip-NeRF.
[cite_start][cite: 413] Blinn. 1977. A homogeneous formulation for lines in 3 space.
[cite_start][cite: 414] Botsch et al. 2005. High-quality surface splatting.
[cite_start][cite: 416] Chen et al. 2022. TensoRF.
[cite_start][cite: 418] Chen et al. 2023b. NeuSG.
[cite_start][cite: 419] Chen et al. 2023a. Mobilenerf.
[cite_start][cite: 422] Chen et al. 2023c. NeuRBF.
[cite_start][cite: 425] Fridovich-Keil et al. 2022. Plenoxels.
[cite_start][cite: 428] Fu et al. 2022. Geo-Neus.
[cite_start][cite: 430] Gao et al. 2023. Relightable 3D Gaussian.
[cite_start][cite: 432] Guédon and Lepetit. 2023. SuGaR.
[cite_start][cite: 434] Hedman et al. 2021. Baking neural radiance fields.
[cite_start][cite: 436] Hu et al. 2023. Tri-MipRF.
[cite_start][cite: 438] Insafutdinov and Dosovitskiy. 2018.
[cite_start][cite: 440] Jensen et al. 2014. Large scale multi-view stereopsis evaluation.
[cite_start][cite: 442] Jiang et al. 2023. GaussianShader.
[cite_start][cite: 444] Kazhdan and Hoppe. 2013. Screened poisson surface reconstruction.
[cite_start][cite: 445] Kerbl et al. 2023. 3D Gaussian Splatting.
[cite_start][cite: 447] Keselman and Hebert. 2022. Approximate differentiable rendering.
[cite_start][cite: 448] Keselman and Hebert. 2023. Flexible techniques for differentiable rendering.
[cite_start][cite: 449] Knapitsch et al. 2017. Tanks and Temples.
[cite_start][cite: 451] Kopanas et al. 2021. Point-Based Neural Rendering.
[cite_start][cite: 453] Lassner and Zollhofer. 2021. Pulsar.
[cite_start][cite: 455] Li et al. 2023. Neuralangelo.
[cite_start][cite: 457] Liang et al. 2023. GS-IR.
[cite_start][cite: 459] Liu et al. 2020. Neural Sparse Voxel Fields.
[cite_start][cite: 460] Luiten et al. 2024. Dynamic 3D Gaussians.
[cite_start][cite: 462] Mescheder et al. 2019. Occupancy Networks.
[cite_start][cite: 464] Mildenhall et al. 2020. NeRF (ECCV).
[cite_start][cite: 466] Mildenhall et al. 2021. NeRF (Commun. ACM).
[cite_start][cite: 468] Müller et al. 2022. Instant Neural Graphics Primitives.
[cite_start][cite: 470] Niemeyer et al. 2020. Differentiable Volumetric Rendering.
[cite_start][cite: 472] Oechsle et al. 2021. UNISURF.
[cite_start][cite: 473] Park et al. 2019. DeepSDF.
[cite_start][cite: 476] Pfister et al. 2000. Surfels.
[cite_start][cite: 479] Qian et al. 2023. Gaussian Avatars.
[cite_start][cite: 481] Reiser et al. 2021. KiloNeRF.
[cite_start][cite: 483] Reiser et al. 2023. Merf.
[cite_start][cite: 486] Rückert et al. 2022. Adop.
[cite_start][cite: 488] Schönberger and Frahm. 2016. Structure-from-Motion Revisited.
[cite_start][cite: 489] Schönberger et al. 2016. Pixelwise View Selection.
[cite_start][cite: 491] Schops et al. 2019. Surfelmeshing.
[cite_start][cite: 493] Shi et al. 2023. GIR.
[cite_start][cite: 495] Sigg et al. 2006. GPU-based ray-casting.
[cite_start][cite: 496] Sun et al. 2022a. Direct Voxel Grid Optimization.
[cite_start][cite: 497] Sun et al. 2022b. Improved Direct Voxel Grid Optimization.
[cite_start][cite: 499] Wang et al. 2021. NeuS.
[cite_start][cite: 502] Wang et al. 2023. NeuS2.
[cite_start][cite: 505] Weyrich et al. 2007. A hardware architecture for surface splatting.
[cite_start][cite: 507] Whelan et al. 2016. ElasticFusion.
[cite_start][cite: 509] Wiles et al. 2020. SynSin.
[cite_start][cite: 511] Xie et al. 2023. PhysGaussian.
[cite_start][cite: 513] Yan et al. 2023. Street Gaussians.
[cite_start][cite: 515] Yao et al. 2018. MVSNet.
[cite_start][cite: 517] Yariv et al. 2021. Volume rendering of neural implicit surfaces.
[cite_start][cite: 519] Yariv et al. 2023. BakedSDF.
[cite_start][cite: 521] Yariv et al. 2020. Multiview Neural Surface Reconstruction.
[cite_start][cite: 523] Yifan et al. 2019. Differentiable surface splatting.
[cite_start][cite: 525] Yu et al. 2021. PlenOctrees.
[cite_start][cite: 528] Yu et al. 2022a. SDFStudio.
[cite_start][cite: 530] Yu et al. 2024. Mip-Splatting.
[cite_start][cite: 532] Yu and Gao. 2020. Fast-MVSNet.
[cite_start][cite: 534] Yu et al. 2022b. MonoSDF.
[cite_start][cite: 536] Zhang et al. 2020. NeRF++.
[cite_start][cite: 537] Zhou et al. 2018. Open3D.
[cite_start][cite: 538] Zielonka et al. 2023. Drivable 3D Gaussian Avatars.
[cite_start][cite: 540] Zwicker et al. 2001a. EWA volume splatting.
[cite_start][cite: 542] Zwicker et al. 2001b. Surface splatting.
[cite_start][cite: 544] Zwicker et al. 2004. Perspective accurate splatting.

---

## Appendix

### A. Details of Depth Distortion
[cite_start]To this end, we adopt an $\mathcal{L}_{2}$ loss and transform the intersected depth $z$ to NDC space to down-weight distant Gaussian primitives, $m=NDC(z)$[cite: 547]. The nested algorithm can be implemented in a single forward pass:

[cite_start]$$\mathcal{L} = \sum_{i=0}^{N-1}\sum_{j=0}^{i-1}\omega_{i}\omega_{j}(m_{i}-m_{j})^{2} = \sum_{i=0}^{N-1}\omega_{i}(m_{i}^{2}A_{i-1}+D_{i-1}^{2}-2m_{i}D_{i-1})$$ [cite: 549]

[cite_start]where $A_{i}=\sum_{j=0}^{i}\omega_{j}$, $D_{i}=\Sigma_{j=0}^{i}\omega_{j}m_{j}$ and $D_{i}^{2}=\Sigma_{j=0}^{i}\omega_{j}m_{j}^{2}$[cite: 550].

### B. Depth Calculations
**Mean depth:**
[cite_start]$$z_{mean} = \frac{\sum_{i} \omega_i z_i}{\sum_{i} \omega_i}$$ [cite: 570]

[cite_start]where $\omega_{i}=T_{i}\alpha_{i}\hat{G}_{i}(u(x))$ is the weight contribution of the i-th Gaussian and $T_{i}=\prod_{j=1}^{i-1}(1-\alpha_{j}\hat{G}_{j}(u(x)))$ measures its visibility[cite: 572].

**Median depth:**
[cite_start]$$z_{median}=max\{z_{i}|T_{i}>0.5\}$$ [cite: 574]

### C. Additional Baselines

[cite_start]**Table 6: Additional baselines on DTU dataset** [cite: 557]

| Method | Accuracy | Completion | Average |
| :--- | :--- | :--- | :--- |
| SuGaR | 1.48 | 1.17 | 1.33 |
| SuGaR+TSDF | 2.47 | 1.90 | 2.18 |
| 3DGS+SPSR (center) | 2.05 | 1.25 | 1.65 |
| 3DGS+TSDF (mean) | 1.93 | 1.99 | 1.96 |
| 2DGS + SPSR (center) | 1.25 | 0.89 | 1.07 |
| 2DGS (affine approx) + TSDF | 0.96 | 1.20 | 1.08 |
| 2DGS (our rasterizer) + TSDF (mean) | 0.79 | 0.98 | 0.88 |
| **2DGS (our rasterizer) + TSDF (median)** | **0.78** | **0.83** | **0.80** |

[cite_start][cite: 560]

### D. Additional Results

[cite_start]**Table 7: PSNR scores for Synthetic NeRF dataset** [cite: 599]
| Method | Mic | Chair | Ship | Materials | Lego | Drums | Ficus | Hotdog | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Plenoxels | 33.26 | 33.98 | 29.62 | 29.14 | 34.10 | 25.35 | 31.83 | 36.81 | 31.76 |
| INGP-Base | 36.22 | 35.00 | 31.10 | 29.78 | 36.39 | 26.02 | 33.51 | 37.40 | 33.18 |
| Mip-NeRF | 36.51 | 35.14 | 30.41 | 30.71 | 35.70 | 25.48 | 33.29 | 37.48 | 33.09 |
| 3DGS | 35.36 | 35.83 | 30.80 | 30.00 | 35.78 | 26.15 | 34.87 | 37.72 | 33.32 |
| **Ours** | 35.09 | 35.05 | 30.60 | 29.74 | 35.10 | 26.05 | 35.57 | 37.36 | 33.07 |

[cite_start][cite: 600-602]

[cite_start]**Table 8: PSNR scores for TnT dataset** [cite: 603]
| Method | Barn | Caterpillar | Courthouse | Ignatius | Meetingroom | Truck | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SuGaR | 28.63 | 23.27 | 23.33 | 20.72 | 25.47 | 24.40 | 24.16 |
| 3DGS | 27.99 | 24.82 | 23.33 | 23.95 | 26.89 | 25.01 | 25.33 |
| **Ours** | 28.79 | 24.23 | 23.51 | 23.82 | 26.15 | 26.85 | 25.56 |

[cite_start][cite: 604]

> **Figure 7:** Visualization of a plane tiled by 2D Gaussians. [cite_start]Affine approximation adopted in 3DGS causes perspective distortion and inaccurate depth, violating normal consistency[cite: 565, 566].
>
> **Figure 8:** We visualize the depth maps generated by MipNeRF360, 3DGS, and our method. The depth map of 3DGS exhibits significant noise. [cite_start]In contrast, our approach generates sampled depth points with normals consistent with the rendered normal map[cite: 610, 614, 615].
>
> [cite_start]**Figure 9:** Comparison of surface reconstruction using our 2DGS and 3DGS[cite: 617].
>
> [cite_start]**Figure 10:** Qualitative studies for the Tanks and Temples dataset[cite: 620].
>
> [cite_start]**Figure 11:** Appearance rendering results from reconstructed 2D Gaussian disks, including DTU, TnT, and Mip-NeRF360 datasets[cite: 621].
>
> **Figure 12:** Illustration of limitations: Our 2DGS struggles with the accurate reconstruction of semi-transparent surfaces (A). [cite_start]Moreover, our method tends to create holes in areas with high light intensity (B)[cite: 622, 623].