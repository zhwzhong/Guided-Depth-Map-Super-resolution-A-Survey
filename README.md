### :book: Guided Depth Map Super-resolution: A Survey
> Accepted by [ACM CSUR](https://dl.acm.org/doi/10.1145/3584860).
#### 
<p align="center">
  <img src="https://github.com/zhwzhong/Guided-Depth-Map-Super-resolution-A-Survey/blob/main/f1.jpg">
</p>

### Citation

If you find this project useful, please cite:

*Zhiwei Zhong, Xianming Liu, Junjun Jiang, Debin Zhao, and Xiangyang Ji. 2023. Guided Depth Map Super-resolution: A Survey. ACM Comput. Surv. Just Accepted (February 2023). https://doi.org/10.1145/3584860*

### Installation

1. Clone repo

   ```bash
   git https://github.com/zhwzhong/Awesome-Guided-Depth-Map-Super-resolution.git
   cd Awesome-Guided-Depth-Map-Super-resolution/code
   ```

2. Install dependent packages

   ```bash
   pip install -r requirements.txt
   ```

### Dataset

1. [NYU 2. ](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)[Sintel](http://sintel.is.tue.mpg.de/) 3.[DIDOE](https://github.com/diode-dataset/diode-devkit?utm_source=catalyzex.com) 4. [SUN RGB](https://rgbd.cs.princeton.edu/) 5. [RGB-D-D](https://dimlrgbd.github.io/) 6. [DIML](https://github.com/lingzhi96/RGB-D-D-Dataset)

### Train

```
 python main.py  --scale=SCALE --dataset_name='NYU' --model_name=MODEL NAME
```

### Test

```
 python main.py  --scale=SCALE --dataset_name='NYU' --model_name=MODEL NAME --test_only
```



## Awesome Guided Depth Map Super-resolution [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

   | Type        |        F        |      P      |       L        |
   | :---------- | :-------------: | :---------: | :------------: |
   | Explanation | Filtering-based | Prior-based | Learning-based |

   


   ### 2023 Venues

| Title                                                        | Venue | Type |                            Code                             |
| :----------------------------------------------------------- | :---: | :--: | :---------------------------------------------------------: |
| Joint depth map super-resolution method via deep hybrid-cross guidance filter |  PR   |  P   |                --                          |
|Recurrent Structure Attention Guidance for Depth Super-Resolution | AAAI | L| -- |
|Structure Flow-Guided Network for Real Depth Super-Resolution | AAAI | L | -- |
| Deep Attentional Guided Filtering | TNNLS | L&F | [Github](https://github.com/zhwzhong/DAGF) |
| Fully Cross-Attention Transformer for Guided Depth Super-Resolution | Sensors | L | -- |
|Spherical Space Feature Decomposition for Guided Depth Map Super-Resolution | arxiv | L&P | -- |

   ### 2022 Venues

| Title                                                        | Venue | Type |                             Code                             |
| :----------------------------------------------------------- | :---: | :--: | :----------------------------------------------------------: |
| PDR-Net: Progressive depth reconstruction network for color guided depth map super-resolution |  NC   |  L   |                              --                              |
| Multi-Modal Convolutional Dictionary Learning                |  TIP  |  L   |                              --                              |
| Toward Unaligned Guided Thermal Super-Resolution             |  TIP  |  L   |       [GitHub](https://github.com/honeygupta/UGSR.git)       |
| Joint image denoising with gradient direction and edge-preserving regularization |  PR   |  P   |                              --                              |
| Learning Graph Regularisation for Guided Super-Resolution    | CVPR  |  P   | [GitHub](https://github.com/prs-eth/graph-super-resolution)  |
| Discrete Cosine Transform Network for Guided Depth Map Super-Resolution | CVPR  |  L   |   [Github](https://github.com/Zhaozixiang1228/GDSR-DCTNet)   |
| Learning Complementary Correlations for Depth Super-Resolution With Incomplete Data in Real World. | TNNLS |  L   |                              --                              |
| Memory-augmented Deep Unfolding Network for Guided Image Super-resolution | IJCV  | L&P  | [Github](https://github.com/manman1995/Awaresome-pansharpening) |
| CODON: On orchestrating cross-domain attentions for depth super-resolution | IJCV  |  L   |         [Github](https://github.com/619862306/CODON)         |
| Local Attention Guided Joint Depth Upsampling                |  VMV  |  L   |                              --                              |
| Depth Map Super-Resolution via Cascaded Transformers Guidance | FRSIP |  L   |                              --                              |

   ### 2021 Venues
| Title                                                        | Venue  | Type |                             Code                             |
| :----------------------------------------------------------- | :----: | :--: | :----------------------------------------------------------: |
| Deformable Kernel Network for Joint Image Filtering          |  IJCV  | L&F  |        [Github](https://github.com/cvlab-yonsei/dkn)         |
| Towards Fast and Accurate Real-World Depth Super-Resolution: Benchmark Dataset and Baseline |  CVPR  |  L   |    [Github](https://github.com/lingzhi96/RGB-D-D-Dataset)    |
| Joint Implicit Image Function for Guided Depth Super-Resolution | ACMMM  |  L   |          [Github](https://github.com/ashawkey/jiif)          |
| BridgeNet: A Joint Learning Network of Depth Map Super-Resolution and Monocular Depth Estimation | ACMMM  |  L   |                              --                              |
| Deformable Enhancement and Adaptive Fusion for Depth Map Super-Resolution |  SPL   |  L   |                              --                              |
| RGB GUIDED DEPTH MAP SUPER-RESOLUTION WITH COUPLED U-NET     |  ICME  |  L   |                              --                              |
| High-resolution Depth Maps Imaging via Attention-based Hierarchical Multi-modal Fusion |  TIP   |  L   |          [Github](https://github.com/zhwzhong/AHMF)          |
| Learning Spatially Variant Linear Representation Models for Joint Filtering | TPAMI  |  L   |         [Github](https://github.com/sunny2109/SVLRM)         |
| Multimodal Deep Unfolding for Guided Image Super-Resolution  |  TIP   |  L   |                              --                              |
| CU-Net+: Deep Fully Interpretable Network for Multi-Modal Image Restoration |  ICIP  |  L   |                              --                              |
| Unsharp Mask Guided Filtering                                |  TIP   |  F   | [Github](https://github.com/shizenglin/Unsharp-Mask-Guided-Filtering) |
| Deep edge map guided depth super resolution                  | SP:IC  |  L   |                              --                              |
| Depth Super-Resolution by Texture-Depth Transformer          |  ICME  |  L   |                              --                              |
| Frequency-Dependent Depth Map Enhancement via Iterative Depth-Guided Affine Transformation and Intensity-Guided Refinement |  TMM   |  L   | [Github](https://github.com/Yifan-Zuo/Frequency-dependent-Depth-Map-Enhancement) |
| Depth Map Super-resolution Based on Dual Normal-depth Regularization and Graph Laplacian Prior | TCSVT  |  P   |                              --                              |
| Dual Regularization Based Depth Map Super-Resolution with Graph Laplacian Prior |  ICME  |  P   |                              --                              |
| MIG-net: Multi-scale Network Alternatively Guided by Intensity and Gradient Features for Depth Map Super-resolution |  TMM   |  L   | [Github](https://github.com/Yifan-Zuo/MIG-net-gradient_guided_depth_enhancement) |
| Depth Map Super-Resolution By Multi-Direction Dictionary And Joint Regularization |  ICME  |  P   |                              --                              |
| Unpaired Depth Super-Resolution in the Wild                  | arXiv  |  L   |                              --                              |
| WAFP-Net: Weighted Attention Fusion based Progressive Residual Learning for Depth Map Super-resolution |  TMM   | L&P  | [Github](https://github.com/PaddlePaddle/PaddleDepth/tree/develop/Depth_super_resolution) |
| Learning Scene Structure Guidance via Cross-Task Knowledge Transfer for Single Depth Super-Resolution |  CVPR  |  L   |    [Github](https://github.com/Sunbaoli/dsr-distillation)    |
| Progressive Multi-scale Reconstruction for Guided Depth Map Super-Resolution via Deep Residual Gate Fusion Network |  CGI   |  L   |                              --                              |
| A Generalized Framework for Edge-preserving and Structure-preserving Image Smoothing | TPAMI  |      | [Github](https://github.com/wliusjtu/Generalized-Smoothing-Framework) |
| Depth Image Super-resolution via Two-Branch Network          | ICCSSP |  L   |                              --                              |
| Depth Map Reconstruction and Enhancement With Local and Patch Manifold Regularized Deep Depth Priors | Access |  P   |                              --                              |
| Single Pair Cross-Modality Super Resolution                  |  CVPR  |  L   |       [Github](https://github.com/camillarhodes/CMSR)        |
| Unpaired Depth Super-Resolution in the Wild                  | arXiv  |  L   |                              --                              |
| Depth map super-resolution based on edge-guided joint trilateral upsampling |  TVC   |  F   |                              --                              |
| Depth Map Super-Resolution Using Guided Deformable Convolution | Access |  L   |                              --                              |
| Fast, High-Quality Hierarchical Depth-Map Super-Resolution   | ACMMM  | L&F  | [Github](https://github.com/YiguoQiao/Fast-High-Quality-HDS) |

   ### 2020 Venues
| Title                                                        | Venue  | Type |                             Code                             |
| :----------------------------------------------------------- | :----: | :--: | :----------------------------------------------------------: |
| Deep Convolutional Neural Network for Multi-Modal Image Restoration and Fusion | TPAMI  | L&P  |   [GitHub](https://github.com/cindydeng1991/TPAMI-CU-Net)    |
| Multimodal Deep Unfolding for Guided Image Super-Resolution  |  TIP   | L&P  |                              --                              |
| Probabilistic Pixel-Adaptive Refinement Networks             |  CVPR  | L&F  |     [Github](https://github.com/visinf/ppac_refinement)      |
| Multi-Direction Dictionary Learning Based Depth Map Super-Resolution With Autoregressive Modeling |  TIP   |  P   |                              --                              |
| Single depth map super-resolution via joint non-local and local modeling |  MMSP  |  P   |                              --                              |
| Channel Attention Based Iterative Residual Learning for Depth Map Super-Resolution |  CVPR  |  L   |    [Github](https://github.com/PaddlePaddle/PaddleDepth)     |
| Guided Deep Decoder: Unsupervised Image Pair Fusion          |  ECCV  |  L   | [Github](https://github.com/tuezato/guided-deep-decoder?utm_source=catalyzex.com) |
| PMBANet: Progressive Multi-Branch Aggregation Network for Scene Depth Super-Resolution |  TIP   |  L   | [Github](https://github.com/Sunbaoli/PMBANet_DSR/tree/master/color-guided) |
| Depth Super-Resolution via Deep Controllable Slicing Network | ACMMM  |  L   |                              --                              |
| Learning Factorized Weight Matrix for Joint Filtering        |  ICML  | L&F  |        [Github](https://github.com/xuxy09/FWM_ICML20)        |
| Deep Convolutional Grid Warping Network for Joint Depth Map Upsampling | Access |  L   |                              --                              |
| Guided Depth Map Super-Resolution Using Recumbent Y Network  | Access |  L   |                              --                              |
| DAEANet: Dual auto-encoder attention network for depth map super-resolution |   NN   |  L   |                              --                              |
| Depth upsampling based on deep edge-aware learning           |   PR   |  L   |          [Github](https://github.com/Sunbaoli/DSR)           |
| Coupled Real-Synthetic Domain Adaptation for Real-World Deep Depth Enhancemen |  TIP   |  L   |                              --                              |
| Depth image super-resolution using correlation-controlled color guidance and multi-scale symmetric network |   PR   |  L   |                              --                              |
| Edge-Guided Depth Image Super-Resolution Based on KSVD       | Access |  P   |                              --                              |
| Depth Map Enhancement by Revisiting Multi-Scale Intensity Guidance Within Coarse-to-Fine Stages | TCSVT  |  L   | [Github](https://github.com/Yifan-Zuo/Revisiting-Multi-scale-Intensity-Guidance-) |
| FMPN: Fusing Multiple Progressive CNNs for Depth Map Super-Resolution | Access |  L   |                              --                              |
| Multi-Scale Frequency Reconstruction for Guided Depth Map Super-Resolution via Deep Residual Network | TCSVT  |  L   | [Github](https://github.com/Yifan-Zuo/MFR_depth_enhancement) |
| Learned Dynamic Guidance for Depth Image Reconstruction      | TPAMI  | L&P  |                              --                              |
| Color-Guided Depth Image Recovery With Adaptive Data Fidelity and Transferred Graph Laplacian Regularization | TCSVT  |  P   |                              --                              |
| Weighted Guided Image Filtering With Steering Kerne          |  TIP   |  F   |          [Github](https://github.com/altlp/SKWGIF)           |
| Weakly Aligned Joint Cross-Modality Super Resolution         |  RSVT  |  L   |                              --                              |
| Depth image super-resolution based on joint sparse coding    |  PRL   |  P   |                              --                              |

   ### 2019 Venues
| Title                                                        | Venue  | Type |                             Code                             |
| :----------------------------------------------------------- | :----: | :--: | :----------------------------------------------------------: |
| Perceptual Deep Depth Super-Resolution                       |  ICCV  |  L   | [GitHub](http://adase.group/3ddl/projects/perceptual-depth-sr/) |
| Spatially Variant Linear Representation Models for Joint Filtering |  ICCV  |  L   |                              --                              |
| Deep Coupled ISTA Network for Multi-Modal Image Super-Resolution |  TIP   |  L   | [Github](https://github.com/cindydeng1991/Deep-Coupled-ISTA-Network) |
| Pixel-Adaptive Convolutional Neural Networks                 |  CVPR  | L&F  |          [Github](https://github.com/NVlabs/pacnet)          |
| Joint Image Filtering with Deep Convolutional Networks       | TPAMI  |  L   | [Github](https://sites.google.com/site/yijunlimaverick/deepjointfilter) |
| Guided Super-Resolution As Pixel-to-Pixel Transformation     |  ICCV  |  L   |      [Github](https://github.com/prs-eth/PixTransform)       |
| Deep Color Guided Coarse-to-Fine Convolutional Network Cascade for Depth Image Super-Resolution |  TIP   | L&F  |                              --                              |
| Pyramid-Structured Depth MAP Super-Resolution Based on Deep Dense-Residual Network |  SPL   |  L   |                              --                              |
| A Novel Segmentation Based Depth Map Up-Sampling             |  TMM   |  P   |                              --                              |
| Simultaneous color-depth super-resolution with conditional generative adversarial networks |   PR   |  L   |          [Github](https://github.com/mdcnn/CDcGAN)           |
| Residual dense network for intensity-guided depth map enhancement |  BMVC  |  L   | [Github](https://github.com/Yifan-Zuo/residual-dense-network-for-intensity-guide-depth-map-enhancement) |
| RADAR: Robust Algorithm for Depth Image Super Resolution Based on FRI Theory and Multimodal Dictionary Learning | TCSVT  |  P   |                              --                              |
| Multiscale Directional Fusion for Depth Map Super Resolution with Denoising | ICASSP |  L   |                              --                              |
| Multi-Direction Dictionary Learning Based Depth Map Super-Resolution with Autoregressive Modeling |  TMM   |  P   |                              --                              |
| Photometric Depth Super-Resolution                           | TPAMI  |  L   | [Github](https://github.com/pengsongyou/SRmeetsPS?utm_source=catalyzex.com) |
| Alternately Guided Depth Super-resolution Using Weighted Least Squares and Zero-order Reverse Filtering | ICASSP |  P   |                              --                              |
| Depth Super-Resolution via Joint Color-Guided Internal and External Regularizations |  TIP   |  P   |                              --                              |
| Multiscale Directional Fusion for Depth Map Super Resolution with Denoising | ICASSP |  P   |                              --                              |

   ### 2018 Venues

| Title                                                        | Venue  | Type |                             Code                             |
| :----------------------------------------------------------- | :----: | :--: | :----------------------------------------------------------: |
| Joint Bilateral Filter                                       |  TIP   |  F   |     [GitHub](https://github.com/facebookresearch/LaMCTS)     |
| Hierarchical Features Driven Residual Learning for Depth Map Super-Resolution |  TIP   |  L   |     [Github](https://li-chongyi.github.io/proj_SR.html)      |
| Depth Super-Resolution From RGB-D Pairs With Transform and Spatial Domain Regularization |  TIP   |  P   |                              --                              |
| Reconstruction-based Pairwise Depth Dataset for Depth Image Enhancement Using CNN |  ECCV  |  L   | [Github](https://github.com/JunhoJeon/depth_refine_reconstruct) |
| Mutually Guided Image Filtering                              | TPAMI  |  F   |     [Github](https://sites.google.com/view/xjguo/mugif)      |
| Depth Restoration From RGB-D Data via Joint Adaptive Regularization and Thresholding on Manifolds |  TIP   |  P   |                              --                              |
| Depth Super-Resolution From RGB-D Pairs With Transform and Spatial Domain Regularization |  TIP   |  P   |                              --                              |
| Fast End-to-End Trainable Guided Filter                      |  CVPR  | L&F  |    [Github](https://github.com/wuhuikai/DeepGuidedFilter)    |
| Color-Guided Depth Map Super-Resolution via Joint Graph Laplacian and Gradient Consistency Regularization |  MMSP  |  P   |                              --                              |
| Single Depth Image Super-Resolution Using Convolutional Neural Networks | ICASSP |  L   |                              --                              |
| Fast Depth Map Super-Resolution Using Deep Neural Network    |  ISMA  |  L   |                              --                              |
| Co-occurrent Structural Edge Detection for Color-Guided Depth Map Super-Resolution |  ICMM  |  P   |                              --                              |
| Depth image super-resolution algorithm based on structural features and non-local means |   OL   |  P   |                              --                              |
| Single-Shot Variational Depth Super-Resolution From Shading  |  CVPR  |  L   | [Github](https://github.com/BjoernHaefner/DepthSRfromShading) |
| Joint-Feature Guided Depth Map Super-Resolution With Face Priors |  TYCB  |  L   |                              --                              |
| Explicit Edge Inconsistency Evaluation Model for Color-guided Depth Map Enhancement | TCSVT  |  P   |                              --                              |
| Minimum spanning forest with embedded edge inconsistency measurement model for guided depth map enhancement |  TIP   |  P   |                              --                              |
| Depth image super-resolution reconstruction based on a modified joint trilateral filter |  ISMA  |  F   |                              --                              |
