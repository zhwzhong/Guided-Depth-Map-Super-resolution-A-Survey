### :book: Guided Depth Map Super-resolution: A Survey
> Accepted by ACM Computing Surveys.
#### 
<p align="center">
  <img src="https://github.com/zhwzhong/Guided-Depth-Map-Super-resolution-A-Survey/blob/main/f1.jpg">
</p>

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

1. [NYU](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
4. [Sintel](http://sintel.is.tue.mpg.de/)
5. [DIDOE](https://github.com/diode-dataset/diode-devkit?utm_source=catalyzex.com)
6. [SUN RGB](https://rgbd.cs.princeton.edu/)
6. [RGB-D-D](https://dimlrgbd.github.io/)
6. [DIML](https://github.com/lingzhi96/RGB-D-D-Dataset)

### Train

```
 python main.py  --scale=SCALE --dataset_name='NYU' --model_name=MODEL NAME
```

## Awesome Guided Depth Map Super-resolution [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

   | Type        |        F        |      P      |       L        |
   | :---------- | :-------------: | :---------: | :------------: |
   | Explanation | Filtering-based | Prior-based | Learning-based |

   

   
   ### 2023 Venues
###2022 Venues

| Title                                                        | Venue | Type |                            Code                             |
| :----------------------------------------------------------- | :---: | :--: | :---------------------------------------------------------: |
| PDR-Net: Progressive depth reconstruction network for color guided depth map super-resolution |  NC   |  L   |                             --                              |
| Multi-Modal Convolutional Dictionary Learning                |  TIP  |  L   |                                                             |
| *Toward Unaligned Guided Thermal Super-Resolution*           |  TIP  |  L   |      [GitHub](https://github.com/honeygupta/UGSR.git)       |
| *Joint image denoising with gradient direction and edge-preserving regularization* |  PR   |  P   |                                                             |
| Learning Graph Regularisation for Guided Super-Resolution    | CVPR  |  P   | [GitHub](https://github.com/prs-eth/graph-super-resolution) |
| Discrete Cosine Transform Network for Guided Depth Map Super-Resolution | CVPR  |  L   |  [Github](https://github.com/Zhaozixiang1228/GDSR-DCTNet)   |
|                                                              |       |      |                                                             |
   
   ### 2022 Venues

| Title                                                        | Venue | Type |                            Code                             |
| :----------------------------------------------------------- | :---: | :--: | :---------------------------------------------------------: |
| PDR-Net: Progressive depth reconstruction network for color guided depth map super-resolution |  NC   |  L   |                             --                              |
| Multi-Modal Convolutional Dictionary Learning                |  TIP  |  L   |                                                             |
| *Toward Unaligned Guided Thermal Super-Resolution*           |  TIP  |  L   |      [GitHub](https://github.com/honeygupta/UGSR.git)       |
| *Joint image denoising with gradient direction and edge-preserving regularization* |  PR   |  P   |                                                             |
| Learning Graph Regularisation for Guided Super-Resolution    | CVPR  |  P   | [GitHub](https://github.com/prs-eth/graph-super-resolution) |
| Discrete Cosine Transform Network for Guided Depth Map Super-Resolution | CVPR  |  L   |  [Github](https://github.com/Zhaozixiang1228/GDSR-DCTNet)   |
|                                                              |       |      |                                                             |
   
   ### 2021 Venues
   | Title                                                        | Venue  | Type |                             Code                             |
| :----------------------------------------------------------- | :----: | :--: | :----------------------------------------------------------: |
| *Deep Denoising of Flash and No-Flash Pairs for Photography in Low-Light Environments* |  CVPR  |  L   |         [GitHub](https://github.com/likesum/deepFnF)         |
| Deformable Kernel Network for Joint Image Filtering          |  IJCV  | L&F  |        [Github](https://github.com/cvlab-yonsei/dkn)         |
| Towards Fast and Accurate Real-World Depth Super-Resolution: Benchmark Dataset and Baseline |  CVPR  |  L   |    [Github](https://github.com/lingzhi96/RGB-D-D-Dataset)    |
| Joint Implicit Image Function for Guided Depth Super-Resolution | ACMMM  |  L   |          [Github](https://github.com/ashawkey/jiif)          |
| BridgeNet: A Joint Learning Network of Depth Map Super-Resolution and Monocular Depth Estimation | ACMMM  |  L   |                              --                              |
| *Deformable Enhancement and Adaptive Fusion for Depth Map Super-Resolution* |  SPL   |  L   |                              --                              |
| *Static/dynamic filter with nonlocal regularizer*            |  JEI   |  F   |      [Github](https://github.com/seasonle/NonlocalSD-)       |
| RGB GUIDED DEPTH MAP SUPER-RESOLUTION WITH COUPLED U-NET     |  ICME  |  L   |                              --                              |
| High-resolution Depth Maps Imaging via Attention-based Hierarchical Multi-modal Fusion |  TIP   |  L   |          [Github](https://github.com/zhwzhong/DAGF)          |
| Learning Spatially Variant Linear Representation Models for Joint Filtering | TPAMI  |  L   |         [Github](https://github.com/sunny2109/SVLRM)         |
| Multimodal Deep Unfolding for Guided Image Super-Resolution  |  TIP   |  L   |                              --                              |
| CU-Net+: Deep Fully Interpretable Network for Multi-Modal Image Restoration |  ICIP  |  L   |                              --                              |
| Unsharp Mask Guided Filtering                                |  TIP   |  F   | [Github](https://github.com/shizenglin/Unsharp-Mask-Guided-Filtering) |
|                                                              |        |      |                                                              |
| Attention-Based Multistage Fusion Network for Remote Sensing Image Pansharpening |  TGRS  |  L   |                              --                              |
| *Deep Convolutional Sparse Coding Network For Pansharpening With Guidance Of Side Information* |  ICME  |  L   |                              --                              |
| Deep edge map guided depth super resolution                  | SP:IC  |  L   |                                                              |
| Depth Super-Resolution by Texture-Depth Transformer          |  ICME  |      |                                                              |
| Frequency-Dependent Depth Map Enhancement via Iterative Depth-Guided Affine Transformation and Intensity-Guided Refinement |  TMM   |      |                                                              |
| Depth Map Super-resolution Based on Dual Normal-depth Regularization and Graph Laplacian Prior | TCSVT  |      |                                                              |
| *Dual Regularization Based Depth Map Super-Resolution with Graph Laplacian Prior* |  ICME  |      |                                                              |
| MIG-net: Multi-scale Network Alternatively Guided by Intensity and Gradient Features for Depth Map Super-resolution |  TMM   |      |                                                              |
| Depth Map Super-Resolution By Multi-Direction Dictionary And Joint Regularization |  ICME  |      |                                                              |
| *Unpaired Depth Super-Resolution in the Wild*                | arXiv  |      |                                                              |
| WAFP-Net: Weighted Attention Fusion based Progressive Residual Learning for Depth Map Super-resolution |  TMM   |      |                                                              |
| Learning Scene Structure Guidance via Cross-Task Knowledge Transfer for Single Depth Super-Resolution |  CVPR  |      |    [Github](https://github.com/Sunbaoli/dsr-distillation)    |
| *Progressive Multi-scale Reconstruction for Guided Depth Map Super-Resolution via Deep Residual Gate Fusion Network* |  CGI   |      |                                                              |
| A Generalized Framework for Edge-preserving and Structure-preserving Image Smoothing | TPAMI  |      | [Github](https://github.com/wliusjtu/Generalized-Smoothing-Framework) |
| Depth Image Super-resolution via Two-Branch Network          | ICCSSP |      |                                                              |
| Depth Map Reconstruction and Enhancement With Local and Patch Manifold Regularized Deep Depth Priors | Access |      |                                                              |
| Seeing in Extra Darkness Using a Deep-Red Flash              |  CVPR  |      |                                                              |
| Single Pair Cross-Modality Super Resolution                  |  CVPR  |      |                                                              |
| Unpaired Depth Super-Resolution in the Wild                  | arXiv  |      |                                                              |
| Depth map super-resolution based on edge-guided joint trilateral upsampling |  TVC   |      |                                                              |
| Depth Map Super-Resolution Using Guided Deformable Convolution | Access |      |                                                              |
| Fast, High-Quality Hierarchical Depth-Map Super-Resolution   | ACMMM  |      |                                                              |
| Decoupled Dynamic Filter Networks                            |  CVPR  |      |                                                              |
| Beyond Image to Depth: Improving Depth Prediction using Echoes |  CVPR  |      |                                                              |
| Image Enhancement Based on the Fusion of Visible and Near-Infrared Images |  IASC  |      |                                                              |
| NeuralFusion: Online Depth Fusion in Latent Space            |  CVPR  |      |                                                              |
   
   ### 2020 Venues
   | Title                                                        | Venue  | Type |                          Code                           |
| :----------------------------------------------------------- | :----: | :--: | :-----------------------------------------------------: |
| Deep Convolutional Neural Network for Multi-Modal Image Restoration and Fusion | TPAMI  | L&P  | [GitHub](https://github.com/cindydeng1991/TPAMI-CU-Net) |
| Multimodal Deep Unfolding for Guided Image Super-Resolution  |  TIP   |  L   |                                                         |
| Interpretable Deep Learning for Multimodal Super-Resolution of Medical Images | MICCAI |      |                                                         |
| Probabilistic Pixel-Adaptive Refinement Networks             |  CVPR  |      |                                                         |
| Multi-Direction Dictionary Learning Based Depth Map Super-Resolution With Autoregressive Modeling |  TIP   |  L   |                                                         |
| *Single depth map super-resolution via joint non-local and local modeling* |  MMSP  |  P   |                                                         |
| Self-supervised Depth Denoising Using Lower- and Higher-quality RGB-D sensors |  3DV   |  L   |                                                         |
| Channel Attention Based Iterative Residual Learning for Depth Map Super-Resolution |  CVPR  |  L   |                                                         |
| Guided Deep Decoder: Unsupervised Image Pair Fusion          |  ECCV  |  L   |                                                         |
| PMBANet: Progressive Multi-Branch Aggregation Network for Scene Depth Super-Resolution |  TIP   |  L   |                                                         |
| Depth Super-Resolution via Deep Controllable Slicing Network | ACMMM  |  L   |                                                         |
| Learning Factorized Weight Matrix for Joint Filtering        |  ICML  |      |                                                         |
| Deep Convolutional Grid Warping Network for Joint Depth Map Upsampling | Access |      |                                                         |
| Guided Depth Map Super-Resolution Using Recumbent Y Network  | Access |      |                                                         |
| DAEANet: Dual auto-encoder attention network for depth map super-resolution |   NN   |      |                                                         |
| Depth upsampling based on deep edge-aware learning           |   PR   |      |                                                         |
| Coupled Real-Synthetic Domain Adaptation for Real-World Deep Depth Enhancemen |  TIP   |      |                                                         |
| Depth image super-resolution using correlation-controlled color guidance and multi-scale symmetric network |   PR   |      |                                                         |
| Edge-Guided Depth Image Super-Resolution Based on KSVD       | Access |      |                                                         |
| Depth Map Enhancement by Revisiting Multi-Scale Intensity Guidance Within Coarse-to-Fine Stages | TCSVT  |      |                                                         |
| FMPN: Fusing Multiple Progressive CNNs for Depth Map Super-Resolution | Access |      |                                                         |
| Multi-Scale Frequency Reconstruction for Guided Depth Map Super-Resolution via Deep Residual Network | TCSVT  |      |                                                         |
| Learned Dynamic Guidance for Depth Image Reconstruction      | TPAMI  |      |                                                         |
| Color-Guided Depth Image Recovery With Adaptive Data Fidelity and Transferred Graph Laplacian Regularization | TCSVT  |      |                                                         |
| Joint Filtering of Intensity Images and Neuromorphic Events for High-Resolution Noise-Robust Imaging |  CVPR  |      |                                                         |
| Depth upsampling based on deep edge-aware learning           |   PR   |      |                                                         |
| Weighted Guided Image Filtering With Steering Kerne          |  TIP   |      |                                                         |
| Deep Atrous Guided Filter for Image Restoration in Under Display Cameras | ECCVW  |  F   |                                                         |
| NEAR-INFRARED IMAGE GUIDED REFLECTION REMOVAL                |  ICME  |      |                                                         |
| Weakly Aligned Joint Cross-Modality Super Resolution         |  RSVT  |      |                                                         |
| Depth image super-resolution based on joint sparse coding    |  PRL   |      |                                                         |
| Adaptive Multi-Modality Residual Network for Compression Distorted Multi-View Depth Video Enhancement | Access |      |                                                         |
| RoutedFusion: Learning Real-time Depth Map Fusion            |  CVPR  |      |                                                         |
| Depth Image Denoising Using Nuclear Norm and Learning Graph Model | ACMMM  |      |                                                         |
| Coupled Real-Synthetic Domain Adaptation for Real-World Deep Depth Enhancement.pdf |  TIP   |      |                                                         |
   
   ### 2019 Venues
  | Title                                                        | Venue  | Type |                             Code                             |
| :----------------------------------------------------------- | :----: | :--: | :----------------------------------------------------------: |
| Perceptual Deep Depth Super-Resolution                       |  ICCV  |  L   | [GitHub](http://adase.group/3ddl/projects/perceptual-depth-sr/) |
| Spatially Variant Linear Representation Models for Joint Filtering |  ICCV  |  L   |                              --                              |
| Deep Coupled ISTA Network for Multi-Modal Image Super-Resolution |  TIP   |  L   |                                                              |
| Pixel-Adaptive Convolutional Neural Networks                 |  CVPR  | L&F  |                                                              |
| Joint Image Filtering with Deep Convolutional Networks       | TPAMI  |  L   |                                                              |
| Guided Super-Resolution As Pixel-to-Pixel Transformation     |  ICCV  |  L   |                                                              |
| Self-Supervised Deep Depth Denoising                         |  ICCV  |  L   |                                                              |
| Deep Color Guided Coarse-to-Fine Convolutional Network Cascade for Depth Image Super-Resolution |  TIP   | L&F  |                                                              |
| *Pyramid-Structured Depth MAP Super-Resolution Based on Deep Dense-Residual Network* |  SPL   |      |                                                              |
| Multi-Source Deep Residual Fusion Network for Depth Image Super-resolution |  RSVT  |      |                                                              |
| Dense Deep Joint Image Filter for Upsampling and Denoising   | ITAIC  |      |                                                              |
| A Novel Segmentation Based Depth Map Up-Sampling             |  TMM   |      |                                                              |
| Multi-Modal Deep Guided Filtering for Comprehensible Medical Image Processing |  TMI   |      |                                                              |
| Simultaneous color-depth super-resolution with conditional generative adversarial networks |   PR   |      |                                                              |
| Local activity-driven structural-preserving filtering f or noise removal and image smoothing |   SP   |      |                                                              |
| Iterative range-domain weighted filter for structural preserving image smoothing and de-noising |  MTA   |      |                                                              |
| Multi-Source Deep Residual Fusion Network for Depth Image Super-resolution |  RSVT  |      |                                                              |
| Residual dense network for intensity-guided depth map enhancement |   IF   |      |                                                              |
| RADAR: Robust Algorithm for Depth Image Super Resolution Based on FRI Theory and Multimodal Dictionary Learning | TCSVT  |      |                                                              |
| Multiscale Directional Fusion for Depth Map Super Resolution with Denoising | ICASSP |      |                                                              |
| Multi-Direction Dictionary Learning Based Depth Map Super-Resolution with Autoregressive Modeling |  TMM   |      |                                                              |
| Photometric Depth Super-Resolution                           | TPAMI  |      |                                                              |
| Alternately Guided Depth Super-resolution Using Weighted Least Squares and Zero-order Reverse Filtering | ICASSP |      |                                                              |
| Depth Super-Resolution via Joint Color-Guided Internal and External Regularizations |  TIP   |      |                                                              |
| Color-Guided Restoration and Local Adjustment of Multi-resolution Depth Map | SICCS  |      |                                                              |
| Depth Super-Resolution on RGB-D Video Sequences With Large Displacement 3D Motion |  TIP   |      |                                                              |
| Depth Maps Restoration for Human Using RealSense             | Access |      |                                                              |
| A Lightweight Neural Network Based Human Depth Recovery Method |  ICME  |      |                                                              |
| Multiscale Directional Fusion for Depth Map Super Resolution with Denoising | ICASSP |      |                                                              |
| Simultaneous color-depth super-resolution with conditional generative adversarial networks |   PR   |      |                                                              |
| Multi-frame Super-resolution for Time-of-flight Imaging      |   PR   |      |                                                              |
| Cross-View Multi-Lateral Filter for Compressed Multi-View Depth Video |  TIP   |      |                                                              |
| Local Activity-Driven Structural-Preserving Filtering for Noise Removal and Image Smoothing |   SP   |      |                                                              |
| Self-Supervised Deep Depth Denoising                         |  ICCV  |      |                                                              |
| Learned Dynamic Guidance for Depth Image Reconstruction      | TPAMI  |      |                                                              |
   
   ### 2018 Venues

| Title                                                        | Venue  | Type |                         Code                         |
| :----------------------------------------------------------- | :----: | :--: | :--------------------------------------------------: |
| Joint Bilateral Filter                                       |  TIP   |  F   | [GitHub](https://github.com/facebookresearch/LaMCTS) |
| Hierarchical Features Driven Residual Learning for Depth Map Super-Resolution |  TIP   |  L   |                                                      |
| Deeply Supervised Depth Map Super-Resolution as Novel View Synthesis | TCSVT  |  L   |                                                      |
| Depth Super-Resolution From RGB-D Pairs With Transform and Spatial Domain Regularization |  TIP   |  P   |                                                      |
| Reconstruction-based Pairwise Depth Dataset for Depth Image Enhancement Using CNN |  ECCV  |      |                                                      |
| Mutually Guided Image Filtering                              | TPAMI  |      |                                                      |
| Depth Restoration From RGB-D Data via Joint Adaptive Regularization and Thresholding on Manifolds |  TIP   |      |                                                      |
| Depth Super-Resolution From RGB-D Pairs With Transform and Spatial Domain Regularization |  TIP   |      |                                                      |
| Global Auto-Regressive Depth Recovery via Iterative Non-Local Filtering |   TB   |      |                                                      |
| Fast End-to-End Trainable Guided Filter                      |  CVPR  |      |                                                      |
| Depth Super-Resolution with Deep Edge-Inference Network and Edge-Guided Depth Filling | ICASSP |      |                                                      |
| Color-Guided Depth Map Super-Resolution via Joint Graph Laplacian and Gradient Consistency Regularization |  MMSP  |      |                                                      |
| Single Depth Image Super-Resolution Using Convolutional Neural Networks | ICASSP |      |                                                      |
| Fast Depth Map Super-Resolution Using Deep Neural Network    |  ISMA  |      |                                                      |
| Co-occurrent Structural Edge Detection for Color-Guided Depth Map Super-Resolution |  ICMM  |      |                                                      |
| Depth image super-resolution algorithm based on structural features and non-local means |   OL   |      |                                                      |
| Single-Shot Variational Depth Super-Resolution From Shading  |  CVPR  |      |                                                      |
| Joint-Feature Guided Depth Map Super-Resolution With Face Priors |  TYCB  |      |                                                      |
| Explicit Edge Inconsistency Evaluation Model for Color-guided Depth Map Enhancement | TCSVT  |      |                                                      |
| Minimum spanning forest with embedded edge inconsistency measurement model for guided depth map enhancement |  TIP   |      |                                                      |
| Depth image super-resolution reconstruction based on a modified joint trilateral filter |  ISMA  |      |                                                      |
| Precise depth map upsampling and enhancement based on edge-preserving fusion filters | IETCV  |      |                                                      |
| Robust depth enhancement based on texture and depth consistency | IETCV  |      |                                                      |
|                                                              |        |      |                                                      |
   
