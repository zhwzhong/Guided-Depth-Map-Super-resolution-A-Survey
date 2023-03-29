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
| Joint depth map super-resolution method via deep hybrid-cross guidance filter [PDF](https://www.sciencedirect.com/science/article/pii/S0031320322007397/pdfft?crasolve=1&r=7af4f5ef6dd41991&ts=1680059855276&rtype=https&vrr=UKN&redir=UKN&redir_fr=UKN&redir_arc=UKN&vhash=UKN&host=d3d3LnNjaWVuY2VkaXJlY3QuY29t&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&re=X2JsYW5rXw%3D%3D&ns_h=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ns_e=X2JsYW5rXw%3D%3D&iv=290cdf75b7c8c8e881d9138a1f9ca3aa&token=63326239316239633530383430646364663366643133346534366533326438646534373432343239663064306434383733653433326561303237396534656263323066373262643664316165383338643030386132356361653862333a333833303364333534353036323065376562306664336362&text=37f925b243cfe7b64c684748ed03feeb99e1bdd926cf80421cfda2728e7c319e9f082e1b14762562f410a01d84d0354e5f1bda70e4562d49ac2379412c1303acc0558e34a0d76fec71b4c43cc0c7ba5405ee4ca8b78138adc06752158266df64cf9edd2057fe1f9cbe66c85903f743d70faff2874949a49945c0e98aa088d0ec1be4696bcd0889145066ce3cd0e4cd7bc1bb6c4213a5d449be35c50b82cc598c5147b48a99f394059e8b02dbe26509cf6a7edd9a710234712e1ca68a95c086e05aaf6d425770f2e62cdcaeaa2c8999c231354fef3a9df4490f1103a8469e11ed828d5b0815e5e0730c21642aff5124ceaf6e069e4bdfd8663ad33aeacbcc3484c895c44e1f6c174bb0c589414bfbbaf99b46d3c5d6e10a83ac701dcdac52d1d553c30c458f1a4ba7529fbbe7b2addca8&original=3f6d64353d3165343161346339666433373662303330633039373431353463383334363639267069643d312d73322e302d53303033313332303332323030373339372d6d61696e2e706466)|  PR   |  P   |                --                          |
|Recurrent Structure Attention Guidance for Depth Super-Resolution [PDF](https://arxiv.org/pdf/2301.13419.pdf) | AAAI | L| -- |
|Structure Flow-Guided Network for Real Depth Super-Resolution [PDF](https://arxiv.org/pdf/2301.13416.pdf) | AAAI | L | -- |
| Deep Attentional Guided Filtering [PDF](https://arxiv.org/pdf/2112.06401.pdf) | TNNLS | L&F | [Github](https://github.com/zhwzhong/DAGF) |
| Fully Cross-Attention Transformer for Guided Depth Super-Resolution [PDF](https://arxiv.org/pdf/2303.09307.pdf)| Sensors | L | -- |
|Spherical Space Feature Decomposition for Guided Depth Map Super-Resolution [PDF](https://arxiv.org/pdf/2303.08942.pdf)| arxiv | L&P | -- |
|Depth Super-Resolution from Explicit and Implicit High-Frequency Features [PDF](https://arxiv.org/pdf/2303.09307.pdf)| arxiv | L | --|
|Self-Supervised Learning for RGB-Guided Depth Enhancement by Exploiting the Dependency Between RGB and Depth [PDF](https://ieeexplore.ieee.org/document/9975253/) |TIP|L|[Github](https://github.com/wjcyt/SRDE)|

   ### 2022 Venues

| Title                                                        | Venue | Type |                             Code                             |
| :----------------------------------------------------------- | :---: | :--: | :----------------------------------------------------------: |
| PDR-Net: Progressive depth reconstruction network for color guided depth map super-resolution [PDF](https://pdf.sciencedirectassets.com/271597/1-s2.0-S0925231222X00052/1-s2.0-S0925231222000686/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEIT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDhnzuw4BKIu%2Bawiy3dMW6aXi%2BWc%2B%2F%2BoTdy9DbGH5GiagIhAL62QlCV3lDi9VvSM%2F8c6SD9aJLVLFfmbaugoEZoaYHeKrMFCF0QBRoMMDU5MDAzNTQ2ODY1IgzVwylJv1LYY9vGHY4qkAWIabPuICj31q5ce%2FMiuF60M2bxGStFkkuq6HJnTcZ6JdagXeinQ9p9vgMFVkiTv%2BiY0HOeynqpHrB2xsw2gG0MKRnPabiGeV%2FIZO4rMSF%2Fb4IT1mbU%2FmfE6YImj8bbD7Jd7x1t2PldeAvAl5j0N30H9OXPyNabHpAoaFB1I9m6gPqbg14kdxpQ9ZakRc44R0Z0FXfVhUSwy375xpvkmazElhwPtLfGEb%2BTgbL91EycU1pxLtWrEehrXHjNaHgYwswu8%2BzrfxFyMVMX%2BZw0jXGGozM9l7VV4cZLinm4reoHNQHIcbLjqLQLarI9Oi2g07mu6OCaf4tjGG1F6NPOUMthFbVm56TB18L03pmlxjMU7snpWW13T0wi7PnBbZqUfxBvo9NsYTfdj9kQDz5DydvKef%2FC%2F5QN8gMKBxYBFKegptPzkKUCJPDQfWvlSUfTosVUUQdqlLDNdY8weiMbuwNODsU7g4nnLzFYDnzcNWOlixSLf34J2rAasDZnDCCrUer0ySZ7X8gVSDbr8dwzaPerL8Or4VYTorf2gxhLoqkFFxwhU2nBdlzfzLbVSMKUTZO4MogO0WHFJVjVqoLhnH1SWDuvY4B4m5iON1Xv%2Fey1i1wfotCjFToIGf4M%2FIBcQv76hIFro9Ieh%2BE6pjvRcaNQILoyHPRZkWMBJG4DjrrX8BNs6CQNSPi77l3KCSlURmaTwwbLhJluBrfj9MNiG7DzDii3AAbS3f2y%2FWNsNtdTyC1O8oDCrXRvSI1EKyeKsO72%2FT%2B8mFs0HYR1qoKhpIdvAeFXMn%2FsmD8rfMXu2ezILC%2BpE7qzt6GMJPVO5FSesun3%2F58YIRlGkRLszIKjcG8SrRE%2BWfdKuuYzussQ8mfNwDDiy5ChBjqwAcAzV%2B7SFtnLwJGZ%2B%2BFW6dQjQ0DHoKPSs80VdHY0uUmDsDcTl2LgAnW%2FZUbHpnnlhPGEJDVDk13IoRy9M2kXB5UtMXeD9VzYyCeOLpTgJVtCxOw5fkd3jCThiqPbG9HC%2F3assVhGqGspc6TwQC6osn14VMh0bOhUGokeD0uJn6v1FlrVi1F%2FSBDu%2BefBJf%2Fppk7E6RLMOgGgbizZl2yHCxHfQXDpbupAvW9XPHvNcjN2&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230329T124916Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYYXMZWJOK%2F20230329%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=2f85dc357f2204b8ff4951efb92178dfdfb927efe64b18f38fd179596bad6517&hash=ab4e5b7f230e4f3448ceb6935e2e43e1a6ba3f559073dc238ce370163e803bee&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0925231222000686&tid=spdf-3324a176-b731-431b-92d0-8dfc0b40857a&sid=4740a84052150640a50ae0205997118b06e8gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0d0c570e005b560457&rr=7af83b5d5ce2096f&cc=cn) |  NC   |  L   |                              --                              |
| Multi-Modal Convolutional Dictionary Learning [PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9681224&tag=1) |  TIP  |  L   |                              --                              |
| Toward Unaligned Guided Thermal Super-Resolution  [PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9633258) |  TIP  |  L   |       [GitHub](https://github.com/honeygupta/UGSR.git)       |
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
