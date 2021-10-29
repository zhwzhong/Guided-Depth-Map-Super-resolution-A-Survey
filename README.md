### :book: Deep Attentional Guided Image Filtering
> [[Paper](https://)] 
> [Zhiwei Zhong](https://github.com/zhwzhong), [Xianming Liu](http://homepage.hit.edu.cn/xmliu?lang=en), [Junjun Jiang](https://scholar.google.com/citations?user=WNH2_rgAAAAJ&hl=en), [Debin Zhao](https://scholar.google.com/citations?user=QXyj0hkAAAAJ&hl=en) ,[Xiangyang Ji](https://ieeexplore.ieee.org/author/37271425200)<br>Harbin Institute of Technology, Tsinghua University

#### Abstract

Guided filter is a fundamental tool in computer vision and computer graphics which aims to transfer structure information from guidance image to target image. Most existing methods construct filter kernels from the guidance itself without considering the mutual dependency between the guidance and the target. However, since there typically exist significantly different edges in the two images, simply transferring all structural information of the guidance to the target would result in various artifacts. To cope with this problem, we propose an effective framework named deep attentional guided image filtering, the filtering process of which can **fully integrate the complementary information** contained in both images. Specifically, we propose an attentional kernel learning module to generate dual sets of filter kernels from the guidance and the target, respectively, and then adaptively combine them by modeling the pixel-wise dependency between the two images. Meanwhile, we propose a multi-scale guided image filtering module to progressively generate the filtering result with the constructed kernels in a **coarse-to-fine manner**. Correspondingly, a multi-scale fusion strategy is introduced to **reuse the intermediate results** in the coarse-to-fine process. Extensive experiments show that the proposed framework compares favorably with the state-of-the-art methods in a wide range of guided image filtering applications, such as *guided super-resolution, cross-modality restoration, texture removal, and semantic segmentation*.

<p align="center">
  <img src="https://github.com/zhwzhong/DAGF/blob/main/fm1.png">
</p>


---

This repository is an official PyTorch implementation of the paper "**Deep Attentional Guided Filtering**"


:sparkles: ##News
Our method won the Real DSR Challenge in ICMR 2021. 

The detail information can be fond [here](https://icmr21-realdsr-challenge.github.io/#Leaderboard).

## :wrench: Dependencies and Installation

- Python >= 3.5 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.2(https://pytorch.org/
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### Installation

1. Clone repo

    ```bash
    git https://github.com/zhwzhong/DAGF.git
    cd DAGF
    ```

1. Install dependent packages

    ```bash
    pip install -r requirements.txt
    ```

### Dataset

1. [NYU](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
2. [Lu](http://web.cecs.pdx.edu/~fliu/project/depth-enhance/)
3. [Middlebury](http://web.cecs.pdx.edu/~fliu/project/depth-enhance/)
4. [Sintel](http://sintel.is.tue.mpg.de/)
5. [ToFMark](http://sintel.is.tue.mpg.de/)
6. [DUT-OMRON](http://saliencydetection.net/dut-omron/)

### Trained Models

You can directly download the trained model and put it in *checkpoints*:

- DAGF (Nearest):[4](https://drive.google.com/file/d/1lFmYV_c2DDhgk3HHT5jK8JcMn0h4lLYC/view?usp=sharing), [8](https://drive.google.com/file/d/1NHAzCB5tCScC2__8IqnvByr3UUkAxERT/view?usp=sharing), [16](https://drive.google.com/file/d/1pcGtFmFUmMWNkKRdBWoKJFZ9f5vswbTX/view?usp=sharing)
- DAGF (Bicubic): [4](https://drive.google.com/file/d/1q0ASMBCkjgfftS8seOtdD7JQtHbHqL5q/view?usp=sharing), [8](https://drive.google.com/file/d/1bo2fPg-z6XoScuE6IVWQLWIMAnEuVLnC/view?usp=sharing), [16](https://drive.google.com/file/d/1pcGtFmFUmMWNkKRdBWoKJFZ9f5vswbTX/view?usp=sharing)

### Train

You can also train by yourself:

```
 python main.py  --scale=16  --save_real --dataset_name='NYU' --model_name='DAGF'
```

*Pay attention to the settings in the option (e.g. gpu id, model_name).*

### Test
We provide the processed test data in 'test_data' and pre-trained models in 'pre_trained'
With the trained model,  you can test and save depth images.

```
python quick_test.py
```

### Acknowledgments

- Thank for [NYU](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), [Lu](http://web.cecs.pdx.edu/~fliu/project/depth-enhance/), [Middlebury](http://web.cecs.pdx.edu/~fliu/project/depth-enhance/), [Sintel](http://sintel.is.tue.mpg.de/) and [DUT-OMRON](http://saliencydetection.net/dut-omron/) datasets.
% - Thank authors of [GF](https://github.com/wuhuikai/DeepGuidedFilter), [DJFR](https://sites.google.com/site/yijunlimaverick/deepjointfilter), [DKN](https://github.com/cvlab-yonsei/dkn), [PacNet](https://github.com/NVlabs/pacnet), [DSRN](http://sintel.is.tue.mpg.de/), [JBU](http://sintel.is.tue.mpg.de/), [Yang](http://sintel.is.tue.mpg.de/), [DGDIE](http://sintel.is.tue.mpg.de/), [DMSG](http://sintel.is.tue.mpg.de/), [TGV](http://sintel.is.tue.mpg.de/), [SDF](http://sintel.is.tue.mpg.de/)  and [FBS](http://sintel.is.tue.mpg.de/)  for sharing their codes.

### TO DO

1. Release the trained models for compared models:
   - DGF: [4](https:), [8](https:), [16](https:)
   - DJF: [4](https:), [8](https:), [16](https:)
   - DMSG: [4](https:), [8](https:), [16](https:)
   - DJFR: [4](https:), [8](https:), [16](https:)
   - DSRN: [4](https:), [8](https:), [16](https:)
   - PAC: [4](https:), [8](https:), [16](https:)
   - DKN: [4](https:), [8](https:), [16](https:)
2. Release the experimental resutls of the compared models.

:e-mail: Contact

If you have any question, please email `zhwzhong@hit.edu.cn` 
