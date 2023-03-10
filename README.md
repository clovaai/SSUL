# SSUL - Official Pytorch Implementation (NeurIPS 2021)

**SSUL: Semantic Segmentation with Unknown Label for Exemplar-based Class-Incremental Learning** <br />
Sungmin Cha<sup>1,2*</sup>, Beomyoung Kim<sup>3*</sup>, YoungJoon Yoo<sup>2,3</sup>, Taesup Moon<sup>1</sup><br>
<sub>\* Equal contribution</sub>

<sup>1</sup> <sub>Department of Electrical and Computer Engineering, Seoul National University</sub><br />
<sup>2</sup> <sub>NAVER AI Lab</sub><br />
<sup>3</sup> <sub>Face, NAVER Clova</sub><br />

NeurIPS 2021 <br />

[![Paper](https://img.shields.io/badge/arXiv-2106.11562-brightgreen)](https://arxiv.org/abs/2106.11562)
<img src = "https://github.com/clovaai/SSUL/blob/main/figures/SSUL_main.png" width="100%" height="100%">

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ssul-semantic-segmentation-with-unknown-label/overlapped-10-1-on-pascal-voc-2012)](https://paperswithcode.com/sota/overlapped-10-1-on-pascal-voc-2012?p=ssul-semantic-segmentation-with-unknown-label)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ssul-semantic-segmentation-with-unknown-label/overlapped-15-1-on-pascal-voc-2012)](https://paperswithcode.com/sota/overlapped-15-1-on-pascal-voc-2012?p=ssul-semantic-segmentation-with-unknown-label)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ssul-semantic-segmentation-with-unknown-label/overlapped-15-5-on-pascal-voc-2012)](https://paperswithcode.com/sota/overlapped-15-5-on-pascal-voc-2012?p=ssul-semantic-segmentation-with-unknown-label)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ssul-semantic-segmentation-with-unknown-label/overlapped-19-1-on-pascal-voc-2012)](https://paperswithcode.com/sota/overlapped-19-1-on-pascal-voc-2012?p=ssul-semantic-segmentation-with-unknown-label)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ssul-semantic-segmentation-with-unknown-label/disjoint-10-1-on-pascal-voc-2012)](https://paperswithcode.com/sota/disjoint-10-1-on-pascal-voc-2012?p=ssul-semantic-segmentation-with-unknown-label)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ssul-semantic-segmentation-with-unknown-label/disjoint-15-1-on-pascal-voc-2012)](https://paperswithcode.com/sota/disjoint-15-1-on-pascal-voc-2012?p=ssul-semantic-segmentation-with-unknown-label)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ssul-semantic-segmentation-with-unknown-label/disjoint-15-5-on-pascal-voc-2012)](https://paperswithcode.com/sota/disjoint-15-5-on-pascal-voc-2012?p=ssul-semantic-segmentation-with-unknown-label)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ssul-semantic-segmentation-with-unknown-label/overlapped-100-5-on-ade20k)](https://paperswithcode.com/sota/overlapped-100-5-on-ade20k?p=ssul-semantic-segmentation-with-unknown-label)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ssul-semantic-segmentation-with-unknown-label/overlapped-100-50-on-ade20k)](https://paperswithcode.com/sota/overlapped-100-50-on-ade20k?p=ssul-semantic-segmentation-with-unknown-label)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ssul-semantic-segmentation-with-unknown-label/overlapped-50-50-on-ade20k)](https://paperswithcode.com/sota/overlapped-50-50-on-ade20k?p=ssul-semantic-segmentation-with-unknown-label)

# Abtract
This paper introduces a solid state-of-the-art baseline for a class-incremental semantic segmentation (CISS) problem. While the recent CISS algorithms utilize variants of the knowledge distillation (KD) technique to tackle the problem, they failed to fully address the critical challenges in CISS causing the catastrophic forgetting; the semantic drift of the background class and the multi-label prediction issue. To better address these challenges, we propose a new method, dubbed **SSUL-M (Semantic Segmentation with Unknown Label with Memory)**, by carefully combining techniques tailored for semantic segmentation. Specifically, we claim three main contributions. (1) defining unknown classes within the background class to help to learn future classes (help plasticity), (2) freezing backbone network and past classifiers with binary cross-entropy loss and pseudo-labeling to overcome catastrophic forgetting (help stability), and (3) utilizing tiny exemplar memory for the first time in CISS to improve both plasticity and stability. The extensively conducted experiments show the effectiveness of our method, achieving significantly better performance than the recent state-of-the-art baselines on the standard benchmark datasets. Furthermore, we justify our contributions with thorough ablation analyses and discuss different natures of the CISS problem compared to the traditional class-incremental learning targeting classification.

# Experimental Results (mIoU all)

|  Method     | VOC 10-1 (11 tasks) | VOC 15-1 (6 tasks) | VOC 5-3 (6 tasks) | VOC 19-1 (2 tasks) | VOC 15-5 (2 tasks) | VOC 5-1 (16 tasks) | VOC 2-1 (19 tasks) |
| :--------:  | :-----------------: | :--------------:   | :---------------: | :----------------: | :----------------: |:----------------: |:----------------: |
| MiB   |         12.65       |  29.29      |   46.71  |  69.15     |   70.08 | 10.03 | 9.88 |
| PLOP  |         30.45       |  54.64      |   18.68  |  73.54     |   70.09 | 6.46 | 4.47 |
| **SSUL**  |         **59.25**       |  **67.61**      |   **56.89**  |  **75.44**     |   **71.22** | **48.65** | **38.32** |
| **SSUL-M**  |       **64.12**      |  **71.37**      |  **58.37**  |  **76.49**     |  **73.02** | **55.11** | **44.74** |

|  Method     | ADE 100-5 (11 tasks) | ADE 100-10 (6 tasks) | ADE 100-50 (2 tasks) | ADE 50-50 (3 tasks) |
| :--------:  | :-----------------: | :--------------:   | :---------------: | :----------------: |
| MiB   |         25.96       |  29.24      |   32.79  |  29.31     |
| PLOP  |         28.75       |  31.59      |   32.94  |  30.40     |
| **SSUL**  |         **32.48**       |  **33.10**      |   **33.58**  |  **29.56**     |
| **SSUL-M**  |       **34.56**      |  **34.46**      |  **34.37**  |  **29.77**     |

# Getting Started

### Requirements
- torch>=1.7.1
- torchvision>=0.8.2
- numpy
- pillow
- scikit-learn
- tqdm
- matplotlib


### Datasets
```
data_root/
    --- VOC2012/
        --- Annotations/
        --- ImageSet/
        --- JPEGImages/
        --- SegmentationClassAug/
        --- saliency_map/
    --- ADEChallengeData2016
        --- annotations
            --- training
            --- validation
        --- images
            --- training
            --- validation
```

Download [SegmentationClassAug](https://github.com/clovaai/SSUL/releases/download/preparation/SegmentationClassAug.zip) and [saliency_map](https://github.com/clovaai/SSUL/releases/download/preparation/saliency_map.zip)

### Class-Incremental Segmentation Segmentation on VOC 2012

```
DATA_ROOT=your_dataset_root_path
DATASET=voc
TASK=15-1 # [15-1, 10-1, 19-1, 15-5, 5-3, 5-1, 2-1, 2-2]
EPOCH=50
BATCH=32
LOSS=bce_loss
LR=0.01
THRESH=0.7
MEMORY=100 # [0 (for SSUL), 100 (for SSUL-M)]

python main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0,1 --crop_val --lr ${LR} --batch_size ${BATCH} --train_epoch ${EPOCH} --loss_type ${LOSS} --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly --pseudo --pseudo_thresh ${THRESH} --freeze --bn_freeze --unknown --w_transfer --amp --mem_size ${MEMORY}
```

### Class-Incremental Segmentation Segmentation on ADE20K

```
DATA_ROOT=your_dataset_root_path
DATASET=ade
TASK=100-5 # [100-5, 100-10, 100-50, 50-50]
EPOCH=100
BATCH=24
LOSS=bce_loss
LR=0.05
THRESH=0.7
MEMORY=300 # [0 (for SSUL), 300 (for SSUL-M)]

python main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0,1 --crop_val --lr ${LR} --batch_size ${BATCH} --train_epoch ${EPOCH} --loss_type ${LOSS} --dataset ${DATASET} --task ${TASK} --overlap --lr_policy warm_poly --pseudo --pseudo_thresh ${THRESH} --freeze --bn_freeze --unknown --w_transfer --amp --mem_size ${MEMORY}
```

### Qualitative Results

<img src = "https://github.com/clovaai/SSUL/blob/main/figures/Qualitative_VOC.png" width="100%" height="100%">
<img src = "https://github.com/clovaai/SSUL/blob/main/figures/Qualitative_ADE.png" width="100%" height="100%">

# Acknowledgement

Our implementation is based on these repositories: [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch), [Torchvision](https://github.com/pytorch/vision).


# License
```
SSUL
Copyright 2021-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
