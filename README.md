# R<sup>3</sup>BiFormer: An Enhanced Vision Transformer for Remote Sensing Foundation Model

<p align="center">
  | <a href="## introduction">Introduction</a> |
  <a href="## Results">Results</a> |
  <a href="## usage">Usage</a> |
</p >

## Introduction

Remote sensing foundation models have made significant progress in visual tasks and vision transformers (ViTs) have been the primary choice due to their their good scalability and representation ability. However, most of the current studies tend to make innovations in remote sensing pretraining strategies and datasets, but ignore the impact of the model itself. It is suboptimal to apply the models that perform well in natural images directly to the field of remote sensing images because of the difference in feature distribution between them. To deal with complex backgrounds and objects of different sizes in RS images, we propose a vision transformer with bi-level routing attention based on routing region relocation (R$^3$BiFormer), which is an enhanced version of BiFormer in the field of remote sensing. Routing Region Relocation mechanism implements the secondary directional routing of the default region based on the original algorithm, and enables a more flexible allocation of computations with content awareness via a novel dynamic sparse attention.Comprehensive experiments are carried out on semantic segmentation, object detection and change detection, and the results show that R$^3$BiFormer achieves better performance than other advanced foundation models on multiple downstream tasks.

<figure>
<img src=RBiFormer.png>
<figcaption align = "center"><b>Fig.  The overall framework of R<sup>3</sup>BiFormer model</b></figcaption>
</figure>
<figure>
<div align="center">
<img src=RBRA.png>
</div>
<figcaption align = "center"><b>Fig. The difference of R<sup>3</sup>BRA mechanism and BRA mechanism</b></figcaption>
</figure>

## Results
### Semantic Segmantation
|Dataset|Backbone | Input size | Params (M) | Fscore| 
|-------|-------- | ----------  | ----- | ----- |
| Potsdam |R<sup>3</sup>BiFormer | 512 × 512 | 26M| 92.03 |
| Vaihingen |R<sup>3</sup>BiFormer |  512 × 512 | 26M |97.20 |

### Object Detection
|Dataset|Backbone | Input size | Params (M) | mAP| 
|-------|-------- | ----------  | ----- | ----- |
| HRSC2016 |R<sup>3</sup>BiFormer | 800 × 800 | 26M| 90.70 |
| DOTA |R<sup>3</sup>BiFormer |  1024 × 1024 | 26M |77.50 |

### Change Detection
|Dataset|Backbone | Input size | Params (M) | mAP| 
|-------|-------- | ----------  | ----- | ----- |
| LEVIR-CD |R<sup>3</sup>BiFormer | 512 × 512 | 26M| 92.05 |
| S2Looking |R<sup>3</sup>BiFormer |  512 × 512 | 26M |67.02 |

## Usage
### Installation
The code framework is mainly borrowed from RSP. Thus,
please refer to [RSP-README.md](https://github.com/ViTAE-Transformer/RSP/blob/main/README.md) for installing main packeges such as python, pytorch, etc.

If there is some problem running the R<sup>3</sup>BiFormer, please try the following environment:
- Python 3.8.5
- Pytorch 1.9.0+cu111
- torchvision 0.10.0+cu111
- timm 0.4.12
- mmcv-full 1.4.1
  
### Training
#### Semantic Segmentation
Training the R<sup>3</sup>BiFormer on LEVIR-CD dataset: 

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=40001 tools/train.py \
configs/upernet/BiFormer/upernet_R3BiFormer_512x512_160k_potsdam_rgb_dpr10_lr6e5_lrd90_ps16_class5_ignore5_optimizer_model.py \
--launcher 'pytorch'
```
#### Object Detection
Training the R<sup>3</sup>BiFormer on DOTA dataset: 

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=50002 tools/train.py \
configs/obb/oriented_rcnn/faster_rcnn_orpn_our_imp_biformer_fpn_1x_dota20.py \
--launcher 'pytorch' --options 'find_unused_parameters'=True
```

#### ChangeDeteciton
Training the R<sup>3</sup>BiFormer on S2Looking dataset: 

```
python tools/train.py configs/DSQNet/DSQNet_vitae_imp_512x512_80k_s2looking_lr1e-4_bs8_wd0.01.py --work-dir ./DSQNet_vitae_imp_512x512_80k_s2looking_lr1e-4_bs8_wd0.01.py
```
### Inference
#### Sematic Segmentation
Evaluation using R<sup>3</sup>BiFormer backbone on Potsdam dataset
```
python tools/test.py configs/upernet/BiFormer/upernet_R3BiFormer_512x512_160k_potsdam_rgb_dpr10_lr6e5_lrd90_ps16_class5_ignore5_optimizer_model.py \ \
[model path] --show-dir [img save path] \
--eval mIoU 'mFscore'
```

### Object Detection
Evaluation using R<sup>3</sup>BiFormer backbone on DOTA dataset
```
python tools/test.py configs/obb/oriented_rcnn/faster_rcnn_orpn_our_imp_biformer_fpn_1x_dota20.py \
--out [result file] --eval 'mAP' \
--show-dir [saved map path]
```
### Change Detection

Evaluation using R<sup>3</sup>BiFormer backbone on S2Looking dataset

```
python tools/test.py configs/DSQNet/DSQNet_vitae_imp_512x512_80k_s2looking_lr1e-4_bs8_wd0.01.py [model pth] --show-dir visualization/S2Looking
```
