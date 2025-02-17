# R<sup>3</sup>BiFormer: An Enhanced Vision Transformer for Remote Sensing Foundation Model

<p align="center">
  | <a href="##introduction">Introduction</a> |
  <a href="##Results">Results</a> |
  <a href="##usage">Usage</a> |
</p >

## Introduction

Remote sensing foundation models have made significant progress in visual tasks and vision transformers (ViTs) have been the primary choice due to their their good scalability and representation ability. However, most of the current studies tend to make innovations in remote sensing pretraining strategies and datasets, but ignore the impact of the model itself. It is suboptimal to apply the models that perform well in natural images directly to the field of remote sensing images because of the difference in feature distribution between them. To deal with complex backgrounds and objects of different sizes in RS images, we propose a vision transformer with bi-level routing attention based on routing region relocation (R$^3$BiFormer), which is an enhanced version of BiFormer in the field of remote sensing. Routing Region Relocation mechanism implements the secondary directional routing of the default region based on the original algorithm, and enables a more flexible allocation of computations with content awareness via a novel dynamic sparse attention.Comprehensive experiments are carried out on semantic segmentation, object detection and change detection, and the results show that R$^3$BiFormer achieves better performance than other advanced foundation models on multiple downstream tasks.

<figure>
<img src=DSQNet_framework.png>
</figure>
<figure>
<div align="center">
<img src=myvisualization.png>
</div>
</figure>

## Results
### Semantic Segmantation
|Dataset|Backbone | Input size | Params (M) | Fscore| 
|-------|-------- | ----------  | ----- | ----- |
| Potsdam |R<sup>3</sup>BiFormer | 512 × 512 | 26M| 92.03 |
| Vaihingen |R<sup>3</sup>BiFormer |  512 × 512 | 26M |97.20 |
