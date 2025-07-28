# SO-Det: A Cross-Layer Weighted Architecture with Channel-Optimized Downsampling and Enhanced Attention Fusion of Small Object Detector

![Framework](images\mainfig1c.jpg) <!-- Add your framework diagram here -->

[![Paper]()]() <!-- Paper link to be updated -->
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


Official implementation of **SO-Det**, a novel architecture for small object detection featuring:
- **Cross-Layer Weighted Architecture (CLWA)**
- **Channel-Optimized Downsampling (CDown)**
- **Enhanced Attention Fusion (EAFusion)**

## Abstract
In object detection tasks, numerous practical applications require the detection of small targets. However, small objects are characterized by low resolution and susceptibility to noise, while conventional object detectors contain redundant designs that struggle to effectively address the specific challenges of small object detection. This study proposes SO-Det, which incorporates Channel-Optimized Downsampling (CDown) and Enhanced Attention Fusion (EAFusion) through a Cross-Layer Weighted Architecture (CLWA). First, CLWA eliminates the traditional 20×20 scale sampling in detectors and employs cross-layer connections at larger scale layers. Second, the cross-layer connections are fused with original feature maps using the EAFusion module. The EAFusion module adopts Enhanced Content-Guided Attention (ECGA), which utilizes combined spatial and channel attention weights to perform pixel attention mechanism rearrangement, forming the final fused features with dynamically weighted fusion feature maps. Finally, conventional strided convolutions in the downsampling component are replaced by CDown, which employs space-to-depth operations to enable stride-1 convolutions. The resulting rich channel information is then weighted by importance relationships through ECA. The experiments show that the proposed method outperforms mainstream general detection models of similar scales on small object datasets. On the VisDrone dataset, the largest version achieves a 5.8% improvement in mAP50 and a 3.6% increase in mAP50-95.

## Installation
```bash
