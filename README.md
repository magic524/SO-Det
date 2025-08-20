# SO-Det: A Cross-Layer Weighted Architecture with Channel-Optimized Downsampling and Enhanced Attention Fusion of Small Object Detector

![Framework](images/mainfig1c.jpg) <!-- Add your framework diagram here -->

[![Paper]()]() <!-- Paper link to be updated -->
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


Official implementation of **SO-Det**, a novel architecture for small object detection featuring:
- **Cross-Layer Weighted Architecture (CLWA)**
- **Channel-Optimized Downsampling (CDown)**
- **Enhanced Attention Fusion (EAFusion)**

## Abstract
Small object detection has broad application prospects in remote sensing and traffic surveillance. The inherent low resolution and noise sensitivity of small objects make sampling more challenging. Existing studies mainly enhance feature fusion architectures or improve feature extraction modules, while redundancy and information loss during sampling remain critical bottlenecks. This work proposes a cross-layer weighted architecture~(CLWA) that removes the P5 scale sampling to suppress noise and incorporates weighted cross-layer connections to preserve original information. A channel-optimized downsampling~(CDown) module is introduced to mitigate information loss caused by conventional strided downsampling. It replaces strided convolution with space-to-depth~(SPD) transformation and applies ECA to weight the enriched channels from SPD. An enhanced attention fusion~(EAFusion) module is designed for the cross-layer connections, where enhanced content-guided attention~(ECGA) integrates spatial and channel attention for pixel-level attention rearrangement. This process produces dynamically weighted fusion feature maps, further enhancing the contribution of original information. Based on these designs, a three-scale small object detector~(SO-Det) is developed. Experimental results on small object datasets demonstrate that SO-Det surpasses multiple baselines and state-of-the-art methods.

## Models

<details open>
<summary>Detection (VisDrone)</summary>

See [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset) for details about this 10-class dataset.

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50 | mAP<sup>val<br>50-95 | Precision | Recall | Params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | ----------------- | -------------------- | --------- | ------ | ------------------ | ----------------- |
| [SO-Det-s](https://github.com/magic524/SO-Det/releases/download/VisDrone200/SO-Dets.pt)    | 640                   | 40.2              | 23.7                 | 49.9      | 38.9   | 1.1                | 16.3              |
| [SO-Det-m](https://github.com/magic524/SO-Det/releases/download/VisDrone200/SO-Detm.pt)    | 640                   | 46.8              | 28.0                 | 53.7      | 46.2   | 3.5                | 50.2              |
| [SO-Det-l](https://github.com/magic524/SO-Det/releases/download/VisDrone200/SO-Detx.pt)    | 640                   | 51.7              | 32.0                 | 59.6      | 50.0   | 13.1               | 182.1             |

- **Metrics** measured on VisDrone val set with input resolution 640x640.  
- Reproduce by `python val.py --data visdrone.yaml --weights so-det-s.pt --img 640`
</details>

<details>
<summary>Detection (TinyPerson)</summary>

See [TinyPerson Dataset](https://github.com/ucas-vg/TinyBenchmark) for details.

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50 | mAP<sup>val<br>50-95 | Precision | Recall | Params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | ----------------- | -------------------- | --------- | ------ | ------------------ | ----------------- |
| [SO-Det-s](https://github.com/magic524/SO-Det/releases/download/TinyPerson200/SO-Dets.pt)    | 640                   | 19.3              | 6.3                  | 32.6      | 26.0   | 1.1                | 16.3              |
| [SO-Det-m](https://github.com/magic524/SO-Det/releases/download/TinyPerson200/SO-Detm.pt)    | 640                   | 24.7              | 7.7                  | 37.4      | 28.8   | 3.5                | 50.2              |
| [SO-Det-l](https://github.com/magic524/SO-Det/releases/download/TinyPerson200/SO-Detl.pt)    | 640                   | 26.1              | 8.3                  | 41.5      | 30.6   | 13.1               | 182.1             |

- **Metrics** measured on TinyPerson val set with input resolution 640x640.  
- Reproduce by `python val.py --data tinyperson.yaml --weights so-det-s.pt --img 640`
</details>


## Installation
```bash
