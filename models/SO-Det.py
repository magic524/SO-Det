import torch
import torch.nn as nn
from torch.nn import Upsample
from models.common import (Conv, C3, SPPF, Concat, 
                          Detect, C3k2, CDown, 
                          C2PSA, WeightedConcat, EAFusion)

class SODet(nn.Module):
    def __init__(self, nc=80, scale='n'):
        super().__init__()
        
        # Scale parameters
        scales = {
            'n': [0.50, 0.25, 1024],
            's': [0.50, 0.50, 1024],
            'm': [0.50, 1.00, 512],
            'l': [1.00, 1.00, 512],
            'x': [1.00, 1.50, 512]
        }
        depth, width, max_channels = scales[scale]
        
        # Backbone
        self.backbone = nn.Sequential(
            Conv(3, int(64 * width), 3, 2),  # 0-P1/1 (320x320)
            CDown(int(64 * width), int(128 * width), 3, 1, 1),  # 1-P2/2 (160x160)
            *[C3k2(int(128 * width), int(256 * width), n=2, shortcut=False, g=0.25)],  # 2
            CDown(int(256 * width), int(256 * width), 3, 1, 1),  # 3-P3/4 (80x80)
            *[C3k2(int(256 * width), int(512 * width), n=2, shortcut=False, g=0.25)],  # 4
            CDown(int(512 * width), int(512 * width), 3, 1, 1),  # 5-P4/4 (40x40)
            *[C3k2(int(512 * width), int(512 * width), n=2, shortcut=True)],  # 6
            SPPF(int(512 * width), int(512 * width), 5),  # 7 (40x40)
            *[C2PSA(int(512 * width), int(512 * width), n=2)]  # 8 (40x40)
        )
        
        # Head
        self.upsample1 = Upsample(scale_factor=2, mode='nearest')  # 9 (40x40->80x80)
        self.concat1 = Concat(dimension=1)  # 10
        self.c3k2_1 = C3k2(int(768 * width), int(256 * width), n=2, shortcut=False)  # 11
        
        self.upsample2 = Upsample(scale_factor=2, mode='nearest')  # 12 (80x80->160x160)
        self.concat2 = Concat(dimension=1)  # 13
        self.c3k2_2 = C3k2(int(384 * width), int(256 * width), n=2, shortcut=False)  # 14
        
        self.conv1 = Conv(int(256 * width), int(256 * width), 3, 2)  # 15 (160x160->80x80)
        self.wconcat = WeightedConcat(dimension=1, weights=[1, 3])  # 16
        self.c3k2_3 = C3k2(int(768 * width), int(512 * width), n=2, shortcut=False)  # 17
        
        self.conv2 = Conv(int(512 * width), int(512 * width), 3, 2)  # 18 (80x80->40x40)
        self.concat3 = Concat(dimension=1)  # 19
        self.c3k2_4 = C3k2(int(1024 * width), int(512 * width), n=2, shortcut=True)  # 20
        
        self.EAFusion = EAFusion()  # 21
        self.detect = Detect(nc)  # 22 (Detect layer)
        
        # Save layer indices for routing
        self.save = [2, 4, 8]  # Layers to save features from
        
    def forward(self, x):
        # Backbone
        x1 = self.backbone[:3](x)  # Through layer 2
        x2 = self.backbone[3:5](x1)  # Through layer 4
        x3 = self.backbone[5:](x2)  # Through layer 8
        
        # Head - Upsample path
        y1 = self.upsample1(x3)  # 9
        y1 = self.concat1([y1, x2])  # 10
        y1 = self.c3k2_1(y1)  # 11
        
        y2 = self.upsample2(y1)  # 12
        y2 = self.concat2([y2, x1])  # 13
        y2 = self.c3k2_2(y2)  # 14
        
        # Head - Downsample path
        z1 = self.conv1(y2)  # 15
        z1 = self.wconcat([z1, y1, x2])  # 16
        z1 = self.c3k2_3(z1)  # 17
        
        z2 = self.conv2(z1)  # 18
        z2 = self.concat3([z2, x3])  # 19
        z2 = self.c3k2_4(z2)  # 20
        
        # Fusion and detection
        fused = self.EAFusion([x1, y2])  # 21
        return self.detect([fused, z1, z2])  # 22