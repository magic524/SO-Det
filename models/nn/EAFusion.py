import math
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from einops import rearrange
 
from ultralytics.nn.modules.conv import Conv, autopad
 
################################################SWSConv###############################################
    
class EAFusion(nn.Module):  # Cross-Layer Attention Fusion
    def __init__(self, dim, reduction=8):
        super(EAFusion, self).__init__()
        self.sa = SpatialAttention_CGA_1()  # Internal reference updated
        self.ca = ChannelAttention_CGA_1(dim, reduction)  # Internal reference updated
        self.pa = PixelAttention_CGA_1(dim)  # Internal reference updated
        self.sigmoid = nn.Sigmoid()
        self.SWS = Conv_SWS_1(dim, dim)  # Internal reference updated

        self.initial_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
        self.final_act = nn.SiLU()  
 
    def forward(self, data):
        x, y = data
        initial = x + y  
        
        cattn = self.ca(initial)
        sattn = self.sa(initial)

        pattn1 = sattn * cattn  # multiplicative fusion emphasizes co-saliency
        
        # pattn2 = self.sigmoid(self.pa(initial, pattn1))
        pattn2 = self.pa(initial, pattn1) ### Experienced double sigmoid, tried removing one 

        result = self.initial_weight * initial + pattn2 * x + (1 - pattn2) * y

        result = self.final_act(result)
        
        result = self.SWS(result)
        # result = self.conv(result)  # Original code commented out
        return result
    
class CGAFusion_SWS_ECA(nn.Module): ###Cross-Layer Attention Fusion 
    def __init__(self, dim, reduction=8):
        super(CGAFusion_SWS_ECA, self).__init__()
        self.sa = SpatialAttention_CGA_1()   
        self.ca = ECA_CGA_1(dim, reduction)   
        self.pa = PixelAttention_CGA_1(dim)  
        # self.conv = nn.Conv2d(dim, dim, 1, bias=True) 
        self.sigmoid = nn.Sigmoid()
        self.SWS = Conv_SWS_1(dim, dim) #  

        self.initial_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.final_act = nn.SiLU() 
 
    def forward(self, data):
        x, y = data
        initial = x + y 
        
        cattn = self.ca(initial)
        sattn = self.sa(initial)

        # pattn1 = sattn + cattn # 
        pattn1 = sattn * cattn 
        
        pattn2 = self.sigmoid(self.pa(initial, pattn1))

        result = self.initial_weight * initial + pattn2 * x + (1 - pattn2) * y

        result = self.final_act(result)
        
        result = self.SWS(result)
        #result = self.conv(result) 
        return result