import torch
import torch.nn as nn
import torch.nn.functional as F

class SEAttention(nn.Module):
    """Squeeze-and-Excitation Attention module."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class CAM(nn.Module):
    """Channel Attention Module."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)
    
class GEAttention(nn.Module):
    """Gather-Excite Attention module."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.sigmoid = nn.Sigmoid()

        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

    def forward(self, x):
        b, c, h, w = x.size()

        y = self.pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)
    
class scSE(nn.Module):
    """Concurrent Spatial and Channel Squeeze & Excitation."""
    def __init__(self, channel):
        super().__init__()
        # Channel excitation
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//16, channel, 1),
            nn.Sigmoid()
        )
        # Spatial excitation
        self.sSE = nn.Sequential(
            nn.Conv2d(channel, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
    
class ECAAttention(nn.Module):
    """Efficient Channel Attention module."""
    def __init__(self, c, b=1, gamma=2):
        super().__init__()
       
        k_size = int(abs((math.log(c, 2) + b) / gamma))
        k_size = k_size if k_size % 2 else k_size + 1 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x) 
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    
class space_to_depth(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
    
class CDowns(nn.Module):
    """Module combining Conv and space_to_depth with ECA attention mechanism"""
    
    def __init__(self, c1, c2, k=1, s=1, dimension=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = Conv(c1, c2//4, k, s, p, g, d, act) 
        self.space_to_depth = space_to_depth(dimension)
        self.eca = ECAAttention(c2 * 4) 

    def forward(self, x):
        x = self.conv(x)
        x = self.space_to_depth(x)
        return self.eca(x) 
    
class CDown(nn.Module):
    """Module combining Conv and space_to_depth with ECA attention mechanism"""
    
    def __init__(self, c1, c2, k=1, s=1, dimension=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g, d, act) 
        self.space_to_depth = space_to_depth(dimension)
        self.eca = ECAAttention(c2)  
        
    def forward(self, x):
        x = self.conv(x)
        x = self.space_to_depth(x)
        return self.eca(x)