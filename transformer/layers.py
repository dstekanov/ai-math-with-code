import torch
import torch.nn as nn
import torch.nn.init as init

from typing import Optional

class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = (2.0 / (in_features + out_features)) ** 0.5
        a = -3 * std
        b = 3 * std
        init.trunc_normal_(self.W, a=a, b=b, std=std)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...d,od->...o", x, self.W)