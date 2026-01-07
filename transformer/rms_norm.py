import torch
import torch.nn as nn
import torch.nn.init as init

from typing import Optional

class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        rms = (mean_sq + self.eps).sqrt()
        y = x / rms
        y = y * self.weight
        return y.to(in_dtype)