import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional
from torch import Tensor
from jaxtyping import Float

class SwiGLU(nn.Module):

    def __init__(
        self, 
        d_model: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1_weight = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype))
        self.w3_weight = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        # FFN(x) = SwiGLU(x,W1,W2,W3) = W2(SiLU(W1x) ⊙ W3x)
        gate = torch.einsum("...d,od->...o", x, self.w1_weight)
        value = torch.einsum("...d,od->...o", x, self.w3_weight)
        hidden = self.silu(gate) * value  
        out = torch.einsum("...d,od->...o", hidden, self.w2_weight)
        return out

    def silu(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return x * torch.sigmoid(x)
