import torch
import torch.nn as nn
import torch.nn.init as init

from typing import Optional

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        std = 1.0
        a = -3 * std
        b = 3 * std
        init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        out = self.weight[token_ids]
        return out