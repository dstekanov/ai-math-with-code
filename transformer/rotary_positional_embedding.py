import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        k = torch.arange(0, self.d_k // 2, device=self.device)  # [0, 1, 2, ..., d_k/2-1]
        theta_k = self.theta ** (-2.0 * k / self.d_k)  # shape: (d_k/2,)

        angles = token_positions.unsqueeze(-1) * theta_k  # broadcasting

        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        
        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_even * sin + x_odd * cos
        
        return torch.stack([x_even_rotated, x_odd_rotated], dim=-1).flatten(-2)