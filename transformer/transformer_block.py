import torch
import torch.nn as nn

from transformer.rms_norm import RMSNorm
from transformer.multihead_self_attention import MultiHeadSelfAttention
from transformer.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with:
    - Sublayer 1: x + MultiHeadSelfAttention(RMSNorm(x))
    - Sublayer 2: y + SwiGLU(RMSNorm(y))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_rope: bool = True,
        theta: float = 10000.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Sublayer 1: RMSNorm + Multi-Head Self-Attention
        self.norm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
            use_rope=use_rope,
            theta=theta
        )
        
        # Sublayer 2: RMSNorm + SwiGLU Feed-Forward
        self.norm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Sublayer 1: Multi-Head Self-Attention with residual connection
        y = x + self.attn(self.norm1(x))
        
        # Sublayer 2: Feed-Forward Network with residual connection
        z = y + self.ffn(self.norm2(y))
        
        return z
