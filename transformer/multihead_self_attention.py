import torch
import torch.nn as nn

from transformer.attention import scaled_dot_product_attention
from transformer.rotary_positional_embedding import RotaryPositionalEmbedding


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048, device: torch.device | None = None, dtype: torch.dtype | None = None, use_rope: bool = True, theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.use_rope = use_rope
        
        # Learnable parameters
        self.W_Q = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.W_K = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.W_V = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.W_O = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        
        # RoPE for positional embeddings (optional)
        if use_rope:
            d_k = d_model // num_heads
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=d_k,
                max_seq_len=max_seq_len,
                device=device
            )
        else:
            self.rope = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        d_k = self.d_model // self.num_heads
        
        # 1. Project to Q, K, V
        Q = torch.einsum("...d,od->...o", x, self.W_Q)  # (batch, seq_len, d_model)
        K = torch.einsum("...d,od->...o", x, self.W_K)
        V = torch.einsum("...d,od->...o", x, self.W_V)
        
        # 2. Reshape for multi-head: (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
        Q = Q.view(batch, seq_len, self.num_heads, d_k)
        K = K.view(batch, seq_len, self.num_heads, d_k)
        V = V.view(batch, seq_len, self.num_heads, d_k)
        
        # 3. Transpose: (batch, seq_len, num_heads, d_k) → (batch, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 4. Apply RoPE to Q and K (not V!) if enabled
        if self.use_rope:
            token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        # 5. Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        
        # 6. Scaled dot-product attention
        output = scaled_dot_product_attention(Q, K, V, mask)  # (batch, num_heads, seq_len, d_k)
        
        # 7. Transpose back and concat heads: (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        
        # 8. Output projection
        output = torch.einsum("...d,od->...o", output, self.W_O)
        
        return output