import torch
import torch.nn as nn

from transformer.embedding import Embedding
from transformer.transformer_block import TransformerBlock
from transformer.rms_norm import RMSNorm


class TransformerLM(nn.Module):
    """
    Transformer Language Model.
    
    Architecture:
    1. Token Embedding
    2. num_layers × TransformerBlock
    3. Final RMSNorm
    4. Output projection (to vocabulary logits)
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_rope: bool = True,
        theta: float = 10000.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # 1. Token Embedding
        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )
        
        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                device=device,
                dtype=dtype,
                use_rope=use_rope,
                theta=theta
            )
            for _ in range(num_layers)
        ])
        
        # 3. Final RMSNorm
        self.final_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        
        # 4. Output projection (LM head)
        # Maps from d_model to vocab_size to get logits
        self.lm_head = nn.Parameter(torch.empty(vocab_size, d_model, device=device, dtype=dtype))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Input token IDs of shape (batch, seq_len)
        
        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        # 1. Token Embedding: (batch, seq_len) → (batch, seq_len, d_model)
        x = self.token_embedding(token_ids)
        
        # 2. Pass through all Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # 3. Final normalization
        x = self.final_norm(x)
        
        # 4. Project to vocabulary: (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
        logits = torch.einsum("...d,vd->...v", x, self.lm_head)
        
        return logits
