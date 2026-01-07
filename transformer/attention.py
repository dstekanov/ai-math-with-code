import torch
from jaxtyping import Float, Bool
from torch import Tensor

from transformer.softmax import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... values d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    """
    Compute scaled dot-product attention.
    
    Formula: Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
    
    Args:
        Q: Query tensor of shape (..., queries, d_k)
        K: Key tensor of shape (..., keys, d_k)
        V: Value tensor of shape (..., values, d_v)
        mask: Optional boolean mask of shape (..., queries, keys)
              True = attend, False = don't attend (set to -inf before softmax)
              
    Returns:
        Output tensor of shape (..., queries, d_v)
    """
    # Get d_k for scaling
    d_k = Q.shape[-1]
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    # Q: (..., queries, d_k)
    # K^T: (..., d_k, keys)
    # scores: (..., queries, keys)
    scores = torch.einsum("...qd,...kd->...qk", Q, K) / (d_k ** 0.5)
    
    # Apply mask if provided
    if mask is not None:
        # Where mask is False, set scores to -inf
        # This makes softmax output 0 for those positions
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Apply softmax along the keys dimension (last dimension)
    attention_weights = softmax(scores, dim=-1)
    
    # Compute weighted sum of values
    # attention_weights: (..., queries, keys)
    # V: (..., keys, d_v)
    # output: (..., queries, d_v)
    output = torch.einsum("...qk,...kv->...qv", attention_weights, V)
    
    return output
