import torch
from jaxtyping import Float
from torch import Tensor


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """
    Apply softmax operation on a tensor along the specified dimension.
    
    Uses numerical stability trick: subtract max value before exp to avoid overflow.
    
    Formula: softmax(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))
    
    Args:
        x: Input tensor of any shape
        dim: Dimension along which to apply softmax
        
    Returns:
        Tensor with same shape as input, with softmax applied along dim
    """
    # Subtract max for numerical stability
    # keepdim=True preserves dimension for broadcasting
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    
    # Compute exp
    exp_x = torch.exp(x_shifted)
    
    # Normalize by sum
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    
    return exp_x / sum_exp