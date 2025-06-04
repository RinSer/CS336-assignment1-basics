import torch
from jaxtyping import Float


def softmax(x: Float[torch.Tensor, ""], dim: int = -1) -> Float[torch.Tensor, ""]:    
    """
    Compute the softmax of a tensor along a specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the softmax. Default is -1 (last dimension).

    Returns:
        torch.Tensor: Softmax of the input tensor.
    """
    max_values, _ = torch.max(x, dim=dim, keepdim=True)
    x_sub = x - max_values
    x_exp = torch.exp(x_sub)
    exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    softmax_result = x_exp / exp_sum
    return softmax_result
