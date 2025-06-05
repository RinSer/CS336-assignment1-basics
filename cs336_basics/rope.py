import einops
import torch
from torch import Tensor
from jaxtyping import Int, Float


class RotaryPositionalEmbedding:

    def __init__(
        self, 
        theta: float,
        d_k: int, 
        max_seq_len: int, 
        device=None):
        """
        RoPE (Rotary Positional Embedding) implementation.

        Args:
            theta (float): Θ value for the RoPE
            d_k (int): dimension of query and key vectors
            max_seq_len (int): Maximum sequence length that will be inputted
            device (torch.device | None): Device to store the buffer on
        """
        pos: Int[Tensor, "max_seq_len 1"] = torch.arange(max_seq_len, device=device).unsqueeze(1)
        k: Float[Tensor, "1 d_k//2"] = torch.arange(d_k // 2, device=device).unsqueeze(0)
        theta_i_k: Float[Tensor, "max_seq_len d_k//2"] = pos / (theta ** (2 * k / d_k))
        self.cos: Float[Tensor, "max_seq_len d_k//2"] = torch.cos(theta_i_k)
        self.sin: Float[Tensor, "max_seq_len d_k//2"] = torch.sin(theta_i_k)

    def forward(
        self, 
        x: Float[Tensor, " ... sequence_length d_k"], 
        token_positions: Int[Tensor, " ... sequence_length"]
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        """
        Apply RoPE to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (..., sequence_length, d_k)
            token_positions (Tensor): Token positions of shape (..., sequence_length)

        Returns:
            Tensor: Tensor with RoPE applied
        """
        # Gather cos/sin for each position in the batch
        cos: Float[Tensor, "... seq_len d_k//2"] = self.cos[token_positions]
        sin: Float[Tensor, "... seq_len d_k//2"] = self.sin[token_positions]

        # Split x into even and odd parts
        x_even: Float[Tensor, "... seq_len d_k//2"] = x[..., 0::2]
        x_odd: Float[Tensor, "... seq_len d_k//2"]  = x[..., 1::2]

        # Apply rotation
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd  = x_even * sin + x_odd * cos

        # Interleave even and odd back together
        x_rotated = einops.rearrange(
            torch.stack([x_rotated_even, x_rotated_odd], dim=-1),
            "... sequence_length d_k two -> ... sequence_length (d_k two)"
        )
        return x_rotated
