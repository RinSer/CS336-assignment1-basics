import einops
import torch
from torch import Tensor
from jaxtyping import Float, Int
from .scaled_dot_product_attention import scaled_dot_product_attention
from .rope import RotaryPositionalEmbedding


class MultiheadSelfAttention(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None,
        theta: float | None = None,
        max_seq_len: int | None = None,
        q_proj_weight: Float[Tensor, " d_k d_in"] | None = None,
        k_proj_weight: Float[Tensor, " d_k d_in"] | None = None,
        v_proj_weight: Float[Tensor, " d_v d_in"] | None = None,
        o_proj_weight: Float[Tensor, " d_model d_v"] | None = None):
        """
        Multi-head self-attention

        Args:
            d_model (int): Dimensionality of the Transformer block inputs.
            num_heads (int): Number of heads to use in multi-head self-attention.
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
            theta (float | None): Θ value for the RoPE
            max_seq_len (int | None): Maximum sequence length to pre-cache.
            q_proj_weight (Float[Tensor, "d_k d_in"] | None): Weights for the Q projection
            k_proj_weight (Float[Tensor, "d_k d_in"] | None): Weights for the K projection
            v_proj_weight (Float[Tensor, "d_k d_in"] | None): Weights for the V projection
            o_proj_weight (Float[Tensor, "d_model d_v"] | None): Weights for the output projection
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        d_k = d_model // num_heads
        Q_weights: Float[Tensor, "d_model d_model"] = q_proj_weight if q_proj_weight is not None else \
            torch.empty((self.d_model, d_model), device=device, dtype=dtype)
        K_weights: Float[Tensor, "d_model d_model"] = k_proj_weight if k_proj_weight is not None else \
            torch.empty((self.d_model, d_model), device=device, dtype=dtype)
        V_weights: Float[Tensor, "d_model d_model"] = v_proj_weight if v_proj_weight is not None else \
            torch.empty((self.d_model, d_model), device=device, dtype=dtype)
        self.OW: Float[Tensor, "d_model d_model"] = o_proj_weight if o_proj_weight is not None else \
            torch.nn.Parameter(torch.empty((d_model, self.d_model), device=device, dtype=dtype))
        if q_proj_weight is None:
            torch.nn.init.xavier_uniform_(Q_weights)
        if k_proj_weight is None:
            torch.nn.init.xavier_uniform_(K_weights)
        if v_proj_weight is None:
            torch.nn.init.xavier_uniform_(V_weights)
        if o_proj_weight is None:
            torch.nn.init.xavier_uniform_(self.OW)
        # Divide weights into num_heads (h) batches
        self.QW = torch.nn.Parameter(
            einops.rearrange(Q_weights, '(h d_k) d_in -> h d_k d_in', h=self.num_heads))
        self.KW = torch.nn.Parameter(
            einops.rearrange(K_weights, '(h d_k) d_in -> h d_k d_in', h=self.num_heads))
        self.VW = torch.nn.Parameter(
            einops.rearrange(V_weights, '(h d_k) d_in -> h d_k d_in', h=self.num_heads))
        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device, dtype)

    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        """
        Batched multi-head self-attention implementation

        Args:
            in_features (Float[Tensor, "... sequence_length d_in"]): Input Tensor
            token_positions (Int[Tensor, " ... sequence_length"] | None): Tensor with the positions of the tokens

        Returns:
            Float[Tensor, " ... sequence_length d_out"]: Output Tensor
        """
        seq_len = in_features.shape[-2]
        # Causal mask: (seq_len, seq_len), True where j <= i
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))

        # Project input: (..., seq_len, d_in) x (num_heads, d_k, d_in) -> (..., num_heads, seq_len, d_k)
        Q = einops.einsum(in_features, self.QW, "... seq_len d_in, num_heads d_k d_in -> ... num_heads seq_len d_k")
        K = einops.einsum(in_features, self.KW, "... seq_len d_in, num_heads d_k d_in -> ... num_heads seq_len d_k")
        V = einops.einsum(in_features, self.VW, "... seq_len d_in, num_heads d_k d_in -> ... num_heads seq_len d_k")

        if token_positions is not None and self.rope is not None:
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)

        heads = scaled_dot_product_attention(Q, K, V, causal_mask)

        # Concatenate heads: (..., seq_len, num_heads * d_k)
        heads = einops.rearrange(heads, "... h seq_len d_k -> ... seq_len (h d_k)")

        # Output projection: (..., seq_len, d_model)
        output = einops.einsum(heads, self.OW, "... seq_len d_v, d_model d_v -> ... seq_len d_model")
        return output
