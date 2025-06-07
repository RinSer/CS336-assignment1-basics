import torch
from jaxtyping import Float, Int

from .rmsnorm import RMSNorm
from .multihead_self_attention import MultiheadSelfAttention
from .positionwise_feedforward import swiglu


class TransformerBlock:

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weights: dict[str, torch.Tensor] = None):
        """
        Transformer block

        Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        device (torch.device | None): Device to store the parameters on
        dtype (torch.dtype | None): Data type of the parameters
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.rmsnorm1 = RMSNorm(d_model, device=device, dtype=dtype, 
            weights=weights["ln1.weight"] if weights and "ln1.weight" in weights else None)
        self.mhsa = MultiheadSelfAttention(d_model, num_heads, device=device, dtype=dtype,
            theta=theta, max_seq_len=max_seq_len,
            q_proj_weight=weights["attn.q_proj.weight"] 
                if weights and "attn.q_proj.weight" in weights else None,
            k_proj_weight=weights["attn.k_proj.weight"] 
                if weights and "attn.k_proj.weight" in weights else None,
            v_proj_weight=weights["attn.v_proj.weight"] 
                if weights and "attn.v_proj.weight" in weights else None,
            o_proj_weight=weights["attn.output_proj.weight"] 
                if weights and "attn.output_proj.weight" in weights else None,
        )
        self.rmsnorm2 = RMSNorm(d_model, device=device, dtype=dtype, 
            weights=weights["ln2.weight"] if weights and "ln2.weight" in weights else None)
        self.w1 = weights["ffn.w1.weight"] if weights and "ffn.w1.weight" in weights else \
            torch.nn.Parameter(
                torch.empty(
                    (d_model, d_ff),
                    device=device,
                    dtype=dtype
                )
            )
        self.w2 = weights["ffn.w2.weight"] if weights and "ffn.w2.weight" in weights else \
            torch.nn.Parameter(
                torch.empty(
                    (d_ff, d_model),
                    device=device,
                    dtype=dtype
                )
            )
        self.w3 = weights["ffn.w3.weight"] if weights and "ffn.w3.weight" in weights else \
            torch.nn.Parameter(
                torch.empty(
                    (d_model, d_ff),
                    device=device,
                    dtype=dtype
                )
            )

    def forward(
        self,
        in_features: Float[torch.Tensor, " batch sequence_length d_model"],
        token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
    ) -> Float[torch.Tensor, " batch sequence_length d_model"]:
        x = in_features

        rmsnorm1 = self.rmsnorm1.forward(x)

        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)
        mhsa = self.mhsa.forward(rmsnorm1, token_positions)

        x = x + mhsa

        rmsnorm2 = self.rmsnorm2.forward(x)

        ff = swiglu(
            self.d_model,
            self.d_ff,
            self.w1,
            self.w2,
            self.w3,
            rmsnorm2
        )

        x = x + ff
        
        return x
