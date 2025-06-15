import torch
from jaxtyping import Int, Float

from .embedding import Embedding
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm
from .linear import Linear


class TransformerLM(torch.nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float, 
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weights: dict[str, torch.Tensor] | None = None):
        """
        Full Transformer LM

        Args:
            vocab_size (int): The number of unique items in the output vocabulary to be predicted.
            context_length (int): The maximum number of tokens to process at once.
            d_model (int): The dimensionality of the model embeddings and sublayer outputs.
            num_layers (int): The number of Transformer layers to use.
            num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
                evenly divisible by `num_heads`.
            d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
            rope_theta (float): The RoPE Theta parameter.
            weights (dict[str, Tensor]): 
                State dict of our reference implementation. {num_layers} refers to an
                integer between `0` and `num_layers - 1` (the layer index).
                The keys of this dictionary are:
                - `token_embeddings.weight`
                    Token embedding matrix. Shape is (vocab_size, d_model).
                - `layers.{num_layers}.attn.q_proj.weight`
                    The query projections for all `num_heads` attention heads.
                    Shape is (num_heads * (d_model / num_heads), d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
                - `layers.{num_layers}.attn.k_proj.weight`
                    The key projections for all `num_heads` attention heads.
                    Shape is (num_heads * (d_model / num_heads), d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
                - `layers.{num_layers}.attn.v_proj.weight`
                    The value projections for all `num_heads` attention heads.
                    Shape is (num_heads * (d_model / num_heads), d_model).
                    The rows are ordered by matrices of shape (num_heads, d_v),
                    so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
                - `layers.{num_layers}.attn.output_proj.weight`
                    Weight of the multi-head self-attention output projection
                    Shape is ((d_model / num_heads) * num_heads, d_model).
                - `layers.{num_layers}.ln1.weight`
                    Weights of affine transform for the first RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
                - `layers.{num_layers}.ffn.w1.weight`
                    Weight of the first linear transformation in the FFN.
                    Shape is (d_model, d_ff).
                - `layers.{num_layers}.ffn.w2.weight`
                    Weight of the second linear transformation in the FFN.
                    Shape is (d_ff, d_model).
                - `layers.{num_layers}.ffn.w3.weight`
                    Weight of the third linear transformation in the FFN.
                    Shape is (d_model, d_ff).
                - `layers.{num_layers}.ln2.weight`
                    Weights of affine transform for the second RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
                - `ln_final.weight`
                    Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                    Shape is (d_model, ).
                - `lm_head.weight`
                    Weights of the language model output embedding.
                    Shape is (vocab_size, d_model).
        """
        super().__init__()
        self.device, self.dtype = device, dtype
        self.token_embeddings = Embedding(
            vocab_size, d_model,
            device=device, dtype=dtype,
            weights=weights["token_embeddings.weight"] \
                if weights and "token_embeddings.weight" in weights else None
        )
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, context_length, theta=rope_theta,
                device=device, dtype=dtype,
                weights={
                    k.replace(f"layers.{i}.", ""): v for k, v in weights.items() 
                    if k.startswith(f"layers.{i}.")
                } if weights else None
            ) for i in range(num_layers)
        ])
        self.ln_final = RMSNorm(
            d_model, device=device, dtype=dtype, 
            weights=weights["ln_final.weight"] \
                if weights and "ln_final.weight" in weights else None
        )
        self.lm_head = Linear(
            d_model, vocab_size,
            device=device, dtype=dtype,
            weights=weights["lm_head.weight"] \
                if weights and "lm_head.weight" in weights else None
        )

    def forward(
        self,
        in_indices: Int[torch.Tensor, " batch_size sequence_length"]
    ) -> Float[torch.Tensor, " batch_size sequence_length vocab_size"]:
        """
        Given the weights of a Transformer language model and input indices,
        returns the output of running a forward pass on the input indices.

        Args:
            in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
                `sequence_length` is at most `context_length`.

        Returns:
            Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
            next-word distribution for each token.
        """
        x = self.token_embeddings.forward(in_indices)
        for block in self.layers:
            x = block.forward(x)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)
        return x
