import einops
import torch
from jaxtyping import Float


class SwiGLU(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        w1_weight: Float[torch.Tensor, " d_ff d_model"] | None = None,
        w2_weight: Float[torch.Tensor, " d_model d_ff"] | None = None,
        w3_weight: Float[torch.Tensor, " d_ff d_model"] | None = None,
    ):
        """
        SwiGLU module

        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
            w1_weight (Float[torch.Tensor, "d_ff d_model"]): Stored weights for W1
            w2_weight (Float[torch.Tensor, "d_model d_ff"]): Stored weights for W2
            w3_weight (Float[torch.Tensor, "d_ff d_model"]): Stored weights for W3
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = w1_weight if w1_weight is not None else \
            torch.nn.Parameter(
                torch.empty(
                    (d_ff, d_model),
                    device=device,
                    dtype=dtype
                )
            )
        self.w2 = w2_weight if w2_weight is not None else \
            torch.nn.Parameter(
                torch.empty(
                    (d_model, d_ff),
                    device=device,
                    dtype=dtype
                )
            )
        self.w3 = w3_weight if w3_weight is not None else \
            torch.nn.Parameter(
                torch.empty(
                    (d_ff, d_model),
                    device=device,
                    dtype=dtype
                )
            )
        if w1_weight is None:
            torch.nn.init.xavier_uniform_(self.w1)
        if w2_weight is None:
            torch.nn.init.xavier_uniform_(self.w2)
        if w3_weight is None:
            torch.nn.init.xavier_uniform_(self.w3)


    def forward(
        self,
        in_features: Float[torch.Tensor, " ... d_model"],
    ) -> Float[torch.Tensor, " ... d_model"]:
        """Given the weights of a SwiGLU network, return
        the output of your implementation with these weights.

        Args:
            in_features (Float[torch.Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

        Returns:
            Float[torch.Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
        """
        # Project input to d_ff using W1 and W3
        W1x = einops.einsum(in_features, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        W3x = einops.einsum(in_features, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        # Elementwise product
        gated = silu(W1x) * W3x
        # Project back to d_model using W2
        output = einops.einsum(gated, self.w2, "... d_ff, d_model d_ff -> ... d_model")
        return output


def silu(tensor: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, " ..."]:
    return tensor * torch.sigmoid(tensor)
