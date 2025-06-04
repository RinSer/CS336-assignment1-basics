import einops
import torch
from jaxtyping import Float


def swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[torch.Tensor, " d_ff d_model"],
    w2_weight: Float[torch.Tensor, " d_model d_ff"],
    w3_weight: Float[torch.Tensor, " d_ff d_model"],
    in_features: Float[torch.Tensor, " ... d_model"],
) -> Float[torch.Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[torch.Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[torch.Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[torch.Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[torch.Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[torch.Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Check dimensions
    assert w1_weight.shape == (d_ff, d_model), "W1 weight shape mismatch"
    assert w2_weight.shape == (d_model, d_ff), "W2 weight shape mismatch"
    assert w3_weight.shape == (d_ff, d_model), "W3 weight shape mismatch"
    assert in_features.shape[-1] == d_model, "Input features last dimension must match d_model"
    # Project input to d_ff using W1 and W3
    W1x = einops.einsum(in_features, w1_weight, "... d_model, d_ff d_model -> ... d_ff")
    W3x = einops.einsum(in_features, w3_weight, "... d_model, d_ff d_model -> ... d_ff")
    # Apply SiLU activation to W1x
    SiLU = W1x * torch.sigmoid(W1x)
    # Elementwise product
    gated = SiLU * W3x
    # Project back to d_model using W2
    output = einops.einsum(gated, w2_weight, "... d_ff, d_model d_ff -> ... d_model")
    return output
