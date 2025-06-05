import einops
from torch import Tensor
from jaxtyping import Float
from .softmax import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of scaled dot product attention.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    scores = einops.einsum(Q, K, 
        "... queries d_k, ... keys d_k -> ... queries keys") / d_k**0.5

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    normalized = softmax(scores)
    output = einops.einsum(normalized, V,
        "... queries keys, ... keys d_v -> ... queries d_v")

    return output