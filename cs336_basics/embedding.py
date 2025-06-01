import torch
from jaxtyping import Float


class Embedding(torch.nn.Module):

    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weights: Float[torch.Tensor, "num_embeddings embedding_dim"] | None = None):
        """
        Construct an embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors, i.e., d_model
            device (torch.device | None): Device to store the parameters on
            dtype: (torch.dtype | None): Data type of the parameters
            weights (Float[Tensor, "vocab_size d_model"] | None): The embedding vectors to fetch from
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weights = weights
        if self.weights is None:
            self.weights = torch.nn.Parameter(
                torch.empty(
                    (num_embeddings, embedding_dim), 
                    device=device,
                    dtype=dtype
                )
            )
            torch.nn.init.trunc_normal_(
                self.weights, mean=0.0, std=1, a=-3.0, b=3.0
            )

    def forward(self, x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "... embedding_dim"]:
        """
        Lookup the embedding vectors for the given token IDs.

        Args:
            x (torch.Tensor): Input tensor of shape (...)

        Returns:
            torch.Tensor: Output tensor of shape (... embedding_dim)
        """
        return self.weights[x]
