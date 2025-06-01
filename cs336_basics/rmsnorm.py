import torch
from jaxtyping import Float


class  RMSNorm(torch.nn.Module):

    def __init__(
        self, 
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weights: Float[torch.Tensor, "num_embeddings embedding_dim"] | None = None):
        """
        Construct the RMSNorm module.
        
        Args:
            d_model (int): Hidden dimension of the model
            eps (float): Epsilon value for numerical stability
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = weights
        if self.weights is None:
            self.weights = torch.nn.Parameter(
                torch.empty(
                    d_model, 
                    device=device,
                    dtype=dtype
                )
            )
            torch.nn.init.ones_(self.weights)

    def forward(self, x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) 
        and return a tensor of the same shape.

        Args:
            x (torch.Tensor): Input tensor of shape (...)

        Returns:
            torch.Tensor: Output tensor of shape (...)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        result = x * self.weights / rms
        return result.to(in_dtype)
