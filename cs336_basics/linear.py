import einops
import torch
from jaxtyping import Float


class Linear(torch.nn.Module):

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None,
        weights: Float[torch.Tensor, "out_features in_features"] | None = None):
        """
        Construct a linear transformation module.

        Args:
            in_features (int): Final dimension of the input
            out_features (int): Final dimension of the output
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
            weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = weights
        if self.weights is None:
            self.weights = torch.nn.Parameter(
                torch.empty(
                    (out_features, in_features), 
                    device=device, 
                    dtype=dtype
                )
            )
            std = 2.0 / (in_features + out_features)
            torch.nn.init.trunc_normal_(
                self.weights, mean=0.0, std=std, a=-3.0*std, b=3.0*std
            )

    def forward(self, x: Float[torch.Tensor, "... in_features"]) -> Float[torch.Tensor, "... out_features"]:
        """
        Apply the linear transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features)

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features)
        """
        return einops.einsum(x, self.weights,
            "... in_features, out_features in_features -> ... out_features")
