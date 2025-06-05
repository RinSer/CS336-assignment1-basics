import torch
from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], 
    targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, 
    computes the average cross-entropy loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # Subtract max for numerical stability
    max_logits, _ = torch.max(inputs, dim=-1, keepdim=True)
    logits_stable = inputs - max_logits

    # Compute log-sum-exp
    logsumexp = torch.log(torch.sum(torch.exp(logits_stable), dim=-1, keepdim=True))

    # Gather the logits at the target indices
    target_logits = torch.gather(logits_stable, dim=-1, index=targets.unsqueeze(-1))

    # Cross entropy: -log(e^target_logit/sum(exp)) = 
    # -(log(e^target_logit) - log(sum(exp))) =
    # log(sum(exp))) - target_logit
    loss = logsumexp - target_logits

    return loss.mean()
