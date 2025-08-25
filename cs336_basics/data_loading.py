import numpy as np
import torch


def data_loading(
    dataset: np.typing.ArrayLike, 
    batch_size: int, 
    context_length: int, 
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (torch.device): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    data_len = len(dataset)
    # Randomly sample start indices so that context_length+1 tokens fit
    starts = np.random.randint(0, data_len - context_length, size=batch_size, dtype=np.uint32)
    # Gather input and label sequences
    input_seqs = np.array([dataset[s:s+context_length] for s in starts])
    label_seqs = np.array([dataset[s+1:s+context_length+1] for s in starts])
    # Convert to tensors and move to device
    input_tensor = torch.tensor(input_seqs, device=device)
    label_tensor = torch.tensor(label_seqs, device=device)
    return input_tensor, label_tensor
