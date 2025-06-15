import numpy as np
import torch
import matplotlib.pyplot as plt

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.data_loading import data_loading
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.gradient_clipping import gradient_clipping


def training_loop(
    num_iterations: int,
    checkpoint_path: str,
    checkpoints_step: int,
    data_path: str,
    batch_size: int,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    max_norm: int = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    from_save: bool = False,
    val_data_path: str = None,
    val_steps: int = 1000,) -> tuple[dict[int, float], dict[int, float]]:
    model = TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        device=device,
        dtype=dtype
    )
    optimizer = AdamW(
        model.parameters(),
        lr,
        betas,
        eps,
        weight_decay
    )
    iteration = 0
    if from_save:
        iteration = load_checkpoint(checkpoint_path, model, optimizer)
    dataset = np.memmap(data_path, mode="r", dtype=np.uint16)
    model.train()
    losses, val_losses = {}, {}
    for i in range(iteration, num_iterations):
        # Get batch
        x, y = data_loading(dataset, batch_size, context_length, device)
        x, y = x.int(), y.long()
        # Forward pass
        logits = model.forward(x)
        # Count loss
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        gradient_clipping(model.parameters(), max_norm)

        optimizer.step()

        losses[i + 1] = loss.item()
        # Validation logging
        if val_data_path is not None and (i + 1) % val_steps == 0 or i == num_iterations - 1:
            val_dataset = np.memmap(data_path, mode="r")
            model.eval()
            with torch.no_grad():
                val_x, val_y = data_loading(val_dataset, batch_size, context_length, device)
                val_x, val_y = val_x.int(), val_y.long()
                val_logits = model.forward(val_x)
                val_loss = cross_entropy(
                    val_logits.view(-1, val_logits.size(-1)),
                    val_y.view(-1)
                )
                print(f"Iteration {i + 1}/{num_iterations}, validation loss: {val_loss.item()}")
                val_losses[i + 1] = val_loss.item()
            model.train()

        if (i + 1) % checkpoints_step == 0 or i == num_iterations - 1:
            save_checkpoint(model, optimizer, i + 1, checkpoint_path)
            print(f"Checkpoint saved at iteration {i + 1}, loss: {loss.item()}")

        if (i + 1) % 100 == 0 or i == num_iterations - 1:
            print(f"Iteration {i + 1}/{num_iterations}, loss: {loss.item()}")
    
    return losses, val_losses


if __name__ == "__main__":
    path_pref = "./data"
    losses, val_losses = training_loop(
        num_iterations=2500,
        checkpoint_path=f"{path_pref}/test_training_loop_2.dat",
        checkpoints_step=1000,
        data_path=f"{path_pref}/tinystories_train_encoded.npy",
        batch_size=10,
        vocab_size=10_000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10_000,
        device=torch.device("cuda"),
        dtype=torch.float32,
        val_data_path=f"{path_pref}/tinystories_valid_encoded.npy",
        val_steps=1000
    )
    plt.plot(list(losses.keys()), list(losses.values()), color="red")
    plt.plot(list(val_losses.keys()), list(val_losses.values()), color="blue")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()
