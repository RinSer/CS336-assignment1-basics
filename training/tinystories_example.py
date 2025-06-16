import torch
from training_together import training_loop


if __name__ == "__main__":
    path_pref = "./data"
    training_loop(
        num_iterations=40_000,
        checkpoint_path=f"{path_pref}/tinystories.dat",
        checkpoints_step=1000,
        data_path=f"{path_pref}/tinystories_train_encoded.npy",
        batch_size=32,
        vocab_size=10_000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10_000,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5,
        max_norm=2.0,
        lr_max=1e-3,
        lr_min=1e-4,
        t_w=100,
        t_c=20_000,
        device=torch.device("cuda"),
        dtype=torch.float32,
        val_data_path=f"{path_pref}/tinystories_valid_encoded.npy",
        val_steps=2,
        from_save=False
    )
