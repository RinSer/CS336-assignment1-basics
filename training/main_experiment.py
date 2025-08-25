import torch
from training_together import training_loop


if __name__ == "__main__":
    path_pref = "./data"
    training_loop(
        num_iterations=40_000,
        checkpoint_path=f"{path_pref}/owt_std.dat",
        checkpoints_step=1000,
        data_path=f"{path_pref}/owt_train_encoded.npy",
        batch_size=32,
        vocab_size=32_000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10_000,
        device=torch.device("cuda"),
        dtype=torch.float32,
        val_data_path=f"{path_pref}/owt_valid_encoded.npy",
        val_steps=10,
        from_save=False
    )
