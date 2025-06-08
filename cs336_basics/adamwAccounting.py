from transformer_accounting import count_parameters, count_flops


def adamw_memory_usage(
    vocab_size: int,
    num_layers: int, 
    d_model: int,
    dsize: int = 4,
) -> int:
    d_ff = 4 * d_model
    total_parameters = count_parameters(
        vocab_size, num_layers, d_model, d_ff
    )
    total_parameters *= 4  # + gradient + m + v
    total_parameters += 8 * vocab_size  # + logits
    total_bytes = total_parameters * dsize  # bytes
    return total_bytes / (1024 ** 3)  # GB


def adamw_flops_usage(
    vocab_size: int,
    num_layers: int, 
    d_model: int,
) -> int:
    d_ff = 4 * d_model
    total_parameters = count_parameters(
        vocab_size, num_layers, d_model, d_ff
    )
    return 12 * total_parameters + 2 * total_parameters #  With weight decay


if __name__ == "__main__":
    total_memory = adamw_memory_usage(
        vocab_size=50_257,
        num_layers=48,
        d_model=1600,
    )
    print(f"Total memory usage for batch 1: {total_memory}")
    print(f"Batch size to with 80GB limit: {80 / total_memory}")
    adamw_flops = adamw_flops_usage(
        vocab_size=50_257,
        num_layers=48,
        d_model=1600,
    )
    print(f"AdamW FLOPs usage: {adamw_flops}")
    effective_flops_per_second = 10**13  # ~ 10 TeraFLOPs
    forward_path = count_flops(
        vocab_size=50_257,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400
    )
    forward_path *= 3  # forward + backward
    total_flops = (forward_path + adamw_flops) * 400_000
    total_days = (total_flops / effective_flops_per_second) / (3600 * 24)
    print(f"Total days to train on one A100: {total_days}")
    print(f"Total years to train on one A100: {total_days / 360:.2f}")
