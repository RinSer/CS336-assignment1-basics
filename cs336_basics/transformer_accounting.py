def count_flops(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
) -> int:
    transformer_blocks = num_layers * transformer_block_flops(
        context_length, d_model, num_heads, d_ff
    )
    fin_rmsnorm = rmsnorm_flops(context_length, d_model)
    linear = 2 * d_model**2 * vocab_size
    total = transformer_blocks + fin_rmsnorm + linear
    print(f"Transfomer blocks to total: {transformer_blocks / total}")
    print(f"Final RMSNorm to total: {fin_rmsnorm / total}")
    print(f"Linear to total: {linear / total}")
    return total


def rmsnorm_flops(
    context_length: int, 
    d_model: int) -> int:
    return context_length * (4 * d_model) + 2


def mhsa_flops(
    context_length: int,
    d_model: int,
    num_heads: int
) -> int:
    d_k = d_model // num_heads
    qkv_flops = 3 * 2 * d_k * num_heads * context_length * d_model**2
    rope = 2 * 3 * d_k * context_length
    sdpa = 2 * d_model**2 * d_k**2 + \
        4 * d_model**2 - d_model + \
        2 * d_model**4
    out_flops = 2 * context_length * d_model**3
    total = qkv_flops + rope + sdpa + out_flops
    print(f"QKV to total: {qkv_flops / total}")
    print(f"RoPE to total: {rope / total}")
    print(f"SDPA to total: {sdpa / total}")
    print(f"Out to total: {out_flops / total}")
    return total


def swiglu_flops(
    d_model: int,
    d_ff: int
) -> int:
    return 4 * d_model**2 * d_ff + \
        2 * d_ff**2 * d_model


def transformer_block_flops(
    context_length: int,
    d_model: int,
    num_heads: int,
    d_ff: int
) -> int:
    sums = 2 * context_length * d_model
    rmsnorms = 2 * rmsnorm_flops(context_length, d_model)
    mhsa = mhsa_flops(context_length, d_model, num_heads)
    swiglu = swiglu_flops(d_model, d_ff)
    total = sums + rmsnorms + mhsa + swiglu
    print(f"Sums to total: {sums / total}")
    print(f"RMSNorms to total: {rmsnorms / total}")
    print(f"MHSA to total: {mhsa / total}")
    print(f"SwiGLU to total: {swiglu / total}")
    return total


def count_parameters(
    vocab_size: int,
    num_layers: int,
    d_model: int,
    d_ff: int,
) -> int:
    embedding = vocab_size * d_model
    transfomer_blocks = num_layers * (
        (2 * d_model) + (4 * d_model**2) + (3 * d_ff * d_model)
    )
    fin_rmsmorm = d_model
    linear = d_model * vocab_size
    return embedding + transfomer_blocks + fin_rmsmorm + linear


if __name__ == "__main__":
    print("Question a")
    print(count_parameters(
        vocab_size=50_257,
        num_layers=48,
        d_model=1600,
        d_ff=6400
    ))
    print("Question b")
    print(count_flops(
        vocab_size=50_257,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400
    ))
    print("Question d")
    print(count_flops(
        vocab_size=50_257,
        context_length=1024,
        num_layers=12,
        d_model=768,
        num_heads=12,
        d_ff=6400
    ))
    print(count_flops(
        vocab_size=50_257,
        context_length=1024,
        num_layers=24,
        d_model=1024,
        num_heads=16,
        d_ff=6400
    ))
    print(count_flops(
        vocab_size=50_257,
        context_length=1024,
        num_layers=36,
        d_model=1280,
        num_heads=20,
        d_ff=6400
    ))
    print("Question e")
    print(count_flops(
        vocab_size=50_257,
        context_length=16_384,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400
    ))
