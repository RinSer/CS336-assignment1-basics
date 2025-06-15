import torch
from jaxtyping import Float

from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.softmax import softmax
from cs336_basics.adamw import AdamW
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.train_bpe import SPECIAL_TOKEN


def top_p_sampling(
    logits: Float[torch.Tensor, " batch_size sequence_length vocab_size"], 
    p: float
) -> Float[torch.Tensor, " batch_size sequence_length vocab_size"]:
    """Apply nucleus (top-p) sampling to logits."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above threshold p
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -float('Inf')
    return logits


def decode(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> str:
    model.eval()
    generated = tokenizer.encode(prompt)
    initial_len = len(generated)
    generated = torch.tensor(generated, device=model.device, dtype=torch.int)
    eot = tokenizer.mapping[SPECIAL_TOKEN.encode("utf-8")]
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model.forward(generated)
            logits = logits[-1]  # consider only the last token
            # Apply temperature
            logits /= temperature
            # Apply top-p sampling
            logits = top_p_sampling(logits, top_p)
            probs = softmax(logits)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token))
            if next_token.item() == eot:
                break
    return tokenizer.decode(
        generated[initial_len:].cpu().numpy().astype("uint16"))


if __name__ == "__main__":
    ts_tokenizer = Tokenizer.from_pickles(
        "./data/bpe_tinystories.pkl",
        [SPECIAL_TOKEN],
    )
    model = TransformerLM(
        vocab_size=10_000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10_000,
        device=torch.device("cuda"),
        dtype=torch.float32,
    )
    optimizer = AdamW(
        model.parameters(),
        1e-3,
        (0.9, 0.999),
        1e-8,
        1e-2,
    )
    d = load_checkpoint("./data/test_training_loop.dat", model, optimizer)
    prompt = "test prompt"
    print(f"Prompt: {prompt}")
    result = decode(
        model,
        ts_tokenizer,
        prompt,
        12
    )
    print(f"Answer: {result}")
