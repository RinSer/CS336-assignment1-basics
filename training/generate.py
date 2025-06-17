import os
import torch

from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.train_bpe import SPECIAL_TOKEN
from decoding import decode


if __name__ == "__main__":
    ts_tokenizer = Tokenizer.from_pickles(
        "./data/bpe_tinystories.pkl",
        [SPECIAL_TOKEN],
    )
    model1 = TransformerLM(
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
    model2 = TransformerLM(
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
    _ = load_checkpoint("./data/tinystories.dat", model1)
    _ = load_checkpoint("./data/tinystories_std.dat", model2)
    prompt = "Can you tell an interesting story?"
    for i in range(1, 21, 4):
        temperature = i / 10
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Prompt: {prompt} (temperature={temperature})")
        result1 = decode(
            model1,
            ts_tokenizer,
            prompt,
            256,
            temperature=temperature
        )
        result2 = decode(
            model2,
            ts_tokenizer,
            prompt,
            256,
            temperature=temperature
        )
        print(f"Answer model1: {result1}")
        print(f"Answer model2: {result2}")
    for i in range(1, 11, 2):
        top_p = i / 10
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Prompt: {prompt} (top_p={top_p})")
        result1 = decode(
            model1,
            ts_tokenizer,
            prompt,
            256,
            top_p=top_p
        )
        result2 = decode(
            model2,
            ts_tokenizer,
            prompt,
            256,
            top_p=top_p
        )
        print(f"Answer model1: {result1}")
        print(f"Answer model2: {result2}")
