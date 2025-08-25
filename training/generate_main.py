import os
import torch

from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.train_bpe import SPECIAL_TOKEN
from decoding import decode


if __name__ == "__main__":
    ts_tokenizer = Tokenizer.from_pickles(
        "./data/bpe_expts_owt.pkl",
        [SPECIAL_TOKEN],
    )
    model = TransformerLM(
        vocab_size=32_000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10_000,
        device=torch.device("cuda"),
        dtype=torch.float32,
    )
    _ = load_checkpoint("./data/owt_std.dat", model)
    prompt = "Can you tell an interesting story?"
    for i in range(1, 21, 4):
        temperature = i / 10
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Prompt: {prompt} (temperature={temperature})")
        result = decode(
            model,
            ts_tokenizer,
            prompt,
            256,
            temperature=temperature
        )
        print(f"Answer: {result}")
    for i in range(1, 11, 2):
        top_p = i / 10
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Prompt: {prompt} (top_p={top_p})")
        result = decode(
            model,
            ts_tokenizer,
            prompt,
            256,
            top_p=top_p
        )
        print(f"Answer: {result}")
