import pickle
import time
from cs336_basics.train_bpe import train_bpe


PICKLE_FILE = "./bpe_tinystories.pkl"


def run_train_bpe():
    """
    Train BPE on the tinystories dataset.
    """
    start = time.time()
    
    vocab, merges = train_bpe(
        input_path="../data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
        num_processes=1,
        debug=True)
    
    print(f"Time taken to train BPE: {time.time() - start:.2f} seconds")

    with open(PICKLE_FILE, "wb") as f:
        pickle.dump({
            "vocab": vocab,
            "merges": merges,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    run_train_bpe()
    with open(PICKLE_FILE, "rb") as f:
        data = pickle.load(f)
        vocab = data["vocab"]
        print(sorted(vocab.values(), key=lambda v: len(v), reverse=True)[:5])
