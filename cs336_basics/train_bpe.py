import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries


SPECIAL_TOKEN = "<|endoftext|>"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 16) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input file.
    """
    vocab = {chr(i):i+1 for i in range(256)}
    vocab[SPECIAL_TOKEN] = 0
    merges = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, SPECIAL_TOKEN.encode("utf-8"))
            
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Split the chunk by the special token
            docs = re.split(re.escape("|".join(special_tokens)), chunk)
            for doc in docs:
                if doc.strip():  # Ignore empty parts
                    words: dict[tuple[bytes], int] = {}
                    for word in re.finditer(PAT, doc):
                        if word:
                            word = tuple(word.group(0).encode("utf-8"))
                            words[word] = words.get(word, 0) + 1
                    print(f"Processed chunk: {doc.strip()}")
    raise NotImplementedError


if __name__ == "__main__":
    train_bpe(
        input_path="data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10_000,
        special_tokens=[SPECIAL_TOKEN])