import regex as re
import time
from collections import defaultdict
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries


SPECIAL_TOKEN = "<|endoftext|>"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Word:
    def __init__(self, word: bytes):
        self.word = word
        self.tokens = tuple([bytes([w]) for w in word])

    def __eq__(self, other):
        return self.word.__eq__(other.word)
    
    def __hash__(self):
        return hash(self.word)
    
    def __len__(self):
        return len(self.word)
    
    def pairs(self) -> list[tuple[bytes, bytes]]:
        return [
            (self.tokens[i], self.tokens[i + 1]) 
            for i in range(len(self.tokens) - 1)
        ]
    
    def merge(self, pair: tuple[bytes, bytes], merged: bytes) -> None:
        new_tokens, i = [], 0
        while i < len(self.tokens):
            if self.tokens[i:i + 2] == pair:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(self.tokens[i])
                i += 1
        self.tokens = tuple(new_tokens)


def pre_tokenize_async(
    text: str,
    special_tokens: list[str]
) -> tuple[list[tuple[str]], list[tuple[str]], dict[bytes, list[Word]]]:
    pairs: dict[bytes, int] = defaultdict(int)
    words: dict[Word, int] = defaultdict(int)
    pair2words: dict[bytes, set[Word]] = defaultdict(set)
    docs = re.split(re.escape("|".join(special_tokens)), text)
    for doc in docs:
        if doc.strip():  # Ignore empty parts
            for word in re.finditer(PAT, doc):
                if word:
                    word = word.group(0).replace("\r", "")
                    if not word:
                        continue
                    word = Word(word.encode("utf-8"))
                    words[word] += 1
                    word_pairs = word.pairs()
                    for pair in word_pairs:
                        pairs[pair] += 1
                        pair2words[pair].add(word)
    return (pairs, words, pair2words)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 1,
    batch_size: int = 1,
    debug: bool = False) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input file.
    """
    if debug:
        begin = time.time()
    
    vocab = {}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    l = len(vocab)  # initialize our vocabulary with the 256 byte values
    vocab |= {i + l: bytes([i]) for i in range(256)}
    merges = []

    pairs: dict[bytes, int] = defaultdict(int)
    words: dict[Word, int] = defaultdict(int)
    pair2words: dict[bytes, set[Word]] = defaultdict(set)

    def pre_tokenize(text: str) -> None:
        docs = re.split(re.escape("|".join(special_tokens)), text)
        for doc in docs:
            if doc.strip():  # Ignore empty parts
                for word in re.finditer(PAT, doc):
                    if word:
                        word = word.group(0).replace("\r", "")
                        if not word:
                            continue
                        word = Word(word.encode("utf-8"))
                        words[word] += 1
                        word_pairs = word.pairs()
                        for pair in word_pairs:
                            pairs[pair] += 1
                            pair2words[pair].add(word)

    # Pre-tokenize the input file
    if num_processes > 1:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_processes, SPECIAL_TOKEN.encode("utf-8"))
            boundaries = list(zip(boundaries[:-1], boundaries[1:]))
            for i in range(batch_size):
                if debug:
                    print(f"Running batch {i + 1} of {batch_size}")
                batch = i * num_processes
                with Pool(num_processes) as pool:
                    results = []
                    for start, end in boundaries[batch:batch + num_processes]:
                        f.seek(start)
                        chunk = f.read(end - start).decode("utf-8", errors="ignore")
                        result = pool.apply_async(
                            pre_tokenize_async,
                            (chunk, special_tokens)
                        )
                        results.append(result)
                    pool.close()
                    for result in results:
                        p, w, pw = result.get()
                        for pair, count in p.items():
                            pairs[pair] += count
                        for word, count in w.items():
                            words[word] += count
                        for pair, word_set in pw.items():
                            pair2words[pair].update(word_set)
                if debug:
                    print(f"Completed batch {i + 1} of {batch_size}")
    else:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_processes, SPECIAL_TOKEN.encode("utf-8"))
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                pre_tokenize(chunk)
    
    if debug:
        print(f"Pre-tokenization took {time.time() - begin:.2f} seconds")
        begin = time.time()

    # Merge pairs
    while len(vocab) < vocab_size:
        pair, _ = max(pairs.items(), key=lambda x: (x[1], x[0]))
        merged = b"".join(pair)
        merges.append((pair[0], pair[1]))
        vocab[len(vocab)] = merged
        for word in pair2words[pair]:
            count = words[word]
            old_pairs = [p for p in word.pairs() if p != pair]
            for old_pair in old_pairs:
                pairs[old_pair] -= count
            word.merge(pair, merged)
            new_pairs = [p for p in word.pairs() if p != pair]
            for new_pair in new_pairs:
                pairs[new_pair] += count
                pair2words[new_pair].add(word)
        del pairs[pair]
        del pair2words[pair]

    if debug:
        print(f"Merge took {time.time() - begin:.2f} seconds")

    return (vocab, merges)


if __name__ == "__main__":
    vocab, merges = train_bpe(
        input_path="tests/fixtures/tinystories_sample_5M.txt",
        vocab_size=1000,
        special_tokens=[SPECIAL_TOKEN],
        num_processes=16,
        debug=True)
