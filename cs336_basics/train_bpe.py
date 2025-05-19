import regex as re
import time
from multiprocessing import Pool, Manager
from cs336_basics.pretokenization_example import find_chunk_boundaries
import pickle


SPECIAL_TOKEN = "<|endoftext|>"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Word:
    def __init__(self, word: str):
        self.word = word
        self.tokens = tuple(word)

    def __eq__(self, other):
        return self.word.__eq__(other.word)
    
    def __hash__(self):
        return hash(self.word)
    
    def __len__(self):
        return len(self.word)
    
    def pairs(self) -> list[tuple[str, str]]:
        return [
            (self.tokens[i], self.tokens[i + 1]) 
            for i in range(len(self.tokens) - 1)
        ]
    
    def merge(self, pair: tuple[str, str], merged: str) -> None:
        new_tokens, i = [], 0
        while i < len(self.tokens):
            if self.tokens[i:i + 2] == pair:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(self.tokens[i])
                i += 1
        self.tokens = tuple(new_tokens)


def pre_tokenize(
        text: str,
        special_tokens: list[str],
        words: list[tuple[str]]) -> None:
    docs = re.split(re.escape("|".join(special_tokens)), text)
    for doc in docs:
        if doc.strip():  # Ignore empty parts
            for word in re.finditer(PAT, doc):
                if word:
                    word = word.group(0)
                    words.append(word)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 1,
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

    pairs: dict[str, int] = {}
    words: dict[Word, int] = {}
    pair2words: dict[str, list[Word]] = {}

    def pre_tokenize(text: str) -> None:
        docs = re.split(re.escape("|".join(special_tokens)), text)
        for doc in docs:
            if doc.strip():  # Ignore empty parts
                for word in re.finditer(PAT, doc):
                    if word:
                        word = word.group(0).replace("\r", "")
                        if not word:
                            continue
                        word = Word(word)
                        words[word] = words.get(word, 0) + 1
                        word_pairs = word.pairs()
                        for pair in word_pairs:
                            pairs[pair] = pairs.get(pair, 0) + 1
                            if pair not in pair2words:
                                pair2words[pair] = set()
                            pair2words[pair].add(word)

    # Pre-tokenize the input file
    if num_processes > 1:
        all_words = Manager().list()
        with Pool(num_processes) as pool:
            with open(input_path, "rb") as f:
                boundaries = find_chunk_boundaries(
                    f, num_processes, SPECIAL_TOKEN.encode("utf-8"))
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    f.seek(start)
                    chunk = f.read(end - start).decode("utf-8", errors="ignore")
                    pool.apply_async(pre_tokenize, (chunk, special_tokens, all_words))      
            pool.close()
            pool.join()
    else:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_processes, SPECIAL_TOKEN.encode("utf-8"))
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                pre_tokenize(chunk)

    if num_processes > 1:
        for word in all_words:
            word = Word(word)
            words[word] = words.get(word, 0) + 1
            pairs = word.pairs()
            for pair in pairs:
                pairs[pair] = pairs.get(pair, 0) + 1
                if pair not in pair2words:
                    pair2words[pair] = set()
                pair2words[pair].add(word)
    
    if debug:
        print(f"Pre-tokenization took {time.time() - begin:.2f} seconds")
        begin = time.time()

    # Merge pairs
    while len(vocab) < vocab_size:
        pair, _ = max(pairs.items(), key=lambda x: (x[1], x[0]))
        merged = "".join(pair)
        merges.append((pair[0].encode("utf-8"), pair[1].encode("utf-8")))
        vocab[len(vocab)] = merged.encode("utf-8")
        for word in pair2words[pair]:
            count = words[word]
            old_pairs = [p for p in word.pairs() if p != pair]
            for old_pair in old_pairs:
                pairs[old_pair] -= count
            word.merge(pair, merged)
            new_pairs = [p for p in word.pairs() if p != pair]
            for new_pair in new_pairs:
                pairs[new_pair] = pairs.get(new_pair, 0) + count
                if new_pair not in pair2words:
                    pair2words[new_pair] = set()
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
        num_processes=1,
        debug=True)
    
    actual = {
        "vocab_keys": set(vocab.keys()),
        "vocab_values": set(vocab.values()),
        "merges": merges,
    }

    # Load the snapshot
    with open("tests/_snapshots/test_train_bpe_special_tokens.pkl", "rb") as f:
        expected_data = pickle.load(f)
    
    if isinstance(actual, dict):
        for key in actual: 
            assert actual[key] == expected_data[key]
