import heapq as hq
import regex as re
import time
from multiprocessing import Pool, Manager
from cs336_basics.pretokenization_example import find_chunk_boundaries


SPECIAL_TOKEN = "<|endoftext|>"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenize(
        text: str,
        special_tokens: list[str],
        words: list[bytes]) -> None:
    docs = re.split(re.escape("|".join(special_tokens)), text)
    for doc in docs:
        if doc.strip():  # Ignore empty parts
            for word in re.finditer(PAT, doc):
                if word:
                    word = word.group(0).encode("utf-8")
                    words.append(word)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 8) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input file.
    """
    # begin = time.time()

    vocab = {i:chr(i).encode('utf-8') for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    merges = []

    # Pre-tokenize the input file
    # all_words = Manager().list()
    # with Pool(num_processes) as pool:
    #     with open(input_path, "rb") as f:
    #         boundaries = find_chunk_boundaries(
    #             f, num_processes, SPECIAL_TOKEN.encode("utf-8"))
    #         for start, end in zip(boundaries[:-1], boundaries[1:]):
    #             f.seek(start)
    #             chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #             pool.apply_async(pre_tokenize, (chunk, special_tokens, all_words))      
    #     pool.close()
    #     pool.join()

    all_words = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, SPECIAL_TOKEN.encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            pre_tokenize(chunk, special_tokens, all_words)

    # print(f"Pre-tokenization took {time.time() - begin:.2f} seconds")

    # begin = time.time()

    pairs: dict[bytes, int] = {}
    words: dict[bytes, int] = {}
    pair2words: dict[bytes, list[bytes]] = {}
    for word in all_words:
        words[word] = words.get(word, 0) + 1
        if len(word) > 1:
            for i in range(len(word) - 1):
                pair = (bytes([word[i]]), bytes([word[i + 1]]))
                pairs[pair] = pairs.get(pair, 0) + 1
                if pair not in pair2words:
                    pair2words[pair] = []
                pair2words[pair].append(word)

    # print(f"Preparing data took {time.time() - begin:.2f} seconds")

    # begin = time.time()

    # Merge pairs
    counts = []
    for pair, count in pairs.items():
        hq.heappush(counts, (-count, pair))

    processed = set()
    while len(vocab) < vocab_size:
        pair = hq.heappop(counts)[1]
        merged = b"".join(pair)
        if merged in processed:
            continue
        merges.append(pair)
        vocab[len(vocab)] = merged
        processed.add(merged)
        new_pairs = {}
        for word in pair2words[pair]:
            count = words[word]
            lp = len(merged)
            if len(word) > lp:
                indices = [m.start() for m in re.finditer(merged, word)]
                for i in indices:
                    new_pair = None
                    if i > 0:
                        new_pair = (word[i - 1:i], word[i:i + lp])
                    if i + lp < len(word):
                        new_pair = (word[i:i + lp], word[i + lp:i + lp + 1])
                    if new_pair:
                        new_pairs[new_pair] = new_pairs.get(new_pair, 0) + count
                        if new_pair not in pair2words:
                            pair2words[new_pair] = []
                        pair2words[new_pair].append(word)
        for new_pair, count in new_pairs.items():
            hq.heappush(counts, (-count, new_pair))

    # print(f"Merge took {time.time() - begin:.2f} seconds")

    return (vocab, merges)


if __name__ == "__main__":
    train_bpe(
        #input_path="data/TinyStoriesV2-debug.txt",
        input_path="tests/fixtures/corpus.en",
        vocab_size=500,
        special_tokens=[SPECIAL_TOKEN],
        num_processes=8)
