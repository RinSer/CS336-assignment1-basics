import regex as re
from typing import Iterable, Iterator
import json
from cs336_basics.tokenizer_word import Word, PAT


class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab: dict[int, bytes] = vocab
        self.mapping: dict[bytes, int] = {v:k for k, v in vocab.items()}
        self.merges: dict[tuple[bytes, bytes], int] = {
            merge: idx for idx, merge in enumerate(merges)
        }
        self.special_tokens: list[str] | None = special_tokens
        if special_tokens:
            for special_token in self.special_tokens:
                special_token = special_token.encode("utf-8")
                if special_token not in self.mapping:
                    idx = len(self.vocab)
                    self.vocab[idx] = special_token
                    self.mapping[special_token] = idx

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath) as f:
            vocab = json.load(f)
        merges = []
        with open(merges_filepath) as m:
            for line in m:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    merges.append(tuple(cleaned_line.split(" ")))
        return Tokenizer(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        res = []
        if self.special_tokens:
            resplit = "|".join([
                "(" + re.escape(s) + ")" 
                for s in sorted(self.special_tokens, key=len, reverse=True)
            ])
            docs = re.split(resplit, text)
        else:
            docs = [text]
        for doc in docs:
            if not doc:
                continue
            if self.special_tokens and doc in self.special_tokens:
                res.append(self.mapping[doc.encode("utf-8")])
                continue
            for word in re.finditer(PAT, doc):
                if word:
                    word: str = word.group(0).replace("\r", "")
                    if not word:
                        continue
                    word = Word(word.encode("utf-8"))
                    if len(word) > 1:
                        pairs = [p for p in word.pairs() if p in self.merges]
                        while len(word.tokens) > 1 and len(pairs) > 0:
                            best = min(pairs, key=lambda p: self.merges[p])
                            word.merge(best, b"".join(best))
                            pairs = [p for p in word.pairs() if p in self.merges]
                    res.extend([self.mapping[token] for token in word.tokens])
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        chunk = ""
        for part in self._load_until_special(iterable):
            chunk += part
        for encoded in self.encode(chunk):
            yield encoded

    def decode(self, ids: list[int]) -> str:
        return (b"".join([
            self.vocab[token] for token in ids
        ])).decode("utf-8", errors="replace")

    def _load_until_special(self, iterable: Iterable[str]) -> Iterator[str]:
        """
        Yields chunks from iterable, each ending with one of the special tokens.
        """
        buffer: str = ""
        for chunk in iterable:
            buffer += chunk
            if self.special_tokens:
                for token in self.special_tokens:
                    if buffer.endswith(token):
                        yield buffer
                        buffer = ""
                        break
        if buffer:
            yield buffer
