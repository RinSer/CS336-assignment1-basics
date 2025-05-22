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
