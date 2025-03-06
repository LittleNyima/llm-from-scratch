from abc import ABC
from collections import OrderedDict
from typing import List


class Encoding:
    def __init__(self, ids: List[int], tokens: List[str]):
        self.ids = ids
        self.tokens = tokens


class BaseTokenizer(ABC):
    def __init__(self):
        self.vocab: OrderedDict[str, int] = OrderedDict()
        self.ranks: OrderedDict[int, str] = OrderedDict()

    def token_to_id(self, token: str) -> int:
        return self.vocab[token]

    def id_to_token(self, id: int) -> str:
        return self.ranks[id]

    def encode(self, text: str) -> Encoding:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    def save_pretrained(self, path: str):
        raise NotImplementedError

    def from_pretrained(self, pretrained_tokenizer: str):
        raise NotImplementedError
