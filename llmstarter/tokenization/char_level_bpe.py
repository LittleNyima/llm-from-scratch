import argparse
import json
import os
from collections import Counter, OrderedDict, deque
from typing import List, Optional, Tuple

from tqdm import tqdm

from llmstarter.tokenization import BaseTokenizer, Encoding
from llmstarter.tokenization.visualize import visualize_encoding


# This implementation references https://sebastianraschka.com/blog/2025/bpe-from-scratch.html
class BPETokenizer(BaseTokenizer):
    def __init__(self):
        super(BPETokenizer, self).__init__()
        # A dictionary that maps a pair of token IDs to a new token ID
        self.merge_rules: OrderedDict[Tuple[int, int], int] = OrderedDict()

    def encode(self, text: str) -> Encoding:
        # Tokenize the text into token IDs
        token_ids = [self.token_to_id(char) for char in text]
        # Merge tokens until no more merges are possible
        can_merge = True
        while can_merge and len(token_ids) > 1:
            can_merge = False
            new_tokens = []
            i = 0
            # Iterate over the token IDs and merge tokens if possible
            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i + 1])
                if pair in self.merge_rules:
                    # Merge the pair into a new token
                    merged_token_id = self.merge_rules[pair]
                    new_tokens.append(merged_token_id)
                    i += 2
                    can_merge = True
                else:
                    # Append the token if it cannot be merged
                    new_tokens.append(token_ids[i])
                    i += 1
            # Append the last token if there are any remaining tokens
            if i < len(token_ids):
                new_tokens.append(token_ids[i])
            token_ids = new_tokens

        return Encoding(
            ids=token_ids,
            tokens=[self.id_to_token(token_id) for token_id in token_ids],
        )

    def decode(self, ids: List[int]) -> str:
        return "".join([self.id_to_token(id) for id in ids])

    def save_pretrained(self, path: str):
        data = {
            "vocab": self.vocab,
            "merge_rules": list(self.merge_rules.items()),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def from_pretrained(self, pretrained_tokenizer: str):
        with open(pretrained_tokenizer, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = OrderedDict(data["vocab"])
        self.ranks = OrderedDict({index: token for token, index in self.vocab.items()})
        self.merge_rules = OrderedDict()
        for (first_id, second_id), new_id in data["merge_rules"]:
            self.merge_rules[(first_id, second_id)] = new_id


class BPETokenizerTrainer:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size: int = vocab_size

    def train(self, text: str, bpe_tokenizer: BPETokenizer):
        # Initialize the vocabulary with all ASCII characters and unique characters in the text
        initial_vocab: list[str] = [chr(i) for i in range(256)]
        initial_vocab.extend(char for char in sorted(set(text)) if char not in initial_vocab)

        # Create a dictionary to map tokens to their indices and vice versa
        bpe_tokenizer.vocab = OrderedDict({token: index for index, token in enumerate(initial_vocab)})
        bpe_tokenizer.ranks = OrderedDict(dict(enumerate(initial_vocab)))

        # Tokenize the text into token IDs
        token_ids = [bpe_tokenizer.token_to_id(char) for char in text]

        # Train the BPE tokenizer
        for new_id in tqdm(range(len(initial_vocab), self.vocab_size), desc="Training BPE tokenizer"):
            # Get the most frequent pair to merge
            pair = self.get_most_frequent_pair(token_ids)
            if pair is None:  # No more pairs to merge
                break
            # Merge the pair into a new token
            token_ids = self.merge_tokens(token_ids, pair, new_id)

            # Update the vocabulary and ranks
            first_id, second_id = pair
            bpe_tokenizer.merge_rules[(first_id, second_id)] = new_id
            new_token = bpe_tokenizer.id_to_token(first_id) + bpe_tokenizer.id_to_token(second_id)
            bpe_tokenizer.vocab[new_token] = new_id
            bpe_tokenizer.ranks[new_id] = new_token

    def get_most_frequent_pair(self, token_ids: List[int]) -> Optional[Tuple[int, int]]:
        counter = Counter(zip(token_ids[:-1], token_ids[1:], strict=True))
        if len(counter) == 0:
            return None
        return max(counter, key=counter.get)

    def merge_tokens(self, token_ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        dq = deque(token_ids)
        new_token_ids = []
        first_id, second_id = pair
        while len(dq) > 0:
            leftmost = dq.popleft()
            if len(dq) > 0 and leftmost == first_id and dq[0] == second_id:
                new_token_ids.append(new_id)
                dq.popleft()
            else:
                new_token_ids.append(leftmost)
        return new_token_ids


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", action="store_true")
    argparser.add_argument("--vocab-size", type=int, default=10000)
    argparser.add_argument("--corpus", type=str, default="downloads/data/a-tale-of-two-cities.txt")
    argparser.add_argument("--save-path", type=str, default="downloads/models/bpe_tokenizer.json")
    args = argparser.parse_args()

    if args.train:
        bpe_tokenizer = BPETokenizer()
        trainer = BPETokenizerTrainer(vocab_size=args.vocab_size)
        with open(args.corpus, "r") as f:
            text = f.read()
        trainer.train(text, bpe_tokenizer)
        bpe_tokenizer.save_pretrained(args.save_path)
    else:
        bpe_tokenizer = BPETokenizer()
        bpe_tokenizer.from_pretrained(args.save_path)
        test_text = (
            "Tokenization is the process of creating a digital representation of a real thing. "
            "Tokenization can also be used to protect sensitive data or to efficiently process "
            "large amounts of data."
        )
        encoding = bpe_tokenizer.encode(test_text)
        visualize_encoding(encoding)
        print()
        print(bpe_tokenizer.decode(encoding.ids) == test_text)
