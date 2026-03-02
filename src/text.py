import re
from collections import Counter
import torch

class Tokenizer:
    def __init__(self, vocab_size=20000, max_len=96, lowercase=True):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.lowercase = lowercase

        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.cls_token = '<cls>'
        self.sep_token = '<sep>'

        self.vocab = {}
        self.inv_vocab = {}
        self.is_fitted = False

    def tokenize(self, text):
        if not isinstance(text, str):
            text = ""
        if self.lowercase:
            text = text.lower()
        # Keep punctuation by adding spaces around it
        text = re.sub(r'([.,!?()])', r' \1 ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        tokens = text.strip().split()
        return tokens

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # Keep top (vocab_size - 4) tokens
        most_common = counter.most_common(self.vocab_size - 4)

        self.vocab = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.cls_token: 2,
            self.sep_token: 3
        }

        for idx, (token, _) in enumerate(most_common):
            self.vocab[token] = idx + 4

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.is_fitted = True

    def numericalize(self, text):
        if not self.is_fitted:
            raise RuntimeError("Tokenizer is not fitted. Call build_vocab first.")

        tokens = self.tokenize(text)

        # Add special tokens
        tokens = [self.cls_token] + tokens + [self.sep_token]

        # Truncate
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
            tokens[-1] = self.sep_token

        # Map to ids
        token_ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

        # Pad
        if len(token_ids) < self.max_len:
            pad_amount = self.max_len - len(token_ids)
            token_ids.extend([self.vocab[self.pad_token]] * pad_amount)

        return torch.tensor(token_ids, dtype=torch.long)
