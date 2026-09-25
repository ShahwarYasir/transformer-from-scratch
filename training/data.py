# Dataset loading and batching utilities, Section 5.1
"""
Data pipeline for Multi30k (German -> English), paper Section 5.1.

Requires (install once, e.g. in Colab):
    pip install datasets tokenizers

Run this file directly to train the tokenizer and sanity-check a batch:
    python data.py
"""

import os

import torch
from torch.utils.data import DataLoader, Dataset

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]
PAD_IDX = 0  # position of PAD in SPECIAL_TOKENS, must match masking.py's default pad_idx=0

TOKENIZER_PATH = "bpe_tokenizer.json"
VOCAB_SIZE = 8000  # smaller than the paper's 37,000 since Multi30k is much smaller than WMT


def load_multi30k():
    """Returns (train, val, test) splits, each a list of {'de': str, 'en': str}."""
    from datasets import load_dataset

    ds = load_dataset("bentrevett/multi30k")
    return ds["train"], ds["validation"], ds["test"]


def train_tokenizer(train_split, vocab_size=VOCAB_SIZE, save_path=TOKENIZER_PATH):
    """
    Trains one shared BPE tokenizer on both German and English text
    (paper Section 5.1: shared source-target vocabulary via byte-pair encoding).
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)

    def text_iterator():
        for row in train_split:
            yield row["de"]
            yield row["en"]

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.save(save_path)
    return tokenizer


def load_tokenizer(path=TOKENIZER_PATH):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


class Multi30kDataset(Dataset):
    """Tokenizes and encodes a Multi30k split into (src_ids, tgt_ids) pairs."""

    def __init__(self, split, tokenizer, max_len=128):
        self.split = split
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bos_id = tokenizer.token_to_id(BOS)
        self.eos_id = tokenizer.token_to_id(EOS)

    def __len__(self):
        return len(self.split)

    def _encode(self, text):
        ids = self.tokenizer.encode(text).ids[: self.max_len - 2]  # room for bos/eos
        return [self.bos_id] + ids + [self.eos_id]

    def __getitem__(self, idx):
        row = self.split[idx]
        src_ids = self._encode(row["de"])
        tgt_ids = self._encode(row["en"])
        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(batch, pad_idx=PAD_IDX):
    """Pads a batch of variable-length (src, tgt) pairs to the same length."""
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]
    max_src, max_tgt = max(src_lens), max(tgt_lens)

    src_padded = torch.full((len(batch), max_src), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), pad_idx, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, : len(s)] = s
        tgt_padded[i, : len(t)] = t

    return src_padded, tgt_padded


def get_dataloaders(batch_size=32, vocab_size=VOCAB_SIZE, max_len=128):
    """Full pipeline: load data, train (or load cached) tokenizer, build dataloaders."""
    train_split, val_split, test_split = load_multi30k()

    if os.path.exists(TOKENIZER_PATH):
        tokenizer = load_tokenizer()
    else:
        tokenizer = train_tokenizer(train_split, vocab_size)

    train_ds = Multi30kDataset(train_split, tokenizer, max_len)
    val_ds = Multi30kDataset(val_split, tokenizer, max_len)
    test_ds = Multi30kDataset(test_split, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, tokenizer


if __name__ == "__main__":
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(batch_size=8)
    src, tgt = next(iter(train_loader))
    print("src shape:", src.shape, "tgt shape:", tgt.shape)
    print("vocab size:", tokenizer.get_vocab_size())
    print("sample src ids:", src[0].tolist())
    print("decoded:", tokenizer.decode(src[0].tolist()))