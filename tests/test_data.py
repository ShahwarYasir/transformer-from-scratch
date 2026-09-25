"""
These tests avoid any real download: no HuggingFace `datasets` or `tokenizers`
network calls. They check the parts of the pipeline that are pure logic
(padding/collation, bos/eos wrapping) using small fake stand-ins.

The full pipeline (load_multi30k, train_tokenizer, get_dataloaders) needs
internet access and the `datasets` + `tokenizers` packages -- run data.py
directly (e.g. in Colab) to exercise those.
"""

import torch

from training.data import PAD_IDX, Multi30kDataset, collate_fn


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    """Minimal stand-in for a trained BPE tokenizer, no download needed."""

    def __init__(self):
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    def token_to_id(self, tok):
        return self.vocab[tok]

    def encode(self, text):
        # toy "tokenization": one id per character, offset past special tokens
        return FakeEncoding([4 + (ord(c) % 20) for c in text])


def test_dataset_wraps_with_bos_eos():
    split = [{"de": "hallo", "en": "hi"}]
    tok = FakeTokenizer()
    ds = Multi30kDataset(split, tok, max_len=128)
    src_ids, tgt_ids = ds[0]
    assert src_ids[0].item() == tok.token_to_id("<bos>")
    assert src_ids[-1].item() == tok.token_to_id("<eos>")
    assert tgt_ids[0].item() == tok.token_to_id("<bos>")
    assert tgt_ids[-1].item() == tok.token_to_id("<eos>")


def test_dataset_respects_max_len():
    split = [{"de": "a" * 200, "en": "b"}]
    tok = FakeTokenizer()
    ds = Multi30kDataset(split, tok, max_len=10)
    src_ids, _ = ds[0]
    assert len(src_ids) == 10  # bos + 8 content tokens + eos, truncated


def test_collate_pads_to_max_length_in_batch():
    batch = [
        (torch.tensor([1, 5, 6, 2]), torch.tensor([1, 9, 2])),
        (torch.tensor([1, 7, 2]), torch.tensor([1, 8, 8, 8, 2])),
    ]
    src, tgt = collate_fn(batch, pad_idx=PAD_IDX)
    assert src.shape == (2, 4)  # longest src in batch has 4 tokens
    assert tgt.shape == (2, 5)  # longest tgt in batch has 5 tokens
    # shorter sequence padded with PAD_IDX at the end
    assert torch.equal(src[1], torch.tensor([1, 7, 2, PAD_IDX]))
    assert torch.equal(tgt[0], torch.tensor([1, 9, 2, PAD_IDX, PAD_IDX]))


def test_collate_preserves_real_tokens():
    batch = [
        (torch.tensor([1, 5, 6, 2]), torch.tensor([1, 9, 2])),
    ]
    src, tgt = collate_fn(batch, pad_idx=PAD_IDX)
    assert torch.equal(src[0], torch.tensor([1, 5, 6, 2]))  # no padding needed, unchanged
    assert torch.equal(tgt[0], torch.tensor([1, 9, 2]))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)