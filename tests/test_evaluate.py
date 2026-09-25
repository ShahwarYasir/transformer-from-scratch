"""
Tests the logic in evaluate.py (sentence translation wrapper, BLEU scoring)
using a fake tokenizer and a tiny untrained model, no real Multi30k download
or sacrebleu-quality expectations. Confirms the plumbing works; run evaluate.py
directly (e.g. in Colab) for real BLEU numbers against a trained checkpoint.
"""

import torch

from evaluation.evaluate import evaluate_bleu, translate_sentence
from model.transformer import Transformer

VOCAB_SIZE = 20


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    """Deterministic char-based stand-in tokenizer; no training/download needed."""

    def __init__(self):
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    def token_to_id(self, tok):
        return self.vocab[tok]

    def get_vocab_size(self):
        return VOCAB_SIZE

    def encode(self, text):
        return FakeEncoding([4 + (ord(c) % (VOCAB_SIZE - 4)) for c in text[:6]])

    def decode(self, ids):
        return " ".join(str(i) for i in ids)  # doesn't need to be real language, just deterministic


def build_tiny_model():
    torch.manual_seed(0)
    model = Transformer(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=16, h=2, d_ff=32, n_layers=1, dropout=0.0,
    )
    model.eval()
    return model


def test_translate_sentence_returns_string():
    model = build_tiny_model()
    tok = FakeTokenizer()
    out = translate_sentence(model, tok, "hallo welt", device="cpu", max_len=10, beam_size=2)
    assert isinstance(out, str)


def test_translate_sentence_greedy_vs_beam_both_run():
    model = build_tiny_model()
    tok = FakeTokenizer()
    greedy_out = translate_sentence(model, tok, "hallo", device="cpu", max_len=10, use_beam=False)
    beam_out = translate_sentence(model, tok, "hallo", device="cpu", max_len=10, beam_size=3, use_beam=True)
    assert isinstance(greedy_out, str)
    assert isinstance(beam_out, str)


def test_evaluate_bleu_on_fake_split_runs_end_to_end():
    model = build_tiny_model()
    tok = FakeTokenizer()
    fake_split = [
        {"de": "hallo welt", "en": "hello world"},
        {"de": "guten tag", "en": "good day"},
    ]
    bleu, hyps, refs = evaluate_bleu(model, tok, fake_split, device="cpu", max_len=10, beam_size=2)
    assert isinstance(bleu, float)
    assert bleu >= 0.0
    assert len(hyps) == 2
    assert refs == ["hello world", "good day"]


def test_evaluate_bleu_respects_limit():
    model = build_tiny_model()
    tok = FakeTokenizer()

    class FakeSplit(list):
        def select(self, indices):
            return FakeSplit(self[i] for i in indices)

    fake_split = FakeSplit([
        {"de": "a", "en": "a"}, {"de": "b", "en": "b"}, {"de": "c", "en": "c"},
    ])
    _, hyps, _ = evaluate_bleu(model, tok, fake_split, device="cpu", max_len=6, beam_size=1, limit=2)
    assert len(hyps) == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)