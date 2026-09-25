import torch

from evaluation.decode import beam_search_decode, greedy_decode, length_penalty
from model.transformer import Transformer

VOCAB_SIZE = 20
BOS, EOS, PAD = 1, 2, 0


def build_tiny_model():
    torch.manual_seed(0)
    model = Transformer(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=16, h=2, d_ff=32, n_layers=1, dropout=0.0,
    )
    model.eval()
    return model


def test_length_penalty_hand_example():
    lp5 = length_penalty(5, alpha=0.6)
    lp10 = length_penalty(10, alpha=0.6)
    assert abs(lp5 - 1.3587) < 1e-3
    assert abs(lp10 - 1.7329) < 1e-3
    assert lp10 > lp5  # longer sequences get a bigger (less punishing) denominator


def test_greedy_decode_stops_at_eos_or_max_len():
    model = build_tiny_model()
    src = torch.randint(3, VOCAB_SIZE, (1, 6))
    out = greedy_decode(model, src, BOS, EOS, PAD, max_len=15)
    assert len(out) <= 14  # at most max_len - 1 generated tokens
    assert isinstance(out, list)
    assert all(isinstance(t, int) for t in out)


def test_greedy_decode_is_deterministic():
    model = build_tiny_model()
    src = torch.randint(3, VOCAB_SIZE, (1, 6))
    out1 = greedy_decode(model, src, BOS, EOS, PAD, max_len=15)
    out2 = greedy_decode(model, src, BOS, EOS, PAD, max_len=15)
    assert out1 == out2  # no randomness anywhere in the path


def test_beam_search_with_beam_size_1_matches_greedy():
    # With beam_size=1, beam search has no alternatives to consider at each step,
    # so it must reduce to exactly greedy decoding.
    model = build_tiny_model()
    src = torch.randint(3, VOCAB_SIZE, (1, 6))
    greedy_out = greedy_decode(model, src, BOS, EOS, PAD, max_len=12)
    beam_out = beam_search_decode(model, src, BOS, EOS, PAD, max_len=12, beam_size=1)
    assert greedy_out == beam_out


def test_beam_search_output_shape_and_type():
    model = build_tiny_model()
    src = torch.randint(3, VOCAB_SIZE, (1, 6))
    out = beam_search_decode(model, src, BOS, EOS, PAD, max_len=12, beam_size=4)
    assert isinstance(out, list)
    assert all(isinstance(t, int) for t in out)
    assert len(out) <= 11


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)