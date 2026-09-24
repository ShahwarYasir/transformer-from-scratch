# Tests for embeddings and positional encoding, Section 3.4 and 3.5
import math

import torch

from model.embeddings import Embeddings, PositionalEncoding


def test_embedding_scaling():
    torch.manual_seed(0)
    emb = Embeddings(vocab_size=10, d_model=512)
    x = torch.tensor([[1, 2, 3]])
    out = emb(x)
    raw = emb.lut(x)
    assert torch.allclose(out, raw * math.sqrt(512), atol=1e-5)
    assert out.shape == (1, 3, 512)


def test_positional_encoding_hand_example():
    pe = PositionalEncoding(d_model=4, max_len=10, dropout=0.0)
    # position 0 is always [sin0, cos0, sin0, cos0] = [0, 1, 0, 1]
    assert torch.allclose(pe.pe[0, 0], torch.tensor([0.0, 1.0, 0.0, 1.0]), atol=1e-5)
    # position 1, computed by hand: [sin(1), cos(1), sin(0.01), cos(0.01)]
    expected = torch.tensor([0.8415, 0.5403, 0.0100, 0.99995])
    assert torch.allclose(pe.pe[0, 1], expected, atol=1e-3)


def test_positional_encoding_adds_to_input():
    pe = PositionalEncoding(d_model=8, max_len=20, dropout=0.0)
    x = torch.zeros(2, 5, 8)  # zeros so output == the PE table itself
    out = pe(x)
    assert torch.allclose(out, pe.pe[:, :5].expand(2, -1, -1))
    assert out.shape == x.shape  # unchanged shape, needed downstream


def test_different_positions_get_different_encodings():
    pe = PositionalEncoding(d_model=16, max_len=50, dropout=0.0)
    assert not torch.allclose(pe.pe[0, 0], pe.pe[0, 1])
    assert not torch.allclose(pe.pe[0, 5], pe.pe[0, 6])


def test_pe_is_not_a_trainable_parameter():
    pe = PositionalEncoding(d_model=16, max_len=50)
    param_names = [name for name, _ in pe.named_parameters()]
    assert "pe" not in param_names  # registered as buffer, not nn.Parameter
    buffer_names = [name for name, _ in pe.named_buffers()]
    assert "pe" in buffer_names


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)