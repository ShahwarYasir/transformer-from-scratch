import torch
import torch.nn as nn

from model.sublayer import SublayerConnection


def test_layernorm_hand_example():
    ln = nn.LayerNorm(4)
    with torch.no_grad():
        ln.weight.fill_(1.0)
        ln.bias.fill_(0.0)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = ln(x)
    expected = torch.tensor([-1.3416, -0.4472, 0.4472, 1.3416])
    assert torch.allclose(out[0], expected, atol=1e-3)


def test_shape_preserved():
    sub = SublayerConnection(d_model=512, dropout=0.0)
    x = torch.randn(8, 10, 512)
    identity = lambda t: t  # trivial sublayer
    out = sub(x, identity)
    assert out.shape == x.shape


def test_residual_actually_adds_input():
    # With a zero sublayer, output should just be LayerNorm(x), not LayerNorm(0)
    torch.manual_seed(0)
    sub = SublayerConnection(d_model=16, dropout=0.0)
    x = torch.randn(2, 5, 16)
    zero_sublayer = lambda t: torch.zeros_like(t)
    out = sub(x, zero_sublayer)
    assert torch.allclose(out, sub.norm(x), atol=1e-5)


def test_output_is_normalized():
    # Regardless of input scale, LayerNorm output should have ~zero mean per token
    sub = SublayerConnection(d_model=32, dropout=0.0)
    x = torch.randn(4, 6, 32) * 100  # large scale input
    identity = lambda t: torch.zeros_like(t)
    out = sub(x, identity)
    means = out.mean(dim=-1)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-3)


def test_wraps_a_real_attention_like_sublayer():
    # Simulate wrapping something like multi-head attention: same d_model in and out
    sub = SublayerConnection(d_model=8, dropout=0.0)
    linear = nn.Linear(8, 8)
    x = torch.randn(2, 3, 8)
    out = sub(x, linear)
    assert out.shape == x.shape


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)