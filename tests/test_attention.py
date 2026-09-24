# Tests for multi-head attention, Section 3.2
import torch

from model.attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention


def test_hand_example():
    # The 3-token example worked by hand: Q = K = V = X, d_k = 2
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    out, w = scaled_dot_product_attention(x, x, x)
    assert torch.allclose(w[0], torch.tensor([0.401, 0.198, 0.401]), atol=1e-3)
    assert torch.allclose(out[0], torch.tensor([0.802, 0.599]), atol=1e-3)
    assert torch.allclose(out[2], torch.tensor([0.752, 0.752]), atol=1e-3)
    assert torch.allclose(w.sum(-1), torch.ones(3))  # rows sum to 1


def test_causal_mask_on_hand_example():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    out, w = scaled_dot_product_attention(x, x, x, mask=causal_mask(3))
    assert torch.allclose(w[0], torch.tensor([1.0, 0.0, 0.0]))  # token 1 sees only itself
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0]))
    assert torch.all(torch.triu(w, diagonal=1) == 0)  # nothing above the diagonal


def test_multihead_shapes():
    mha = MultiHeadAttention(d_model=512, h=8)
    x = torch.randn(32, 10, 512)
    out = mha(x, x, x)
    assert out.shape == (32, 10, 512)  # same in, same out (needed for residuals)
    assert mha.attn.shape == (32, 8, 10, 10)  # (B, heads, n, n)


def test_cross_attention_shapes():
    mha = MultiHeadAttention(d_model=512, h=8)
    dec = torch.randn(4, 7, 512)  # target side, n_q = 7
    enc = torch.randn(4, 12, 512)  # source side, n_k = 12
    out = mha(dec, enc, enc)
    assert out.shape == (4, 7, 512)  # follows the query length
    assert mha.attn.shape == (4, 8, 7, 12)


def test_multihead_is_causal():
    # Changing FUTURE tokens must not change earlier outputs
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=64, h=4).eval()
    x = torch.randn(1, 6, 64)
    x2 = x.clone()
    x2[:, 4:] = torch.randn(1, 2, 64)  # change the last 2 tokens
    m = causal_mask(6)
    y1 = mha(x, x, x, mask=m)
    y2 = mha(x2, x2, x2, mask=m)
    assert torch.allclose(y1[:, :4], y2[:, :4], atol=1e-5)  # first 4 unchanged
    assert not torch.allclose(y1[:, 4:], y2[:, 4:], atol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)