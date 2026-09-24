# Tests for the encoder stack, Section 3.1
import torch

from model.encoder import Encoder, EncoderLayer


def test_encoder_layer_shape():
    layer = EncoderLayer(d_model=512, h=8, d_ff=2048, dropout=0.0)
    x = torch.randn(4, 10, 512)
    out = layer(x)
    assert out.shape == x.shape  # required for stacking layers


def test_encoder_stack_shape():
    enc = Encoder(d_model=512, h=8, d_ff=2048, dropout=0.0, n_layers=6)
    x = torch.randn(4, 10, 512)
    out = enc(x)
    assert out.shape == (4, 10, 512)


def test_encoder_has_six_independent_layers():
    enc = Encoder(d_model=32, h=4, d_ff=64, n_layers=6)
    assert len(enc.layers) == 6
    # deepcopy means weights START identical, but must be SEPARATE objects,
    # so training one layer's weights doesn't affect the others
    w0 = enc.layers[0].self_attn.w_q.weight
    w1 = enc.layers[1].self_attn.w_q.weight
    assert torch.equal(w0, w1)  # same values initially (expected with deepcopy)
    assert w0.data_ptr() != w1.data_ptr()  # but different memory, i.e. independent parameters
    with torch.no_grad():
        w0.add_(1.0)  # simulate a training update on layer 0 only
    assert not torch.equal(enc.layers[0].self_attn.w_q.weight, enc.layers[1].self_attn.w_q.weight)


def test_encoder_respects_padding_mask():
    # Two identical real tokens, differing only in a padded position, should
    # produce identical outputs at the real positions when that padding is masked out
    torch.manual_seed(0)
    enc = Encoder(d_model=16, h=2, d_ff=32, dropout=0.0, n_layers=2).eval()
    x = torch.randn(1, 3, 16)
    x2 = x.clone()
    x2[:, 2] = torch.randn(16)  # change the padded (3rd) token's content

    # mask: allow attending to positions 0,1 only (position 2 is "padding")
    mask = torch.tensor([[[1, 1, 0], [1, 1, 0], [1, 1, 0]]])  # (1, 3, 3)

    out1 = enc(x, mask)
    out2 = enc(x2, mask)
    assert torch.allclose(out1[:, :2], out2[:, :2], atol=1e-4)


def test_final_norm_applied():
    enc = Encoder(d_model=16, h=2, d_ff=32, n_layers=1)
    x = torch.randn(1, 5, 16) * 50  # large scale
    out = enc(x)
    # after final LayerNorm, per-token mean should be ~0
    assert torch.allclose(out.mean(dim=-1), torch.zeros(1, 5), atol=1e-3)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)