# Tests for the decoder stack, Section 3.1
import torch

from model.attention import causal_mask
from model.decoder import Decoder, DecoderLayer


def test_decoder_layer_shape():
    layer = DecoderLayer(d_model=512, h=8, d_ff=2048, dropout=0.0)
    x = torch.randn(4, 7, 512)       # target: 7 tokens
    memory = torch.randn(4, 12, 512)  # source: 12 tokens
    out = layer(x, memory)
    assert out.shape == x.shape  # follows target length, not source length


def test_decoder_stack_shape():
    dec = Decoder(d_model=512, h=8, d_ff=2048, dropout=0.0, n_layers=6)
    x = torch.randn(4, 7, 512)
    memory = torch.randn(4, 12, 512)
    out = dec(x, memory)
    assert out.shape == (4, 7, 512)


def test_decoder_respects_causal_mask():
    # Changing a FUTURE target token must not change earlier decoder outputs
    torch.manual_seed(0)
    dec = Decoder(d_model=16, h=2, d_ff=32, dropout=0.0, n_layers=2).eval()
    x = torch.randn(1, 5, 16)
    memory = torch.randn(1, 8, 16)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(1, 2, 16)  # change the last 2 target tokens

    mask = causal_mask(5).unsqueeze(0)  # (1, 5, 5)
    out1 = dec(x, memory, tgt_mask=mask)
    out2 = dec(x2, memory, tgt_mask=mask)
    assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-4)


def test_decoder_uses_encoder_memory():
    # Changing the encoder output should change the decoder output
    # (proves cross-attention is actually wired to `memory`, not ignored)
    torch.manual_seed(0)
    dec = Decoder(d_model=16, h=2, d_ff=32, dropout=0.0, n_layers=2).eval()
    x = torch.randn(1, 4, 16)
    memory1 = torch.randn(1, 6, 16)
    memory2 = torch.randn(1, 6, 16)

    out1 = dec(x, memory1)
    out2 = dec(x, memory2)
    assert not torch.allclose(out1, out2, atol=1e-4)


def test_decoder_handles_different_src_tgt_lengths():
    dec = Decoder(d_model=32, h=4, d_ff=64, n_layers=2)
    x = torch.randn(2, 3, 32)       # short target
    memory = torch.randn(2, 20, 32)  # long source
    out = dec(x, memory)
    assert out.shape == (2, 3, 32)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)