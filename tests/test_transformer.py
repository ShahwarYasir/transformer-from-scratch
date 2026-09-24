import torch

from model.masking import decoder_mask, padding_mask
from model.transformer import Transformer


def test_padding_mask_hand_example():
    seq = torch.tensor([[5, 7, 0, 0]])  # pad_idx=0, 2 real tokens + 2 pad
    mask = padding_mask(seq, pad_idx=0)
    assert mask.shape == (1, 1, 4)
    assert torch.equal(mask[0, 0], torch.tensor([1, 1, 0, 0]))


def test_decoder_mask_combines_causal_and_padding():
    tgt = torch.tensor([[5, 7, 9, 0]])  # last token is padding
    mask = decoder_mask(tgt, pad_idx=0)
    assert mask.shape == (1, 4, 4)
    expected = torch.tensor([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 0],  # column 4 is always 0 (padding), even though row 4 "could" see itself
    ])
    assert torch.equal(mask[0], expected)


def test_transformer_output_shape():
    torch.manual_seed(0)
    model = Transformer(
        src_vocab_size=100, tgt_vocab_size=120,
        d_model=32, h=4, d_ff=64, n_layers=2,
    )
    src = torch.randint(1, 100, (2, 7))   # (B=2, n_src=7)
    tgt = torch.randint(1, 120, (2, 5))   # (B=2, n_tgt=5)
    logits = model(src, tgt)
    assert logits.shape == (2, 5, 120)  # (B, n_tgt, tgt_vocab_size)


def test_transformer_weight_sharing():
    model = Transformer(src_vocab_size=50, tgt_vocab_size=50, d_model=16, h=2, d_ff=32, n_layers=1)
    assert model.generator.weight.data_ptr() == model.tgt_embed.lut.weight.data_ptr()


def test_transformer_respects_masks():
    torch.manual_seed(0)
    model = Transformer(src_vocab_size=30, tgt_vocab_size=30, d_model=16, h=2, d_ff=32, n_layers=2).eval()
    src = torch.tensor([[3, 4, 5, 0]])  # last token padded
    tgt = torch.tensor([[3, 4, 0, 0]])  # last 2 padded

    src_mask = padding_mask(src, pad_idx=0)
    tgt_mask = decoder_mask(tgt, pad_idx=0)

    logits1 = model(src, tgt, src_mask, tgt_mask)

    # Change the padded src token; real-token outputs at position 1 (first target token)
    # should be unaffected since it can only attend to non-pad source tokens
    src2 = src.clone()
    src2[:, 3] = 7
    logits2 = model(src2, tgt, src_mask, tgt_mask)

    assert torch.allclose(logits1[:, 0], logits2[:, 0], atol=1e-4)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)

