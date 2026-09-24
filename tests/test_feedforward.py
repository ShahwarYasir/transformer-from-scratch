# Tests for position-wise feed-forward network, Section 3.3
import torch

from model.feedforward import PositionwiseFeedForward


def test_hand_example():
    ffn = PositionwiseFeedForward(d_model=2, d_ff=3, dropout=0.0)
    with torch.no_grad():
        ffn.w1.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]))
        ffn.w1.bias.zero_()
        ffn.w2.weight.copy_(torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))
        ffn.w2.bias.zero_()

    x = torch.tensor([[1.0, -1.0]])  # one token
    out = ffn(x)
    assert torch.allclose(out[0], torch.tensor([1.0, 2.0]), atol=1e-5)


def test_shapes_match_input_for_residual():
    ffn = PositionwiseFeedForward(d_model=512, d_ff=2048)
    x = torch.randn(32, 10, 512)
    out = ffn(x)
    assert out.shape == x.shape  # required for the residual connection


def test_positions_are_independent():
    # Changing token 2 must not change the FFN output of token 1
    torch.manual_seed(0)
    ffn = PositionwiseFeedForward(d_model=16, d_ff=32, dropout=0.0).eval()
    x = torch.randn(1, 3, 16)
    x2 = x.clone()
    x2[:, 1] = torch.randn(16)  # perturb only token 2
    y1 = ffn(x)
    y2 = ffn(x2)
    assert torch.allclose(y1[:, 0], y2[:, 0], atol=1e-6)
    assert torch.allclose(y1[:, 2], y2[:, 2], atol=1e-6)
    assert not torch.allclose(y1[:, 1], y2[:, 1], atol=1e-6)


def test_relu_zeros_negatives():
    ffn = PositionwiseFeedForward(d_model=4, d_ff=8, dropout=0.0)
    with torch.no_grad():
        ffn.w1.weight.fill_(-1.0)  # forces all pre-activations negative for positive input
        ffn.w1.bias.zero_()
    x = torch.ones(1, 1, 4)
    hidden_input = ffn.relu(ffn.w1(x))
    assert torch.all(hidden_input == 0)  # ReLU killed every negative activation


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)