# Encoder stack, Section 3.1
import copy

import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feedforward import PositionwiseFeedForward
from model.sublayer import SublayerConnection


class EncoderLayer(nn.Module):
    """
    One encoder layer (paper, Section 3.1): self-attention, then feed-forward,
    each wrapped in a residual + LayerNorm sublayer connection.
    """

    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, mask=None):
        # x: (B, n, d_model)
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer2(x, self.feed_forward)
        return x


class Encoder(nn.Module):
    """Stack of N identical encoder layers, with a final LayerNorm (paper, Section 3.1)."""

    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1, n_layers=6):
        super().__init__()
        layer = EncoderLayer(d_model, h, d_ff, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)