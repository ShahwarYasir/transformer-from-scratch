# Decoder stack with masked self-attention and cross-attention, Section 3.1
import copy

import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feedforward import PositionwiseFeedForward
from model.sublayer import SublayerConnection


class DecoderLayer(nn.Module):
    """
    One decoder layer (paper, Section 3.1): masked self-attention, then
    cross-attention over the encoder output, then feed-forward. Each wrapped
    in a residual + LayerNorm sublayer connection.
    """

    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.cross_attn = MultiHeadAttention(d_model, h)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # x: (B, n_tgt, d_model) - target sequence so far
        # memory: (B, n_src, d_model) - encoder output
        # tgt_mask: causal mask (+ target padding, if any) for self-attention
        # src_mask: padding mask over the source, for cross-attention
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer2(x, lambda x: self.cross_attn(x, memory, memory, src_mask))
        x = self.sublayer3(x, self.feed_forward)
        return x


class Decoder(nn.Module):
    """Stack of N identical decoder layers, with a final LayerNorm (paper, Section 3.1)."""

    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1, n_layers=6):
        super().__init__()
        layer = DecoderLayer(d_model, h, d_ff, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)