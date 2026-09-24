# Sublayer connection with residual connection and layer norm, Section 3.1
import torch.nn as nn


class SublayerConnection(nn.Module):
    """
    Residual connection around a sub-layer, followed by LayerNorm
    (paper, Section 3.1, post-norm as written in the paper):

        output = LayerNorm(x + Dropout(Sublayer(x)))

    `sublayer` is passed in as a callable (e.g. a lambda wrapping self-attention
    or the feed-forward network) so this class stays agnostic to what it wraps.
    """

    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # x: (B, n, d_model); sublayer(x) must return the same shape
        return self.norm(x + self.dropout(sublayer(x)))