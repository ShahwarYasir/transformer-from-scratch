# Input embeddings and positional encoding, Section 3.4 and 3.5
import math

import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """
    Token embeddings, scaled by sqrt(d_model) (paper, Section 3.4).

    The scaling brings embedding magnitudes up to roughly the same scale as
    the positional encodings (which live in [-1, 1]) before the two are added.
    """

    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x: (B, n) token ids -> (B, n, d_model)
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (paper, Section 3.5):
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Not learned. Precomputed once for max_len positions and added to the
    (already-scaled) embeddings.
    """

    def __init__(self, d_model=512, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        # div_term = 10000^(2i/d_model), computed in log-space for stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) so it broadcasts over batch
        self.register_buffer("pe", pe)  # not a parameter, but moves with .to(device)

    def forward(self, x):
        # x: (B, n, d_model)
        n = x.size(1)
        x = x + self.pe[:, :n]
        return self.dropout(x)