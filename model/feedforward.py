# Position-wise feed-forward network, Section 3.3
import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, x W1 + b1) W2 + b2   (paper, Section 3.3)

    Applied identically and independently to every position in the sequence.
    Base model: d_model=512, d_ff=2048.
    """

    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, n, d_model) -> (B, n, d_ff) -> (B, n, d_model)
        return self.w2(self.dropout(self.relu(self.w1(x))))