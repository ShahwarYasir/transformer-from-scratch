# Multi-head attention mechanism, Section 3.2
import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V   (paper, Section 3.2.1)

    q: (..., n_q, d_k)
    k: (..., n_k, d_k)
    v: (..., n_k, d_v)
    mask: broadcastable to (..., n_q, n_k); 1 = keep, 0 = block
    returns: output (..., n_q, d_v), weights (..., n_q, n_k)
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (..., n_q, n_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)  # each row sums to 1
    return weights @ v, weights


def causal_mask(n, device=None):
    """Lower-triangular mask: position i may attend to positions <= i."""
    return torch.tril(torch.ones(n, n, device=device))  # (n, n)


class MultiHeadAttention(nn.Module):
    """Multi-head attention (paper, Section 3.2.2). Base model: d_model=512, h=8."""

    def __init__(self, d_model=512, h=8):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.h = h
        self.d_k = d_model // h  # 512 / 8 = 64
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn = None  # saved weights, handy for heatmaps later

    def _split_heads(self, x):
        # (B, n, d_model) -> (B, h, n, d_k)
        B, n, _ = x.shape
        return x.view(B, n, self.h, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        # q: (B, n_q, d_model); k, v: (B, n_k, d_model)
        B, n_q, _ = q.shape

        q = self._split_heads(self.w_q(q))  # (B, h, n_q, d_k)
        k = self._split_heads(self.w_k(k))  # (B, h, n_k, d_k)
        v = self._split_heads(self.w_v(v))  # (B, h, n_k, d_k)

        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, n_q, n_k) -> (B, 1, n_q, n_k), shared across heads

        out, self.attn = scaled_dot_product_attention(q, k, v, mask)  # (B, h, n_q, d_k)

        out = out.transpose(1, 2).contiguous().view(B, n_q, self.h * self.d_k)  # concat heads
        return self.w_o(out)  # (B, n_q, d_model)