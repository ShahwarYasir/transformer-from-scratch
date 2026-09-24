# Padding mask and causal (look-ahead) mask utilities, Section 3.2.3
import torch


def padding_mask(seq, pad_idx=0):
    """
    Mask out <pad> positions so attention never looks at them.

    seq: (B, n) token ids
    returns: (B, 1, n) mask, 1 = real token, 0 = padding.
             Broadcasts against the query dimension in attention: (B, 1, n) -> (B, n_q, n).
    """
    return (seq != pad_idx).unsqueeze(1).long()


def decoder_mask(tgt, pad_idx=0):
    """
    Combined causal + padding mask for decoder self-attention.

    tgt: (B, n) target token ids (shifted-right input to the decoder)
    returns: (B, n, n) mask, 1 = allowed, 0 = blocked.
    """
    B, n = tgt.shape
    pad = (tgt != pad_idx).unsqueeze(1).long()  # (B, 1, n)
    causal = torch.tril(torch.ones(n, n, device=tgt.device)).long()  # (n, n)
    return pad & causal  # broadcasts to (B, n, n)