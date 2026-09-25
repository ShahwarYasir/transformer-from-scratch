# Label-smoothed cross-entropy loss, Section 5.4
import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    """
    Label-smoothed cross-entropy (paper, Section 5.4).

    Instead of a one-hot target, the correct class gets (1 - smoothing) and the
    remaining `smoothing` mass is spread evenly across the other (V - 1) classes.
    Padding positions (target == pad_idx) are excluded from the loss entirely.

    KLDivLoss is used (as in the paper's reference implementation) since we're
    comparing two full distributions rather than a single correct index.
    """

    def __init__(self, vocab_size, pad_idx=0, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, logits, target):
        """
        logits: (B*n, vocab_size) raw scores (pre-softmax)
        target: (B*n,) correct token indices; positions equal to pad_idx are ignored
        """
        log_probs = torch.log_softmax(logits, dim=-1)

        true_dist = torch.full_like(logits, self.smoothing / (self.vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        pad_mask = target == self.pad_idx
        true_dist[pad_mask] = 0.0  # padding positions contribute nothing

        n_tokens = (~pad_mask).sum().clamp(min=1)  # avoid div-by-zero on an all-pad batch
        return self.criterion(log_probs, true_dist) / n_tokens