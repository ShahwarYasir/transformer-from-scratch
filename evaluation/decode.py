# Greedy and beam-search decoding, Section 3.1
import torch
import torch.nn.functional as F

from model.masking import decoder_mask, padding_mask


@torch.no_grad()
def greedy_decode(model, src, bos_idx, eos_idx, pad_idx, max_len, device="cpu"):
    """
    Generate one token at a time, always picking the single highest-probability
    next token. Fast, but can't recover from an early suboptimal choice.

    src: (1, n_src) a single source sequence (batch size 1)
    returns: list[int], the generated token ids (excluding <bos>)
    """
    model.eval()
    src = src.to(device)
    src_mask = padding_mask(src, pad_idx=pad_idx)
    memory = model.encode(src, src_mask)

    ys = torch.full((1, 1), bos_idx, dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        tgt_mask = decoder_mask(ys, pad_idx=pad_idx)
        out = model.decode(ys, memory, src_mask, tgt_mask)
        logits = model.generator(out[:, -1])  # (1, vocab_size), last position only
        next_token = logits.argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        if next_token.item() == eos_idx:
            break

    return ys[0, 1:].tolist()  # drop the leading <bos>


def length_penalty(length, alpha=0.6):
    """Paper's beam search length penalty (Section 6.1): ((5 + len) / 6) ^ alpha."""
    return ((5 + length) / 6) ** alpha


@torch.no_grad()
def beam_search_decode(model, src, bos_idx, eos_idx, pad_idx, max_len, beam_size=4, alpha=0.6, device="cpu"):
    """
    Keeps the top `beam_size` partial hypotheses alive at each step, scored by
    length-normalized log-probability. Paper's setup (Section 6.1): beam=4, alpha=0.6.

    src: (1, n_src) a single source sequence (batch size 1)
    returns: list[int], the best completed hypothesis's token ids (excluding <bos>)
    """
    model.eval()
    src = src.to(device)
    src_mask = padding_mask(src, pad_idx=pad_idx)
    memory = model.encode(src, src_mask)

    # Each beam: (token_ids tensor (1, n), cumulative log-prob, finished flag)
    beams = [(torch.full((1, 1), bos_idx, dtype=torch.long, device=device), 0.0, False)]
    completed = []

    for _ in range(max_len - 1):
        candidates = []
        for ys, score, finished in beams:
            if finished:
                candidates.append((ys, score, True))
                continue

            tgt_mask = decoder_mask(ys, pad_idx=pad_idx)
            out = model.decode(ys, memory, src_mask, tgt_mask)
            log_probs = F.log_softmax(model.generator(out[:, -1]), dim=-1)  # (1, vocab_size)

            topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)
            for k in range(beam_size):
                next_id = topk_ids[0, k].item()
                next_score = score + topk_log_probs[0, k].item()
                new_ys = torch.cat([ys, topk_ids[0, k].view(1, 1)], dim=1)
                candidates.append((new_ys, next_score, next_id == eos_idx))

        # Keep only the best `beam_size` candidates, ranked by length-normalized score
        candidates.sort(key=lambda c: c[1] / length_penalty(c[0].size(1), alpha), reverse=True)
        beams = candidates[:beam_size]

        for ys, score, finished in beams:
            if finished:
                completed.append((ys, score))

        if all(finished for _, _, finished in beams):
            break

    if not completed:
        completed = [(ys, score) for ys, score, _ in beams]

    best_ys, _ = max(completed, key=lambda c: c[1] / length_penalty(c[0].size(1), alpha))
    return best_ys[0, 1:].tolist()  # drop the leading <bos>