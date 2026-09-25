"""
Visualizes cross-attention: for each generated output word, which source
words did the decoder attend to? Run in Colab, after training (needs a
checkpoint, the tokenizer, and matplotlib).

    !python -m evaluation.attention_viz --checkpoint checkpoints/epoch16.pt \
        --sentence "Ein Boston Terrier läuft über saftig-grünes Gras."
"""

import argparse

import matplotlib.pyplot as plt
import torch

from evaluation.evaluate import load_config, load_model_from_checkpoint
from model.masking import decoder_mask, padding_mask
from training.data import load_tokenizer


@torch.no_grad()
def translate_with_attention(model, tokenizer, src_text, device, max_len=50, layer=-1, head=None):
    """
    Greedy-decodes src_text while recording the last decoder layer's
    cross-attention weights at every step, so we get a full
    (generated_len, src_len) attention matrix at the end.

    head=None averages over all heads; pass an int to inspect one head only.
    """
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")

    src_ids = [bos_id] + tokenizer.encode(src_text).ids + [eos_id]
    src = torch.tensor([src_ids]).to(device)
    src_mask = padding_mask(src, pad_idx=pad_id)
    memory = model.encode(src, src_mask)

    ys = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
    attn_rows = []
    for _ in range(max_len - 1):
        tgt_mask = decoder_mask(ys, pad_idx=pad_id)
        out = model.decode(ys, memory, src_mask, tgt_mask)

        # cross_attn.attn shape: (1, h, n_tgt_so_far, n_src); we want the LAST query row
        cross_attn = model.decoder.layers[layer].cross_attn.attn
        last_row = cross_attn[0, :, -1, :] if head is None else cross_attn[0, head, -1, :]
        if last_row.dim() == 2:  # averaging over heads
            last_row = last_row.mean(dim=0)
        attn_rows.append(last_row.cpu())

        next_token = model.generator(out[:, -1]).argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        if next_token.item() == eos_id:
            break

    src_tokens = [tokenizer.id_to_token(i) for i in src_ids]
    tgt_tokens = [tokenizer.id_to_token(t) for t in ys[0, 1:].tolist()]
    attn_matrix = torch.stack(attn_rows)  # (n_tgt, n_src)
    return src_tokens, tgt_tokens, attn_matrix


def plot_attention(src_tokens, tgt_tokens, attn_matrix, save_path=None):
    fig, ax = plt.subplots(figsize=(max(6, len(src_tokens) * 0.6), max(4, len(tgt_tokens) * 0.5)))
    im = ax.imshow(attn_matrix.numpy(), cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens)
    ax.set_xlabel("Source (German)")
    ax.set_ylabel("Generated (English)")
    ax.set_title("Decoder cross-attention: which source words each output word attended to")

    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"saved: {save_path}")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/epoch16.pt")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--out", default="attention_heatmap.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(args.config)
    tokenizer = load_tokenizer()
    model = load_model_from_checkpoint(args.checkpoint, cfg, tokenizer.get_vocab_size(), device)

    src_tokens, tgt_tokens, attn = translate_with_attention(model, tokenizer, args.sentence, device)
    print("source: ", " ".join(src_tokens))
    print("output: ", " ".join(tgt_tokens))
    plot_attention(src_tokens, tgt_tokens, attn, save_path=args.out)


if __name__ == "__main__":
    main()