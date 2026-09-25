# BLEU score evaluation, Section 6
"""
Loads a trained checkpoint and reports BLEU on the Multi30k test set.

Run in Colab (needs the trained checkpoint + dataset + sacrebleu):
    !pip install datasets tokenizers pyyaml sacrebleu -q
    !python evaluate.py --checkpoint checkpoints/epoch16.pt
"""

import argparse

import torch
import yaml

from training.data import get_dataloaders, load_tokenizer, load_multi30k
from evaluation.decode import beam_search_decode, greedy_decode
from model.transformer import Transformer


def load_config(path="configs/base.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_from_checkpoint(checkpoint_path, cfg, vocab_size, device):
    model = Transformer(
        src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
        d_model=cfg["d_model"], h=cfg["num_heads"], d_ff=cfg["d_ff"],
        n_layers=cfg["num_layers"], dropout=cfg["dropout"], max_len=cfg["max_len"],
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded checkpoint from epoch {ckpt['epoch']} (val_loss {ckpt['val_loss']:.4f})")
    return model


def translate_sentence(model, tokenizer, src_text, device, max_len=128, beam_size=4, use_beam=True):
    """Encodes a raw source sentence, decodes it, and detokenizes back to text."""
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")

    src_ids = [bos_id] + tokenizer.encode(src_text).ids[: max_len - 2] + [eos_id]
    src = torch.tensor([src_ids])

    if use_beam:
        out_ids = beam_search_decode(model, src, bos_id, eos_id, pad_id, max_len, beam_size, device=device)
    else:
        out_ids = greedy_decode(model, src, bos_id, eos_id, pad_id, max_len, device=device)

    out_ids = [i for i in out_ids if i != eos_id]  # drop eos before detokenizing
    return tokenizer.decode(out_ids)


def evaluate_bleu(model, tokenizer, test_split, device, max_len=128, beam_size=4, use_beam=True, limit=None):
    """
    Translates every sentence in test_split and scores against the references
    with sacrebleu. Returns (bleu_score, hypotheses, references).
    """
    import sacrebleu

    hypotheses, references = [], []
    rows = test_split if limit is None else test_split.select(range(limit))

    for i, row in enumerate(rows):
        hyp = translate_sentence(model, tokenizer, row["de"], device, max_len, beam_size, use_beam)
        hypotheses.append(hyp)
        references.append(row["en"])
        if (i + 1) % 100 == 0:
            print(f"  translated {i + 1}/{len(rows)}")

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score, hypotheses, references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/epoch16.pt")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="only evaluate on first N test sentences")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(args.config)

    _, _, test_loader, tokenizer = get_dataloaders(
        batch_size=cfg["batch_size"], vocab_size=cfg["vocab_size"], max_len=cfg["max_len"],
    )
    _, _, test_split = load_multi30k()

    model = load_model_from_checkpoint(args.checkpoint, cfg, tokenizer.get_vocab_size(), device)
    model = load_model_from_checkpoint(args.checkpoint, cfg, tokenizer.get_vocab_size(), device)

    bleu, hyps, refs = evaluate_bleu(
        model, tokenizer, test_split, device,
        max_len=cfg["max_len"], beam_size=args.beam_size, limit=args.limit,
    )

    print(f"\nBLEU: {bleu:.2f}")
    print(f"(paper's Transformer-base EN-DE, trained on full WMT 2014: 27.3 BLEU — not directly")
    print(f" comparable, since this model trained on ~150x less data)")

    print("\nSample translations:")
    for i in range(min(3, len(hyps))):
        print(f"  DE:  {test_split[i]['de']}")
        print(f"  REF: {refs[i]}")
        print(f"  HYP: {hyps[i]}\n")


if __name__ == "__main__":
    main()