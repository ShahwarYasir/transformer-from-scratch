# Training loop and checkpoint saving, Section 5
"""
Training loop for the Transformer on Multi30k.

Run in Colab (needs GPU + the dataset from data.py):
    !pip install datasets tokenizers pyyaml -q
    !python train.py
"""

import os
import time

import torch
import yaml

from training.data import PAD_IDX, get_dataloaders
from training.loss import LabelSmoothingLoss
from model.masking import decoder_mask, padding_mask
from training.optimizer import NoamScheduler, build_optimizer
from model.transformer import Transformer


def load_config(path="configs/base.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run_epoch(model, dataloader, criterion, scheduler, device, train=True):
    """One pass over the data. If train=True, updates weights; otherwise just measures loss."""
    model.train() if train else model.eval()
    total_loss, total_tokens = 0.0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]  # shift right, as in toy_task

            src_mask = padding_mask(src, pad_idx=PAD_IDX)
            tgt_mask = decoder_mask(tgt_input, pad_idx=PAD_IDX)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            if train:
                scheduler.zero_grad()
                loss.backward()
                scheduler.step()

            n_tokens = (tgt_output != PAD_IDX).sum().item()
            total_loss += loss.item() * n_tokens  # criterion already averages per-token
            total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"epoch{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)
    return path


def train(config_path="configs/base.yaml"):
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        batch_size=cfg["batch_size"], vocab_size=cfg["vocab_size"], max_len=cfg["max_len"],
    )
    vocab_size = tokenizer.get_vocab_size()

    model = Transformer(
        src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
        d_model=cfg["d_model"], h=cfg["num_heads"], d_ff=cfg["d_ff"],
        n_layers=cfg["num_layers"], dropout=cfg["dropout"], max_len=cfg["max_len"],
    ).to(device)

    optimizer = build_optimizer(model)
    scheduler = NoamScheduler(optimizer, d_model=cfg["d_model"], warmup_steps=cfg["warmup_steps"])
    criterion = LabelSmoothingLoss(
        vocab_size=vocab_size, pad_idx=cfg["pad_idx"], smoothing=cfg["label_smoothing"],
    )

    best_val_loss = float("inf")
    for epoch in range(1, cfg["num_epochs"] + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, criterion, scheduler, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, scheduler, device, train=False)
        elapsed = time.time() - start

        print(f"epoch {epoch:2d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | {elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = save_checkpoint(model, optimizer, epoch, val_loss, cfg["checkpoint_dir"])
            print(f"  saved new best checkpoint: {path}")

    return model, tokenizer


if __name__ == "__main__":
    train()