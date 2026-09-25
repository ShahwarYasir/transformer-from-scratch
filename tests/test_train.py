"""
Tests the training-loop mechanics (run_epoch, checkpointing) using a tiny fake
dataset, no Multi30k download or GPU required. The real end-to-end run (with
the actual dataset) happens in Colab via train.py's __main__ block.
"""

import os
import shutil

import torch
from torch.utils.data import DataLoader

from training.loss import LabelSmoothingLoss
from model.masking import decoder_mask, padding_mask
from training.optimizer import NoamScheduler, build_optimizer
from training.train import run_epoch, save_checkpoint
from model.transformer import Transformer

VOCAB_SIZE = 20
PAD_IDX = 0


def fake_dataloader(n_batches=5, batch_size=4, seq_len=6):
    """Random token batches, standing in for a real Multi30k DataLoader."""
    data = []
    for _ in range(n_batches):
        src = torch.randint(2, VOCAB_SIZE, (batch_size, seq_len))
        tgt = torch.randint(2, VOCAB_SIZE, (batch_size, seq_len + 1))  # +1: shifted in run_epoch
        data.append((src, tgt))
    return data  # a plain list works fine as a DataLoader stand-in (iterable of batches)


def build_tiny_model():
    return Transformer(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=16, h=2, d_ff=32, n_layers=1, dropout=0.0,
    )


def test_run_epoch_train_reduces_loss_over_calls():
    torch.manual_seed(0)
    model = build_tiny_model()
    optimizer = build_optimizer(model)
    scheduler = NoamScheduler(optimizer, d_model=16, warmup_steps=10)
    criterion = LabelSmoothingLoss(vocab_size=VOCAB_SIZE, pad_idx=PAD_IDX, smoothing=0.1)

    loader = fake_dataloader(n_batches=8)
    loss1 = run_epoch(model, loader, criterion, scheduler, device="cpu", train=True)
    loss2 = run_epoch(model, loader, criterion, scheduler, device="cpu", train=True)
    loss3 = run_epoch(model, loader, criterion, scheduler, device="cpu", train=True)
    # training on the same fake batches repeatedly should push loss down
    assert loss3 < loss1


def test_run_epoch_eval_does_not_update_weights():
    torch.manual_seed(0)
    model = build_tiny_model()
    optimizer = build_optimizer(model)
    scheduler = NoamScheduler(optimizer, d_model=16, warmup_steps=10)
    criterion = LabelSmoothingLoss(vocab_size=VOCAB_SIZE, pad_idx=PAD_IDX, smoothing=0.1)

    loader = fake_dataloader(n_batches=3)
    before = model.generator.weight.clone()
    run_epoch(model, loader, criterion, scheduler, device="cpu", train=False)
    after = model.generator.weight
    assert torch.equal(before, after)  # eval mode: no gradient step happened


def test_checkpoint_saves_and_reloads():
    torch.manual_seed(0)
    model = build_tiny_model()
    optimizer = build_optimizer(model)
    tmp_dir = "test_checkpoints_tmp"

    path = save_checkpoint(model, optimizer, epoch=1, val_loss=2.5, checkpoint_dir=tmp_dir)
    assert os.path.exists(path)

    ckpt = torch.load(path, weights_only=True)
    assert ckpt["epoch"] == 1
    assert ckpt["val_loss"] == 2.5

    new_model = build_tiny_model()
    new_model.load_state_dict(ckpt["model_state_dict"])
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)

    shutil.rmtree(tmp_dir)  # cleanup


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)