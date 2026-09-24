import torch
import torch.nn as nn

from model.masking import decoder_mask, padding_mask
from model.transformer import Transformer

BOS = 1  # start-of-sequence token
PAD = 0  # padding token (unused for fixed-length toy sequences, kept for mask compatibility)
VOCAB_SIZE = 12  # tokens 2..11 are "real" data tokens; 0=pad, 1=bos


def generate_batch(batch_size, seq_len, device="cpu"):
    """
    Random copy-task batch. Returns:
      src:        (B, seq_len)      the sequence to copy, tokens in [2, VOCAB_SIZE)
      tgt_input:  (B, seq_len)      <bos> + src[:-1], fed into the decoder
      tgt_output: (B, seq_len)      src itself, what the decoder must predict
    """
    src = torch.randint(2, VOCAB_SIZE, (batch_size, seq_len), device=device)
    bos_col = torch.full((batch_size, 1), BOS, device=device)
    tgt_input = torch.cat([bos_col, src[:, :-1]], dim=1)
    tgt_output = src
    return src, tgt_input, tgt_output


def train_copy_task(steps=500, batch_size=32, seq_len=8, d_model=64, h=4, d_ff=128, n_layers=2, lr=3e-4, device="cpu"):
    """Trains a small Transformer on the copy task. Returns the trained model and final loss."""
    model = Transformer(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=d_model, h=h, d_ff=d_ff, n_layers=n_layers,dropout=0.0,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    final_loss = None
    for step in range(steps):
        src, tgt_input, tgt_output = generate_batch(batch_size, seq_len, device)
        src_mask = padding_mask(src, pad_idx=PAD)
        tgt_mask = decoder_mask(tgt_input, pad_idx=PAD)

        logits = model(src, tgt_input, src_mask, tgt_mask)  # (B, seq_len, VOCAB_SIZE)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()

    return model, final_loss


@torch.no_grad()
def greedy_decode(model, src, max_len, device="cpu"):
    """Greedy autoregressive decoding: generate one token at a time."""
    model.eval()
    src_mask = padding_mask(src, pad_idx=PAD)
    memory = model.encode(src, src_mask)

    ys = torch.full((src.size(0), 1), BOS, device=device)
    for _ in range(max_len):
        tgt_mask = decoder_mask(ys, pad_idx=PAD)
        out = model.decode(ys, memory, src_mask, tgt_mask)
        logits = model.generator(out[:, -1])  # only need the last position's prediction
        next_token = logits.argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
    return ys[:, 1:]  # drop the leading <bos>


if __name__ == "__main__":
    torch.manual_seed(0)
    model, loss = train_copy_task(steps=500)
    print(f"final training loss: {loss:.4f}")

    src, _, _ = generate_batch(batch_size=4, seq_len=8)
    pred = greedy_decode(model, src, max_len=8)
    print("input: ", src.tolist())
    print("output:", pred.tolist())
    accuracy = (pred == src).float().mean().item()
    print(f"token accuracy: {accuracy:.2%}")