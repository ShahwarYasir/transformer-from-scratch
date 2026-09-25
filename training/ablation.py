"""
Small ablation: how does the number of attention heads affect how well the
model learns? Mirrors the paper's Table 3 (varying h), but run on the toy
copy task instead of full Multi30k training, since that lets us compare
several configs in a couple of minutes on CPU instead of hours on a GPU.

Run locally:
    python -m training.ablation
"""

import torch

from training.toy_task import generate_batch, greedy_decode, train_copy_task


def run_ablation(head_counts=(1, 2, 4, 8), steps=800, seq_len=8, d_model=64, seed=0):
    results = []
    for h in head_counts:
        torch.manual_seed(seed)
        model, final_loss = train_copy_task(
            steps=steps, batch_size=32, seq_len=seq_len,
            d_model=d_model, h=h, d_ff=d_model * 2, n_layers=2,
        )
        torch.manual_seed(seed + 1)  # different data than training, for a fair eval
        src, _, _ = generate_batch(batch_size=50, seq_len=seq_len)
        pred = greedy_decode(model, src, max_len=seq_len)
        accuracy = (pred == src).float().mean().item()

        results.append({"heads": h, "final_loss": final_loss, "accuracy": accuracy})
        print(f"h={h:2d} | final_loss={final_loss:.4f} | copy accuracy={accuracy:.2%}")

    return results


if __name__ == "__main__":
    print("Ablation: number of attention heads (paper Table 3 style, on the copy task)\n")
    run_ablation()