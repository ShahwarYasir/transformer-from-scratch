# End-to-end test on a small toy copy task, Section 5
import torch

from training.toy_task import VOCAB_SIZE, generate_batch, greedy_decode, train_copy_task


def test_batch_shapes_and_shift():
    src, tgt_input, tgt_output = generate_batch(batch_size=4, seq_len=6)
    assert src.shape == (4, 6)
    assert tgt_input.shape == (4, 6)
    assert tgt_output.shape == (4, 6)
    assert torch.equal(tgt_output, src)  # target output IS the source, unchanged
    assert torch.all(tgt_input[:, 0] == 1)  # every row starts with <bos>=1
    assert torch.equal(tgt_input[:, 1:], src[:, :-1])  # shifted-right by one


def test_loss_decreases_with_training():
    torch.manual_seed(0)
    # Two short runs: loss after more steps should be lower than after fewer steps
    _, loss_short = train_copy_task(steps=20, seq_len=6, d_model=32, h=2, d_ff=64, n_layers=1)
    torch.manual_seed(0)
    _, loss_long = train_copy_task(steps=300, seq_len=6, d_model=32, h=2, d_ff=64, n_layers=1)
    assert loss_long < loss_short


def test_model_learns_the_copy_task():
    # The real end-to-end sanity check: after training, greedy decode should
    # reproduce the input with high accuracy. This is the test that would catch
    # a broken mask, a broken residual connection, or any wiring mistake.
    torch.manual_seed(0)
    model, final_loss = train_copy_task(
        steps=1500, batch_size=32, seq_len=8, d_model=64, h=4, d_ff=128, n_layers=2,
    )
    torch.manual_seed(1)  # different seed for eval data than training data
    src, _, _ = generate_batch(batch_size=20, seq_len=8)
    pred = greedy_decode(model, src, max_len=8)

    accuracy = (pred == src).float().mean().item()
    assert accuracy > 0.9, f"copy task accuracy too low: {accuracy:.2%} (final loss {final_loss:.4f})"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)