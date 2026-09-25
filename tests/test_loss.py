import torch

from training.loss import LabelSmoothingLoss


def test_smoothed_target_hand_example():
    # vocab=5, correct token=2, smoothing=0.1
    criterion = LabelSmoothingLoss(vocab_size=5, pad_idx=0, smoothing=0.1)
    logits = torch.zeros(1, 5)  # doesn't matter for this check, we inspect true_dist directly
    target = torch.tensor([2])

    log_probs = torch.log_softmax(logits, dim=-1)
    true_dist = torch.full_like(logits, criterion.smoothing / (criterion.vocab_size - 1))
    true_dist.scatter_(1, target.unsqueeze(1), criterion.confidence)

    expected = torch.tensor([[0.025, 0.025, 0.9, 0.025, 0.025]])
    assert torch.allclose(true_dist, expected, atol=1e-6)
    assert torch.allclose(true_dist.sum(dim=-1), torch.tensor([1.0]))


def test_padding_excluded_from_loss():
    torch.manual_seed(0)
    criterion = LabelSmoothingLoss(vocab_size=10, pad_idx=0, smoothing=0.1)
    logits = torch.randn(4, 10)
    target_no_pad = torch.tensor([1, 2, 3, 4])
    target_with_pad = torch.tensor([1, 2, 3, 0])  # last one is padding

    loss_no_pad = criterion(logits, target_no_pad)
    # Changing the logits at the padded position should not affect the loss
    logits2 = logits.clone()
    logits2[3] = torch.randn(10) * 100  # wildly different padded-row logits
    loss_with_pad_a = criterion(logits, target_with_pad)
    loss_with_pad_b = criterion(logits2, target_with_pad)
    assert torch.allclose(loss_with_pad_a, loss_with_pad_b, atol=1e-4)


def test_loss_is_lower_for_correct_confident_prediction():
    criterion = LabelSmoothingLoss(vocab_size=5, pad_idx=0, smoothing=0.1)
    target = torch.tensor([2])

    confident_correct = torch.tensor([[0.0, 0.0, 10.0, 0.0, 0.0]])  # strongly predicts class 2
    confident_wrong = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])     # strongly predicts class 0

    loss_correct = criterion(confident_correct, target)
    loss_wrong = criterion(confident_wrong, target)
    assert loss_correct < loss_wrong


def test_label_smoothing_never_reaches_zero_loss():
    # Even a "perfect" one-hot prediction has nonzero loss, because the smoothed
    # target itself isn't one-hot (this is the whole point of label smoothing)
    criterion = LabelSmoothingLoss(vocab_size=5, pad_idx=0, smoothing=0.1)
    target = torch.tensor([2])
    near_perfect_logits = torch.tensor([[-1e4, -1e4, 1e4, -1e4, -1e4]])
    loss = criterion(near_perfect_logits, target)
    assert loss.item() > 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)