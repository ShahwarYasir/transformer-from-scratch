import torch
import torch.nn as nn

from training.optimizer import NoamScheduler, build_optimizer


def test_adam_betas_and_eps():
    model = nn.Linear(4, 4)
    opt = build_optimizer(model)
    group = opt.param_groups[0]
    assert group["betas"] == (0.9, 0.98)
    assert group["eps"] == 1e-9


def test_lr_hand_example_step1():
    model = nn.Linear(4, 4)
    opt = build_optimizer(model)
    sched = NoamScheduler(opt, d_model=512, warmup_steps=4000)
    lr = sched.rate(step=1)
    assert abs(lr - 1.7469e-7) < 1e-10


def test_lr_hand_example_peak_at_warmup():
    model = nn.Linear(4, 4)
    opt = build_optimizer(model)
    sched = NoamScheduler(opt, d_model=512, warmup_steps=4000)
    lr = sched.rate(step=4000)
    assert abs(lr - 6.9877e-4) < 1e-7


def test_lr_increases_during_warmup():
    model = nn.Linear(4, 4)
    opt = build_optimizer(model)
    sched = NoamScheduler(opt, d_model=512, warmup_steps=4000)
    lr_early = sched.rate(step=100)
    lr_later = sched.rate(step=2000)
    assert lr_later > lr_early  # still rising before warmup ends


def test_lr_decreases_after_warmup():
    model = nn.Linear(4, 4)
    opt = build_optimizer(model)
    sched = NoamScheduler(opt, d_model=512, warmup_steps=4000)
    lr_peak = sched.rate(step=4000)
    lr_later = sched.rate(step=16000)
    assert lr_later < lr_peak  # decaying after warmup


def test_step_updates_optimizer_lr():
    model = nn.Linear(4, 4)
    opt = build_optimizer(model)
    sched = NoamScheduler(opt, d_model=512, warmup_steps=4000)

    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    sched.step()  # should call opt.step() and set param_groups[0]['lr']

    assert opt.param_groups[0]["lr"] == sched.rate(step=1)
    assert sched.step_num == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)