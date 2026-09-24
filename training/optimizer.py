# Adam optimizer with custom learning-rate schedule, Section 5.3
import torch


def build_optimizer(model, lr=1.0):
    """
    Adam with the paper's betas and epsilon (Section 5.3).
    lr=1.0 here is a placeholder; the real per-step LR is set by NoamScheduler below,
    which overwrites param_group['lr'] before every optimizer.step().
    """
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)


class NoamScheduler:
    """
    The paper's warmup-then-decay learning rate schedule (Section 5.3):

        lr = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)

    LR rises linearly for `warmup_steps`, then decays proportional to 1/sqrt(step).
    Peak LR occurs exactly at step == warmup_steps.

    Named "Noam" after the paper's reference implementation convention.
    """

    def __init__(self, optimizer, d_model=512, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self._last_lr = 0.0

    def rate(self, step=None):
        if step is None:
            step = self.step_num
        step = max(step, 1)  # avoid step=0 -> division by zero
        return (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))

    def step(self):
        self.step_num += 1
        lr = self.rate()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self._last_lr = lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()