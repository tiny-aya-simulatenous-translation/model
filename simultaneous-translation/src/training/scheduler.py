"""Linear-warmup + cosine-decay LR scheduler.

WHY THIS EXISTS
---------------
``WarmupCosineScheduler`` is the standard recipe for transformer
fine-tuning: ramp the LR linearly from 0 to ``base_lr`` over the
warmup window, then decay along a cosine to ``min_lr_ratio * base_lr``
by ``total_steps``.

The class deliberately does *not* inherit from
``torch.optim.lr_scheduler._LRScheduler`` -- the parent's hidden
``last_epoch`` accounting confuses our explicit ``step(int)`` API
and needlessly complicates checkpoint state.

This file is device-agnostic; nothing inside touches CUDA or XLA.
"""

import math


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def get_lr_multiplier(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay

    def step(self, step: int):
        m = self.get_lr_multiplier(step)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * m

    def get_last_lrs(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {
            "base_lrs": self.base_lrs,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
        }

    def load_state_dict(self, sd: dict):
        self.base_lrs = sd["base_lrs"]
        self.warmup_steps = sd["warmup_steps"]
        self.total_steps = sd["total_steps"]
        self.min_lr_ratio = sd["min_lr_ratio"]
