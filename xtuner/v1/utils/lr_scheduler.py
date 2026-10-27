"""Learning rate schedulers shared by the standard training loop."""

import math
from typing import TYPE_CHECKING

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, LRScheduler, SequentialLR


if TYPE_CHECKING:
    from xtuner.v1.config import LRConfig


class WarmupDecayLR(LRScheduler):
    """Explicit-horizon warmup/decay with absolute LRs and a fixed decay floor.

    Args:
        optimizer (Optimizer): Optimizer with per-group peak learning rates.
        style (str): Validated constant, linear, cosine or wsd schedule.
        minimum (float): Minimum LR shared by the parameter groups.
        warmup (int): Number of warmup updates starting from zero LR.
        decay (int): Full schedule horizon in optimizer updates.
        tail (int | None): WSD cosine-tail length, otherwise None.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        style: str,
        minimum: float,
        warmup: int,
        decay: int,
        tail: int | None,
    ) -> None:
        self.style = style
        self.minimum = minimum
        self.warmup = warmup
        self.decay = decay
        self.tail = tail
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Compute absolute LRs for the next update from completed updates.

        Returns:
            list[float]: Per-group LRs for the next update.
        """
        step = self.last_epoch
        if self.warmup > 0 and step <= self.warmup:
            return [peak * float(step) / float(self.warmup) for peak in self.base_lrs]
        if self.style == "constant":
            return list(self.base_lrs)
        if step > self.decay:
            return [self.minimum for _ in self.base_lrs]
        if self.style == "wsd":
            assert self.tail is not None
            start = self.decay - self.tail
            ratio = 0.0 if step <= start else float(step - start) / float(self.tail)
        else:
            ratio = float(step - self.warmup) / float(self.decay - self.warmup)
        coefficient = 1.0 - ratio if self.style == "linear" else 0.5 * (math.cos(math.pi * ratio) + 1.0)
        return [self.minimum + coefficient * (peak - self.minimum) for peak in self.base_lrs]


def build_lr_scheduler(optimizer: Optimizer, config: "LRConfig", total_steps: int) -> LRScheduler:
    """Build a scheduler whose initial LR is used by the first optimizer
    update.

    Explicit step counts use completed optimizer updates, rather than microsteps.
    With fixed GBS this is equivalent to a sample-count schedule. The
    caller advances the scheduler after each update; resume restores its state.
    Existing recipes without explicit counts retain their legacy schedulers.

    Args:
        optimizer (Optimizer): Optimizer with the configured peak LR in each group.
        config (LRConfig): Schedule type, floor and optional explicit horizons.
        total_steps (int): Run length used only when no decay horizon is specified.

    Returns:
        LRScheduler: Initialized scheduler, including the LR for update zero.
    """
    decay = config.decay_steps if config.decay_steps is not None else total_steps
    warmup = config.warmup_steps
    if warmup is None:
        warmup = int(config.warmup_ratio * decay if config.warmup_ratio < 1 else config.warmup_ratio)
    if warmup < 0 or decay <= 0 or config.lr_min < 0:
        raise ValueError("Warmup and minimum LR must be nonnegative; decay_steps must be positive")
    explicit = config.lr_type == "wsd" or config.warmup_steps is not None or config.decay_steps is not None
    if not explicit:
        # Preserve existing recipes, including SequentialLR checkpoint format.
        warmup_scheduler = LambdaLR(optimizer, lambda step: step / warmup if step < warmup else 1)
        scheduler: LRScheduler
        if config.lr_type == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=config.lr_min / optimizer.defaults["lr"],
                total_iters=decay - warmup,
            )
        elif config.lr_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=decay - warmup, eta_min=config.lr_min)
        elif config.lr_type == "constant":
            scheduler = LambdaLR(optimizer, lambda step: 1.0)
        else:
            raise ValueError(f"Unsupported lr type: {config.lr_type}")
        return SequentialLR(optimizer, [warmup_scheduler, scheduler], milestones=[warmup])

    if warmup >= decay:
        raise ValueError("warmup_steps must be smaller than the full decay_steps horizon")
    tail = config.wsd_decay_steps
    if config.lr_type == "wsd" and (tail is None or tail <= 0 or tail > decay - warmup):
        raise ValueError("WSD requires 0 < wsd_decay_steps <= decay_steps - warmup_steps")
    if config.lr_type != "wsd" and tail is not None:
        raise ValueError("wsd_decay_steps is only valid for a WSD schedule")
    peaks = [group["lr"] for group in optimizer.param_groups]
    if any(peak <= 0 or config.lr_min > peak for peak in peaks):
        raise ValueError("Each peak LR must be positive and at least lr_min")

    return WarmupDecayLR(optimizer, config.lr_type, config.lr_min, warmup, decay, tail)
