"""Local sample-count-based WSD schedule for Qwen35 training."""
import math
from typing import Any


def sample_count_wsd_lr(step: int, global_batch_size: int, config: dict[str, Any]) -> float:
    """Compute the current recipe's WSD LR before an optimizer update.

    Args:
        step (int): Number of completed optimizer updates.
        global_batch_size (int): Samples consumed per optimizer update.
        config (dict[str, Any]): Peak/minimum LR and warmup/decay iteration counts.

    Returns:
        float: Learning rate for this optimizer update.
    """
    consumed = step * global_batch_size
    warmup = config["warmup_iters"] * global_batch_size
    decay = config["decay_iters"] * global_batch_size
    wsd = config["wsd_decay_iters"] * global_batch_size
    maximum, minimum = config["lr"], config["min_lr"]
    if warmup < 0 or decay <= warmup or not 0 < wsd <= decay or not 0 <= minimum <= maximum:
        raise ValueError("Invalid warmup/WSD schedule")
    if warmup > 0 and consumed <= warmup:
        return maximum * float(consumed) / float(warmup)
    if consumed > decay:
        return minimum
    coefficient = 1.0
    if consumed > decay - wsd:
        ratio = float(consumed - (decay - wsd)) / float(wsd)
        coefficient = 0.5 * (math.cos(math.pi * ratio) + 1.0)
    return minimum + coefficient * (maximum - minimum)
