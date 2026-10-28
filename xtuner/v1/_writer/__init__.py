from pathlib import Path
from typing import Protocol

from .jsonl_writer import JsonlWriter
from .tb_writer import TensorboardWriter


class Writer(Protocol):
    def __init__(
        self,
        log_dir: str | Path | None = None,
    ): ...

    def add_scalar(
        self,
        *,
        tag: str,
        scalar_value: float,
        global_step: int,
    ): ...

    def add_scalars(
        self,
        *,
        tag_scalar_dict: dict[str, float],
        global_step: int,
    ): ...

    def close(self) -> None: ...


class CombinedWriter:
    """Write identical scalar values to TensorBoard and full-precision JSONL.

    Args:
        log_dir (str | Path | None): Directory for both independent outputs.
    """

    def __init__(self, log_dir: str | Path | None = None) -> None:
        self.tensorboard = TensorboardWriter(log_dir=log_dir)
        self.jsonl = JsonlWriter(log_dir=log_dir)

    def add_scalar(self, *, tag: str, scalar_value: float, global_step: int) -> None:
        """Record a scalar in both formats.

        Args:
            tag (str): Metric name.
            scalar_value (float): Metric value.
            global_step (int): Completed update count.
        """
        self.tensorboard.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)
        self.jsonl.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)

    def add_scalars(self, *, tag_scalar_dict: dict[str, float], global_step: int) -> None:
        """Record a step in both formats.

        Args:
            tag_scalar_dict (dict[str, float]): Metrics for this update.
            global_step (int): Completed update count.
        """
        self.tensorboard.add_scalars(tag_scalar_dict=tag_scalar_dict, global_step=global_step)
        self.jsonl.add_scalars(tag_scalar_dict=tag_scalar_dict, global_step=global_step)

    def close(self) -> None:
        """Flush and close both outputs."""
        try:
            self.tensorboard.close()
        finally:
            self.jsonl.close()


def get_writer(
    *,
    writer_type: str,
    log_dir: str | Path | None = None,
) -> Writer:
    if writer_type == "both":
        return CombinedWriter(log_dir=log_dir)
    if writer_type == "jsonl":
        return JsonlWriter(log_dir=log_dir)
    elif writer_type == "tensorboard":
        return TensorboardWriter(log_dir=log_dir)
    else:
        raise ValueError(f"Unknown writer type: {writer_type}")


__all__ = ["JsonlWriter", "TensorboardWriter"]
