"""CPU-readable capture validation and ordered SP1 batch access."""

import json
from pathlib import Path
from typing import Any

import torch

from batch_contract import validate_batch

WORLD_SIZE = 8


def batch_path(root: Path, step: int, micro: int, rank: int) -> Path:
    """Locate an atomically completed production batch.

    Args:
        root (Path): Capture directory.
        step (int): Zero-based update.
        micro (int): Rank-local microbatch.
        rank (int): Original data-parallel rank.

    Returns:
        Path: Captured tensor file.
    """
    return root / "batches" / f"step{step:06d}.micro{micro:04d}.rank{rank:02d}.pt"


def capture_metadata(root: Path) -> dict[str, int]:
    """Validate completion and file identities without caching all tensors.

    Args:
        root (Path): Successful eight-rank capture.

    Returns:
        dict[str, int]: Fixed batch/sequence dimensions and update count.
    """
    completed = [json.loads((root / f"megatron_complete.rank{rank:02d}.json").read_text())
                 for rank in range(WORLD_SIZE)]
    first = completed[0]
    if not first or any(value != first for value in completed):
        raise ValueError("Megatron capture is incomplete or ranks consumed different steps")
    steps = sorted(int(step) for step in first)
    if steps != list(range(len(steps))) or any(str(step) not in first for step in steps):
        raise ValueError("SP1 replay requires contiguous zero-based captured updates")
    counts = list(first.values())
    if any(type(count) is not int or count <= 0 or count != counts[0] for count in counts):
        raise ValueError("SP1 replay requires fixed positive microbatch counts")
    expected = {batch_path(root, step, micro, rank).name for step in steps
                for micro in range(counts[0]) for rank in range(WORLD_SIZE)}
    actual = {path.name for path in (root / "batches").glob("*.pt")}
    if actual != expected:
        raise ValueError("Captured batch files are missing, duplicated or outside the completion range")
    lengths = []
    for rank in range(WORLD_SIZE):
        payload = torch.load(batch_path(root, 0, 0, rank), map_location="cpu", weights_only=True)
        validate_batch(payload, rank, WORLD_SIZE)
        if payload["step"] != 0 or payload["micro"] != 0:
            raise ValueError("Captured file name and payload identity differ")
        if payload["global_batch_size"] != counts[0] * WORLD_SIZE:
            raise ValueError("GBS differs from the completion microbatch count")
        lengths.append(payload["tokens"].shape[1])
    if any(length != lengths[0] for length in lengths):
        raise ValueError("Captured ranks disagree on sequence length")
    return {"steps": len(steps), "global_batch_size": counts[0] * WORLD_SIZE,
            "sequence_length": lengths[0], "microbatches_per_update": counts[0]}


class RecordedBatchSource:
    """Read one captured rank/update at a time, without applying its recorded LR.

    Args:
        root (Path): Capture directory; it is never modified.
        rank (int): Original data-parallel rank.
        metadata (dict[str, int]): Validated fixed capture dimensions.
    """

    def __init__(self, root: Path, rank: int, metadata: dict[str, int]) -> None:
        self.root = root.resolve()
        self.rank = rank
        self.metadata = metadata
        self.cursor = 0
        if not 0 <= rank < WORLD_SIZE:
            raise ValueError("Captured rank must be in [0, 8)")
        expected = {str(step): metadata["microbatches_per_update"] for step in range(metadata["steps"])}
        actual = json.loads((root / f"megatron_complete.rank{rank:02d}.json").read_text())
        if actual != expected:
            raise ValueError("Capture completion changed after recipe preparation")

    def next_update(self) -> list[dict[str, Any]]:
        """Read and validate the next complete local update.

        Returns:
            list[dict[str, Any]]: Original payloads in microbatch order.
        """
        if self.cursor >= self.metadata["steps"]:
            raise StopIteration
        payloads = []
        for micro in range(self.metadata["microbatches_per_update"]):
            payload = torch.load(batch_path(self.root, self.cursor, micro, self.rank),
                                 map_location="cpu", weights_only=True)
            validate_batch(payload, self.rank, WORLD_SIZE)
            if payload["step"] != self.cursor or payload["micro"] != micro:
                raise ValueError("Captured microbatch order or payload identity differs")
            if (payload["global_batch_size"] != self.metadata["global_batch_size"]
                    or payload["tokens"].shape[1] != self.metadata["sequence_length"]):
                raise ValueError("Captured batch dimensions changed within the replay")
            payloads.append(payload)
        self.cursor += 1
        return payloads

    def get_state_dict(self) -> dict[str, Any]:
        """Get the next unread update cursor.

        Returns:
            dict[str, Any]: Capture identity and delivered update count.
        """
        return {"capture": str(self.root), "rank": self.rank, "metadata": self.metadata, "cursor": self.cursor}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a cursor only for the identical capture/rank/shape contract.

        Args:
            state (dict[str, Any]): State returned by get_state_dict.
        """
        if any(state.get(key) != value for key, value in self.get_state_dict().items() if key != "cursor"):
            raise ValueError("Replay checkpoint refers to another capture or rank")
        cursor = state["cursor"]
        if type(cursor) is not int or not 0 <= cursor <= self.metadata["steps"]:
            raise ValueError("Invalid replay cursor")
        self.cursor = cursor


def collated_update(payloads: list[dict[str, Any]], context_class: Any) -> list[dict[str, Any]]:
    """Build native CPU items without tokenization, reshifting or repacking.

    Args:
        payloads (list[dict[str, Any]]): Validated rank-local microbatches.
        context_class (Any): Native SequenceContext; injectable in CPU tests.

    Returns:
        list[dict[str, Any]]: Ready-to-consume native Trainer items.
    """
    items = []
    for payload in payloads:
        tokens = payload["tokens"]
        boundaries = payload["cu_seqlens"].tolist()
        documents = tuple(tokens[:, start:end] for start, end in zip(boundaries, boundaries[1:]))
        context = context_class.from_input_ids(documents, device="cpu")
        if not torch.equal(context.position_ids, payload["position_ids"]):
            raise ValueError("Native SequenceContext position IDs differ from capture")
        labels = payload["labels"].masked_fill(payload["loss_mask"] == 0, -100)
        items.append({"seq_ctx": context, "shifted_labels": labels, "loss_mask": payload["loss_mask"]})
    return items
