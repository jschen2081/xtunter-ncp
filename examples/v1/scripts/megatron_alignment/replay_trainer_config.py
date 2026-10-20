"""Native Trainer data adapter for captured SP1 inputs; no training math."""

import json
import os
from pathlib import Path
from typing import Any, Iterator

from pydantic import ConfigDict
from recorded_batch_source import RecordedBatchSource, collated_update
from train_qwen35 import build_trainer_config

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.datasets.dataloader import BaseDataloader
from xtuner.v1.ops.flash_attn import flash_attn_varlen_func
from xtuner.v1.train import TrainerConfig


class RecordedBatchDataloader(BaseDataloader):
    """Deliver complete captured updates to Trainer and audit input identities.

    Args:
        source (RecordedBatchSource): Validated ordered CPU source.
        audit_dir (Path): New writable native-run directory.
    """

    def __init__(self, source: RecordedBatchSource, audit_dir: Path) -> None:
        self.source = source
        self.audit = audit_dir / f"replay_inputs.rank{source.rank:02d}.jsonl"
        self.audit.open("x").close()

    def __len__(self) -> int:
        return self.source.metadata["steps"]

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        while self.source.cursor < len(self):
            payloads = self.source.next_update()
            items = collated_update(payloads, SequenceContext)
            row = {"step": payloads[0]["step"], "batch_sha256": [p["sha256"] for p in payloads],
                   "valid_targets": sum(int(item["shifted_labels"].ne(-100).sum()) for item in items)}
            with self.audit.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            yield items

    def set_epoch(self, epoch: int) -> None:
        if self.source.cursor >= len(self):
            raise ValueError("Replay exhausted; captured updates must not be recycled")

    def get_state_dict(self) -> dict[str, Any]:
        """Get the delivered update cursor.

        Returns:
            dict[str, Any]: Native checkpoint data state.
        """
        return self.source.get_state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the cursor with capture-identity validation.

        Args:
            state_dict (dict[str, Any]): Previously saved data state.
        """
        self.source.load_state_dict(state_dict)


class RecordedBatchDataloaderConfig(DataloaderConfig):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    capture_dir: Path
    audit_dir: Path
    capture_metadata: dict[str, int]

    def build(
        self, tokenizer: Any, dp_mesh: Any, global_batch_size: int,
        micro_batch_size: int, seed: int, shuffle: bool = False, total_step: int | None = None,
    ) -> RecordedBatchDataloader:
        """Build the native data adapter; input order is fixed by the capture.

        Args:
            tokenizer (Any): Unused; captured tokens are already processed.
            dp_mesh (Any): Native eight-rank SP1 data mesh.
            global_batch_size (int): Samples per update.
            micro_batch_size (int): Rank-local samples per update, not engine MBS.
            seed (int): Unused for captured input ordering.
            shuffle (bool): Unused; captured input order is preserved.
            total_step (int | None): Must match the complete capture.

        Returns:
            RecordedBatchDataloader: Native update iterator and checkpoint state.
        """
        if (dp_mesh.size() != 8 or global_batch_size != self.capture_metadata["global_batch_size"]
                or micro_batch_size != self.capture_metadata["microbatches_per_update"]
                or total_step != self.capture_metadata["steps"]):
            raise ValueError("Native SP1 data topology or batch dimensions differ from capture")
        source = RecordedBatchSource(self.capture_dir, dp_mesh.get_local_rank(), self.capture_metadata)
        return RecordedBatchDataloader(source, self.audit_dir)


def build_replay_config(config: dict[str, Any]) -> TrainerConfig:
    """Reuse the independent recipe and replace only its input source.

    Args:
        config (dict[str, Any]): Saved capture identity and independent LR settings.

    Returns:
        TrainerConfig: Standard CLI configuration for captured SP1 inputs.
    """
    trainer = build_trainer_config(config)
    backend = flash_attn_varlen_func.__name__
    if os.environ.get("XTUNER_USE_FA3", "0") == "1" and backend != "gpu_flash_varlen_attn_v3":
        raise ValueError("FA3 requested but a different attention backend was selected")
    print(f"ATTENTION_BACKEND {backend}", flush=True)
    if int(os.environ.get("WORLD_SIZE", "8")) != 8 or config["sp_size"] != 1:
        raise ValueError("Native replay currently supports eight ranks with SP1")
    trainer.dataloader_cfg = RecordedBatchDataloaderConfig(
        capture_dir=Path(config["capture_dir"]), audit_dir=Path(config["run_dir"]),
        capture_metadata=config["capture_metadata"], pack_level="none",
        pack_max_length=config["sequence_length"] + 1, pack_to_max_length=True, group_by_length=False,
    )
    return trainer
