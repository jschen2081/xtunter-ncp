"""Native Trainer data adapter for captured DP/SP inputs; no training math."""

import json
import os
from pathlib import Path
from typing import Any, Iterator

import torch.distributed as dist
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
            row = {
                "step": payloads[0]["step"],
                "batch_sha256": [p["sha256"] for p in payloads],
                "valid_targets": sum(int(item["shifted_labels"].ne(-100).sum()) for item in items),
            }
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
    sp_size: int = 1

    def build(
        self,
        tokenizer: Any,
        dp_mesh: Any,
        global_batch_size: int,
        micro_batch_size: int,
        seed: int,
        shuffle: bool = False,
        total_step: int | None = None,
    ) -> RecordedBatchDataloader:
        """Build the native data adapter; input order is fixed by the capture.

        Args:
            tokenizer (Any): Unused; captured tokens are already processed.
            dp_mesh (Any): Native data-parallel mesh; SP peers share full inputs.
            global_batch_size (int): Samples per update.
            micro_batch_size (int): Rank-local samples per update, not engine MBS.
            seed (int): Unused for captured input ordering.
            shuffle (bool): Unused; captured input order is preserved.
            total_step (int | None): Must match the complete capture.

        Returns:
            RecordedBatchDataloader: Native update iterator and checkpoint state.
        """
        if (
            self.sp_size not in (1, 8)
            or dp_mesh.size() != 8 // self.sp_size
            or global_batch_size != self.capture_metadata["global_batch_size"]
            or micro_batch_size != self.capture_metadata["microbatches_per_update"] * self.sp_size
            or total_step != self.capture_metadata["steps"]
        ):
            raise ValueError("Native replay data topology or batch dimensions differ from capture")
        rank = dist.get_rank() if self.sp_size > 1 else dp_mesh.get_local_rank()
        if rank // self.sp_size != dp_mesh.get_local_rank():
            raise ValueError("Native replay requires contiguous SP groups")
        source = RecordedBatchSource(self.capture_dir, rank, self.capture_metadata, self.sp_size)
        return RecordedBatchDataloader(source, self.audit_dir)


def build_replay_config(config: dict[str, Any]) -> TrainerConfig:
    """Reuse the independent recipe and replace only its input source.

    Args:
        config (dict[str, Any]): Saved capture identity and independent LR settings.

    Returns:
        TrainerConfig: Standard CLI configuration for captured DP/SP inputs.
    """
    backend = flash_attn_varlen_func.__name__
    if os.environ.get("XTUNER_USE_FA3", "0") == "1" and backend != "gpu_flash_varlen_attn_v3":
        raise ValueError("FA3 requested but a different attention backend was selected")
    print(f"ATTENTION_BACKEND {backend}", flush=True)
    if int(os.environ.get("WORLD_SIZE", "8")) != 8 or config["sp_size"] not in (1, 8):
        raise ValueError("Native replay supports eight ranks with SP1 or SP8")
    dataloader_cfg = RecordedBatchDataloaderConfig(
        capture_dir=Path(config["capture_dir"]),
        audit_dir=Path(config["run_dir"]),
        capture_metadata=config["capture_metadata"],
        sp_size=config["sp_size"],
        pack_level="none",
        pack_max_length=config["sequence_length"] + 1,
        pack_to_max_length=True,
        group_by_length=False,
    )
    return build_trainer_config(config, dataloader_cfg=dataloader_cfg)
