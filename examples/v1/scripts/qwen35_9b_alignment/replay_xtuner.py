"""Captured batch validation, native replay adapter and SP1/SP8 replay CLI.

Training uses the standard Trainer and independently computed LR. Native
framework imports are delayed until loading the generated training recipe.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import torch
import torch.distributed as dist
from train_qwen35 import build_trainer_config, collect_metrics, env_bool
from training_environment import capture_training_environment


if TYPE_CHECKING:
    from xtuner.v1.datasets.config import DataloaderConfig
    from xtuner.v1.train import TrainerConfig

TENSOR_FIELDS = ("tokens", "position_ids", "labels", "loss_mask", "cu_seqlens")

META_FIELDS = ("step", "micro", "rank", "world_size", "global_batch_size", "lr", "packed")


def batch_digest(payload):
    digest = hashlib.sha256()
    digest.update(json.dumps({key: payload[key] for key in META_FIELDS}, sort_keys=True).encode())
    for key in TENSOR_FIELDS:
        value = payload[key].detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_batch(payload, rank, world):
    import torch

    if payload["rank"] != rank or payload["world_size"] != world:
        raise ValueError("Captured rank/world size differs")
    if payload["sha256"] != batch_digest(payload):
        raise ValueError("Captured batch checksum differs")
    tokens = payload["tokens"]
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] % 4:
        raise ValueError("Replay requires MBS=1 and chunk-aligned sequences")
    for key in ("position_ids", "labels", "loss_mask"):
        if payload[key].shape != tokens.shape:
            raise ValueError(f"Invalid {key} shape")
    mask = payload["loss_mask"]
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Fractional loss masks require weighted CE; refusing to discard weights")
    if torch.any((payload["labels"] < 0) & (mask != 0)):
        raise ValueError("Active production labels must be nonnegative")
    bounds = payload["cu_seqlens"].tolist()
    if bounds[0] != 0 or bounds[-1] != tokens.shape[1]:
        raise ValueError("Boundaries do not cover tokens")
    if any(b <= a or (b - a) % 4 for a, b in zip(bounds, bounds[1:])):
        raise ValueError("Each sequence must align to concept chunks")
    expected_positions = torch.cat([torch.arange(b - a) for a, b in zip(bounds, bounds[1:])])[None]
    if not torch.equal(payload["position_ids"], expected_positions):
        raise ValueError("Unsupported position resets inside a captured attention sequence")
    if payload["global_batch_size"] % world:
        raise ValueError("GBS must be divisible by eight")


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
    completed = [
        json.loads((root / f"megatron_complete.rank{rank:02d}.json").read_text()) for rank in range(WORLD_SIZE)
    ]
    first = completed[0]
    if not first or any(value != first for value in completed):
        raise ValueError("Megatron capture is incomplete or ranks consumed different steps")
    steps = sorted(int(step) for step in first)
    if steps != list(range(len(steps))) or any(str(step) not in first for step in steps):
        raise ValueError("Replay requires contiguous zero-based captured updates")
    counts = list(first.values())
    if any(type(count) is not int or count <= 0 or count != counts[0] for count in counts):
        raise ValueError("Replay requires fixed positive microbatch counts")
    expected = {
        batch_path(root, step, micro, rank).name
        for step in steps
        for micro in range(counts[0])
        for rank in range(WORLD_SIZE)
    }
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
    return {
        "steps": len(steps),
        "global_batch_size": counts[0] * WORLD_SIZE,
        "sequence_length": lengths[0],
        "microbatches_per_update": counts[0],
    }


class RecordedBatchSource:
    """Read each update for one SP group, without applying its recorded LR.

    Args:
        root (Path): Capture directory; it is never modified.
        rank (int): Current global training rank.
        metadata (dict[str, int]): Validated fixed capture dimensions.
        sp_size (int): Number of peers sharing each full input sequence.
    """

    def __init__(self, root: Path, rank: int, metadata: dict[str, int], sp_size: int = 1) -> None:
        self.root = root.resolve()
        self.rank = rank
        self.metadata = metadata
        self.cursor = 0
        if sp_size not in (1, 8):
            raise ValueError("Replay supports SP1 or SP8")
        self.sp_size = sp_size
        self.source_ranks = range(rank // sp_size * sp_size, (rank // sp_size + 1) * sp_size)
        if not 0 <= rank < WORLD_SIZE:
            raise ValueError("Captured rank must be in [0, 8)")
        expected = {str(step): metadata["microbatches_per_update"] for step in range(metadata["steps"])}
        for source_rank in self.source_ranks:
            actual = json.loads((root / f"megatron_complete.rank{source_rank:02d}.json").read_text())
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
            for source_rank in self.source_ranks:
                payload = torch.load(
                    batch_path(self.root, self.cursor, micro, source_rank), map_location="cpu", weights_only=True
                )
                validate_batch(payload, source_rank, WORLD_SIZE)
                if payload["step"] != self.cursor or payload["micro"] != micro:
                    raise ValueError("Captured microbatch order or payload identity differs")
                if (
                    payload["global_batch_size"] != self.metadata["global_batch_size"]
                    or payload["tokens"].shape[1] != self.metadata["sequence_length"]
                ):
                    raise ValueError("Captured batch dimensions changed within the replay")
                payloads.append(payload)
        self.cursor += 1
        return payloads

    def get_state_dict(self) -> dict[str, Any]:
        """Get the next unread update cursor.

        Returns:
            dict[str, Any]: Capture identity and delivered update count.
        """
        return {
            "capture": str(self.root),
            "rank": self.rank,
            "sp_size": self.sp_size,
            "metadata": self.metadata,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a cursor only for the identical capture/rank/shape contract.

        Args:
            state (dict[str, Any]): State returned by get_state_dict.
        """
        if any(
            state.get(key, 1 if key == "sp_size" else None) != value
            for key, value in self.get_state_dict().items()
            if key != "cursor"
        ):
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


def _replay_dataloader_config_type() -> type["DataloaderConfig"]:
    from pydantic import ConfigDict

    from xtuner.v1.data_proto import SequenceContext
    from xtuner.v1.datasets.config import DataloaderConfig
    from xtuner.v1.datasets.dataloader import BaseDataloader

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

    return RecordedBatchDataloaderConfig


def build_replay_config(config: dict[str, Any]) -> "TrainerConfig":
    """Reuse the independent recipe and replace only its input source.

    Args:
        config (dict[str, Any]): Saved capture identity and independent LR settings.

    Returns:
        TrainerConfig: Standard CLI configuration for captured DP/SP inputs.
    """
    from xtuner.v1.ops.flash_attn import flash_attn_varlen_func

    backend = flash_attn_varlen_func.__name__
    if os.environ.get("XTUNER_USE_FA3", "0") == "1" and backend != "gpu_flash_varlen_attn_v3":
        raise ValueError("FA3 requested but a different attention backend was selected")
    print(f"ATTENTION_BACKEND {backend}", flush=True)
    if int(os.environ.get("WORLD_SIZE", "8")) != 8 or config["sp_size"] not in (1, 8):
        raise ValueError("Native replay supports eight ranks with SP1 or SP8")
    config_type = _replay_dataloader_config_type()
    dataloader_cfg = config_type(
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


def prepare_replay(args: argparse.Namespace) -> Path:
    """Prepare an immutable input contract and native CLI configuration.

    Args:
        args (argparse.Namespace): Capture path and independent training settings.

    Returns:
        Path: Generated native Trainer configuration.
    """
    root = args.run_dir.resolve()
    output = root / f"xtuner_sp{args.sp_size}_native"
    if output.exists() or any(
        (root / name).exists() for name in (f"xtuner_sp{args.sp_size}.jsonl", "xtuner_complete.json")
    ):
        raise FileExistsError("Use a fresh prepared capture directory; replay results already exist")
    metadata = capture_metadata(root)
    contract = json.loads((root / "contract.json").read_text())
    if args.sp_size not in (1, 8):
        raise ValueError("Native replay supports SP1 or SP8")
    if metadata["sequence_length"] % (4 * args.sp_size):
        raise ValueError("Captured sequence length must be divisible by 4 * SP size")
    if args.profile_step <= 0 or args.warmup_iters < 0 or args.decay_iters <= args.warmup_iters:
        raise ValueError("Require positive profile-step and 0 <= warmup-iters < decay-iters")
    if not 0 <= args.min_lr <= args.lr or args.lr <= 0:
        raise ValueError("Require 0 <= min-lr <= lr and lr > 0")
    if args.lr_decay_style == "wsd" and not 0 < args.wsd_decay_iters <= args.decay_iters - args.warmup_iters:
        raise ValueError("WSD tail must be positive and fit after warmup")
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        capture_dir=str(root),
        run_dir=str(output),
        capture_metadata=metadata,
        **metadata,
        sp_size=args.sp_size,
        hf_path=str(Path(args.hf_path or contract["hf_path"]).resolve()),
        # Captured inputs are final tensors; the native loader does not use a tokenizer.
        tokenizer_path=None,
        capture_tokenizer=contract.get("tokenizer"),
        data_format="indexed",
        data_path=str(root / "batches"),
        num_workers=0,
        pack_workers=1,
        lr_source="xtuner_scheduler",
        replayed_fields=["tokens", "position_ids", "labels", "loss_mask", "cu_seqlens"],
        recorded_lr_usage="checksum and post-run comparison only; never applied to optimizer",
    )
    output.mkdir()
    config_path = output / "training_config.json"
    config["kernel_env"] = capture_training_environment()
    config_path.write_text(json.dumps(config, indent=2))
    scripts = args.xtuner_root.resolve() / "examples/v1/scripts/qwen35_9b_alignment"
    recipe = (
        "import json\nimport sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(scripts)!r})\n"
        "from training_environment import validate_training_environment\n"
        f"_recipe = json.loads(Path({str(config_path)!r}).read_text())\n"
        'validate_training_environment(_recipe.get("kernel_env"))\n'
        "from replay_xtuner import build_replay_config\n"
        "trainer = build_replay_config(_recipe)\n"
    )
    target = output / "trainer_config.py"
    target.write_text(recipe)
    (output / "status").write_text("PREPARED\n")
    print(
        "CHANGE_SCOPE Megatron unchanged; xTuner config/data/audit adapter; existing model/loss/Muon reused",
        flush=True,
    )
    print("TRAINING_ENTRY standard CLI -> Trainer.fit -> TrainEngine", flush=True)
    print("REPLAY_SOURCE inputs/masks/positions/boundaries; shared initial checkpoint", flush=True)
    print(
        f"LR_SOURCE xTuner scheduler: {args.lr_decay_style}; peak={args.lr} min={args.min_lr} "
        f"warmup={args.warmup_iters} horizon={args.decay_iters} tail={args.wsd_decay_iters}",
        flush=True,
    )
    print(
        f"NATIVE_CONFIG {target}\nTENSORBOARD {output / 'logs/exp_tracking'} "
        f"profiling={'enabled' if args.profile_trace else 'disabled'}",
        flush=True,
    )
    return target


def collect_replay(root: Path, sp_size: int = 1) -> None:
    """Join native post-update metrics with validated per-rank input audits.

    Args:
        root (Path): Prepared capture directory after successful torchrun exit.
        sp_size (int): Sequence parallel size used for the native run.
    """
    if sp_size not in (1, 8):
        raise ValueError("Native replay supports SP1 or SP8")
    output = root / f"xtuner_sp{sp_size}_native"
    config = json.loads((output / "training_config.json").read_text())
    if config["sp_size"] != sp_size:
        raise ValueError("Native replay configuration has a different SP size")
    expected_steps = list(range(config["steps"]))
    audits = []
    reference_lr: list[float] = []
    for rank in range(WORLD_SIZE):
        path = output / f"replay_inputs.rank{rank:02d}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if [row["step"] for row in rows] != expected_steps or any(
            len(row["batch_sha256"]) != config["microbatches_per_update"] * sp_size for row in rows
        ):
            raise ValueError(f"Replay input audit is incomplete for rank {rank}")
        tracker = output / f"logs/exp_tracking/rank{rank}/tracker.jsonl"
        native = [json.loads(line) for line in tracker.read_text().splitlines() if line.strip()]
        if [row["step"] for row in native] != [step + 1 for step in expected_steps]:
            raise ValueError(f"Native training metrics are incomplete for rank {rank}")
        if any(
            not math.isfinite(row[key])
            for row in native
            for key in (
                "lr",
                "grad_norm",
                "loss/reduced_total_loss",
                "loss/reduced_lm_loss",
                "loss/reduced_vq_token_loss",
                "loss/reduced_hlm_token_loss",
                "time/step_time",
            )
        ):
            raise ValueError(f"Nonfinite native training metric at rank {rank}")
        if rank == 0:
            reference_lr = [row["lr"] for row in native]
        elif [row["lr"] for row in native] != reference_lr:
            raise ValueError("Ranks used different independently computed LR schedules")
        audits.append(rows)
    if (root / f"xtuner_sp{sp_size}.jsonl").exists() or (root / "xtuner_complete.json").exists():
        raise FileExistsError("Refusing to overwrite replay results")
    for group_start in range(0, WORLD_SIZE, sp_size):
        if any(audits[rank] != audits[group_start] for rank in range(group_start + 1, group_start + sp_size)):
            raise ValueError("SP peers consumed different captured updates")
    metrics = output / "train_metrics.jsonl"
    if not metrics.exists():
        collect_metrics(output)
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    if [row["step"] for row in rows] != expected_steps:
        raise ValueError("Exported native metrics have missing or duplicate steps")
    if [row["lr"] for row in rows] != reference_lr:
        raise ValueError("Exported LR differs from the actual native training logs")
    for row in rows:
        step = row["step"]
        if row["valid_targets"] != sum(audits[rank][step]["valid_targets"] for rank in range(0, WORLD_SIZE, sp_size)):
            raise ValueError("Native targets differ from the actual replayed inputs")
        row.update(
            batch_sha256_by_rank=[
                audits[rank // sp_size * sp_size][step]["batch_sha256"][rank % sp_size :: sp_size]
                for rank in range(WORLD_SIZE)
            ],
            accumulation_steps=config["microbatches_per_update"] * sp_size,
            lr_source="xtuner_scheduler",
        )
    target = root / f"xtuner_sp{sp_size}.jsonl"
    temporary = target.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(target)
    target = root / "xtuner_complete.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "steps": expected_steps,
                "training_entry": "Trainer.fit",
                "lr_source": "xtuner_scheduler",
                "sp_size": sp_size,
            }
        )
    )
    temporary.replace(target)


def main() -> None:
    """Prepare configuration or collect metrics; training runs through the CLI."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--xtuner-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--sp-size", type=int, choices=(1, 8), default=os.environ.get("SP_SIZE", "1"))
    parser.add_argument("--hf-path", type=Path)
    parser.add_argument("--zero-stage", type=int, choices=(2, 3), default=os.environ.get("ZERO_STAGE", "3"))
    parser.add_argument(
        "--reduce-dtype",
        type=str.lower,
        choices=("fp32", "bf16"),
        default=os.environ.get("REDUCE_DTYPE", "fp32"),
        help="FSDP gradient reduction precision",
    )
    parser.add_argument("--recompute-ratio", type=float, default=os.environ.get("RECOMPUTE_RATIO", "1.0"))
    parser.add_argument(
        "--lm-head-chunk-size",
        type=int,
        default=os.environ.get("LOSS_CHUNK_SIZE", "4096"),
        help="Tokens per LM-head CE chunk",
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=env_bool("ENABLE_COMPILE", False))
    parser.add_argument(
        "--gradient-accumulation-no-sync",
        action=argparse.BooleanOptionalAction,
        default=env_bool("GRADIENT_ACCUMULATION_NO_SYNC", False),
    )
    parser.add_argument(
        "--profile-trace", action=argparse.BooleanOptionalAction, default=env_bool("ENABLE_TRACE", False)
    )
    parser.add_argument("--profile-step", type=int, default=7)
    parser.add_argument("--seed", type=int, default=os.environ.get("SEED", "1234"))
    parser.add_argument("--lr", type=float, default=os.environ.get("LR", "8e-4"))
    parser.add_argument("--min-lr", type=float, default=os.environ.get("MIN_LR", "8e-5"))
    parser.add_argument(
        "--lr-decay-style",
        type=str.lower,
        choices=("wsd", "cosine", "linear", "constant"),
        default=os.environ.get("LR_DECAY_STYLE", "wsd"),
    )
    parser.add_argument("--warmup-iters", type=int, default=os.environ.get("LR_WARMUP_ITERS", "2000"))
    parser.add_argument("--decay-iters", type=int, default=os.environ.get("LR_DECAY_ITERS", "143052"))
    parser.add_argument("--wsd-decay-iters", type=int, default=os.environ.get("LR_WSD_DECAY_ITERS", "14306"))
    parser.add_argument("--collect-metrics", action="store_true")
    args = parser.parse_args()
    if args.lm_head_chunk_size <= 0:
        parser.error("lm-head-chunk-size must be positive")
    if args.collect_metrics:
        collect_replay(args.run_dir, args.sp_size)
    else:
        prepare_replay(args)


if __name__ == "__main__":
    main()
