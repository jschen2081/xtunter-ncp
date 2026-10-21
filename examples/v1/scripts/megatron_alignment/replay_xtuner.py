"""Prepare native SP1/SP8 captured-input training with independently computed LR."""

import argparse
import json
import math
import os
from pathlib import Path

from training_environment import capture_training_environment
from recorded_batch_source import WORLD_SIZE, capture_metadata
from train_qwen35 import collect_metrics, env_bool


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
    scripts = args.xtuner_root.resolve() / "examples/v1/scripts/megatron_alignment"
    recipe = (
        "import json\nimport sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(scripts)!r})\n"
        "from training_environment import validate_training_environment\n"
        f"_recipe = json.loads(Path({str(config_path)!r}).read_text())\n"
        "validate_training_environment(_recipe.get(\"kernel_env\"))\n"
        "from replay_trainer_config import build_replay_config\n"
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
        "--reduce-dtype", type=str.lower, choices=("fp32", "bf16"),
        default=os.environ.get("REDUCE_DTYPE", "fp32"), help="FSDP gradient reduction precision",
    )
    parser.add_argument("--recompute-ratio", type=float, default=os.environ.get("RECOMPUTE_RATIO", "1.0"))
    parser.add_argument(
        "--lm-head-chunk-size", type=int, default=os.environ.get("LOSS_CHUNK_SIZE", "4096"),
        help="Tokens per LM-head CE chunk",
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=env_bool("ENABLE_COMPILE", False))
    parser.add_argument(
        "--gradient-accumulation-no-sync", action=argparse.BooleanOptionalAction,
        default=env_bool("GRADIENT_ACCUMULATION_NO_SYNC", False),
    )
    parser.add_argument("--profile-trace", action=argparse.BooleanOptionalAction, default=env_bool("ENABLE_TRACE", False))
    parser.add_argument("--profile-step", type=int, default=7)
    parser.add_argument("--seed", type=int, default=os.environ.get("SEED", "1234"))
    parser.add_argument("--lr", type=float, default=os.environ.get("LR", "8e-4"))
    parser.add_argument("--min-lr", type=float, default=os.environ.get("MIN_LR", "8e-5"))
    parser.add_argument(
        "--lr-decay-style", type=str.lower, choices=("wsd", "cosine", "linear", "constant"),
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
