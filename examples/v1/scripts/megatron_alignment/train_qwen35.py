"""Prepare a native Trainer CLI recipe or export its completed metrics.

This script does not implement a training loop. Run the generated configuration
through xtuner.v1.train.cli.sft, which calls Trainer.fit().
"""

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kernel_config import configure_training_kernels
from runtime_requirements import require_fla_version


if TYPE_CHECKING:
    from xtuner.v1.datasets.config import DataloaderConfig
    from xtuner.v1.train import TrainerConfig


def build_trainer_config(
    config: dict[str, Any],
    *,
    dataloader_cfg: "DataloaderConfig | None" = None,
) -> "TrainerConfig":
    """Translate the independent training recipe into native configuration.

    Args:
        config (dict[str, Any]): Saved launch arguments; no reference batch/state.
        dataloader_cfg (DataloaderConfig | None): Input adapter; otherwise build the independent data pipeline.

    Returns:
        TrainerConfig: Configuration consumed by the standard CLI and Trainer.
    """
    configure_training_kernels()
    require_fla_version()

    import torch
    from qwen35_training_observer import ConceptLMTrainingObserver

    from xtuner.v1.config import FSDPConfig, LRConfig, MuonConfig
    from xtuner.v1.datasets import PretrainTokenizeFunctionConfig
    from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
    from xtuner.v1.loss import CELossConfig, ConceptLMLossConfig
    from xtuner.v1.model.dense.conceptlm_qwen35_9b import ConceptLMQwen35V2Config
    from xtuner.v1.train import TrainerConfig
    from xtuner.v1.train.trainer import HooksConfig

    torch.set_float32_matmul_precision("medium")
    if dataloader_cfg is None:
        indexed = config["data_format"] == "indexed"
        if indexed:
            datasets = [
                {
                    "dataset": DatasetConfig(
                        anno_path=config["data_path"],
                        class_name="MegatronIndexedDataset",
                        name="tokenized",
                        sequence_length=config["sequence_length"] + 1,
                        enable_sequential_sampler=False,
                    ),
                    "tokenize_fn": PretrainTokenizeFunctionConfig(add_eos_token=False, add_bos_token=False),
                }
            ]
        else:
            files = json.loads(Path(config["text_manifest"]).read_text())
            datasets = [
                {
                    "dataset": DatasetConfig(
                        anno_path=item["normalized"],
                        class_name="JsonlDataset",
                        name=f"text_{index}",
                        cache_dir=config["data_cache"],
                        enable_sequential_sampler=False,
                    ),
                    "tokenize_fn": PretrainTokenizeFunctionConfig(add_eos_token=True, add_bos_token=False),
                }
                for index, item in enumerate(files)
            ]
        dataloader_cfg = DataloaderConfig(
            dataset_config_list=datasets,
            pack_level="none" if indexed else "hard",
            global_pack=True,
            pack_max_length=config["sequence_length"] + 1,
            pack_to_max_length=True,
            collator="single_sequence_sft_llm_collator",
            shuffle=True,
            group_by_length=False,
            num_workers=config["num_workers"],
            pack_workers=config["pack_workers"],
        )
    lm_loss = CELossConfig(mode="chunk", chunk_size=128, loss_reduction="token")
    return TrainerConfig(
        load_from=config["hf_path"],
        tokenizer_path=config["tokenizer_path"],
        strict_load=True,
        model_cfg=ConceptLMQwen35V2Config(
            compile_cfg=config["compile"],
            concept_loss_cfg=ConceptLMLossConfig(),
            lm_loss_cfg=lm_loss,
        ),
        loss_cfg=lm_loss,
        optim_cfg=MuonConfig(
            megatron_compatible=True,
            lr=config["lr"],
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            momentum=0.95,
            nesterov=True,
            max_grad_norm=1.0,
            adjust_lr="rms_norm",
            extra_scale_factor=0.2,
            clip_grad_mode="all",
            one_d_weight_decay=0.0,
            embedding_weight_decay=0.1,
            adamw_foreach=False,
        ),
        fsdp_cfg=FSDPConfig(
            requires_grad=True,
            recompute_ratio=config["recompute_ratio"],
            reshard_after_forward=config["zero_stage"] == 3,
            cpu_offload=False,
            gradient_accumulation_no_sync=config["gradient_accumulation_no_sync"],
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            fp32_lm_head=False,
        ),
        dataloader_cfg=dataloader_cfg,
        lr_cfg=LRConfig(
            lr_type=config["lr_decay_style"],
            lr_min=config["min_lr"],
            warmup_steps=config["warmup_iters"],
            decay_steps=config["decay_iters"],
            wsd_decay_steps=config["wsd_decay_iters"] if config["lr_decay_style"] == "wsd" else None,
        ),
        global_batch_size=config["global_batch_size"],
        intra_layer_micro_batch=1,
        sp_size=config["sp_size"],
        total_step=config["steps"],
        seed=config["seed"],
        work_dir=config["run_dir"],
        log_dir=str(Path(config["run_dir"]) / "logs"),
        exp_tracker="both",
        debug_skip_save=True,
        grad_norm_dtype="float32",
        hooks_config=HooksConfig(after_train_step=ConceptLMTrainingObserver()),
        profile_step=[config["profile_step"] - 1, config["profile_step"]] if config["profile_trace"] else None,
    )


def collect_metrics(run_dir: Path) -> None:
    """Validate and export native step records without rerunning training.

    Native metric names are retained; aliases only normalize step/time/TPS.
    This is not a strict batch-replay completion certificate.

    Args:
        run_dir (Path): Completed CLI run with a full-precision native tracker.
    """
    config = json.loads((run_dir / "training_config.json").read_text())
    tracker = run_dir / "logs/exp_tracking/rank0/tracker.jsonl"
    rows = [json.loads(line) for line in tracker.read_text().splitlines() if line.strip()]
    if [row["step"] for row in rows] != list(range(1, config["steps"] + 1)):
        raise ValueError("Native training step records are incomplete or duplicated")
    for row in rows:
        if not all(math.isfinite(row[key]) for key in ("lr", "grad_norm", "time/step_time")):
            raise ValueError("Nonfinite native training metric")
        if row["time/step_time"] <= 0:
            raise ValueError("Invalid iteration time")
        row.update(
            total_loss=row["loss/reduced_total_loss"],
            lm_loss=row["loss/reduced_lm_loss"],
            vq_token_loss=row["loss/reduced_vq_token_loss"],
            hlm_token_loss=row["loss/reduced_hlm_token_loss"],
            valid_targets=row["loss/valid_targets"],
            step=row["step"] - 1,
            seconds=row["time/step_time"],
            tokens_per_second_per_gpu=config["global_batch_size"]
            * config["sequence_length"]
            / row["time/step_time"]
            / 8,
            sp_size=config["sp_size"],
            dp_size=8 // config["sp_size"],
            timing_scope="native_trainer_step_including_input_preparation",
        )
    output = run_dir / "train_metrics.jsonl"
    if output.exists():
        raise FileExistsError(output)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run_dir / "native_training_complete.json").write_text(json.dumps({"steps": [row["step"] for row in rows]}))
    (run_dir / "status").write_text("COMPLETED\n")


def main() -> None:
    """Prepare configuration on CPU; use the standard CLI for GPU training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--xtuner-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--recompute-ratio", type=float, default=1.0)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the model default torch.compile configuration",
    )
    parser.add_argument("--zero-stage", type=int, choices=(2, 3), default=3)
    parser.add_argument("--sp-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--gradient-accumulation-no-sync", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile-trace", action=argparse.BooleanOptionalAction, default=False)
    base = Path("/mnt/shared-storage-user/llmrazor-share/heqiushi")
    parser.add_argument("--hf-path", type=Path, default=base / "ncp/qwen35/ncp_qwen35_v2_2551e48_iter0000002_hf")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--min-lr", type=float, default=8e-5)
    parser.add_argument("--warmup-iters", type=int, default=2000)
    parser.add_argument("--decay-iters", type=int, default=143052)
    parser.add_argument("--wsd-decay-iters", type=int, default=14306)
    parser.add_argument("--text-manifest", type=Path)
    parser.add_argument("--data-format", choices=("indexed", "text"), default="indexed")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(
            "/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/data/"
            "dolma3_mix-150B-1025/megatron_qwen35_8192/indexed"
        ),
    )
    parser.add_argument("--data-cache", type=Path, default=base / "xtuner_text_data_cache/tokenization")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path(
            "/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/data/"
            "dolma3_mix-150B-1025/megatron_qwen35_8192/tokenizer/Qwen3.5-0.8B-Base"
        ),
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pack-workers", type=int, default=1)
    parser.add_argument(
        "--lr-decay-style",
        type=str.lower,
        choices=("wsd", "cosine", "linear", "constant"),
        default="wsd",
    )
    parser.add_argument(
        "--profile-step", type=int, default=7, help="First one-based update of the two-step profiling window"
    )
    parser.add_argument("--collect-metrics", action="store_true", help="Export native logs after CLI training")
    args = parser.parse_args()
    if args.collect_metrics:
        collect_metrics(args.run_dir)
        return
    # FLA is validated by the GPU launcher and build_trainer_config, not by
    # CPU-only recipe preparation or postprocessing.
    if (
        args.steps <= 0
        or args.global_batch_size <= 0
        or args.sequence_length <= 0
        or args.sequence_length % (4 * args.sp_size)
    ):
        parser.error("steps, GBS, sequence length must be positive; sequence length must be divisible by 4 * SP size")
    if args.global_batch_size % (8 // args.sp_size):
        parser.error("GBS must be divisible by the data parallel size")
    if args.profile_step <= 0:
        parser.error("profile-step must be positive")
    if args.data_format == "text" and args.text_manifest is None:
        parser.error("text input requires --text-manifest")
    if args.run_dir.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.run_dir}")
    if args.warmup_iters < 0 or args.decay_iters <= args.warmup_iters:
        parser.error("Require 0 <= warmup-iters < decay-iters")
    if not 0 <= args.min_lr <= args.lr or args.lr <= 0:
        parser.error("Require 0 <= min-lr <= lr and lr > 0")
    if args.lr_decay_style == "wsd" and not 0 < args.wsd_decay_iters <= args.decay_iters - args.warmup_iters:
        parser.error("Require 0 < wsd-decay-iters <= decay-iters - warmup-iters")
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    args.run_dir.mkdir(parents=True)
    config_path = args.run_dir.resolve() / "training_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    recipe = (
        "import json\nimport sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
        "from train_qwen35 import build_trainer_config\n"
        f"trainer = build_trainer_config(json.loads(Path({str(config_path)!r}).read_text()))\n"
    )
    (args.run_dir / "trainer_config.py").write_text(recipe)
    (args.run_dir / "status").write_text("PREPARED\n")
    print(f"Native CLI config: {args.run_dir / 'trainer_config.py'}", flush=True)
    print("No captured batches, reference LR or reference gradients are used for training.", flush=True)


if __name__ == "__main__":
    main()
