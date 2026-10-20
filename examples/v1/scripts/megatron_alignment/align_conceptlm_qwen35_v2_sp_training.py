#!/usr/bin/env python3
"""Compare full-parameter ConceptLM Qwen3.5 v2 training across Ulysses SP sizes.

Run from the repository root with eight GPUs::

    PYTHONPATH=. torchrun --standalone --nproc-per-node=8 \
        examples/v1/scripts/megatron_alignment/align_conceptlm_qwen35_v2_sp_training.py \
        --steps 5 --lengths 128 --output logs/ulysses-sp-training.json

Every mode reloads the same checkpoint and starts a fresh AdamW optimizer.
The global batch contains WORLD_SIZE copies of the --lengths pattern, with
different token contents. SP=1 assigns one pattern to each rank. SP=N uses N
gradient accumulation microbatches per DP group, each with one pattern split
across N SP ranks. Thus global documents, labels, valid tokens, and optimizer
settings are fixed. With --lengths 8192, SP=8 holds 1024 tokens per rank per
microbatch and accumulates eight microbatches for the same global batch of eight.

TrainEngine owns forward/backward accumulation, replicated-gradient synchronization,
gradient clipping, and optimizer updates. Loss contexts use the model's standard
build_loss_ctx_batch entry point and native chunked CE with global calibration.
Document-final labels and SP padding are ignored. FSDP averages parameter
gradients; the native autograd loss reduction supplies the matching scaling.
Grad norms count each unique FSDP gradient once and are measured before clipping.
Independent optimizer trajectories are retained across steps; no parameters
are synchronized from SP=1 after an update. Production fused kernels are used
by default; --compile additionally enables the model's torch.compile policy.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT = "/mnt/shared-storage-user/llmrazor-share/heqiushi/ncp/qwen35/ncp_qwen35_v2_2551e48_iter0000002_hf"


def parse_args() -> argparse.Namespace:
    """Parse training and comparison options.

    Returns:
        argparse.Namespace: Validated command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", type=Path, default=Path(os.environ.get("NCP_QWEN35_V2_PATH", DEFAULT_CHECKPOINT)))
    parser.add_argument("--sp-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--lengths", type=int, nargs="+", default=[128], help="document lengths per baseline DP rank")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="0 disables clipping")
    parser.add_argument("--recompute-ratio", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--ce-chunk-size", type=int, default=128)
    parser.add_argument("--loss-atol", type=float, default=1e-2)
    parser.add_argument("--loss-rtol", type=float, default=1e-2)
    parser.add_argument("--grad-norm-atol", type=float, default=1e-3)
    parser.add_argument("--grad-norm-rtol", type=float, default=5e-2)
    parser.add_argument("--output", type=Path, default=Path("logs/ulysses-sp-training.json"))
    parser.add_argument("--data-path", type=Path, help="fixed CPU batches; generated once if the file does not exist")
    parser.add_argument(
        "--tensorboard-dir", type=Path, help="TensorBoard root containing sp1, sp8 and comparison runs"
    )
    parser.add_argument("--report-only", action="store_true", help="report tolerance failures without a nonzero exit")
    args = parser.parse_args()
    if args.steps < 1 or not args.lengths or any(length < 2 for length in args.lengths):
        parser.error("--steps must be positive and every --lengths entry must be at least two")
    if args.sp_sizes[0] != 1 or len(args.sp_sizes) != len(set(args.sp_sizes)):
        parser.error("--sp-sizes must start with the SP=1 reference and contain unique values")
    if any(size < 1 for size in args.sp_sizes):
        parser.error("--sp-sizes must be positive")
    if args.ce_chunk_size < 1 or not 0 <= args.recompute_ratio <= 1:
        parser.error("--ce-chunk-size must be positive and --recompute-ratio must be in [0, 1]")
    for name in ("lr", "eps"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name} must be finite and positive")
    for name in ("weight_decay", "max_grad_norm", "loss_atol", "loss_rtol", "grad_norm_atol", "grad_norm_rtol"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if any(not 0 <= beta < 1 for beta in args.betas):
        parser.error("--betas must be in [0, 1)")
    args.data_path = args.data_path or args.output.with_suffix(".batches.pt")
    args.tensorboard_dir = args.tensorboard_dir or args.output.with_suffix(".tensorboard")
    return args


def main() -> None:
    """Run independent SP training trajectories and write per-step comparisons."""
    args = parse_args()
    _configure_kernels()

    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from xtuner.v1.utils.misc import set_deterministic

    set_deterministic()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    try:
        if not args.hf_path.is_dir():
            raise ValueError(f"checkpoint does not exist: {args.hf_path}")
        if any(world_size % size or 16 % size or 24 % size for size in args.sp_sizes):
            raise ValueError("SP sizes must divide WORLD_SIZE, the 16 GDN key heads, and the 24 MLA heads")
        batches, data_sha256 = _load_batches(args, rank, world_size)
        initial_weights = None
        results: dict[int, list[dict[str, Any]]] = {}
        report: dict[str, Any] = {
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "world_size": world_size,
            "global_documents": world_size * len(args.lengths),
            "global_valid_targets": world_size * sum(length - 1 for length in args.lengths),
            "grad_norm": "global L2 of unique parameter gradients before clipping",
            "engine": "xtuner.v1.engine.train_engine.TrainEngine",
            "runs": {},
            "comparisons": [],
            "data_sha256": data_sha256,
            "initial_weight_shards": {},
        }
        failures = 0
        for sp_size in args.sp_sizes:
            mesh = init_device_mesh("cuda", (world_size // sp_size, sp_size), mesh_dim_names=("dp", "sp"))
            metrics, weight_shards = _run_training(args, mesh["sp"], rank, world_size, batches, initial_weights)
            if initial_weights is None:
                initial_weights = weight_shards
            report["initial_weight_shards"][str(sp_size)] = weight_shards
            results[sp_size] = metrics
            report["runs"][str(sp_size)] = metrics
            if sp_size != 1:
                for reference, actual in zip(results[1], metrics, strict=True):
                    comparison = _compare_step(reference, actual, sp_size, args)
                    report["comparisons"].append(comparison)
                    failures += not comparison["passed"]
                    if rank == 0:
                        print(
                            f"COMPARE step={actual['step']} sp=1/{sp_size} "
                            f"loss={reference['loss']:.8f}/{actual['loss']:.8f} "
                            f"loss_rel_gap={comparison['loss_relative_gap']:.3%} "
                            f"grad_norm={reference['grad_norm']:.8e}/{actual['grad_norm']:.8e} "
                            f"grad_norm_rel_gap={comparison['grad_norm_relative_gap']:.3%} "
                            f"[{'PASS' if comparison['passed'] else 'FAIL'}]",
                            flush=True,
                        )
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            dist.barrier()
        if rank == 0:
            _write_comparison_events(args, report)
            print(
                f"{'PASS' if failures == 0 else 'FAIL'}: {failures} failed step comparisons; {args.output}",
                flush=True,
            )
        if failures and not args.report_only:
            raise SystemExit(1)
    finally:
        dist.destroy_process_group()


def _configure_kernels() -> None:
    from examples.v1.scripts.megatron_alignment.kernel_config import configure_training_kernels

    configure_training_kernels()


def _load_engine(args: argparse.Namespace) -> Any:
    """Build the standard engine and load the checkpoint into its model."""
    import torch

    from xtuner.v1.config import AdamWConfig, FSDPConfig
    from xtuner.v1.engine.train_engine import TrainEngine
    from xtuner.v1.loss import CELossConfig
    from xtuner.v1.model.dense.conceptlm_qwen35_9b import ConceptLMQwen35V2Config

    _seed_step(args.seed)
    engine = TrainEngine(
        model_cfg=ConceptLMQwen35V2Config(
            compile_cfg=args.compile,
            lm_loss_cfg=CELossConfig(mode="chunk", chunk_size=args.ce_chunk_size, loss_reduction="token"),
        ),
        optim_cfg=AdamWConfig(
            lr=args.lr,
            betas=tuple(args.betas),
            eps=args.eps,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            foreach=False,
            fused=False,
        ),
        fsdp_cfg=FSDPConfig(
            requires_grad=True,
            recompute_ratio=args.recompute_ratio,
            reshard_after_forward=True,
            cpu_offload=False,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            fp32_lm_head=False,
        ),
    )
    engine.from_hf(args.hf_path, strict=True)
    engine.model.train()
    return engine


def _make_documents(args: argparse.Namespace, step: int, world_size: int, vocab_size: int) -> tuple[list[Any], str]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(args.seed + step)
    documents = [
        torch.randint(2, vocab_size, (1, length), generator=generator, dtype=torch.long)
        for _ in range(world_size)
        for length in args.lengths
    ]
    return documents, _batch_digest(documents)


def _batch_digest(documents: list[Any]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(str(tuple(document.shape)).encode())
        digest.update(document.numpy().tobytes())
    return digest.hexdigest()


def _load_batches(args: argparse.Namespace, rank: int, world_size: int) -> tuple[list[list[Any]], str]:
    import torch
    import torch.distributed as dist

    metadata = {"seed": args.seed, "steps": args.steps, "lengths": args.lengths, "world_size": world_size}
    if rank == 0 and not args.data_path.exists():
        batches = [_make_documents(args, step, world_size, 248320)[0] for step in range(args.steps)]
        args.data_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.data_path.with_suffix(".tmp")
        torch.save({"metadata": metadata, "batches": batches}, temporary)
        temporary.replace(args.data_path)
    dist.barrier()
    raw = args.data_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if payload["metadata"] != metadata or len(payload["batches"]) != args.steps:
        raise ValueError(f"fixed-batch metadata does not match this run: {payload['metadata']} != {metadata}")
    fingerprints: list[Any] = [None] * world_size
    dist.all_gather_object(fingerprints, digest)
    if len(set(fingerprints)) != 1:
        raise RuntimeError(f"ranks loaded different data files: {fingerprints}")
    if rank == 0:
        print(f"FIXED_DATA path={args.data_path} sha256={digest} metadata={metadata}", flush=True)
    return payload["batches"], digest


def _weight_fingerprints(model: Any, world_size: int) -> list[str]:
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        cpu = local.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(parameter.shape)}:{cpu.dtype}:{tuple(cpu.shape)}".encode())
        if cpu.numel():
            digest.update(memoryview(cpu.numpy()).cast("B"))
    fingerprints: list[Any] = [None] * world_size
    dist.all_gather_object(fingerprints, digest.hexdigest())
    return fingerprints


def _seed_step(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_comparison_events(args: argparse.Namespace, report: dict[str, Any]) -> None:
    from torch.utils.tensorboard import SummaryWriter

    with SummaryWriter(str(args.tensorboard_dir / "comparison")) as writer:
        for comparison in report["comparisons"]:
            for name in ("loss_abs_gap", "loss_relative_gap", "grad_norm_abs_gap", "grad_norm_relative_gap"):
                writer.add_scalar(f"sp{comparison['sp_size']}/{name}", comparison[name], comparison["step"] + 1)
            writer.add_scalar(
                f"sp{comparison['sp_size']}/same_batch_and_gradient_count",
                int(comparison["same_batch_and_gradient_count"]),
                comparison["step"] + 1,
            )


def _gradient_count(parameters: list[Any]) -> int:
    import torch
    import torch.distributed as dist

    gradient_count = sum(parameter.grad is not None for parameter in parameters)
    counts = torch.tensor([gradient_count, gradient_count], device="cuda", dtype=torch.int64)
    dist.all_reduce(counts[:1], op=dist.ReduceOp.MIN)
    dist.all_reduce(counts[1:], op=dist.ReduceOp.MAX)
    if counts[0].item() != counts[1].item() or gradient_count == 0:
        raise RuntimeError(f"inconsistent or empty gradient counts: {counts.tolist()}")
    return gradient_count


def _run_training(
    args: argparse.Namespace,
    sp_mesh: Any,
    rank: int,
    world_size: int,
    batches: list[list[Any]],
    expected_initial_weights: list[str] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    import torch
    import torch.distributed as dist
    from torch.utils.tensorboard import SummaryWriter

    from xtuner.v1.data_proto import SequenceContext
    from xtuner.v1.model.base import ModelItem

    engine = _load_engine(args)
    model = engine.model
    weight_shards = _weight_fingerprints(model, world_size)
    if expected_initial_weights is not None and weight_shards != expected_initial_weights:
        raise RuntimeError("SP modes did not load bitwise identical initial parameter shards")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    sp_size = sp_mesh.size()
    first_document = (rank // sp_size) * sp_size * len(args.lengths)
    writer = None
    metrics_file = None
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(args.tensorboard_dir / f"sp{sp_size}"), flush_secs=10)
        writer.add_text("config", json.dumps(vars(args), default=str, indent=2), 0)
        writer.add_text("initial_weight_shards", json.dumps(weight_shards), 0)
        metrics_file = args.output.with_suffix(f".sp{sp_size}.jsonl").open("w")
        print(f"INITIAL_WEIGHTS sp={sp_size} shards={weight_shards}", flush=True)
        print(f"BATCH sp={sp_size} accumulation={sp_size} local_tokens={sum(args.lengths) // sp_size}", flush=True)
    rows = []
    try:
        for step in range(args.steps):
            _seed_step(args.seed + step)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            engine.optimizer.zero_grad(set_to_none=True)
            global_documents = batches[step]
            digest = _batch_digest(global_documents)
            data_batch = []
            for microbatch in range(sp_size):
                start = first_document + microbatch * len(args.lengths)
                documents = tuple(document.cuda() for document in global_documents[start : start + len(args.lengths)])
                labels = torch.cat(
                    [torch.cat((doc[:, 1:], doc.new_full((1, 1), -100)), dim=1) for doc in documents], dim=1
                )
                seq_ctx = SequenceContext.from_input_ids(documents, device="cuda")
                if sp_size > 1:
                    seq_ctx = seq_ctx.split(sp_mesh)
                data_batch.append({"seq_ctx": seq_ctx, "shifted_labels": labels})
            loss_contexts = model.build_loss_ctx_batch(data_batch, sp_mesh=sp_mesh if sp_size > 1 else None)
            engine_input = [
                ModelItem(seq_ctx=data["seq_ctx"], loss_ctx=loss_ctx)
                for data, loss_ctx in zip(data_batch, loss_contexts, strict=True)
            ]
            target_count = sum(loss_ctx["lm"].loss_kwargs.shifted_labels.ne(-100).sum() for loss_ctx in loss_contexts)
            dist.all_reduce(target_count, op=dist.ReduceOp.SUM)
            expected_targets = world_size * sum(length - 1 for length in args.lengths)
            if target_count.item() != expected_targets:
                raise RuntimeError(f"target count mismatch: {target_count.item()} != {expected_targets}")
            step_info = engine.train_step(engine_input)
            loss_value = step_info["total_loss"]
            grad_norm_tensor = engine.clip_grad_norm(do_clip=args.max_grad_norm > 0, dtype=torch.float32)
            grad_norm = grad_norm_tensor.item()
            gradient_count = _gradient_count(parameters)
            if not math.isfinite(loss_value) or not math.isfinite(grad_norm):
                raise RuntimeError(
                    f"non-finite metrics: sp={sp_size}, step={step}, loss={loss_value}, norm={grad_norm}"
                )
            clip_coefficient = min(1.0, args.max_grad_norm / (grad_norm + 1e-6)) if args.max_grad_norm else 1.0
            engine.step_optimizer(grad_norm_tensor)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            row = {
                "step": step,
                "loss": loss_value,
                "grad_norm": grad_norm,
                "clip_coefficient": clip_coefficient,
                "lr": engine.optimizer.param_groups[0]["lr"],
                "tensors_with_grad": gradient_count,
                "valid_targets": expected_targets,
                "batch_sha256": digest,
                "step_seconds": seconds,
                "global_tokens_per_second": world_size * sum(args.lengths) / seconds,
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gradient_accumulation_steps": sp_size,
                "local_sequence_tokens": engine_input[0]["seq_ctx"].input_ids.shape[1],
            }
            rows.append(row)
            if rank == 0:
                assert writer is not None and metrics_file is not None
                print(f"TRAIN sp={sp_size} " + json.dumps(row, allow_nan=False), flush=True)
                metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
                metrics_file.flush()
                for name in (
                    "loss",
                    "grad_norm",
                    "clip_coefficient",
                    "lr",
                    "step_seconds",
                    "global_tokens_per_second",
                    "peak_memory_gib",
                    "valid_targets",
                ):
                    writer.add_scalar(f"train/{name}", row[name], step + 1)
                writer.flush()
            del data_batch, loss_contexts, engine_input, step_info, seq_ctx, documents, labels
            dist.barrier()
    finally:
        if writer is not None:
            writer.close()
        if metrics_file is not None:
            metrics_file.close()
    del engine, model, parameters
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return rows, weight_shards


def _compare_step(
    reference: dict[str, Any], actual: dict[str, Any], sp_size: int, args: argparse.Namespace
) -> dict[str, Any]:
    gaps = {name: abs(actual[name] - reference[name]) for name in ("loss", "grad_norm")}
    limits = {
        "loss": args.loss_atol + args.loss_rtol * abs(reference["loss"]),
        "grad_norm": args.grad_norm_atol + args.grad_norm_rtol * abs(reference["grad_norm"]),
    }
    same_batch = all(
        actual[name] == reference[name] for name in ("step", "batch_sha256", "valid_targets", "tensors_with_grad")
    )
    return {
        "step": actual["step"],
        "sp_size": sp_size,
        "loss_abs_gap": gaps["loss"],
        "loss_relative_gap": gaps["loss"] / max(abs(reference["loss"]), 1e-30),
        "loss_limit": limits["loss"],
        "grad_norm_abs_gap": gaps["grad_norm"],
        "grad_norm_relative_gap": gaps["grad_norm"] / max(abs(reference["grad_norm"]), 1e-30),
        "grad_norm_limit": limits["grad_norm"],
        "same_batch_and_gradient_count": same_batch,
        "passed": same_batch and all(math.isfinite(gaps[name]) and gaps[name] <= limits[name] for name in gaps),
    }


if __name__ == "__main__":
    main()
