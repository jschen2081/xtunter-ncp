#!/usr/bin/env python3
"""Align several 9B ConceptLM Qwen3.5 v2 training steps with its HF export.

The XTuner and HF models run sequentially so two 9B models never occupy rank
0 at the same time. XTuner first writes one CPU snapshot per step, then rank 0
replays the exact batches with HF and compares forward values, selected
parameter gradients, updated FP32 master parameters, and AdamW states.

By default only four small norm weights are trainable. They sit near the start
of each tower, so their gradients exercise almost the complete downstream
autograd graph without allocating optimizer states for all 9B parameters.
Pass ``--full-parameters`` to train every parameter and compare only the loss
and global gradient norm. This avoids materializing full-model CPU snapshots.

Launch from the repository root::

    torchrun --standalone --nproc-per-node=8 \
        examples/v1/scripts/align_conceptlm_qwen35_v2_hf_training.py \
        --hf-path "$NCP_QWEN35_V2_PATH" --steps 3 --length 8
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# This must be set before importing xtuner.v1/fla. It pins Triton autotuning for
# the controlled parity path; production training does not set this switch.
os.environ.setdefault("XTUNER_DETERMINISTIC", "true")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import align_conceptlm_qwen35_v2_hf as forward_alignment  # noqa: E402


DEFAULT_PARAMETERS = (
    "encoder.layers.0.input_layernorm.weight",
    "predictor.layers.0.input_layernorm.weight",
    "decoder.layers.0.input_layernorm.weight",
    "decoder.final_norm.weight",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns:
        argparse.Namespace: Parsed training and comparison options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", default=forward_alignment.DEFAULT_HF)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--recompute-ratio", type=float, default=1.0)
    parser.add_argument(
        "--kernel-path",
        choices=("controlled", "production"),
        default="controlled",
        help="controlled: eager reference kernels; production: compile and fused production kernels",
    )
    parser.add_argument("--parameters", nargs="+", default=list(DEFAULT_PARAMETERS))
    parser.add_argument(
        "--full-parameters",
        action="store_true",
        help="train all parameters and compare per-step loss/global grad norm only",
    )
    parser.add_argument("--logit-atol", type=float, default=5e-3)
    parser.add_argument("--loss-atol", type=float, default=1e-4)
    parser.add_argument("--grad-atol", type=float, default=0.0)
    parser.add_argument("--parameter-atol", type=float, default=0.0)
    parser.add_argument("--optimizer-atol", type=float, default=0.0)
    parser.add_argument("--grad-norm-atol", type=float, default=1e-3)
    parser.add_argument("--grad-norm-rtol", type=float, default=5e-2)
    return parser.parse_args()


def validate_options(args: argparse.Namespace) -> None:
    """Validate options before loading a 9B model.

    Args:
        args (argparse.Namespace): Parsed command-line options.
    """
    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}")
    forward_alignment.validate_lengths([args.length])
    if args.lr <= 0:
        raise ValueError(f"--lr must be positive, got {args.lr}")
    if len(args.betas) != 2 or not all(0.0 <= beta < 1.0 for beta in args.betas):
        raise ValueError(f"--betas must contain two values in [0, 1), got {args.betas}")
    if args.eps <= 0:
        raise ValueError(f"--eps must be positive, got {args.eps}")
    if args.weight_decay < 0:
        raise ValueError(f"--weight-decay must be non-negative, got {args.weight_decay}")
    if not 0.0 <= args.recompute_ratio <= 1.0:
        raise ValueError(f"--recompute-ratio must be in [0, 1], got {args.recompute_ratio}")
    if not args.parameters or len(args.parameters) != len(set(args.parameters)):
        raise ValueError("--parameters must contain at least one unique XTuner parameter name")
    tolerances = {
        "--logit-atol": args.logit_atol,
        "--loss-atol": args.loss_atol,
        "--grad-atol": args.grad_atol,
        "--parameter-atol": args.parameter_atol,
        "--optimizer-atol": args.optimizer_atol,
        "--grad-norm-atol": getattr(args, "grad_norm_atol", 1e-3),
        "--grad-norm-rtol": getattr(args, "grad_norm_rtol", 5e-2),
    }
    invalid = {name: value for name, value in tolerances.items() if value < 0 or not math.isfinite(value)}
    if invalid:
        raise ValueError(f"tolerances must be finite and non-negative, got {invalid}")


def resolve_parameter_mapping(model: Any, selected_names: list[str]) -> dict[str, str]:
    """Resolve selected XTuner parameter names to unique HF names.

    Args:
        model (Any): XTuner model exposing ``named_parameters`` and ``to_hf_key_list``.
        selected_names (list[str]): Exact XTuner parameter names requested by the user.

    Returns:
        dict[str, str]: XTuner-to-HF parameter-name mapping in requested order.
    """
    available = dict(model.named_parameters())
    missing = [name for name in selected_names if name not in available]
    if missing:
        suggestions = {
            name: [candidate for candidate in available if name.split(".")[-1] in candidate][:5]
            for name in missing
        }
        raise ValueError(f"unknown XTuner parameters {missing}; possible matches: {suggestions}")
    mapping: dict[str, str] = {}
    for name in selected_names:
        hf_names = model.to_hf_key_list(name)
        if len(hf_names) != 1:
            raise ValueError(f"training alignment requires a 1:1 HF mapping for {name}, got {hf_names}")
        mapping[name] = hf_names[0]
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"selected XTuner parameters do not map to unique HF parameters: {mapping}")
    return mapping


def tensor_max_abs(left: Any, right: Any) -> float:
    """Return the maximum absolute FP32 difference, failing closed on shape mismatch.

    Args:
        left (Any): First tensor-like value.
        right (Any): Second tensor-like value.

    Returns:
        float: Maximum absolute difference, or infinity for incompatible shapes/non-finite values.
    """
    import torch

    left_tensor = torch.as_tensor(left).detach().float().cpu()
    right_tensor = torch.as_tensor(right).detach().float().cpu()
    if left_tensor.shape != right_tensor.shape:
        return math.inf
    if left_tensor.numel() == 0:
        return 0.0
    gap = (left_tensor - right_tensor).abs().max().item()
    return gap if math.isfinite(gap) else math.inf


def gate_step_metrics(metrics: dict[str, float], tolerances: dict[str, float]) -> list[str]:
    """Return all metric failures for one training step.

    Args:
        metrics (dict[str, float]): Metric names and observed maximum absolute differences.
        tolerances (dict[str, float]): Metric names and accepted absolute differences.

    Returns:
        list[str]: Human-readable failures; empty means the step passed.
    """
    failures = []
    for name, limit in tolerances.items():
        value = metrics[name]
        if not math.isfinite(value):
            failures.append(f"{name} is not finite")
        elif value > limit:
            failures.append(f"{name} {value:.3e}>{limit:.3e}")
    return failures


def _set_trainable_parameters(model: Any, selected_names: list[str]) -> dict[str, Any]:
    selected = set(selected_names)
    parameters = dict(model.named_parameters())
    missing = selected.difference(parameters)
    if missing:
        raise ValueError(f"parameters disappeared before sharding: {sorted(missing)}")
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in selected)
    return {name: parameters[name] for name in selected_names}


def _full_cpu(tensor: Any) -> Any:
    import torch
    from torch.distributed.tensor import DTensor

    value = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
    return value.detach().float().cpu().clone()


def _capture_xtuner_step(
    gradients: dict[str, Any],
    master_parameters: dict[str, Any],
    optimizer: Any,
    logits: Any,
    loss: Any,
    rank: int,
) -> dict[str, Any] | None:
    # Every rank must enter DTensor.full_tensor collectives in the same order.
    captured_gradients = {name: _full_cpu(gradient) for name, gradient in gradients.items()}
    parameters = {name: _full_cpu(parameter) for name, parameter in master_parameters.items()}
    optimizer_states: dict[str, dict[str, Any]] = {}
    for name, parameter in master_parameters.items():
        state = optimizer.state[parameter]
        optimizer_states[name] = {
            key: _full_cpu(value) for key, value in state.items() if key in {"step", "exp_avg", "exp_avg_sq"}
        }
    if rank != 0:
        return None
    return {
        "logits": logits.detach().cpu(),
        "loss": loss.detach().float().cpu(),
        "gradients": captured_gradients,
        "parameters": parameters,
        "optimizer": optimizer_states,
    }


def _load_xtuner_training_model(
    hf_path: Path,
    rank: int,
    selected_names: list[str],
    recompute_ratio: float,
    compile_model: bool,
    full_parameters: bool = False,
) -> tuple[Any, dict[str, Any], dict[str, str]]:
    import torch

    from xtuner.v1.config import FSDPConfig
    from xtuner.v1.model.base import TorchCompileOption
    from xtuner.v1.model.dense.conceptlm_qwen35_v2 import ConceptLMQwen35V2Config

    diagnostic_targets = os.environ.get("NCP_DIAG_COMPILE_TARGETS", "")
    if diagnostic_targets:
        compile_options = (
            {"emulate_precision_casts": True}
            if os.environ.get("NCP_DIAG_EMULATE_PRECISION_CASTS") == "1"
            else {}
        )
        compile_model = {
            target: TorchCompileOption(fullgraph=True, options=compile_options)
            for target in diagnostic_targets.split(",")
            if target
        }
        if rank == 0:
            print(f"diagnostic compile targets: {list(compile_model)}", flush=True)
    with torch.device("meta"):
        model = ConceptLMQwen35V2Config(compile_cfg=compile_model).build()
    if full_parameters:
        mapping = {}
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        mapping = resolve_parameter_mapping(model, selected_names)
        _set_trainable_parameters(model, selected_names)
    model.fully_shard(
        FSDPConfig(
            requires_grad=True,
            recompute_ratio=recompute_ratio,
            reshard_after_forward=True,
            cpu_offload=False,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            fp32_lm_head=False,
        )
    )
    model.train()
    loaded, unloaded, missing = model.from_hf(hf_path, strict=True)
    if rank == 0:
        print(f"xtuner from_hf: loaded={len(loaded)} unloaded={len(unloaded)} missing={len(missing)}")
        if full_parameters:
            print(f"trainable parameters: all {sum(1 for _ in model.parameters())}")
        else:
            print("trainable parameter mapping")
            for xtuner_name, hf_name in mapping.items():
                print(f"  {xtuner_name} -> {hf_name}")
    sharded_parameters = dict(model.named_parameters())
    parameters_by_original_name: dict[str, Any] = {}
    for sharded_name, parameter in sharded_parameters.items():
        original_name = sharded_name.replace("._checkpoint_wrapped_module", "")
        if original_name in parameters_by_original_name:
            raise RuntimeError(f"duplicate parameter after removing checkpoint wrapper: {original_name}")
        parameters_by_original_name[original_name] = parameter
    effective_names = list(parameters_by_original_name) if full_parameters else selected_names
    missing_after_sharding = set(effective_names).difference(parameters_by_original_name)
    if missing_after_sharding:
        raise RuntimeError(
            f"parameters disappeared after sharding/checkpoint wrapping: {sorted(missing_after_sharding)}"
        )
    selected_parameters = {name: parameters_by_original_name[name] for name in effective_names}
    return model, selected_parameters, mapping


def _build_optimizer(parameters: Any, args: argparse.Namespace) -> Any:
    import torch

    return torch.optim.AdamW(
        parameters,
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
        foreach=False,
        fused=False,
    )


def _global_grad_norm(parameters: Any, distributed: bool) -> tuple[float, int]:
    """Compute an L2 norm without gathering sharded full-model gradients."""
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, Replicate, Shard

    total_square = torch.zeros((), device="cuda", dtype=torch.float64)
    gradient_count = 0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient_count += 1
        if isinstance(gradient, DTensor):
            unsupported = [
                placement
                for placement in gradient.placements
                if not isinstance(placement, (Shard, Replicate))
            ]
            if unsupported:
                raise RuntimeError(f"unsupported DTensor gradient placements: {gradient.placements}")
            contribution = gradient.to_local().double().square().sum()
            for mesh_dim, placement in enumerate(gradient.placements):
                if isinstance(placement, Replicate):
                    contribution /= gradient.device_mesh.size(mesh_dim)
            total_square += contribution
        else:
            contribution = gradient.double().square().sum()
            if distributed:
                contribution /= dist.get_world_size()
            total_square += contribution
    if distributed:
        dist.all_reduce(total_square, op=dist.ReduceOp.SUM)
        count_tensor = torch.tensor(gradient_count, device="cuda", dtype=torch.int64)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.MAX)
        gradient_count = int(count_tensor.item())
    return total_square.sqrt().item(), gradient_count


def _run_xtuner_full_steps(
    model: Any,
    parameters: dict[str, Any],
    batches: list[Any],
    output_directory: Path,
    rank: int,
    args: argparse.Namespace,
) -> None:
    import torch
    import torch.distributed as dist

    optimizer = _build_optimizer(parameters.values(), args)
    for step, input_ids in enumerate(batches):
        torch.manual_seed(args.seed + step)
        torch.cuda.manual_seed_all(args.seed + step)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_alignment.xtuner_train_logits(model, input_ids.cuda())
        loss, _ = forward_alignment.mean_ce(logits, input_ids.cuda())
        loss.backward()
        grad_norm, gradient_count = _global_grad_norm(parameters.values(), distributed=True)
        if rank == 0:
            torch.save(
                {"loss": loss.detach().float().cpu(), "grad_norm": grad_norm},
                output_directory / f"step-{step:04d}.pt",
            )
            print(
                f"xtuner step={step} loss={loss.detach().float().item():.8f}"
                f" grad_norm={grad_norm:.8e} tensors_with_grad={gradient_count}",
                flush=True,
            )
        optimizer.step()
        del logits, loss
        dist.barrier()
    del optimizer


def _run_hf_full_steps(
    hf_path: Path,
    batches: list[Any],
    output_directory: Path,
    args: argparse.Namespace,
) -> int:
    import torch

    overrides = {}
    for name in (
        "NCP_Q35V2_FA3",
        "NCP_Q35V2_FLA_L2",
        "NCP_Q35V2_GDN_TE_OUT_NORM",
    ):
        diagnostic_name = f"NCP_DIAG_{name}"
        if diagnostic_name in os.environ:
            os.environ[name] = os.environ[diagnostic_name]
            overrides[name] = os.environ[name]
    print(f"HF diagnostic overrides: {overrides}", flush=True)
    model = forward_alignment.load_hf(hf_path)
    forward_alignment.relax_hf_fla_pin(model)
    forward_alignment.relax_hf_mla_pin(model)
    forward_alignment.relax_hf_te_runtime(model)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = _build_optimizer(model.parameters(), args)
    failed_steps = 0
    for step, input_ids in enumerate(batches):
        torch.manual_seed(args.seed + step)
        torch.cuda.manual_seed_all(args.seed + step)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_alignment.hf_train_logits(model, input_ids.cuda())
        loss, _ = forward_alignment.mean_ce(logits, input_ids.cuda())
        loss.backward()
        grad_norm, gradient_count = _global_grad_norm(model.parameters(), distributed=False)
        missing_gradients = [name for name, parameter in model.named_parameters() if parameter.grad is None]
        print(
            f"HF missing gradients ({len(missing_gradients)}): {missing_gradients}",
            flush=True,
        )
        reference = torch.load(
            output_directory / f"step-{step:04d}.pt", map_location="cpu", weights_only=True
        )
        xtuner_loss = float(reference["loss"])
        xtuner_grad_norm = float(reference["grad_norm"])
        hf_loss = loss.detach().float().item()
        loss_gap = abs(hf_loss - xtuner_loss)
        grad_norm_gap = abs(grad_norm - xtuner_grad_norm)
        grad_norm_relative_gap = grad_norm_gap / max(abs(xtuner_grad_norm), 1e-30)
        failures = []
        if not math.isfinite(loss_gap) or loss_gap > args.loss_atol:
            failures.append(f"loss {loss_gap:.3e}>{args.loss_atol:.3e}")
        grad_norm_limit = args.grad_norm_atol + args.grad_norm_rtol * abs(xtuner_grad_norm)
        if not math.isfinite(grad_norm_gap) or grad_norm_gap > grad_norm_limit:
            failures.append(
                f"grad_norm {grad_norm_gap:.3e}>{grad_norm_limit:.3e}"
            )
        status = "PASS" if not failures else "FAIL " + "; ".join(failures)
        print(
            f"step={step} xtuner_loss={xtuner_loss:.8f} hf_loss={hf_loss:.8f}"
            f" loss_gap={loss_gap:.3e} xtuner_grad_norm={xtuner_grad_norm:.8e}"
            f" hf_grad_norm={grad_norm:.8e} grad_norm_gap={grad_norm_gap:.3e}"
            f" grad_norm_rel_gap={grad_norm_relative_gap:.3%}"
            f" tensors_with_grad={gradient_count} [{status}]",
            flush=True,
        )
        failed_steps += bool(failures)
        optimizer.step()
        del logits, loss, reference
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return failed_steps


def _run_xtuner_steps(
    model: Any,
    selected_parameters: dict[str, Any],
    batches: list[Any],
    output_directory: Path,
    rank: int,
    args: argparse.Namespace,
) -> None:
    import torch
    import torch.distributed as dist

    masters = _build_master_parameters(selected_parameters)
    optimizer = _build_optimizer(masters.values(), args)
    for step, input_ids in enumerate(batches):
        torch.manual_seed(args.seed + step)
        torch.cuda.manual_seed_all(args.seed + step)
        for parameter in selected_parameters.values():
            parameter.grad = None
        optimizer.zero_grad(set_to_none=True)
        logits = forward_alignment.xtuner_train_logits(model, input_ids.cuda())
        loss, _ = forward_alignment.mean_ce(logits, input_ids.cuda())
        loss.backward()
        gradients = _copy_model_grads_and_step(selected_parameters, masters, optimizer)
        snapshot = _capture_xtuner_step(gradients, masters, optimizer, logits, loss, rank)
        if rank == 0:
            assert snapshot is not None
            import torch

            torch.save(snapshot, output_directory / f"step-{step:04d}.pt")
            print(f"xtuner step={step} loss={loss.detach().float().item():.8f}")
        del logits, loss, gradients, snapshot
        dist.barrier()


def _build_master_parameters(selected_parameters: dict[str, Any]) -> dict[str, Any]:
    import torch

    return {
        name: torch.nn.Parameter(parameter.detach().float().clone(), requires_grad=True)
        for name, parameter in selected_parameters.items()
    }


def _prepare_hf_training(
    model: Any, mapping: dict[str, str], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    import torch

    hf_parameters = dict(model.named_parameters())
    missing = set(mapping.values()).difference(hf_parameters)
    if missing:
        raise ValueError(f"HF model is missing selected parameters: {sorted(missing)}")
    selected_hf = {xt_name: hf_parameters[hf_name] for xt_name, hf_name in mapping.items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    masters = _build_master_parameters(selected_hf)
    for parameter in selected_hf.values():
        parameter.requires_grad_(True)
    optimizer = _build_optimizer(masters.values(), args)
    return selected_hf, masters, optimizer


def _copy_model_grads_and_step(
    selected_parameters: dict[str, Any], masters: dict[str, Any], optimizer: Any
) -> dict[str, Any]:
    gradients = {}
    for name, parameter in selected_parameters.items():
        if parameter.grad is None:
            raise RuntimeError(f"parameter {name} did not receive a gradient")
        gradient = parameter.grad.detach().float()
        master = masters[name]
        if gradient.numel() != master.numel():
            raise RuntimeError(f"HF/master shape mismatch for {name}: {gradient.shape} vs {master.shape}")
        master.grad = gradient.reshape_as(master).clone()
        gradients[name] = master.grad.detach().clone()
    optimizer.step()
    with __import__("torch").no_grad():
        for name, parameter in selected_parameters.items():
            parameter.copy_(masters[name].reshape_as(parameter).to(dtype=parameter.dtype))
    return gradients


# Kept as a narrow compatibility alias for the CPU tests and existing callers.
_copy_master_grads_and_step = _copy_model_grads_and_step


def _compare_hf_step(
    reference: dict[str, Any],
    logits: Any,
    loss: Any,
    gradients: dict[str, Any],
    masters: dict[str, Any],
    optimizer: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[str]]:
    metrics = {
        "logit": tensor_max_abs(logits, reference["logits"]),
        "loss": tensor_max_abs(loss, reference["loss"]),
        "gradient": max(tensor_max_abs(gradients[name], reference["gradients"][name]) for name in masters),
        "parameter": max(
            tensor_max_abs(master.detach(), reference["parameters"][name]) for name, master in masters.items()
        ),
        "optimizer": 0.0,
    }
    for name, master in masters.items():
        hf_state = optimizer.state[master]
        reference_state = reference["optimizer"][name]
        if set(hf_state) != set(reference_state):
            metrics["optimizer"] = math.inf
            break
        metrics["optimizer"] = max(
            metrics["optimizer"],
            *(tensor_max_abs(hf_state[key], reference_state[key]) for key in hf_state),
        )
    tolerances = {
        "logit": args.logit_atol,
        "loss": args.loss_atol,
        "gradient": args.grad_atol,
        "parameter": args.parameter_atol,
        "optimizer": args.optimizer_atol,
    }
    return metrics, gate_step_metrics(metrics, tolerances)


def _run_hf_steps(
    hf_path: Path,
    mapping: dict[str, str],
    batches: list[Any],
    output_directory: Path,
    args: argparse.Namespace,
) -> int:
    import torch

    model = forward_alignment.load_hf(hf_path)
    forward_alignment.relax_hf_fla_pin(model)
    forward_alignment.relax_hf_mla_pin(model)
    forward_alignment.relax_hf_te_runtime(model)
    model.train()
    selected_hf, masters, optimizer = _prepare_hf_training(model, mapping, args)
    failed_steps = 0
    for step, input_ids in enumerate(batches):
        torch.manual_seed(args.seed + step)
        torch.cuda.manual_seed_all(args.seed + step)
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_alignment.hf_train_logits(model, input_ids.cuda())
        loss, _ = forward_alignment.mean_ce(logits, input_ids.cuda())
        loss.backward()
        gradients = _copy_model_grads_and_step(selected_hf, masters, optimizer)
        reference = torch.load(output_directory / f"step-{step:04d}.pt", map_location="cpu", weights_only=True)
        metrics, failures = _compare_hf_step(reference, logits, loss, gradients, masters, optimizer, args)
        status = "PASS" if not failures else "FAIL " + "; ".join(failures)
        print(
            f"step={step} hf_loss={loss.detach().float().item():.8f}"
            f" logit={metrics['logit']:.3e} loss={metrics['loss']:.3e}"
            f" grad={metrics['gradient']:.3e} parameter={metrics['parameter']:.3e}"
            f" optimizer={metrics['optimizer']:.3e} [{status}]"
        )
        failed_steps += bool(failures)
        del logits, loss, gradients, reference
    del model, selected_hf, masters, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return failed_steps


def main() -> None:
    """Run sequential XTuner-FSDP and HF multi-step training alignment."""
    forward_alignment.configure_hf_modules_cache()

    import torch
    import torch.distributed as dist

    args = parse_args()
    validate_options(args)
    if not torch.cuda.is_available():
        raise SystemExit("this script needs CUDA GPUs")
    rank, world = forward_alignment.setup_distributed()
    if world != 8:
        dist.destroy_process_group()
        raise SystemExit(f"XTuner FSDP expects 8 ranks, found {world}")
    hf_path = forward_alignment.require_hf_dir(Path(args.hf_path))
    mla_kernel, gdn_kernel = forward_alignment.configure_xtuner_kernels(args.kernel_path)
    if rank == 0:
        print(f"kernel_path={args.kernel_path}")
        print(f"  XTuner MLA: {mla_kernel}")
        print(f"  XTuner GDN: {gdn_kernel}")
        print(f"  XTuner torch.compile: {args.kernel_path == 'production'}")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    batches = [
        torch.randint(0, 248320, (1, args.length), generator=generator, dtype=torch.long)
        for _ in range(args.steps)
    ]
    shared_directory = [tempfile.mkdtemp(prefix="ncp-qwen35-v2-training-") if rank == 0 else None]
    dist.broadcast_object_list(shared_directory, src=0)
    output_directory = Path(shared_directory[0])

    result: list[Any] = [0, None]
    try:
        model, selected_parameters, mapping = _load_xtuner_training_model(
            hf_path,
            rank,
            args.parameters,
            args.recompute_ratio,
            compile_model=False,
            full_parameters=args.full_parameters,
        )
        if args.full_parameters:
            _run_xtuner_full_steps(
                model, selected_parameters, batches, output_directory, rank, args
            )
        else:
            _run_xtuner_steps(model, selected_parameters, batches, output_directory, rank, args)
        del model, selected_parameters
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        if rank == 0:
            try:
                if args.full_parameters:
                    failed_steps = _run_hf_full_steps(
                        hf_path, batches, output_directory, args
                    )
                else:
                    failed_steps = _run_hf_steps(
                        hf_path, mapping, batches, output_directory, args
                    )
                result[0] = failed_steps
                print(
                    f"{'PASS' if not failed_steps else 'FAIL'}"
                    f" {args.steps - failed_steps}/{args.steps} training steps"
                )
            except Exception as error:  # Keep peers from hanging in a collective on rank-0 HF failure.
                result[:] = [-1, f"{type(error).__name__}: {error}"]
        dist.broadcast_object_list(result, src=0)
    finally:
        dist.barrier()
        if rank == 0:
            shutil.rmtree(output_directory)

    dist.destroy_process_group()
    if result[0] == -1:
        raise RuntimeError(f"HF replay failed: {result[1]}")
    if result[0]:
        raise SystemExit(f"FAIL {result[0]}/{args.steps} training steps")


if __name__ == "__main__":
    main()
