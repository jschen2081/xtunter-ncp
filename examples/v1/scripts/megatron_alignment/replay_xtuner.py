"""Eight-GPU xTuner SP1 training on batches captured from current Megatron."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from batch_contract import validate_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--xtuner-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--recompute-ratio", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true", help="Enable the model default torch.compile configuration")
    parser.add_argument("--zero-stage", type=int, choices=(2, 3), default=3)
    parser.add_argument("--gradient-accumulation-no-sync", action="store_true")
    parser.add_argument("--profile-trace", action="store_true")
    parser.add_argument("--sp-size", type=int, choices=(1, 8), default=1)
    args = parser.parse_args()
    contract = json.loads((args.run_dir / "contract.json").read_text())
    args.hf_path = Path(contract["hf_path"])
    complete = [json.loads((args.run_dir / f"megatron_complete.rank{rank:02d}.json").read_text())
                for rank in range(8)]
    if not complete[0] or any(item != complete[0] for item in complete):
        raise ValueError("Megatron capture is incomplete or ranks consumed different steps")
    result_path = args.run_dir / f"xtuner_sp{args.sp_size}.jsonl"
    if result_path.exists():
        raise FileExistsError(f"Use a fresh capture directory: {result_path}")
    sys.path.insert(0, str(args.xtuner_root))
    sys.path.insert(0, str(args.xtuner_root / "examples/v1/scripts"))
    from align_conceptlm_qwen35_v2_sp_training import _configure_kernels, _seed_step

    _configure_kernels()
    import torch
    import torch.distributed as dist
    from xtuner.v1.config import FSDPConfig, MuonConfig
    from xtuner.v1.data_proto import SequenceContext
    from xtuner.v1.engine.train_engine import TrainEngine
    from xtuner.v1.loss import CELossConfig, ConceptLMLossConfig
    from xtuner.v1.model.base import ModelItem
    from xtuner.v1.model.dense.conceptlm_qwen35_v2 import ConceptLMQwen35V2, ConceptLMQwen35V2Config

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise ValueError("Comparison requires exactly eight ranks, SP1")
    from xtuner.v1.ops.flash_attn import flash_attn_varlen_func
    attention_backend = flash_attn_varlen_func.__name__
    if os.environ.get("XTUNER_USE_FA3", "0") == "1" and attention_backend != "gpu_flash_varlen_attn_v3":
        raise ValueError("FA3 requested but a different attention backend was selected")
    if rank == 0:
        print(f"ATTENTION_BACKEND {attention_backend}", flush=True)
    # Match the production Muon fp32 matmul policy.
    torch.set_float32_matmul_precision("medium")
    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh("cuda", (world // args.sp_size, args.sp_size), mesh_dim_names=("replay_dp", "replay_sp"))
    sp_mesh = mesh["replay_sp"]
    source_ranks = range((rank // args.sp_size) * args.sp_size, (rank // args.sp_size + 1) * args.sp_size)
    paths = sorted(path for source_rank in source_ranks
                   for path in (args.run_dir / "batches").glob(f"*.rank{source_rank:02d}.pt"))
    grouped = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_batch(payload, payload["rank"], world)
        grouped.setdefault(payload["step"], []).append(payload)
    actual_counts = {str(step): len(items) for step, items in grouped.items()}
    if actual_counts != {step: count * args.sp_size for step, count in complete[rank].items()}:
        raise ValueError("Captured batches differ from successful Megatron completion record")
    for step, payloads in grouped.items():
        payloads.sort(key=lambda value: (value["micro"], value["rank"]))
        expected = payloads[0]["global_batch_size"] // world
        if [(p["micro"], p["rank"]) for p in payloads] != [
                (micro, source_rank) for micro in range(expected) for source_rank in source_ranks]:
            raise ValueError(f"Missing/duplicate microbatches in step {step}")
        if any(p["lr"] != payloads[0]["lr"] or p["global_batch_size"] != payloads[0]["global_batch_size"]
               for p in payloads):
            raise ValueError("Microbatches disagree on step optimizer contract")
    step_contracts = [None] * world
    dist.all_gather_object(step_contracts, [(s, len(ps), ps[0]["lr"]) for s, ps in sorted(grouped.items())])
    if any(item != step_contracts[0] for item in step_contracts):
        raise ValueError("Ranks disagree on optimizer step schedule")

    _seed_step(1234)
    engine = TrainEngine(
        model_cfg=ConceptLMQwen35V2Config(
            compile_cfg=args.compile,
            concept_loss_cfg=ConceptLMLossConfig(),
            lm_loss_cfg=CELossConfig(mode="chunk", chunk_size=128, loss_reduction="token")),
        optim_cfg=MuonConfig(megatron_compatible=True, lr=8e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
                            momentum=0.95, nesterov=True, max_grad_norm=1.0,
                            adjust_lr="rms_norm", extra_scale_factor=0.2,
                            clip_grad_mode="all", one_d_weight_decay=0.0,
                            embedding_weight_decay=0.1, adamw_foreach=False),
        fsdp_cfg=FSDPConfig(requires_grad=True, recompute_ratio=args.recompute_ratio,
                           reshard_after_forward=args.zero_stage == 3, cpu_offload=False,
                           gradient_accumulation_no_sync=args.gradient_accumulation_no_sync,
                           param_dtype=torch.bfloat16, reduce_dtype=torch.float32, fp32_lm_head=False),
    )
    engine.from_hf(args.hf_path, strict=True)
    engine.model.train()
    optimizer_contract = dict(optimizer_class=type(engine.optimizer).__name__,
                              optimizer_config=engine.optim_cfg.model_dump())
    if rank == 0:
        optimizer_contract.update(
            xtuner_repository=str(args.xtuner_root.resolve()),
            xtuner_commit=subprocess.check_output(
                ["git", "-C", str(args.xtuner_root), "rev-parse", "HEAD"], text=True).strip(),
            sp_size=args.sp_size, world_size=world, recompute_ratio=args.recompute_ratio, compile=args.compile,
            zero_stage=args.zero_stage, reshard_after_forward=engine.fsdp_cfg.reshard_after_forward,
            gradient_accumulation_no_sync=engine.fsdp_cfg.gradient_accumulation_no_sync,
            attention_backend=attention_backend, use_fa3=os.environ.get("XTUNER_USE_FA3", "0") == "1",
            profile_trace=args.profile_trace, trace_schedule={"wait": 5, "warmup": 1, "active": 2, "repeat": 1},
            hf_path=str(args.hf_path), megatron_commit=contract["commit"])
        (args.run_dir / "xtuner_optimizer_contract.json").write_text(json.dumps(optimizer_contract, indent=2))
    # Current Megatron branch uses Muon for all 32 codebooks.
    muon_ids = {id(p) for group in engine.optimizer.param_groups
                if group.get("algorithm") == "muon" for p in group["params"]}
    if not all(id(p) in muon_ids for p in engine.model.codebooks):
        raise ValueError("Codebooks must use Muon on the current branch")
    profiler = None
    if args.profile_trace:
        trace_dir = args.run_dir / "traces"
        trace_dir.mkdir(exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=1, active=2, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(trace_dir), worker_name=f"rank{rank:02d}", use_gzip=True),
            record_shapes=True, profile_memory=False, with_stack=False,
        )
        profiler.start()
    try:
        for step, payloads in sorted(grouped.items()):
            batches = []
            for payload in payloads:
                tokens = payload["tokens"].cuda()
                bounds = payload["cu_seqlens"].tolist()
                documents = tuple(tokens[:, start:end] for start, end in zip(bounds, bounds[1:]))
                context = SequenceContext.from_input_ids(documents, device="cuda")
                if not torch.equal(context.position_ids, payload["position_ids"].cuda()):
                    raise ValueError("Position IDs differ from production")
                labels = payload["labels"].cuda().masked_fill(payload["loss_mask"].cuda() == 0, -100)
                if args.sp_size > 1:
                    context = context.split(sp_mesh)
                batches.append(dict(seq_ctx=context, shifted_labels=labels, loss_mask=payload["loss_mask"].cuda()))
            target_count = sum(b["shifted_labels"].ne(-100).sum() for b in batches)
            dist.all_reduce(target_count)
            if target_count.item() <= 0:
                raise ValueError("No effective training tokens")
            contexts = engine.model.build_loss_ctx_batch(batches, sp_mesh=sp_mesh if args.sp_size > 1 else None)
            items = []
            for batch, context in zip(batches, contexts):
                items.append(ModelItem(seq_ctx=batch["seq_ctx"], loss_ctx=context))
            for group in engine.optimizer.param_groups:
                group["lr"] = payloads[0]["lr"]
            engine.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            info = engine.train_step(items)
            grad_norm = engine.clip_grad_norm(dtype=torch.float32)
            if not torch.isfinite(grad_norm):
                raise ValueError("Non-finite xTuner gradient norm")
            engine.step_optimizer(grad_norm)
            torch.cuda.synchronize()
            elapsed = torch.tensor(time.perf_counter() - started, device="cuda")
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            stats = torch.stack((torch.tensor(info["total_loss"], device="cuda"),
                                 info["extra_info"]["concept_vq_numerator"].sum(),
                                 info["extra_info"]["concept_hlm_numerator"].sum()))
            dist.all_reduce(stats)
            total_loss = stats[0].item() / world
            vq, hlm = (4 * stats[1:] / target_count).tolist()
            fingerprints = [None] * world
            if args.sp_size == 1:
                dist.all_gather_object(fingerprints, [p["sha256"] for p in payloads])
            else:
                fingerprints = [[p["sha256"] for p in payloads if p["rank"] == source_rank]
                                for source_rank in range(world)]
            row = dict(step=step, sp_size=args.sp_size, accumulation_steps=len(payloads),
                       total_loss=total_loss, lm_loss=total_loss - vq - hlm,
                       vq_token_loss=vq, hlm_token_loss=hlm, grad_norm=float(grad_norm),
                       lr=payloads[0]["lr"], valid_targets=int(target_count) // args.sp_size,
                       batch_sha256_by_rank=fingerprints, seconds=float(elapsed),
                       peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                       tokens_per_second_per_gpu=payloads[0]["global_batch_size"] *
                           payloads[0]["tokens"].shape[1] / float(elapsed) / world)
            if rank == 0:
                with result_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print("XTUNER_PRODUCTION " + json.dumps(row), flush=True)
                print(
                    f"XTUNER_PERFORMANCE step={step} "
                    f"elapsed_time_per_iteration={row['seconds']:.3f}s "
                    f"TPS/GPU={row['tokens_per_second_per_gpu']:.2f}",
                    flush=True,
                )
            if profiler is not None:
                profiler.step()
        if rank == 0:
            (args.run_dir / "xtuner_complete.json").write_text(json.dumps({"steps": sorted(grouped)}))
    finally:
        try:
            if profiler is not None:
                profiler.stop()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
