#!/usr/bin/env python3
"""Align ConceptLMQwen35V2 logits and next-token loss with the HF NCP v2 export.

HF's CUDA default uses Transformer Engine, FlashAttention-3 and a Triton
cross-MUDD. Those are turned off before either model is imported, except the
decoder dual Cross-MUDD kernel, which both sides run. GDN convolution and the
delta rule are the same FLA calls HF uses for one unpadded sequence. MLA uses
SDPA on both sides.

XTuner uses the repo FSDP path (``fully_shard``). Parameters are allgathered
inside each layer, and the forward already returns the full sequence. HF runs
on rank 0's single GPU.

Launch from the xtuner-ncp repo::

    torchrun --standalone --nproc-per-node=8 \\
        examples/v1/scripts/align_conceptlm_qwen35_v2_hf.py
"""

import argparse
import gc
import json
import os
import stat
import sys
from pathlib import Path

# Read by the HF remote code and by XTuner's GDN dispatcher at call time.
os.environ.update(
    {
        # F.rms_norm
        # "NCP_Q35V2_TORCH_RMS": "1",
        # "NCP_Q35V2_MUDD_RMS": "1",
        # "NCP_Q35V2_TE_RMS": "0",
        # 手写 fp32。ZeroCentered 关 TORCH_RMS 和 TE_RMS；Mudd 另关 MUDD_RMS。
        # "NCP_Q35V2_TORCH_RMS": "0",
        # "NCP_Q35V2_MUDD_RMS": "0",
        # "NCP_Q35V2_TE_RMS": "0",
        # TE Triton，只作用于 ZeroCenteredRMSNorm，且优先于 TORCH_RMS。Mudd 仍走 F.rms_norm。
        "NCP_Q35V2_TORCH_RMS": "0",
        "NCP_Q35V2_MUDD_RMS": "1",
        "NCP_Q35V2_TE_RMS": "1",
 
        "NCP_Q35V2_FLA_L2": "0",
        "NCP_Q35V2_FA3": "0",
        "NCP_Q35V2_TE_MLA_PROJ": "0",
        "NCP_Q35V2_MLA_TE_INPUT_NORM": "0",
        "NCP_Q35V2_GDN_TORCH_INPUT_NORM": "1",
        "NCP_Q35V2_GDN_TE_OUT_NORM": "0",
        "NCP_Q35V2_GDN_PACK_PROJECTION": "0",
        "NCP_Q35V2_TE_MLP_FC1": "0",
        "NCP_Q35V2_TE_FINAL_NORM": "0",
        "NCP_Q35V2_TRITON_CROSS_MUDD": "0",

        # 1打开dual_cross_mudd的triton优化路径
        # "NCP_Q35V2_TRITON_DUAL_CROSS_MUDD": "0",
        "NCP_Q35V2_TRITON_DUAL_CROSS_MUDD": "1",  

        "NCP_Q35V2_COMPILE_CROSS_ROUTER": "0",
        "NCP_Q35V2_SHARE_CROSS_INV_RMS": "1",
        "XTUNER_HF_IMPL": "true",
    }
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_HF = (
    "/mnt/shared-storage-gpfs2/plm-gpfs/yxwang/model_exports/"
    "ncp_qwen35_v2_2551e48_iter0000002_hf"
)
# chunk=4, sequences shorter than 8 are padded to 8, and concept 1 first
# lands on a real token at position 7.
# 3: every real token is still inside the right-shift gap
# 4: position 3 is the first real token that reads concept 0
# 5: padded to 8, with one real token inside the second chunk
# 7: last real token still reads concept 0; concept 1 starts on the pad
# 8: no padding, and concept 1 is read by a real token
# 9: first length padded to 12 instead of clamped to 8
# 128, 1024: chunk-aligned, no padding
# 8191: one pad token at the context edge
# 8192: exact context limit, no padding
DEFAULT_LENGTHS = [3, 4, 5, 7, 8, 9, 128, 1024, 8191, 8192]
SPOT_KEYS = (
    ("embed_tokens.weight", "model.embed_tokens.weight"),
    ("fusion_alpha", "model.fusion_alpha"),
    ("fusion_encoder_norm.weight", "model.fusion_encoder_norm.weight"),
    ("lm_head.weight", "lm_head.weight"),
    ("codebooks.0", "model.codebooks.0"),
    ("prediction_heads.0.weight", "model.prediction_heads.0.weight"),
    ("encoder.layers.0.self_attn.A_log", "model.encoder.layers.0.attention.A_log"),
    ("encoder.layers.3.self_attn.q_down.weight", "model.encoder.layers.3.attention.q_down.weight"),
    ("mudd.routes.encoder.0.router_w1.weight", "model.mudd.routes.encoder.0.router_w1.weight"),
    ("mudd.routes.encoder.7.post_norm", "model.mudd.routes.encoder.7.post_norm"),
    ("mudd.routes.encoder.7.static_weight", "model.mudd.routes.encoder.7.static_weight"),
    ("mudd.routes.predictor.15.router_w1.weight", "model.mudd.routes.predictor.15.router_w1.weight"),
    ("mudd.routes.decoder.7.static_weight", "model.mudd.routes.decoder.7.static_weight"),
    ("predictor.cross_route_groups.0.0.static_logits", "model.predictor.cross_route_groups.0.0.static_logits"),
    ("decoder.cross_route_groups.1.7.gate_param", "model.decoder.cross_route_groups.1.7.gate_param"),
    ("decoder.concept_routes.0.final_diag", "model.decoder.concept_routes.0.final_diag"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", default=DEFAULT_HF)
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2e-2, help="max |logit| gap at the context limit")
    parser.add_argument("--loss-atol", type=float, default=1e-3, help="max |mean CE| gap at the context limit")
    return parser.parse_args()


def acceptance(length: int, atol: float, loss_atol: float) -> dict[str, float]:
    """Short boundaries have less room for bf16 drift than the 8192 forward."""
    if length <= 9:
        return {
            "atol": min(atol, 5e-3),
            "loss_atol": min(loss_atol, 1e-4),
            "token_loss_atol": min(atol, 5e-3),
            "top1_min": 1.0,
        }
    if length < 8191:
        return {
            "atol": min(atol, 1e-2),
            "loss_atol": min(loss_atol, 5e-4),
            "token_loss_atol": min(atol, 1e-2),
            "top1_min": 1.0,
        }
    return {
        "atol": atol,
        "loss_atol": loss_atol,
        "token_loss_atol": atol,
        "top1_min": 1.0 - 8.0 / length,
    }


def gate_failures(row: dict[str, float], limits: dict[str, float]) -> list[str]:
    failed: list[str] = []
    if row["max_abs"] > limits["atol"]:
        failed.append(f"logit {row['max_abs']:.3e}>{limits['atol']:.3e}")
    if row["loss_abs"] > limits["loss_atol"]:
        failed.append(f"loss {row['loss_abs']:.3e}>{limits['loss_atol']:.3e}")
    if row["token_loss_max"] > limits["token_loss_atol"]:
        failed.append(f"token_loss {row['token_loss_max']:.3e}>{limits['token_loss_atol']:.3e}")
    if row["top1"] < limits["top1_min"]:
        failed.append(f"top1 {row['top1']:.6f}<{limits['top1_min']:.6f}")
    return failed


def use_hf_eager_kernels() -> None:
    """Point XTuner's CUDA helpers at the same FLA and SDPA calls HF uses here."""
    import torch
    import torch.nn.functional as F

    import xtuner.v1.module.attention.ncp_qwen35_v2 as v2_attention
    import xtuner.v1.ops.gated_deltanet as gdn_ops

    def sdpa(query, key, value, seq_ctx, scale):
        del seq_ctx
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=True,
            scale=scale,
        ).transpose(1, 2)

    def fla_conv(x, weight, bias=None, activation="silu", seq_idx=None):
        del bias, seq_idx
        from fla.modules.convolution import causal_conv1d

        out, _ = causal_conv1d(
            x,
            weight,
            bias=None,
            activation=activation,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=None,
        )
        return out

    def fla_delta(query, key, value, g, beta, use_qk_l2norm_in_kernel=False, cu_seqlens=None):
        del cu_seqlens
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        return chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    v2_attention._causal_attention = sdpa
    gdn_ops.get_causal_conv1d_fn = lambda: fla_conv
    gdn_ops.get_chunk_gated_delta_rule_fn = lambda: fla_delta
    del torch


def spot_check(model, hf_path: Path, rank: int) -> None:
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from safetensors import safe_open

    weight_map = json.loads((hf_path / "model.safetensors.index.json").read_text())["weight_map"]
    state = model.state_dict()
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for local_name, hf_name in SPOT_KEYS:
        by_shard.setdefault(weight_map[hf_name], []).append((local_name, hf_name))
    if rank == 0:
        print("weight spot check")
    worst = 0.0
    for shard, pairs in by_shard.items():
        with safe_open(hf_path / shard, framework="pt", device="cpu") as handle:
            for local_name, hf_name in pairs:
                loaded = state[local_name]
                if isinstance(loaded, DTensor):
                    loaded = loaded.full_tensor()
                if rank == 0:
                    reference = handle.get_tensor(hf_name).float()
                    gap = (loaded.detach().float().cpu() - reference).abs().max().item()
                    worst = max(worst, gap)
                    print(f"  {hf_name}: max_abs={gap:.3e}")
    flag = torch.tensor([worst], device="cuda")
    dist.broadcast(flag, src=0)
    if flag.item() > 0:
        raise SystemExit(f"loaded weights differ from the HF file (max_abs={flag.item():.3e})")


def mean_ce(logits, input_ids):
    import torch
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
    )
    return token_loss.mean(), token_loss


def require_hf_dir(path: Path) -> Path:
    path = path.expanduser()
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        raise SystemExit(f"{path} does not exist on this machine.") from None
    except PermissionError as exc:
        raise SystemExit(f"cannot access {path}: {exc}") from None
    if not stat.S_ISDIR(mode):
        raise SystemExit(f"{path} is not a directory.")
    if not os.access(path, os.R_OK | os.X_OK):
        raise SystemExit(f"{path} is {stat.filemode(mode)} and this user cannot list it.")
    return path


def setup_distributed() -> tuple[int, int]:
    import torch
    import torch.distributed as dist

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise SystemExit("launch with torchrun --standalone --nproc-per-node=8")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size()


def load_xtuner(hf_path: Path, rank: int):
    import torch

    from xtuner.v1.config import FSDPConfig
    from xtuner.v1.model.dense.conceptlm_qwen35_v2 import ConceptLMQwen35V2Config

    with torch.device("meta"):
        model = ConceptLMQwen35V2Config().build()
    model.fully_shard(
        FSDPConfig(
            requires_grad=False,
            recompute_ratio=0.0,
            reshard_after_forward=True,
            cpu_offload=False,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            fp32_lm_head=False,
        )
    )
    model.eval()
    loaded, unloaded, missing = model.from_hf(hf_path, strict=True)
    if rank == 0:
        print(f"xtuner from_hf: loaded={len(loaded)} unloaded={len(unloaded)} missing={len(missing)}")
    return model


def xtuner_logits(model, input_ids):
    import torch

    from xtuner.v1.data_proto import SequenceContext

    seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cuda")
    with torch.no_grad():
        outputs = model(seq_ctx)
    return outputs.logits


def _distribution_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def relax_hf_fla_pin(model) -> None:
    """HF accepts CUDA GDN only when both FLA distributions are exactly 0.5.1."""
    import importlib

    fla = _distribution_version("flash-linear-attention")
    core = _distribution_version("fla-core")
    print(f"flash-linear-attention={fla}, fla-core={core}; HF requires both == 0.5.1")

    def kernels():
        from fla.modules.convolution import causal_conv1d
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        return causal_conv1d, chunk_gated_delta_rule

    modeling = importlib.import_module(type(model).__module__)
    modeling._fla_gdn_kernels = kernels


def load_hf(hf_path: Path):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = {"trust_remote_code": True, "dtype": torch.bfloat16}
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_path, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(hf_path, **kwargs)
    model.eval()
    return model.to("cuda")


def hf_logits(model, input_ids):
    import torch

    with torch.inference_mode():
        return model(input_ids=input_ids).logits


def release(model) -> None:
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    import torch
    import torch.distributed as dist

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this script needs a CUDA GPU")
    rank, world = setup_distributed()
    if world != 8:
        dist.destroy_process_group()
        raise SystemExit(f"XTuner FSDP expects 8 ranks, found {world}")
    hf_path = require_hf_dir(Path(args.hf_path))
    use_hf_eager_kernels()

    lengths = args.lengths
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    batches = {
        length: torch.randint(0, 248320, (1, length), generator=generator, dtype=torch.long)
        for length in lengths
    }

    xtuner = load_xtuner(hf_path, rank)
    spot_check(xtuner, hf_path, rank)
    xtuner_out = {}
    for length, ids in batches.items():
        logits = xtuner_logits(xtuner, ids.cuda())
        if rank == 0:
            xtuner_out[length] = logits.float().cpu()
            print(f"xtuner forward length={length} logits={tuple(logits.shape)}")
        del logits
    release(xtuner)
    dist.barrier()

    failed_count = 0
    row_count = 0
    if rank == 0:
        hf = load_hf(hf_path)
        relax_hf_fla_pin(hf)
        rows = []
        for length, ids in batches.items():
            logits = hf_logits(hf, ids.cuda()).float().cpu()
            reference = xtuner_out[length]
            if logits.shape != reference.shape:
                print(f"length {length}: logit shape {tuple(logits.shape)} vs {tuple(reference.shape)}")
                failed_count = len(lengths)
                row_count = len(lengths)
                break
            gap = (logits - reference).abs()
            hf_loss, hf_token = mean_ce(logits, ids)
            xt_loss, xt_token = mean_ce(reference, ids)
            top1 = logits.argmax(-1).eq(reference.argmax(-1)).float().mean().item()
            token_gap = (hf_token - xt_token).abs()
            limits = acceptance(length, args.atol, args.loss_atol)
            row = {
                "length": length,
                "max_abs": gap.max().item(),
                "mean_abs": gap.mean().item(),
                "top1": top1,
                "hf_loss": hf_loss.item(),
                "xt_loss": xt_loss.item(),
                "loss_abs": abs(hf_loss.item() - xt_loss.item()),
                "token_loss_max": token_gap.max().item(),
            }
            row["failures"] = gate_failures(row, limits)
            rows.append(row)
            status_text = "PASS" if not row["failures"] else "FAIL " + "; ".join(row["failures"])
            print(
                f"length={length:4d}  logit max_abs={row['max_abs']:.3e}"
                f"  mean_abs={row['mean_abs']:.3e}  top1={top1:.6f}"
                f"  hf_loss={row['hf_loss']:.6f}  xt_loss={row['xt_loss']:.6f}"
                f"  loss_abs={row['loss_abs']:.3e}"
                f"  token_loss_max={row['token_loss_max']:.3e}"
                f"  [{status_text}]"
            )
        release(hf)
        if not failed_count:
            failed = [row for row in rows if row["failures"]]
            failed_count = len(failed)
            row_count = len(rows)
        if failed_count:
            print(f"FAIL {failed_count}/{row_count} lengths")
        else:
            print(f"PASS {row_count}/{row_count} lengths")

    status = torch.tensor([failed_count, row_count], device="cuda")
    dist.broadcast(status, src=0)
    dist.barrier()
    dist.destroy_process_group()
    if status[0].item():
        raise SystemExit(f"FAIL {int(status[0].item())}/{int(status[1].item())} lengths")


if __name__ == "__main__":
    main()
