#!/usr/bin/env python3
"""Align ConceptLMQwen35V2 logits and next-token loss with the HF NCP v2 export.

The default ``controlled`` path runs XTuner without ``torch.compile`` and
uses the same eager GDN/SDPA references as HF. In ``production``, both models
use FlashAttention-3; XTuner enables its compile-friendly GDN custom-op
wrappers, fused kernels, and default performance ``torch.compile`` options.

XTuner uses the repo FSDP path (``fully_shard``). Parameters are allgathered
inside each layer, and the forward already returns the full sequence. HF runs
on rank 0's single GPU.

Launch from the xtuner-ncp repo::

    torchrun --standalone --nproc-per-node=8 \\
        examples/v1/scripts/qwen35_9b_alignment/align_conceptlm_qwen35_v2_hf.py

    torchrun --standalone --nproc-per-node=8 \\
        examples/v1/scripts/qwen35_9b_alignment/align_conceptlm_qwen35_v2_hf.py \\
        --kernel-path production

    torchrun --standalone --nproc-per-node=8 \\
        examples/v1/scripts/qwen35_9b_alignment/align_conceptlm_qwen35_v2_hf.py \\
        --kernel-path production --lengths 8 128 --backward-length 8
"""

import argparse
import gc
import json
import math
import os
import stat
import sys
import tempfile
from pathlib import Path

# Read by the HF remote code and by XTuner's GDN dispatcher at call time.
os.environ.update(
    {
        # F.rms_norm
        "NCP_Q35V2_TORCH_RMS": "1",
        "NCP_Q35V2_MUDD_RMS": "1",
        "NCP_Q35V2_TE_RMS": "0",

        # Exercise XTuner's generic Triton RMSNorm on the production path.
        "XTUNER_USE_NATIVE_RMSNORM": "0",
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

        "NCP_Q35V2_TRITON_DUAL_CROSS_MUDD": "0",
        # The current dual Cross-MUDD Triton implementation is forward-only.

        "NCP_Q35V2_COMPILE_CROSS_ROUTER": "0",
        "NCP_Q35V2_SHARE_CROSS_INV_RMS": "1",
    }
)
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_HF = (
    "/mnt/shared-storage-user/llmrazor-share/heqiushi/ncp/qwen35/"
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



def configure_hf_modules_cache() -> None:
    """Use a writable remote-code cache when the default home cache is inaccessible."""
    if "HF_MODULES_CACHE" in os.environ:
        return
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    hf_home = Path(os.environ.get("HF_HOME", str(cache_home / "huggingface")))
    modules_cache = hf_home / "modules"
    try:
        modules_cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=modules_cache):
            pass
    except PermissionError:
        # Each torchrun worker owns its cache; no shared-directory creation races.
        fallback = tempfile.mkdtemp(prefix="xtuner-hf-modules-")
        os.environ["HF_MODULES_CACHE"] = fallback
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(f"HF modules cache {modules_cache} is inaccessible; using {fallback}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", default=DEFAULT_HF)
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--kernel-path",
        choices=("controlled", "production"),
        default="controlled",
        help=(
            "controlled: both eager; production: XTuner compile/RMSNorm/GDN paths plus FA3 on both models"
        ),
    )
    parser.add_argument("--atol", type=float, default=1.5e-1, help="max |logit| gap at the context limit")
    parser.add_argument("--loss-atol", type=float, default=1e-3, help="max |mean CE| gap at the context limit")
    parser.add_argument(
        "--token-loss-atol",
        type=float,
        default=1e-1,
        help="max per-token CE gap at the context limit",
    )
    parser.add_argument(
        "--top1-min",
        type=float,
        default=0.997,
        help="minimum top-1 agreement at the context limit",
    )
    parser.add_argument(
        "--backward-length",
        type=int,
        default=None,
        help="also compare dL/d(embedding output) at this short sequence length",
    )
    parser.add_argument(
        "--backward-atol",
        type=float,
        default=0.0,
        help="maximum absolute embedding-gradient gap for --backward-length",
    )
    return parser.parse_args()


def acceptance(
    length: int,
    atol: float,
    loss_atol: float,
    token_loss_atol: float,
    top1_min: float,
) -> dict[str, float]:
    """Short boundaries have less room for bf16 drift than the 8192 forward."""
    if length <= 9:
        return {
            "atol": min(atol, 5e-3),
            "loss_atol": min(loss_atol, 1e-4),
            "token_loss_atol": min(token_loss_atol, 5e-3),
            "top1_min": 1.0,
        }
    if length < 8191:
        return {
            "atol": min(atol, 1e-2),
            "loss_atol": min(loss_atol, 5e-4),
            "token_loss_atol": min(token_loss_atol, 1e-2),
            "top1_min": 1.0,
        }
    return {
        "atol": atol,
        "loss_atol": loss_atol,
        "token_loss_atol": token_loss_atol,
        "top1_min": top1_min,
    }


def gate_failures(row: dict[str, float], limits: dict[str, float]) -> list[str]:
    failed: list[str] = []
    for metric in ("max_abs", "loss_abs", "token_loss_max", "top1"):
        if not math.isfinite(row[metric]):
            failed.append(f"{metric} is not finite")
    if failed:
        return failed
    if row["max_abs"] > limits["atol"]:
        failed.append(f"logit {row['max_abs']:.3e}>{limits['atol']:.3e}")
    if row["loss_abs"] > limits["loss_atol"]:
        failed.append(f"loss {row['loss_abs']:.3e}>{limits['loss_atol']:.3e}")
    if row["token_loss_max"] > limits["token_loss_atol"]:
        failed.append(f"token_loss {row['token_loss_max']:.3e}>{limits['token_loss_atol']:.3e}")
    if row["top1"] < limits["top1_min"]:
        failed.append(f"top1 {row['top1']:.6f}<{limits['top1_min']:.6f}")
    return failed


def validate_lengths(lengths: list[int], max_length: int = 8192) -> list[int]:
    """Validate lengths before allocating model inputs or computing next-token loss."""
    if not lengths:
        raise ValueError("at least one sequence length is required")
    if len(set(lengths)) != len(lengths):
        raise ValueError("sequence lengths must be unique")
    invalid = [length for length in lengths if length < 2 or length > max_length]
    if invalid:
        raise ValueError(f"sequence lengths must be in [2, {max_length}], got {invalid}")
    return lengths


def use_hf_mla_kernel() -> str:
    """Point XTuner MLA at the same SDPA call HF uses here."""
    import torch.nn.functional as F

    import xtuner.v1.module.attention.concept_qwen35 as v2_attention

    def sdpa(query, key, value, seq_ctx, scale):
        del seq_ctx
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=True,
            scale=scale,
        ).transpose(1, 2)

    v2_attention._causal_attention = sdpa
    return "torch SDPA"


def use_hf_gdn_kernels() -> str:
    """Point XTuner controlled GDN at the same direct FLA calls HF uses."""
    import xtuner.v1.ops.gated_deltanet as gdn_ops

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

    gdn_ops.get_causal_conv1d_fn = lambda **_kwargs: fla_conv
    gdn_ops.get_chunk_gated_delta_rule_fn = lambda: fla_delta
    return "direct FLA calls"


def use_xtuner_mla_kernel() -> str:
    """Validate and retain XTuner's MLA FlashAttention path."""
    from xtuner.v1.ops import flash_attn_varlen_func

    if flash_attn_varlen_func is None:
        raise SystemExit("XTuner FlashAttention is unavailable on this CUDA environment")
    return f"{flash_attn_varlen_func.__module__}.{flash_attn_varlen_func.__name__}"


def use_xtuner_gdn_kernels() -> str:
    """Validate and retain XTuner's GDN production dispatch paths."""
    from xtuner.v1.ops.gated_deltanet import get_causal_conv1d_fn, get_chunk_gated_delta_rule_fn

    conv = get_causal_conv1d_fn()
    delta = get_chunk_gated_delta_rule_fn()
    expected_prefix = "xtuner.v1.ops.gated_deltanet."
    for label, function in (("causal conv", conv), ("delta rule", delta)):
        if not function.__module__.startswith(expected_prefix):
            raise SystemExit(
                f"production {label} did not select an XTuner custom-op wrapper: "
                f"{function.__module__}.{function.__name__}"
            )
    return f"{conv.__module__}.{conv.__name__} + {delta.__module__}.{delta.__name__}"


def configure_xtuner_hf_impl(kernel_path: str) -> None:
    """Enable broad HF compatibility only for reference acceptance paths."""
    if kernel_path == "production":
        os.environ.pop("XTUNER_HF_IMPL", None)
    else:
        os.environ["XTUNER_HF_IMPL"] = "true"


def configure_xtuner_kernels(kernel_path: str) -> tuple[str, str]:
    """Select the eager controlled path or compiled production path."""
    configure_xtuner_hf_impl(kernel_path)
    # These must be selected before either model imports its attention ops.
    xtuner_fa3 = kernel_path == "production"
    hf_fa3 = kernel_path == "production"
    production = kernel_path == "production"
    os.environ["XTUNER_USE_FA3"] = "1" if xtuner_fa3 else "0"
    os.environ["NCP_Q35V2_FA3"] = "1" if hf_fa3 else "0"
    os.environ["XTUNER_GDN_FUSED_QK_L2"] = "true" if production else "false"
    os.environ["XTUNER_GDN_FUSED_NORM_GATE"] = "true" if production else "false"
    # HF keeps its existing kernels but selects the closest production
    # numerical paths for comparison with the fused XTuner implementation.
    os.environ["NCP_Q35V2_FLA_L2"] = "1" if production else "0"
    os.environ["NCP_Q35V2_GDN_TE_OUT_NORM"] = "1" if production else "0"
    if kernel_path == "controlled":
        os.environ["XTUNER_GDN_CUSTOM_OP"] = "false"
        return use_hf_mla_kernel(), use_hf_gdn_kernels()
    if kernel_path == "production":
        os.environ["XTUNER_GDN_CUSTOM_OP"] = "true"
        return use_xtuner_mla_kernel(), use_xtuner_gdn_kernels()
    raise ValueError(f"unknown kernel path: {kernel_path}")


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


def comparison_metrics(hf_logits, reference, input_ids, chunk_tokens: int = 64) -> dict[str, float]:
    """Compare full-vocabulary logits in token chunks to bound temporary memory."""
    import torch
    import torch.nn.functional as F

    if hf_logits.shape != reference.shape:
        raise ValueError(f"logit shape {tuple(hf_logits.shape)} vs {tuple(reference.shape)}")
    max_abs = 0.0
    abs_sum = 0.0
    element_count = 0
    top1_matches = 0
    token_count = hf_logits.shape[1]
    hf_loss_sum = 0.0
    reference_loss_sum = 0.0
    token_loss_max = 0.0
    loss_token_count = 0
    for start in range(0, token_count, chunk_tokens):
        end = min(start + chunk_tokens, token_count)
        hf_chunk = hf_logits[:, start:end].float()
        reference_chunk = reference[:, start:end].to(device=hf_logits.device).float()
        gap = (hf_chunk - reference_chunk).abs()
        max_abs = max(max_abs, gap.max().item())
        abs_sum += gap.sum().item()
        element_count += gap.numel()
        top1_matches += hf_chunk.argmax(-1).eq(reference_chunk.argmax(-1)).sum().item()

        loss_end = min(end, token_count - 1)
        if start < loss_end:
            labels = input_ids[:, start + 1 : loss_end + 1].to(hf_logits.device)
            hf_token_loss = F.cross_entropy(
                hf_chunk[:, : loss_end - start].reshape(-1, hf_chunk.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            )
            reference_token_loss = F.cross_entropy(
                reference_chunk[:, : loss_end - start].reshape(-1, reference_chunk.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            )
            hf_loss_sum += hf_token_loss.sum().item()
            reference_loss_sum += reference_token_loss.sum().item()
            loss_token_count += hf_token_loss.numel()
            token_loss_max = max(token_loss_max, (hf_token_loss - reference_token_loss).abs().max().item())
        del gap, hf_chunk, reference_chunk

    hf_loss = hf_loss_sum / loss_token_count
    reference_loss = reference_loss_sum / loss_token_count
    return {
        "max_abs": max_abs,
        "mean_abs": abs_sum / element_count,
        "top1": top1_matches / token_count,
        "hf_loss": hf_loss,
        "xt_loss": reference_loss,
        "loss_abs": abs(hf_loss - reference_loss),
        "token_loss_max": token_loss_max,
    }


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


def load_xtuner(hf_path: Path, rank: int, compile_model: bool):
    import torch

    from xtuner.v1.config import FSDPConfig
    from xtuner.v1.model.dense.conceptlm_qwen35_9b import ConceptLMQwen35V2Config

    with torch.device("meta"):
        model = ConceptLMQwen35V2Config(compile_cfg=compile_model).build()
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


def input_embedding_gradient(model, input_ids, forward) -> tuple[object, object]:
    """Return logits and dL/d(embedding output) without allocating parameter gradients."""
    captured = {}

    def make_leaf(_module, _inputs, output):
        leaf = output.detach().requires_grad_(True)
        captured["embedding"] = leaf
        return leaf

    embedding = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else model.embed_tokens
    handle = embedding.register_forward_hook(make_leaf)
    try:
        logits = forward(model, input_ids)
        loss, _ = mean_ce(logits, input_ids)
        (gradient,) = __import__("torch").autograd.grad(loss, captured["embedding"])
    finally:
        handle.remove()
    return logits.detach(), gradient.detach()


def xtuner_train_logits(model, input_ids):
    from xtuner.v1.data_proto import SequenceContext

    return model(SequenceContext.from_input_ids((input_ids,), device="cuda")).logits


def hf_train_logits(model, input_ids):
    return model(input_ids=input_ids, use_cache=False).logits


def _distribution_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def relax_hf_fla_pin(model) -> None:
    """Use the installed FLA kernels instead of the HF export's fla-core pin."""
    import importlib

    fla = _distribution_version("flash-linear-attention")
    core = _distribution_version("fla-core")
    print(f"flash-linear-attention={fla}, fla-core={core}; HF requires fla-core == 0.5.1")

    def kernels():
        from fla.modules.convolution import causal_conv1d
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        return causal_conv1d, chunk_gated_delta_rule

    modeling = importlib.import_module(type(model).__module__)
    modeling._fla_gdn_kernels = kernels



def relax_hf_mla_pin(model: object) -> None:
    """Use the installed FA3 interface when the HF export's pinned import is unavailable."""
    import importlib

    if os.environ.get("NCP_Q35V2_FA3") != "1":
        return
    modeling = importlib.import_module(type(model).__module__)
    try:
        modeling._flash_attention_kernel()
    except RuntimeError:
        # XTuner's FA3 wrapper imports this same runtime (including 3.0.0b1).
        # Keep HF on FA3 rather than changing the numerical comparison to SDPA.
        from flash_attn_interface import flash_attn_func

        modeling._flash_attention_kernel = lambda: flash_attn_func
        print(f"HF MLA: using installed flash-attn-3={_distribution_version('flash-attn-3')} interface")


def relax_hf_te_runtime(model: object) -> None:
    """Use the export's pinned TE-equivalent kernel if its native extension cannot load."""
    import importlib

    if os.environ.get("NCP_Q35V2_GDN_TE_OUT_NORM", "0") != "1":
        return
    try:
        # The PyTorch package registers transformer_engine_torch, unlike the
        # top-level package imported by the HF export.
        importlib.import_module("transformer_engine.pytorch")
    except (ImportError, OSError) as error:
        modeling = importlib.import_module(type(model).__module__)
        modeling._te_zero_rms_norm = modeling.zero_rmsnorm_cuda
        print(f"HF GDN RMSNorm: using export's pinned TE-equivalent Triton kernel ({error})")


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
    model.requires_grad_(False)
    return model.to("cuda")


def hf_logits(model, input_ids):
    import torch

    with torch.inference_mode():
        return model(input_ids=input_ids).logits


def main() -> None:
    configure_hf_modules_cache()

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
    if args.backward_length is not None:
        validate_lengths([args.backward_length])
    mla_kernel, gdn_kernel = configure_xtuner_kernels(args.kernel_path)
    if rank == 0:
        print(f"kernel_path={args.kernel_path}")
        print(f"  XTUNER_HF_IMPL: {os.environ.get('XTUNER_HF_IMPL', 'unset')}")
        print(f"  XTuner MLA: {mla_kernel}")
        print(f"  HF MLA: {'FlashAttention-3' if os.environ['NCP_Q35V2_FA3'] == '1' else 'torch SDPA'}")
        print(f"  XTuner GDN: {gdn_kernel}")
        print(f"  XTuner GDN in-kernel QK L2: {os.environ['XTUNER_GDN_FUSED_QK_L2']}")
        print(f"  XTuner GDN fused norm-gate: {os.environ['XTUNER_GDN_FUSED_NORM_GATE']}")
        print(f"  XTuner torch.compile: {args.kernel_path == 'production'}")

    lengths = validate_lengths(args.lengths)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    batches = {
        length: torch.randint(0, 248320, (1, length), generator=generator, dtype=torch.long)
        for length in lengths
    }
    backward_ids = (
        torch.randint(0, 248320, (1, args.backward_length), generator=generator, dtype=torch.long)
        if args.backward_length is not None
        else None
    )

    xtuner = load_xtuner(hf_path, rank, compile_model=args.kernel_path == "production")
    spot_check(xtuner, hf_path, rank)
    xtuner_backward = None
    if backward_ids is not None:
        logits, gradient = input_embedding_gradient(
            xtuner,
            backward_ids.cuda(),
            xtuner_train_logits,
        )
        if rank == 0:
            xtuner_backward = (logits.cpu(), gradient.cpu())
        del logits, gradient
    output_directory = tempfile.TemporaryDirectory(prefix="ncp-qwen35-v2-logits-") if rank == 0 else None
    xtuner_out: dict[int, Path] = {}
    for length, ids in batches.items():
        logits = xtuner_logits(xtuner, ids.cuda())
        if rank == 0:
            assert output_directory is not None
            output_path = Path(output_directory.name) / f"xtuner-{length}.pt"
            torch.save(logits.cpu(), output_path)
            xtuner_out[length] = output_path
            print(f"xtuner forward length={length} logits={tuple(logits.shape)}")
        del logits
    del xtuner
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    failed_count = 0
    row_count = 0
    if rank == 0:
        hf = load_hf(hf_path)
        relax_hf_fla_pin(hf)
        relax_hf_mla_pin(hf)
        relax_hf_te_runtime(hf)
        if backward_ids is not None:
            assert xtuner_backward is not None
            hf_backward = input_embedding_gradient(
                hf,
                backward_ids.cuda(),
                hf_train_logits,
            )
            backward_logit_gap = (hf_backward[0].float().cpu() - xtuner_backward[0].float()).abs().max().item()
            backward_grad_gap = (hf_backward[1].float().cpu() - xtuner_backward[1].float()).abs().max().item()
            print(
                f"backward length={args.backward_length}  logit max_abs={backward_logit_gap:.3e}"
                f"  embedding_grad max_abs={backward_grad_gap:.3e}"
            )
            row_count += 1
            backward_limits = acceptance(
                args.backward_length,
                args.atol,
                args.loss_atol,
                args.token_loss_atol,
                args.top1_min,
            )
            if (
                not math.isfinite(backward_logit_gap)
                or backward_logit_gap > backward_limits["atol"]
                or not math.isfinite(backward_grad_gap)
                or backward_grad_gap > args.backward_atol
            ):
                failed_count += 1
            del hf_backward
        rows = []
        for length, ids in batches.items():
            logits = hf_logits(hf, ids.cuda())
            reference = torch.load(xtuner_out[length], map_location="cpu", weights_only=True, mmap=True)
            if logits.shape != reference.shape:
                print(f"length {length}: logit shape {tuple(logits.shape)} vs {tuple(reference.shape)}")
                remaining = len(lengths) - len(rows)
                failed_count += remaining
                row_count += remaining
                break
            limits = acceptance(
                length,
                args.atol,
                args.loss_atol,
                args.token_loss_atol,
                args.top1_min,
            )
            row = {"length": length, **comparison_metrics(logits, reference, ids)}
            row["failures"] = gate_failures(row, limits)
            rows.append(row)
            status_text = "PASS" if not row["failures"] else "FAIL " + "; ".join(row["failures"])
            print(
                f"length={length:4d}  logit max_abs={row['max_abs']:.3e}"
                f"  mean_abs={row['mean_abs']:.3e}  top1={row['top1']:.6f}"
                f"  hf_loss={row['hf_loss']:.6f}  xt_loss={row['xt_loss']:.6f}"
                f"  loss_abs={row['loss_abs']:.3e}"
                f"  token_loss_max={row['token_loss_max']:.3e}"
                f"  [{status_text}]"
            )
            del logits, reference
        del hf
        gc.collect()
        torch.cuda.empty_cache()
        failed = [row for row in rows if row["failures"]]
        failed_count += len(failed)
        row_count += len(rows)
        if failed_count:
            print(f"FAIL {failed_count}/{row_count} lengths")
        else:
            print(f"PASS {row_count}/{row_count} lengths")
        assert output_directory is not None
        output_directory.cleanup()

    status = torch.tensor([failed_count, row_count], device="cuda")
    dist.broadcast(status, src=0)
    dist.barrier()
    dist.destroy_process_group()
    if status[0].item():
        raise SystemExit(f"FAIL {int(status[0].item())}/{int(status[1].item())} lengths")


if __name__ == "__main__":
    main()
