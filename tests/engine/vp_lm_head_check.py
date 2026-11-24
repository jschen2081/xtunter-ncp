"""Standalone EP=2 parity check for the vocab-parallel LM head.

Run on an 8-card NPU host:

    ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        tests/engine/vp_lm_head_check.py

Both engines use EP=2 so the expert layout is identical; only the language
model head differs. Weights are copied between them, one train step runs on the
same input, and loss / gradients / grad norm are compared.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from xtuner.v1.config import AdamWConfig, FSDPConfig  # noqa: E402
from xtuner.v1.engine.train_engine import TrainEngine  # noqa: E402
from xtuner.v1.loss.ce_loss import CELossConfig  # noqa: E402
from xtuner.v1.model.base import ModelItem  # noqa: E402
from xtuner.v1.model.moe.moe import SequenceContext  # noqa: E402
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig  # noqa: E402
from xtuner.v1.module.attention import MHAConfig  # noqa: E402
from xtuner.v1.module.router.greedy import GreedyRouterConfig  # noqa: E402
from xtuner.v1.utils.device import get_device, get_torch_device_module  # noqa: E402


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()

_TINY_LAYERS = 2
_SEQ_LEN = 32
_VOCAB_SIZE = 128
_HIDDEN_SIZE = 128
_EP_SIZE = 2


def _build_cfg() -> Qwen3MoEConfig:
    return Qwen3MoEConfig(
        vocab_size=_VOCAB_SIZE,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        num_hidden_layers=_TINY_LAYERS,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=256,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        hidden_act="silu",
        attention=MHAConfig(num_attention_heads=4, num_key_value_heads=2, head_dim=32, qk_norm=True),
        tie_word_embeddings=False,
        n_routed_experts=4,
        n_shared_experts=0,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        hidden_factor=1.0,
        moe_intermediate_size=64,
        router=GreedyRouterConfig(scoring_func="softmax", norm_topk_prob=True, router_scaling_factor=1.0),
        ep_size=_EP_SIZE,
        dispatcher="all2all",
        compile_cfg=False,
        balancing_loss_cfg=None,
        z_loss_cfg=None,
    )


def _build_engine(vocab_parallel_size: int) -> TrainEngine:
    return TrainEngine(
        model_cfg=_build_cfg(),
        optim_cfg=AdamWConfig(),
        fsdp_cfg=FSDPConfig(ep_size=_EP_SIZE, cpu_offload=False, vocab_parallel_size=vocab_parallel_size),
        intra_layer_micro_batch=1,
    )


def _make_input(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    seed = 12345
    if os.environ.get("VP_CHECK_DISTINCT_DATA") == "1":
        seed += dist.get_rank()
    torch.manual_seed(seed)
    full_ids = torch.randint(0, _VOCAB_SIZE, (1, _SEQ_LEN), dtype=torch.long, device=device)
    return full_ids[:, :-1], full_ids[:, 1:]


def _make_items(loss_cfg: CELossConfig, input_ids: torch.Tensor, labels: torch.Tensor) -> list[ModelItem]:
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
    loss_ctx = loss_cfg.build(data={"shifted_labels": labels.to(DEVICE)}, sp_mesh=None)
    assert loss_ctx is not None
    loss_ctx_list = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])
    return [ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": ctx}) for ctx in loss_ctx_list]


def _full(tensor: torch.Tensor | DTensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _param(engine: TrainEngine, suffix: str) -> torch.nn.Parameter:
    for name, param in engine.model.named_parameters():
        if name.replace("._checkpoint_wrapped_module", "").endswith(suffix):
            return param
    raise AssertionError(f"Cannot find parameter ending with {suffix}")


def _grad(engine: TrainEngine, suffix: str) -> torch.Tensor:
    grad = _param(engine, suffix).grad
    assert grad is not None, f"Missing gradient for {suffix}"
    return _full(grad).detach().float().cpu()


def _copy_weights(src: TrainEngine, dst: TrainEngine) -> None:
    src_params = dict(src.model.named_parameters())
    with torch.no_grad():
        for name, dst_param in dst.model.named_parameters():
            src_full = _full(src_params[name].detach()).to(device=dst_param.device, dtype=dst_param.dtype)
            if isinstance(dst_param, DTensor):
                dst_param.copy_(distribute_tensor(src_full, dst_param.device_mesh, dst_param.placements))
            else:
                dst_param.copy_(src_full)


def _as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    return torch.tensor(float(value))


def _rel_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    DEVICE_MODULE.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()

    torch.manual_seed(12345 + rank)
    engine_ref = _build_engine(vocab_parallel_size=1)
    engine_ref.init_model_weights()

    engine_vp = _build_engine(vocab_parallel_size=_EP_SIZE)
    engine_vp.init_model_weights()
    _copy_weights(engine_ref, engine_vp)
    dist.barrier()

    head = _param(engine_vp, "lm_head.weight")
    assert isinstance(head, DTensor)
    local_shape = tuple(head.to_local().shape)
    assert local_shape == (_VOCAB_SIZE // _EP_SIZE, _HIDDEN_SIZE), f"unexpected local head {local_shape}"

    input_ids, labels = _make_input(torch.device(DEVICE, local_rank))
    loss_cfg = CELossConfig(mode=os.environ.get("VP_CHECK_MODE", "chunk"), chunk_size=8)

    step_ref = engine_ref.train_step(_make_items(loss_cfg, input_ids, labels))
    norm_ref = engine_ref.clip_grad_norm(do_clip=False)
    loss_ref = step_ref["logs_info"]["reduced_llm_loss"]

    step_vp = engine_vp.train_step(_make_items(loss_cfg, input_ids, labels))
    norm_vp = engine_vp.clip_grad_norm(do_clip=False)
    loss_vp = step_vp["logs_info"]["reduced_llm_loss"]

    checks = {
        "loss": (_as_tensor(loss_vp), _as_tensor(loss_ref)),
        "lm_head.grad": (_grad(engine_vp, "lm_head.weight"), _grad(engine_ref, "lm_head.weight")),
        "embed.grad": (_grad(engine_vp, "embed_tokens.weight"), _grad(engine_ref, "embed_tokens.weight")),
        "grad_norm": (_as_tensor(norm_vp), _as_tensor(norm_ref)),
    }

    ok = True
    if rank == 0:
        print(f"local head shard: {local_shape}")
        for name, (got, want) in checks.items():
            got_f = _as_tensor(got)
            want_f = _as_tensor(want)
            rel = _rel_norm(got_f, want_f)
            absd = (got_f - want_f).abs().max().item()
            good = bool(torch.isfinite(got_f).all()) and rel < 2e-2
            ok = ok and good
            print(f"{name:16s} vp={got_f.reshape(-1)[:1].item():.6f} "
                  f"ref={want_f.reshape(-1)[:1].item():.6f} max_abs={absd:.3e} rel_norm={rel:.3e} "
                  f"{'OK' if good else 'FAIL'}")
        print("RESULT:", "PASS" if ok else "FAIL")

    if os.environ.get("VP_CHECK_SAVE") == "1":
        import shutil
        import tempfile

        tmpdir = os.path.join(tempfile.gettempdir(), f"vp_check_{os.environ.get('MASTER_PORT', '0')}")
        if rank == 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
            os.makedirs(tmpdir, exist_ok=True)
        dist.barrier()
        if True:
            engine_vp.model.save_hf(tmpdir)
            dist.barrier()
            engine_new = _build_engine(vocab_parallel_size=_EP_SIZE)
            engine_new.init_model_weights()
            engine_new.model.from_hf(tmpdir)
            got = _full(_param(engine_new, "lm_head.weight").detach()).float().cpu()
            want = _full(_param(engine_vp, "lm_head.weight").detach()).float().cpu()
            rel = _rel_norm(got, want)
            good = rel < 2e-2
            ok = ok and good
            if rank == 0:
                print(f"{'checkpoint':16s} rel_norm={rel:.3e} {'OK' if good else 'FAIL'}")
            engine_new.close()
            dist.barrier()
            if rank == 0:
                shutil.rmtree(tmpdir, ignore_errors=True)

    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
