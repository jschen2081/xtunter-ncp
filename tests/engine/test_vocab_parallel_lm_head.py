"""Verify the vocab-parallel LM head against the non-parallel layout.

Topology: world_size = EP * DP = 2 * 1. Both engines run EP=2 so the expert
layout is identical; only the language-model head differs. Weights are copied
between the two engines, then one train step is executed on the same input and
loss / gradients / grad norm are compared.
"""

from __future__ import annotations

import parametrize
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router.greedy import GreedyRouterConfig
from xtuner.v1.utils.device import get_device


DEVICE = get_device()

_TINY_LAYERS = 2
_SEQ_LEN = 32
_VOCAB_SIZE = 128
_HIDDEN_SIZE = 128


def _build_tiny_moe_cfg(ep_size: int = 1) -> Qwen3MoEConfig:
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
        ep_size=ep_size,
        dispatcher="all2all" if ep_size > 1 else None,
        compile_cfg=False,
        balancing_loss_cfg=None,
        z_loss_cfg=None,
    )


def _build_engine(ep_size: int, vocab_parallel_size: int) -> TrainEngine:
    return TrainEngine(
        model_cfg=_build_tiny_moe_cfg(ep_size),
        optim_cfg=AdamWConfig(),
        fsdp_cfg=FSDPConfig(ep_size=ep_size, cpu_offload=False, vocab_parallel_size=vocab_parallel_size),
        intra_layer_micro_batch=1,
    )


def _make_input(device: torch.device, seed: int = 12345) -> tuple[torch.Tensor, torch.Tensor]:
    # Each rank owns a distinct data shard, matching the SFT data mesh
    # (``dp = world / (tp * sp)``), so the gather path is exercised for real.
    torch.manual_seed(seed + dist.get_rank())
    full_ids = torch.randint(0, _VOCAB_SIZE, (1, _SEQ_LEN), dtype=torch.long, device=device)
    return full_ids[:, :-1], full_ids[:, 1:]


def _make_engine_items(loss_cfg: CELossConfig, input_ids: torch.Tensor, labels: torch.Tensor) -> list[ModelItem]:
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
    loss_ctx = loss_cfg.build(data={"shifted_labels": labels.to(DEVICE)}, sp_mesh=None)
    assert loss_ctx is not None
    loss_ctx_list = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])
    return [ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx}) for loss_ctx in loss_ctx_list]


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
                dst_param.copy_(
                    distribute_tensor(src_full, dst_param.device_mesh, dst_param.placements)
                )
            else:
                dst_param.copy_(src_full)


class TestVocabParallelLMHead(DeterministicDDPTestCase):
    """The head's loss and gradients must match the unsharded layout."""

    @parametrize.parametrize("device,vocab_parallel_size", [("cuda", 2)])
    def test_vocab_parallel_matches_full_head(self, device: str, vocab_parallel_size: int) -> None:
        pg = self.create_pg(device)

        engine_ref = _build_engine(ep_size=2, vocab_parallel_size=1)
        engine_ref.init_model_weights()

        engine_vp = _build_engine(ep_size=2, vocab_parallel_size=vocab_parallel_size)
        engine_vp.init_model_weights()
        _copy_weights(engine_ref, engine_vp)
        dist.barrier()

        # The head must only keep its own vocabulary slice.
        head = _param(engine_vp, "lm_head.weight")
        assert isinstance(head, DTensor)
        assert head.to_local().shape == (_VOCAB_SIZE // vocab_parallel_size, _HIDDEN_SIZE)

        input_ids, labels = _make_input(torch.device(device, dist.get_rank() % torch.cuda.device_count()))
        loss_cfg = CELossConfig(mode="chunk", chunk_size=8)

        step_ref = engine_ref.train_step(_make_engine_items(loss_cfg, input_ids, labels))
        norm_ref = engine_ref.clip_grad_norm(do_clip=False)
        loss_ref = step_ref["logs_info"]["reduced_llm_loss"]

        step_vp = engine_vp.train_step(_make_engine_items(loss_cfg, input_ids, labels))
        norm_vp = engine_vp.clip_grad_norm(do_clip=False)
        loss_vp = step_vp["logs_info"]["reduced_llm_loss"]

        assert torch.isfinite(torch.tensor(loss_vp))
        torch.testing.assert_close(torch.tensor(loss_vp), torch.tensor(loss_ref), rtol=2e-2, atol=1e-5)

        head_grad_ref = _grad(engine_ref, "lm_head.weight")
        head_grad_vp = _grad(engine_vp, "lm_head.weight")
        torch.testing.assert_close(head_grad_vp, head_grad_ref, rtol=2e-2, atol=1e-5)

        embed_grad_ref = _grad(engine_ref, "embed_tokens.weight")
        embed_grad_vp = _grad(engine_vp, "embed_tokens.weight")
        torch.testing.assert_close(embed_grad_vp, embed_grad_ref, rtol=2e-2, atol=1e-5)

        torch.testing.assert_close(norm_vp.float(), norm_ref.float(), rtol=2e-2, atol=1e-5)

        dist.barrier()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    @property
    def world_size(self) -> int:
        return 2

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False
