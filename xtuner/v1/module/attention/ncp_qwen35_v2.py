"""NCP Qwen3.5 v2 attention modules.

These modules implement the repository attention contract while preserving the
parameter layout and numerical order of the pinned HF checkpoint.
"""

import math
import os
from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn
from torch.distributed.tensor import DTensor

from xtuner.v1.config import GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all
from xtuner.v1.ops.gated_deltanet.gen_seq_idx import gen_seq_idx

from ..linear import build_linear
from ..rms_norm.ncp_qwen35_v2 import ZeroCenteredRMSNorm, env_on
from .attn_outputs import AttnOutputs


# Keep separate call sites until the Q/K/V collectives are fused. Dynamo otherwise
# reuses one callable with incompatible tensor shapes across MLA and GDN.
def _mla_query_all_to_all(x: Tensor, seq_ctx: SequenceContext) -> Tensor:
    assert seq_ctx.sequence_parallel_mesh is not None
    return ulysses_all_to_all(x, scatter_dim=2, gather_dim=1, mesh=seq_ctx.sequence_parallel_mesh)


def _mla_key_all_to_all(x: Tensor, seq_ctx: SequenceContext) -> Tensor:
    assert seq_ctx.sequence_parallel_mesh is not None
    return ulysses_all_to_all(x, scatter_dim=2, gather_dim=1, mesh=seq_ctx.sequence_parallel_mesh)


def _mla_value_all_to_all(x: Tensor, seq_ctx: SequenceContext) -> Tensor:
    assert seq_ctx.sequence_parallel_mesh is not None
    return ulysses_all_to_all(x, scatter_dim=2, gather_dim=1, mesh=seq_ctx.sequence_parallel_mesh)


def _mla_output_all_to_all(x: Tensor, seq_ctx: SequenceContext) -> Tensor:
    assert seq_ctx.sequence_parallel_mesh is not None
    return ulysses_all_to_all(x, scatter_dim=1, gather_dim=2, mesh=seq_ctx.sequence_parallel_mesh)


def _gdn_query_all_to_all(x: Tensor, seq_ctx: SequenceContext) -> Tensor:
    assert seq_ctx.sequence_parallel_mesh is not None
    return ulysses_all_to_all(x, scatter_dim=1, gather_dim=2, mesh=seq_ctx.sequence_parallel_mesh)


def _gdn_key_all_to_all(x: Tensor, seq_ctx: SequenceContext) -> Tensor:
    assert seq_ctx.sequence_parallel_mesh is not None
    return ulysses_all_to_all(x, scatter_dim=1, gather_dim=2, mesh=seq_ctx.sequence_parallel_mesh)


def _gdn_value_all_to_all(x: Tensor, seq_ctx: SequenceContext) -> Tensor:
    assert seq_ctx.sequence_parallel_mesh is not None
    return ulysses_all_to_all(x, scatter_dim=1, gather_dim=2, mesh=seq_ctx.sequence_parallel_mesh)


def _gdn_decay_all_to_all(x: Tensor, seq_ctx: SequenceContext) -> Tensor:
    assert seq_ctx.sequence_parallel_mesh is not None
    return ulysses_all_to_all(x, scatter_dim=1, gather_dim=2, mesh=seq_ctx.sequence_parallel_mesh)


def _gdn_beta_all_to_all(x: Tensor, seq_ctx: SequenceContext) -> Tensor:
    assert seq_ctx.sequence_parallel_mesh is not None
    return ulysses_all_to_all(x, scatter_dim=1, gather_dim=2, mesh=seq_ctx.sequence_parallel_mesh)


def _gdn_output_all_to_all(x: Tensor, seq_ctx: SequenceContext) -> Tensor:
    assert seq_ctx.sequence_parallel_mesh is not None
    return ulysses_all_to_all(x, scatter_dim=1, gather_dim=2, mesh=seq_ctx.sequence_parallel_mesh)


class NCPQwen35V2MLAConfig(BaseModel):
    model_config = ConfigDict(title="NCP Qwen3.5 v2 MLA config", extra="forbid")

    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_head_dim: int
    qk_pos_emb_head_dim: int
    v_head_dim: int
    rotary_base: float
    rms_norm_eps: float

    def build(
        self,
        hidden_size: int,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        layer_idx: int = 0,
        rope_scaling_cfg: RopeScalingConfig | None = None,
        generate_config: GenerateConfig | None = None,
        float8_cfg: Float8Config | None = None,
    ) -> "NCPQwen35V2MLA":
        del layer_type, rope_scaling_cfg, generate_config
        return NCPQwen35V2MLA(
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            float8_cfg=float8_cfg,
            **self.model_dump(),
        )


class NCPQwen35V2GatedDeltaNetConfig(BaseModel):
    model_config = ConfigDict(title="NCP Qwen3.5 v2 GatedDeltaNet config", extra="forbid")

    num_value_heads: int
    num_key_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_dim: int
    hidden_act: str
    rms_norm_eps: float

    def build(
        self,
        hidden_size: int,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        layer_idx: int = 0,
        rope_scaling_cfg: RopeScalingConfig | None = None,
        generate_config: GenerateConfig | None = None,
        float8_cfg: Float8Config | None = None,
    ) -> "NCPQwen35V2GatedDeltaNet":
        del layer_type, rope_scaling_cfg, generate_config
        return NCPQwen35V2GatedDeltaNet(
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            float8_cfg=float8_cfg,
            **self.model_dump(),
        )


def _production_feature_enabled(name: str) -> bool:
    override = os.environ.get(name)
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    return not env_on("XTUNER_HF_IMPL")


def _fused_zero_rms_norm_gate(
    x: Tensor,
    gate: Tensor,
    zero_centered_weight: Tensor,
    eps: float,
) -> Tensor:
    from xtuner.v1.ops.gated_deltanet.ncp_qwen35_v2 import zero_rms_norm_gate

    if isinstance(zero_centered_weight, DTensor):
        zero_centered_weight = zero_centered_weight.to_local()
    return zero_rms_norm_gate(x, gate, zero_centered_weight, eps)


def _apply_partial_rope(x: Tensor, positions: Tensor, rotary_base: float, width: int) -> Tensor:
    inv_freq = 1.0 / (rotary_base ** (torch.arange(0, width, 2, device=x.device).float() / width))
    phase = positions.float().unsqueeze(-1) * inv_freq
    phase = torch.cat((phase, phase), dim=-1).unsqueeze(-2)
    part = x[..., :width]
    part = torch.cat((part[..., 0::2], part[..., 1::2]), dim=-1)
    first, second = part.chunk(2, dim=-1)
    rotated = part * phase.cos().to(x.dtype) + torch.cat((-second, first), dim=-1) * phase.sin().to(x.dtype)
    return torch.cat((rotated, x[..., width:]), dim=-1)


def _causal_attention(query: Tensor, key: Tensor, value: Tensor, seq_ctx: SequenceContext, scale: float) -> Tensor:
    cu = seq_ctx.cu_seq_lens_q
    if query.is_cuda:
        from xtuner.v1.ops import flash_attn_varlen_func

        if flash_attn_varlen_func is not None:
            # The repository FA2 wrapper assumes equal Q/K/V head widths;
            # MLA uses a smaller V width. FA3 also requires alignment to 8.
            # Keep the explicit QK scale and trim
            # the zero-padded value channels after attention.
            value_width = value.shape[-1]
            head_width = math.ceil(max(query.shape[-1], value_width) / 8) * 8
            if query.shape[-1] != head_width:
                query = F.pad(query, (0, head_width - query.shape[-1]))
                key = F.pad(key, (0, head_width - key.shape[-1]))
            if value_width != head_width:
                value = F.pad(value, (0, head_width - value_width))
            output = flash_attn_varlen_func(
                query.squeeze(0).contiguous(),
                key.squeeze(0).contiguous(),
                value.squeeze(0).contiguous(),
                cu_seqlens_q=cu,
                cu_seqlens_k=seq_ctx.cu_seq_lens_k,
                max_seqlen_q=seq_ctx.max_length_q,
                max_seqlen_k=seq_ctx.max_length_k,
                causal=True,
                softmax_scale=scale,
            )
            return output[..., :value_width].unsqueeze(0)
    if cu.numel() == 2:
        return F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), is_causal=True, scale=scale
        ).transpose(1, 2)
    length = query.shape[1]
    doc_ids = torch.empty(length, dtype=torch.long, device=query.device)
    bounds = cu.detach().cpu().tolist()
    for document, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        doc_ids[int(start) : int(end)] = document
    index = torch.arange(length, device=query.device)
    allowed = (doc_ids[:, None] == doc_ids[None, :]) & (index[None, :] <= index[:, None])
    return F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=allowed.view(1, 1, length, length),
        scale=scale,
    ).transpose(1, 2)


class NCPQwen35V2MLA(nn.Module):
    """QK-normalized MLA: Q reads the Q stream and KV reads the K stream."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_head_dim: int,
        qk_pos_emb_head_dim: int,
        v_head_dim: int,
        rotary_base: float,
        rms_norm_eps: float,
        layer_idx: int = 0,
        float8_cfg: Float8Config | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_head_dim
        self.qk_pos_emb_head_dim = qk_pos_emb_head_dim
        self.v_head_dim = v_head_dim
        self.rotary_base = rotary_base
        self.q_down = build_linear(hidden_size, q_lora_rank, bias=False, float8_cfg=float8_cfg)
        self.q_norm = ZeroCenteredRMSNorm(q_lora_rank, rms_norm_eps)
        self.q_up = build_linear(
            q_lora_rank,
            num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim),
            bias=False,
            float8_cfg=float8_cfg,
        )
        self.kv_down = build_linear(
            hidden_size,
            kv_lora_rank + qk_pos_emb_head_dim,
            bias=False,
            float8_cfg=float8_cfg,
        )
        self.kv_norm = ZeroCenteredRMSNorm(kv_lora_rank, rms_norm_eps)
        self.kv_up = build_linear(
            kv_lora_rank,
            num_attention_heads * (qk_head_dim + v_head_dim),
            bias=False,
            float8_cfg=float8_cfg,
        )
        self.q_content_norm = nn.RMSNorm(qk_head_dim, eps=rms_norm_eps)
        self.k_content_norm = nn.RMSNorm(qk_head_dim, eps=rms_norm_eps)
        self.q_pos_norm = nn.RMSNorm(qk_pos_emb_head_dim, eps=rms_norm_eps)
        self.k_pos_norm = nn.RMSNorm(qk_pos_emb_head_dim, eps=rms_norm_eps)
        self.out_proj = build_linear(
            num_attention_heads * v_head_dim,
            hidden_size,
            bias=False,
            float8_cfg=float8_cfg,
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor] | None,
        seq_ctx: SequenceContext,
        mudd_inputs: tuple[Tensor, Tensor, Tensor] | None = None,
    ) -> AttnOutputs:
        del position_embeddings
        q_input, k_input, _ = mudd_inputs if mudd_inputs is not None else (hidden_states,) * 3
        if seq_ctx.position_ids is None:
            raise ValueError("NCP Qwen3.5 v2 MLA requires position_ids")
        positions = seq_ctx.position_ids

        query = self.q_up(self.q_norm(self.q_down(q_input)))
        latent, key_pos = self.kv_down(k_input).split((self.kv_lora_rank, self.qk_pos_emb_head_dim), dim=-1)
        kv_up = self.kv_up(self.kv_norm(latent))
        query = query.reshape(*q_input.shape[:2], self.num_heads, -1)
        key, value = kv_up.reshape(*k_input.shape[:2], self.num_heads, -1).split(
            (self.qk_head_dim, self.v_head_dim), dim=-1
        )
        query_content, query_pos = query.split((self.qk_head_dim, self.qk_pos_emb_head_dim), dim=-1)
        query_pos = _apply_partial_rope(
            self.q_pos_norm(query_pos), positions, self.rotary_base, self.qk_pos_emb_head_dim
        )
        key_pos = _apply_partial_rope(
            self.k_pos_norm(key_pos.unsqueeze(2)), positions, self.rotary_base, self.qk_pos_emb_head_dim
        ).expand(-1, -1, self.num_heads, -1)
        query = torch.cat((self.q_content_norm(query_content), query_pos), dim=-1)
        key = torch.cat((self.k_content_norm(key), key_pos), dim=-1)
        sp_mesh = seq_ctx.sequence_parallel_mesh
        if sp_mesh is not None and sp_mesh.size() > 1:
            if self.num_heads % sp_mesh.size():
                raise ValueError(
                    f"MLA heads ({self.num_heads}) must be divisible by SP size ({sp_mesh.size()})"
                )
            query = _mla_query_all_to_all(query, seq_ctx)
            key = _mla_key_all_to_all(key, seq_ctx)
            value = _mla_value_all_to_all(value, seq_ctx)
        scale = 1.0 / math.sqrt(self.qk_head_dim + self.qk_pos_emb_head_dim)
        raw_output = _causal_attention(query, key, value, seq_ctx, scale)
        if sp_mesh is not None and sp_mesh.size() > 1:
            raw_output = _mla_output_all_to_all(raw_output, seq_ctx)
        raw_output = raw_output.reshape(*q_input.shape[:2], -1)
        return {"raw_output": raw_output, "projected_output": self.out_proj(raw_output)}


def _cpu_delta_rule(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor) -> Tensor:
    state = torch.zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3], device=q.device, dtype=torch.float32)
    outputs: list[Tensor] = []
    for index in range(q.shape[1]):
        state = state * g[:, index].float().exp()[..., None, None]
        key = k[:, index].float()
        residual = v[:, index].float() - torch.einsum("bhk,bhkv->bhv", key, state)
        state = state + key[..., None] * (residual * beta[:, index].float()[..., None])[..., None, :]
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, index].float() / math.sqrt(q.shape[-1]), state))
    return torch.stack(outputs, dim=1).to(v.dtype)


def _cpu_conv_documents(qkv: Tensor, weight: Tensor, cu_seq_lens: Tensor) -> Tensor:
    bounds = cu_seq_lens.detach().cpu().tolist()
    pieces: list[Tensor] = []
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        piece = qkv[:, int(start) : int(end)]
        convolved = F.conv1d(
            piece.transpose(1, 2),
            weight.unsqueeze(1),
            padding=weight.shape[1] - 1,
            groups=weight.shape[0],
        )
        pieces.append(F.silu(convolved[..., : piece.shape[1]]).transpose(1, 2))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)


def _cpu_delta_documents(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, cu_seq_lens: Tensor) -> Tensor:
    bounds = cu_seq_lens.detach().cpu().tolist()
    pieces: list[Tensor] = []
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        start_i, end_i = int(start), int(end)
        pieces.append(
            _cpu_delta_rule(
                q[:, start_i:end_i],
                k[:, start_i:end_i],
                v[:, start_i:end_i],
                g[:, start_i:end_i],
                beta[:, start_i:end_i],
            )
        )
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)


def _delta_rule_cu_seqlens(seq_ctx: SequenceContext) -> Tensor | None:
    cu_seqlens = seq_ctx.cu_seq_lens_q
    return cu_seqlens if cu_seqlens.numel() > 2 else None


class NCPQwen35V2GatedDeltaNet(nn.Module):
    """GDN preserving the checkpoint's single fused input projection."""

    def __init__(
        self,
        hidden_size: int,
        num_value_heads: int,
        num_key_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_dim: int,
        hidden_act: str,
        rms_norm_eps: float,
        layer_idx: int = 0,
        float8_cfg: Float8Config | None = None,
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError("NCP Qwen3.5 v2 GDN requires silu")
        self.layer_idx = layer_idx
        self.num_k_heads = num_key_heads
        self.num_v_heads = num_value_heads
        self.head_k_dim = key_head_dim
        self.head_v_dim = value_head_dim
        self.key_dim = num_key_heads * key_head_dim
        self.value_dim = num_value_heads * value_head_dim
        self.conv_kernel_dim = conv_kernel_dim
        self.use_fused_qk_l2 = _production_feature_enabled("XTUNER_GDN_FUSED_QK_L2")
        self.use_fused_norm_gate = _production_feature_enabled("XTUNER_GDN_FUSED_NORM_GATE")
        conv_dim = 2 * self.key_dim + self.value_dim
        self.in_proj = nn.Linear(hidden_size, conv_dim + self.value_dim + 2 * self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, conv_kernel_dim, groups=conv_dim, padding=conv_kernel_dim - 1, bias=False
        )
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads))
        self.out_norm = ZeroCenteredRMSNorm(self.head_v_dim, rms_norm_eps)
        self.out_proj = build_linear(
            self.value_dim,
            hidden_size,
            bias=False,
            float8_cfg=float8_cfg,
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor] | None,
        seq_ctx: SequenceContext,
        mudd_inputs: tuple[Tensor, Tensor, Tensor] | None = None,
    ) -> AttnOutputs:
        del position_embeddings
        q_input, k_input, v_input = mudd_inputs if mudd_inputs is not None else (hidden_states,) * 3
        sizes = (self.key_dim, self.key_dim, self.value_dim, self.value_dim, self.num_v_heads, self.num_v_heads)
        weights = self.in_proj.weight.split(sizes, dim=0)
        query = F.linear(q_input, weights[0])
        key = F.linear(k_input, weights[1])
        value = F.linear(v_input, weights[2])
        z = F.linear(q_input, weights[3])
        beta = F.linear(q_input, weights[4])
        alpha = F.linear(q_input, weights[5])
        weight = self.conv1d.weight.squeeze(1)
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        sp_mesh = seq_ctx.sequence_parallel_mesh
        if sp_mesh is not None and sp_mesh.size() > 1:
            sp_size = sp_mesh.size()
            if self.num_k_heads % sp_size or self.num_v_heads % sp_size:
                raise ValueError(
                    "GDN key/value heads "
                    f"({self.num_k_heads}/{self.num_v_heads}) must be divisible by SP size ({sp_size})"
                )
            query = _gdn_query_all_to_all(query.transpose(1, 2), seq_ctx).transpose(1, 2).contiguous()
            key = _gdn_key_all_to_all(key.transpose(1, 2), seq_ctx).transpose(1, 2).contiguous()
            value = _gdn_value_all_to_all(value.transpose(1, 2), seq_ctx).transpose(1, 2).contiguous()
            query_weight, key_weight, value_weight = weight.split(
                (self.key_dim, self.key_dim, self.value_dim), dim=0
            )
            rank = seq_ctx.sp_rank
            weight = torch.cat(
                (
                    query_weight.chunk(sp_size, dim=0)[rank],
                    key_weight.chunk(sp_size, dim=0)[rank],
                    value_weight.chunk(sp_size, dim=0)[rank],
                ),
                dim=0,
            )
        qkv = torch.cat((query, key, value), dim=-1)
        seq_idx = gen_seq_idx(qkv.shape[1], seq_ctx.cu_seq_lens_q)
        if qkv.is_cuda:
            from xtuner.v1.ops.gated_deltanet import get_causal_conv1d_fn

            qkv = get_causal_conv1d_fn()(
                x=qkv, weight=weight, bias=None, activation="silu", seq_idx=seq_idx
            )
        else:
            qkv = _cpu_conv_documents(qkv, weight.to(qkv.dtype), seq_ctx.cu_seq_lens_q)
        sp_size = sp_mesh.size() if sp_mesh is not None else 1
        local_key_dim = self.key_dim // sp_size
        local_value_dim = self.value_dim // sp_size
        query_key, value = qkv.split((2 * local_key_dim, local_value_dim), dim=-1)
        local_k_heads = self.num_k_heads // sp_size
        local_v_heads = self.num_v_heads // sp_size
        query_key = query_key.reshape(*qkv.shape[:2], 2 * local_k_heads, self.head_k_dim)
        value = value.reshape(*qkv.shape[:2], local_v_heads, self.head_v_dim)
        use_kernel_qk_l2 = qkv.is_cuda and self.use_fused_qk_l2
        if not use_kernel_qk_l2:
            query_key = query_key.float() * torch.rsqrt(
                query_key.float().square().sum(-1, keepdim=True) + 1.0e-6
            )
            query_key = query_key.to(qkv.dtype)
        query, key = query_key.split(local_k_heads, dim=2)
        repeats = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
        a_log = self.A_log.to_local() if isinstance(self.A_log, DTensor) else self.A_log
        dt_bias = self.dt_bias.to_local() if isinstance(self.dt_bias, DTensor) else self.dt_bias
        decay = -a_log.float().exp() * F.softplus(alpha.float() + dt_bias)
        beta = beta.sigmoid()
        if sp_mesh is not None and sp_mesh.size() > 1:
            decay = _gdn_decay_all_to_all(decay.transpose(1, 2), seq_ctx).transpose(1, 2).contiguous()
            beta = _gdn_beta_all_to_all(beta.transpose(1, 2), seq_ctx).transpose(1, 2).contiguous()
        if qkv.is_cuda:
            from xtuner.v1.ops.gated_deltanet import get_chunk_gated_delta_rule_fn

            output, _ = get_chunk_gated_delta_rule_fn()(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                g=decay.contiguous(),
                beta=beta.contiguous(),
                use_qk_l2norm_in_kernel=use_kernel_qk_l2,
                cu_seqlens=_delta_rule_cu_seqlens(seq_ctx),
            )
        else:
            output = _cpu_delta_documents(query, key, value, decay, beta, seq_ctx.cu_seq_lens_q)
        if sp_mesh is not None and sp_mesh.size() > 1:
            output = _gdn_output_all_to_all(output, seq_ctx)
        gate = z.reshape_as(output)
        if output.is_cuda and self.use_fused_norm_gate:
            gated = _fused_zero_rms_norm_gate(output, gate, self.out_norm.weight, self.out_norm.eps)
        else:
            normalized = self.out_norm(output)
            gated = (normalized.float() * F.silu(gate.float())).to(qkv.dtype)
        raw_output = gated.flatten(-2)
        return {"raw_output": raw_output, "projected_output": self.out_proj(raw_output)}


__all__ = [
    "NCPQwen35V2GatedDeltaNet",
    "NCPQwen35V2GatedDeltaNetConfig",
    "NCPQwen35V2MLA",
    "NCPQwen35V2MLAConfig",
]
