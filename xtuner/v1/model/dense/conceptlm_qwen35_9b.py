"""Qwen3.5 ConceptLM v2: the NCP three-tower contract on the 9B tower.

Same training shell as :class:`ConceptLMQwen35` (packed ``SequenceContext``,
FSDP, LM loss). The forward matches HuggingFace ``NCPQwen35V2ForCausalLM``:

    tokens -> encoder (block self-MUDD)
           -> chunk-4 mean pool
           -> predictor (self-MUDD + encoder cross-MUDD)
           -> softmax over 32x128 codebooks
           -> fusion RMSNorm mix into the decoder
           -> decoder (self-MUDD, encoder/predictor cross-MUDD, per-layer concept read)
           -> LM head

Full-attention layers are QK-normed MLA. GatedDeltaNet keeps the 9B head
layout. The inherited 9B ``attention`` MHA spec is unused.
"""

import math
from itertools import accumulate
from typing import Literal, Self, cast

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as distF
import torch.nn.functional as F
from pydantic import model_validator
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from tqdm import tqdm
from typing_extensions import override

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss import BaseLossContext
from xtuner.v1.loss.conceptlm_loss import ConceptLMLossConfig, ConceptLMLossContext
from xtuner.v1.model.base import DEFAULT_FLOAT8_CFG, BaseModel, HFSaveCfg, TorchCompileOption
from xtuner.v1.model.utils import apply_activation_checkpointing
from xtuner.v1.module import LMHead, RMSNorm
from xtuner.v1.module.attention.concept_qwen35 import (
    NCPQwen35V2GatedDeltaNetConfig,
    NCPQwen35V2MLAConfig,
)
from xtuner.v1.module.concept_sequence import document_lengths, expand_concepts, merge_chunks
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer
from xtuner.v1.module.decoder_layer.fused_swiglu import FusedSwiGLUMLP
from xtuner.v1.module.linear import build_linear
from xtuner.v1.module.rms_norm.concept_qwen35 import (
    GatedDeltaNetInputRMSNorm,
    MuddRMSNorm,
    ZeroCenteredRMSNorm,
)
from xtuner.v1.module.rms_norm.concept_qwen35 import (
    effective_weight as _effective_weight,
)
from xtuner.v1.module.rms_norm.concept_qwen35 import (
    env_on as _env_on,
)
from xtuner.v1.module.rms_norm.concept_qwen35 import (
    mudd_rms_norm as _mudd_rms_norm,
)
from xtuner.v1.ops.mudd.source_scale import scale_source
from xtuner.v1.ops.rms_norm.native_backward import rms_norm_native_backward
from xtuner.v1.utils import get_device, log_rank0

from .dense import DenseModelOutputs
from .qwen3_5_text import Qwen3_5_VLTextDense9BConfig


DEVICE = get_device()
Streams = tuple[Tensor, Tensor, Tensor, Tensor]

CONCEPTLM_QWEN35_V2_COMPILE_CFG: dict[str, TorchCompileOption] = {
    "xtuner.v1.module.decoder_layer.dense_decoder_layer.DenseDecoderLayer.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.model.dense.conceptlm_qwen35_9b.SelfMUDDRoute.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.model.dense.conceptlm_qwen35_9b.CrossMUDDRoute.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.model.dense.conceptlm_qwen35_9b.PredictedConceptInjection.forward": TorchCompileOption(fullgraph=True),
    **DEFAULT_FLOAT8_CFG,
}


def _block_end_indices(num_layers: int, block_size: int) -> list[int]:
    indices = list(range(block_size - 1, num_layers, block_size))
    if num_layers % block_size:
        indices.append(num_layers - 1)
    return indices


def _arange_positions(lengths: list[int], device: torch.device) -> Tensor:
    return torch.cat([torch.arange(length, device=device) for length in lengths], dim=0).unsqueeze(0)


def _hf_padded_lengths(lengths: list[int], chunk_size: int) -> list[int]:
    """HF pads each sequence to at least two chunks and a chunk multiple."""
    padded: list[int] = []
    for length in lengths:
        padded.append(max(2 * chunk_size, math.ceil(length / chunk_size) * chunk_size))
    return padded


def _context_from_lengths(
    input_ids: Tensor | None,
    inputs_embeds: Tensor | None,
    lengths: list[int],
    device: torch.device,
) -> SequenceContext:
    cu = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device)
    return SequenceContext(
        input_ids=cast(torch.LongTensor | None, input_ids),
        inputs_embeds=cast(torch.FloatTensor | None, inputs_embeds),
        cu_seq_lens_q=cast(torch.IntTensor, cu),
        cu_seq_lens_k=cast(torch.IntTensor, cu),
        max_length_q=max(lengths),
        max_length_k=max(lengths),
        num_padding=0,
        device=device,
        position_ids=cast(torch.LongTensor, _arange_positions(lengths, device)),
    )


def _gather_sequence(hidden_states: Tensor, sp_mesh: DeviceMesh | None) -> Tensor:
    if sp_mesh is None or sp_mesh.size() == 1:
        return hidden_states
    gathered = distF.all_gather(hidden_states.contiguous(), group=sp_mesh.get_group())
    return torch.cat(gathered, dim=1).contiguous()


def _split_sequence(hidden_states: Tensor, sp_mesh: DeviceMesh | None) -> Tensor:
    if sp_mesh is None or sp_mesh.size() == 1:
        return hidden_states
    if hidden_states.shape[1] % sp_mesh.size():
        raise ValueError(f"sequence length ({hidden_states.shape[1]}) must be divisible by SP size ({sp_mesh.size()})")
    shard_size = hidden_states.shape[1] // sp_mesh.size()
    shard_start = sp_mesh.get_local_rank() * shard_size
    return hidden_states[:, shard_start : shard_start + shard_size].contiguous()


def _sp_context_from_lengths(
    inputs_embeds: Tensor,
    lengths: list[int],
    sp_mesh: DeviceMesh,
    chunk_size: int = 1,
) -> SequenceContext:
    total_length = sum(lengths)
    if total_length != inputs_embeds.shape[1] * sp_mesh.size():
        raise ValueError("SP context local tensor does not cover the global packed sequence")
    if total_length % (sp_mesh.size() * chunk_size):
        raise ValueError("SP context length must preserve chunk-aligned equal shards")
    cu = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=inputs_embeds.device)
    full_positions = _arange_positions(lengths, inputs_embeds.device)
    shard_size = inputs_embeds.shape[1]
    shard_start = sp_mesh.get_local_rank() * shard_size
    return SequenceContext(
        input_ids=None,
        inputs_embeds=cast(torch.FloatTensor, inputs_embeds),
        cu_seq_lens_q=cast(torch.IntTensor, cu),
        cu_seq_lens_k=cast(torch.IntTensor, cu),
        max_length_q=max(lengths),
        max_length_k=max(lengths),
        num_padding=0,
        device=inputs_embeds.device,
        position_ids=cast(torch.LongTensor, full_positions[:, shard_start : shard_start + shard_size]),
        sequence_parallel_mesh=sp_mesh,
        shard_start=shard_start,
        shard_size=shard_size,
        valid_sequence_length=total_length,
    )


def _crop_documents(hidden: Tensor, orig_lengths: list[int], padded_lengths: list[int]) -> Tensor:
    if orig_lengths == padded_lengths:
        return hidden
    pieces: list[Tensor] = []
    start = 0
    for orig, padded in zip(orig_lengths, padded_lengths, strict=True):
        pieces.append(hidden[:, start : start + orig])
        start += padded
    return torch.cat(pieces, dim=1)


class SelfMUDDRoute(nn.Module):
    """Block self-MUDD.

    Parameter names and shapes match ``model.mudd.routes.*``.
    """

    def __init__(self, hidden_size: int, layer_index: int, depth: int, block_size: int, eps: float) -> None:
        super().__init__()
        self.num_sources = 2 + (layer_index - 1) // block_size
        self.final_only_r = layer_index == depth
        self.compute_streams = 1 if self.final_only_r else 4
        router_width = self.num_sources if self.final_only_r else 4 * self.num_sources
        self.eps = eps
        self.source_norm = MuddRMSNorm(hidden_size, eps)
        self.router_norm = MuddRMSNorm(hidden_size, eps)
        self.post_norm = nn.Parameter(torch.full((self.compute_streams, hidden_size), -0.999))
        self.router_w1 = nn.Linear(hidden_size, router_width, bias=False)
        self.router_w2 = nn.Linear(router_width, self.compute_streams * self.num_sources, bias=False)
        self.static_weight = nn.Parameter(torch.zeros(self.compute_streams, self.num_sources))

    def forward(self, current_raw: Tensor, history_raw: tuple[Tensor, ...]) -> Streams:
        if len(history_raw) + 1 != self.num_sources:
            raise ValueError(f"Self-MUDD expected {self.num_sources - 1} history sources, got {len(history_raw)}")
        hidden = F.gelu(self.router_w1(self.router_norm(current_raw)))
        weights = self.router_w2(hidden) + self.static_weight.reshape(1, 1, -1)
        weights = weights.reshape(*current_raw.shape[:2], self.compute_streams, self.num_sources).softmax(dim=-1)
        sources = tuple(self.source_norm(source) for source in (*history_raw, current_raw))
        mixed = []
        for role in range(self.compute_streams):
            mixed_role = sources[0] * weights[..., role, 0:1]
            for source_index, source in enumerate(sources[1:], start=1):
                mixed_role = mixed_role + source * weights[..., role, source_index : source_index + 1]
            mixed.append(mixed_role)
        outputs = tuple(
            current_raw + _mudd_rms_norm(value, self.post_norm[role], self.eps) for role, value in enumerate(mixed)
        )
        if self.final_only_r:
            return (outputs[0], outputs[0], outputs[0], outputs[0])
        return cast(Streams, outputs)


def _cross_source_units(candidate_pool: tuple[Tensor, ...], eps: float) -> tuple[Tensor, ...]:
    """Normalize an immutable source bank once per stack forward.

    Keep this cache on the autograd graph: each route has its own scale, but
    all routes in a group share the same source tensors and epsilon.
    """
    return tuple(rms_norm_native_backward(source, None, eps) for source in candidate_pool)


class CrossMUDDRoute(nn.Module):
    """Four-role cross route.

    The mix is added onto the current self-MUDD streams.
    """

    def __init__(self, hidden_size: int, num_sources: int, eps: float) -> None:
        super().__init__()
        self.num_sources = int(num_sources)
        self.eps = float(eps)
        width = 4 * self.num_sources
        self.router_norm_weight = nn.Parameter(torch.zeros(hidden_size))
        self.source_norm_weight = nn.Parameter(torch.zeros(hidden_size))
        self.post_norm_weight = nn.Parameter(torch.zeros(4, hidden_size))
        self.router_w1 = nn.Linear(hidden_size, width, bias=False)
        self.router_w2 = nn.Linear(width, width, bias=False)
        self.static_logits = nn.Parameter(torch.zeros(4, self.num_sources))
        self.gate_param = nn.Parameter(torch.full((4,), math.atanh(0.02)))

    def route_weights(self, current_raw: Tensor, current_inv_rms: Tensor | None) -> Tensor:
        scale = _effective_weight(self.router_norm_weight)
        if current_inv_rms is None:
            router_hidden = _mudd_rms_norm(current_raw, self.router_norm_weight, self.eps)
        else:
            router_hidden = current_raw * current_inv_rms * scale
        logits = F.linear(F.gelu(F.linear(router_hidden, self.router_w1.weight)), self.router_w2.weight)
        logits = logits + self.static_logits.reshape(1, 1, -1)
        return logits.reshape(*current_raw.shape[:2], 4, self.num_sources).softmax(dim=-1)

    def forward(
        self,
        current_raw: Tensor,
        candidate_pool: tuple[Tensor, ...],
        streams: Streams,
        current_inv_rms: Tensor | None = None,
        source_units: tuple[Tensor, ...] | None = None,
    ) -> Streams:
        if len(candidate_pool) != self.num_sources:
            raise ValueError(f"Cross-MUDD expected {self.num_sources} sources, got {len(candidate_pool)}")
        weights = self.route_weights(current_raw, current_inv_rms)
        gates = torch.tanh(self.gate_param)
        if source_units is None:
            sources = tuple(_mudd_rms_norm(source, self.source_norm_weight, self.eps) for source in candidate_pool)
        else:
            if len(source_units) != self.num_sources:
                raise ValueError("Cross-MUDD cached source count differs from the source bank")
            scale = _effective_weight(self.source_norm_weight)
            sources = tuple(scale_source(unit, scale) for unit in source_units)
        outputs: list[Tensor] = []
        for role, stream in enumerate(streams):
            mixed = cast(
                Tensor, sum(source * weights[..., role, index : index + 1] for index, source in enumerate(sources))
            )
            delta = gates[role] * _mudd_rms_norm(mixed, self.post_norm_weight[role], self.eps)
            outputs.append(stream + delta)
        return (outputs[0], outputs[1], outputs[2], outputs[3])


class PredictedConceptInjection(nn.Module):
    """Per-decoder-layer read of the aligned predicted concept."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = MuddRMSNorm(hidden_size, eps)
        self.final_diag = nn.Parameter(torch.full((hidden_size,), 0.05))

    def forward(self, concept: Tensor) -> Tensor:
        normalized = rms_norm_native_backward(concept, _effective_weight(self.norm.weight), self.norm.eps)
        return normalized * self.final_diag


class MUDDRoutes(nn.Module):
    """Self-MUDD parameters, one route per layer of each tower."""

    def __init__(self, config: "ConceptLMQwen35V2Config") -> None:
        super().__init__()
        block = config.self_mudd_block_size
        eps = config.mudd_epsilon

        def routes(depth: int) -> nn.ModuleList:
            return nn.ModuleList(
                SelfMUDDRoute(config.hidden_size, index + 1, depth, block, eps) for index in range(depth)
            )

        self.routes = nn.ModuleDict(
            {
                "encoder": routes(config.concept_encoder_layers),
                "predictor": routes(config.concept_predictor_layers),
                "decoder": routes(config.concept_decoder_layers),
            }
        )


class ConceptLMQwen35V2Stack(nn.Module):
    """One tower: hybrid layers, block self-MUDD, optional cross routes and concept reads."""

    _self_routes: nn.ModuleList

    def __init__(
        self,
        config: "ConceptLMQwen35V2Config",
        depth: int,
        self_routes: nn.ModuleList,
        cross_source_counts: tuple[int, ...] = (),
        inject_concept: bool = False,
    ) -> None:
        super().__init__()
        self.block_size = config.self_mudd_block_size
        self.cache_cross_source_rms = _env_on("NCP_Q35V2_CACHE_CROSS_SOURCE_RMS")
        # HF shares one inv-RMS using the tower layernorm epsilon, not the cross-route epsilon.
        self.route_eps = config.rms_norm_eps
        self.layers = nn.ModuleDict({str(index): self._build_layer(config, index) for index in range(depth)})
        self.final_norm = ZeroCenteredRMSNorm(config.hidden_size, config.rms_norm_eps)
        object.__setattr__(self, "_self_routes", self_routes)
        self.cross_route_groups = nn.ModuleList(
            nn.ModuleList(CrossMUDDRoute(config.hidden_size, count, config.cross_mudd_epsilon) for _ in range(depth))
            for count in cross_source_counts
        )
        self.concept_routes = (
            nn.ModuleList(
                PredictedConceptInjection(config.hidden_size, config.concept_route_epsilon) for _ in range(depth)
            )
            if inject_concept
            else None
        )

    @staticmethod
    def _build_layer(config: "ConceptLMQwen35V2Config", index: int) -> DenseDecoderLayer:
        attention_config: NCPQwen35V2MLAConfig | NCPQwen35V2GatedDeltaNetConfig
        input_norm: nn.Module
        if index % 4 == 3:
            attention_config = NCPQwen35V2MLAConfig(
                num_attention_heads=config.mla_num_heads,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_head_dim=config.qk_head_dim,
                qk_pos_emb_head_dim=config.qk_pos_emb_head_dim,
                v_head_dim=config.v_head_dim,
                rotary_base=config.mla_rotary_base,
                rms_norm_eps=config.rms_norm_eps,
            )
            input_norm = ZeroCenteredRMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            assert config.linear_attention is not None
            attention_config = NCPQwen35V2GatedDeltaNetConfig(**config.linear_attention.model_dump())
            input_norm = GatedDeltaNetInputRMSNorm(config.hidden_size, config.rms_norm_eps)
        return DenseDecoderLayer(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            mlp_bias=False,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            attention_config=attention_config,
            rope_scaling_cfg=config.rope_scaling_cfg,
            generate_config=config.generate_config,
            float8_cfg=config.float8_cfg,
            layer_type="full_attention" if index % 4 == 3 else None,
            layer_idx=index,
            mlp=FusedSwiGLUMLP(
                config.hidden_size,
                config.intermediate_size,
                bias=False,
                float8_cfg=config.float8_cfg,
            ),
            input_layernorm=input_norm,
            post_attention_layernorm=ZeroCenteredRMSNorm(config.hidden_size, config.rms_norm_eps),
            attention_output_norm=MuddRMSNorm(config.hidden_size, config.rms_norm_eps),
            mlp_output_norm=MuddRMSNorm(config.hidden_size, config.rms_norm_eps),
        )

    def forward(
        self,
        hidden_states: Tensor,
        seq_ctx: SequenceContext,
        cross_banks: tuple[tuple[Tensor, ...], ...] = (),
        aligned_predicted_concept: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        if len(cross_banks) != len(self.cross_route_groups):
            raise ValueError("cross route and bank counts differ")
        use_dual_kernel = len(self.cross_route_groups) == 2 and _env_on("NCP_Q35V2_TRITON_DUAL_CROSS_MUDD")
        if use_dual_kernel and (self.training or torch.is_grad_enabled()):
            raise RuntimeError("Dual Cross-MUDD is inference-only; use eval() and disable autograd")
        # Scope the cache to this invocation so weights/inputs cannot go stale
        # across micro-batches, optimizer updates, or checkpoint recomputation.
        use_source_cache = (
            self.cache_cross_source_rms and _env_on("NCP_Q35V2_MUDD_RMS", "1") and not use_dual_kernel
        )
        source_units: tuple[tuple[Tensor, ...], ...] = ()
        if use_source_cache:
            source_units = tuple(
                _cross_source_units(bank, routes[0].eps) if len(routes) else ()
                for routes, bank in zip(self.cross_route_groups, cross_banks, strict=True)
            )
        streams: Streams = (hidden_states,) * 4
        self_history: tuple[Tensor, ...] = (hidden_states,)
        raw_outputs: list[Tensor] = []
        for idx, layer in self.layers.items():
            index = int(idx)
            current_raw = layer(
                streams[3],
                position_embeddings=None,
                seq_ctx=seq_ctx,
                mudd_streams=streams,
            )
            assert isinstance(current_raw, Tensor)
            raw_outputs.append(current_raw)
            streams = self._self_routes[index](current_raw, self_history)
            shared_inv_rms = None
            if self.cross_route_groups:
                shared_inv_rms = torch.rsqrt(
                    current_raw.float().square().mean(dim=-1, keepdim=True) + self.route_eps
                ).to(dtype=current_raw.dtype)
            # Explicit inference experiment; training uses the differentiable routes.
            if use_dual_kernel:
                from xtuner.v1.ops.mudd.concept_qwen35 import dual_cross_mudd_streams

                route0 = cast(CrossMUDDRoute, cast(nn.ModuleList, self.cross_route_groups[0])[index])
                route1 = cast(CrossMUDDRoute, cast(nn.ModuleList, self.cross_route_groups[1])[index])
                bank0, bank1 = cross_banks
                if len(bank0) != route0.num_sources or len(bank1) != route1.num_sources:
                    raise ValueError("dual Cross-MUDD source count mismatch")
                streams = dual_cross_mudd_streams(
                    route0.route_weights(current_raw, shared_inv_rms),
                    bank0,
                    _effective_weight(route0.source_norm_weight),
                    _effective_weight(route0.post_norm_weight),
                    torch.tanh(route0.gate_param),
                    route0.eps,
                    route1.route_weights(current_raw, shared_inv_rms),
                    bank1,
                    _effective_weight(route1.source_norm_weight),
                    _effective_weight(route1.post_norm_weight),
                    torch.tanh(route1.gate_param),
                    route1.eps,
                    streams,
                )
            else:
                for group_index, (routes, bank) in enumerate(zip(self.cross_route_groups, cross_banks, strict=True)):
                    if use_source_cache:
                        streams = routes[index](current_raw, bank, streams, shared_inv_rms, source_units[group_index])
                    else:
                        streams = routes[index](current_raw, bank, streams, shared_inv_rms)
            if self.concept_routes is not None:
                if aligned_predicted_concept is None:
                    raise ValueError("decoder concept injection requires concepts")
                delta = self.concept_routes[index](aligned_predicted_concept)
                streams = tuple(stream + delta for stream in streams)
            if index % self.block_size == self.block_size - 1:
                self_history = (*self_history, current_raw)
        return self.final_norm(streams[3]), tuple(raw_outputs)


def _cross_source_count(num_layers: int, block_size: int) -> int:
    return 1 + (num_layers + block_size - 1) // block_size


class ConceptLMQwen35V2Config(Qwen3_5_VLTextDense9BConfig):
    """Qwen3.5-9B ConceptLM v2 recipe.

    Depths and MLA match the pinned HF checkpoint.
    """

    # stack depths
    concept_encoder_layers: int = 8
    concept_predictor_layers: int = 16
    concept_decoder_layers: int = 8
    # concept transform
    concept_chunk_size: int = 4
    concept_chunk_merge_method: Literal["meanpooling", "first", "last"] = "meanpooling"
    concept_shift_feature: bool = True
    # residual VQ
    concept_codebook_size: int = 128
    concept_num_codebooks: int = 32
    concept_loss_cfg: ConceptLMLossConfig | None = None
    mla_num_heads: int = 24
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_head_dim: int = 192
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 256
    mla_rotary_base: float = 10_000_000.0
    self_mudd_block_size: int = 4
    cross_mudd_block_size: int = 4
    mudd_epsilon: float = 1.0e-6
    cross_mudd_epsilon: float = 1.0e-6
    concept_route_epsilon: float = 1.0e-5
    max_position_embeddings: int = 8192
    pad_token_id: int | None = 248044
    # The 9B tower remaps ``model.`` into ``model.language_model.``. NCP v2 stores the
    # backbone directly under ``model.``.
    hf_key_mapping: dict[str, str] | None = None
    hf_save_cfg: HFSaveCfg = HFSaveCfg(
        fp32_keys_pattern=[
            r"model\.(?:encoder|predictor|decoder)\.layers\.\d+\.attention\.A_log",
            r"model\.(?:encoder|predictor|decoder)\.layers\.\d+\.attention\.conv1d\.weight",
        ],
    )

    @model_validator(mode="after")
    def _check_v2_contract(self) -> Self:
        self.num_hidden_layers = (
            self.concept_encoder_layers + self.concept_predictor_layers + self.concept_decoder_layers
        )
        if self.hidden_size % self.concept_num_codebooks:
            raise ValueError("hidden_size must be divisible by concept_num_codebooks")
        if self.linear_attention is None:
            raise ValueError("v2 requires the 9B GatedDeltaNet config")
        if self.linear_attention.num_value_heads % self.linear_attention.num_key_heads:
            raise ValueError("linear value heads must be divisible by key heads")
        if self.self_mudd_block_size <= 0 or self.cross_mudd_block_size <= 0:
            raise ValueError("v2 requires positive MUDD block sizes")
        if self.qk_pos_emb_head_dim % 2:
            raise ValueError("MLA positional head dim must be even")
        if self.concept_loss_cfg is not None and self.concept_loss_cfg.concept_chunk_size != self.concept_chunk_size:
            raise ValueError("ConceptLM loss and model chunk sizes must agree")
        return self

    def build(self) -> "ConceptLMQwen35V2":  # type: ignore[override]
        return ConceptLMQwen35V2(self)


class ConceptLMQwen35V2(BaseModel):
    """Three-tower ConceptLM whose tensor contract follows NCP Qwen3.5 v2."""

    config: ConceptLMQwen35V2Config

    def __init__(self, config: ConceptLMQwen35V2Config):
        super().__init__(config)
        # HF keeps the pad row as a learned vector. padding_idx would zero it.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.mudd = MUDDRoutes(config)
        encoder_sources = _cross_source_count(config.concept_encoder_layers, config.cross_mudd_block_size)
        predictor_sources = _cross_source_count(config.concept_predictor_layers, config.cross_mudd_block_size)
        self.encoder = ConceptLMQwen35V2Stack(config, config.concept_encoder_layers, self.mudd.routes["encoder"])
        self.predictor = ConceptLMQwen35V2Stack(
            config,
            config.concept_predictor_layers,
            self.mudd.routes["predictor"],
            cross_source_counts=(encoder_sources,),
        )
        self.decoder = ConceptLMQwen35V2Stack(
            config,
            config.concept_decoder_layers,
            self.mudd.routes["decoder"],
            cross_source_counts=(encoder_sources, predictor_sources),
            inject_concept=True,
        )
        self.prediction_heads = nn.ModuleList(
            [
                build_linear(
                    config.hidden_size,
                    config.concept_codebook_size,
                    bias=True,
                    float8_cfg=config.float8_cfg,
                )
                for _ in range(config.concept_num_codebooks)
            ]
        )
        codebook_width = config.hidden_size // config.concept_num_codebooks
        self.codebooks = nn.ParameterList(
            nn.Parameter(torch.zeros(config.concept_codebook_size, codebook_width))
            for _ in range(config.concept_num_codebooks)
        )
        self.fusion_encoder_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, type="default")
        self.fusion_repeated_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, type="default")
        # FSDP rejects a 0-dim parameter. numel 1 keeps the same broadcast as HF's scalar.
        self.fusion_alpha = nn.Parameter(torch.ones(1))
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, bias=False)
        self._init_load_spec()
        self._maybe_enable_compile(self.compile_cfg)

    @property
    @override
    def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
        return CONCEPTLM_QWEN35_V2_COMPILE_CFG

    # --------------------------------------------------------------- forward
    def to_hf_key_list(self, key: str) -> list[str]:
        if key.startswith("lm_head."):
            return [key]
        key = key.replace(".self_attn.", ".attention.")
        key = key.replace(".input_layernorm.", ".input_norm.")
        key = key.replace(".post_attention_layernorm.", ".post_norm.")
        key = key.replace(".attention_output_norm.", ".self_attn_output_norm.")
        key = key.replace(".mlp.fc1.", ".fc1.")
        key = key.replace(".mlp.fc2.", ".fc2.")
        return [f"model.{key}"]

    def hf_tensor_to_canonical(self, name: str, loaded_tensor: Tensor) -> Tensor:
        if name == "fusion_alpha" and loaded_tensor.ndim == 0:
            return loaded_tensor.reshape(1)
        return loaded_tensor

    def param_to_safetensor(self, safetensor: Tensor, hf_param_name: str) -> Tensor:
        if hf_param_name == "model.fusion_alpha" and safetensor.shape == (1,):
            return safetensor.reshape(())
        return safetensor

    @override
    def _mark_dynamic(self, seq_ctx: SequenceContext, dim: int = 0) -> None:
        # GDN specializes the single-document dense path (two offsets). Allow
        # that specialization while keeping packed document counts dynamic.
        torch._dynamo.maybe_mark_dynamic(seq_ctx.cu_seq_lens_q, dim)
        torch._dynamo.maybe_mark_dynamic(seq_ctx.cu_seq_lens_k, dim)

    @staticmethod
    def _original_document_lengths(seq_ctx: SequenceContext) -> list[int]:
        sp_mesh = seq_ctx.sequence_parallel_mesh
        if sp_mesh is None or sp_mesh.size() == 1:
            return document_lengths(seq_ctx)
        valid_length = seq_ctx.valid_sequence_length
        boundaries = [int(value) for value in seq_ctx.cu_seq_lens_q.detach().cpu().tolist()]
        logical_boundaries = [value for value in boundaries if value <= valid_length]
        if not logical_boundaries or logical_boundaries[0] != 0:
            logical_boundaries.insert(0, 0)
        if logical_boundaries[-1] != valid_length:
            logical_boundaries.append(valid_length)
        return [end - start for start, end in zip(logical_boundaries[:-1], logical_boundaries[1:], strict=True)]

    def _token_context(
        self, seq_ctx: SequenceContext, orig_lengths: list[int]
    ) -> tuple[SequenceContext, list[int], list[int]]:
        config = self.config
        padded_lengths = _hf_padded_lengths(orig_lengths, config.concept_chunk_size)
        sp_mesh = seq_ctx.sequence_parallel_mesh
        use_sp = sp_mesh is not None and sp_mesh.size() > 1
        if padded_lengths == orig_lengths and not use_sp:
            return seq_ctx, padded_lengths, padded_lengths
        input_ids: Tensor | None = seq_ctx.raw_input_ids if use_sp else seq_ctx.input_ids
        if input_ids is None:
            raise ValueError("v2 pads short documents with pad_token_id and needs input_ids")
        pieces: list[Tensor] = []
        start = 0
        pad_id = config.pad_token_id if config.pad_token_id is not None else config.eos_token_id
        for orig, padded in zip(orig_lengths, padded_lengths, strict=True):
            document = input_ids[:, start : start + orig]
            start += orig
            if padded > orig:
                pad = document.new_full((1, padded - orig), pad_id)
                document = torch.cat((document, pad), dim=1)
            pieces.append(document)
        input_ids = torch.cat(pieces, dim=1)
        compute_lengths = list(padded_lengths)
        if not use_sp:
            token_ctx = _context_from_lengths(input_ids, None, compute_lengths, input_ids.device)
            return token_ctx, padded_lengths, compute_lengths

        assert sp_mesh is not None
        alignment = sp_mesh.size() * config.concept_chunk_size
        tail_padding = (-input_ids.shape[1]) % alignment
        if tail_padding:
            input_ids = torch.cat((input_ids, input_ids.new_full((1, tail_padding), pad_id)), dim=1)
            # Keep the logical document count identical to the non-SP path.
            # Treating alignment padding as a synthetic document changes the
            # packed-kernel branch (notably GDN's cu_seqlens path), and those
            # per-layer numerical differences are amplified by the full model.
            # Extending the final document is causal-safe: appended tokens
            # cannot affect its valid prefix.
            compute_lengths[-1] += tail_padding
        local_input_ids = _split_sequence(input_ids, sp_mesh)
        full_positions = _arange_positions(compute_lengths, input_ids.device)
        local_positions = _split_sequence(full_positions, sp_mesh)
        cu = torch.tensor([0, *accumulate(compute_lengths)], dtype=torch.int32, device=input_ids.device)
        shard_size = local_input_ids.shape[1]
        shard_start = seq_ctx.sp_rank * shard_size
        token_ctx = SequenceContext(
            input_ids=cast(torch.LongTensor, local_input_ids),
            inputs_embeds=None,
            cu_seq_lens_q=cast(torch.IntTensor, cu),
            cu_seq_lens_k=cast(torch.IntTensor, cu),
            max_length_q=max(compute_lengths),
            max_length_k=max(compute_lengths),
            num_padding=0,
            sequence_parallel_mesh=sp_mesh,
            device=input_ids.device,
            position_ids=cast(torch.LongTensor, local_positions),
            raw_input_ids=cast(torch.LongTensor, input_ids),
            shard_start=shard_start,
            shard_size=shard_size,
            valid_sequence_length=sum(compute_lengths),
        )
        return token_ctx, padded_lengths, compute_lengths

    @staticmethod
    def _restore_output_shard(
        hidden_states: Tensor,
        seq_ctx: SequenceContext,
        orig_lengths: list[int],
        padded_lengths: list[int],
    ) -> Tensor:
        sp_mesh = seq_ctx.sequence_parallel_mesh
        full_hidden = _gather_sequence(hidden_states, sp_mesh)
        cropped = _crop_documents(full_hidden, orig_lengths, padded_lengths)
        if sp_mesh is None or sp_mesh.size() == 1:
            return cropped
        if seq_ctx.input_ids is None:
            raise ValueError("SP output restoration requires the original local input shard")
        output_length = seq_ctx.input_ids.shape[1] * sp_mesh.size()
        if cropped.shape[1] < output_length:
            padding = cropped.new_zeros(cropped.shape[0], output_length - cropped.shape[1], cropped.shape[2])
            cropped = torch.cat((cropped, padding), dim=1)
        return _split_sequence(cropped, sp_mesh)

    def get_muon_metadata(self):
        """Model-owned physical projection layouts and scalar fallback
        options."""
        splits, gates, scalar = {}, {}, {}
        for name, param in self.named_parameters():
            if name.endswith((".post_norm", ".post_norm_weight", ".static_weight", ".static_logits")):
                scalar[param] = 0.0
            elif name.endswith(".attention.conv1d.weight") or name == "fusion_alpha":
                # Megatron conv is 3D, xTuner stores its equivalent 2D kernel.
                # Megatron fusion_alpha is 0D (weight decay enabled), xTuner is [1].
                scalar[param] = 0.1
        for module in self.modules():
            kind = type(module).__name__
            # FSDP/checkpoint wrappers expose the underlying modules during traversal.
            if "NCPQwen35V2MLA" in kind and hasattr(module, "q_up"):
                splits[module.q_up.weight] = (module.qk_head_dim + module.qk_pos_emb_head_dim,) * module.num_heads
                splits[module.kv_up.weight] = (module.qk_head_dim, module.v_head_dim) * module.num_heads
                splits[module.kv_down.weight] = (module.kv_lora_rank, module.qk_pos_emb_head_dim)
            elif "NCPQwen35V2GatedDeltaNet" in kind and hasattr(module, "in_proj"):
                rows = (module.head_k_dim,) * (2 * module.num_k_heads) + (module.head_v_dim,) * module.num_v_heads
                splits[module.in_proj.weight] = rows
                # [Q heads | K heads | V heads | z | beta | alpha].
                gates[module.in_proj.weight] = (sum(rows), module.in_proj.weight.shape[0])
            elif "FusedSwiGLUMLP" in kind and hasattr(module, "fc1"):
                width = module.fc1.weight.shape[0] // 2
                splits[module.fc1.weight] = (width, width)
            elif kind.endswith(("SelfMUDDRoute", "CrossMUDDRoute")):
                streams = getattr(module, "compute_streams", 4)
                splits[module.router_w2.weight] = (module.num_sources,) * streams
        return {
            "split_sizes": splits,
            "adamw_ranges": gates,
            "param_options": {param: {"algorithm": "adamw", "weight_decay": decay} for param, decay in scalar.items()},
        }

    def build_loss_ctx_batch(self, data_batch, sp_mesh=None):
        contexts = super().build_loss_ctx_batch(data_batch, sp_mesh)
        if self.config.concept_loss_cfg is not None:
            auxiliary = [self.config.concept_loss_cfg.build(data, sp_mesh) for data in data_batch]
            auxiliary = ConceptLMLossContext.build_batches(auxiliary)
            for context, aux_ctx in zip(contexts, auxiliary):
                context["concept"] = aux_ctx
        return contexts

    def forward(  # type: ignore[override]
        self,
        seq_ctx: SequenceContext,
        loss_ctx: dict[str, BaseLossContext | list[BaseLossContext]] | None = None,
        return_intermediates: bool = False,
    ) -> DenseModelOutputs | tuple[DenseModelOutputs, dict[str, torch.Tensor]]:
        config = self.config
        sp_mesh = seq_ctx.sequence_parallel_mesh
        orig_lengths = self._original_document_lengths(seq_ctx)
        token_ctx, padded_lengths, compute_lengths = self._token_context(seq_ctx, orig_lengths)
        input_ids = token_ctx.input_ids
        if input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = token_ctx.inputs_embeds
        assert hidden_states is not None
        self._mark_dynamic(token_ctx)
        chunk_size = config.concept_chunk_size

        encoder_hidden, encoder_raw = self.encoder(hidden_states, token_ctx)
        encoder_summaries = tuple(
            encoder_raw[index] for index in _block_end_indices(len(encoder_raw), config.cross_mudd_block_size)
        )
        full_encoder_sources = tuple(
            _gather_sequence(source, sp_mesh) for source in (hidden_states, *encoder_summaries)
        )
        full_encoder_hidden = _gather_sequence(encoder_hidden, sp_mesh)
        full_concept_hidden, concept_lengths = merge_chunks(
            full_encoder_hidden, compute_lengths, chunk_size, config.concept_chunk_merge_method
        )
        concept_hidden = _split_sequence(full_concept_hidden, sp_mesh)
        if sp_mesh is not None and sp_mesh.size() > 1:
            concept_ctx = _sp_context_from_lengths(concept_hidden, concept_lengths, sp_mesh)
        else:
            concept_ctx = _context_from_lengths(None, concept_hidden, concept_lengths, concept_hidden.device)
        encoder_cross_bank = tuple(
            _split_sequence(
                merge_chunks(source, compute_lengths, chunk_size, config.concept_chunk_merge_method)[0],
                sp_mesh,
            )
            for source in full_encoder_sources
        )
        predicted_hidden, predictor_raw = self.predictor(
            concept_hidden,
            concept_ctx,
            cross_banks=(encoder_cross_bank,),
        )
        prediction_logits = torch.stack([head(predicted_hidden) for head in self.prediction_heads], dim=2)
        predicted = torch.einsum(
            "bshn,hnd->bshd", prediction_logits.softmax(dim=-1), torch.stack(tuple(self.codebooks))
        ).flatten(-2)
        full_predicted = _gather_sequence(predicted, sp_mesh)
        expanded = _split_sequence(
            expand_concepts(
                full_predicted,
                compute_lengths,
                concept_lengths,
                chunk_size,
                config.concept_shift_feature,
            ),
            sp_mesh,
        )
        decoder_input = self.fusion_encoder_norm(encoder_hidden) + (
            self.fusion_alpha * self.fusion_repeated_norm(expanded)
        )
        predictor_summaries = tuple(
            predictor_raw[index] for index in _block_end_indices(len(predictor_raw), config.cross_mudd_block_size)
        )
        predictor_bank = tuple(
            _split_sequence(
                expand_concepts(
                    _gather_sequence(source, sp_mesh),
                    compute_lengths,
                    concept_lengths,
                    chunk_size,
                    config.concept_shift_feature,
                ),
                sp_mesh,
            )
            for source in (concept_hidden, *predictor_summaries)
        )
        decoder_hidden, _ = self.decoder(
            decoder_input,
            token_ctx,
            cross_banks=((hidden_states, *encoder_summaries), predictor_bank),
            aligned_predicted_concept=expanded,
        )
        hidden_states = self._restore_output_shard(decoder_hidden, seq_ctx, orig_lengths, padded_lengths)

        output: dict = {}
        intermediates = {
            "encoder_hidden": encoder_hidden,
            "concept_hidden": concept_hidden,
            "predicted": predicted,
            "decoder_input": decoder_input,
            "decoder_hidden": decoder_hidden,
        }
        if loss_ctx is None:
            _, (logits, _) = self.lm_head(hidden_states, None)
            output["logits"] = logits
        else:
            lm_loss, (logits, extra_info) = self.lm_head(hidden_states, loss_ctx["lm"])  # type: ignore[call-overload]
            if self.config.concept_loss_cfg is not None:
                if "concept" not in loss_ctx:
                    raise ValueError("Build ConceptLM loss contexts with model.build_loss_ctx_batch")
                concept_loss_ctx = cast(ConceptLMLossContext, loss_ctx["concept"])
                auxiliary, statistics = concept_loss_ctx(
                    full_concept_hidden if sp_mesh is not None and sp_mesh.size() > 1 else concept_hidden,
                    full_predicted if sp_mesh is not None and sp_mesh.size() > 1 else predicted,
                    self.codebooks,
                )
                lm_loss = lm_loss + auxiliary
                extra_info.update(statistics)
            output["loss"] = lm_loss
            output["logits"] = logits
            output["extra_info"] = extra_info
        if return_intermediates:
            return DenseModelOutputs(**output), intermediates
        return DenseModelOutputs(**output)

    # ----------------------------------------------------------- FSDP wiring
    def _stacks(self) -> list[nn.ModuleDict]:
        return [self.encoder.layers, self.predictor.layers, self.decoder.layers]

    def _init_device_mesh(self, fsdp_config: FSDPConfig):
        self.fsdp_config = fsdp_config
        world_size = dist.get_world_size()
        if self.fsdp_config.hsdp_sharding_size is None:
            model_mesh = init_device_mesh(
                DEVICE,
                (world_size, 1),
                mesh_dim_names=(f"{self.config.mesh_prefix}.fsdp", f"{self.config.mesh_prefix}.others"),
            )
            self.fsdp_mesh = model_mesh[f"{self.config.mesh_prefix}.fsdp"]
        else:
            model_mesh = init_device_mesh(
                DEVICE,
                (world_size // self.fsdp_config.hsdp_sharding_size, self.fsdp_config.hsdp_sharding_size),
                mesh_dim_names=(
                    f"{self.config.mesh_prefix}.hsdp_replicate",
                    f"{self.config.mesh_prefix}.hsdp_shard",
                ),
            )
            self.hsdp_mesh = model_mesh
            self.fsdp_mesh = self.hsdp_mesh[f"{self.config.mesh_prefix}.hsdp_shard"]
        self._world_mesh = model_mesh

    @override
    def fully_shard(self, fsdp_config: FSDPConfig) -> Self:
        self.fsdp_config = fsdp_config
        self._init_device_mesh(fsdp_config)

        if self.config.float8_cfg is not None:
            from xtuner.v1.float8.float8_handler import Float8Handler

            Float8Handler.pad_for_fsdp(
                self,
                cast(DeviceMesh, self.fsdp_mesh),
                callback_after_pad=self._init_load_spec,
            )  # type: ignore[arg-type]

        checkpoint_preserve_rng_state = fsdp_config.checkpoint_preserve_rng_state
        if not checkpoint_preserve_rng_state and self.config.attention.dropout > 0.0:
            checkpoint_preserve_rng_state = True
            log_rank0.warning("When using dropout, checkpoint_preserve_rng_state is set to True to avoid issues.")

        assert self.fsdp_mesh is not None
        if self.fsdp_config.requires_grad:
            for module in self.modules():
                for p_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        setattr(module, p_name, nn.Parameter(param.to(dtype=torch.float32)))
        else:
            for param in self.parameters():
                param.requires_grad = False

        mp_policy = MixedPrecisionPolicy(param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype)
        lm_head_mp_policy = (
            MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
            if fsdp_config.fp32_lm_head
            else mp_policy
        )
        offload_policy = CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None
        shard_mesh = self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh

        # Flatten the three stacks so the recompute budget is spread over all layers.
        stack_layers = [(stack, key, layer) for stack in self._stacks() for key, layer in stack.items()]
        num_recompute_layers = int(len(stack_layers) * self.fsdp_config.recompute_ratio)
        generator = torch.Generator()
        generator.manual_seed(0)
        shuffled = torch.randperm(len(stack_layers), generator=generator)
        for position, flat_idx in enumerate(tqdm(shuffled, desc="[FSDP Sharding]")):
            stack, key, layer = stack_layers[int(flat_idx)]
            if position < num_recompute_layers:
                layer = apply_activation_checkpointing(layer, preserve_rng_state=checkpoint_preserve_rng_state)
                if self.compile_cfg:
                    compiled_forward = torch.compile(type(layer).forward, fullgraph=False)
                    layer.forward = compiled_forward.__get__(layer, type(layer))  # type: ignore[method-assign]
            stack[key] = layer
            self._fully_shard(
                mesh=shard_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=self.fsdp_config.reshard_after_forward,
                offload_policy=offload_policy,
                module=layer,
            )

        # NOTE: ``prediction_heads`` and ``MUDDRoutes`` have no ``forward`` of their own,
        # so their parameters are sharded together with the root module below.
        for module, policy in (
            (self.embed_tokens, mp_policy),
            (self.lm_head, lm_head_mp_policy),
            (self.fusion_encoder_norm, mp_policy),
            (self.fusion_repeated_norm, mp_policy),
            (self.encoder.final_norm, mp_policy),
            (self.predictor.final_norm, mp_policy),
            (self.decoder.final_norm, mp_policy),
        ):
            self._fully_shard(
                mesh=shard_mesh,
                mp_policy=policy,
                reshard_after_forward=self.fsdp_config.reshard_after_forward,
                offload_policy=offload_policy,
                module=module,
            )

        self._fully_shard(
            mesh=shard_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.fsdp_config.reshard_after_forward,
            offload_policy=offload_policy,
        )
        self._init_load_spec()
        self._to_empty_meta()
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        return self
