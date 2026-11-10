import os
from functools import partial
from typing import Callable, Literal, Protocol, TypeAlias, TypedDict, cast

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch.autograd.function import Function
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from xtuner.v1.config.generate import GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8 import Float8Config
from xtuner.v1.module import (
    AttnOutputs,
    GatedDeltaNet,
    GatedDeltaNetConfig,
    GreedyRouterConfig,
    KDAConfig,
    KimiDeltaAttention,
    MHAConfig,
    MLAConfig,
    MultiHeadAttention,
    MultiLatentAttention,
    NoAuxRouterConfig,
    RMSNorm,
    RouterResults,
)
from xtuner.v1.module.dispatcher import (
    CombineResult,
    DispatchResult,
    EPExecutionRuntime,
    ExpertWeightLayout,
    PostDispatchResult,
    PreCombineResult,
    PreDispatchResult,
    build_dispatcher,
)
from xtuner.v1.module.grouped_linear.moe_group_linear import build_grouped_linear
from xtuner.v1.module.moe_backend import SonicMoEBackendConfig
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.ops.act_fn import get_act_fn, get_mlp_act_fn
from xtuner.v1.utils import ForwardState, get_device

from ..linear import build_linear


RouterLogits: TypeAlias = torch.Tensor
RouterWeights: TypeAlias = torch.Tensor
RouterTopKIds: TypeAlias = torch.Tensor
HiddenStates: TypeAlias = torch.Tensor


def _prepare_rollout_routed_experts_for_router(
    rollout_routed_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    *,
    offload_rollout_routed_experts: bool,
) -> torch.Tensor:
    """Move one layer's routed-expert IDs to the router index dtype."""
    if offload_rollout_routed_experts and rollout_routed_experts.device != hidden_states.device:
        rollout_routed_experts = rollout_routed_experts.contiguous()
    return rollout_routed_experts.to(device=hidden_states.device, dtype=torch.long)


def _intra_layer_interleave_enabled() -> bool:
    """Whether in-layer micro-batches pipeline dispatch against experts GEMM.

    ``MOONEP_DISPATCH_OVERLAP=1`` submits micro-batch ``i``'s routed dispatch and
    then immediately issues micro-batch ``i-1``'s postprocess + experts GEMM, so
    the AIV dispatch runs while the Cube GEMM of the previous micro-batch runs
    on the compute stream.

    Issuing the compute only after *every* dispatch has been submitted is not
    enough on Ascend: the plan ends in a small D2H that blocks the host until
    the preceding dispatch kernel has retired, so the GEMMs would all be
    submitted after the dispatches already finished on device and nothing would
    overlap. Handing the previous micro-batch's compute to the host during that
    wait costs it almost nothing (no D2H there) and is what makes the overlap
    real.

    Only the *dispatch* is reordered. The routed weight prefetch is the one part
    of ``dispatch`` with a cross-micro-batch hazard — the expert home section
    holds a single set of B borrow slots per generation, so micro-batch i+1's
    prefetch would overwrite the experts micro-batch i's GEMM still reads — and
    it is deferred to ``prepare_experts`` for that call instead of running
    inside ``dispatch``. See ``_async_parallel_enabled`` in the MoonEP
    dispatcher. Default on, and Ascend-only: the CUDA MoonEP Buffer has no
    prefetch hook or micro-batch slot index.
    """
    return get_device() == "npu" and os.environ.get("MOONEP_DISPATCH_OVERLAP", "1") == "1"


def _combine_overlap_enabled() -> bool:
    """Whether each in-layer micro-batch's combine is issued right after its GEMM.

    ``MOONEP_COMBINE_OVERLAP=1`` submits micro-batch ``i``'s combine as soon as
    its experts GEMM is queued, instead of collecting every micro-batch's
    combine at the layer tail. The combine is launched on MoonEP's own stream
    and returns to the host immediately, so micro-batch ``i+1``'s GEMM is
    submitted beside it rather than after it. Output order is unchanged.
    Default on, and Ascend-only for the same reason as the dispatch overlap.
    """
    return get_device() == "npu" and os.environ.get("MOONEP_COMBINE_OVERLAP", "1") == "1"


class MoEDecoderLayerOutput(TypedDict):
    """Per-micro-batch outputs of one :class:`MoEDecoderLayer` forward."""

    hidden_states: HiddenStates
    router_logits: RouterLogits
    router_weights: RouterWeights
    router_topk_ids: RouterTopKIds


class MoEDecoderLayerMicroBatchOutput(TypedDict):
    """Outputs of one :class:`MoEDecoderLayer` forward over several micro-
    batches (domino EP).

    Each field holds one entry per micro-batch, in input order.
    """

    hidden_states: list[HiddenStates]
    router_logits: list[RouterLogits]
    router_weights: list[RouterWeights]
    router_topk_ids: list[RouterTopKIds]


class MoEActFnProtocol(Protocol):
    def __call__(self, fused_x: torch.Tensor, split_dim: int = -1) -> torch.Tensor: ...


class MoEActFnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    act_type: Literal["clipped_swiglu", "swiglu", "situ"] = "swiglu"

    clip_alpha: float | None = None
    clip_limit: float | None = None
    situ_beta: float = 1.0
    situ_linear_beta: float | None = None

    def build(self) -> MoEActFnProtocol:
        act_fn = get_act_fn(self.act_type)

        if self.act_type == "clipped_swiglu":
            act_fn = partial(act_fn, alpha=self.clip_alpha, limit=self.clip_limit)
        elif self.act_type == "situ":
            act_fn = partial(act_fn, beta=self.situ_beta, linear_beta=self.situ_linear_beta)
        return act_fn


class MoEMLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        n_shared_experts: int,
        moe_intermediate_size: int,
        hidden_act: str,
        mlp_bias: bool = False,
        float8_cfg: Float8Config | None = None,
        situ_beta: float = 1.0,
        situ_linear_beta: float | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = moe_intermediate_size * n_shared_experts
        self.gate_proj = build_linear(self.hidden_size, self.intermediate_size, bias=mlp_bias, float8_cfg=float8_cfg)
        self.up_proj = build_linear(self.hidden_size, self.intermediate_size, bias=mlp_bias, float8_cfg=float8_cfg)
        self.down_proj = build_linear(self.intermediate_size, self.hidden_size, bias=mlp_bias, float8_cfg=float8_cfg)
        self.act_fn = get_mlp_act_fn(hidden_act, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x), self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        router_config: GreedyRouterConfig | NoAuxRouterConfig,
        gate_bias: bool = False,
        router_compute_dtype: Literal["float32", "native"] = "float32",
    ):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.router_compute_dtype = router_compute_dtype

        self.gating_dim = hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))

        self.router = router_config.build(
            n_routed_experts=self.n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

        self.gate_bias = gate_bias
        if self.gate_bias:
            self.bias = nn.Parameter(torch.zeros(self.n_routed_experts))

    def forward(
        self, hidden_states: torch.Tensor, rollout_routed_experts: torch.Tensor | None = None
    ) -> RouterResults:
        _, _, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)

        if isinstance(self.weight, DTensor):
            weight = self.weight.to_local()
        else:
            weight = self.weight

        bias = None
        if self.gate_bias:
            bias = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias

        if self.router_compute_dtype == "native":
            logits = F.linear(hidden_states, weight, bias)
        else:
            bias = bias.float() if bias is not None else None
            logits = F.linear(hidden_states.float(), weight.float(), bias)
        return self.router(logits, rollout_routed_experts)

        # Debug for aligning with hf implementation.
        # logits = F.linear(hidden_states, weight, bias)
        # gate = self.router(logits, rollout_routed_experts)
        # gate['topk_weights'] = gate['topk_weights'].float()
        # return gate


class MoEBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int,
        moe_bias: bool = False,
        ep_mesh: DeviceMesh | None = None,
        expert_tp_mesh: DeviceMesh | None = None,
        float8_cfg: Float8Config | None = None,
        moe_act_fn_cfg: MoEActFnConfig,
        ep_tp_mesh: DeviceMesh | None = None,
        expert_backend: Literal["grouped_gemm", "sonicmoe"] = "grouped_gemm",
        sonicmoe_cfg: SonicMoEBackendConfig | None = None,
    ):
        super().__init__()
        if expert_backend == "sonicmoe":
            if ep_mesh is not None and ep_mesh.size() > 1:
                raise ValueError("The initial SonicMoE integration does not support EP.")
            if expert_tp_mesh is not None and expert_tp_mesh.size() > 1:
                raise ValueError("The initial SonicMoE integration does not support ExpertTP.")
            if float8_cfg is not None:
                raise ValueError("The initial SonicMoE integration does not support FP8.")
            if moe_act_fn_cfg.act_type != "swiglu":
                raise ValueError("The initial SonicMoE integration supports SwiGLU only.")
        self.hidden_size = hidden_size
        self.intermediate_size = moe_intermediate_size
        self.num_routed_experts = n_routed_experts
        self.ep_mesh = ep_mesh
        # self.fused_w1 = GroupedLinear(self.hidden_size, self.intermediate_size, self.num_routed_experts, ep_mesh)
        # self.fused_w3 = GroupedLinear(self.hidden_size, self.intermediate_size, self.num_routed_experts, ep_mesh)
        self.fused_w1w3 = build_grouped_linear(
            self.hidden_size,
            2 * self.intermediate_size,
            self.num_routed_experts,
            moe_bias=moe_bias,
            ep_mesh=self.ep_mesh,
            expert_tp_mesh=expert_tp_mesh,
            parallel_style="column",
            float8_cfg=float8_cfg,
            ep_tp_mesh=ep_tp_mesh,
            num_fused_projections=2,
        )
        self.fused_w2 = build_grouped_linear(
            self.intermediate_size,
            self.hidden_size,
            self.num_routed_experts,
            moe_bias=moe_bias,
            ep_mesh=self.ep_mesh,
            expert_tp_mesh=expert_tp_mesh,
            parallel_style="row",
            float8_cfg=float8_cfg,
            ep_tp_mesh=ep_tp_mesh,
        )
        self.moe_act = moe_act_fn_cfg.build()
        self.sonicmoe = (sonicmoe_cfg or SonicMoEBackendConfig()).build() if expert_backend == "sonicmoe" else None

    @property
    def uses_sonicmoe(self) -> bool:
        """Return whether this expert block uses the SonicMoE backend.

        Returns:
            bool: Whether SonicMoE is enabled.
        """
        return self.sonicmoe is not None

    @staticmethod
    def _local_parameter(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    def forward_routed(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        router_weights: torch.Tensor,
        *,
        decoding: bool = False,
    ) -> torch.Tensor:
        """Run SonicMoE directly from XTuner's unpermuted router output.

        Args:
            hidden_states (torch.Tensor): Unpermuted expert input with shape ``[T, H]``.
            topk_ids (torch.Tensor): Original top-k expert ids with shape ``[T, K]``.
            topk_weights (torch.Tensor): Original top-k router weights with shape ``[T, K]``.
            router_weights (torch.Tensor): Full router weights with shape ``[T, E]``.
            decoding (bool): Whether the call is an inference decode step.

        Returns:
            torch.Tensor: Combined routed-expert output with shape ``[T, H]``.
        """
        if self.sonicmoe is None:
            raise RuntimeError("forward_routed is only available when expert_backend='sonicmoe'.")

        w1w3 = self._local_parameter(self.fused_w1w3.weight).view(
            self.num_routed_experts, 2 * self.intermediate_size, self.hidden_size
        )
        w2 = self._local_parameter(self.fused_w2.weight).view(
            self.num_routed_experts, self.hidden_size, self.intermediate_size
        )
        w1w3_bias = None
        w2_bias = None
        if self.fused_w1w3.moe_bias:
            w1w3_bias = self._local_parameter(self.fused_w1w3.bias)
            w2_bias = self._local_parameter(self.fused_w2.bias)

        output, _ = self.sonicmoe(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            router_weights=router_weights,
            fused_w1w3=w1w3,
            fused_w2=w2,
            fused_w1w3_bias=w1w3_bias,
            fused_w2_bias=w2_bias,
            training=self.training and not decoding,
        )
        return output

    def forward(
        self,
        x: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        *,
        weight_layout: ExpertWeightLayout,
    ) -> torch.Tensor:
        trainable = weight_layout.trainable_weights or (None, None)
        # X8 (grad_weight_out): dispatcher-supplied heap gradient slot views.
        # When present, GroupedLinear → _GMMWithGradWeightOut writes dW directly
        # into the heap (home add_ + dup copy_), bypassing the autograd dW path;
        # the bridge still fires via dx to trigger reduce_grad_bf16 + FSDP writeback.
        gwo = weight_layout.grad_weight_out or (None, None)
        # grad_weight_first = first producer of this FSDP call (the home has
        # already been zeroed by the dispatcher) -> the GEMM dw writes the
        # whole heap block directly, with zero materialization.
        gwo_first = getattr(weight_layout, "grad_weight_first", False)
        gate_up_out = self.fused_w1w3(
            x, tokens_per_expert, trainable_weight=trainable[0],
            grad_weight_out=gwo[0], grad_weight_first=gwo_first,
        )
        out = self.moe_act(gate_up_out, split_dim=-1)
        res = self.fused_w2(
            out, tokens_per_expert, trainable_weight=trainable[1],
            grad_weight_out=gwo[1], grad_weight_first=gwo_first,
        )
        return res


class MoEDecoderLayer(nn.Module):
    """MoE decoder layer."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        mlp_bias: bool = False,
        gate_bias: bool = False,
        moe_bias: bool = False,
        hidden_act: str,
        rms_norm_eps: float = 1e-6,
        rms_norm_type: Literal["default", "zero_centered"] = "default",
        num_experts_per_tok: int,
        n_routed_experts: int,
        n_shared_experts: int,
        with_shared_expert_gate: bool = False,
        hidden_factor: float = 1.0,
        attention_config: MHAConfig | MLAConfig | GatedDeltaNetConfig | KDAConfig,
        rope_scaling_cfg: RopeScalingConfig | None = None,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        generate_config: GenerateConfig | None = None,
        router_config: GreedyRouterConfig | NoAuxRouterConfig,
        router_compute_dtype: Literal["float32", "native"] = "float32",
        moe_act_fn_cfg: MoEActFnConfig,
        expert_backend: Literal["grouped_gemm", "sonicmoe"] = "grouped_gemm",
        sonicmoe_cfg: SonicMoEBackendConfig | None = None,
        float8_cfg: Float8Config | None = None,
        layer_idx: int = 0,
        dispatcher: Literal["deepep", "all2all", "agrs", "moonep"] | None,
        ep_mesh: DeviceMesh | None = None,
        expert_tp_mesh: DeviceMesh | None = None,
        ep_tp_mesh: DeviceMesh | None = None,
        situ_beta: float = 1.0,
        situ_linear_beta: float | None = None,
        routed_expert_hidden_size: int | None = None,
        latent_moe_use_norm: bool = False,
        ep_runtime: EPExecutionRuntime | None = None,
        layer_fqn: str | None = None,
    ):
        super().__init__()
        if expert_backend == "sonicmoe" and dispatcher is not None:
            raise ValueError("SonicMoE owns local token dispatch, so dispatcher must be None in the initial version.")
        if routed_expert_hidden_size is not None and expert_backend == "sonicmoe":
            raise NotImplementedError("LatentMoE does not support sonicmoe yet")
        self.ep_mesh = ep_mesh
        self.ep_tp_mesh = ep_tp_mesh
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.hidden_factor = hidden_factor
        self.routed_expert_hidden_size = routed_expert_hidden_size
        self.latent_moe_use_norm = latent_moe_use_norm

        # Overlap the shared-experts GEMM (compute stream) with the dispatch
        # all-to-all (comm stream). MoonEP's in-layer interleave refactor reads
        # this attribute; the upstream XTUNER_MOE_SHARED_OVERLAP feature itself
        # is not part of this branch, so it defaults to off here.
        self.shared_overlap = os.environ.get("XTUNER_MOE_SHARED_OVERLAP", "0") == "1"

        self.self_attn: MultiHeadAttention | MultiLatentAttention | GatedDeltaNet | KimiDeltaAttention = (
            attention_config.build(
                hidden_size=hidden_size,
                layer_idx=layer_idx,
                generate_config=generate_config,
                rope_scaling_cfg=rope_scaling_cfg,
                layer_type=layer_type,
                float8_cfg=float8_cfg,
            )
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.layer_idx = layer_idx

        self.with_shared_expert_gate = with_shared_expert_gate
        self.shared_expert_gate: nn.Module | None
        self.shared_experts: MoEMLP | None

        if n_shared_experts > 0:
            self.shared_experts = self.build_shared_experts(
                hidden_size=hidden_size,
                n_shared_experts=n_shared_experts,
                moe_intermediate_size=moe_intermediate_size,
                hidden_act=hidden_act,
                mlp_bias=mlp_bias,
                float8_cfg=float8_cfg,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            )
            if with_shared_expert_gate:
                self.shared_expert_gate = build_linear(hidden_size, 1, bias=False)
            else:
                self.shared_expert_gate = None
        else:
            self.shared_experts = None
            self.shared_expert_gate = None

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)

        self.gate = MoEGate(
            hidden_size=hidden_size,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            router_config=router_config,
            gate_bias=gate_bias,
            router_compute_dtype=router_compute_dtype,
        )

        expert_hidden_size = routed_expert_hidden_size if routed_expert_hidden_size is not None else hidden_size
        if routed_expert_hidden_size is not None:
            self.routed_expert_down_proj = build_linear(
                hidden_size, routed_expert_hidden_size, bias=False, float8_cfg=float8_cfg
            )
            self.routed_expert_up_proj = build_linear(
                routed_expert_hidden_size, hidden_size, bias=False, float8_cfg=float8_cfg
            )
            if latent_moe_use_norm:
                self.routed_expert_norm = RMSNorm(routed_expert_hidden_size, eps=rms_norm_eps)

        self.experts = self.build_experts(
            hidden_size=expert_hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            n_routed_experts=n_routed_experts,
            moe_bias=moe_bias,
            ep_mesh=ep_mesh,
            expert_tp_mesh=expert_tp_mesh,
            float8_cfg=float8_cfg,
            moe_act_fn_cfg=moe_act_fn_cfg,
            ep_tp_mesh=ep_tp_mesh,
            expert_backend=expert_backend,
            sonicmoe_cfg=sonicmoe_cfg,
        )
        # TODO: (yehaochen) Maybe should be replaced by build_dispatcher
        process_group = ep_mesh.get_group() if ep_mesh is not None else None
        tp_group = expert_tp_mesh.get_group() if expert_tp_mesh is not None else None
        ep_tp_group = ep_tp_mesh._flatten().get_group() if ep_tp_mesh is not None else None
        # EP membership is static for the layer. Keep the decision outside the
        # compiled forward; Naive EP=1 execution retains its synchronous API.
        self._async_combine = process_group is not None and process_group.size() > 1
        self.dispatcher = None
        if not self.experts.uses_sonicmoe:
            self.dispatcher = build_dispatcher(
                dispatcher=dispatcher,
                n_routed_experts=n_routed_experts,
                ep_group=process_group,
                tp_group=tp_group,
                ep_tp_group=ep_tp_group,
                training_dtype="fp8" if float8_cfg is not None else "bf16",
                generate_dtype=generate_config.dtype if generate_config is not None else "bf16",
                ep_runtime=ep_runtime,
                layer_fqn=layer_fqn,
                projections=(self.experts.fused_w1w3, self.experts.fused_w2),
            )

    def build_shared_experts(
        self,
        *,
        hidden_size: int,
        n_shared_experts: int,
        moe_intermediate_size: int,
        hidden_act: str,
        mlp_bias: bool,
        float8_cfg: Float8Config | None,
        situ_beta: float = 1.0,
        situ_linear_beta: float | None = None,
    ) -> MoEMLP:
        """Construct shared-expert MLP; subclasses may override."""
        return MoEMLP(
            hidden_size=hidden_size,
            n_shared_experts=n_shared_experts,
            moe_intermediate_size=moe_intermediate_size,
            hidden_act=hidden_act,
            mlp_bias=mlp_bias,
            float8_cfg=float8_cfg,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )

    def build_experts(
        self,
        *,
        hidden_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int,
        moe_bias: bool,
        ep_mesh: DeviceMesh | None,
        expert_tp_mesh: DeviceMesh | None,
        float8_cfg: Float8Config | None,
        moe_act_fn_cfg: MoEActFnConfig,
        ep_tp_mesh: DeviceMesh | None,
        expert_backend: Literal["grouped_gemm", "sonicmoe"],
        sonicmoe_cfg: SonicMoEBackendConfig | None,
    ) -> MoEBlock:
        """Construct routed ``MoEBlock``; subclasses may override."""
        return MoEBlock(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            n_routed_experts=n_routed_experts,
            moe_bias=moe_bias,
            ep_mesh=ep_mesh,
            expert_tp_mesh=expert_tp_mesh,
            float8_cfg=float8_cfg,
            moe_act_fn_cfg=moe_act_fn_cfg,
            ep_tp_mesh=ep_tp_mesh,
            expert_backend=expert_backend,
            sonicmoe_cfg=sonicmoe_cfg,
        )

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        *,
        seq_ctx: SequenceContext | list[SequenceContext],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
    ) -> MoEDecoderLayerOutput | MoEDecoderLayerMicroBatchOutput:
        """Forward pass of the MoE decoder layer.

        Passing lists runs several equal-shaped micro-batches in one layer invocation (domino EP).
        ExpertTP overlaps dispatch/combine communication with expert compute; the local BF16 NaiveDispatcher
        instead concatenates tokens and invokes routed experts once.

        Args:
            hidden_states (torch.Tensor | list[torch.Tensor]): Input hidden states, one tensor per
                micro-batch.
            seq_ctx (SequenceContext | list[SequenceContext]): Sequence context, aligned with
                ``hidden_states``.
            position_embeddings (tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]]):
                Rotary position embeddings ``(cos, sin)``, aligned with ``hidden_states``.

        Returns:
            MoEDecoderLayerOutput | MoEDecoderLayerMicroBatchOutput: Hidden states and router
            results. Scalar fields for a single ``hidden_states`` tensor, per-micro-batch lists for
            a list of them.
        """
        if not isinstance(hidden_states, list):
            assert isinstance(seq_ctx, SequenceContext), (
                f"seq_ctx should be a SequenceContext instance but got {seq_ctx}"
            )
            assert isinstance(position_embeddings, tuple) and len(position_embeddings) == 2, (
                "position_embeddings should be a tuple of two tensors (position_ids, position_embeds)"
            )
            return self._forward(
                hidden_states=hidden_states,
                seq_ctx=seq_ctx,
                position_embeddings=position_embeddings,
            )

        assert isinstance(seq_ctx, list) and len(seq_ctx) == len(hidden_states), (
            "seq_ctx should be a list of SequenceContext instances with the same length as hidden_states"
        )
        assert isinstance(position_embeddings, list) and len(position_embeddings) == len(hidden_states), (
            "position_embeddings should be a list of tuples with the same length as hidden_states"
        )

        return self._micro_batch_forward(
            hidden_states_list=hidden_states,
            seq_ctx_list=seq_ctx,
            position_embeddings_list=position_embeddings,
        )

    def _hf_expert_forward_for_debug(self, hidden_states: torch.Tensor, router_results: RouterResults, origin_shape):
        # xtuner: num_experts * 2 * expert_dim, hidden_size
        # hf: num_experts, 2 * expert_dim, hidden_size
        origin_gate_up_proj = self.experts.fused_w1w3.weight
        gate_up_proj = origin_gate_up_proj.view(
            self.n_routed_experts, 2 * self.experts.intermediate_size, self.hidden_size
        )

        # xtuner: num_experts * hidden_size, expert_dim
        # hf: num_experts, hidden_size, expert_dim
        origin_down_proj = self.experts.fused_w2.weight
        down_proj = origin_down_proj.view(self.n_routed_experts, self.hidden_size, self.experts.intermediate_size)

        from transformers.activations import ACT2FN

        act_fn = ACT2FN["silu"]

        hidden_states_reshaped = hidden_states.view(-1, hidden_states.size(-1))
        combined_hidden_states = torch.zeros_like(hidden_states_reshaped)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(router_results["topk_ids"], num_classes=self.n_routed_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.n_routed_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_reshaped[token_idx]
            gate, up = nn.functional.linear(current_state, gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, down_proj[expert_idx])
            current_hidden_states = current_hidden_states * router_results["topk_weights"][token_idx, top_k_pos, None]
            combined_hidden_states.index_add_(0, token_idx, current_hidden_states.to(combined_hidden_states.dtype))

        combined_hidden_states = combined_hidden_states.view(*origin_shape)
        return combined_hidden_states

    def _project_to_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Optional LatentMoE down-proj before routed-expert dispatch."""
        if self.routed_expert_hidden_size is None:
            return hidden_states
        return self.routed_expert_down_proj(hidden_states)

    def _project_from_experts(self, combined_hidden_states: torch.Tensor) -> torch.Tensor:
        """Optional LatentMoE norm + up-proj after routed-expert combine."""
        if self.routed_expert_hidden_size is None:
            return combined_hidden_states
        if self.latent_moe_use_norm:
            combined_hidden_states = self.routed_expert_norm(combined_hidden_states)
        return self.routed_expert_up_proj(combined_hidden_states)

    def _dispatch_experts_combine(
        self,
        *,
        hidden_states: torch.Tensor,
        router_results: RouterResults,
        layer_state: object | None = None,
    ) -> torch.Tensor:
        """Routed path: project → dispatch → experts → combine → project back.

        Preserves ``(batch, seq, *)`` layout; last dim may change across the
        LatentMoE projections.
        """
        assert self.dispatcher is not None
        origin_b, origin_s = hidden_states.shape[:2]
        expert_in = self._project_to_experts(hidden_states)
        pre_dispatched = self.dispatcher.dispatch_preprocess(
            hidden_states=expert_in.view(-1, expert_in.shape[-1]),
            topk_ids=router_results["topk_ids"],
            topk_weights=router_results["topk_weights"],
            tokens_per_expert=router_results["tokens_per_expert"],
            layer_state=layer_state,
        )
        dispatched = self.dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            topk_weights=router_results["topk_weights"],
            decoding=False,
        )  # type: ignore[call-overload]
        post_dispatched = self.dispatcher.dispatch_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
        )
        if self.ep_mesh is not None:
            # MoEBlock is fullgraph-compiled and shared by all decoder layers. Only the routed-token
            # dimension varies, so make it dynamic before entering the compile boundary to keep one
            # AOT Autograd save plan across the original forward and checkpoint replay.
            torch._dynamo.mark_dynamic(post_dispatched["hidden_states"], 0)
        experts_out = self.experts(
            post_dispatched["hidden_states"],
            post_dispatched["tokens_per_expert"],
            weight_layout=post_dispatched["expert_weight_layout"],
        )
        pre_combined = self.dispatcher.combine_preprocess(
            hidden_states=experts_out,
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            decoding=False,
        )
        combined = self.dispatcher.combine(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            decoding=False,
        )
        post_combined = self.dispatcher.combine_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            combined=combined,
        )
        combined_hidden_states = post_combined["hidden_states"].view(origin_b, origin_s, -1)
        return self._project_from_experts(combined_hidden_states)

    def _forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_kwargs: dict[str, object] | None = None,
    ) -> MoEDecoderLayerOutput:
        # MoonEP uses this identity seam to place Join before attention and
        # carries its opaque invocation token into dispatch phase 1. The SonicMoE
        # backend keeps no dispatcher at all, so it skips the seam.
        if self.experts.uses_sonicmoe:
            layer_inputs, layer_states = [hidden_states], [None]
        else:
            assert self.dispatcher is not None
            layer_inputs, layer_states = self.dispatcher.prepare_layer_inputs([hidden_states])
        hidden_states, layer_state = layer_inputs[0], layer_states[0]
        residual, hidden_states, router_results, attn_outputs = self._pre_moe_forward(
            hidden_states=hidden_states,
            seq_ctx=seq_ctx,
            position_embeddings=position_embeddings,
            state=ForwardState.TRAINING,
            attention_kwargs=attention_kwargs,
        )

        if self.experts.uses_sonicmoe:
            hidden_states = self._sonicmoe_forward(
                hidden_states=hidden_states,
                residual=residual,
                router_results=router_results,
            )
            return self._build_output(
                hidden_states=hidden_states,
                router_results=router_results,
                attn_outputs=attn_outputs,
            )

        combined_hidden_states = self._dispatch_experts_combine(
            hidden_states=hidden_states,
            router_results=router_results,
            layer_state=layer_state,
        )

        # debug for aligning with hf implementation.
        # combined_hidden_states = self._hf_expert_forward_for_debug(hidden_states, router_results, origin_shape)

        if self.n_shared_experts > 0:
            shared_experts_out = self._shared_experts_forward(hidden_states=hidden_states)
        else:
            shared_experts_out = None

        hidden_states = self._post_moe_forward(
            combined_hidden_states=combined_hidden_states,
            residual=residual,
            shared_experts_out=shared_experts_out,
        )
        return self._build_output(
            hidden_states=hidden_states,
            router_results=router_results,
            attn_outputs=attn_outputs,
        )

    def _build_output(
        self,
        *,
        hidden_states: torch.Tensor,
        router_results: RouterResults,
        attn_outputs: AttnOutputs,
    ) -> MoEDecoderLayerOutput:
        """Build the public output; model-specific decoders may extend it."""
        return {
            "hidden_states": hidden_states,
            "router_logits": router_results["logits"],
            "router_weights": router_results["router_weights"],
            "router_topk_ids": router_results["topk_ids"],
        }

    def _micro_batch_forward(
        self,
        hidden_states_list: list[torch.Tensor],
        seq_ctx_list: list[SequenceContext],
        position_embeddings_list: list[tuple[torch.Tensor, torch.Tensor]],
        attention_kwargs_list: list[dict[str, object]] | None = None,
    ) -> MoEDecoderLayerMicroBatchOutput:
        if self.experts.uses_sonicmoe:
            return self._sonicmoe_micro_batch_forward(
                hidden_states_list=hidden_states_list,
                seq_ctx_list=seq_ctx_list,
                position_embeddings_list=position_embeddings_list,
                attention_kwargs_list=attention_kwargs_list,
            )

        assert self.dispatcher is not None
        if self.dispatcher.supports_micro_batch_pack:
            return self._local_packed_micro_batch_forward(
                hidden_states_list=hidden_states_list,
                seq_ctx_list=seq_ctx_list,
                position_embeddings_list=position_embeddings_list,
                attention_kwargs_list=attention_kwargs_list,
            )

        origin_shape = hidden_states_list[0].shape
        assert all(hidden_states.shape == origin_shape for hidden_states in hidden_states_list), (
            "All hidden states should have the same shape"
        )
        intra_layer_micro_batch = len(hidden_states_list)
        residual_list: list[torch.Tensor] = []
        router_results_list: list[RouterResults] = []
        attn_outputs_list: list[AttnOutputs] = []
        if attention_kwargs_list is None:
            attention_kwargs_list = [{} for _ in hidden_states_list]
        assert len(attention_kwargs_list) == intra_layer_micro_batch

        pre_dispatched_list: list[PreDispatchResult] = []
        dispatched_list: list[DispatchResult] = []
        pre_moe_forward_out_list: list[torch.Tensor] = []

        # A single multi-output Join must precede every branch (including
        # residuals), so shared expert gradients reach FSDP exactly once.
        hidden_states_list, layer_states = self.dispatcher.prepare_layer_inputs(hidden_states_list)

        # Attention + gate + pre-dispatch
        for (
            hidden_states,
            layer_state,
            attention_kwargs,
            seq_ctx,
            position_embeddings,
        ) in zip(
            hidden_states_list,
            layer_states,
            attention_kwargs_list,
            seq_ctx_list,
            position_embeddings_list,
        ):
            residual, hidden_states, router_results, attn_outputs = self._pre_moe_forward(
                hidden_states=hidden_states,
                seq_ctx=seq_ctx,
                position_embeddings=position_embeddings,
                state=ForwardState.TRAINING,
                attention_kwargs=attention_kwargs,
            )
            pre_moe_forward_out_list.append(hidden_states)
            expert_in = self._project_to_experts(hidden_states)
            expert_in = expert_in.view(-1, expert_in.shape[-1])
            pre_dispatched = self.dispatcher.dispatch_preprocess(
                hidden_states=expert_in,
                topk_ids=router_results["topk_ids"],
                topk_weights=router_results["topk_weights"],
                tokens_per_expert=router_results["tokens_per_expert"],
                layer_state=layer_state,
                async_op=True,
            )
            pre_dispatched_list.append(pre_dispatched)
            residual_list.append(residual)
            router_results_list.append(router_results)
            attn_outputs_list.append(attn_outputs)

        post_dispatched_list: list[PostDispatchResult] = []
        experts_out_list: list[torch.Tensor] = []
        pre_combined_list: list[PreCombineResult] = []
        combined_list: list[CombineResult] = []
        # Pre-init so the overlap path can append inside the dispatch loop; the
        # serial path (shared_overlap off) and the n_shared_experts == 0 case
        # are both reassigned in the later shared-experts loop.
        shared_experts_out_list: list[torch.Tensor | None] = []

        # dispatch + experts + pre-combine
        #
        # ``MOONEP_DISPATCH_OVERLAP=1`` pipelines micro-batch i's routed dispatch
        # against micro-batch i-1's experts GEMM, so the AIV dispatch runs on
        # the comm stream while the Cube GEMM runs on the compute stream. The
        # dispatch contract is already "launch now, join in
        # dispatch_postprocess", so the reordering adds no waiting. Default off
        # — see ``_intra_layer_interleave_enabled``.
        interleave = _intra_layer_interleave_enabled()

        def _issue_routed_compute(i: int) -> None:
            """Issue micro-batch ``i``'s postprocess + experts GEMM + pre-combine.

            Must be called in increasing ``i``: the three output lists are
            appended in call order and consumed positionally downstream.
            """
            pre_dispatched = pre_dispatched_list[i]
            dispatched = dispatched_list[i]
            # Overlap: issue the shared-experts GEMM on the compute stream
            # while the dispatch all-to-all runs on the comm stream, then
            # wait for the a2a. Shared input is the attention output
            # (pre_moe_forward_out), independent of the routed tokens.
            if self.shared_overlap and self.n_shared_experts > 0:
                shared_experts_out = self._shared_experts_forward(
                    hidden_states=pre_moe_forward_out_list[i],
                )
                shared_experts_out_list.append(shared_experts_out)
            # wait for pre-dispatch event
            post_dispatched = self.dispatcher.dispatch_postprocess(
                pre_dispatched=pre_dispatched,
                dispatched=dispatched,
                async_op=True,
            )
            if self.ep_mesh is not None:
                # Preserve the same dynamic-token compile contract for every in-layer micro-batch.
                torch._dynamo.mark_dynamic(post_dispatched["hidden_states"], 0)
            experts_out = self.experts(
                post_dispatched["hidden_states"],
                post_dispatched["tokens_per_expert"],
                weight_layout=post_dispatched["expert_weight_layout"],
            )

            pre_combined = self.dispatcher.combine_preprocess(
                hidden_states=experts_out,
                pre_dispatched=pre_dispatched,
                dispatched=dispatched,
                post_dispatched=post_dispatched,
                async_op=True,
            )

            post_dispatched_list.append(post_dispatched)
            experts_out_list.append(experts_out)
            pre_combined_list.append(pre_combined)

        def _issue_combine(i: int) -> torch.Tensor:
            """Issue micro-batch ``i``'s routed combine and return its output.

            Reads the lists by index, so it must follow ``_issue_routed_compute(i)``.
            """
            return self.dispatcher.combine(
                pre_combined=pre_combined_list[i],
                pre_dispatched=pre_dispatched_list[i],
                dispatched=dispatched_list[i],
                post_dispatched=post_dispatched_list[i],
                async_op=True,
            )

        combine_overlap = _combine_overlap_enabled()
        combined_by_index: dict[int, torch.Tensor] = {}

        if interleave:
            # Software pipeline: submit dispatch i, then immediately issue the
            # compute of micro-batch i-1. That compute needs no host-side data
            # from dispatch i, so it lands on the compute stream while dispatch
            # i still runs on the comm stream.
            issued = 0
            for i, router_results in enumerate(router_results_list):
                dispatched_list.append(
                    self.dispatcher.dispatch(
                        pre_dispatched=pre_dispatched_list[i],
                        topk_weights=router_results["topk_weights"],
                        async_op=True,
                    )
                )
                if issued < i:
                    _issue_routed_compute(issued)
                    if combine_overlap:
                        combined_by_index[issued] = _issue_combine(issued)
                    issued += 1
            while issued < len(dispatched_list):
                _issue_routed_compute(issued)
                if combine_overlap:
                    combined_by_index[issued] = _issue_combine(issued)
                issued += 1
        else:
            for i, pre_dispatched in enumerate(pre_dispatched_list):
                dispatched_list.append(
                    self.dispatcher.dispatch(
                        pre_dispatched=pre_dispatched,
                        topk_weights=router_results_list[i]["topk_weights"],
                        async_op=True,
                    )
                )
                _issue_routed_compute(i)
                if combine_overlap:
                    combined_by_index[i] = _issue_combine(i)

        for i in range(len(pre_combined_list)):
            if i not in combined_by_index:
                combined_by_index[i] = self.dispatcher.combine(
                    pre_combined=pre_combined_list[i],
                    pre_dispatched=pre_dispatched_list[i],
                    dispatched=dispatched_list[i],
                    post_dispatched=post_dispatched_list[i],
                    async_op=True,
                )
            combined_list.append(combined_by_index.pop(i))

        # One cross-rank barrier per layer. ``mb_idx`` already keeps one
        # micro-batch's slot reset out of another's way, so only the handover to
        # the next layer's dispatch needs one — and doing it per micro-batch
        # would cut across the combine/GEMM overlap MOONEP_COMBINE_OVERLAP builds.
        layer_barrier = getattr(self.dispatcher, "layer_barrier", None)
        if layer_barrier is not None:
            layer_barrier()

        if self.n_shared_experts > 0 and not self.shared_overlap:
            # Serial path: shared-experts GEMM after combine (no overlap).
            shared_experts_out_list = []
            for pre_moe_forward_out in pre_moe_forward_out_list:
                shared_experts_out = self._shared_experts_forward(
                    hidden_states=pre_moe_forward_out,
                )
                shared_experts_out_list.append(shared_experts_out)
        elif self.n_shared_experts > 0:
            # shared_overlap: shared forward already issued inside the dispatch
            # loop above; nothing to do here.
            pass
        else:
            shared_experts_out_list = [None] * intra_layer_micro_batch

        hidden_states_out_list: list[torch.Tensor] = []
        for i in range(intra_layer_micro_batch):
            post_combined = self.dispatcher.combine_postprocess(
                pre_dispatched=pre_dispatched_list[i],
                dispatched=dispatched_list[i],
                post_dispatched=post_dispatched_list[i],
                pre_combined=pre_combined_list[i],
                combined=combined_list[i],
                async_op=True,
            )
            pre_shape = pre_moe_forward_out_list[i].shape
            combined_hidden_states = post_combined["hidden_states"].view(pre_shape[0], pre_shape[1], -1)
            combined_hidden_states = self._project_from_experts(combined_hidden_states)
            hidden_states = self._post_moe_forward(
                combined_hidden_states=combined_hidden_states,
                residual=residual_list[i],
                shared_experts_out=shared_experts_out_list[i],
            )
            hidden_states_out_list.append(hidden_states)

        return self._build_micro_batch_output(
            hidden_states_list=hidden_states_out_list,
            router_results_list=router_results_list,
            attn_outputs_list=attn_outputs_list,
        )

    def _local_packed_micro_batch_forward(
        self,
        hidden_states_list: list[torch.Tensor],
        seq_ctx_list: list[SequenceContext],
        position_embeddings_list: list[tuple[torch.Tensor, torch.Tensor]],
        attention_kwargs_list: list[dict[str, object]] | None = None,
    ) -> MoEDecoderLayerMicroBatchOutput:
        """Run routed experts once over the concatenated local micro-batch
        tokens."""
        origin_shape = hidden_states_list[0].shape
        assert all(hidden_states.shape == origin_shape for hidden_states in hidden_states_list), (
            "All hidden states should have the same shape"
        )
        if attention_kwargs_list is None:
            attention_kwargs_list = [{} for _ in hidden_states_list]
        assert len(attention_kwargs_list) == len(hidden_states_list)

        residual_list: list[torch.Tensor] = []
        router_results_list: list[RouterResults] = []
        attn_outputs_list: list[AttnOutputs] = []
        pre_moe_forward_out_list: list[torch.Tensor] = []
        expert_inputs: list[torch.Tensor] = []
        topk_ids_list: list[torch.Tensor] = []
        topk_weights_list: list[torch.Tensor] = []

        for hidden_states, attention_kwargs, seq_ctx, position_embeddings in zip(
            hidden_states_list,
            attention_kwargs_list,
            seq_ctx_list,
            position_embeddings_list,
        ):
            residual, hidden_states, router_results, attn_outputs = self._pre_moe_forward(
                hidden_states=hidden_states,
                seq_ctx=seq_ctx,
                position_embeddings=position_embeddings,
                state=ForwardState.TRAINING,
                attention_kwargs=attention_kwargs,
            )
            pre_moe_forward_out_list.append(hidden_states)
            expert_in = self._project_to_experts(hidden_states)
            expert_inputs.append(expert_in.view(-1, expert_in.shape[-1]))
            topk_ids_list.append(router_results["topk_ids"].reshape(-1, router_results["topk_ids"].shape[-1]))
            topk_weights_list.append(
                router_results["topk_weights"].reshape(-1, router_results["topk_weights"].shape[-1])
            )
            residual_list.append(residual)
            router_results_list.append(router_results)
            attn_outputs_list.append(attn_outputs)

        token_counts = [expert_in.shape[0] for expert_in in expert_inputs]
        expert_in = torch.cat(expert_inputs, dim=0)
        topk_ids = torch.cat(topk_ids_list, dim=0)
        topk_weights = torch.cat(topk_weights_list, dim=0)
        assert topk_ids.shape == topk_weights.shape
        assert topk_ids.shape[0] == expert_in.shape[0]

        assert self.dispatcher is not None
        pre_dispatched = self.dispatcher.dispatch_preprocess(
            hidden_states=expert_in,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            tokens_per_expert=torch.bincount(topk_ids.flatten(), minlength=self.n_routed_experts),
            layer_state=None,
            async_op=False,
        )
        dispatched = self.dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            topk_weights=topk_weights,
            async_op=False,
            decoding=False,
        )
        post_dispatched = self.dispatcher.dispatch_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            async_op=False,
        )
        if self.ep_mesh is not None:
            torch._dynamo.mark_dynamic(post_dispatched["hidden_states"], 0)
        experts_out = self.experts(
            post_dispatched["hidden_states"],
            post_dispatched["tokens_per_expert"],
            weight_layout=post_dispatched["expert_weight_layout"],
        )
        pre_combined = self.dispatcher.combine_preprocess(
            hidden_states=experts_out,
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            async_op=False,
            decoding=False,
        )
        combined = self.dispatcher.combine(
            pre_combined=pre_combined,
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            async_op=False,
            decoding=False,
        )
        post_combined = self.dispatcher.combine_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            combined=combined,
            async_op=False,
        )
        combined_hidden_states_list = torch.split(post_combined["hidden_states"], token_counts, dim=0)

        shared_experts_out_list: list[torch.Tensor | None]
        if self.n_shared_experts > 0:
            shared_experts_out_list = [
                self._shared_experts_forward(hidden_states=hidden_states) for hidden_states in pre_moe_forward_out_list
            ]
        else:
            shared_experts_out_list = [None] * len(hidden_states_list)

        hidden_states_out_list: list[torch.Tensor] = []
        for i, combined_hidden_states in enumerate(combined_hidden_states_list):
            pre_shape = pre_moe_forward_out_list[i].shape
            combined_hidden_states = combined_hidden_states.view(pre_shape[0], pre_shape[1], -1)
            combined_hidden_states = self._project_from_experts(combined_hidden_states)
            hidden_states = self._post_moe_forward(
                combined_hidden_states=combined_hidden_states,
                residual=residual_list[i],
                shared_experts_out=shared_experts_out_list[i],
            )
            hidden_states_out_list.append(hidden_states)

        return self._build_micro_batch_output(
            hidden_states_list=hidden_states_out_list,
            router_results_list=router_results_list,
            attn_outputs_list=attn_outputs_list,
        )

    def _build_micro_batch_output(
        self,
        *,
        hidden_states_list: list[torch.Tensor],
        router_results_list: list[RouterResults],
        attn_outputs_list: list[AttnOutputs],
    ) -> MoEDecoderLayerMicroBatchOutput:
        """Build the public micro-batch output; model-specific decoders may
        extend it."""
        return {
            "hidden_states": hidden_states_list,
            "router_logits": [router_results["logits"] for router_results in router_results_list],
            "router_weights": [router_results["router_weights"] for router_results in router_results_list],
            "router_topk_ids": [router_results["topk_ids"] for router_results in router_results_list],
        }

    def _sonicmoe_micro_batch_forward(
        self,
        hidden_states_list: list[torch.Tensor],
        seq_ctx_list: list[SequenceContext],
        position_embeddings_list: list[tuple[torch.Tensor, torch.Tensor]],
        attention_kwargs_list: list[dict[str, object]] | None = None,
    ) -> MoEDecoderLayerMicroBatchOutput:
        """Execute intra-layer micro-batches without the EP dispatcher
        pipeline."""
        origin_shape = hidden_states_list[0].shape
        assert all(hidden_states.shape == origin_shape for hidden_states in hidden_states_list), (
            "All hidden states should have the same shape"
        )

        hidden_states_out_list: list[torch.Tensor] = []
        router_results_list: list[RouterResults] = []
        attn_outputs_list: list[AttnOutputs] = []
        if attention_kwargs_list is None:
            attention_kwargs_list = [{} for _ in hidden_states_list]
        assert len(attention_kwargs_list) == len(hidden_states_list)
        for hidden_states, attention_kwargs, seq_ctx, position_embeddings in zip(
            hidden_states_list,
            attention_kwargs_list,
            seq_ctx_list,
            position_embeddings_list,
        ):
            residual, expert_inputs, router_results, attn_outputs = self._pre_moe_forward(
                hidden_states=hidden_states,
                seq_ctx=seq_ctx,
                position_embeddings=position_embeddings,
                state=ForwardState.TRAINING,
                attention_kwargs=attention_kwargs,
            )
            output = self._sonicmoe_forward(
                hidden_states=expert_inputs,
                residual=residual,
                router_results=router_results,
            )
            hidden_states_out_list.append(output)
            router_results_list.append(router_results)
            attn_outputs_list.append(attn_outputs)

        return self._build_micro_batch_output(
            hidden_states_list=hidden_states_out_list,
            router_results_list=router_results_list,
            attn_outputs_list=attn_outputs_list,
        )

    def _sonicmoe_forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        router_results: RouterResults,
    ) -> torch.Tensor:
        combined_hidden_states = self.experts.forward_routed(
            hidden_states.view(-1, hidden_states.shape[-1]),
            router_results["topk_ids"],
            router_results["topk_weights"],
            router_results["router_weights"],
        ).view_as(hidden_states)
        shared_experts_out = (
            self._shared_experts_forward(hidden_states=hidden_states) if self.n_shared_experts > 0 else None
        )
        return self._post_moe_forward(
            combined_hidden_states=combined_hidden_states,
            residual=residual,
            shared_experts_out=shared_experts_out,
        )

    def _pre_moe_forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        state: ForwardState,
        past_key_values: list[list[torch.Tensor]] | None = None,
        attention_kwargs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RouterResults, AttnOutputs]:
        # NOTE: In order to allow `torch.compile` to compile the ops before and after attention as much as possible,
        # attention, post-layernorm and gate are implemented in one function
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if state == ForwardState.TRAINING:
            attention_forward = cast(Callable[..., AttnOutputs], self.self_attn)
            attn_outputs = attention_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
                **(attention_kwargs or {}),
            )
            hidden_states = attn_outputs["projected_output"]
        elif state == ForwardState.PREFILLING:
            attn_outputs = {}
            assert past_key_values is not None, "past_key_values should be provided in pre-filling state"
            hidden_states = self.self_attn.prefilling(  # type: ignore
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
                past_key_values=past_key_values,
            )
        elif state == ForwardState.DECODING:
            attn_outputs = {}
            assert past_key_values is not None, "past_key_values should be provided in decoding state"
            hidden_states = self.self_attn.decoding(  # type: ignore
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
                past_key_values=past_key_values,
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if seq_ctx.rollout_routed_experts is not None and self.layer_idx < seq_ctx.rollout_routed_experts.shape[1]:
            rollout_routed_experts = seq_ctx.rollout_routed_experts[:, self.layer_idx, :]  # seq_l, expert
            # TODO: pin_memory() + to(device, non_blocking=True) on a CUDA stream would allow overlapping the transfer
            # with prior-layer compute
            rollout_routed_experts = _prepare_rollout_routed_experts_for_router(
                rollout_routed_experts,
                hidden_states,
                offload_rollout_routed_experts=seq_ctx.offload_rollout_routed_experts,
            )
        else:
            rollout_routed_experts = None
        router_results: RouterResults = self.gate(hidden_states, rollout_routed_experts)
        return residual, hidden_states, router_results, attn_outputs

    def _shared_experts_forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert self.shared_experts is not None, "Shared experts should be initialized when n_shared_experts > 0"
        shared_experts_out = self.shared_experts(hidden_states)

        if self.with_shared_expert_gate:
            assert self.shared_expert_gate is not None, (
                "Shared expert gate should be initialized when with_shared_expert_gate is True"
            )
            shared_experts_out = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_experts_out

        return shared_experts_out

    def _post_moe_forward(
        self,
        combined_hidden_states: torch.Tensor,
        residual: torch.Tensor,
        shared_experts_out: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.n_shared_experts > 0:
            shared_experts_out = cast(torch.Tensor, shared_experts_out)
            combined_hidden_states = combined_hidden_states + shared_experts_out
        return combined_hidden_states * self.hidden_factor + residual

    def build_kv_cache(
        self, max_batch_size: int | None = None, max_length: int | None = None, block_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.self_attn.build_kv_cache(  # type: ignore
            max_batch_size=max_batch_size,
            max_length=max_length,
            block_size=block_size,
        )


class _BackwardSync(Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        previous_backward_event: torch.cuda.Event | None = None,
        finished_backward_event: torch.cuda.Event | None = None,
        name=None,
    ) -> torch.Tensor:
        ctx.previous_backward_event = previous_backward_event
        ctx.finished_backward_event = finished_backward_event
        ctx.name = name
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        current_stream = torch.cuda.current_stream()

        if ctx.previous_backward_event is not None:
            current_stream.wait_event(ctx.previous_backward_event)
        if ctx.finished_backward_event is not None:
            current_stream.record_event(ctx.finished_backward_event)

        return grad_output, None, None, None


backward_sync = _BackwardSync.apply
