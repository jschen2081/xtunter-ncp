"""GatedDeltaNet op-level dispatchers.

`XTUNER_HF_IMPL` controls which implementations XTuner's `GatedDeltaNet` module uses,
mirroring how `xtuner/v1/ops/attn_imp.py::get_attn_impl_fn` and the rms_norm selector
switch between fast / fused paths and HF-exact paths. Under `XTUNER_HF_IMPL=true`:

* `chunk_gated_delta_rule` is the canonical `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
  (same callable HF's `Qwen3_5GatedDeltaNet` uses), bypassing XTuner's
  `torch.library.custom_op` wrap.
* `causal_conv1d_fn` is the high-level `causal_conv1d.causal_conv1d_fn` adapted to XTuner's
  channel-last call site: the adapter transposes to channel-first, calls the package wrapper
  with ``seq_idx=None`` (HF's non-packed convention), and transposes back. XTuner's own wrap
  binds the channel-last/seq_idx convention together via an internal transpose, which gives a
  different backward op graph from HF's call pattern even though forward is bitwise.

`get_causal_conv1d_fn(hf_backend="fla")` selects the FLA eager convolution for
NCP Qwen3.5 v2 HF parity, preserving packed document boundaries via ``cu_seqlens``.

`XTUNER_GDN_CUSTOM_OP` independently overrides the delta-rule dispatcher, allowing
compile-friendly custom ops while keeping the other HF implementations selected.

These switches are only meant for the bitwise-parity tests. Production / training stays on the
XTuner path (compile-friendly custom_op wraps + seq_idx-aware kernel dispatch).

Shared FSDP helpers used by both ``GatedDeltaNet`` and ``KDA``:

* ``fused_rms_norm_gated.FusedRMSNormGated`` — DTensor-aware gated RMSNorm
* ``dtensor`` — ``to_local`` / ``full_tensor`` / depthwise-conv weight materialization
"""

import os
from collections.abc import Callable
from typing import Literal

import torch


_TRUTHY = {"true", "1", "yes", "on"}


def _hf_impl_enabled() -> bool:
    return os.getenv("XTUNER_HF_IMPL", "").strip().lower() in _TRUTHY


def _gdn_ascendc_enabled() -> bool:
    from xtuner.v1.utils import get_device

    return os.environ.get("XTUNER_NPU_GDN_ASCENDC", "1") == "1" and get_device() == "npu"


def _npu_conv_enabled() -> bool:
    from xtuner.v1.utils import get_device

    return os.environ.get("XTUNER_NPU_CONV", "1") == "1" and get_device() == "npu"


def _custom_op_enabled() -> bool:
    override = os.getenv("XTUNER_GDN_CUSTOM_OP")
    if override is not None:
        return override.strip().lower() in _TRUTHY
    return not _hf_impl_enabled()


def _hf_causal_conv1d_adapter(x, weight, bias, activation, seq_idx):
    from causal_conv1d import causal_conv1d_fn as _hf_causal_conv1d_fn

    # XTuner's GatedDeltaNet supplies `x` in channel-last (``(batch, seq, dim)``) and always
    # passes a `seq_idx` tensor; HF's call site uses channel-first and `seq_idx=None` for
    # non-packed batches. Adapt: transpose, drop seq_idx, transpose back.
    x_cf = x.transpose(1, 2)
    out = _hf_causal_conv1d_fn(x=x_cf, weight=weight, bias=bias, activation=activation, seq_idx=None)
    return out.transpose(1, 2)


def _fla_causal_conv1d_adapter(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    seq_idx: torch.Tensor | None,
) -> torch.Tensor:
    from fla.modules.convolution import causal_conv1d

    cu_seqlens = None
    original_shape = x.shape
    if seq_idx is not None:
        if seq_idx.shape != x.shape[:2]:
            raise ValueError("seq_idx must match the convolution batch and sequence dimensions")
        # Only the eager parity path converts sequence IDs to FLA document offsets.
        # Leave ordinary batches unchanged to match the HF reference call exactly.
        changes = seq_idx[:, 1:] != seq_idx[:, :-1]
        if bool(changes.any()):
            starts = torch.ones_like(seq_idx, dtype=torch.bool)
            starts[:, 1:] = changes
            offsets = starts.flatten().nonzero().flatten().to(torch.int32)
            end = offsets.new_tensor([seq_idx.numel()])
            cu_seqlens = torch.cat((offsets, end))
            x = x.reshape(1, -1, x.shape[-1])
    output, _ = causal_conv1d(
        x,
        weight,
        bias=bias,
        activation=activation,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    return output.reshape(original_shape)


def get_chunk_gated_delta_rule_fn():
    if not _custom_op_enabled():
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _hf_chunk_gated_delta_rule

        return _hf_chunk_gated_delta_rule
    if _gdn_ascendc_enabled():
        from .chunk_gated_delta_rule_npu import ascendc_chunk_gated_delta_rule_fn

        fn = ascendc_chunk_gated_delta_rule_fn()
        if fn is not None:
            return fn
    from .chunk_gated_delta_rule import chunk_gated_delta_rule as _xtuner_chunk_gated_delta_rule

    return _xtuner_chunk_gated_delta_rule


def get_causal_conv1d_fn(*, hf_backend: Literal["cuda", "fla"] = "cuda") -> Callable[..., torch.Tensor]:
    """Select convolution while preserving the production kernel dispatch.

    Args:
        hf_backend (Literal["cuda", "fla"]): Eager backend used only in HF mode.

    Returns:
        Callable: Convolution accepting channel-last inputs and sequence IDs.
    """
    if hf_backend not in {"cuda", "fla"}:
        raise ValueError(f"Unsupported HF convolution backend: {hf_backend}")
    if _hf_impl_enabled():
        return _fla_causal_conv1d_adapter if hf_backend == "fla" else _hf_causal_conv1d_adapter
    if _npu_conv_enabled():
        from .causal_conv1d_npu import causal_conv1d as _npu_causal_conv1d

        return _npu_causal_conv1d
    from .causal_conv1d import causal_conv1d_fn as _xtuner_causal_conv1d_fn

    return _xtuner_causal_conv1d_fn
