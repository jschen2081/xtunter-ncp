import os

import torch


def npu_group_gemm(
    x: torch.Tensor,
    weights: torch.Tensor | list[torch.Tensor],
    split_sizes: torch.Tensor,
    *,
    grad_weight_out: torch.Tensor | None = None,
    grad_weight_first: bool = False,
) -> torch.Tensor:
    # A non-empty ``grad_weight_out`` can only come from the MoonEP dispatcher,
    # so it selects the MoonEP kernel on its own. Relying on ``DISPATCHER``
    # alone would let a stale environment silently drop the direct-to-heap dW
    # (the mindspeed call has no such argument) and lose the expert gradients.
    if grad_weight_out is not None or os.environ.get("DISPATCHER", "all2all") == "moonep":
        return npu_group_gemm_moonep(
            x,
            weights,
            split_sizes,
            grad_weight_out=grad_weight_out,
            grad_weight_first=grad_weight_first,
        )
    return npu_group_gemm_mindspeed(x, weights, split_sizes)

def npu_group_gemm_mindspeed(x: torch.Tensor, weights: torch.Tensor, split_sizes: torch.Tensor) -> torch.Tensor:
    from mindspeed.core.fusions.grouped_matmul import Ops
    weights = weights.transpose(1, 2)
    out = Ops.gmm(x, weights, split_sizes, trans_b=False)
    return out

def npu_group_gemm_moonep(
    x: torch.Tensor,
    weights: torch.Tensor | list[torch.Tensor],
    split_sizes: torch.Tensor,
    *,
    grad_weight_out: torch.Tensor | None = None,
    grad_weight_first: bool = False,
) -> torch.Tensor:
    """Grouped GEMM via MoonEP Ops.gmm (in-house, mindspeed dropped) with grad_weight_out.

    Args:
        x: [num_tokens, in_features] bf16 input activations.
        weights: [num_experts, out_features, in_features] bf16 weights —
            the MoonEP-Ascend [E+B, O, I] heap view (home+dup); the
            home+dup window is physically contiguous, so a single-segment
            full_view suffices (no cat materialization, -2.4GB@GLM).
        split_sizes: [num_experts] int tensor — tokens per expert group.
        grad_weight_out: optional [num_experts, out_features, in_features]
            bf16 tensor — the weight gradient is written straight into the
            heap slots (X8).
    """
    group_list = torch.cumsum(split_sizes, dim=0).to(x.device)
    # Single segment [E, I, O]
    weights_t = weights.transpose(1, 2)  # [E, I, O] — npu_gmm expects this
    if grad_weight_out is not None:
        # grad_weight_out is the [E,O,I] heap slots passed straight through
        #   (X8 backward does grad_t@x -> [E,O,I] direct write, no transpose).
        # MOONEP_X8_DW controls the direct-write strength (grad_weight_first
        #   carries the dispatcher's home-zero contract; it is no longer used
        #   as a gate):
        #   0 = full materialization fallback (A/B); 1 (default) = first
        #   producer whole-block direct write; 2 = first producer whole-block
        #   direct write + non-first producer dup half direct write + home
        #   materialized accumulation (branched inside gmm.py).
        from .gmm import npu_gmm_with_grad_weight_out

        dw_mode = os.environ.get("MOONEP_X8_DW", "1")
        return npu_gmm_with_grad_weight_out(
            x, weights_t, group_list=group_list,
            grad_weight_out=grad_weight_out, original_weight=None,
            grad_weight_first=grad_weight_first and dw_mode != "0",
        )
    else:
        from .gmm import npu_gmm
        return npu_gmm(x, weights_t, bias=None, group_list=group_list,
                       group_type=0)
