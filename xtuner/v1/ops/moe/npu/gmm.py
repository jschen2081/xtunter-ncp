"""Forked MindSpeed GMM with grad_weight_out support — MoonEP-Ascend fork.

Drops the mindspeed dependency entirely: npu_gmm / npu_gmm_backward are both
implemented by MoonEP-Ascend ops/gmm (moonep_ops._gmm, which calls CANN
aclnnGroupedMatmulV4 directly). Public semantics stay compatible with the
original.

``npu_gmm_with_grad_weight_out``: dw is computed by MoonEP npu_gmm_backward and
written into the caller-supplied output tensor (home add_ + dup copy_), while dx
stays on the activation gradient path (X8 bridge).
"""

import os

import torch

try:
    import moonep_ops
except ImportError:
    raise ImportError(
        "npu_gmm requires MoonEP-Ascend ops (moonep_ops); "
        "ensure MoonEP-Ascend is built and on PYTHONPATH"
    )

__all__ = ["npu_gmm", "npu_gmm_v2", "npu_gmm_with_grad_weight_out",
]


# ═══════════════════════════════════════════════════════════════════════
# MoonEP-Ascend GMMFunction — calls moonep_ops._gmm directly
# ═══════════════════════════════════════════════════════════════════════

class GMMFunction(torch.autograd.Function):
    """MoonEP grouped GEMM autograd (mindspeed dependency removed).

    forward: moonep_ops.npu_gmm (single x/weight + group_list, original semantics).
    backward: moonep_ops.npu_gmm_backward -> (dx, dw).
    """

    @staticmethod
    def forward(ctx, original_weight, x, weight, bias, group_args):
        del original_weight  # MoonEP has no original_weight semantics
        group_list, group_type, gemm_fusion, group_list_type, group_list_data_type = group_args
        if bias is not None and bias.requires_grad:
            raise ValueError("Bias is not supported to compute gradient!")
        if (x.requires_grad or weight.requires_grad) and group_type != 0:
            raise ValueError("group_type must be zero to compute gradients of x and weight!")
        del gemm_fusion  # not supported by MoonEP (always False in XTuner)
        bias_t = bias if bias is not None else None
        if group_list_data_type == 0:
            ctx.save_for_backward(x, weight)
        else:
            ctx.save_for_backward(x, weight, group_list)
        ctx.group_list = group_list
        ctx.group_list_type = group_list_type
        return moonep_ops.npu_gmm(x, weight, bias_t, group_list, group_type,
                                  group_list_type)

    @staticmethod
    def backward(ctx, grad_outputs):
        saved = ctx.saved_tensors
        x, weight = saved[0], saved[1]
        group_list = ctx.group_list
        dx, dw = moonep_ops.npu_gmm_backward(grad_outputs, x, weight,
                                             group_list, ctx.group_list_type)
        return None, dx, dw, None, None


# ═══════════════════════════════════════════════════════════════════════
# XTuner addition: public API with grad_weight_out (X8 direct heap write)
# ═══════════════════════════════════════════════════════════════════════

class _GMMWithGradWeightOut(torch.autograd.Function):
    """GMM wrapper that writes dw into a caller-supplied tensor.

    Forward: moonep_ops.npu_gmm.
    Backward: dx + dw via moonep_ops.npu_gmm_backward, dw written into
    grad_weight_out as home add_ + dup copy_.

    grad_weight_first: the first producer's GEMM writes dw directly over the
    whole gwo block (home+dup, empty-group rows written as 0) — overwrite ==
    add_ (home initial value) and == copy_ (dup). This removes the full dw
    materialization + add_ + copy_ (~8.4GB/step of extra traffic -> zero) and
    also frees the dispatcher from pre-zeroing home.
    Non-first producers fall back to the add_/copy_ path under MOONEP_X8_DW=1
    (the default), because a home accumulation cannot be expressed as an
    overwrite-style GEMM;
    under MOONEP_X8_DW=2 non-first producers write the dup half directly plus
    materialize the home half accumulation (dup direct write == copy_,
    equivalent per producer).
    """

    @staticmethod
    def forward(ctx, x, weight, group_list, group_list_type, grad_weight_out,
                grad_weight_first=False):
        outputs = moonep_ops.npu_gmm(x, weight, None, group_list, 0,
                                     group_list_type)
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        ctx.group_list_type = group_list_type
        ctx.grad_weight_out = grad_weight_out
        ctx.grad_weight_first = bool(grad_weight_first)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        group_list = ctx.group_list
        group_list_type = ctx.group_list_type
        gwo = ctx.grad_weight_out

        if x.shape[0] == 0:
            return None, None, None, None, None, None

        # MOONEP_X8_DW: 0 = full materialization fallback; 1 = first producer
        #                whole-block direct write (default); 2 = first producer
        #                whole-block direct write + non-first producer dup half
        #                direct write + home half materialized accumulation.
        dw_mode = os.environ.get("MOONEP_X8_DW", "1")
        # dw direct write uses "swap A/B + contiguous out", eliminating the
        # internal Transpose materialization that aclnnGroupedMatmulV4 inserts
        # for a transposed-view output.
        # Set to 0 to fall back to the old form (transposed view out), used for
        # A/B comparisons on the same library.
        dw_swap = os.environ.get("MOONEP_GMM_DW_SWAP", "1") != "0"

        if getattr(ctx, "grad_weight_first", False) and dw_mode != "0":
            # First producer: the GEMM writes dw directly over the whole gwo
            # block, overwriting all 2B rows including empty-group ones.
            # dw is organized as (x^T @ grad)^T == grad^T @ x so the GEMM
            # produces a contiguous out in heap-slot layout [E,O,I], removing
            # the "write transposed view -> CANN internal Transpose
            # materialization" step (measured 40x(0.57+0.26) ms/step; both
            # paths bitwise identical).
            # Zero dw materialization / zero add_ / zero copy_.
            if dw_swap:
                wt = weight.transpose(-1, -2)
                dx = moonep_ops.npu_gmm(grad_output, wt, None, group_list, 0,
                                        group_list_type)
                moonep_ops.npu_gmm_moep([grad_output.transpose(-1, -2)], [x],
                                        None, group_list, 2, group_list_type,
                                        gwo)
            else:  # A/B comparison only: pre-change form (transposed view out, CANN inserts Transpose)
                dx, _dw = moonep_ops.npu_gmm_backward(
                    grad_output, x, weight, group_list, group_list_type,
                    dw_out=gwo.transpose(1, 2))
            return dx, None, None, None, None, None

        if dw_mode == "2":
            # Non-first producer: dup half written directly by the GEMM plus
            # home half materialized accumulation.
            #   home accumulates across micro-batches -> GEMM materializes
            #   dw_home + add_;
            #   dup is slot-local and every row is fully written by this
            #   producer's copy_ -> GEMM direct write == copy_ (including
            #   empty-group rows written as 0, no stale data), equivalent per
            #   producer, no cross-producer invariant required.
            #   Split: x's group order matches the heap row order (home groups
            #   first, then dup groups), boundary = group_list[b-1] (total
            #   tokens of the home groups).
            #   Cost: one .item() sync to read boundary (see env comment; only
            #   taken when G>1).
            b = gwo.shape[0] // 2
            boundary = int(group_list[b - 1].item())
            # dx: same form as npu_gmm_backward (grad @ W^T, group_type=0, single group over all M)
            wt = weight.transpose(-1, -2)
            dx = moonep_ops.npu_gmm(grad_output, wt, None, group_list, 0,
                                    group_list_type)
            # home dw: materialize + add_ (cross-micro-batch accumulator,
            # cannot overwrite)
            # Same as the first producer: swap A/B so dw_home is directly
            #   [b,O,I] contiguous, and add_ runs on contiguous read/write
            #   (no strided reads, no internal Transpose materialization).
            if boundary > 0:
                if dw_swap:
                    dw_home = moonep_ops.npu_gmm(
                        grad_output[:boundary].transpose(-1, -2), x[:boundary],
                        None, group_list[:b], 2, group_list_type)
                else:  # A/B comparison only: old form, dw_home is [b,I,O] then transposed add_
                    dw_home = moonep_ops.npu_gmm(
                        x[:boundary].transpose(-1, -2), grad_output[:boundary],
                        None, group_list[:b], 2, group_list_type).transpose(1, 2)
                gwo[:b].add_(dw_home)
            # dup dw: GEMM direct write into the dup half (empty-group rows written 0 - self-cleaning, equivalent to copy_)
            if x.shape[0] - boundary > 0:
                gl_dup = (group_list[b:] - boundary).contiguous()
                if dw_swap:
                    moonep_ops.npu_gmm_moep(
                        [grad_output[boundary:].transpose(-1, -2)],
                        [x[boundary:]], None, gl_dup, 2, group_list_type,
                        gwo[b:])
                else:  # A/B comparison only: old form, transposed view out
                    moonep_ops.npu_gmm_moep(
                        [x[boundary:].transpose(-1, -2)],
                        [grad_output[boundary:]], None, gl_dup, 2,
                        group_list_type, gwo[b:].transpose(1, 2))
            else:
                gwo[b:].zero_()
            return dx, None, None, None, None, None

        # Fallback path (dw_mode == "0", or a non-first producer under "1"):
        # home accumulation -> materialize dw + add_ + dup copy_.
        # home/dup split: gwo = [epn+B, O, I], epn==B -> b = shape[0]//2.
        b = gwo.shape[0] // 2

        # dx + dw in one call: MoonEP npu_gmm_backward -> (dx, dw), dw shape ==
        # weight.sizes() = [E,I,O] (npu_gmm weight convention [E,I,O];
        # group_gemm.py feeds weights.transpose(1,2)).
        dx, dw = moonep_ops.npu_gmm_backward(grad_output, x, weight,
                                             group_list, group_list_type)
        # dw [E,I,O] -> heap slot [E,O,I] non-contiguous view (zero materialization)
        dw = dw.transpose(1, 2)

        # Write to heap: home is a cross-micro-batch accumulator -> add_;
        # dup is slot-local -> copy_.
        # Invariant: the dup copy_ overwrite semantics rely on planning
        # _dedup_dst guaranteeing "one unique writer per target row" - with
        # multiple writers this must become add_, and the direct-write premise
        # disappears.
        gwo[:b].add_(dw[:b])
        gwo[b:].copy_(dw[b:])

        # dx re-enable: with X8 wired, _MoonEPExpertGradBridge.backward is
        # triggered by dx flowing back into the expert-block input ->
        # start_gradient_completion -> reduce_grad_bf16.
        return dx, None, None, None, None, None


def npu_gmm_param_verification(x, weight, *, bias=None, group_list=None,
                               group_type=0, group_list_type=0):
    """Parameter validation (same semantics as the original)."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}.")
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be a torch.Tensor, got {type(weight)}.")
    check_optional_tensor(bias, x.device, "bias")
    if group_list is not None and not isinstance(group_list, torch.Tensor):
        raise TypeError(f"group_list must be a torch.Tensor or None, got {type(group_list)}.")
    check_optional_tensor(group_list, x.device, "group_list")
    if group_list is not None and group_list.dtype != torch.int64:
        raise TypeError(f"group_list must be int64, got {group_list.dtype}!")


def check_optional_tensor(tensor, device, name):
    if tensor is not None and tensor.device != device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device, but found at "
            f"least two devices, {device}(arg0) and {tensor.device}({name})!"
        )


def npu_gmm(x, weight, *, bias=None, group_list=None, group_type=0,
            gemm_fusion=False, original_weight=None):
    """Grouped GEMM forward (MoonEP implementation, compatible with npu_gmm semantics)."""
    del gemm_fusion  # not supported by MoonEP (always False in XTuner)
    npu_gmm_param_verification(x, weight, bias=bias, group_list=group_list,
                               group_type=group_type, group_list_type=0)
    group_args = (group_list, group_type, False, 0,
                  1 if isinstance(group_list, (torch.Tensor, type(None))) else 0)
    return GMMFunction.apply(original_weight, x, weight, bias, group_args)


def npu_gmm_v2(x, weight, *, bias=None, group_list=None, group_type=0,
               gemm_fusion=False, original_weight=None):
    """group_list_type=1 variant - MoonEP routes it through npu_gmm (group_list Tensor)."""
    del gemm_fusion
    npu_gmm_param_verification(x, weight, bias=bias, group_list=group_list,
                               group_type=group_type, group_list_type=1)
    group_args = (group_list, group_type, False, 1,
                  1 if isinstance(group_list, (torch.Tensor, type(None))) else 0)
    return GMMFunction.apply(original_weight, x, weight, bias, group_args)


def npu_gmm_with_grad_weight_out(x, weight, *, group_list,
                                 grad_weight_out=None, original_weight=None,
                                 grad_weight_first=False):
    """GMM with optional grad_weight_out (MoonEP-Ascend path).

    grad_weight_out=None -> same as npu_gmm (backward compatible).
    When provided, backward writes dw directly into grad_weight_out
    (home add_ + dup copy_).
    """
    if grad_weight_out is None:
        return npu_gmm(x, weight, bias=None, group_list=group_list,
                       group_type=0, group_list_type=0)
    npu_gmm_param_verification(x, weight, bias=None, group_list=group_list,
                               group_type=0, group_list_type=0)
    return _GMMWithGradWeightOut.apply(x, weight, group_list, 0,
                                       grad_weight_out, grad_weight_first)
