"""Keep CUDA RMSNorm backward native when its caller is compiled."""

import os
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor


def rms_norm_native_backward(x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """Apply last-dimension RMSNorm without decomposing its CUDA backward.

    Args:
        x (Tensor): Input activations.
        weight (Tensor | None): Effective scale, including any zero-centered offset.
        eps (float): Epsilon added to the mean square.

    Returns:
        Tensor: Normalized activations with the input shape and dtype.

    Set NCP_Q35V2_NATIVE_RMS_BACKWARD=1 before compilation to opt in.
    """
    if (
        os.environ.get("NCP_Q35V2_NATIVE_RMS_BACKWARD", "0").strip().lower() not in {"1", "true", "yes", "on"}
        or x.device.type != "cuda"
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return F.rms_norm(x, (x.shape[-1],), weight, eps)
    return cast(Tensor, _NativeBackwardRMSNorm.apply(x, weight, eps))


@torch.library.custom_op(
    "xtuner::rms_norm_native_backward",
    mutates_args=(),
    schema=(
        "(Tensor grad_out, Tensor x, Tensor rstd, Tensor? weight, bool need_input, bool need_weight)"
        " -> (Tensor?, Tensor?)"
    ),
)
def _native_backward(
    grad_out: Tensor,
    x: Tensor,
    rstd: Tensor,
    weight: Tensor | None,
    need_input: bool,
    need_weight: bool,
) -> tuple[Tensor | None, Tensor | None]:
    # Keep the native CUDA input/weight reductions opaque to AOTAutograd and Inductor.
    return torch.ops.aten._fused_rms_norm_backward.default(
        grad_out, x, [x.shape[-1]], rstd, weight, [need_input, need_weight]
    )


@_native_backward.register_fake
def _native_backward_fake(
    grad_out: Tensor,
    x: Tensor,
    rstd: Tensor,
    weight: Tensor | None,
    need_input: bool,
    need_weight: bool,
) -> tuple[Tensor | None, Tensor | None]:
    dx = torch.empty_like(x, memory_format=torch.contiguous_format) if need_input else None
    dw = torch.empty_like(weight) if need_weight and weight is not None else None
    return dx, dw


class _NativeBackwardRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
        output, rstd = torch.ops.aten._fused_rms_norm.default(x, [x.shape[-1]], weight, eps)
        ctx.save_for_backward(x, rstd, weight)
        return output

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor | None, Tensor | None, None]:
        x, rstd, weight = ctx.saved_tensors
        dx, dw = _native_backward(
            grad_out, x, rstd, weight, ctx.needs_input_grad[0], weight is not None and ctx.needs_input_grad[1]
        )
        return dx, dw, None


