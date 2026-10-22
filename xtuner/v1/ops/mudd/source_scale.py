"""Tiled weight-gradient reduction for cached Cross-MUDD source scaling."""

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor


def scale_source(x: Tensor, scale: Tensor) -> Tensor:
    """Multiply a source by its channel scale with a tiled CUDA backward.

    Args:
        x (Tensor): Source tensor with channels in its last dimension.
        scale (Tensor): One-dimensional channel scale.

    Returns:
        Tensor: Scaled source, retaining ordinary multiplication semantics.
    """
    if (
        x.device.type != "cuda"
        or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or scale.dtype != x.dtype
    ):
        # Mixed dtypes promote in forward; only take the custom path when the
        # backward cast to x.dtype is a no-op, keeping gradients eager-exact.
        return x * scale
    return cast(Tensor, _ScaleSource.apply(x, scale))


@triton.jit
def _partial_weight_gradient(
    source, grad, partial, rows: tl.constexpr, width: tl.constexpr,
    ROWS: tl.constexpr, COLS: tl.constexpr,
):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.program_id(1) * COLS + tl.arange(0, COLS)
    offsets = r[:, None] * width + c[None, :]
    mask = (r[:, None] < rows) & (c[None, :] < width)
    x = tl.load(source + offsets, mask, other=0).to(tl.float32)
    dy = tl.load(grad + offsets, mask, other=0).to(tl.float32)
    # Ordinary BF16/FP16 mul rounds before sum; preserve that intermediate rounding.
    product = (x * dy).to(source.dtype.element_ty).to(tl.float32)
    sums = tl.sum(product, axis=0)
    tl.store(partial + tl.program_id(0) * width + c, sums, c < width)


@torch.library.custom_op("xtuner::mudd_scale_weight_gradient", mutates_args=())
def _weight_gradient(x: Tensor, grad: Tensor) -> Tensor:
    width = x.shape[-1]
    rows = x.numel() // width
    if rows == 0:
        return torch.zeros((width,), device=x.device, dtype=x.dtype)
    x = x.contiguous()
    grad = grad.contiguous()
    partial = torch.empty((triton.cdiv(rows, 32), width), device=x.device, dtype=torch.float32)
    _partial_weight_gradient[(triton.cdiv(rows, 32), triton.cdiv(width, 128))](
        x, grad, partial, rows, width, 32, 128, num_warps=4
    )
    return partial.sum(dim=0).to(x.dtype)


@_weight_gradient.register_fake
def _weight_gradient_fake(x: Tensor, grad: Tensor) -> Tensor:
    return x.new_empty((x.shape[-1],))


class _ScaleSource(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, scale: Tensor) -> Tensor:
        ctx.save_for_backward(x, scale)
        return x * scale

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor | None, Tensor | None]:
        x, scale = ctx.saved_tensors
        dx = (grad * scale).to(x.dtype) if ctx.needs_input_grad[0] else None
        if ctx.needs_input_grad[1]:
            if grad.dtype == x.dtype and scale.dtype == x.dtype:
                dw = _weight_gradient(x, grad)
            else:
                product = grad * x
                reduce_dims = tuple(range(x.ndim - 1))
                dw = (product.sum(dim=reduce_dims) if reduce_dims else product).to(scale.dtype)
        else:
            dw = None
        return dx, dw
