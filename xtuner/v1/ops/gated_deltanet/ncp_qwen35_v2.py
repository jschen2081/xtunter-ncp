"""Zero-centered GDN norm-gate with the checkpoint's BF16 rounding order."""

from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _zero_norm_gate_kernel(x, gate, weight, out, eps: tl.constexpr, width: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    if width == 128:
        # Match the pinned TE-equivalent GDN reduction (32 lanes, vector 4).
        lane = tl.arange(0, 32)
        acc = tl.full((32,), 0, tl.float32)
        for element in tl.static_range(4):
            value = tl.load(x + row * width + lane * 4 + element).to(tl.float32)
            acc = tl.fma(value, value, acc)
        for step in tl.static_range(5):
            acc = acc + tl.gather(acc, lane ^ (1 << step), 0)
        rstd = tl.rsqrt(tl.fma(acc, 1.0 / width, eps))
        for element in tl.static_range(4):
            offset = lane * 4 + element
            value = tl.load(x + row * width + offset).to(tl.float32)
            gamma = tl.load(weight + offset).to(tl.float32) + 1.0
            normalized = (rstd * value * gamma).to(x.dtype.element_ty).to(tl.float32)
            g = tl.load(gate + row * width + offset).to(tl.float32)
            activated = g / (1.0 + libdevice.exp(-g))
            tl.store(out + row * width + offset, normalized * activated)
    else:
        offset = tl.arange(0, block)
        value = tl.load(x + row * width + offset, offset < width, 0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(value * value, 0) / width + eps)
        gamma = tl.load(weight + offset, offset < width, 0).to(tl.float32) + 1.0
        normalized = (rstd * value * gamma).to(x.dtype.element_ty).to(tl.float32)
        g = tl.load(gate + row * width + offset, offset < width, 0).to(tl.float32)
        tl.store(out + row * width + offset, normalized * (g / (1.0 + libdevice.exp(-g))), offset < width)


@torch.library.custom_op("xtuner::ncp_v2_zero_norm_gate", mutates_args=())
def zero_rms_norm_gate(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fuse GDN normalization and SiLU while preserving the normalized BF16 boundary."""
    x, gate, weight = x.contiguous(), gate.contiguous(), weight.contiguous()
    out = torch.empty_like(x)
    width = x.shape[-1]
    _zero_norm_gate_kernel[(x.numel() // width,)](
        x, gate, weight, out, eps, width, triton.next_power_of_2(width), num_warps=1, enable_fp_fusion=False
    )
    return out


@zero_rms_norm_gate.register_fake
def _zero_rms_norm_gate_fake(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    return torch.empty_like(x)


def _setup_context(ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float], output: torch.Tensor) -> None:
    x, gate, weight, ctx.eps = inputs
    ctx.save_for_backward(x, gate, weight)


def _backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    x, gate, weight = ctx.saved_tensors
    xf, gf, wf, dy = x.float(), gate.float(), weight.float() + 1.0, grad.float()
    rstd = torch.rsqrt(xf.square().mean(-1, keepdim=True) + ctx.eps)
    unit = xf * rstd
    normalized = (unit * wf).to(x.dtype).float()
    sigmoid = gf.sigmoid()
    dnorm = (dy * (gf * sigmoid)).to(x.dtype).float()
    du = dnorm * wf
    dx = rstd * (du - unit * (du * unit).mean(-1, keepdim=True))
    dg = dy * normalized * sigmoid * (1.0 + gf * (1.0 - sigmoid))
    dw = (dnorm * unit).sum(tuple(range(x.ndim - 1)))
    return dx.to(x.dtype), dg.to(gate.dtype), dw.to(weight.dtype), None


zero_rms_norm_gate.register_autograd(_backward, setup_context=_setup_context)
