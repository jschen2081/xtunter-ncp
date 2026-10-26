"""RMSNorm variants used by the NCP Qwen3.5 v2 checkpoint."""

import os

import torch
from torch import Tensor, nn

from xtuner.v1.ops.rms_norm.native_backward import rms_norm_native_backward


def env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


@torch.library.custom_op("xtuner::low_precision_rms_weight", mutates_args=())
def _low_precision_rms_weight(weight: Tensor) -> Tensor:
    # An explicit boundary preserves BF16/FP16 addition rounding when the
    # effective scale is subsequently promoted for cached source multiplication.
    return weight + 1.0


@_low_precision_rms_weight.register_fake
def _low_precision_rms_weight_fake(weight: Tensor) -> Tensor:
    return torch.empty_like(weight)


def _low_precision_rms_weight_backward(ctx, grad: Tensor) -> Tensor:
    return grad


_low_precision_rms_weight.register_autograd(_low_precision_rms_weight_backward)


def effective_weight(weight: Tensor) -> Tensor:
    """Convert a zero-centered gamma to its effective scale."""
    if torch.compiler.is_compiling() and weight.dtype in (torch.float16, torch.bfloat16):
        return _low_precision_rms_weight(weight)
    return weight + 1.0


def eager_zero_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (normalized * effective_weight(weight).float()).to(x.dtype)


def zero_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Differentiable zero-centered normalization used by the model."""
    if env_on("NCP_Q35V2_TORCH_RMS"):
        return rms_norm_native_backward(x, effective_weight(weight), eps)
    return eager_zero_rms_norm(x, weight, eps)


def mudd_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Route/output norm used by MUDD blocks in the reference model."""
    enabled = os.environ.get("NCP_Q35V2_MUDD_RMS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if not enabled:
        return eager_zero_rms_norm(x, weight, eps)
    return rms_norm_native_backward(x, effective_weight(weight), eps)


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm whose stored weight is the offset from one."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        return zero_rms_norm(x, self.weight, self.eps)


class GatedDeltaNetInputRMSNorm(ZeroCenteredRMSNorm):
    """Input norm matching the HF GDN layer's default torch RMSNorm path."""

    def forward(self, x: Tensor) -> Tensor:
        if env_on("NCP_Q35V2_GDN_TORCH_INPUT_NORM", "1"):
            return rms_norm_native_backward(x, effective_weight(self.weight), self.eps)
        return super().forward(x)


class MuddRMSNorm(nn.Module):
    """Zero-centered RMSNorm with the MUDD numerical dispatch contract."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        return mudd_rms_norm(x, self.weight, self.eps)


__all__ = [
    "GatedDeltaNetInputRMSNorm",
    "MuddRMSNorm",
    "ZeroCenteredRMSNorm",
    "effective_weight",
    "env_on",
    "mudd_rms_norm",
]
