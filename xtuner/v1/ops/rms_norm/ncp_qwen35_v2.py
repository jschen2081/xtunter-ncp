"""Pinned zero-centered RMSNorm kernel for NCP Qwen3.5 v2."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _general_norm(x_ptr, weight_ptr, output_ptr, eps: tl.constexpr, width: tl.constexpr, vector: tl.constexpr):
        row = tl.program_id(0)
        lane = tl.arange(0, 32)
        acc = tl.full((32,), 0, tl.float32)
        for chunk in tl.static_range(width // (32 * vector)):
            for element in tl.static_range(vector):
                offset = lane * vector + chunk * 32 * vector + element
                value = tl.load(x_ptr + row * width + offset).to(tl.float32)
                acc = tl.fma(value, value, acc)
        for step in tl.static_range(5):
            acc = acc + tl.gather(acc, lane ^ (1 << step), 0)
        reciprocal_std = tl.rsqrt(tl.fma(acc, 1.0 / width, eps))
        for chunk in tl.static_range(width // (32 * vector)):
            for element in tl.static_range(vector):
                offset = lane * vector + chunk * 32 * vector + element
                value = tl.load(x_ptr + row * width + offset).to(tl.float32)
                gamma = tl.load(weight_ptr + offset).to(tl.float32) + 1.0
                tl.store(output_ptr + row * width + offset, reciprocal_std * value * gamma)

    @triton.jit
    def _tuned_norm(x_ptr, weight_ptr, output_ptr, eps: tl.constexpr, width: tl.constexpr):
        row = tl.program_id(0)
        lane = tl.arange(0, 32)
        mean = tl.full((32,), 0, tl.float32)
        for chunk in tl.static_range(width // 256):
            for element in tl.static_range(8):
                offset = lane * 8 + chunk * 256 + element
                mean += tl.load(x_ptr + row * width + offset).to(tl.float32)
        for step in tl.static_range(5):
            mean = mean + tl.gather(mean, lane ^ (1 << step), 0)
        mean *= 1.0 / width
        second_moment = tl.full((32,), 0, tl.float32)
        for chunk in tl.static_range(width // 256):
            for element in tl.static_range(8):
                offset = lane * 8 + chunk * 256 + element
                delta = tl.load(x_ptr + row * width + offset).to(tl.float32) - mean
                second_moment = tl.fma(delta, delta, second_moment)
        for step in tl.static_range(5):
            second_moment = second_moment + tl.gather(second_moment, lane ^ (1 << step), 0)
        reciprocal_std = tl.rsqrt(tl.fma(second_moment, 1.0 / width, mean * mean) + eps)
        for chunk in tl.static_range(width // 256):
            for element in tl.static_range(8):
                offset = lane * 8 + chunk * 256 + element
                value = tl.load(x_ptr + row * width + offset).to(tl.float32)
                gamma = tl.load(weight_ptr + offset).to(tl.float32) + 1.0
                tl.store(output_ptr + row * width + offset, reciprocal_std * value * gamma)


def zero_rmsnorm_cuda(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Match the validated TE 2.14.1 reduction for supported BF16 widths."""
    if triton is None:
        raise RuntimeError("CUDA RMSNorm requires the pinned Triton runtime")
    width = x.shape[-1]
    rows = x.numel() // width
    if not x.is_cuda or x.dtype != torch.bfloat16 or width not in (128, 512, 1536, 4096):
        raise ValueError("NCP RMSNorm requires CUDA BF16 with width 128/512/1536/4096")
    x = x.contiguous()
    weight = weight.contiguous()
    output = torch.empty_like(x)
    if width in (512, 4096) and rows % 4 == 0:
        _tuned_norm[(rows,)](x, weight, output, eps, width, num_warps=1)
    else:
        vector = 4 if width == 128 else 8
        _general_norm[(rows,)](x, weight, output, eps, width, vector, num_warps=1)
    return output


__all__ = ["zero_rmsnorm_cuda"]
