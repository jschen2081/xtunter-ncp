"""Forward-only dual Cross-MUDD kernels for explicit inference experiments.

Call the model with eval() inside torch.no_grad() or torch.inference_mode(). This implementation does not provide
autograd and is not a training backend.
"""

from typing import cast

import torch
from torch import Tensor


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

Streams = tuple[Tensor, Tensor, Tensor, Tensor]
_MAX_CROSS_MUDD_SOURCES = 25


def dual_cross_mudd_streams(
    weights0: Tensor,
    candidate_pool0: tuple[Tensor, ...],
    source_scale0: Tensor,
    post_scale0: Tensor,
    gates0: Tensor,
    eps0: float,
    weights1: Tensor,
    candidate_pool1: tuple[Tensor, ...],
    source_scale1: Tensor,
    post_scale1: Tensor,
    gates1: Tensor,
    eps1: float,
    streams: Streams,
) -> Streams:
    """Fused decoder Cross-MUDD pair.

    Two mixes, one post-norm residual kernel.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("Dual Cross-MUDD is inference-only; disable autograd explicitly")
    raise RuntimeError("Triton dual Cross-MUDD requires CUDA and triton")


if triton is not None:

    @triton.jit
    def _cross_mudd_mul(a, b, bf16: tl.constexpr, fp16: tl.constexpr):
        value = a.to(tl.float32) * b.to(tl.float32)
        if bf16:
            value = value.to(tl.bfloat16).to(tl.float32)
        elif fp16:
            value = value.to(tl.float16).to(tl.float32)
        return value

    @triton.jit
    def _cross_mudd_scaled_fwd(
        weights,
        scale,
        s0,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        s8,
        s9,
        s10,
        s11,
        s12,
        s13,
        s14,
        s15,
        s16,
        s17,
        s18,
        s19,
        s20,
        s21,
        s22,
        s23,
        s24,
        out0,
        out1,
        out2,
        out3,
        nsrc: tl.constexpr,
        hidden: tl.constexpr,
        block: tl.constexpr,
        bf16: tl.constexpr,
        fp16: tl.constexpr,
    ):
        sources = (
            s0,
            s1,
            s2,
            s3,
            s4,
            s5,
            s6,
            s7,
            s8,
            s9,
            s10,
            s11,
            s12,
            s13,
            s14,
            s15,
            s16,
            s17,
            s18,
            s19,
            s20,
            s21,
            s22,
            s23,
            s24,
        )
        row = tl.program_id(0)
        hidden_block = tl.program_id(1)
        offsets = hidden_block * block + tl.arange(0, block)
        mask = offsets < hidden
        pointer = row * hidden + offsets
        weight_pointer = row * 4 * nsrc
        learned_scale = tl.load(scale + offsets, mask=mask, other=0)
        value0 = tl.zeros((block,), dtype=tl.float32)
        value1 = tl.zeros((block,), dtype=tl.float32)
        value2 = tl.zeros((block,), dtype=tl.float32)
        value3 = tl.zeros((block,), dtype=tl.float32)
        for source_index in tl.static_range(0, 25):
            if nsrc > source_index:
                source = tl.load(sources[source_index] + pointer, mask=mask, other=0)
                scaled = _cross_mudd_mul(source, learned_scale, bf16, fp16)
                value0 += _cross_mudd_mul(scaled, tl.load(weights + weight_pointer + source_index), bf16, fp16)
                value1 += _cross_mudd_mul(scaled, tl.load(weights + weight_pointer + nsrc + source_index), bf16, fp16)
                value2 += _cross_mudd_mul(
                    scaled, tl.load(weights + weight_pointer + 2 * nsrc + source_index), bf16, fp16
                )
                value3 += _cross_mudd_mul(
                    scaled, tl.load(weights + weight_pointer + 3 * nsrc + source_index), bf16, fp16
                )
        tl.store(out0 + pointer, value0, mask=mask)
        tl.store(out1 + pointer, value1, mask=mask)
        tl.store(out2 + pointer, value2, mask=mask)
        tl.store(out3 + pointer, value3, mask=mask)

    @triton.jit
    def _cross_mudd_dual_post_gate_residual_fwd(
        update0,
        update1,
        current,
        post_scale0,
        post_scale1,
        gate0,
        gate1,
        output,
        hidden: tl.constexpr,
        block: tl.constexpr,
        eps0: tl.constexpr,
        eps1: tl.constexpr,
        bf16: tl.constexpr,
        fp16: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < hidden
        pointer = row * hidden + offsets
        value0 = tl.load(update0 + pointer, mask=mask, other=0).to(tl.float32)
        value1 = tl.load(update1 + pointer, mask=mask, other=0).to(tl.float32)
        current_value = tl.load(current + pointer, mask=mask, other=0).to(tl.float32)
        gamma0 = tl.load(post_scale0 + offsets, mask=mask, other=0)
        gamma1 = tl.load(post_scale1 + offsets, mask=mask, other=0)
        gate_value0 = tl.load(gate0)
        gate_value1 = tl.load(gate1)
        inverse_rms0 = tl.rsqrt(tl.sum(value0 * value0, axis=0) / hidden + eps0)
        inverse_rms1 = tl.rsqrt(tl.sum(value1 * value1, axis=0) / hidden + eps1)
        normalized0 = _cross_mudd_mul(value0 * inverse_rms0, gamma0, bf16, fp16)
        normalized1 = _cross_mudd_mul(value1 * inverse_rms1, gamma1, bf16, fp16)
        gated0 = _cross_mudd_mul(normalized0, gate_value0, bf16, fp16)
        gated1 = _cross_mudd_mul(normalized1, gate_value1, bf16, fp16)
        intermediate = current_value + gated0
        if bf16:
            intermediate = intermediate.to(tl.bfloat16).to(tl.float32)
        elif fp16:
            intermediate = intermediate.to(tl.float16).to(tl.float32)
        tl.store(output + pointer, intermediate + gated1, mask=mask)

    def dual_cross_mudd_streams(  # type: ignore[no-redef]
        weights0: Tensor,
        candidate_pool0: tuple[Tensor, ...],
        source_scale0: Tensor,
        post_scale0: Tensor,
        gates0: Tensor,
        eps0: float,
        weights1: Tensor,
        candidate_pool1: tuple[Tensor, ...],
        source_scale1: Tensor,
        post_scale1: Tensor,
        gates1: Tensor,
        eps1: float,
        streams: Streams,
    ) -> Streams:
        if torch.is_grad_enabled():
            raise RuntimeError("Dual Cross-MUDD is inference-only; disable autograd explicitly")
        if not weights0.is_cuda:
            raise RuntimeError("Triton dual Cross-MUDD requires CUDA and triton")
        if not 2 <= len(candidate_pool0) <= _MAX_CROSS_MUDD_SOURCES:
            raise ValueError("unsupported first Cross-MUDD source count")
        if not 2 <= len(candidate_pool1) <= _MAX_CROSS_MUDD_SOURCES:
            raise ValueError("unsupported second Cross-MUDD source count")
        shape = candidate_pool0[0].shape
        if candidate_pool1[0].shape != shape:
            raise ValueError("dual Cross-MUDD candidate banks have different shapes")
        hidden = shape[-1]

        def prepare_sources(candidate_pool: tuple[Tensor, ...], eps: float) -> tuple[Tensor, ...]:
            return tuple(
                (source * torch.rsqrt(source.float().square().mean(dim=-1, keepdim=True) + eps).to(source.dtype))
                .reshape(-1, hidden)
                .contiguous()
                for source in candidate_pool
            )

        unit_sources0 = prepare_sources(candidate_pool0, eps0)
        unit_sources1 = prepare_sources(candidate_pool1, eps1)
        nsrc0 = len(unit_sources0)
        nsrc1 = len(unit_sources1)
        weight_matrix0 = weights0.reshape(-1, 4, nsrc0).contiguous()
        weight_matrix1 = weights1.reshape(-1, 4, nsrc1).contiguous()
        source_pointers0 = unit_sources0 + (unit_sources0[0],) * (_MAX_CROSS_MUDD_SOURCES - nsrc0)
        source_pointers1 = unit_sources1 + (unit_sources1[0],) * (_MAX_CROSS_MUDD_SOURCES - nsrc1)
        mixed0 = tuple(torch.empty_like(unit_sources0[0]) for _ in range(4))
        mixed1 = tuple(torch.empty_like(unit_sources1[0]) for _ in range(4))
        mix_block = min(256, triton.next_power_of_2(hidden))
        launch = (weight_matrix0.shape[0], triton.cdiv(hidden, mix_block))
        _cross_mudd_scaled_fwd[launch](
            weight_matrix0,
            source_scale0.contiguous(),
            *source_pointers0,
            *mixed0,
            nsrc=nsrc0,
            hidden=hidden,
            block=mix_block,
            bf16=weights0.dtype == torch.bfloat16,
            fp16=weights0.dtype == torch.float16,
            num_warps=4,
        )
        _cross_mudd_scaled_fwd[launch](
            weight_matrix1,
            source_scale1.contiguous(),
            *source_pointers1,
            *mixed1,
            nsrc=nsrc1,
            hidden=hidden,
            block=mix_block,
            bf16=weights1.dtype == torch.bfloat16,
            fp16=weights1.dtype == torch.float16,
            num_warps=4,
        )
        stream_matrices = tuple(stream.reshape(-1, hidden).contiguous() for stream in streams)
        outputs = tuple(torch.empty_like(unit_sources0[0]) for _ in range(4))
        post_block = triton.next_power_of_2(hidden)
        for role in range(4):
            _cross_mudd_dual_post_gate_residual_fwd[(weight_matrix0.shape[0],)](
                mixed0[role],
                mixed1[role],
                stream_matrices[role],
                post_scale0[role].contiguous(),
                post_scale1[role].contiguous(),
                gates0[role].contiguous(),
                gates1[role].contiguous(),
                outputs[role],
                hidden=hidden,
                block=post_block,
                eps0=eps0,
                eps1=eps1,
                bf16=weights0.dtype == torch.bfloat16,
                fp16=weights0.dtype == torch.float16,
                num_warps=8,
            )
        return cast(Streams, tuple(output.reshape(shape) for output in outputs))


__all__ = ["dual_cross_mudd_streams"]
