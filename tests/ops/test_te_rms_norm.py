"""Numerical coverage for every TE-equivalent RMSNorm reduction branch."""

import pytest
import torch

from xtuner.v1.ops.rms_norm.concept_qwen35 import zero_rmsnorm_cuda


class TestTEEquivalentRMSNorm:
    def test_autograd_is_rejected(self) -> None:
        x = torch.randn(2, 128, requires_grad=True)
        with pytest.raises(RuntimeError, match="forward-only"):
            zero_rmsnorm_cuda(x, torch.zeros(128), 1e-6)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")
    @pytest.mark.parametrize("width", [128, 512, 1536, 4096])
    @pytest.mark.parametrize("rows", [1, 4, 5, 8])
    @pytest.mark.parametrize("eps", [1e-6, 1e-4])
    @torch.no_grad()
    def test_matches_high_precision_reference(self, width: int, rows: int, eps: float) -> None:
        torch.manual_seed(37)
        x = torch.randn(rows, width * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
        weight = (torch.randn(width * 2, device="cuda", dtype=torch.bfloat16) * 0.1)[::2]
        assert not x.is_contiguous() and not weight.is_contiguous()
        output = zero_rmsnorm_cuda(x, weight, eps)
        xf = x.double()
        reference = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * (weight.double() + 1.0)
        assert output.shape == x.shape and output.dtype == x.dtype
        assert torch.isfinite(output).all()
        # TE-style FP32 reductions and final BF16 rounding need not be bitwise
        # equal to FP64; use the same BF16 tolerance as the fused norm tests.
        torch.testing.assert_close(output, reference.to(x.dtype), rtol=0.02, atol=0.02)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")
    @pytest.mark.parametrize("width,dtype", [(64, torch.bfloat16), (128, torch.float32)])
    @torch.no_grad()
    def test_unsupported_inputs_are_rejected(self, width: int, dtype: torch.dtype) -> None:
        with pytest.raises(ValueError, match="CUDA BF16"):
            zero_rmsnorm_cuda(
                torch.randn(4, width, device="cuda", dtype=dtype), torch.zeros(width, device="cuda"), 1e-6
            )
