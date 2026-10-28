"""GDN fused norm-gate forward rounding and backward numerical coverage."""

import pytest
import torch

from xtuner.v1.module.attention.concept_qwen35 import _fused_zero_rms_norm_gate


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the CUDA FLA fused norm-gate kernel")
def test_v2_fused_zero_rms_norm_gate_supports_forward_and_backward() -> None:
    pytest.importorskip("fla")
    torch.manual_seed(0)
    eps = 1.0e-6
    x = torch.randn(64, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    weight = torch.randn(16, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    output = _fused_zero_rms_norm_gate(x, gate, weight, eps)
    normalized = (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * (weight.float() + 1.0)
    ).to(x.dtype)
    reference = (normalized.float() * torch.nn.functional.silu(gate.float())).to(x.dtype)
    torch.testing.assert_close(output, reference, rtol=3.0e-2, atol=3.0e-2)

    output.float().square().mean().backward()
    for tensor in (x, gate, weight):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton normalization")
class TestGDNZeroCenteredNormGate:
    @pytest.mark.parametrize("rows", [8, 65])
    def test_preserves_te_weight_and_bf16_rounding(self, rows: int) -> None:
        from xtuner.v1.module.attention.concept_qwen35 import _fused_zero_rms_norm_gate
        from xtuner.v1.ops.rms_norm.concept_qwen35 import zero_rmsnorm_cuda

        torch.manual_seed(42)
        x = torch.randn(1, rows, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        gate = torch.randn_like(x, requires_grad=True)
        weight = (torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.01).requires_grad_()
        actual = _fused_zero_rms_norm_gate(x, gate, weight, 1e-6)
        normalized = zero_rmsnorm_cuda(x.detach(), weight.detach(), 1e-6)
        expected = (normalized.float() * torch.nn.functional.silu(gate.float())).to(x.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        xf, gf, wf = (tensor.detach().float().requires_grad_() for tensor in (x, gate, weight))
        normalized_reference = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
        normalized_reference = (normalized_reference * (wf + 1.0)).to(x.dtype).float()
        reference = (normalized_reference * torch.nn.functional.silu(gf)).to(x.dtype)
        upstream = torch.randn_like(x)
        actual_grads = torch.autograd.grad(actual, (x, gate, weight), upstream)
        expected_grads = torch.autograd.grad(reference, (xf, gf, wf), upstream)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad.float(), expected_grad, rtol=0.02, atol=0.02)
