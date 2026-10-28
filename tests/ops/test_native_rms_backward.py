"""Regression coverage for the native ConceptLM RMS backward boundary."""

import pytest
import torch
import torch.nn.functional as F

from xtuner.v1.ops.rms_norm.native_backward import rms_norm_native_backward


class TestNativeRMSBackward:
    @pytest.fixture(autouse=True)
    def enable_native_backward(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NCP_Q35V2_NATIVE_RMS_BACKWARD", "1")

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("enabled", [None, "0", "1"])
    def test_switch_controls_custom_backward(self, monkeypatch: pytest.MonkeyPatch, enabled: str | None) -> None:
        if enabled is None:
            monkeypatch.delenv("NCP_Q35V2_NATIVE_RMS_BACKWARD")
        else:
            monkeypatch.setenv("NCP_Q35V2_NATIVE_RMS_BACKWARD", enabled)
        x = torch.randn(2, 17, 64, device="cuda", requires_grad=True)
        w = torch.randn(64, device="cuda", requires_grad=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            rms_norm_native_backward(x, w, 1e-6).sum().backward()
        keys = {event.key for event in profiler.key_averages()}
        assert ("xtuner::rms_norm_native_backward" in keys) == (enabled == "1")

    def test_cpu_gradcheck(self) -> None:
        x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
        w = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda a, b: rms_norm_native_backward(a, b, 1e-6), (x, w))

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("mode", ["both", "input", "weight", "unweighted"])
    @pytest.mark.parametrize("compiled", [False, True])
    def test_output_and_gradients(self, dtype: torch.dtype, mode: str, compiled: bool) -> None:
        torch.manual_seed(7)
        # Exercise non-contiguous inputs as well as zero-centered effective weights.
        x = torch.randn(2, 17, 128, device="cuda", dtype=dtype)[..., ::2].detach()
        x.requires_grad_(mode != "weight")
        w = None if mode == "unweighted" else torch.randn(64, device="cuda", dtype=dtype)
        if w is not None:
            w.requires_grad_(mode != "input")
        fn = torch.compile(rms_norm_native_backward, fullgraph=True) if compiled else rms_norm_native_backward
        scale = None if w is None else w + 1.0
        result = fn(x, scale, 1e-6)
        reference = F.rms_norm(x, (64,), scale, 1e-6)
        grad = torch.randn_like(result)
        inputs = [t for t in (x, w) if t is not None and t.requires_grad]
        actual = torch.autograd.grad(result, inputs, grad, retain_graph=True)
        expected = torch.autograd.grad(reference, inputs, grad)
        tolerance = 0.02 if dtype == torch.bfloat16 else 2e-5
        torch.testing.assert_close(result, reference, atol=tolerance, rtol=tolerance)
        for observed, target in zip(actual, expected, strict=True):
            torch.testing.assert_close(observed, target, atol=tolerance, rtol=tolerance)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mixed_dtype_uses_torch_fallback(self, dtype: torch.dtype) -> None:
        torch.manual_seed(17)
        x = torch.randn(2, 7, 64, device="cuda", dtype=dtype, requires_grad=True)
        weight = torch.randn(64, device="cuda", dtype=torch.float32, requires_grad=True)
        result = rms_norm_native_backward(x, weight, 1e-6)
        reference = F.rms_norm(x, (64,), weight, 1e-6)
        assert result.dtype == dtype
        torch.testing.assert_close(result, reference, atol=0, rtol=0)
        grad = torch.randn_like(result)
        actual = torch.autograd.grad(result, (x, weight), grad, retain_graph=True)
        expected = torch.autograd.grad(reference, (x, weight), grad)
        for observed, target in zip(actual, expected, strict=True):
            torch.testing.assert_close(observed, target, atol=0, rtol=0)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_compiled_effective_weight_keeps_low_precision_rounding(self, dtype: torch.dtype) -> None:
        from xtuner.v1.module.rms_norm.concept_qwen35 import effective_weight

        weight = torch.linspace(-0.9, 0.9, 65, device="cuda", dtype=dtype, requires_grad=True)
        # Promotion after addition must not remove its low-precision rounding.
        fn = torch.compile(lambda w: effective_weight(w).float(), fullgraph=True)
        result = fn(weight)
        reference = (weight + 1.0).float()
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(result.sum(), weight, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(reference.sum(), weight)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
