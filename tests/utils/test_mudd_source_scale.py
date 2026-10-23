"""Check the tiled Cross-MUDD scale backward against ordinary multiplication."""
import math

import pytest
import torch

from xtuner.v1.ops.mudd.source_scale import _ScaleSource, scale_source


class TestMuddSourceScale:
    @pytest.fixture(autouse=True)
    def enable_source_scale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NCP_Q35V2_TRITON_SOURCE_SCALE", "1")

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("enabled", [None, "0", "1"])
    def test_switch_controls_custom_backward(self, monkeypatch: pytest.MonkeyPatch, enabled: str | None) -> None:
        if enabled is None:
            monkeypatch.delenv("NCP_Q35V2_TRITON_SOURCE_SCALE")
        else:
            monkeypatch.setenv("NCP_Q35V2_TRITON_SOURCE_SCALE", enabled)
        x = torch.randn(1, 17, 65, device="cuda", requires_grad=True)
        w = torch.randn(65, device="cuda", requires_grad=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            scale_source(x, w).sum().backward()
        keys = {event.key for event in profiler.key_averages()}
        assert ("xtuner::mudd_scale_weight_gradient" in keys) == (enabled == "1")

    @pytest.mark.parametrize("shape", [(65,), (17, 65), (2, 17, 65)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_mixed_dtype_backward(self, shape: tuple[int, ...], dtype: torch.dtype, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA required")
        torch.manual_seed(23)
        x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        w = torch.randn(shape[-1], device=device, dtype=torch.float32, requires_grad=True)
        # Exercise the CUDA autograd implementation on CPU too; mixed dtypes
        # take its ordinary tensor fallback and do not launch the Triton kernel.
        fn = _ScaleSource.apply if device == "cpu" else scale_source
        actual = fn(x, w)
        expected = x * w
        dy = torch.randn_like(actual)
        observed = torch.autograd.grad(actual, (x, w), dy)
        reference = torch.autograd.grad(expected, (x, w), dy)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for a, b in zip(observed, reference, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("shape", [(17, 65), (2048, 4096), (8192, 4096)])
    def test_shared_scale_compiled(self, dtype: torch.dtype, shape: tuple[int, int]) -> None:
        torch.manual_seed(17)
        x = torch.randn(1, *shape, device="cuda", dtype=dtype, requires_grad=True)
        w = torch.randn(shape[-1], device="cuda", dtype=dtype, requires_grad=True)
        def forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return scale_source(a, b + 1) + scale_source(a * 0.75, b + 1)
        fn = torch.compile(forward, fullgraph=True)
        actual = fn(x, w)
        expected = x * (w + 1) + (x * 0.75) * (w + 1)
        dy = torch.randn_like(actual) / math.sqrt(shape[0])
        ga = torch.autograd.grad(actual, (x, w), dy)
        ge = torch.autograd.grad(expected, (x, w), dy)
        tol = 0.02 if dtype == torch.bfloat16 else 0.003 if dtype == torch.float16 else 2e-5
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
        for a, b in zip(ga, ge, strict=True):
            torch.testing.assert_close(a, b, atol=tol, rtol=tol)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("mode", ["input", "weight", "both"])
    def test_noncontiguous_and_frozen_inputs(self, mode: str) -> None:
        x = torch.randn(2, 19, 130, device="cuda")[..., ::2].detach().requires_grad_(mode != "weight")
        w = torch.randn(65, device="cuda", requires_grad=mode != "input")
        inputs = [v for v in (x, w) if v.requires_grad]
        dy = torch.randn_like(x)
        a = torch.autograd.grad(scale_source(x, w), inputs, dy)
        b = torch.autograd.grad(x * w, inputs, dy)
        for actual, expected in zip(a, b, strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
