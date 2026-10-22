"""Compare fused chunk weight accumulation against the original chunk algorithm."""
from functools import partial
from typing import Any

import pytest
import torch

from xtuner.v1.loss.ce_loss import CELossConfig, CELossKwargs, LMHeadLossContext
from xtuner.v1.loss.chunk_loss import ChunkLoss


class TestChunkGradAccumulation:
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("case", ["single", "tail_padded", "ignored", "frozen"])
    def test_loss_and_gradients(self, dtype: torch.dtype, case: str) -> None:
        self._compare(dtype, case, "cuda")

    def test_cpu_fallback(self) -> None:
        self._compare(torch.float32, "tail_padded", "cpu")

    @staticmethod
    def _compare(
        dtype: torch.dtype, case: str, device: str, context_cls: type[LMHeadLossContext] = LMHeadLossContext
    ) -> None:
        torch.manual_seed(19)
        length = 5 if case == "single" else 19
        x = (torch.randn(2, length, 64, device=device, dtype=dtype) * 0.1).requires_grad_()
        w = (torch.randn(272, 64, device=device, dtype=dtype) * 0.1).requires_grad_(case != "frozen")
        labels = torch.randint(0, 257, (2, length), device=device)
        labels[:, -2:] = -100
        if case == "ignored":
            labels.fill_(-100)
        kwargs = CELossKwargs(
            shifted_labels=labels, loss_weight=torch.full_like(labels, 1 / (2 * length), dtype=torch.float32)
        )
        config = CELossConfig(mode="chunk", chunk_size=8)
        ctx = context_cls(config, kwargs)
        loss_fn = partial(ctx.loss_fn, valid_vocab_size=257)
        reference, _ = ChunkLoss.apply(x, w, None, loss_fn, kwargs.chunk(8), 8)
        inputs = (x, w) if w.requires_grad else (x,)
        expected = torch.autograd.grad(reference * 0.37, inputs)
        actual, _ = ctx.chunk_mode(x, w, None, kwargs, valid_vocab_size=257)
        result = torch.autograd.grad(actual * 0.37, inputs)
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
        tolerance = 2e-5 if dtype == torch.float32 else 2e-3 if dtype == torch.float16 else 2e-2
        for a, b in zip(result, expected, strict=True):
            torch.testing.assert_close(a, b, atol=tolerance * 0.01, rtol=tolerance)
        if w.requires_grad:
            assert torch.count_nonzero(result[1][257:]) == 0

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_uses_addmm_after_first_chunk(self) -> None:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            self._compare(torch.bfloat16, "tail_padded", "cuda")
        counts = {event.key: event.count for event in profiler.key_averages()}
        assert counts["aten::addmm_"] == 2


    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_loss_subclass_keeps_original_contract(self) -> None:
        class LegacyContext(LMHeadLossContext):
            def loss_fn(
                self,
                hidden_states: torch.Tensor,
                head_weight: torch.Tensor,
                head_bias: torch.Tensor | None,
                loss_kwargs: CELossKwargs,
                valid_vocab_size: int | None = None,
            ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
                return super().loss_fn(hidden_states, head_weight, head_bias, loss_kwargs, valid_vocab_size)

        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            self._compare(torch.bfloat16, "tail_padded", "cuda", LegacyContext)
        assert "aten::addmm_" not in {event.key for event in profiler.key_averages()}
