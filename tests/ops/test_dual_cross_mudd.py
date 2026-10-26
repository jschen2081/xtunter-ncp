"""Dual Cross-MUDD inference agrees with independent eager route arithmetic."""

import pytest
import torch
from torch import Tensor

from xtuner.v1.ops.mudd.concept_qwen35 import dual_cross_mudd_streams


def _route_reference(
    weights: Tensor,
    bank: tuple[Tensor, ...],
    source_scale: Tensor,
    post_scale: Tensor,
    gates: Tensor,
    eps: float,
) -> tuple[Tensor, ...]:
    units = tuple(
        source * torch.rsqrt(source.float().square().mean(-1, keepdim=True) + eps).to(source.dtype) for source in bank
    )
    updates = []
    for role in range(4):
        # Match CrossMUDDRoute's ordinary weighted sum, including activation
        # dtype rounding, rather than reproducing the fused kernel's reduction.
        mixed = sum((unit * source_scale) * weights[..., role, index, None] for index, unit in enumerate(units))
        normalized = mixed.float() * torch.rsqrt(mixed.float().square().mean(-1, keepdim=True) + eps)
        normalized = (normalized * post_scale[role].float()).to(mixed.dtype)
        updates.append(normalized * gates[role])
    return tuple(updates)


class TestDualCrossMudd:
    def test_autograd_is_rejected(self) -> None:
        tensor = torch.ones(1, requires_grad=True)
        with pytest.raises(RuntimeError, match="inference-only"):
            dual_cross_mudd_streams(
                tensor,
                (tensor, tensor),
                tensor,
                tensor,
                tensor,
                1e-6,
                tensor,
                (tensor, tensor),
                tensor,
                tensor,
                tensor,
                1e-6,
                (tensor,) * 4,
            )

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("counts,width", [((2, 3), 65), ((3, 5), 128), ((25, 25), 512)])
    @torch.no_grad()
    def test_forward_matches_eager(self, dtype: torch.dtype, counts: tuple[int, int], width: int) -> None:
        torch.manual_seed(41)

        def tensor(*shape: int) -> Tensor:
            return torch.randn(*shape, device="cuda", dtype=dtype)

        streams = tuple(tensor(1, 7, width * 2)[..., ::2] for _ in range(4))
        banks = [tuple(tensor(1, 7, width * 2)[..., ::2] for _ in range(count)) for count in counts]
        weights = [tensor(1, 7, 4, count).softmax(-1) for count in counts]
        scales = [tensor(width) * 0.1 + 1 for _ in range(2)]
        post = [tensor(4, width) * 0.1 + 1 for _ in range(2)]
        gates = [tensor(4).sigmoid() for _ in range(2)]
        eps = (1e-6, 1e-4)
        actual = dual_cross_mudd_streams(
            weights[0],
            banks[0],
            scales[0],
            post[0],
            gates[0],
            eps[0],
            weights[1],
            banks[1],
            scales[1],
            post[1],
            gates[1],
            eps[1],
            streams,
        )
        first = _route_reference(weights[0], banks[0], scales[0], post[0], gates[0], eps[0])
        second = _route_reference(weights[1], banks[1], scales[1], post[1], gates[1], eps[1])
        tolerance = 0.03 if dtype == torch.bfloat16 else 0.003 if dtype == torch.float16 else 2e-5
        for output, current, a, b in zip(actual, streams, first, second, strict=True):
            expected = (current + a) + b
            assert output.shape == current.shape and output.dtype == dtype
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")
    @pytest.mark.parametrize("invalid", ["first_count", "second_count", "shape"])
    @torch.no_grad()
    def test_invalid_banks_are_rejected(self, invalid: str) -> None:
        x = torch.zeros(1, 2, 64, device="cuda")
        first, second = (x, x), (x, x)
        if invalid == "first_count":
            first = (x,)
        elif invalid == "second_count":
            second = (x,) * 26
        else:
            second = (x[..., :32],) * 2
        message = "different shapes" if invalid == "shape" else "source count"
        with pytest.raises(ValueError, match=message):
            dual_cross_mudd_streams(x, first, x, x, x, 1e-6, x, second, x, x, x, 1e-6, (x,) * 4)
