"""NCP v2 HF convolution selects FLA without changing production dispatch."""

import pytest
import torch

from xtuner.v1.ops import gated_deltanet as ops


@pytest.mark.parametrize("hf_mode", ["true", "false"])
def test_fla_backend_only_changes_hf_dispatch(monkeypatch: pytest.MonkeyPatch, hf_mode: str) -> None:
    monkeypatch.setenv("XTUNER_HF_IMPL", hf_mode)
    if hf_mode == "true":
        assert ops.get_causal_conv1d_fn(hf_backend="fla") is ops._fla_causal_conv1d_adapter
        assert ops.get_causal_conv1d_fn() is ops._hf_causal_conv1d_adapter
    else:
        assert ops.get_causal_conv1d_fn(hf_backend="fla") is ops.get_causal_conv1d_fn()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FLA convolution")
@pytest.mark.parametrize("batch_size", [1, 2])
def test_packed_documents_do_not_share_forward_or_backward_state(batch_size: int) -> None:
    torch.manual_seed(123)
    x = torch.randn(batch_size, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ids = torch.tensor([0] * 8 + [1] * 8, device="cuda", dtype=torch.int32).expand(batch_size, -1)
    output = ops._fla_causal_conv1d_adapter(x, weight, None, "silu", ids)
    altered = x.detach().clone()
    altered[:, 8:] += 10
    changed = ops._fla_causal_conv1d_adapter(altered, weight, None, "silu", ids)
    torch.testing.assert_close(output[:, :8], changed[:, :8], rtol=0, atol=0)
    dx, dw = torch.autograd.grad(output[:, 8:].float().sum(), (x, weight))
    assert torch.count_nonzero(dx[:, :8]) == 0
    assert torch.isfinite(dx).all() and torch.isfinite(dw).all()
    # Each document also agrees with a separate FLA call, without packed offsets.
    from fla.modules.convolution import causal_conv1d

    for batch in range(batch_size):
        for start in (0, 8):
            expected, _ = causal_conv1d(x[batch : batch + 1, start : start + 8], weight, activation="silu")
            torch.testing.assert_close(output[batch : batch + 1, start : start + 8], expected, rtol=0, atol=0)
