"""Packed document boundaries must preserve deterministic convolution gradients."""

import pytest
import torch

from xtuner.v1.module.attention.concept_qwen35 import _cpu_conv_documents


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [64, 256])
@pytest.mark.parametrize("with_bias", [False, True])
def test_packed_document_gradients(activation, batch_size, channels, with_bias):
    from xtuner.v1.ops.gated_deltanet.causal_conv1d import causal_conv1d_fn

    old = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(123)
        lengths = [64, 127, 129]
        x = torch.randn(batch_size, sum(lengths), channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(channels, 4, device="cuda", dtype=torch.float32, requires_grad=True)
        bias = torch.randn(channels, device="cuda", dtype=torch.float32, requires_grad=True) if with_bias else None
        params = (x, weight, bias) if with_bias else (x, weight)
        dout = torch.randn_like(x)
        ids = (
            torch.repeat_interleave(
                torch.arange(3, device="cuda", dtype=torch.int32), torch.tensor(lengths, device="cuda")
            )
            .expand(batch_size, -1)
            .contiguous()
        )
        out = causal_conv1d_fn(x, weight, bias=bias, seq_idx=ids, activation=activation)
        grads = torch.autograd.grad(out, params, dout)
        dx, dw = grads[:2]

        expected_dw = torch.zeros_like(weight)
        expected_db = torch.zeros_like(bias) if with_bias else None
        expected_out, expected_dx = torch.empty_like(out), torch.empty_like(x)
        for b in range(batch_size):
            start = 0
            for length in lengths:
                end = start + length
                doc = x[b : b + 1, start:end].detach().contiguous().requires_grad_()
                doc_out = causal_conv1d_fn(
                    doc,
                    weight,
                    bias=bias,
                    seq_idx=torch.zeros((1, length), dtype=torch.int32, device="cuda"),
                    activation=activation,
                )
                doc_params = (doc, weight, bias) if with_bias else (doc, weight)
                doc_grads = torch.autograd.grad(doc_out, doc_params, dout[b : b + 1, start:end].contiguous())
                expected_dw.add_(doc_grads[1])
                if with_bias:
                    expected_db.add_(doc_grads[2])
                expected_out[b : b + 1, start:end] = doc_out.detach()
                expected_dx[b : b + 1, start:end] = doc_grads[0]
                start = end
        assert dw.dtype == torch.float32
        torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
        torch.testing.assert_close(dx, expected_dx, rtol=0, atol=0)
        torch.testing.assert_close(dw, expected_dw, rtol=0, atol=0)
        if with_bias:
            assert grads[2].dtype == torch.float32
            torch.testing.assert_close(grads[2], expected_db, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warn_only)


@pytest.mark.parametrize("lengths", [[16], [3, 13]])
def test_gdn_cpu_convolution_preserves_sp_head_shards(lengths: list[int]) -> None:
    torch.manual_seed(11)
    qkv = torch.randn(1, sum(lengths), 12)
    weight = torch.randn(12, 4)
    boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    # One rank's channels from each of the Q, K and V projection blocks.
    channels = torch.tensor([0, 1, 4, 5, 8, 9])
    expected = _cpu_conv_documents(qkv, weight, boundaries).index_select(2, channels)
    actual = _cpu_conv_documents(qkv.index_select(2, channels), weight.index_select(0, channels), boundaries)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
