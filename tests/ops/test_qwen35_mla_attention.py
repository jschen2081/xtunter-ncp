"""MLA FlashAttention output and gradient comparison against SDPA."""

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext


def _context(lengths: list[int]) -> SequenceContext:
    total = sum(lengths)
    boundaries = [0]
    positions = []
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
        positions.append(torch.arange(length))
    cu = torch.tensor(boundaries, dtype=torch.int32)
    return SequenceContext(
        input_ids=torch.randint(0, 64, (1, total)),
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=max(lengths),
        max_length_k=max(lengths),
        num_padding=0,
        sequence_parallel_mesh=None,
        device=torch.device("cpu"),
        position_ids=torch.cat(positions).unsqueeze(0),
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashAttention")
class TestMLAFlashAttention:
    @pytest.mark.parametrize("lengths", [[8], [5, 3]])
    @pytest.mark.parametrize("qk_width,value_width", [(12, 8), (8, 16)])
    def test_matches_sdpa_output_and_gradients(self, lengths: list[int], qk_width: int, value_width: int) -> None:
        from xtuner.v1.module.attention.concept_qwen35 import _causal_attention

        torch.manual_seed(1)
        total = sum(lengths)
        tensors = tuple(
            torch.randn(1, total, 2, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for width in (qk_width, qk_width, value_width)
        )
        reference = tuple(tensor.detach().float().requires_grad_() for tensor in tensors)
        doc_ids = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths)).cuda()
        mask = (doc_ids[:, None] == doc_ids[None, :]) & torch.ones(
            total, total, device="cuda", dtype=torch.bool
        ).tril()
        scale = qk_width**-0.5
        expected = torch.nn.functional.scaled_dot_product_attention(
            *(tensor.transpose(1, 2) for tensor in reference), attn_mask=mask, scale=scale
        ).transpose(1, 2)
        actual = _causal_attention(*tensors, _context(lengths).to("cuda"), scale)
        torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, tensors, upstream)
        expected_grads = torch.autograd.grad(expected, reference, upstream.float())
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad.float(), expected_grad, rtol=0.03, atol=0.02)
