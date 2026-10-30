import pytest
import torch

from xtuner.v1.loss.chunk_loss import ChunkLoss


def _loss_forward(hidden_states, head_weight, head_bias, scale):
    assert head_bias is None
    logits = hidden_states @ head_weight.T
    loss = logits.square().sum() * scale
    return loss, (None, {"logits_max": logits.detach().max()})


def _reference_loss(hidden_states, head_weight, scales, chunk_size):
    hidden_states_chunks = torch.split(hidden_states, chunk_size, dim=1)
    return sum(
        _loss_forward(hidden_states_chunk, head_weight, None, scale)[0]
        for hidden_states_chunk, scale in zip(hidden_states_chunks, scales, strict=True)
    )


@pytest.mark.parametrize("upstream_gradient", [1.0, 0.37])
def test_chunk_loss_matches_reference_with_trainable_head(upstream_gradient):
    torch.manual_seed(0)
    chunk_size = 3
    scales = [0.5, 1.25, 0.75]
    hidden_states = torch.randn(2, 7, 4, dtype=torch.float32)
    head_weight = torch.randn(6, 4, dtype=torch.float32)

    reference_hidden_states = hidden_states.clone().requires_grad_()
    reference_head_weight = head_weight.clone().requires_grad_()
    reference_loss = _reference_loss(reference_hidden_states, reference_head_weight, scales, chunk_size)
    reference_loss.backward(gradient=torch.tensor(upstream_gradient, dtype=torch.float32))

    chunk_hidden_states = hidden_states.clone().requires_grad_()
    chunk_head_weight = head_weight.clone().requires_grad_()
    chunk_loss, _ = ChunkLoss.apply(
        chunk_hidden_states,
        chunk_head_weight,
        None,
        _loss_forward,
        scales,
        chunk_size,
    )
    chunk_loss.backward(gradient=torch.tensor(upstream_gradient, dtype=torch.float32))

    torch.testing.assert_close(chunk_loss, reference_loss)
    torch.testing.assert_close(chunk_hidden_states.grad, reference_hidden_states.grad)
    torch.testing.assert_close(chunk_head_weight.grad, reference_head_weight.grad)
