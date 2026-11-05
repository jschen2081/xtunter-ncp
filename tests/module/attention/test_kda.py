"""CPU checks of the real FLA call boundary without launching GPU kernels."""

from types import SimpleNamespace

import pytest
import torch

from xtuner.v1.module.attention import kda


pytestmark = pytest.mark.skipif(not kda.has_fla_kda, reason="FLA KDA dependencies unavailable")


@pytest.mark.parametrize("sp", [False, True])
def test_sigmoid_beta_reaches_fla_with_gradients(monkeypatch, sp):
    from fla.ops.kda.chunk import ChunkKDAFunction

    torch.manual_seed(0)
    attention = kda.KDAConfig(num_heads=2, head_dim=4).build(hidden_size=8)
    with torch.no_grad():
        attention.b_proj.weight.zero_()
        attention.b_proj.weight[:, 0] = torch.tensor([-2.0, 2.0])

    # Convolution and gated norm kernels need CUDA. Keep the projections and
    # actual FLA Python entrypoint, intercepting only the GPU computations.
    monkeypatch.setattr(kda.KDAShortConvolution, "forward", lambda self, x, **kwargs: (x, None))
    monkeypatch.setattr(attention.o_norm, "forward", lambda x, g: x * g.sigmoid())
    monkeypatch.setattr(kda, "ulysses_all_to_all", lambda x, **kwargs: x)
    real_chunk_kda = kda.chunk_kda

    def checked_chunk_kda(**kwargs):
        assert "use_beta_sigmoid_in_kernel" not in kwargs
        assert kwargs["state_v_first"] is True
        return real_chunk_kda(**kwargs)

    monkeypatch.setattr(kda, "chunk_kda", checked_chunk_kda)
    seen = {}

    def kernel_boundary(*args):
        seen["beta"] = args[4]
        seen["cu_seqlens"] = args[12]
        return args[2] * args[4].unsqueeze(-1), None

    monkeypatch.setattr(ChunkKDAFunction, "apply", kernel_boundary)
    hidden_states = torch.ones(1, 4, 8, requires_grad=True)
    boundaries = torch.tensor([0, 2, 4], dtype=torch.int32)
    context = SimpleNamespace(
        cu_seq_lens_q=boundaries,
        sequence_parallel_mesh=SimpleNamespace(size=lambda: 1) if sp else None,
        sp_rank=0,
    )
    # A size-one mesh exercises every SP call site without collectives.
    forward = attention.forward_for_sp if sp else attention.forward
    output = forward(hidden_states, context)["projected_output"]
    expected = torch.tensor([-2.0, 2.0]).sigmoid().view(1, 1, 2).expand(1, 4, 2)
    torch.testing.assert_close(seen["beta"], expected, rtol=0, atol=0)
    assert seen["cu_seqlens"] is boundaries
    output.square().sum().backward()
    assert torch.isfinite(hidden_states.grad).all()
    assert torch.isfinite(attention.b_proj.weight.grad).all()
    assert attention.b_proj.weight.grad.count_nonzero() > 0
