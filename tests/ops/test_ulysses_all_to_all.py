"""Direct collective regression for movedim contiguity and its backward."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all


def _check_all_to_all(rank: int, rendezvous: str, device: str, compiled: bool) -> None:
    if device == "cuda":
        torch.cuda.set_device(rank)
    target = torch.device(device, rank) if device == "cuda" else torch.device("cpu")
    dist.init_process_group(
        "nccl" if device == "cuda" else "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        mesh = init_device_mesh(device, (2,))
        for scatter, gather in ((0, 1), (1, 2), (2, 1)):
            x = torch.arange(4 * 6 * 16, device=target, dtype=torch.float32).reshape(4, 6, 16) + rank * 1000
            x = x[..., ::2].detach().requires_grad_()
            assert not x.is_contiguous()

            def exchange(value: torch.Tensor) -> torch.Tensor:
                return ulysses_all_to_all(value, scatter, gather, mesh)

            function = (
                torch.compile(exchange, fullgraph=True, backend="inductor" if device == "cuda" else "aot_eager")
                if compiled
                else exchange
            )
            output = function(x)
            peers = [torch.empty_like(x) for _ in range(2)]
            dist.all_gather(peers, x.detach().contiguous())
            expected = torch.cat([peer.chunk(2, dim=scatter)[rank] for peer in peers], dim=gather)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

            upstream = torch.arange(output.numel(), device=target, dtype=x.dtype).reshape(output.shape) + rank * 100
            (gradient,) = torch.autograd.grad(output, x, upstream)
            peer_gradients = [torch.empty_like(upstream) for _ in range(2)]
            dist.all_gather(peer_gradients, upstream)
            expected_gradient = torch.cat([grad.chunk(2, dim=gather)[rank] for grad in peer_gradients], dim=scatter)
            torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class TestUlyssesAllToAll:
    @pytest.mark.parametrize("compiled", [False, True])
    def test_cpu_noncontiguous_forward_backward(self, tmp_path: Path, compiled: bool) -> None:
        mp.spawn(_check_all_to_all, args=((tmp_path / "rendezvous").as_uri(), "cpu", compiled), nprocs=2, join=True)

    @pytest.mark.gpu
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
    @pytest.mark.parametrize("compiled", [False, True])
    def test_cuda_noncontiguous_forward_backward(self, tmp_path: Path, compiled: bool) -> None:
        mp.spawn(_check_all_to_all, args=((tmp_path / "rendezvous").as_uri(), "cuda", compiled), nprocs=2, join=True)
