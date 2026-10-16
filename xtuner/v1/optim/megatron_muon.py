"""Megatron Muon update semantics on xTuner's native DTensor optimizer.

Uses fresh optimizer state; cross-topology checkpoint migration is unsupported.
"""
import math

import torch
import torch.nn.functional as F
from torch.optim._functional import adamw

from .muon import Muon


def local_row_range(param, start, end):
    from torch.distributed.tensor import DTensor, Replicate, Shard

    if not isinstance(param, DTensor):
        return start, end
    if len(param.placements) != 1:
        raise ValueError("Alignment gate updates require a 1D FSDP mesh")
    placement = param.placements[0]
    if isinstance(placement, Replicate):
        return start, end
    if not isinstance(placement, Shard) or placement.dim != 0:
        raise ValueError("Alignment gate updates require row sharding")
    width = math.ceil(param.shape[0] / param.device_mesh.size())
    offset = width * param.device_mesh.get_local_rank()
    local_rows = param.to_local().shape[0]
    return max(0, min(local_rows, start - offset)), max(0, min(local_rows, end - offset))


def production_momentum(G, M, momentum, nesterov):
    """Production EMA; keep FP32 through communication before normalization."""
    mu = float(momentum)
    updates = []
    for grad, buffer in zip(G, M):
        grad = grad.to(buffer.dtype)
        buffer.lerp_(grad, 1 - mu)
        updates.append(grad.lerp(buffer, mu) if nesterov else buffer.clone())
    return updates


def production_newton_schulz(value, epsilon=1e-7, num_experts=1):
    """Production-specific fused BF16 polynomial, isolated from the legacy kernel."""
    shape = value.shape
    matrices = value.float().reshape(num_experts, -1, value.shape[-1])
    results = []
    # Production operates on individual logical matrices. addmm and baddbmm
    # can round differently in BF16 even with identical coefficients.
    for matrix in matrices:
        transpose = matrix.shape[-2] > matrix.shape[-1]
        if transpose:
            matrix = matrix.mT
        matrix = F.normalize(matrix, p=2, dim=(-2, -1), eps=1e-7).to(torch.bfloat16)
        for a, b, c in ((4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377),
                        (3.7418, -5.5913, 2.3037), (2.8769, -3.1427, 1.2046),
                        (2.8366, -3.0525, 1.2012)):
            gram = matrix @ matrix.mT
            polynomial = torch.addmm(gram, gram, gram, alpha=c, beta=b)
            matrix = torch.addmm(matrix, polynomial, matrix, alpha=1.0, beta=a)
        results.append(matrix.float().mT if transpose else matrix.float())
    return torch.stack(results).reshape(shape)


def adam_gate_update(target, grad, state, group):
    """Apply Adam direction after the fused parameter's existing decay step."""
    # Reuse PyTorch AdamW rather than modifying the shared xTuner AdamW kernel.
    # The fused Muon parameter already received its weight decay once.
    adamw(
        [target], [grad], [state["production_gate_m"]], [state["production_gate_v"]], [],
        [torch.tensor(float(state["production_gate_step"] - 1))],
        amsgrad=False, beta1=group["beta1"], beta2=group["beta2"], lr=group["lr"],
        weight_decay=0.0, eps=group["epsilon"], maximize=False, foreach=False, fused=False,
    )

class MegatronMuon(Muon):
    """EMA Muon with semantic projection splits and AdamW control rows."""

    def __init__(self, params, *, muon_adamw_ranges=None, **kwargs):
        super().__init__(params, pre_orthogonalize_fn=production_momentum,
                         newton_schulz_func=production_newton_schulz, **kwargs)
        self._muon_adamw_ranges = muon_adamw_ranges or {}
        for param, (start, end) in self._muon_adamw_ranges.items():
            if param not in self._muon_split_sizes or start != sum(self._muon_split_sizes[param]):
                raise ValueError("AdamW control rows must follow the independent Muon matrices")
            if not 0 < start < end == param.shape[0]:
                raise ValueError("Invalid fused projection AdamW range")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("MegatronMuon does not support closures")
        updates = []
        for group in self.param_groups:
            for param in group["params"]:
                if param not in self._muon_adamw_ranges or param.grad is None:
                    continue
                lo, hi = local_row_range(param, *self._muon_adamw_ranges[param])
                if lo == hi:
                    continue
                local = param.to_local() if hasattr(param, "to_local") else param
                grad = param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
                state = self.state[param]
                if "production_gate_m" not in state:
                    state["production_gate_m"] = torch.zeros_like(local[lo:hi])
                    state["production_gate_v"] = torch.zeros_like(local[lo:hi])
                state["production_gate_step"] = state.get("production_gate_step", 0) + 1
                updates.append((local[lo:hi], grad[lo:hi].clone(), state, group))
        result = super().step()
        # Native split NS leaves these rows zero but already applies decay.
        for target, grad, state, group in updates:
            adam_gate_update(target, grad, state, group)
        return result
