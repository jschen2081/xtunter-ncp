"""Muon variant for ConceptLM models: split projection layouts on xTuner's native DTensor optimizer.

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
        raise ValueError("Gate row updates require a 1D FSDP mesh")
    placement = param.placements[0]
    if isinstance(placement, Replicate):
        return start, end
    if not isinstance(placement, Shard) or placement.dim != 0:
        raise ValueError("Gate row updates require row sharding")
    width = math.ceil(param.shape[0] / param.device_mesh.size())
    offset = width * param.device_mesh.get_local_rank()
    local_rows = param.to_local().shape[0]
    return max(0, min(local_rows, start - offset)), max(0, min(local_rows, end - offset))


def ema_momentum(G, M, momentum, nesterov):
    """EMA momentum; buffers stay FP32 so reduced gradients keep full
    precision."""
    mu = float(momentum)
    updates = []
    for grad, buffer in zip(G, M):
        grad = grad.to(buffer.dtype)
        buffer.lerp_(grad, 1 - mu)
        updates.append(grad.lerp(buffer, mu) if nesterov else buffer.clone())
    return updates


def quintic_newton_schulz(value, epsilon=1e-7, num_experts=1):
    """Fused BF16 quintic polynomial Newton-Schulz, isolated from the legacy
    kernel."""
    shape = value.shape
    matrices = value.float().reshape(num_experts, -1, value.shape[-1])
    results = []
    # Orthogonalize individual logical matrices. addmm and baddbmm can round
    # differently in BF16 even with identical coefficients.
    for matrix in matrices:
        transpose = matrix.shape[-2] > matrix.shape[-1]
        if transpose:
            matrix = matrix.mT
        matrix = F.normalize(matrix, p=2, dim=(-2, -1), eps=1e-7).to(torch.bfloat16)
        for a, b, c in (
            (4.0848, -6.8946, 2.9270),
            (3.9505, -6.3029, 2.6377),
            (3.7418, -5.5913, 2.3037),
            (2.8769, -3.1427, 1.2046),
            (2.8366, -3.0525, 1.2012),
        ):
            gram = matrix @ matrix.mT
            polynomial = torch.addmm(gram, gram, gram, alpha=c, beta=b)
            matrix = torch.addmm(matrix, polynomial, matrix, alpha=1.0, beta=a)
        results.append(matrix.float().mT if transpose else matrix.float())
    return torch.stack(results).reshape(shape)


def adam_gate_update(target, grad, m, v, state, group):
    """Apply Adam direction after the fused parameter's existing decay step."""
    # Reuse PyTorch AdamW rather than modifying the shared xTuner AdamW kernel.
    # The fused Muon parameter already received its weight decay once.
    # FP32 states reject BF16 operands; update an FP32 copy and write back.
    target_fp32 = target.float()
    adamw(
        [target_fp32],
        [grad.to(m.dtype)],
        [m],
        [v],
        [],
        [torch.tensor(float(state["gate_step"] - 1))],
        amsgrad=False,
        beta1=group["beta1"],
        beta2=group["beta2"],
        lr=group["lr"],
        weight_decay=0.0,
        eps=group["epsilon"],
        maximize=False,
        foreach=False,
        fused=False,
    )
    target.copy_(target_fp32)


class ConceptMuon(Muon):
    """EMA Muon that orthogonalizes each role-specific block of fused
    projections independently.

    Optimizer state (momentum/variance and the control-row Adam buffers) is kept in FP32 by default while parameters
    may stay low-precision.
    """

    def __init__(self, params, *, muon_adamw_ranges=None, optim_state_dtype=torch.float32, **kwargs):
        super().__init__(
            params,
            pre_orthogonalize_fn=ema_momentum,
            newton_schulz_func=quintic_newton_schulz,
            optim_state_dtype=optim_state_dtype,
            **kwargs,
        )
        self._muon_adamw_ranges = muon_adamw_ranges or {}
        for param, (start, end) in self._muon_adamw_ranges.items():
            if param not in self._muon_split_sizes or start != sum(self._muon_split_sizes[param]):
                raise ValueError("AdamW control rows must follow the independent Muon matrices")
            if not 0 < start < end == param.shape[0]:
                raise ValueError("Invalid fused projection AdamW range")
        for group in self.param_groups:
            if group["algorithm"] != "muon":
                continue
            for param in group["params"]:
                if param.ndim != 2:
                    # ConceptMuon has no flattened/batched Newton-Schulz path: >2D
                    # tensors (e.g. GatedDeltaNet conv1d) must be routed to AdamW
                    # through get_muon_metadata param_options instead.
                    raise ValueError(
                        f"ConceptMuon only orthogonalizes 2D parameters, got {tuple(param.shape)}; "
                        "route higher-rank tensors to AdamW via get_muon_metadata param_options"
                    )

    def load_state_dict(self, state_dict):
        legacy = [key for state in state_dict["state"].values() for key in state if key.startswith("production_gate_")]
        if legacy:
            # Legacy MegatronMuon checkpoints store gate Adam state as per-rank
            # local row slices under production_gate_*; ConceptMuon keeps them
            # param-shaped and sharded like the fused parameter. The layouts are
            # not convertible, and silently accepting the old keys would reinitialize
            # the gate state (momentum zeroed, step reset) on the next step.
            raise ValueError(
                f"Checkpoint contains legacy MegatronMuon gate state keys {sorted(set(legacy))}; "
                "ConceptMuon gate state is not layout-compatible with it. "
                "Resume with a fresh optimizer state instead."
            )
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("ConceptMuon does not support closures")
        updates = []
        for group in self.param_groups:
            for param in group["params"]:
                if param not in self._muon_adamw_ranges:
                    continue
                state = self.state[param]
                # Parameter-shaped states follow the fused parameter's sharding, so
                # distributed checkpoints preserve every rank's rows. The step
                # counter must stay consistent on ranks holding no gate rows.
                if "gate_momentum" not in state:
                    dtype = self._optim_state_dtype or torch.float32
                    state["gate_momentum"] = torch.zeros_like(param, dtype=dtype)
                    state["gate_variance"] = torch.zeros_like(param, dtype=dtype)
                    state["gate_step"] = 0
                if param.grad is None:
                    continue
                # Advance only on steps with a gradient so AdamW bias correction
                # counts real updates, but before the local-rows check so ranks
                # holding no gate rows stay in lockstep. The gate projection is
                # dense, so grad-None steps happen identically on all ranks.
                state["gate_step"] += 1
                lo, hi = local_row_range(param, *self._muon_adamw_ranges[param])
                if lo == hi:
                    continue
                local = param.to_local() if hasattr(param, "to_local") else param
                grad = param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
                m = state["gate_momentum"]
                v = state["gate_variance"]
                if hasattr(m, "to_local"):
                    m = m.to_local()
                    v = v.to_local()
                updates.append((local[lo:hi], grad[lo:hi].clone(), m[lo:hi], v[lo:hi], state, group))
        result = super().step()
        # Native split NS leaves these rows zero but already applies decay.
        for target, grad, m, v, state, group in updates:
            adam_gate_update(target, grad, m, v, state, group)
        return result
