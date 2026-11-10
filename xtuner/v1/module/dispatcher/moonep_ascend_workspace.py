"""Ascend (cann-shmem) variant of the MoonEP expert workspace.

This module replaces the GPU-only VMM allocation path
(``_ExpertVMMWorkspace.allocate`` → ``nvl_dist_alloc`` / ``nvl_dist_map`` /
CUDA IPC FD exchange) with a lightweight adapter that pulls zero-copy views
from ``BufferXtuner``'s symmetric heap (``xt_*_view``).  The runtime methods
inherited from ``_ExpertVMMWorkspace`` — ``prefetch_weights``,
``return_expert_gradients``, ``local_compute_view``, ``landing``,
``generation_for`` — consume only ``_WorkspaceLayout`` and are fully
shape-compatible with the Ascend views.

Layout correspondence (``_WorkspaceLayout`` ↔ ``xt_*_view``):

  landings[gen][proj]               = xt_weight_home_view(gen, proj)            [epn, O_p, I_p]
  global_weights[gen][proj]         = xt_weight_full_view(gen, proj)            [epn+B, O_p, I_p]
  local_weights[gen][proj]          = xt_weight_full_view(gen, proj)            [epn+B, O_p, I_p]
  local_grad_outputs[slot][proj]    = xt_grad_full_view(slot, proj)             [epn+B, O_p, I_p]
  distributed_duplicate_grads[slot][proj] = placeholder (kernel self-fetches)  [R, B, O_p, I_p]
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import torch
import torch.distributed as dist
from typing_extensions import TypedDict

from .moonep_workspace import _ExpertVMMWorkspace, _WorkspaceLayout


def _layout_from_buffer(
    buffer: Any,
    *,
    ep_size: int,
    gradient_slots: int,
    home_generations: int,
) -> _WorkspaceLayout:
    """Build ``_WorkspaceLayout`` from BufferXtuner's symmetric-heap views.

    Shared by the eager path (``allocate(buffer=...)``) and the deferred path
    (``build_from_buffer()``), which differ only in when the Buffer exists.

    Projection indexing: proj=0 → w13 [B, 2Hp, H], proj=1 → w2 [B, H, Hp].
    """
    # landings[G][P], each [epn, O_p, I_p] — FSDP AllGather targets.
    landings: list[list[torch.Tensor]] = []
    # global_weights[G][P], each [epn+B, O_p, I_p] — prefetch addresses.
    global_weights: list[list[torch.Tensor]] = []
    # local_weights[G][P], each [epn+B, O_p, I_p] — grouped GEMM compute alias.
    local_weights: list[list[torch.Tensor]] = []

    for gen in range(home_generations):
        gen_landings: list[torch.Tensor] = []
        gen_globals: list[torch.Tensor] = []
        gen_locals: list[torch.Tensor] = []
        for proj in range(2):
            gen_landings.append(buffer.xt_weight_home_view(gen, proj))  # [B, O_p, I_p]
            # home and dup are physically contiguous inside the home window, so
            # global/local weights can take the single-segment [epn+B] view with
            # no materialization. torch.cat would allocate outside the heap and
            # the prefetch data_ptr check rejects such tensors.
            full_view = buffer.xt_weight_full_view(gen, proj)  # [epn+B, O_p, I_p]
            gen_globals.append(full_view)
            gen_locals.append(full_view)
        landings.append(gen_landings)
        global_weights.append(gen_globals)
        local_weights.append(gen_locals)

    # local_grad_outputs[N][P], each [epn+B, O_p, I_p] — [home, duplicate] grad.
    # distributed_duplicate_grads[N][P], each [R, B, O_p, I_p] — on Ascend the
    # kernel self-fetches peer duplicate slots through symmetric heap
    # addressing, so this stays a zero-allocation contract-check placeholder
    # (reduce_grad_bf16 only asserts the arity, never reads it).
    local_grad_outputs: list[list[torch.Tensor]] = []
    distributed_duplicate_grads: list[list[torch.Tensor]] = []

    for slot in range(gradient_slots):
        slot_grads: list[torch.Tensor] = []
        slot_dist: list[torch.Tensor] = []
        for proj in range(2):
            # Full heap view (home+dup contiguous, zero copy). Building it with
            # torch.cat would materialize ~1.6GB@GLM that nothing reads.
            slot_grads.append(buffer.xt_grad_full_view(slot, proj))  # [epn+B, O_p, I_p]
            grad_dup = buffer.xt_grad_duplicate_view(slot, proj)
            # .expand(...).contiguous() would materialize [R, B, O, I] = R×dup
            # (R=2 → 17GB/rank permanent, R=16 → 138GB, guaranteed OOM).
            slot_dist.append(grad_dup.unsqueeze(0).expand(ep_size, *grad_dup.shape))
        local_grad_outputs.append(slot_grads)
        distributed_duplicate_grads.append(slot_dist)

    # keepalives: hold strong refs to the heap views so their storage is not
    # reclaimed early; BufferXtuner owns the underlying heap.
    keepalives = (
        *sum(landings, []),
        *sum(global_weights, []),
        *sum(local_weights, []),
        *sum(local_grad_outputs, []),
    )

    return {
        "landings": tuple(tuple(p) for p in landings),
        "global_weights": tuple(tuple(p) for p in global_weights),
        "local_weights": tuple(tuple(p) for p in local_weights),
        "local_grad_outputs": tuple(tuple(p) for p in local_grad_outputs),
        "distributed_duplicate_grads": tuple(tuple(p) for p in distributed_duplicate_grads),
        "keepalives": keepalives,
    }


class _ExpertAscendWorkspace(_ExpertVMMWorkspace):
    """Ascend workspace: views from BufferXtuner symmetric heap, no VMM/IPC.

    Inherits all runtime methods from ``_ExpertVMMWorkspace`` unchanged —
    they read only ``_WorkspaceLayout`` via ``self._views``, which is
    constructed here from ``BufferXtuner.xt_*_view`` instead of VMM mappings.
    """

    # ---------------------------------------------------------------------------
    # Deferred layout: build_from_buffer() fills _layout after buffer creation.
    # ---------------------------------------------------------------------------

    def __init__(self, *, layout, ep_group, ep_rank, num_experts,
                 experts_per_rank, home_generations):
        # Allow deferred layout (None) — filled by build_from_buffer().
        self._layout = layout
        self._ep_group = ep_group
        self._ep_rank = ep_rank
        self._num_experts = num_experts
        self._experts_per_rank = experts_per_rank
        self._home_generations = home_generations
        # If layout is deferred, set _gradient_slots from gradient_slots param;
        # otherwise infer from layout (same as parent).
        if layout is not None:
            self._gradient_slots = len(layout["local_grad_outputs"])
        else:
            self._gradient_slots = 0  # updated in build_from_buffer()
        self._destroyed = False

    def build_from_buffer(self, buffer: Any, gradient_slots: int) -> None:
        """Fill the deferred layout from BufferXtuner's symmetric heap views.

        Called once from ``buffer_for()`` after the Buffer is created.
        Constructs ``_WorkspaceLayout`` from ``xt_*_view`` and sets
        ``_gradient_slots``.
        """
        if self._layout is not None:
            raise RuntimeError("Ascend workspace layout already built")
        if self._destroyed:
            raise RuntimeError("Ascend workspace has been destroyed")

        self._layout = _layout_from_buffer(
            buffer,
            ep_size=dist.get_world_size(self._ep_group),
            gradient_slots=gradient_slots,
            home_generations=self._home_generations,
        )
        self._gradient_slots = gradient_slots

    @classmethod
    def allocate(
        cls,
        *,
        projection_shapes: Sequence[tuple[int, int]],
        num_experts: int,
        ep_group: dist.ProcessGroup,
        gradient_slots: int,
        home_generations: int = 1,
        buffer: Any = None,
    ) -> _ExpertAscendWorkspace:
        """Build the workspace layout from BufferXtuner's symmetric heap views.

        Unlike the GPU ``_ExpertVMMWorkspace.allocate`` which allocates VMM
        chunks, exchanges IPC FDs, and maps cross-rank views via
        ``nvl_dist_map``, this method pulls zero-copy views directly from the
        already-allocated cann-shmem symmetric heap owned by BufferXtuner.

        Args:
            projection_shapes: ((2*Hp, H), (H, Hp)) — w13 / w2 projection shapes.
            num_experts: total routed experts E.
            ep_group: EP process group (must be initialized).
            gradient_slots: N gradient slots (= intra_layer_micro_batch).
            home_generations: G home generations (1 by default, 2 = double-buffered).
            buffer: BufferXtuner instance (must be created and workspace-ready).
        """
        ep_size = dist.get_world_size(ep_group)
        ep_rank = dist.get_rank(ep_group)
        experts_per_rank = num_experts // ep_size

        if buffer is None:
            # Deferred mode: layout built later by ``build_from_buffer()``.
            # Used when install_after_fsdp runs before the first forward
            # (Buffer is lazily created in buffer_for once Fixed-S is known).
            return cls(
                layout=None,  # deferred — filled by build_from_buffer()
                ep_group=ep_group,
                ep_rank=ep_rank,
                num_experts=num_experts,
                experts_per_rank=experts_per_rank,
                home_generations=home_generations,
            )
        # Home generation switch: only 1 (single generation, halves home memory)
        # or 2 (double generation) is allowed.
        if home_generations not in (1, 2):
            raise ValueError(f"Ascend workspace requires home_generations 1 or 2, got {home_generations}")
        if len(projection_shapes) != 2:
            raise ValueError("MoonEP requires fused w1/w3 and w2 projections (projection_shapes len=2)")

        layout = _layout_from_buffer(
            buffer,
            ep_size=ep_size,
            gradient_slots=gradient_slots,
            home_generations=home_generations,
        )

        return cls(
            layout=layout,
            ep_group=ep_group,
            ep_rank=ep_rank,
            num_experts=num_experts,
            experts_per_rank=experts_per_rank,
            home_generations=home_generations,
        )

    # ---------------------------------------------------------------------------
    # Override: destroy() — no VMM/FD teardown; just drop layout refs.
    # The BufferXtuner owns the symmetric heap and handles SHMEM finalize.
    # ---------------------------------------------------------------------------

    def destroy(self) -> None:
        """Release layout references; BufferXtuner owns heap lifecycle."""
        if self._destroyed:
            return
        # No torch.cuda.synchronize() — NPU uses torch.npu.synchronize().
        # BufferXtuner.destroy() handles SHMEM finalize; we only drop views.
        dist.barrier(group=self._ep_group)
        self._layout = None
        self._destroyed = True

    # ---------------------------------------------------------------------------
    # Asynchronous weight prefetch. BufferXtuner takes the generation index and
    # can run the transfer on its own stream, which the CUDA Buffer cannot; the
    # dispatcher's overlap switches call these instead of
    # ``_ExpertVMMWorkspace.prefetch_weights``.
    # ---------------------------------------------------------------------------

    def start_prefetch(
        self,
        *,
        buffer: Any,
        plan: Any,
        generation: int,
        stream: Any = None,
    ) -> Any:
        """Launch the weight prefetch without touching the launching stream.

        Returns the completion event so a caller can run the transfer
        concurrently with its own work and join later.
        """
        return buffer.prefetch_weight(
            plan=plan,
            projections=self._views["global_weights"][generation],
            generation=generation,
            async_finish=True,
            stream=stream,
        )

    def prefetch_weights(
        self,
        *,
        buffer: Any,
        plan: Any,
        generation: int,
        async_finish: bool = False,
        stream: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prefetch the global ``[epn+B]`` weights and return this generation's
        local compute aliases.

        Args:
            async_finish (bool): Launch on ``stream`` and return immediately
                instead of blocking the launching stream.
            stream (Any): Target stream when ``async_finish`` is set; ``None``
                leaves the choice to the buffer (its comm stream).
        """
        buffer.prefetch_weight(
            plan=plan,
            projections=self._views["global_weights"][generation],
            generation=generation,
            async_finish=async_finish,
            stream=stream,
        )
        return self.local_weights(generation=generation)

    # ---------------------------------------------------------------------------
    # X8 accessors: public hooks for the dispatcher to pull per-slot heap
    # gradient views so it can inject them as ``grad_weight_out`` into the
    # expert GroupedLinear forward calls. ``local_grad_outputs[slot][proj]``
    # is the full [epn+B, O, I] home+dup view (X8 writes home add_ + dup copy_).
    # ---------------------------------------------------------------------------

    def grad_view(self, slot: int, projection: int) -> torch.Tensor:
        """Full [epn+B, O, I] heap gradient slot view (home+dup) for X8."""
        if self._layout is None:
            raise RuntimeError("Ascend workspace layout not built (buffer deferred?)")
        return self._views["local_grad_outputs"][slot][projection]

    def home_grad_view(self, slot: int, projection: int) -> torch.Tensor:
        """Home-only [B, O, I] heap gradient slot view (for the first-mb zero_)."""
        if self._layout is None:
            raise RuntimeError("Ascend workspace layout not built (buffer deferred?)")
        return self._views["local_grad_outputs"][slot][projection][: self._experts_per_rank]

    def __del__(self) -> None:
        if getattr(self, "_destroyed", True) or getattr(self, "_layout", None) is None:
            return
        warnings.warn(
            "MoonEP Ascend workspace was not destroyed explicitly; "
            "resources may leak.",
            ResourceWarning,
        )
        # Keep layout alive for rank-divergence safety (same as GPU path).
        from .moonep_workspace import _UNDISPOSED_WORKSPACE_TENSORS
        _UNDISPOSED_WORKSPACE_TENSORS.append(self._layout)
        self._layout = None
