"""ConceptLM VQ/HLM objectives and globally calibrated loss contexts."""

from itertools import accumulate

import torch
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import Field
from torch.distributed.nn.functional import all_reduce

from xtuner.v1.module.concept_quantizer import quantize_concepts
from xtuner.v1.module.concept_sequence import next_concept_mse, padded_document_lengths

from .base_loss_ctx import BaseLossConfig, BaseLossContext, BaseLossKwargs


def auxiliary_numerators(
    concept, predicted, codebooks, mask, boundaries, *, chunk_size=4, commitment_cost=0.25, norm_eps=1e-6
):
    """Compose existing quantization and document-aware HLM primitives."""

    def norm(value: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(value, (value.shape[-1],), eps=norm_eps)

    vq_input = norm(concept.detach())
    valid = mask.bool().reshape(mask.shape[0], -1, chunk_size).any(-1)
    quantized = quantize_concepts(
        vq_input.transpose(0, 1),
        torch.stack(tuple(codebooks)),
        valid.transpose(0, 1),
        commitment_cost=commitment_cost,
    )
    concept_lengths = [(end - start) // chunk_size for start, end in zip(boundaries, boundaries[1:])]
    target = norm(concept).detach()
    hlm_num, _ = next_concept_mse(norm(predicted), target, concept_lengths, valid_mask=valid)
    return quantized.numerator, hlm_num


class ConceptLMLossKwargs(BaseLossKwargs):
    loss_mask: torch.Tensor
    boundaries: torch.Tensor
    global_num_tokens: torch.Tensor | None = None
    # Full masks are replicated across SP peers; per-peer target counts must be
    # normalized by sp_size before any WORLD-level aggregation.
    sp_size: int = 1


class ConceptLMLossConfig(BaseLossConfig):
    """Optional full ConceptLM objective; labels must already be shifted."""

    concept_chunk_size: int = Field(default=4, gt=0)
    commitment_cost: float = Field(default=0.25, ge=0)
    vq_weight: float = Field(default=1.0, ge=0)
    hlm_weight: float = Field(default=1.0, ge=0)
    norm_eps: float = Field(default=1e-6, gt=0)

    @property
    def loss_ctx_cls(self):
        return ConceptLMLossContext

    @property
    def _loss_kwargs_cls(self):
        return ConceptLMLossKwargs

    def build(self, data, sp_mesh=None):
        seq_ctx = data["seq_ctx"]
        # Auxiliary loss uses full gathered concept features and full token masks.
        # SP replicas intentionally count replicated masks in WORLD calibration;
        # this balances differentiable WORLD reduction and FSDP gradient averaging.
        labels = data["shifted_labels"]
        mask = data.get("loss_mask", labels.ne(self.ignore_idx))
        if mask.shape != labels.shape or mask.ndim != 2 or mask.shape[0] != 1:
            raise ValueError("ConceptLM loss requires a single packed token sequence")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("ConceptLM loss requires a binary token mask")
        if not torch.equal(mask.bool(), labels.ne(self.ignore_idx)):
            raise ValueError("ConceptLM and LM target masks must agree")
        boundaries = seq_ctx.cu_seq_lens_q
        if sp_mesh is None:
            sp_mesh = getattr(seq_ctx, "sequence_parallel_mesh", None)
        sp_size = sp_mesh.size() if sp_mesh is not None else 1
        if sp_size > 1:
            # Under SP the model strips padding documents (collator pack padding and
            # shard alignment padding live beyond the valid region), so its concept
            # stream covers only the valid tokens. Scope the loss mask and document
            # boundaries to the same region, or the concept count would differ.
            valid_length = getattr(seq_ctx, "valid_sequence_length", None)
            if valid_length is None:
                valid_length = mask.shape[1] - int(getattr(seq_ctx, "num_padding", 0))
            valid_length = int(valid_length)
            boundaries = boundaries[boundaries <= valid_length]
            if boundaries.numel() == 0 or boundaries[0].item() != 0:
                boundaries = torch.cat((boundaries.new_zeros(1), boundaries))
            if boundaries[-1].item() != valid_length:
                boundaries = torch.cat((boundaries, boundaries.new_tensor([valid_length])))
            mask = mask[:, :valid_length]
        lengths = [int(value) for value in (boundaries[1:] - boundaries[:-1]).tolist()]
        if any(length <= 0 for length in lengths):
            raise ValueError("ConceptLM loss does not support empty documents")
        padded_lengths = padded_document_lengths(lengths, self.concept_chunk_size, sp_size)
        if padded_lengths != lengths:
            # The model pads short or chunk-unaligned documents (and the SP tail)
            # with pad tokens, so its concept stream is longer than the raw token
            # mask. Grow the mask the same way with zeros: padded concepts carry no
            # valid target and drop out of both auxiliary objectives.
            pieces = []
            start = 0
            for length, padded in zip(lengths, padded_lengths, strict=True):
                pieces.append(mask[:, start : start + length])
                start += length
                if padded > length:
                    pieces.append(mask.new_zeros((1, padded - length)))
            mask = torch.cat(pieces, dim=1)
            boundaries = torch.tensor(
                [0, *accumulate(padded_lengths)], dtype=boundaries.dtype, device=boundaries.device
            )
        if boundaries[0].item() != 0 or boundaries[-1].item() != mask.shape[1]:
            raise ValueError("ConceptLM boundaries must cover all tokens")
        alignment = sp_size * self.concept_chunk_size
        if mask.shape[1] % alignment:
            raise ValueError(
                f"ConceptLM loss requires total tokens ({mask.shape[1]}) to be divisible by "
                f"SP size ({sp_size}) * concept chunk size ({self.concept_chunk_size}) = {alignment}"
            )
        # Native Trainer keeps collated labels on CPU; the context owns its
        # device-local mask, as the CE context does. Preserve full SP masks.
        mask = mask.to(seq_ctx.input_ids.device)
        return ConceptLMLossContext(self, ConceptLMLossKwargs(loss_mask=mask, boundaries=boundaries, sp_size=sp_size))


class ConceptLMLossContext(BaseLossContext):
    @staticmethod
    def build_batches(contexts, **kwargs):
        if not contexts:
            raise ValueError("ConceptLM needs at least one microbatch")
        count = torch.stack([ctx.loss_kwargs.loss_mask.sum().float() for ctx in contexts]).sum()
        if dist.is_initialized():
            dist.all_reduce(count)
        if count.item() <= 0:
            raise ValueError("ConceptLM batch has no effective target tokens")
        for ctx in contexts:
            ctx.loss_kwargs.global_num_tokens = count
            ctx._batch_size = len(contexts)
        return contexts

    def forward(self, concept, predicted, codebooks):
        config, kwargs = self.loss_cfg, self.loss_kwargs
        if kwargs.global_num_tokens is None:
            raise RuntimeError("ConceptLM contexts must be calibrated with build_batches")
        vq, hlm = auxiliary_numerators(
            concept,
            predicted,
            codebooks,
            kwargs.loss_mask,
            kwargs.boundaries.tolist(),
            chunk_size=config.concept_chunk_size,
            commitment_cost=config.commitment_cost,
            norm_eps=config.norm_eps,
        )
        loss = config.concept_chunk_size * (config.vq_weight * vq + config.hlm_weight * hlm)
        loss = loss / kwargs.global_num_tokens
        # Same differentiable WORLD SUM convention as xTuner's CE context.
        if dist.is_initialized():
            loss = all_reduce(loss, group=dist.group.WORLD)
        # Logging metrics carry their global step contribution: each rank's share is
        # summed over WORLD (SP peers hold identical full masks, so target counts are
        # normalized by sp_size first), and microbatch sums then give step totals.
        metrics = torch.stack(
            [
                config.concept_chunk_size * config.vq_weight * vq.detach() / kwargs.global_num_tokens,
                config.concept_chunk_size * config.hlm_weight * hlm.detach() / kwargs.global_num_tokens,
                kwargs.loss_mask.sum().detach().float() / kwargs.sp_size,
            ]
        )
        if dist.is_initialized():
            dist.all_reduce(metrics, group=dist.group.WORLD)
        return loss, {
            "concept_vq_numerator": vq.detach(),
            "concept_hlm_numerator": hlm.detach(),
            "concept_vq_loss": metrics[0],
            "concept_hlm_loss": metrics[1],
            "concept_valid_targets": metrics[2],
        }
