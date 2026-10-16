"""ConceptLM VQ/HLM objectives and globally calibrated loss contexts (SP1)."""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import Field
from torch.distributed.nn.functional import all_reduce

from xtuner.v1.module.concept_quantizer import quantize_concepts
from xtuner.v1.module.concept_sequence import next_concept_mse

from .base_loss_ctx import BaseLossConfig, BaseLossContext, BaseLossKwargs


def auxiliary_numerators(concept, predicted, codebooks, mask, boundaries, *,
                         chunk_size=4, commitment_cost=0.25, norm_eps=1e-6):
    """Compose existing quantization and document-aware HLM primitives."""
    norm = lambda value: F.rms_norm(value, (value.shape[-1],), eps=norm_eps)
    vq_input = norm(concept.detach())
    valid = mask.bool().reshape(mask.shape[0], -1, chunk_size).any(-1)
    quantized = quantize_concepts(
        vq_input.transpose(0, 1), torch.stack(tuple(codebooks)), valid.transpose(0, 1),
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
        mesh = sp_mesh if sp_mesh is not None else seq_ctx.sequence_parallel_mesh
        if mesh is not None and mesh.size() > 1:
            raise NotImplementedError("ConceptLM auxiliary losses currently require SP1")
        labels = data["shifted_labels"]
        mask = data.get("loss_mask", labels.ne(self.ignore_idx))
        if mask.shape != labels.shape or mask.ndim != 2 or mask.shape[0] != 1:
            raise ValueError("ConceptLM loss requires a single packed token sequence")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("ConceptLM loss requires a binary token mask")
        if not torch.equal(mask.bool(), labels.ne(self.ignore_idx)):
            raise ValueError("ConceptLM and LM target masks must agree")
        boundaries = seq_ctx.cu_seq_lens_q
        lengths = boundaries[1:] - boundaries[:-1]
        if torch.any(lengths <= 0) or torch.any(lengths % self.concept_chunk_size):
            raise ValueError("ConceptLM sequence lengths must align to concept chunks")
        if boundaries[0].item() != 0 or boundaries[-1].item() != mask.shape[1]:
            raise ValueError("ConceptLM boundaries must cover all tokens")
        return ConceptLMLossContext(self, ConceptLMLossKwargs(loss_mask=mask, boundaries=boundaries))


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
            concept, predicted, codebooks, kwargs.loss_mask, kwargs.boundaries.tolist(),
            chunk_size=config.concept_chunk_size, commitment_cost=config.commitment_cost,
            norm_eps=config.norm_eps)
        loss = config.concept_chunk_size * (config.vq_weight * vq + config.hlm_weight * hlm)
        loss = loss / kwargs.global_num_tokens
        # Same differentiable WORLD SUM convention as xTuner's CE context.
        if dist.is_initialized():
            loss = all_reduce(loss, group=dist.group.WORLD)
        return loss, {"concept_vq_numerator": vq.detach(), "concept_hlm_numerator": hlm.detach()}
