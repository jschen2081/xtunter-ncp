# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Annotated, Any, Literal, Sequence, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from cyclopts import Parameter
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_reduce

from xtuner.v1.utils.device import get_device

# from xtuner.v1.profiler.prober import ProberList
from .base_loss_ctx import BaseLossConfig, BaseLossContext, BaseLossKwargs
from .chunk_loss import ChunkLoss, linear_with_grad_accumulation
from .utils import sp_gather, sp_split


DEVICE = get_device()


def _all_gather_flat(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Concatenate one flat tensor from every rank of *group* along dim 0.

    Args:
        tensor (torch.Tensor): 1-D tensor owned by this rank.
        group (dist.ProcessGroup): Group whose ranks contribute equally sized tensors.

    Returns:
        torch.Tensor: Concatenation ordered by group rank.
    """
    world = dist.get_world_size(group)
    if world == 1:
        return tensor
    tensor = tensor.contiguous()
    local_size = torch.tensor([tensor.size(0)], dtype=torch.long, device=tensor.device)
    sizes = [torch.empty_like(local_size) for _ in range(world)]
    dist.all_gather(sizes, local_size, group=group)
    if any(size.item() != tensor.size(0) for size in sizes):
        raise NotImplementedError(
            "Vocabulary parallelism requires every rank of the group to hold the same token count; "
            f"got sizes {[size.item() for size in sizes]}"
        )
    gathered = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(gathered, tensor, group=group)
    return torch.cat(gathered, dim=0)


class _VocabParallelLinear(torch.autograd.Function):
    """Linear over the vocabulary slice of this rank, applied to every rank's tokens.

    Each rank gathers the whole group's hidden states and projects them with its
    own ``[V / P, H]`` slice, so the language-model loss sees every token on every
    rank while the head weight is never materialised in full. The input gradient
    is the group-wide sum of the per-slice partial products; the weight gradient
    covers every token already and is divided by the group size to match the
    data-parallel mean the rest of the model uses.
    """

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        vocab_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        local_tokens = hidden_states.numel() // hidden_states.size(-1)
        flat_hidden = hidden_states.reshape(local_tokens, hidden_states.size(-1))
        flat_hidden = _all_gather_flat(flat_hidden, vocab_group)
        # The head lives outside FSDP, so nothing casts the fp32 master to the
        # activation dtype for us; mirror what FSDP's mixed precision would do.
        compute_weight = weight.to(flat_hidden.dtype)
        ctx.save_for_backward(flat_hidden, compute_weight)
        ctx.weight_dtype = weight.dtype
        ctx.vocab_group = vocab_group
        ctx.local_tokens = local_tokens
        ctx.input_shape = hidden_states.shape
        ctx.rank = dist.get_rank(vocab_group)
        return F.linear(flat_hidden, compute_weight)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        flat_hidden, compute_weight = ctx.saved_tensors
        grad_input = None
        grad_weight = None
        flat_grad_output = grad_output.reshape(-1, grad_output.size(-1))
        if ctx.needs_input_grad[0]:
            grad_full = flat_grad_output.matmul(compute_weight)
            # Every rank projected the same tokens, so each token's gradient is
            # only a partial sum over the vocabulary slices.
            dist.all_reduce(grad_full, op=dist.ReduceOp.SUM, group=ctx.vocab_group)
            start = ctx.rank * ctx.local_tokens
            grad_input = grad_full[start : start + ctx.local_tokens].reshape(ctx.input_shape)
        if ctx.needs_input_grad[1]:
            # The gather covers every rank's tokens, so this row slice already
            # holds the group-wide sum. Convert it to the data-parallel mean the
            # rest of the model uses.
            grad_weight = flat_grad_output.t().matmul(flat_hidden)
            grad_weight = (grad_weight / dist.get_world_size(ctx.vocab_group)).to(ctx.weight_dtype)
        return grad_input, grad_weight, None


class CELossConfig(BaseLossConfig):
    """Cross-entropy loss configuration.

    Args:
        ignore_idx (int): The index to ignore in the loss computation.
            Defaults to -100.
        mode (str): The mode for loss computation. Options are "eager" and "chunk".
            Defaults to "eager".
        chunk_size (int | None): The chunk size for "chunk" mode. Ignored if mode is "eager".
            Defaults to 1024.
        loss_reduction (str): The reduction mode for the loss. Options are "token", "sample", and "square".
    """

    mode: Annotated[Literal["eager", "chunk", "liger"], Parameter(help="loss calculation mode")] = "eager"  # type: ignore
    loss_reduction: Annotated[Literal["token", "sample", "square"], Parameter(help="loss reduction mode")] = "token"

    @property
    def loss_ctx_cls(self) -> type["CELossContext"]:
        return CELossContext

    @property
    def _loss_kwargs_cls(self) -> type["CELossKwargs"]:
        return CELossKwargs

    def model_post_init(self, _context: Any) -> None:
        if self.mode == "liger":
            assert self.loss_reduction == "token", "Currently, cannot use liger kernel with sample or square reduction"

    def build(
        self,
        data: dict,
        sp_mesh: DeviceMesh | None = None,
    ) -> "CELossContext | None":
        """Build CELossContext from data dict.

        Args:
            data (dict): Data dict containing loss-related fields.
                Required: shifted_labels
            sp_mesh (DeviceMesh | None): Sequence parallel mesh.

        Returns:
            CELossContext | None: Built loss context. Returns None if shifted_labels
                is not present in data dict.
        """
        if "shifted_labels" not in data:
            return None
        # Extract required fields from data
        shifted_labels = data["shifted_labels"]

        loss_kwargs = CELossKwargs(shifted_labels=shifted_labels).to(DEVICE)
        if sp_mesh is not None and sp_mesh.size() > 1:
            loss_kwargs = loss_kwargs.sp_split(sp_mesh)
        loss_ctx = self.loss_ctx_cls(self, loss_kwargs)
        return loss_ctx


class CELossKwargs(BaseLossKwargs):
    """Keyword arguments for cross-entropy loss computation.

    Args:
        shifted_labels (torch.Tensor): The shifted labels for the input sequences.
        loss_weight (torch.Tensor): The weight for each token in the loss computation.
    """

    shifted_labels: torch.Tensor
    loss_weight: torch.Tensor | None = None

    def sp_split(self, sp_mesh: DeviceMesh) -> "CELossKwargs":
        self.shifted_labels = sp_split(self.shifted_labels, sp_mesh=sp_mesh, split_dim=1, padding_value=-100)
        return self

    def to(self, device: torch.device | str) -> "CELossKwargs":
        self.shifted_labels = self.shifted_labels.to(device)
        return self


class LMHeadLossContext(BaseLossContext):
    """Cross-entropy loss context for language models.

    Args:
        loss_cfg (CELossConfig): The configuration for the cross-entropy loss.
        loss_kwargs (CELossKwargs): The keyword arguments for the cross-entropy loss.
    """

    loss_cfg: CELossConfig
    loss_kwargs: CELossKwargs

    def __init__(self, loss_cfg: CELossConfig, loss_kwargs: CELossKwargs):
        super().__init__(loss_cfg, loss_kwargs)

        if loss_cfg.mode == "liger":
            # Device-dispatched FLCE: NPU uses the in-package implementation
            # (xtuner.v1.loss.get_flce_loss_cls), CUDA uses liger_kernel's
            # stock class. reduction stays 'sum' on both devices.
            from xtuner.v1.loss import get_flce_loss_cls

            self.liger_loss_fct = get_flce_loss_cls()(
                reduction="sum",
                accum_dtype=torch.float32,
            )
        else:
            self.liger_loss_fct = None

    @staticmethod
    def build_batches(  # type: ignore[override]
        loss_ctx_list: list["CELossContext"],
        cu_seq_lens_list: Sequence[torch.IntTensor] | None = None,
        sp_mesh: DeviceMesh | None = None,
    ) -> list["CELossContext"]:
        assert len(loss_ctx_list) > 0, "loss_ctx_list can not be empty"
        loss_cfg = loss_ctx_list[0].loss_cfg

        loss_weight_list: list[torch.Tensor] = []
        for i, loss_ctx in enumerate(loss_ctx_list):
            shifted_labels = loss_ctx.loss_kwargs.shifted_labels
            if loss_cfg.loss_reduction == "token":
                loss_weight = torch.ones_like(shifted_labels, dtype=torch.float32)
            else:
                assert cu_seq_lens_list is not None, "cu_seq_lens_list must be provided for sample or square reduction"
                cu_seq_lens = cu_seq_lens_list[i].to(shifted_labels.device)
                boundaries = cu_seq_lens[1:]
                num_tokens = cu_seq_lens[1:] - cu_seq_lens[:-1]

                if sp_mesh is not None:
                    # gather shifted_labels from different sp ranks to compute the correct loss weight
                    shifted_labels = sp_gather(shifted_labels, sp_mesh=sp_mesh, dim=1)

                mask = (shifted_labels != loss_cfg.ignore_idx).int()
                num_grad_tokens = torch.zeros_like(boundaries, dtype=torch.int32)
                prev_idx = 0
                for j, boundary in enumerate(boundaries):
                    num_grad_tokens[j] = mask[0, prev_idx:boundary].sum()
                    prev_idx = boundary
                if loss_cfg.loss_reduction == "sample":
                    loss_weight = 1.0 / num_grad_tokens
                elif loss_cfg.loss_reduction == "square":
                    loss_weight = 1.0 / torch.sqrt(num_grad_tokens.float())
                else:
                    raise NotImplementedError(loss_cfg.loss_reduction)
                loss_weight = loss_weight.repeat_interleave(num_tokens).unsqueeze(0)

                if sp_mesh is not None:
                    loss_weight = sp_split(loss_weight, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
                    shifted_labels = sp_split(shifted_labels, sp_mesh=sp_mesh, split_dim=1, padding_value=-100)

            loss_weight[shifted_labels == loss_cfg.ignore_idx] = 0.0
            if torch.isnan(loss_weight).any() or torch.isinf(loss_weight).any():
                raise AssertionError(
                    "loss_weight contains NaN or Inf values. Please filter out samples with no valid tokens."
                )
            loss_ctx.loss_kwargs.loss_weight = loss_weight
            loss_weight_list.append(loss_weight)

        # Compute the denominator used in the global calibration of the loss
        rank_denominator = sum(loss_weight.sum() for loss_weight in loss_weight_list)
        rank_denominator = cast(torch.Tensor, rank_denominator)
        global_denominator = rank_denominator
        if dist.is_initialized():
            dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)

        for loss_ctx in loss_ctx_list:
            loss_ctx._batch_size = len(loss_ctx_list)
            assert loss_ctx.loss_kwargs.loss_weight is not None
            loss_ctx.loss_kwargs.loss_weight /= global_denominator + 1e-12
        return loss_ctx_list

    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: CELossKwargs,
        valid_vocab_size: int | None = None,
        grad_weight_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        # We do linear forward here to simplify the implementation of chunk loss (saving memory).
        if grad_weight_buffer is None:
            logits = F.linear(hidden_states, head_weight, head_bias)
        else:
            logits = linear_with_grad_accumulation(hidden_states, head_weight, grad_weight_buffer)
        physical_vocab_size = logits.size(-1)
        if valid_vocab_size is None:
            valid_vocab_size = physical_vocab_size
        if not 0 < valid_vocab_size <= physical_vocab_size:
            raise ValueError("valid_vocab_size must be positive and no larger than the physical vocabulary")
        if valid_vocab_size < physical_vocab_size:
            # Keep the aligned physical GEMM shape, then exclude padded rows
            # before the dtype conversion makes the logical logits contiguous.
            # This preserves the exact logical softmax and gives padded rows a
            # gradient of zero without copying the BF16 tensor separately.
            logits = logits[..., :valid_vocab_size]
        logits = logits.float()  # (bs, seq_len, vocab_size)

        shifted_labels = loss_kwargs.shifted_labels  # (bs, seq_len)
        loss_weight = loss_kwargs.loss_weight  # (bs, seq_len)
        assert loss_weight is not None, "loss_weight can not be None"

        logits = logits.reshape(-1, logits.size(-1))  # (bs * seq_len, vocab_size)
        shifted_labels = shifted_labels.flatten()
        loss_weight = loss_weight.flatten()

        # ``reduction="none"`` already returns zero for ignored targets, even
        # when an entire chunk is ignored. Avoid checking the number of valid
        # targets in Python because converting that CUDA scalar to ``bool``
        # synchronizes every loss chunk with the host.
        loss = F.cross_entropy(logits, shifted_labels, reduction="none", ignore_index=self.loss_cfg.ignore_idx)
        # Step 2.b in the loss calculation: sum the loss over all tokens
        loss = (loss * loss_weight).sum()

        return loss, (logits, {})

    def vocab_parallel_loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: CELossKwargs,
        vocab_start_index: int,
        vocab_end_index: int,
        vocab_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        """Cross entropy over a vocabulary slice owned by this rank.

        Args:
            hidden_states (torch.Tensor): Hidden states with features in the last dimension.
            head_weight (torch.Tensor): Local ``[V / P, H]`` slice of the LM head.
            head_bias (torch.Tensor | None): Unsupported, must be ``None``.
            loss_kwargs (CELossKwargs): Per-token labels and weights.
            vocab_start_index (int): First global vocabulary id owned by this rank.
            vocab_end_index (int): One past the last global vocabulary id owned by this rank.
            vocab_group (dist.ProcessGroup): Group whose ranks partition the vocabulary.

        Returns:
            tuple[torch.Tensor, tuple[None, dict]]: This rank's ``1 / P`` share of the
            cross entropy, with an empty aux dict. The logits are not returned
            because they are only a vocabulary slice.
        """
        if head_bias is not None:
            raise NotImplementedError("Vocab-parallel CE does not support a head bias")

        logits = _VocabParallelLinear.apply(hidden_states, head_weight, vocab_group)
        local_vocab_size = logits.size(-1)
        logits = logits.float().reshape(-1, local_vocab_size)

        # Every rank projected the whole group's tokens, so the labels and their
        # weights must be gathered in the same rank order as the hidden states.
        assert loss_kwargs.loss_weight is not None, "loss_weight can not be None"
        shifted_labels = _all_gather_flat(loss_kwargs.shifted_labels.flatten(), vocab_group)
        loss_weight = _all_gather_flat(loss_kwargs.loss_weight.flatten(), vocab_group)

        ignore_idx = self.loss_cfg.ignore_idx
        ignore_mask = shifted_labels == ignore_idx
        in_local = (shifted_labels >= vocab_start_index) & (shifted_labels < vocab_end_index) & ~ignore_mask
        target_index = torch.where(in_local, shifted_labels - vocab_start_index, torch.zeros_like(shifted_labels))
        # `gather` records `logits` for its backward, so the buffer must stay
        # untouched; the masked copy below becomes the single work buffer.
        local_target = torch.where(
            in_local,
            logits.gather(-1, target_index[:, None]).squeeze(-1),
            torch.zeros((), dtype=logits.dtype, device=logits.device),
        )

        # One full-size temporary from here on: at 32768 tokens x 18992 columns
        # in fp32, every extra copy of this shape costs 2374 MiB.
        masked_logits = logits.masked_fill(ignore_mask[:, None], float("-inf"))

        # The shift is detached: log-sum-exp is invariant to it, and `-inf`
        # rows mean every rank agrees the token is masked. Detaching also keeps
        # `masked_logits` out of any autograd node that would object to the
        # in-place shift below.
        local_max = masked_logits.detach().max(dim=-1).values.clone()
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=vocab_group)
        global_max = torch.where(torch.isneginf(local_max), torch.zeros_like(local_max), local_max)

        masked_logits.sub_(global_max[:, None])
        masked_logits.exp_()
        local_sum = masked_logits.sum(dim=-1)
        # Autograd-aware SUM: every rank sees the global log-sum-exp and the
        # backward distributes the shared gradient back to each local partial sum.
        global_sum = all_reduce(local_sum, op=dist.ReduceOp.SUM, group=vocab_group)
        lse = global_max + global_sum.log()

        global_target = all_reduce(local_target, op=dist.ReduceOp.SUM, group=vocab_group)

        token_loss = lse - global_target
        # Masked tokens produce -inf above; zero them before they meet a zero weight.
        token_loss = torch.where(ignore_mask, torch.zeros_like(token_loss), token_loss)
        # Every rank of the group computed the same full-token loss, so the
        # outer world-sum reduce would multiply it by P. Divide here: the value
        # lands on the true loss and the gradient is compensated by the same
        # factor when autograd walks back through the world reduce.
        loss = (token_loss * loss_weight).sum() / dist.get_world_size(vocab_group)
        return loss, (None, {})

    def eager_mode(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: CELossKwargs,
        valid_vocab_size: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        return self.loss_fn(
            hidden_states,
            head_weight,
            head_bias,
            loss_kwargs,
            valid_vocab_size=valid_vocab_size,
        )

    def chunk_mode(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: CELossKwargs,
        valid_vocab_size: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        if self.loss_cfg.mode == "chunk":
            assert self.loss_cfg.chunk_size is not None, "chunk_size must be set in chunk mode"

            chunks = loss_kwargs.chunk(self.loss_cfg.chunk_size)
            loss_forward = partial(self.loss_fn, valid_vocab_size=valid_vocab_size)
            # Subclasses with another loss_fn retain the original callable contract.
            accumulate_forward = loss_forward if type(self).loss_fn is LMHeadLossContext.loss_fn else None
            loss, extra_info = ChunkLoss.apply(
                hidden_states,
                head_weight,
                head_bias,
                loss_forward,
                chunks,
                self.loss_cfg.chunk_size,
                accumulate_forward,
            )
            return loss, (None, extra_info)
        else:
            assert self.liger_loss_fct is not None, "liger_loss_fct must be initialized in liger mode"
            if valid_vocab_size is not None and valid_vocab_size < head_weight.size(0):
                raise NotImplementedError("Padded LM heads are not supported by the liger CE loss")
            shifted_labels = loss_kwargs.shifted_labels  # (bs, seq_len)
            loss_weight = loss_kwargs.loss_weight  # (bs, seq_len)
            assert loss_weight is not None, "loss_weight can not be None"

            bs, seq, dim = hidden_states.shape
            hidden_states = hidden_states.reshape(bs * seq, dim)
            shifted_labels = shifted_labels.flatten()
            # liger kernel dont support reduction=="none"
            # step 2.b in the loss calculation: sum the loss over all tokens, then multiply the loss weight (i.e. divide by the global_denominator)
            loss = self.liger_loss_fct(head_weight, hidden_states, shifted_labels)
            # ProberList.record_tensor(loss, "[lm_head.ce_loss][before calibration]loss")
            mask = loss_weight != 0
            # w equals to 1/global_denominator on normal shards. clamp(min=1)
            # guards the all-padding SP-shard where mask.sum()==0 would give
            # 0/0=nan (loss is finite 0 there -> w=0 -> loss*0=0). no-op n_valid>0.
            w = loss_weight.sum() / mask.sum().clamp(min=1)
            loss = loss * w
            return loss, (None, {})

    def _vocab_parallel_mode(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        loss_kwargs: CELossKwargs,
        vocab_start_index: int,
        vocab_end_index: int,
        vocab_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, tuple[None, dict[str, Any]]]:
        if self.loss_cfg.mode == "liger":
            raise NotImplementedError("vocab_parallel_size does not support the liger CE mode")
        loss_forward = partial(
            self.vocab_parallel_loss_fn,
            vocab_start_index=vocab_start_index,
            vocab_end_index=vocab_end_index,
            vocab_group=vocab_group,
        )
        if self.loss_cfg.mode == "eager":
            return loss_forward(hidden_states, head_weight, None, loss_kwargs)
        if self.loss_cfg.mode != "chunk":
            raise NotImplementedError(f"vocab_parallel_size does not support CE mode {self.loss_cfg.mode!r}")
        assert self.loss_cfg.chunk_size is not None, "chunk_size must be set in chunk mode"
        chunks = loss_kwargs.chunk(self.loss_cfg.chunk_size)
        loss, extra_info = ChunkLoss.apply(
            hidden_states,
            head_weight,
            None,
            loss_forward,
            chunks,
            self.loss_cfg.chunk_size,
            None,
        )
        return loss, (None, extra_info)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None = None,
        valid_vocab_size: int | None = None,
        vocab_start_index: int | None = None,
        vocab_end_index: int | None = None,
        vocab_group: dist.ProcessGroup | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        from xtuner.v1.model.utils.misc import ModelForwardExtraLogInfo

        assert self.loss_kwargs is not None, "loss_kwargs must be set before calling forward"
        if head_bias is not None:
            raise NotImplementedError("Loss does not support head_bias yet.")

        if vocab_start_index is not None:
            if valid_vocab_size is not None:
                raise NotImplementedError("Vocabulary parallelism and a padded head cannot be combined")
            assert vocab_end_index is not None and vocab_group is not None
            loss, (logits, extra_info) = self._vocab_parallel_mode(
                hidden_states,
                head_weight,
                self.loss_kwargs,
                vocab_start_index=vocab_start_index,
                vocab_end_index=vocab_end_index,
                vocab_group=vocab_group,
            )
        elif self.loss_cfg.mode == "eager":
            loss, (logits, extra_info) = self.eager_mode(
                hidden_states,
                head_weight,
                head_bias,
                self.loss_kwargs,
                valid_vocab_size=valid_vocab_size,
            )
        else:
            loss, (logits, extra_info) = self.chunk_mode(
                hidden_states,
                head_weight,
                head_bias,
                self.loss_kwargs,
                valid_vocab_size=valid_vocab_size,
            )

        # TODO: yanhuida, should be removed
        if not isinstance(extra_info, ModelForwardExtraLogInfo):
            extra_info = ModelForwardExtraLogInfo(extra_info)

        extra_info["local_base_loss"] = loss.detach().clone()

        # Step 2.c in the loss calculation: reduce the loss over all ranks using all_reduce with autograd support
        if dist.is_initialized():
            loss = all_reduce(loss, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

        return loss, (logits, extra_info)


# Deprecated: Use LMHeadLossContext instead. Will be removed in version 1.1.0
CELossContext = LMHeadLossContext
