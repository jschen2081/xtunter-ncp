"""Checkpoint-backed Ulysses SP acceptance tests for ConceptLM Qwen3.5 v2."""

import gc
import math
import os
import unittest
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.utils.hooks import RemovableHandle

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.dense.conceptlm_qwen35_v2 import (
    ConceptLMQwen35V2,
    ConceptLMQwen35V2Config,
    _crop_documents,
    _hf_padded_lengths,
)
from xtuner.v1.module.attention.ncp_qwen35_v2 import NCPQwen35V2GatedDeltaNet, NCPQwen35V2MLA
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer
from xtuner.v1.module.decoder_layer.fused_swiglu import FusedSwiGLUMLP
from xtuner.v1.module.rms_norm.ncp_qwen35_v2 import (
    GatedDeltaNetInputRMSNorm,
    MuddRMSNorm,
    ZeroCenteredRMSNorm,
)


DEFAULT_CHECKPOINT = Path(
    "/mnt/shared-storage-user/llmrazor-share/heqiushi/ncp/qwen35/"
    "ncp_qwen35_v2_2551e48_iter0000002_hf"
)
CHECKPOINT = Path(os.environ.get("NCP_QWEN35_V2_PATH", DEFAULT_CHECKPOINT))
# Different SP GEMM shapes change floating-point accumulation order. These
# bounds reject non-finite or grossly unrelated outputs without requiring the
# SP=1 and SP=8 logits to be numerically close element by element.
_LOGIT_MEAN_ABS = 5.0e-1
_LOGIT_TOP1 = 0.5


def _capture_embedding_leaf(model: nn.Module, captured: dict[str, Tensor]) -> RemovableHandle:
    def make_leaf(_module: nn.Module, _inputs: tuple[object, ...], output: Tensor) -> Tensor:
        leaf = output.detach().requires_grad_(True)
        captured["embedding"] = leaf
        return leaf

    return model.embed_tokens.register_forward_hook(make_leaf)


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="eight CUDA devices are required for SP=8")
@unittest.skipUnless(CHECKPOINT.is_dir(), f"checkpoint does not exist: {CHECKPOINT}")
class TestConceptLMQwen35V2UlyssesSP(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 8

    @staticmethod
    def _configure_kernels() -> None:
        os.environ.pop("XTUNER_HF_IMPL", None)
        os.environ["XTUNER_GDN_CUSTOM_OP"] = "true"
        os.environ["XTUNER_GDN_FUSED_QK_L2"] = "true"
        os.environ["XTUNER_GDN_FUSED_NORM_GATE"] = "true"
        os.environ["NCP_Q35V2_TORCH_RMS"] = "1"
        os.environ["NCP_Q35V2_TE_RMS"] = "0"
        os.environ["NCP_Q35V2_TRITON_DUAL_CROSS_MUDD"] = "0"

    def _load_model(self) -> ConceptLMQwen35V2:
        self._configure_kernels()
        # The v2 recipe is pinned to the full 9B checkpoint contract. Unlike the
        # generic Qwen configs, it intentionally has no config-level ``from_hf``;
        # weights are loaded below through the model-level adapter.
        config = ConceptLMQwen35V2Config(compile_cfg=False)
        with torch.device("meta"):
            model = config.build()
        model.fully_shard(
            FSDPConfig(
                requires_grad=False,
                recompute_ratio=0.0,
                reshard_after_forward=True,
                cpu_offload=False,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                fp32_lm_head=False,
            )
        )
        loaded, unloaded, missing = model.from_hf(CHECKPOINT, strict=True)
        assert loaded
        assert not unloaded
        assert not missing
        return model.eval()

    @staticmethod
    def _documents(lengths: list[int], seed: int) -> tuple[Tensor, ...]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return tuple(
            torch.randint(2, 248320, (1, length), generator=generator, dtype=torch.long).cuda()
            for length in lengths
        )

    @staticmethod
    def _aligned_reference_documents(
        model: ConceptLMQwen35V2,
        documents: tuple[Tensor, ...],
        sp_mesh: DeviceMesh,
    ) -> tuple[tuple[Tensor, ...], list[int]]:
        """Pad the SP=1 reference to the exact internal SP=8 layout."""
        lengths = [document.shape[1] for document in documents]
        compute_lengths = _hf_padded_lengths(lengths, model.config.concept_chunk_size)
        alignment = sp_mesh.size() * model.config.concept_chunk_size
        compute_lengths[-1] += (-sum(compute_lengths)) % alignment
        pad_id = model.config.pad_token_id
        if pad_id is None:
            pad_id = model.config.eos_token_id
        aligned: list[Tensor] = []
        for document, compute_length in zip(documents, compute_lengths, strict=True):
            padding = compute_length - document.shape[1]
            if padding:
                document = torch.cat((document, document.new_full((1, padding), pad_id)), dim=1)
            aligned.append(document)
        return tuple(aligned), compute_lengths

    def _assert_local_logits(
        self,
        actual: Tensor,
        expected: Tensor,
        lengths: list[int],
        sp_mesh: DeviceMesh,
        chunk_tokens: int = 8,
        boundary_audit: str | None = None,
    ) -> None:
        assert actual.shape == expected.shape
        max_abs = 0.0
        abs_sum = 0.0
        squared_gap_sum = 0.0
        actual_squared_sum = 0.0
        expected_squared_sum = 0.0
        dot_sum = 0.0
        nonfinite_count = 0
        element_count = 0
        top1_matches = 0
        # Output restoration adds zero rows outside the logical sequence.
        # Those rows must not dilute the error or inflate top-1 agreement.
        shard_start = sp_mesh.get_local_rank() * actual.shape[1]
        token_count = max(0, min(actual.shape[1], sum(lengths) - shard_start))
        for start in range(0, token_count, chunk_tokens):
            end = min(start + chunk_tokens, token_count)
            actual_chunk = actual[:, start:end].float()
            expected_chunk = expected[:, start:end].to(device=actual.device).float()
            nonfinite_count += (~torch.isfinite(actual_chunk)).sum().item()
            nonfinite_count += (~torch.isfinite(expected_chunk)).sum().item()
            gap = (actual_chunk - expected_chunk).abs()
            max_abs = max(max_abs, gap.max().item())
            abs_sum += gap.sum().item()
            squared_gap_sum += gap.square().sum().item()
            actual_squared_sum += actual_chunk.square().sum().item()
            expected_squared_sum += expected_chunk.square().sum().item()
            dot_sum += (actual_chunk * expected_chunk).sum().item()
            element_count += gap.numel()
            top1_matches += actual_chunk.argmax(dim=-1).eq(expected_chunk.argmax(dim=-1)).sum().item()

        max_tensor = torch.tensor(max_abs, device=actual.device, dtype=torch.float64)
        totals = torch.tensor(
            [
                abs_sum,
                float(element_count),
                float(top1_matches),
                float(token_count),
                squared_gap_sum,
                actual_squared_sum,
                expected_squared_sum,
                dot_sum,
                float(nonfinite_count),
            ],
            device=actual.device,
            dtype=torch.float64,
        )
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX, group=sp_mesh.get_group())
        dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=sp_mesh.get_group())
        global_max_abs = max_tensor.item()
        global_mean_abs = (totals[0] / totals[1]).item()
        global_top1 = (totals[2] / totals[3]).item()
        global_relative_l2 = (totals[4].sqrt() / totals[6].sqrt().clamp_min(1.0e-12)).item()
        global_cosine = (totals[7] / (totals[5] * totals[6]).sqrt().clamp_min(1.0e-12)).item()
        if self.rank == 0:
            print(
                f"lengths={lengths} max_abs={global_max_abs:.3e} "
                f"mean_abs={global_mean_abs:.3e} relative_l2={global_relative_l2:.3e} "
                f"cosine={global_cosine:.8f} top1={global_top1:.6f}"
            )
        failures: list[str] = []
        if totals[8].item() or not all(
            math.isfinite(value) for value in (global_mean_abs, global_top1, global_relative_l2, global_cosine)
        ):
            failures.append(f"non-finite logits/metrics (elements={int(totals[8].item())})")
        if global_mean_abs > _LOGIT_MEAN_ABS:
            failures.append(f"mean_abs={global_mean_abs:.3e}>{_LOGIT_MEAN_ABS:.3e}")
        if global_top1 < _LOGIT_TOP1:
            failures.append(f"top1={global_top1:.6f}<{_LOGIT_TOP1:.6f}")
        assert not failures, (
            f"lengths={lengths}: {'; '.join(failures)} "
            f"(diagnostic max_abs={global_max_abs:.3e}, mean_abs={global_mean_abs:.3e}, "
            f"relative_l2={global_relative_l2:.3e}, cosine={global_cosine:.8f}, "
            f"top1={global_top1:.6f}; boundary={boundary_audit})"
        )

    @staticmethod
    def _expected_local_logits(full_logits: Tensor, local_length: int, sp_mesh: DeviceMesh) -> Tensor:
        total_length = local_length * sp_mesh.size()
        if full_logits.shape[1] < total_length:
            padding = full_logits.new_zeros(
                full_logits.shape[0], total_length - full_logits.shape[1], full_logits.shape[2]
            )
            full_logits = torch.cat((full_logits, padding), dim=1)
        start = sp_mesh.get_local_rank() * local_length
        return full_logits[:, start : start + local_length].detach().cpu()

    @staticmethod
    def _capture_boundaries(
        model: ConceptLMQwen35V2,
        captured: dict[str, Tensor],
    ) -> list[RemovableHandle]:
        handles: list[RemovableHandle] = []

        def make_hook(name: str) -> Callable[[nn.Module, tuple[object, ...], object], None]:
            def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
                if isinstance(output, dict):
                    output = output.get("projected_output")
                elif isinstance(output, (tuple, list)):
                    output = next((item for item in output if isinstance(item, Tensor)), None)
                if not isinstance(output, Tensor):
                    return
                captured[name] = output.detach()

            return hook

        boundary_types = (
            nn.Embedding,
            nn.Linear,
            nn.RMSNorm,
            GatedDeltaNetInputRMSNorm,
            MuddRMSNorm,
            ZeroCenteredRMSNorm,
            NCPQwen35V2GatedDeltaNet,
            NCPQwen35V2MLA,
            FusedSwiGLUMLP,
            DenseDecoderLayer,
        )
        for name, module in model.named_modules():
            if isinstance(module, boundary_types):
                handles.append(module.register_forward_hook(make_hook(name)))
        return handles

    def _report_first_boundary_divergence(
        self,
        baseline: dict[str, Tensor],
        actual: dict[str, Tensor],
        sp_mesh: DeviceMesh,
    ) -> str:
        assert baseline.keys() == actual.keys()
        for name in baseline:
            local = actual[name]
            expected = baseline[name].float()
            if local.shape == baseline[name].shape:
                full = local.float()
            else:
                gathered = [torch.empty_like(local) for _ in range(sp_mesh.size())]
                dist.all_gather(gathered, local, group=sp_mesh.get_group())
                full = torch.cat(gathered, dim=1).float()
            assert full.shape == expected.shape, (name, full.shape, expected.shape)
            gap = (full - expected).abs()
            mean_abs = gap.mean().item()
            if mean_abs <= 1.0e-3:
                continue
            cosine = torch.nn.functional.cosine_similarity(full.flatten(), expected.flatten(), dim=0).item()
            if self.rank == 0:
                print(
                    f"sp_boundary_audit first={name} max_abs={gap.max().item():.3e} "
                    f"mean_abs={mean_abs:.3e} cosine={cosine:.8f}"
                )
            return f"first={name},max_abs={gap.max().item():.3e},mean_abs={mean_abs:.3e},cosine={cosine:.8f}"
        if self.rank == 0:
            print("sp_boundary_audit all captured boundaries have mean_abs<=1.000e-03")
        return "all captured boundaries have mean_abs<=1.000e-03"

    def _forward_pair(
        self,
        model: ConceptLMQwen35V2,
        documents: tuple[Tensor, ...],
        sp_mesh: DeviceMesh,
        audit_boundaries: bool = False,
    ) -> tuple[Tensor, Tensor, str | None]:
        lengths = [document.shape[1] for document in documents]
        reference_documents, compute_lengths = self._aligned_reference_documents(model, documents, sp_mesh)
        baseline_ctx = SequenceContext.from_input_ids(reference_documents, device="cuda")
        baseline_boundaries: dict[str, Tensor] = {}
        handles = self._capture_boundaries(model, baseline_boundaries) if audit_boundaries else []
        try:
            with torch.no_grad():
                baseline_logits = model(baseline_ctx).logits
        finally:
            for handle in handles:
                handle.remove()
        assert baseline_logits is not None
        baseline_logits = _crop_documents(baseline_logits, lengths, compute_lengths)
        sp_ctx = SequenceContext.from_input_ids(documents, device="cuda").split(sp_mesh)
        assert sp_ctx.input_ids is not None
        expected = self._expected_local_logits(baseline_logits, sp_ctx.input_ids.shape[1], sp_mesh)
        del baseline_logits
        gc.collect()
        torch.cuda.empty_cache()
        actual_boundaries: dict[str, Tensor] = {}
        handles = self._capture_boundaries(model, actual_boundaries) if audit_boundaries else []
        try:
            with torch.no_grad():
                local_logits = model(sp_ctx).logits
        finally:
            for handle in handles:
                handle.remove()
        assert local_logits is not None
        boundary_audit = None
        if audit_boundaries:
            boundary_audit = self._report_first_boundary_divergence(baseline_boundaries, actual_boundaries, sp_mesh)
        return local_logits, expected, boundary_audit

    def _assert_embedding_gradient(
        self,
        model: ConceptLMQwen35V2,
        documents: tuple[Tensor, ...],
        lengths: list[int],
        sp_mesh: DeviceMesh,
    ) -> None:
        reference_documents, compute_lengths = self._aligned_reference_documents(model, documents, sp_mesh)
        baseline_ctx = SequenceContext.from_input_ids(reference_documents, device="cuda")
        baseline_capture: dict[str, Tensor] = {}
        handle = _capture_embedding_leaf(model, baseline_capture)
        try:
            baseline_logits = model(baseline_ctx).logits
            assert baseline_logits is not None
            baseline_logits = _crop_documents(baseline_logits, lengths, compute_lengths)
            baseline_loss = baseline_logits.float().square().mean()
            (baseline_gradient,) = torch.autograd.grad(baseline_loss, baseline_capture["embedding"])
        finally:
            handle.remove()
        baseline_gradient = _crop_documents(baseline_gradient, lengths, compute_lengths)
        del baseline_logits, baseline_loss

        sp_ctx = SequenceContext.from_input_ids(documents, device="cuda").split(sp_mesh)
        sp_capture: dict[str, Tensor] = {}
        handle = _capture_embedding_leaf(model, sp_capture)
        try:
            local_logits = model(sp_ctx).logits
            assert local_logits is not None
            global_elements = sum(lengths) * local_logits.shape[-1]
            local_loss = local_logits.float().square().sum() / global_elements
            (local_gradient,) = torch.autograd.grad(local_loss, sp_capture["embedding"])
        finally:
            handle.remove()
        if local_gradient.shape[1] == sum(compute_lengths):
            full_gradient = local_gradient.detach().clone()
            dist.all_reduce(full_gradient, op=dist.ReduceOp.SUM, group=sp_mesh.get_group())
        else:
            gathered_gradients = [torch.empty_like(local_gradient) for _ in range(sp_mesh.size())]
            dist.all_gather(gathered_gradients, local_gradient.detach(), group=sp_mesh.get_group())
            full_gradient = torch.cat(gathered_gradients, dim=1)
        actual_gradient = _crop_documents(full_gradient, lengths, compute_lengths).float()
        expected_gradient = baseline_gradient.float()
        relative_l2 = (
            (actual_gradient - expected_gradient).norm() / expected_gradient.norm().clamp_min(1.0e-12)
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            actual_gradient.flatten(), expected_gradient.flatten(), dim=0
        ).item()
        assert torch.isfinite(actual_gradient).all(), f"lengths={lengths}: non-finite SP embedding gradient"
        assert actual_gradient.norm().item() > 0, f"lengths={lengths}: empty SP embedding gradient"
        assert math.isfinite(relative_l2) and math.isfinite(cosine)
        if self.rank == 0:
            print(f"lengths={lengths} embedding_gradient relative_l2={relative_l2:.3e} cosine={cosine:.6f}")

    def test_short_forward_backward_matches_sp1(self) -> None:
        self.create_pg("cuda")
        model = self._load_model()
        sp_mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("sp",))
        for case_index, lengths in enumerate(([9], [5, 11])):
            with self.subTest(lengths=lengths):
                documents = self._documents(lengths, seed=7 + case_index)
                actual, expected, boundary_audit = self._forward_pair(model, documents, sp_mesh, audit_boundaries=True)
                self._assert_local_logits(actual, expected, lengths, sp_mesh, boundary_audit=boundary_audit)
                del actual, expected
                torch.cuda.empty_cache()
                self._assert_embedding_gradient(model, documents, lengths, sp_mesh)
                torch.cuda.empty_cache()

    def test_long_forward_matches_sp1_through_16k(self) -> None:
        self.create_pg("cuda")
        model = self._load_model()
        sp_mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("sp",))
        cases = ([4096], [8192], [16384], [8000, 8384])
        failures: list[str] = []
        for case_index, lengths in enumerate(cases):
            with self.subTest(lengths=lengths):
                documents = self._documents(lengths, seed=17 + case_index)
                actual, expected, _ = self._forward_pair(model, documents, sp_mesh)
                try:
                    self._assert_local_logits(actual, expected, lengths, sp_mesh)
                except AssertionError as error:
                    # Metrics are reduced over SP first, so every rank makes
                    # the same decision and can safely proceed to the next case.
                    failures.append(str(error))
                del actual, expected, documents
                gc.collect()
                torch.cuda.empty_cache()
        assert not failures, "\n".join(failures)
