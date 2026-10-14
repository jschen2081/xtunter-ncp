"""CPU regression tests for the alignment script runtime compatibility."""

import math
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch


@pytest.fixture
def alignment_script(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    script = Path(__file__).resolve().parents[2] / "examples/v1/scripts/align_conceptlm_qwen35_v2_hf.py"
    monkeypatch.setattr(os, "environ", os.environ.copy())
    return runpy.run_path(str(script))


class TestHFModulesCache:
    def test_inaccessible_default_uses_writable_fallback(
        self, alignment_script: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("HF_MODULES_CACHE", raising=False)
        blocked = tmp_path / "blocked"
        blocked.mkdir(mode=0o500)
        monkeypatch.setenv("HF_HOME", str(blocked))
        try:
            alignment_script["configure_hf_modules_cache"]()
            cache = Path(os.environ["HF_MODULES_CACHE"])
            (cache / "probe").write_text("writable")
        finally:
            blocked.chmod(0o700)

    def test_writable_default_is_preserved(
        self, alignment_script: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("HF_MODULES_CACHE", raising=False)
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        alignment_script["configure_hf_modules_cache"]()
        assert "HF_MODULES_CACHE" not in os.environ
        assert (tmp_path / "modules").is_dir()

    def test_explicit_cache_is_preserved(
        self, alignment_script: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        explicit = str(tmp_path / "explicit")
        monkeypatch.setenv("HF_MODULES_CACHE", explicit)
        alignment_script["configure_hf_modules_cache"]()
        assert os.environ["HF_MODULES_CACHE"] == explicit


class TestXTunerHFImplementation:
    def test_production_clears_broad_hf_override(
        self, alignment_script: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XTUNER_HF_IMPL", "true")

        alignment_script["configure_xtuner_hf_impl"]("production")

        assert "XTUNER_HF_IMPL" not in os.environ

    def test_reference_paths_enable_broad_hf_override(
        self, alignment_script: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XTUNER_HF_IMPL", raising=False)

        alignment_script["configure_xtuner_hf_impl"]("controlled")

        assert os.environ["XTUNER_HF_IMPL"] == "true"


class TestKernelConfiguration:
    @pytest.mark.parametrize(
        "kernel_path,production",
        [("controlled", False), ("production", True)],
    )
    def test_sets_complete_environment_matrix(
        self,
        alignment_script: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        kernel_path: str,
        production: bool,
    ) -> None:
        globals_ = alignment_script["configure_xtuner_kernels"].__globals__
        monkeypatch.setitem(globals_, "use_hf_mla_kernel", lambda: "hf-mla")
        monkeypatch.setitem(globals_, "use_hf_gdn_kernels", lambda: "hf-gdn")
        monkeypatch.setitem(globals_, "use_xtuner_mla_kernel", lambda: "xt-mla")
        monkeypatch.setitem(globals_, "use_xtuner_gdn_kernels", lambda: "xt-gdn")

        selected = alignment_script["configure_xtuner_kernels"](kernel_path)

        enabled = "1" if production else "0"
        boolean = "true" if production else "false"
        assert os.environ["XTUNER_USE_FA3"] == enabled
        assert os.environ["NCP_Q35V2_FA3"] == enabled
        assert os.environ["NCP_Q35V2_FLA_L2"] == enabled
        assert os.environ["NCP_Q35V2_GDN_TE_OUT_NORM"] == enabled
        assert os.environ["XTUNER_GDN_CUSTOM_OP"] == boolean
        assert os.environ["XTUNER_GDN_FUSED_QK_L2"] == boolean
        assert os.environ["XTUNER_GDN_FUSED_NORM_GATE"] == boolean
        assert selected == (("xt-mla", "xt-gdn") if production else ("hf-mla", "hf-gdn"))

    def test_rejects_unknown_kernel_path(
        self, alignment_script: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        globals_ = alignment_script["configure_xtuner_kernels"].__globals__
        monkeypatch.setitem(globals_, "configure_xtuner_hf_impl", lambda _path: None)
        with pytest.raises(ValueError, match="unknown kernel path"):
            alignment_script["configure_xtuner_kernels"]("unknown")


class TestHFMLARuntime:
    @pytest.mark.parametrize("pinned_available", [True, False])
    def test_selects_fa3_interface(
        self, alignment_script: dict[str, Any], monkeypatch: pytest.MonkeyPatch, pinned_available: bool
    ) -> None:
        modeling = ModuleType("fake_hf_modeling")
        runtime = ModuleType("flash_attn_interface")
        pinned_kernel = object()
        installed_kernel = object()

        def get_pinned_kernel() -> object:
            if not pinned_available:
                raise RuntimeError("CUDA MLA requires the pinned flash-attn-3==3.0.0 runtime")
            return pinned_kernel

        modeling._flash_attention_kernel = get_pinned_kernel
        runtime.flash_attn_func = installed_kernel
        model_type = type("HFModel", (), {"__module__": modeling.__name__})
        monkeypatch.setitem(sys.modules, modeling.__name__, modeling)
        monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
        monkeypatch.setenv("NCP_Q35V2_FA3", "1")
        alignment_script["relax_hf_mla_pin"](model_type())
        assert modeling._flash_attention_kernel() is (pinned_kernel if pinned_available else installed_kernel)

    def test_sdpa_mode_skips_fa3_loading(
        self, alignment_script: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NCP_Q35V2_FA3", "0")
        alignment_script["relax_hf_mla_pin"](object())


class TestHFGDNNormRuntime:
    @pytest.mark.parametrize("load_error", [None, ImportError("ABI mismatch"), OSError("missing shared library")])
    def test_initializes_te_or_uses_export_kernel(
        self, alignment_script: dict[str, Any], monkeypatch: pytest.MonkeyPatch, load_error: Exception | None
    ) -> None:
        import importlib

        modeling = ModuleType("fake_hf_gdn_modeling")
        native_norm, pinned_norm = object(), object()
        modeling._te_zero_rms_norm = native_norm
        modeling.zero_rmsnorm_cuda = pinned_norm
        model_type = type("HFModel", (), {"__module__": modeling.__name__})
        original_import = importlib.import_module
        imported = []

        def import_runtime(name: str) -> ModuleType:
            imported.append(name)
            if name == "transformer_engine.pytorch":
                if load_error is not None:
                    raise load_error
                return ModuleType(name)
            return original_import(name)

        monkeypatch.setitem(sys.modules, modeling.__name__, modeling)
        monkeypatch.setattr(importlib, "import_module", import_runtime)
        monkeypatch.setenv("NCP_Q35V2_GDN_TE_OUT_NORM", "1")
        alignment_script["relax_hf_te_runtime"](model_type())
        assert "transformer_engine.pytorch" in imported
        assert modeling._te_zero_rms_norm is (native_norm if load_error is None else pinned_norm)

    def test_controlled_skips_te_loading(
        self, alignment_script: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NCP_Q35V2_GDN_TE_OUT_NORM", "0")
        alignment_script["relax_hf_te_runtime"](object())


class TestAcceptanceGates:
    @pytest.mark.parametrize(
        "length,token_loss_atol,expected",
        [
            (9, 2e-3, {"atol": 5e-3, "loss_atol": 1e-4, "token_loss_atol": 2e-3, "top1_min": 1.0}),
            (128, 2e-3, {"atol": 1e-2, "loss_atol": 5e-4, "token_loss_atol": 2e-3, "top1_min": 1.0}),
            (8191, 0.1, {"atol": 0.15, "loss_atol": 1e-3, "token_loss_atol": 0.1, "top1_min": 0.997}),
        ],
    )
    def test_acceptance_uses_each_independent_limit(
        self,
        alignment_script: dict[str, Any],
        length: int,
        token_loss_atol: float,
        expected: dict[str, float],
    ) -> None:
        actual = alignment_script["acceptance"](length, 0.15, 1e-3, token_loss_atol, 0.997)
        assert actual == expected

    @pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
    @pytest.mark.parametrize("metric", ["max_abs", "loss_abs", "token_loss_max", "top1"])
    def test_non_finite_metrics_fail_closed(
        self, alignment_script: dict[str, Any], metric: str, bad_value: float
    ) -> None:
        row = {"max_abs": 0.0, "loss_abs": 0.0, "token_loss_max": 0.0, "top1": 1.0}
        row[metric] = bad_value
        limits = {"atol": 1.0, "loss_atol": 1.0, "token_loss_atol": 1.0, "top1_min": 0.0}

        assert alignment_script["gate_failures"](row, limits) == [f"{metric} is not finite"]

    def test_reports_every_exceeded_limit(self, alignment_script: dict[str, Any]) -> None:
        row = {"max_abs": 2.0, "loss_abs": 2.0, "token_loss_max": 2.0, "top1": 0.5}
        limits = {"atol": 1.0, "loss_atol": 1.0, "token_loss_atol": 1.0, "top1_min": 0.9}

        assert len(alignment_script["gate_failures"](row, limits)) == 4


class TestLengthValidation:
    def test_accepts_unique_next_token_lengths(self, alignment_script: dict[str, Any]) -> None:
        lengths = [2, 8, 8192]
        assert alignment_script["validate_lengths"](lengths) is lengths

    @pytest.mark.parametrize("lengths", [[], [1], [8193], [8, 8]])
    def test_rejects_invalid_lengths(self, alignment_script: dict[str, Any], lengths: list[int]) -> None:
        with pytest.raises(ValueError):
            alignment_script["validate_lengths"](lengths)


class TestChunkedMetrics:
    def test_matches_direct_metrics(self, alignment_script: dict[str, Any]) -> None:
        torch.manual_seed(0)
        reference = torch.randn(1, 7, 11)
        actual = reference + torch.randn_like(reference) * 0.01
        input_ids = torch.randint(0, 11, (1, 7))

        metrics = alignment_script["comparison_metrics"](actual, reference, input_ids, chunk_tokens=2)
        actual_loss, actual_token_loss = alignment_script["mean_ce"](actual, input_ids)
        reference_loss, reference_token_loss = alignment_script["mean_ce"](reference, input_ids)
        gap = (actual - reference).abs()

        assert metrics["max_abs"] == pytest.approx(gap.max().item())
        assert metrics["mean_abs"] == pytest.approx(gap.mean().item())
        assert metrics["top1"] == pytest.approx(actual.argmax(-1).eq(reference.argmax(-1)).float().mean().item())
        assert metrics["hf_loss"] == pytest.approx(actual_loss.item())
        assert metrics["xt_loss"] == pytest.approx(reference_loss.item())
        assert metrics["loss_abs"] == pytest.approx(abs(actual_loss.item() - reference_loss.item()))
        assert metrics["token_loss_max"] == pytest.approx(
            (actual_token_loss - reference_token_loss).abs().max().item()
        )

    def test_rejects_shape_mismatch(self, alignment_script: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="logit shape"):
            alignment_script["comparison_metrics"](
                torch.zeros(1, 2, 3),
                torch.zeros(1, 3, 3),
                torch.zeros(1, 2, dtype=torch.long),
            )


class TestBackwardHelpers:
    def test_captures_embedding_gradient_without_parameter_gradients(
        self, alignment_script: dict[str, Any]
    ) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = torch.nn.Embedding(7, 5)
                self.head = torch.nn.Linear(5, 7, bias=False)

            def get_input_embeddings(self) -> torch.nn.Module:
                return self.embed

        model = TinyModel().requires_grad_(False)
        input_ids = torch.tensor([[1, 2, 3, 4]])

        logits, gradient = alignment_script["input_embedding_gradient"](
            model,
            input_ids,
            lambda module, ids: module.head(module.embed(ids)),
        )

        assert logits.shape == (1, 4, 7)
        assert gradient.shape == (1, 4, 5)
        assert torch.isfinite(gradient).all()
        assert all(parameter.grad is None for parameter in model.parameters())
        assert not model.embed._forward_hooks
