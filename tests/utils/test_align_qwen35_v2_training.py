"""CPU tests for the Qwen3.5 v2 multi-step training alignment script."""

import argparse
import math
import os
import runpy
from pathlib import Path
from typing import Any

import pytest
import torch


@pytest.fixture
def training_script(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    script = Path(__file__).resolve().parents[2] / "examples/v1/scripts/align_conceptlm_qwen35_v2_hf_training.py"
    monkeypatch.setattr(os, "environ", os.environ.copy())
    return runpy.run_path(str(script))


def _options(**updates: Any) -> argparse.Namespace:
    values = {
        "steps": 3,
        "length": 8,
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "recompute_ratio": 1.0,
        "parameters": ["weight"],
        "logit_atol": 0.0,
        "loss_atol": 0.0,
        "grad_atol": 0.0,
        "parameter_atol": 0.0,
        "optimizer_atol": 0.0,
    }
    values.update(updates)
    return argparse.Namespace(**values)


class TestTrainingOptions:
    @pytest.mark.parametrize(
        "updates,match",
        [
            ({"steps": 0}, "--steps"),
            ({"lr": 0.0}, "--lr"),
            ({"betas": (0.9, 1.0)}, "--betas"),
            ({"recompute_ratio": 1.1}, "--recompute-ratio"),
            ({"parameters": ["weight", "weight"]}, "unique"),
            ({"grad_atol": math.nan}, "tolerances"),
        ],
    )
    def test_rejects_invalid_values(
        self, training_script: dict[str, Any], updates: dict[str, Any], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            training_script["validate_options"](_options(**updates))

    def test_accepts_default_shape(self, training_script: dict[str, Any]) -> None:
        training_script["validate_options"](_options())


class _MappedModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Linear(2, 2, bias=False)
        self.right = torch.nn.Linear(2, 2, bias=False)

    def to_hf_key_list(self, name: str) -> list[str]:
        return [f"model.{name}"]


class TestParameterMapping:
    def test_preserves_requested_order(self, training_script: dict[str, Any]) -> None:
        model = _MappedModule()
        mapping = training_script["resolve_parameter_mapping"](
            model, ["right.weight", "left.weight"]
        )
        assert list(mapping.items()) == [
            ("right.weight", "model.right.weight"),
            ("left.weight", "model.left.weight"),
        ]

    def test_rejects_unknown_parameter(self, training_script: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="unknown XTuner parameters"):
            training_script["resolve_parameter_mapping"](_MappedModule(), ["missing.weight"])


class TestHFMasterWeights:
    def test_adamw_updates_fp32_master_and_copies_bf16_parameter(
        self, training_script: dict[str, Any]
    ) -> None:
        model = _MappedModule().to(dtype=torch.bfloat16)
        selected, masters, optimizer = training_script["_prepare_hf_training"](
            model, {"left.weight": "left.weight"}, _options()
        )
        initial = masters["left.weight"].detach().clone()
        expected = torch.nn.Parameter(initial.clone())
        expected_optimizer = torch.optim.AdamW(
            [expected], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False, fused=False
        )
        gradient = torch.full_like(selected["left.weight"], 0.25)
        selected["left.weight"].grad = gradient
        expected.grad = gradient.float()

        captured = training_script["_copy_master_grads_and_step"](selected, masters, optimizer)
        expected_optimizer.step()

        torch.testing.assert_close(captured["left.weight"], gradient.float(), rtol=0, atol=0)
        torch.testing.assert_close(masters["left.weight"], expected, rtol=0, atol=0)
        torch.testing.assert_close(selected["left.weight"], expected.to(torch.bfloat16), rtol=0, atol=0)
        assert not model.right.weight.requires_grad


class TestMetricGates:
    def test_shape_mismatch_fails_closed(self, training_script: dict[str, Any]) -> None:
        assert math.isinf(training_script["tensor_max_abs"](torch.zeros(2), torch.zeros(3)))

    def test_reports_non_finite_and_exceeded_metrics(self, training_script: dict[str, Any]) -> None:
        metrics = {"loss": math.nan, "gradient": 2.0}
        tolerances = {"loss": 0.0, "gradient": 1.0}
        assert training_script["gate_step_metrics"](metrics, tolerances) == [
            "loss is not finite",
            "gradient 2.000e+00>1.000e+00",
        ]
