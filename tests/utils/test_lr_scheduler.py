"""CPU tests for native schedules across warmup, stable, decay and resume.

The direct source import avoids GPU-only package initializers; all numerical
code exercised here is the real library implementation.
"""

import ast
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("native_lr_scheduler", ROOT / "xtuner/v1/utils/lr_scheduler.py")
LIBRARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIBRARY)
build_lr_scheduler = LIBRARY.build_lr_scheduler


def make_config(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**{
        "lr_type": "wsd", "warmup_ratio": 0.03, "lr_min": 8e-5,
        "warmup_steps": 4, "decay_steps": 40, "wsd_decay_steps": 8,
    } | kwargs)


def make_optimizer() -> torch.optim.SGD:
    return torch.optim.SGD([
        {"params": [torch.nn.Parameter(torch.ones(()))], "lr": 8e-4},
        {"params": [torch.nn.Parameter(torch.ones(()))], "lr": 4e-4},
    ], lr=8e-4)


def reference_lr(step: int, peak: float, config: SimpleNamespace) -> float:
    # Analytic phase definitions, independent of the scheduler implementation.
    if step < config.warmup_steps:
        return peak * step / config.warmup_steps
    if step >= config.decay_steps:
        return config.lr_min
    start = config.decay_steps - config.wsd_decay_steps if config.lr_type == "wsd" else config.warmup_steps
    if step <= start:
        return peak
    progress = (step - start) / (config.decay_steps - start)
    return config.lr_min + (peak - config.lr_min) * (1 + math.cos(math.pi * progress)) / 2


class TestLRScheduler:
    @pytest.mark.parametrize("style", ["wsd", "cosine"])
    @pytest.mark.parametrize("warmup", [0, 4])
    def test_every_phase_and_post_decay_floor(self, style: str, warmup: int) -> None:
        config = make_config(lr_type=style, warmup_steps=warmup,
                             wsd_decay_steps=8 if style == "wsd" else None)
        optimizer = make_optimizer()
        scheduler = build_lr_scheduler(optimizer, config, total_steps=2)
        # The full horizon is 40 despite a two-step smoke run.
        for step in range(51):
            expected = [reference_lr(step, peak, config) for peak in (8e-4, 4e-4)]
            assert scheduler.get_last_lr() == pytest.approx(expected, rel=1e-13, abs=1e-18)
            assert [group["lr"] for group in optimizer.param_groups] == scheduler.get_last_lr()
            # Values above are the LR actually used by this update.
            optimizer.step()
            scheduler.step()

    @pytest.mark.parametrize("style", ["wsd", "cosine"])
    @pytest.mark.parametrize("restore_step", [2, 12, 35, 44])
    def test_resume_in_every_phase(self, style: str, restore_step: int) -> None:
        config = make_config(lr_type=style, wsd_decay_steps=8 if style == "wsd" else None)
        original = make_optimizer()
        scheduler = build_lr_scheduler(original, config, total_steps=2)
        for _ in range(restore_step):
            original.step()
            scheduler.step()
        resumed = make_optimizer()
        resumed_scheduler = build_lr_scheduler(resumed, config, total_steps=2)
        resumed_scheduler.load_state_dict(scheduler.state_dict())
        resumed.load_state_dict(original.state_dict())
        for _ in range(20):
            assert [g["lr"] for g in resumed.param_groups] == [g["lr"] for g in original.param_groups]
            assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()
            original.step()
            resumed.step()
            scheduler.step()
            resumed_scheduler.step()

    @pytest.mark.parametrize("style", ["constant", "linear", "cosine"])
    def test_legacy_recipes_and_checkpoint_format_unchanged(self, style: str) -> None:
        config = make_config(lr_type=style, warmup_steps=None, decay_steps=None,
                             wsd_decay_steps=None, warmup_ratio=0.1)
        actual, reference = make_optimizer(), make_optimizer()
        scheduler = build_lr_scheduler(actual, config, total_steps=40)
        warmup = LambdaLR(reference, lambda step: step / 4 if step < 4 else 1)
        if style == "constant":
            decay = LambdaLR(reference, lambda step: 1)
        elif style == "linear":
            decay = LinearLR(reference, start_factor=1, end_factor=0.1, total_iters=36)
        else:
            decay = CosineAnnealingLR(reference, T_max=36, eta_min=8e-5)
        legacy = SequentialLR(reference, [warmup, decay], milestones=[4])
        assert isinstance(scheduler, SequentialLR)
        for _ in range(41):
            assert scheduler.state_dict() == legacy.state_dict()
            assert scheduler.get_last_lr() == legacy.get_last_lr()
            actual.step()
            reference.step()
            scheduler.step()
            legacy.step()

    @pytest.mark.parametrize("fields", [
        {"warmup_steps": -1}, {"warmup_steps": 40}, {"decay_steps": 0},
        {"wsd_decay_steps": None}, {"wsd_decay_steps": 0}, {"wsd_decay_steps": 37},
        {"lr_min": -1}, {"lr_min": 1},
    ])
    def test_invalid_schedule_rejected(self, fields: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            build_lr_scheduler(make_optimizer(), make_config(**fields), total_steps=2)

    def test_warmup_ratio_uses_full_horizon(self) -> None:
        optimizer = make_optimizer()
        scheduler = build_lr_scheduler(optimizer, make_config(warmup_steps=None, warmup_ratio=0.1), 2)
        optimizer.step()
        scheduler.step()
        assert scheduler.get_last_lr()[0] == pytest.approx(8e-4 / 4)

    @pytest.mark.parametrize("style", ["wsd", "cosine"])
    @pytest.mark.parametrize("gbs", [8, 128])
    def test_full_production_horizon_against_megatron(self, style: str, gbs: int) -> None:
        megatron_root = Path(os.environ.get("MEGATRON_ROOT", ROOT.parent / "megatron_ncp_qwen35"))
        path = megatron_root / "megatron/core/optimizer_param_scheduler.py"
        if not path.exists():
            pytest.skip("Set MEGATRON_ROOT to additionally compare the real Megatron scheduler")
        tree = ast.parse(path.read_text())
        # Only replace a logging import; retain the complete reference scheduler.
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module == "megatron.core.utils"
        )]
        namespace = {"log_single_rank": lambda *args, **kwargs: None}
        exec(compile(tree, str(path), "exec"), namespace)
        config = make_config(lr_type=style, warmup_steps=2000, decay_steps=143052,
                             wsd_decay_steps=14306 if style == "wsd" else None)
        optimizer, reference = make_optimizer(), make_optimizer()
        for group in reference.param_groups:
            group["max_lr"] = group["lr"]
        mg = namespace["OptimizerParamScheduler"](
            reference, init_lr=0, max_lr=8e-4, min_lr=8e-5,
            lr_warmup_steps=2000 * gbs, lr_decay_steps=143052 * gbs,
            lr_decay_style="WSD" if style == "wsd" else "cosine",
            start_wd=0, end_wd=0, wd_incr_steps=143052 * gbs, wd_incr_style="constant",
            wsd_decay_steps=14306 * gbs if style == "wsd" else None, lr_wsd_decay_style="cosine",
        )
        scheduler = build_lr_scheduler(optimizer, config, total_steps=100)
        # Exercise every LR through warmup, plateau, decay and beyond the horizon.
        for _ in range(143055):
            assert [g["lr"] for g in optimizer.param_groups] == [g["lr"] for g in reference.param_groups]
            optimizer.step()
            scheduler.step()
            mg.step(increment=gbs)
