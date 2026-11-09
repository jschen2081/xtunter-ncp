# Copyright (c) OpenMMLab. All rights reserved.
"""CPU tests for Muon/AdamW parameter decay overrides."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from torch import nn

from xtuner.v1.config.optim import MuonConfig


class _DecayModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_routed_experts=1)
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.projection = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.conv1d = nn.Conv1d(4, 4, 3, groups=4, bias=False)

    def trainable_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return list(self.named_parameters())


def _build_optimizer(model: nn.Module, cfg: MuonConfig):
    with (
        patch("xtuner.v1.config.optim.dist.get_rank", return_value=1),
        patch("xtuner.v1.optim.muon.Muon._init_moe_subgroups"),
    ):
        return cfg.build(model)


def _parameter_groups(model: nn.Module, optimizer) -> dict[int, dict]:
    assignments = [(p, group) for group in optimizer.param_groups for p in group["params"]]
    groups = {id(p): group for p, group in assignments}
    assert len(groups) == len(assignments) == len(list(model.parameters()))
    return groups


class TestMuonWeightDecayOverrides:
    @pytest.mark.parametrize(
        ("overrides", "expected_decay"),
        [
            ({}, 0.1),
            ({r"(?:^|\.)lm_head\.": 0.0}, 0.0),
            ({r"(?:^|\.)lm_head\.": 0.03}, 0.03),
        ],
    )
    def test_adamw_decay_override(self, overrides: dict[str, float], expected_decay: float) -> None:
        model = _DecayModel()
        cfg = MuonConfig(
            weight_decay=0.1,
            embedding_weight_decay=0.0,
            one_d_weight_decay=0.0,
            weight_decay_overrides=overrides,
        )
        optimizer = _build_optimizer(model, cfg)
        groups = _parameter_groups(model, optimizer)
        expected = {
            "embed_tokens.weight": ("adamw", 0.0),
            "lm_head.weight": ("adamw", expected_decay),
            "projection.weight": ("muon", 0.1),
            "projection.bias": ("adamw", 0.0),
            "norm.weight": ("adamw", 0.0),
            "norm.bias": ("adamw", 0.0),
            "conv1d.weight": ("adamw", 0.1),
        }
        for name, param in model.named_parameters():
            group = groups[id(param)]
            assert (group["algorithm"], group["weight_decay"]) == expected[name], name

    def test_override_applies_to_muon_and_first_match_wins(self) -> None:
        model = _DecayModel()
        cfg = MuonConfig(
            weight_decay=0.1,
            embedding_weight_decay=0.0,
            one_d_weight_decay=0.0,
            weight_decay_overrides={
                r"^projection\.weight$": 0.03,
                r"^projection\.": 0.2,
            },
        )
        optimizer = _build_optimizer(model, cfg)
        groups = _parameter_groups(model, optimizer)

        projection_weight = groups[id(model.projection.weight)]
        assert projection_weight["algorithm"] == "muon"
        assert projection_weight["weight_decay"] == 0.03
        assert groups[id(model.projection.bias)]["weight_decay"] == 0.2
        assert groups[id(model.lm_head.weight)]["weight_decay"] == 0.1
        assert groups[id(model.norm.weight)]["weight_decay"] == 0.0

    def test_shared_parameter_rejects_conflicting_alias_overrides(self) -> None:
        model = _DecayModel()
        model.lm_head.weight = model.embed_tokens.weight
        cfg = MuonConfig(
            weight_decay_overrides={
                r"^embed_tokens\.": 0.0,
                r"^lm_head\.": 0.1,
            }
        )

        with pytest.raises(ValueError, match="Conflicting weight decay overrides"):
            _build_optimizer(model, cfg)
