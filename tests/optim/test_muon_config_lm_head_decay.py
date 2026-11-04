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


class TestMuonLMHeadDecay:
    @pytest.mark.parametrize("head_decay", [None, 0.0, 0.03])
    def test_lm_head_decay_override(self, head_decay: float | None) -> None:
        model = _DecayModel()
        cfg = MuonConfig(
            weight_decay=0.1,
            embedding_weight_decay=0.0,
            one_d_weight_decay=0.0,
            lm_head_weight_decay=head_decay,
        )
        # Parameter grouping is CPU-only and does not require collectives.
        with (
            patch("xtuner.v1.config.optim.dist.get_rank", return_value=1),
            patch("xtuner.v1.optim.muon.Muon._init_moe_subgroups"),
        ):
            optimizer = cfg.build(model)

        assignments = [(p, group) for group in optimizer.param_groups for p in group["params"]]
        groups = {id(p): group for p, group in assignments}
        assert len(groups) == len(assignments) == len(list(model.parameters()))
        expected = {
            "embed_tokens.weight": ("adamw", 0.0),
            "lm_head.weight": ("adamw", 0.1 if head_decay is None else head_decay),
            "projection.weight": ("muon", 0.1),
            "projection.bias": ("adamw", 0.0),
            "norm.weight": ("adamw", 0.0),
            "norm.bias": ("adamw", 0.0),
            "conv1d.weight": ("adamw", 0.1),
        }
        for name, param in model.named_parameters():
            group = groups[id(param)]
            assert (group["algorithm"], group["weight_decay"]) == expected[name], name
