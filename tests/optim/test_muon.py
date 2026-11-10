# Copyright (c) OpenMMLab. All rights reserved.
"""Muon optimizer tests.

1. TestNewtonSchulz     — Triton vs PyTorch Newton-Schulz precision.
2. Parameter overrides — Algorithm and weight-decay parameter-group overrides.
3. TestMuonSingleGPU   — Single-GPU correctness: production Muon must match the
                          reference within tight tolerances (fp32, no comms).
4. TestMuonFSDP        — End-to-end correctness: one Muon step on a fully-sharded
                          model must match a single-process reference for every
                          parameter, covering all four communication strategies
                          (all_to_all, agrs, local, subgroup_allgather).
"""

import copy
import math
import os
from types import SimpleNamespace
from typing import Callable, Literal, overload
from unittest.mock import patch

import parametrize
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from xtuner._testing.testcase import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.config.optim import MuonConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.model.base import BaseModel, XTunerBaseModelConfig
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseMLP
from xtuner.v1.optim.muon import (
    Muon,
    _get_muon_lr_ratio,
    _muon_qkv_newton_schulz,
    _muonsplit_newton_schulz,
    zeropower_via_newtonschulz5,
)


# ─── Test: Newton-Schulz functions ───────────────────────────────────────────


class TestNewtonSchulz(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 1

    def test_triton_vs_pytorch(self):
        """Triton and PyTorch NS implementations should produce similar results."""
        self.create_pg("cuda")
        from xtuner.v1.optim.newton_schulz_triton import newton_schulz_triton

        # Use a single representative shape per category to limit autotune overhead.
        # Autotune results are cached on disk, so subsequent runs are fast.
        test_cases = [
            (1, 512, 1536),  # regular matrix (M < N)
            (4, 256, 256),  # MoE experts (square)
        ]

        for num_experts, M, N in test_cases:
            G = torch.randn(num_experts * M, N, device="cuda", dtype=torch.float32)

            result1 = zeropower_via_newtonschulz5(G, epsilon=1e-7, num_experts=num_experts)
            result2 = newton_schulz_triton(G, epsilon=1e-7, num_experts=num_experts)

            assert not torch.isnan(result1).any()
            assert not torch.isnan(result2).any()
            torch.testing.assert_close(result1, result2, atol=3e-2, rtol=3e-2)

    def test_muonsplit_matches_independent_blocks(self):
        self.create_pg("cuda")
        split_sizes = (6, 2, 6, 2)
        logical_rows = sum(split_sizes)
        G = torch.randn(logical_rows + 2, 4, device="cuda", dtype=torch.float32)

        result = _muonsplit_newton_schulz(
            G,
            epsilon=1e-7,
            num_experts=1,
            newton_schulz_func=zeropower_via_newtonschulz5,
            split_sizes=split_sizes,
            adjust_lr="rms_norm",
        )

        expected_chunks = []
        offset = 0
        for size in split_sizes:
            chunk = G.narrow(0, offset, size)
            chunk = zeropower_via_newtonschulz5(chunk, epsilon=1e-7)
            chunk.mul_(0.2 * math.sqrt(max(size, G.size(1))))
            expected_chunks.append(chunk)
            offset += size
        expected_chunks.append(torch.zeros_like(G[logical_rows:]))
        expected = torch.cat(expected_chunks)

        torch.testing.assert_close(result, expected)


def test_lr_ratio_modes():
    assert _get_muon_lr_ratio(4096, 4096, "spectral_norm") == 1
    assert _get_muon_lr_ratio(22016, 4096, "spectral_norm") == math.sqrt(22016 / 4096)
    assert _get_muon_lr_ratio(4096, 11008, "spectral_norm") == math.sqrt(4096 / 11008)
    assert _get_muon_lr_ratio(4096, 4096, "rms_norm") == 0.2 * 64
    assert _get_muon_lr_ratio(4096, 4096, "rms_norm", extra_scale_factor=1) == 64
    assert _get_muon_lr_ratio(22016, 4096, "rms_norm", extra_scale_factor=1) == math.sqrt(22016)
    assert _get_muon_lr_ratio(4096, 11008, "rms_norm", extra_scale_factor=1) == math.sqrt(11008)


def test_qkv_split_gathers_before_orthogonalize():
    """Head-interleaved Q/K/V are gathered, then scaled on the gathered shape."""
    seen: list[tuple[int, int]] = []

    def identity(matrix, epsilon, num_experts):
        assert num_experts == 1
        seen.append(tuple(matrix.shape))
        return matrix.clone()

    groups, head_dim, hidden = 2, 2, 4
    rows = groups * 3 * head_dim
    source = torch.arange(rows * hidden, dtype=torch.float32).reshape(rows, hidden)
    result = _muon_qkv_newton_schulz(
        source,
        epsilon=1e-7,
        num_experts=1,
        newton_schulz_func=identity,
        qkv_split_shapes=(head_dim, head_dim, head_dim),
        adjust_lr="rms_norm",
        extra_scale_factor=1,
    )

    gathered_rows = groups * head_dim
    assert seen == [(gathered_rows, hidden)] * 3
    scale = math.sqrt(max(gathered_rows, hidden))
    grouped = source.view(groups, 3 * head_dim, hidden)
    parts = [
        part.reshape(gathered_rows, hidden) * scale for part in grouped.split((head_dim, head_dim, head_dim), dim=1)
    ]
    expected = torch.cat([part.view(groups, head_dim, hidden) for part in parts], dim=1).reshape(rows, hidden)
    torch.testing.assert_close(result, expected)


class _ParamOverrideModel(nn.Module):
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


def _build_param_override_optimizer(model: nn.Module, cfg: MuonConfig):
    with (
        patch("xtuner.v1.config.optim.dist.get_rank", return_value=1),
        patch("xtuner.v1.optim.muon.Muon._init_moe_subgroups"),
    ):
        return cfg.build(model)


def _param_override_groups(model: nn.Module, optimizer) -> dict[int, dict]:
    assignments = [(p, group) for group in optimizer.param_groups for p in group["params"]]
    groups = {id(p): group for p, group in assignments}
    assert len(groups) == len(assignments) == len(list(model.parameters()))
    return groups


class TestMuonParamOverrides:
    @pytest.mark.parametrize(
        ("overrides", "expected_updates"),
        [
            ({}, {}),
            (
                {r"(?:^|\.)lm_head\.": {"weight_decay": 0.0}},
                {"lm_head.weight": ("adamw", 0.0)},
            ),
            (
                {r"(?:^|\.)lm_head\.": {"algorithm": "muon", "weight_decay": 0.03}},
                {"lm_head.weight": ("muon", 0.03)},
            ),
            (
                {
                    r"^projection\.weight$": {"algorithm": "adamw", "weight_decay": 0.03},
                    r"^projection\.": {"weight_decay": 0.2},
                },
                {"projection.weight": ("adamw", 0.03), "projection.bias": ("adamw", 0.2)},
            ),
            (
                {r"^embed_tokens\.": {"weight_decay": 0.2}},
                {"embed_tokens.weight": ("adamw", 0.2)},
            ),
        ],
    )
    def test_param_overrides(self, overrides: dict[str, dict], expected_updates: dict[str, tuple[str, float]]) -> None:
        model = _ParamOverrideModel()
        cfg = MuonConfig(
            weight_decay=0.1,
            embedding_weight_decay=0.0,
            one_d_weight_decay=0.0,
            param_overrides=overrides,
        )
        optimizer = _build_param_override_optimizer(model, cfg)
        groups = _param_override_groups(model, optimizer)
        expected = {
            "embed_tokens.weight": ("adamw", 0.0),
            "lm_head.weight": ("adamw", 0.1),
            "projection.weight": ("muon", 0.1),
            "projection.bias": ("adamw", 0.0),
            "norm.weight": ("adamw", 0.0),
            "norm.bias": ("adamw", 0.0),
            "conv1d.weight": ("adamw", 0.1),
        }
        expected.update(expected_updates)
        for name, param in model.named_parameters():
            group = groups[id(param)]
            assert (group["algorithm"], group["weight_decay"]) == expected[name], name

    @pytest.mark.parametrize(
        ("overrides", "field"),
        [
            (
                {
                    r"^embed_tokens\.": {"weight_decay": 0.0},
                    r"^lm_head\.": {"weight_decay": 0.1},
                },
                "weight_decay",
            ),
            (
                {
                    r"^embed_tokens\.": {"algorithm": "adamw"},
                    r"^lm_head\.": {"algorithm": "muon"},
                },
                "algorithm",
            ),
        ],
    )
    def test_shared_parameter_rejects_conflicting_aliases(self, overrides: dict[str, dict], field: str) -> None:
        model = _ParamOverrideModel()
        model.lm_head.weight = model.embed_tokens.weight
        cfg = MuonConfig(param_overrides=overrides)

        with pytest.raises(ValueError, match=f"Conflicting parameter overrides.*{field}"):
            _build_param_override_optimizer(model, cfg)

    def test_shared_parameter_merges_compatible_aliases(self) -> None:
        model = _ParamOverrideModel()
        model.lm_head.weight = model.embed_tokens.weight
        cfg = MuonConfig(
            param_overrides={
                r"^embed_tokens\.": {"algorithm": "adamw"},
                r"^lm_head\.": {"weight_decay": 0.03},
            }
        )

        optimizer = _build_param_override_optimizer(model, cfg)
        group = _param_override_groups(model, optimizer)[id(model.embed_tokens.weight)]
        assert (group["algorithm"], group["weight_decay"]) == ("adamw", 0.03)

    def test_model_metadata_rejects_conflicting_algorithm(self) -> None:
        model = _ParamOverrideModel()
        model.get_muon_metadata = lambda: {
            "param_options": {model.projection.weight: {"algorithm": "muon", "weight_decay": 0.1}}
        }
        cfg = MuonConfig(
            concept_muon=True,
            param_overrides={r"^projection\.weight$": {"algorithm": "adamw"}},
        )

        with pytest.raises(ValueError, match="Conflicting optimizer algorithms"):
            _build_param_override_optimizer(model, cfg)

    def test_user_decay_overrides_model_metadata_default(self) -> None:
        model = _ParamOverrideModel()
        model.get_muon_metadata = lambda: {
            "param_options": {model.conv1d.weight: {"algorithm": "adamw", "weight_decay": 0.1}}
        }
        cfg = MuonConfig(
            concept_muon=True,
            param_overrides={r"^conv1d\.weight$": {"weight_decay": 0.03}},
        )

        optimizer = _build_param_override_optimizer(model, cfg)
        group = _param_override_groups(model, optimizer)[id(model.conv1d.weight)]
        assert (group["algorithm"], group["weight_decay"]) == ("adamw", 0.03)

    def test_legacy_weight_decay_overrides_are_normalized(self) -> None:
        pattern = r"(?:^|\.)lm_head\."
        cfg = MuonConfig(weight_decay_overrides={pattern: 0.03})
        assert cfg.param_overrides[pattern].weight_decay == 0.03
        assert cfg.param_overrides[pattern].algorithm is None

        with pytest.raises(ValueError, match="cannot be used together"):
            MuonConfig(
                weight_decay_overrides={pattern: 0.0},
                param_overrides={pattern: {"weight_decay": 0.03}},
            )

    def test_embedding_and_one_d_defaults_conflict(self) -> None:
        model = _ParamOverrideModel()
        model.embed_tokens = nn.Embedding(4, 1)
        cfg = MuonConfig(embedding_weight_decay=0.0, one_d_weight_decay=0.1)

        with pytest.raises(ValueError, match="matches both embedding"):
            _build_param_override_optimizer(model, cfg)

    def test_invalid_override_pattern(self) -> None:
        with pytest.raises(ValueError, match="Invalid parameter override pattern"):
            MuonConfig(param_overrides={"[": {"weight_decay": 0.0}})

    def test_muon_override_rejects_vector(self) -> None:
        model = _ParamOverrideModel()
        cfg = MuonConfig(param_overrides={r"^norm\.weight$": {"algorithm": "muon"}})

        with pytest.raises(ValueError, match="Muon requires at least 2D parameters"):
            _build_param_override_optimizer(model, cfg)


# ─── Model for end-to-end tests ──────────────────────────────────────────────


class ToyMoEModelConfig(XTunerBaseModelConfig):
    vocab_size: int = 64
    hidden_size: int = 32
    intermediate_size: int = 64
    n_routed_experts: int = 2
    moe_intermediate_size: int = 16

    def build(self) -> "ToyMoEModel":
        return ToyMoEModel(self)


class ToyMoEModel(BaseModel):
    """Minimal model exercising all Muon param categories and comm strategies.

    Forward skips top-k routing for simplicity — all experts are applied
    unconditionally, guaranteeing every expert weight receives non-zero
    gradients so the optimizer update is exercised on all of them.

    With world_size=4 and n_routed_experts=2, the comm strategy coverage is:
      all_to_all          — fcs (4 same-shape params fill a full batch)
      agrs                — mlp params (remainder batches, grouped by shape)
      local               — fused_w1w3 (ns_experts=4, 4%4==0, 1 expert/rank)
      subgroup_allgather  — fused_w2 (ns_experts=2, 4%2==0, sg_size=2)

    Parameter layout:
      embed_tokens.weight       (64, 32)  — AdamW (name-matched)
      fcs.{0,1,2,3}.weight      (32, 32)  — Muon regular
      mlp.gate_proj.weight      (64, 32)  — Muon regular
      mlp.up_proj.weight        (64, 32)  — Muon regular
      mlp.down_proj.weight      (32, 64)  — Muon regular
      norm.weight               (32,)     — AdamW (1D)
      norm.bias                 (32,)     — AdamW (1D)
      fused_w1w3.weight         (64, 32)  — Muon MoE, num_experts = 2 * 2 = 4
      fused_w2.weight           (64, 16)  — Muon MoE, num_experts = 2
      lm_head.weight            (64, 32)  — AdamW (name-matched)
    """

    config: ToyMoEModelConfig  # type: ignore

    def __init__(self, config: ToyMoEModelConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # 4 linear layers to fill a full batch (Muon regular, all_to_all)
        self.fcs = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size, bias=False) for _ in range(4)])
        # DenseMLP — 3 more Muon regular params (remainder batch, agrs)
        self.mlp = DenseMLP(
            hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, hidden_act="silu"
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        # fused_w1w3: (n_routed_experts * 2 * moe_intermediate_size, hidden_size)
        self.fused_w1w3 = nn.Linear(
            config.hidden_size, config.n_routed_experts * 2 * config.moe_intermediate_size, bias=False
        )
        # fused_w2: (n_routed_experts * hidden_size, moe_intermediate_size)
        self.fused_w2 = nn.Linear(
            config.moe_intermediate_size, config.n_routed_experts * config.hidden_size, bias=False
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Trivial forward that touches all params and returns a scalar loss."""
        h = self.embed_tokens(input_ids)
        h = h + sum(fc(h) for fc in self.fcs)
        h = self.norm(h)
        h = h + self.mlp(h)
        # Simplified MoE: split → SiLU gate → w2 (no routing, all experts applied)
        w1w3_out = self.fused_w1w3(h)
        gate, up = w1w3_out.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        # Per-expert w2 with explicit slices for DTensor-compatible autograd.
        hidden = hidden.view(*hidden.shape[:-1], self.config.n_routed_experts, self.config.moe_intermediate_size)
        w2 = self.fused_w2.weight
        expert_out = torch.stack(
            [
                F.linear(
                    hidden[..., expert, :],
                    w2.narrow(0, expert * self.config.hidden_size, self.config.hidden_size),
                )
                for expert in range(self.config.n_routed_experts)
            ],
            dim=-2,
        ).sum(-2)
        h = h + expert_out
        logits = self.lm_head(h)
        return logits.sum()

    @overload  # type:ignore
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor: ...  # type:ignore

    __call__ = nn.Module.__call__


# ─── Reference optimizer ─────────────────────────────────────────────────────


class ReferenceMuon(torch.optim.Optimizer):
    """Self-contained single-process reference for Muon + AdamW.

    Classifies parameters using the same name-based rules as MuonConfig.build():
    - 2D+ tensors (excluding embed_tokens, lm_head, effectively-1D) → Muon
    - Everything else → AdamW
    """

    FUSED_W1W3_PATTERNS = ("fused_w1w3",)
    OTHER_EXPERT_PATTERNS = ("fused_w2", "fused_w1", "fused_w3")

    NS_CONSTS = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    def __init__(
        self,
        model: ToyMoEModel,
        lr: float,
        mu: float,
        weight_decay: float,
        epsilon: float,
        betas: tuple[float, float] = (0.9, 0.95),
    ):
        n_routed_experts = model.config.n_routed_experts
        muon_groups: list[dict] = []
        adamw_params: list[nn.Parameter] = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_muon = p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name and p.numel() not in p.shape
            if is_muon:
                ne = self._get_num_experts(name, n_routed_experts)
                muon_groups.append(
                    {
                        "params": [p],
                        "type": "muon",
                        "num_experts": ne,
                        "base_lr": lr,
                        "lr": lr * self._rms_lr_ratio(p, ne),
                    }
                )
            else:
                adamw_params.append(p)

        super().__init__(
            muon_groups + [{"params": adamw_params, "type": "adamw", "base_lr": lr, "lr": lr}],
            {"mu": mu, "weight_decay": weight_decay, "epsilon": epsilon, "betas": betas},
        )

    @staticmethod
    def _rms_lr_ratio(p: nn.Parameter, num_experts: int) -> float:
        A = p.shape[-2] // num_experts
        B = p.shape[-1]
        return 0.2 * math.sqrt(max(A, B))

    @classmethod
    def _get_num_experts(cls, name: str, n_routed_experts: int) -> int:
        if any(pat in name for pat in cls.FUSED_W1W3_PATTERNS):
            return 2 * n_routed_experts
        elif any(pat in name for pat in cls.OTHER_EXPERT_PATTERNS):
            return n_routed_experts
        return 1

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["mu"]
            wd = group["weight_decay"]
            eps = group["epsilon"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if group["type"] == "muon":
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p.data, dtype=torch.float32)

                    mom = state["momentum"]
                    num_experts = group["num_experts"]
                    g = p.grad.to(dtype=torch.float32)

                    # Momentum + Nesterov
                    mom.mul_(mu).add_(g)
                    U = mom * mu + g

                    # Newton-Schulz orthogonalization
                    X = U.to(dtype=torch.bfloat16).float()
                    N = X.size(-1)
                    X = X.view(num_experts, -1, N)
                    need_transpose = X.size(-2) > X.size(-1)
                    if need_transpose:
                        X = X.mT
                    norm = X.norm(dim=(-2, -1), keepdim=True)
                    X = (X / torch.maximum(norm, torch.tensor(eps, device=X.device, dtype=X.dtype))).to(
                        dtype=torch.bfloat16
                    )
                    for a, b, c in self.NS_CONSTS:
                        A = X @ X.mT
                        B = b * A + c * (A @ A)
                        X = a * X + B @ X
                    if need_transpose:
                        X = X.mT
                    X = X.view(U.shape)

                    # Weight decay (uses base_lr, not adjusted_lr)
                    base_lr = group["base_lr"]
                    p.data.mul_(1 - base_lr * wd)
                    # Update: multiply in bf16 (matches production _foreach_mul_ on bf16)
                    X.mul_(lr)
                    p.data.sub_(X)

                else:  # AdamW
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)
                        state["step"] = 0

                    state["step"] += 1
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    g = p.grad.float()

                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]
                    m_hat = m / bc1
                    v_hat = v / bc2

                    # Decoupled weight decay
                    p.data.mul_(1 - lr * wd)
                    p.data.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)


# ─── Test: single-GPU correctness ───────────────────────────────────────────


def test_muon_clip_grad_mode_defaults_to_adamw_only():
    assert MuonConfig().clip_grad_mode == "adamw_only"


class TestMuonSingleGPU(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 1

    def test_muon_single_gpu_matches_reference(self):
        """On a single GPU, production Muon must match the reference for every
        parameter within tight tolerances (everything is fp32, no comms).

        fully_shard() is required even here: the production Muon optimizer
        assumes that parameters are DTensors and will fail on plain tensors.
        With world_size=1 it is a no-op sharding-wise.
        """
        self.create_pg("cuda")
        device = "cuda"

        LR = 0.01
        MU = 0.95
        WD = 0.01
        EPSILON = 1e-8
        BETAS = (0.9, 0.95)

        # ── Build two identical models ───────────────────────────────────────
        torch.manual_seed(42)
        config = ToyMoEModelConfig(compile_cfg=False)
        model = config.build().to(device)
        ref_model = copy.deepcopy(model)
        input_ids = torch.randint(0, config.vocab_size, (2, 8), device=device)

        # ── Reference path ───────────────────────────────────────────────────
        ref_loss = ref_model(input_ids)
        ref_loss.backward()
        ref_optim = ReferenceMuon(ref_model, lr=LR, mu=MU, weight_decay=WD, epsilon=EPSILON, betas=BETAS)
        ref_optim.step()

        # ── Production path ──────────────────────────────────────────────────
        fsdp_config = FSDPConfig(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            torch_compile=False,
        )
        model.fully_shard(fsdp_config)

        loss = model(input_ids)
        loss.backward()
        muon_config = MuonConfig(lr=LR, momentum=MU, weight_decay=WD, eps=EPSILON, betas=BETAS)
        optim = muon_config.build(model)
        optim.step()

        # ── Compare all parameters ───────────────────────────────────────────
        for (name, ref_p), (_, prod_p) in zip(ref_model.named_parameters(), model.named_parameters()):
            full = prod_p.data.full_tensor()  # type: ignore[attr-defined]
            abs_diff = (full - ref_p.data).abs()
            rel_diff = abs_diff / (ref_p.data.abs().clamp(min=1e-12))
            torch.testing.assert_close(
                full,
                ref_p.data,
                atol=1e-6,
                rtol=1e-5,
                msg=f"mismatch on '{name}': max_abs={abs_diff.max().item():.2e}, max_rel={rel_diff.max().item():.2e}",
            )

    def test_muon_step_runs_with_epsilon_as_a_plain_scalar(self):
        """One production Muon step must not depend on epsilon's device.

        Regression: the batched update path builds epsilon as a zero-dim CPU tensor, and
        Newton-Schulz compares it with ``clamp_min``, which requires both operands on the
        same device. Every Muon run died here on its first optimizer step. This test does
        not compare against ``ReferenceMuon`` -- that reference still normalizes in the
        pre-384d4d1f order, which is a separate disagreement -- it only pins that the step
        completes and really updates the parameters.
        """
        self.create_pg("cuda")

        config = ToyMoEModelConfig(compile_cfg=False)
        model = config.build().to("cuda")
        model.fully_shard(FSDPConfig(param_dtype=torch.float32, reduce_dtype=torch.float32, torch_compile=False))

        optimizer = MuonConfig(lr=0.01, momentum=0.95, weight_decay=0.01, eps=1e-8, betas=(0.9, 0.95)).build(model)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}

        loss = model(torch.randint(0, config.vocab_size, (2, 8), device="cuda"))
        loss.backward()
        optimizer.step()

        moved = [name for name, p in model.named_parameters() if not torch.equal(p.detach(), before[name])]
        assert moved, "the Muon step left every parameter unchanged"

    @parametrize.parametrize("clip_grad_mode", ["all", "adamw_only"])
    def test_clip_grad_norm_respects_config(self, clip_grad_mode: Literal["all", "adamw_only"]):
        self.create_pg("cuda")
        device = "cuda"
        model = ToyMoEModelConfig(compile_cfg=False).build().to(device)
        model.fully_shard(
            FSDPConfig(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                torch_compile=False,
            )
        )

        optim_config = MuonConfig(max_grad_norm=1.0, clip_grad_mode=clip_grad_mode)
        optimizer = optim_config.build(model)
        total_squared = 0
        clipped_squared = 0
        initial_grad_values = {}
        for group in optimizer.param_groups:
            is_muon = group["algorithm"] == "muon"
            expected_clip_grad = clip_grad_mode == "all" or not is_muon
            assert group["clip_grad"] is expected_clip_grad
            grad_value = 2.0 if is_muon else 1.0
            for param in group["params"]:
                param.grad = torch.full_like(param, grad_value)
                initial_grad_values[id(param)] = grad_value
                total_squared += param.numel() * grad_value**2
                if expected_clip_grad:
                    clipped_squared += param.numel() * grad_value**2
            # The engine must consume only the generic policy, not optimizer-specific metadata.
            group["algorithm"] = "test"

        engine = object.__new__(TrainEngine)
        engine.model = model
        engine.optimizer = optimizer
        engine.optim_cfg = optim_config
        grad_norm = engine.clip_grad_norm()

        torch.testing.assert_close(grad_norm, torch.tensor(math.sqrt(total_squared), device=device))
        clip_coef = min(1.0, optim_config.max_grad_norm / (math.sqrt(clipped_squared) + 1e-6))
        for group in optimizer.param_groups:
            for param in group["params"]:
                grad_value = initial_grad_values[id(param)]
                expected = grad_value * clip_coef if group["clip_grad"] else grad_value
                torch.testing.assert_close(param.grad.to_local(), torch.full_like(param.grad.to_local(), expected))

    @parametrize.parametrize("muon_clip_grad", [True, False])
    def test_clip_grad_policy_is_preserved_on_load(self, muon_clip_grad: bool):
        muon_param = nn.Parameter(torch.empty(4, 4))
        adamw_param = nn.Parameter(torch.empty(4))
        optimizer = Muon(
            [
                {"params": [muon_param], "clip_grad": muon_clip_grad},
                {"params": [adamw_param], "algorithm": "adamw"},
            ]
        )
        expected_policy = [muon_clip_grad, True]
        assert [group["clip_grad"] for group in optimizer.param_groups] == expected_policy

        state_dict = optimizer.state_dict()
        assert all("clip_grad" not in group for group in state_dict["param_groups"])

        loaded_muon_clip_grad = not muon_clip_grad
        state_dict["param_groups"][0]["clip_grad"] = loaded_muon_clip_grad
        state_dict["param_groups"][1]["clip_grad"] = False
        optimizer.load_state_dict(state_dict)
        assert [group["clip_grad"] for group in optimizer.param_groups] == expected_policy
        assert [group["clip_grad"] for group in state_dict["param_groups"]] == [loaded_muon_clip_grad, False]

        optimizer.add_param_group({"params": [nn.Parameter(torch.empty(4, 4))]})
        assert optimizer.param_groups[-1]["clip_grad"] is False


# ─── Test: ConceptMuon gate-state checkpointing ─────────────────────────────


def test_gate_adamw_state_survives_checkpoint():
    """Gate AdamW rows keep parameter-shaped m/v state that round-trips through
    state_dict without changing subsequent updates.

    Regression test: gate states were previously sized to the local gate-row
    slice and created only on ranks holding those rows, so distributed
    checkpoints dropped or misaligned them.
    """
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from xtuner.v1.optim.concept_qwen35_muon import ConceptMuon

    initialized_here = False
    if not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29571")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        initialized_here = True

    try:
        mesh = DeviceMesh("cpu", [0])
        torch.manual_seed(0)
        # BF16 parameters exercise the FP32 optimizer-state path.
        init = torch.randn(8, 4).to(torch.bfloat16)
        grad = torch.randn(8, 4).to(torch.bfloat16)

        def make_param() -> nn.Parameter:
            return nn.Parameter(distribute_tensor(init.clone(), mesh, placements=[Shard(0)]))

        def make_optimizer(param: nn.Parameter) -> ConceptMuon:
            return ConceptMuon([param], muon_split_sizes={param: (6,)}, muon_adamw_ranges={param: (6, 8)})

        def step(optimizer: ConceptMuon, param: nn.Parameter) -> None:
            param.grad = distribute_tensor(grad.clone(), mesh, placements=[Shard(0)])
            optimizer.step()

        # Uninterrupted reference: three steps.
        ref_param = make_param()
        ref_optimizer = make_optimizer(ref_param)
        for _ in range(3):
            step(ref_optimizer, ref_param)

        # Interrupted run: two steps, checkpoint, restore into a fresh optimizer.
        param = make_param()
        optimizer = make_optimizer(param)
        for _ in range(2):
            step(optimizer, param)

        state_dict = optimizer.state_dict()
        state = next(iter(state_dict["state"].values()))
        # States follow the fused parameter's sharding, not the gate-row slice,
        # and optimizer state stays FP32 while parameters may be low-precision.
        assert state["gate_momentum"].shape == param.shape
        assert state["gate_momentum"].dtype == torch.float32
        assert state["gate_variance"].shape == param.shape
        assert state["gate_variance"].dtype == torch.float32
        assert state["momentum"].dtype == torch.float32
        assert state["gate_step"] == 2

        restored = make_optimizer(param)
        restored.load_state_dict(state_dict)
        step(restored, param)

        ref_state = ref_optimizer.state[ref_param]
        restored_state = restored.state[param]
        torch.testing.assert_close(param.to_local(), ref_param.to_local(), atol=0, rtol=0)
        torch.testing.assert_close(
            restored_state["momentum"].to_local(), ref_state["momentum"].to_local(), atol=0, rtol=0
        )
        torch.testing.assert_close(
            restored_state["gate_momentum"].to_local(),
            ref_state["gate_momentum"].to_local(),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            restored_state["gate_variance"].to_local(),
            ref_state["gate_variance"].to_local(),
            atol=0,
            rtol=0,
        )
        assert restored_state["gate_step"] == ref_state["gate_step"] == 3
    finally:
        if initialized_here:
            dist.destroy_process_group()


def test_load_state_dict_rejects_unsharded_state_for_sharded_param():
    """A full (unsharded) checkpoint tensor for a sharded FP32 state must raise
    instead of being silently treated as the local shard."""
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from xtuner.v1.optim.concept_qwen35_muon import ConceptMuon

    initialized_here = False
    if not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29572")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        initialized_here = True

    try:
        mesh = DeviceMesh("cpu", [0])
        torch.manual_seed(0)
        param = nn.Parameter(distribute_tensor(torch.randn(8, 4).to(torch.bfloat16), mesh, placements=[Shard(0)]))
        optimizer = ConceptMuon([param], muon_split_sizes={param: (6,)}, muon_adamw_ranges={param: (6, 8)})
        param.grad = distribute_tensor(torch.randn(8, 4).to(torch.bfloat16), mesh, placements=[Shard(0)])
        optimizer.step()

        state_dict = optimizer.state_dict()
        state = next(iter(state_dict["state"].values()))
        # Simulate a checkpoint that stored the full tensor instead of the shard.
        state["gate_momentum"] = state["gate_momentum"].to_local().clone()

        restored = ConceptMuon([param], muon_split_sizes={param: (6,)}, muon_adamw_ranges={param: (6, 8)})
        try:
            restored.load_state_dict(state_dict)
        except ValueError as error:
            assert "cross-topology" in str(error)
        else:
            raise AssertionError("load_state_dict accepted an unsharded tensor for a sharded state")
    finally:
        if initialized_here:
            dist.destroy_process_group()


def test_load_state_dict_rejects_legacy_gate_keys():
    """Legacy MegatronMuon production_gate_* state must be rejected loudly
    instead of being silently reinitialized on the next step."""
    from xtuner.v1.optim.concept_qwen35_muon import ConceptMuon

    torch.manual_seed(0)
    param = nn.Parameter(torch.randn(8, 4).to(torch.bfloat16))
    optimizer = ConceptMuon([param], muon_split_sizes={param: (6,)}, muon_adamw_ranges={param: (6, 8)})

    state_dict = optimizer.state_dict()
    entry = next(iter(state_dict["state"].values()))
    entry["production_gate_m"] = torch.zeros(2, 4)
    entry["production_gate_step"] = 5

    restored = ConceptMuon([param], muon_split_sizes={param: (6,)}, muon_adamw_ranges={param: (6, 8)})
    try:
        restored.load_state_dict(state_dict)
    except ValueError as error:
        assert "legacy" in str(error)
    else:
        raise AssertionError("load_state_dict accepted legacy production_gate_* state")


def test_load_state_dict_promotes_low_precision_momentum():
    """A BF16 checkpoint momentum under optim_state_dtype=float32 is promoted
    to FP32 instead of being silently kept in BF16."""
    from xtuner.v1.optim.concept_qwen35_muon import ConceptMuon

    torch.manual_seed(0)
    param = nn.Parameter(torch.randn(8, 4).to(torch.bfloat16))
    optimizer = ConceptMuon([param], muon_split_sizes={param: (6,)}, muon_adamw_ranges={param: (6, 8)})
    legacy_momentum = torch.randn(8, 4).to(torch.bfloat16)
    optimizer.state[param]["momentum"] = legacy_momentum.clone()

    state_dict = optimizer.state_dict()
    restored = ConceptMuon([param], muon_split_sizes={param: (6,)}, muon_adamw_ranges={param: (6, 8)})
    restored.load_state_dict(state_dict)

    promoted = restored.state[param]["momentum"]
    assert promoted.dtype == torch.float32
    torch.testing.assert_close(promoted, legacy_momentum.float(), atol=0, rtol=0)


def _uneven_shard_restore_worker(rank: int, world_size: int) -> None:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from xtuner.v1.optim.concept_qwen35_muon import ConceptMuon

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29573"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        mesh = DeviceMesh("cpu", list(range(world_size)))
        torch.manual_seed(0)
        # 7 rows over 2 ranks shard unevenly to (4, 4) and (3, 4).
        param = nn.Parameter(distribute_tensor(torch.randn(7, 4).to(torch.bfloat16), mesh, placements=[Shard(0)]))
        optimizer = ConceptMuon([param], muon_split_sizes={param: (5,)}, muon_adamw_ranges={param: (5, 7)})
        state = optimizer.state[param]
        for name in ("momentum", "gate_momentum", "gate_variance"):
            state[name] = torch.zeros_like(param, dtype=torch.float32)

        state_dict = optimizer.state_dict()
        restored = ConceptMuon([param], muon_split_sizes={param: (5,)}, muon_adamw_ranges={param: (5, 7)})
        restored.load_state_dict(state_dict)

        for name in ("momentum", "gate_momentum", "gate_variance"):
            restored_state = restored.state[param][name]
            # from_local must keep the true global shape, not local_rows*world_size.
            assert tuple(restored_state.shape) == (7, 4), (name, tuple(restored_state.shape))
            assert restored_state.to_local().shape == param.to_local().shape
    finally:
        dist.destroy_process_group()


def test_load_state_dict_preserves_global_shape_on_uneven_shards():
    """Restoring FP32 state on unevenly sharded parameters keeps the true
    global shape instead of local_rows*world_size."""
    torch.multiprocessing.spawn(_uneven_shard_restore_worker, args=(2,), nprocs=2, join=True)


# ─── Test: end-to-end FSDP ───────────────────────────────────────────────────


class TestMuonFSDP(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 4

    @parametrize.parametrize("enable_all2all", [True, False])
    def test_muon_fsdp_matches_reference(self, enable_all2all: bool):
        """One Muon step on a fully-sharded model must match the single-process
        reference for every parameter, across all param categories.

        Each rank processes a different sample (as in real DDP/FSDP training).
        The reference processes all samples and averages the loss to produce
        equivalent gradients.
        """
        self.create_pg("cuda")
        device = "cuda"
        rank = dist.get_rank()
        ws = dist.get_world_size()

        LR = 0.01
        MU = 0.95
        WD = 0.01
        EPSILON = 1e-8
        BETAS = (0.9, 0.95)

        # ── Build model on every rank, then broadcast rank-0 weights ─────────
        config = ToyMoEModelConfig(compile_cfg=False)
        model = config.build().to(device)
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

        ref_model = copy.deepcopy(model)

        # ── Generate input data on rank 0 and scatter ────────────────────────
        if rank == 0:
            torch.manual_seed(7)
            all_input_ids = torch.randint(0, config.vocab_size, (ws, 8), device=device)
        else:
            all_input_ids = torch.empty(ws, 8, dtype=torch.long, device=device)
        dist.broadcast(all_input_ids, src=0)

        # Each rank gets its own sample
        local_input_ids = all_input_ids[rank : rank + 1]

        # ── Reference: forward all samples, average loss ─────────────────────
        ref_loss = ref_model(all_input_ids) / ws
        ref_loss.backward()

        ref_optim = ReferenceMuon(ref_model, lr=LR, mu=MU, weight_decay=WD, epsilon=EPSILON, betas=BETAS)
        ref_optim.step()

        # ── Fully shard the test model (production path) ─────────────────────
        fsdp_config = FSDPConfig(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            torch_compile=False,
        )
        model.fully_shard(fsdp_config)

        # ── Forward + backward on sharded model (each rank sees its sample) ──
        fsdp_loss = model(local_input_ids)
        fsdp_loss.backward()

        # ── Production Muon optimizer step ────────────────────────────────────
        muon_config = MuonConfig(
            lr=LR, momentum=MU, weight_decay=WD, eps=EPSILON, betas=BETAS, enable_all2all=enable_all2all
        )
        optim = muon_config.build(model)
        optim.step()

        # ── Compare all parameters ───────────────────────────────────────────
        for (name, ref_p), (_, fsdp_p) in zip(ref_model.named_parameters(), model.named_parameters()):
            full = fsdp_p.data.full_tensor()  # type: ignore[attr-defined]
            abs_diff = (full - ref_p.data).abs()
            rel_diff = abs_diff / (ref_p.data.abs().clamp(min=1e-12))
            torch.testing.assert_close(
                full,
                ref_p.data,
                atol=1e-6,
                rtol=1e-5,
                msg=f"mismatch on '{name}': max_abs={abs_diff.max().item():.2e}, max_rel={rel_diff.max().item():.2e}",
            )
