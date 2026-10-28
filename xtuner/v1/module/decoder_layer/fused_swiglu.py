"""Fused gate/up SwiGLU MLP used by checkpoint-native model layouts."""

import torch.nn.functional as F
from torch import Tensor, nn

from xtuner.v1.float8.config import Float8Config

from ..linear import build_linear


class FusedSwiGLUMLP(nn.Module):
    """SwiGLU with one checkpoint-compatible ``[gate | up]`` projection."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        float8_cfg: Float8Config | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = build_linear(
            hidden_size,
            2 * intermediate_size,
            bias=bias,
            float8_cfg=float8_cfg,
        )
        self.fc2 = build_linear(
            intermediate_size,
            hidden_size,
            bias=bias,
            float8_cfg=float8_cfg,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate, up = self.fc1(hidden_states).chunk(2, dim=-1)
        return self.fc2((F.silu(gate.float()) * up.float()).to(gate.dtype))


__all__ = ["FusedSwiGLUMLP"]
