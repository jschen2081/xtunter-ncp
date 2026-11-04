"""Qwen3.5 dt_bias must bypass BF16 FSDP casting and HF export."""

from unittest.mock import Mock, patch

import pytest
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy

from xtuner.v1.model.dense.qwen3_5_text import Qwen3_5_VLTextDense4BConfig
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.model.moe.qwen3_5_text_split import Qwen3_5_VLTextMoE35BA3BSplitConfig
from xtuner.v1.module.mtp import MTPConfig


@pytest.fixture(
    params=[
        (Qwen3_5_VLTextDense4BConfig, {}),
        (Qwen3_5_VLTextMoE35BA3BConfig, {}),
        (Qwen3_5_VLTextMoE35BA3BSplitConfig, {}),
        (Qwen3_5_VLTextMoE35BA3BSplitConfig, {"hf_key_mapping": {r"^model\.": "model.language_model."}}),
    ],
    ids=["dense", "moe-fused", "moe-split", "moe-split-vlm-keys"],
)
def model(request: pytest.FixtureRequest) -> torch.nn.Module:
    """Build a four-layer text tower on meta with one linear-attention MTP layer.

    Args:
        request (pytest.FixtureRequest): Model class and optional HF namespace mapping.

    Returns:
        torch.nn.Module: A meta-only model; no weights or CUDA execution are required.
    """
    config_cls, kwargs = request.param
    config = config_cls(
        num_hidden_layers=4,
        max_window_layers=4,
        mtp_config=MTPConfig(num_layers=1, decoder_layer_type="linear_attention"),
        compile_cfg=False,
        **kwargs,
    )
    with torch.device("meta"), patch("torch.cuda.Stream", return_value=None):
        return config.build()


class TestQwen35DTBiasFP32:
    def test_hf_dtype_preserves_backbone_and_mtp_dt_bias(self, model: torch.nn.Module) -> None:
        dt_bias_names = [name for name, _ in model.named_parameters() if name.endswith("dt_bias")]
        assert len(dt_bias_names) == 4
        assert any(name.startswith("mtp_block.") for name in dt_bias_names)
        for name in dt_bias_names:
            hf_keys = model.load_spec_mapping[name].global_hf_keys
            assert all(key.endswith(".linear_attn.dt_bias") for key in hf_keys)
            for key in hf_keys:
                assert model._get_save_dtype(key, torch.bfloat16) == torch.float32, key

        for name, spec in model.load_spec_mapping.items():
            if name.endswith(("A_log", "self_attn.norm.weight")):
                for key in spec.global_hf_keys:
                    assert model._get_save_dtype(key, torch.bfloat16) == torch.float32, key
            if name.endswith("in_proj.weight"):
                for key in spec.global_hf_keys:
                    assert model._get_save_dtype(key, torch.bfloat16) == torch.bfloat16, key

    def test_fsdp_excludes_dt_bias_from_bf16_materialization(self, model: torch.nn.Module) -> None:
        mesh = Mock(ndim=1)
        model._world_mesh = mesh
        with (
            patch("xtuner.v1.model.base.fully_shard") as shard,
            patch("xtuner.v1.model.base.distribute_tensor", side_effect=lambda tensor, *_: tensor),
        ):
            model._fully_shard(
                mesh=mesh,
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16),
                reshard_after_forward=True,
                offload_policy=None,
            )
        ignored = shard.call_args.kwargs["ignored_params"]
        assert ignored is not None
        dt_bias_names = [name for name, _ in model.named_parameters() if name.endswith("dt_bias")]
        assert len(dt_bias_names) == 4
        for name, param in model.named_parameters():
            if name in dt_bias_names:
                assert name in model._fsdp_ignored_param_names
                assert param in ignored
                assert param.dtype == torch.float32
            elif name.endswith("in_proj.weight"):
                assert name not in model._fsdp_ignored_param_names
                assert param not in ignored
