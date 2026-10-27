import pytest
import torch

from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.model.moe.qwen3_5_text_split import Qwen3_5_VLTextMoE35BA3BSplitConfig
from xtuner.v1.module.mtp import MTPConfig


@pytest.mark.parametrize(
    "config_cls",
    [Qwen3_5_VLTextMoE35BA3BConfig, Qwen3_5_VLTextMoE35BA3BSplitConfig],
    ids=["fused", "split"],
)
def test_builds_linear_attention_mtp_and_maps_hf_keys(config_cls) -> None:
    config = config_cls(
        mtp_config=MTPConfig(num_layers=1, decoder_layer_type="linear_attention"),
        compile_cfg=False,
    )

    with torch.device("meta"):
        model = config.build()

    assert model.mtp_block is not None
    mtp_decoder = model.mtp_block.layers[0].decoder_layer
    assert type(mtp_decoder.self_attn) is type(model.layers["0"].self_attn)
    assert model.to_hf_key_list("mtp_block.layers.0.decoder_layer.self_attn.in_proj.weight") == [
        "mtp.layers.0.linear_attn.in_proj.weight"
    ]
