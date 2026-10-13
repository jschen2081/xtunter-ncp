"""NCP Qwen3.5 v2 decoder-layer architecture and CPU contract tests."""

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.dense.conceptlm_qwen35_v2 import ConceptLMQwen35V2Config
from xtuner.v1.module.attention import (
    GatedDeltaNetConfig,
    MHAConfig,
    NCPQwen35V2GatedDeltaNet,
    NCPQwen35V2MLA,
)
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer
from xtuner.v1.module.decoder_layer.fused_swiglu import FusedSwiGLUMLP
from xtuner.v1.module.rms_norm import GatedDeltaNetInputRMSNorm, MuddRMSNorm, ZeroCenteredRMSNorm


class _TinyV2Config(ConceptLMQwen35V2Config):
    vocab_size: int = 64
    max_position_embeddings: int = 64
    pad_token_id: int | None = 0
    eos_token_id: int = 1
    hidden_size: int = 32
    intermediate_size: int = 64
    attention: MHAConfig = MHAConfig(
        with_gate=True,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        qk_norm=True,
        rms_norm_eps=1.0e-6,
        rms_norm_type="zero_centered",
    )
    linear_attention: GatedDeltaNetConfig = GatedDeltaNetConfig(
        num_value_heads=4,
        num_key_heads=2,
        key_head_dim=4,
        value_head_dim=4,
        conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1.0e-6,
    )
    concept_encoder_layers: int = 4
    concept_predictor_layers: int = 1
    concept_decoder_layers: int = 1
    concept_chunk_size: int = 4
    concept_codebook_size: int = 8
    concept_num_codebooks: int = 4
    mla_num_heads: int = 2
    q_lora_rank: int = 16
    kv_lora_rank: int = 8
    qk_head_dim: int = 8
    qk_pos_emb_head_dim: int = 4
    v_head_dim: int = 8


@pytest.fixture
def config() -> _TinyV2Config:
    return _TinyV2Config(compile_cfg=False)


def _context(lengths: list[int]) -> SequenceContext:
    total = sum(lengths)
    boundaries = [0]
    positions = []
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
        positions.append(torch.arange(length))
    cu = torch.tensor(boundaries, dtype=torch.int32)
    return SequenceContext(
        input_ids=torch.randint(0, 64, (1, total)),
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=max(lengths),
        max_length_k=max(lengths),
        num_padding=0,
        sequence_parallel_mesh=None,
        device=torch.device("cpu"),
        position_ids=torch.cat(positions).unsqueeze(0),
    )


def _manual_dense_forward(
    layer: DenseDecoderLayer,
    streams: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    seq_ctx: SequenceContext,
) -> torch.Tensor:
    q_stream, k_stream, v_stream, residual = streams
    mudd_inputs = (
        layer.input_layernorm(q_stream),
        layer.input_layernorm(k_stream),
        layer.input_layernorm(v_stream),
    )
    attention = layer.self_attn(
        hidden_states=mudd_inputs[0],
        position_embeddings=None,
        seq_ctx=seq_ctx,
        mudd_inputs=mudd_inputs,
    )["projected_output"]
    hidden = residual + layer.attention_output_norm(attention)
    mlp = layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden + layer.mlp_output_norm(mlp)


class TestConceptLMQwen35V2DenseLayer:
    def test_stack_uses_dense_decoder_layer_with_v2_modules(self, config: _TinyV2Config):
        model = config.build()
        gdn_layer = model.encoder.layers["0"]
        mla_layer = model.encoder.layers["3"]

        assert isinstance(gdn_layer, DenseDecoderLayer)
        assert isinstance(mla_layer, DenseDecoderLayer)
        assert isinstance(gdn_layer.self_attn, NCPQwen35V2GatedDeltaNet)
        assert isinstance(mla_layer.self_attn, NCPQwen35V2MLA)
        assert isinstance(gdn_layer.mlp, FusedSwiGLUMLP)
        assert isinstance(gdn_layer.input_layernorm, GatedDeltaNetInputRMSNorm)
        assert isinstance(mla_layer.input_layernorm, ZeroCenteredRMSNorm)
        assert isinstance(gdn_layer.attention_output_norm, MuddRMSNorm)
        assert isinstance(gdn_layer.mlp_output_norm, MuddRMSNorm)

    @pytest.mark.parametrize("layer_index", [0, 3])
    def test_dense_forward_preserves_v2_operation_order(self, config: _TinyV2Config, layer_index: int):
        torch.manual_seed(0)
        model = config.build()
        layer = model.encoder.layers[str(layer_index)]
        seq_ctx = _context([3, 2])
        streams = tuple(torch.randn(1, 5, config.hidden_size) for _ in range(4))

        expected = _manual_dense_forward(layer, streams, seq_ctx)
        actual = layer(
            streams[3],
            position_embeddings=None,
            seq_ctx=seq_ctx,
            mudd_streams=streams,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_hf_key_mapping_preserves_checkpoint_layout(self, config: _TinyV2Config):
        model = config.build()
        mappings = {
            "encoder.layers.0.self_attn.in_proj.weight": "model.encoder.layers.0.attention.in_proj.weight",
            "encoder.layers.0.input_layernorm.weight": "model.encoder.layers.0.input_norm.weight",
            "encoder.layers.0.post_attention_layernorm.weight": "model.encoder.layers.0.post_norm.weight",
            "encoder.layers.0.attention_output_norm.weight": (
                "model.encoder.layers.0.self_attn_output_norm.weight"
            ),
            "encoder.layers.0.mlp.fc1.weight": "model.encoder.layers.0.fc1.weight",
            "encoder.layers.0.mlp.fc2.weight": "model.encoder.layers.0.fc2.weight",
            "encoder.layers.0.mlp_output_norm.weight": "model.encoder.layers.0.mlp_output_norm.weight",
        }
        state = model.state_dict()
        for local_key, hf_key in mappings.items():
            assert local_key in state
            assert model.to_hf_key_list(local_key) == [hf_key]

        gdn = model.encoder.layers["0"].self_attn
        assert isinstance(gdn, NCPQwen35V2GatedDeltaNet)
        expected_rows = 2 * gdn.key_dim + 2 * gdn.value_dim + 2 * gdn.num_v_heads
        assert gdn.in_proj.weight.shape == (expected_rows, config.hidden_size)

    def test_tiny_packed_model_forward_runs_on_cpu(self, config: _TinyV2Config):
        torch.manual_seed(0)
        model = config.build().eval()
        seq_ctx = _context([5, 3])
        with torch.no_grad():
            outputs = model(seq_ctx)
        assert outputs["logits"].shape == (1, 8, config.vocab_size)
        assert torch.isfinite(outputs["logits"]).all()


class TestNCPQwen35V2AttentionStreams:
    def test_mla_reads_q_and_k_but_not_v(self, config: _TinyV2Config):
        torch.manual_seed(1)
        model = config.build()
        attention = model.encoder.layers["3"].self_attn
        assert isinstance(attention, NCPQwen35V2MLA)
        seq_ctx = _context([4])
        hidden = torch.randn(1, 4, config.hidden_size)
        q_stream = torch.randn_like(hidden)
        k_stream = torch.randn_like(hidden)
        v_stream = torch.randn_like(hidden)
        changed_v = v_stream + torch.randn_like(v_stream)

        output = attention(hidden, None, seq_ctx, (q_stream, k_stream, v_stream))["projected_output"]
        output_with_changed_v = attention(hidden, None, seq_ctx, (q_stream, k_stream, changed_v))[
            "projected_output"
        ]
        torch.testing.assert_close(output, output_with_changed_v, rtol=0, atol=0)

    def test_gdn_keeps_single_fused_projection(self, config: _TinyV2Config):
        model = config.build()
        attention = model.encoder.layers["0"].self_attn
        assert isinstance(attention, NCPQwen35V2GatedDeltaNet)
        parameter_names = dict(attention.named_parameters())
        assert "in_proj.weight" in parameter_names
        assert not any(name.startswith("in_proj_qkv") for name in parameter_names)
        assert not any(name.startswith("in_proj_z") for name in parameter_names)
        assert not any(name.startswith("in_proj_a") for name in parameter_names)
        assert not any(name.startswith("in_proj_b") for name in parameter_names)
