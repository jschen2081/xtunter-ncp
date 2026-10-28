"""NCP Qwen3.5 v2 decoder-layer architecture and CPU contract tests."""

import json
import os
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.dense.conceptlm_qwen35_9b import (
    ConceptLMQwen35V2Config,
    CrossMUDDRoute,
    PredictedConceptInjection,
    SelfMUDDRoute,
    _crop_documents,
)
from xtuner.v1.module.attention import (
    GatedDeltaNetConfig,
    MHAConfig,
    NCPQwen35V2GatedDeltaNet,
    NCPQwen35V2MLA,
)
from xtuner.v1.module.attention.concept_qwen35 import (
    _delta_rule_cu_seqlens,
)
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer


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
    @pytest.mark.parametrize("concept_muon", [False, True])
    def test_conv_weights_follow_semantic_adamw_routing(
        self,
        config: _TinyV2Config,
        monkeypatch: pytest.MonkeyPatch,
        concept_muon: bool,
    ) -> None:
        from xtuner.v1.config.optim import MuonConfig

        model = config.build()
        conv_weights = {
            name: param for name, param in model.named_parameters() if name.endswith(".self_attn.conv1d.weight")
        }
        assert len(conv_weights) == 5  # Three encoder GDN layers, one predictor and one decoder.
        metadata = model.get_muon_metadata()
        for param in conv_weights.values():
            assert metadata["param_options"][param] == {"algorithm": "adamw", "weight_decay": 0.1}

        # Parameter grouping is CPU-only and does not need a process group.
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        optimizer = MuonConfig(concept_muon=concept_muon, weight_decay=0.03).build(model)
        groups = {id(param): group for group in optimizer.param_groups for param in group["params"]}
        for param in conv_weights.values():
            group = groups[id(param)]
            assert group["algorithm"] == ("adamw" if concept_muon else "muon")
            assert group["weight_decay"] == (0.1 if concept_muon else 0.03)

    @pytest.mark.parametrize("concept_muon", [False, True])
    def test_codebooks_follow_muon_routing(
        self,
        config: _TinyV2Config,
        monkeypatch: pytest.MonkeyPatch,
        concept_muon: bool,
    ) -> None:
        from xtuner.v1.config.optim import MuonConfig

        model = config.build()
        codebooks = {name: param for name, param in model.named_parameters() if name.startswith("codebooks.")}
        assert len(codebooks) == 4
        metadata = model.get_muon_metadata()
        for param in codebooks.values():
            assert param not in metadata["param_options"]

        # Parameter grouping is CPU-only and does not need a process group.
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        optimizer = MuonConfig(concept_muon=concept_muon, weight_decay=0.03).build(model)
        groups = {id(param): group for group in optimizer.param_groups for param in group["params"]}
        for param in codebooks.values():
            group = groups[id(param)]
            assert group["algorithm"] == "muon"
            assert group["weight_decay"] == 0.03

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
            "encoder.layers.0.attention_output_norm.weight": ("model.encoder.layers.0.self_attn_output_norm.weight"),
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

    def test_fp32_save_policy_preserves_gdn_convolution_in_all_stacks(self, config: _TinyV2Config) -> None:
        model = config.build()
        for stack in (model.encoder, model.predictor, model.decoder):
            for layer in stack.layers.values():
                if not isinstance(layer.self_attn, NCPQwen35V2GatedDeltaNet):
                    continue
                parameter = layer.self_attn.conv1d.weight
                name = next(name for name, value in model.named_parameters() if value is parameter)
                hf_name = model.to_hf_key_list(name)[0]
                assert model._get_save_dtype(hf_name, torch.bfloat16) == torch.float32

    def test_tiny_packed_model_forward_runs_on_cpu(self, config: _TinyV2Config):
        torch.manual_seed(0)
        model = config.build().eval()
        seq_ctx = _context([5, 3])
        with torch.no_grad():
            outputs = model(seq_ctx)
        assert outputs["logits"].shape == (1, 8, config.vocab_size)
        assert torch.isfinite(outputs["logits"]).all()

    def test_all_state_keys_have_unique_hf_mappings(self, config: _TinyV2Config) -> None:
        model = config.build()
        mapped = [model.to_hf_key_list(key)[0] for key in model.state_dict()]

        assert len(mapped) == len(set(mapped))
        assert all(key == "lm_head.weight" or key.startswith("model.") for key in mapped)

    def test_fusion_alpha_scalar_checkpoint_round_trip(self, config: _TinyV2Config) -> None:
        model = config.build()
        hf_scalar = torch.tensor(0.25)

        canonical = model.hf_tensor_to_canonical("fusion_alpha", hf_scalar)
        restored = model.param_to_safetensor(canonical, "model.fusion_alpha")

        assert canonical.shape == (1,)
        assert restored.shape == ()
        torch.testing.assert_close(restored, hf_scalar, rtol=0, atol=0)


class TestConceptLMQwen35V2PaddingContract:
    def test_each_packed_document_is_padded_and_cropped_independently(self, config: _TinyV2Config) -> None:
        model = config.build()
        seq_ctx = _context([3, 5])
        seq_ctx.input_ids = torch.arange(8).unsqueeze(0)

        padded_ctx, padded_lengths, compute_lengths = model._token_context(seq_ctx, [3, 5])

        assert padded_lengths == [8, 8]
        assert compute_lengths == padded_lengths
        assert padded_ctx.input_ids is not None
        torch.testing.assert_close(padded_ctx.input_ids[:, :3], torch.tensor([[0, 1, 2]]), rtol=0, atol=0)
        torch.testing.assert_close(padded_ctx.input_ids[:, 8:13], torch.tensor([[3, 4, 5, 6, 7]]), rtol=0, atol=0)
        assert padded_ctx.input_ids[:, 3:8].eq(config.pad_token_id).all()
        assert padded_ctx.input_ids[:, 13:16].eq(config.pad_token_id).all()

        hidden = torch.arange(16).reshape(1, 16, 1)
        cropped = _crop_documents(hidden, [3, 5], padded_lengths)
        torch.testing.assert_close(cropped.flatten(), torch.tensor([0, 1, 2, 8, 9, 10, 11, 12]), rtol=0, atol=0)

    def test_aligned_documents_reuse_original_context(self, config: _TinyV2Config) -> None:
        model = config.build()
        seq_ctx = _context([8, 12])

        token_ctx, padded_lengths, compute_lengths = model._token_context(seq_ctx, [8, 12])

        assert token_ctx is seq_ctx
        assert padded_lengths == [8, 12]
        assert compute_lengths == padded_lengths

    def test_short_inputs_embeds_are_rejected_before_padding(self, config: _TinyV2Config) -> None:
        model = config.build()
        seq_ctx = _context([3])
        seq_ctx.inputs_embeds = torch.randn(1, 3, config.hidden_size)
        seq_ctx.input_ids = None

        with pytest.raises(ValueError, match="needs input_ids"):
            model._token_context(seq_ctx, [3])

    def test_packed_documents_do_not_leak_across_boundaries(self, config: _TinyV2Config) -> None:
        torch.manual_seed(3)
        model = config.build().eval()
        original = _context([3, 5])
        original.input_ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
        changed = _context([3, 5])
        changed.input_ids = original.input_ids.clone()
        changed.input_ids[:, :3] = torch.tensor([[10, 11, 12]])

        with torch.no_grad():
            original_logits = model(original).logits
            changed_logits = model(changed).logits

        assert original_logits is not None
        assert changed_logits is not None
        torch.testing.assert_close(original_logits[:, 3:], changed_logits[:, 3:], rtol=0, atol=0)


class TestConceptLMQwen35V2ConfigContract:
    @pytest.mark.parametrize(
        "updates,error",
        [
            ({"hidden_size": 30}, "hidden_size must be divisible"),
            ({"linear_attention": None}, "valid dictionary or instance of GatedDeltaNetConfig"),
            ({"self_mudd_block_size": 0}, "positive MUDD block sizes"),
            ({"qk_pos_emb_head_dim": 3}, "positional head dim must be even"),
        ],
    )
    def test_rejects_invalid_architecture_contracts(self, updates: dict[str, object], error: str) -> None:
        with pytest.raises(ValidationError, match=error):
            _TinyV2Config(**updates)

    def test_matches_exported_hf_config_when_checkpoint_is_available(self) -> None:
        checkpoint = os.environ.get("NCP_QWEN35_V2_PATH")
        if checkpoint is None:
            pytest.skip("set NCP_QWEN35_V2_PATH to validate the exported HF config")
        payload = json.loads((Path(checkpoint) / "config.json").read_text())
        config = ConceptLMQwen35V2Config(compile_cfg=False)
        mappings = {
            "hidden_size": "hidden_size",
            "intermediate_size": "intermediate_size",
            "vocab_size": "vocab_size",
            "concept_encoder_layers": "encoder_layers",
            "concept_predictor_layers": "predictor_layers",
            "concept_decoder_layers": "decoder_layers",
            "concept_chunk_size": "chunk_size",
            "concept_num_codebooks": "num_codebooks",
            "concept_codebook_size": "codebook_size",
            "mla_num_heads": "num_attention_heads",
            "q_lora_rank": "q_lora_rank",
            "kv_lora_rank": "kv_lora_rank",
            "qk_head_dim": "qk_head_dim",
            "qk_pos_emb_head_dim": "qk_pos_emb_head_dim",
            "v_head_dim": "v_head_dim",
            "max_position_embeddings": "max_position_embeddings",
            "self_mudd_block_size": "self_mudd_block_size",
            "cross_mudd_block_size": "cross_mudd_block_size",
        }
        for local_name, hf_name in mappings.items():
            assert getattr(config, local_name) == payload[hf_name]
        assert config.linear_attention is not None
        assert config.linear_attention.num_key_heads == payload["linear_num_key_heads"]
        assert config.linear_attention.num_value_heads == payload["linear_num_value_heads"]
        assert config.linear_attention.key_head_dim == payload["linear_key_head_dim"]
        assert config.linear_attention.value_head_dim == payload["linear_value_head_dim"]

    def test_checkpoint_index_matches_every_model_parameter_when_available(self) -> None:
        checkpoint = os.environ.get("NCP_QWEN35_V2_PATH")
        if checkpoint is None:
            pytest.skip("set NCP_QWEN35_V2_PATH to validate the checkpoint index")
        weight_map = json.loads((Path(checkpoint) / "model.safetensors.index.json").read_text())["weight_map"]
        with torch.device("meta"):
            model = ConceptLMQwen35V2Config(compile_cfg=False).build()

        mapped = {model.to_hf_key_list(key)[0] for key in model.state_dict()}

        assert mapped == set(weight_map)


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
        output_with_changed_v = attention(hidden, None, seq_ctx, (q_stream, k_stream, changed_v))["projected_output"]
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

    def test_gdn_delta_uses_dense_path_for_single_document(self):
        seq_ctx = _context([8])

        assert _delta_rule_cu_seqlens(seq_ctx) is None

    def test_gdn_delta_preserves_boundaries_for_multiple_documents(self):
        seq_ctx = _context([5, 3])

        cu_seqlens = _delta_rule_cu_seqlens(seq_ctx)
        assert cu_seqlens is seq_ctx.cu_seq_lens_q
        torch.testing.assert_close(cu_seqlens, torch.tensor([0, 5, 8], dtype=torch.int32), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA attention and GDN kernels")
@pytest.mark.parametrize("lengths", [[8], [5, 3], [4, 4, 4]])
def test_v2_default_compile_supports_forward_and_backward(monkeypatch: pytest.MonkeyPatch, lengths: list[int]) -> None:
    pytest.importorskip("causal_conv1d")
    pytest.importorskip("fla")
    from xtuner.v1.utils.compile import is_compiled_function

    # CompileConfig replaces class methods globally; restore them after this test.
    for module_type in (DenseDecoderLayer, SelfMUDDRoute, CrossMUDDRoute, PredictedConceptInjection):
        monkeypatch.setattr(module_type, "forward", module_type.forward)

    monkeypatch.setenv("XTUNER_GDN_CUSTOM_OP", "true")
    monkeypatch.setenv("NCP_Q35V2_TORCH_RMS", "1")
    monkeypatch.setenv("NCP_Q35V2_TE_RMS", "0")
    monkeypatch.setenv("NCP_Q35V2_TRITON_DUAL_CROSS_MUDD", "0")
    monkeypatch.setenv("XTUNER_GDN_FUSED_QK_L2", "true")
    monkeypatch.setenv("XTUNER_GDN_FUSED_NORM_GATE", "true")

    torch.manual_seed(0)
    config = _TinyV2Config(compile_cfg=True)
    model = config.build().cuda().to(torch.bfloat16)
    seq_ctx = _context(lengths).to("cuda")

    outputs = model(seq_ctx)
    assert outputs.logits is not None
    assert outputs.logits.shape == (1, sum(lengths), config.vocab_size)
    assert torch.isfinite(outputs.logits).all()
    outputs.logits.square().mean().backward()

    assert is_compiled_function(DenseDecoderLayer.forward)
    assert is_compiled_function(SelfMUDDRoute.forward)
    assert is_compiled_function(CrossMUDDRoute.forward)
    assert is_compiled_function(PredictedConceptInjection.forward)
    mla = model.encoder.layers["3"].self_attn
    gdn = model.encoder.layers["0"].self_attn
    assert mla.q_down.weight.grad is not None
    assert gdn.out_proj.weight.grad is not None
    for name in (
        "mudd.routes.encoder.0.router_w1.weight",
        "prediction_heads.0.weight",
        "codebooks.0",
        "fusion_alpha",
    ):
        assert model.get_parameter(name).grad is not None, f"missing gradient for {name}"
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
