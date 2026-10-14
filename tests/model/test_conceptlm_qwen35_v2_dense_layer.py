"""NCP Qwen3.5 v2 decoder-layer architecture and CPU contract tests."""

import json
import os
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.model.dense.conceptlm_qwen35_v2 import (
    CONCEPTLM_QWEN35_V2_COMPILE_CFG,
    ConceptLMQwen35V2,
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
from xtuner.v1.module.attention.ncp_qwen35_v2 import (
    NCPQwen35V2GatedDeltaNetConfig,
    NCPQwen35V2MLAConfig,
    _delta_rule_cu_seqlens,
    _fused_zero_rms_norm_gate,
    _production_feature_enabled,
)
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer
from xtuner.v1.module.decoder_layer.fused_swiglu import FusedSwiGLUMLP
from xtuner.v1.module.rms_norm import GatedDeltaNetInputRMSNorm, MuddRMSNorm, ZeroCenteredRMSNorm
from xtuner.v1.ops import gated_deltanet as gdn_ops


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

    def test_attention_projections_use_repository_linear_builder(self, config: _TinyV2Config) -> None:
        model = config.build()
        gdn = model.encoder.layers["0"].self_attn
        mla = model.encoder.layers["3"].self_attn
        assert isinstance(gdn, NCPQwen35V2GatedDeltaNet)
        assert isinstance(mla, NCPQwen35V2MLA)

        assert type(gdn.in_proj) is torch.nn.Linear
        assert type(gdn.out_proj) is not torch.nn.Linear
        for projection in (mla.q_down, mla.q_up, mla.kv_down, mla.kv_up, mla.out_proj):
            assert type(projection) is not torch.nn.Linear

    def test_default_compile_is_enabled_during_model_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        compile_configs: list[object] = []

        def record_compile_config(self: ConceptLMQwen35V2, compile_cfg: object) -> None:
            del self
            compile_configs.append(compile_cfg)

        monkeypatch.setattr(ConceptLMQwen35V2, "_maybe_enable_compile", record_compile_config)
        model = _TinyV2Config(compile_cfg=True).build()

        assert model.default_compile_cfg == CONCEPTLM_QWEN35_V2_COMPILE_CFG
        assert compile_configs == [CONCEPTLM_QWEN35_V2_COMPILE_CFG]

    def test_attention_configs_forward_float8_to_repository_linears(
        self,
        config: _TinyV2Config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import xtuner.v1.module.attention.ncp_qwen35_v2 as v2_attention

        float8_cfg = Float8Config()
        received: list[Float8Config | None] = []

        def record_linear(
            in_features: int,
            out_features: int,
            bias: bool = True,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
            float8_cfg: Float8Config | None = None,
        ) -> torch.nn.Module:
            received.append(float8_cfg)
            return torch.nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)

        monkeypatch.setattr(v2_attention, "build_linear", record_linear)
        mla = NCPQwen35V2MLAConfig(
            num_attention_heads=config.mla_num_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_head_dim=config.qk_head_dim,
            qk_pos_emb_head_dim=config.qk_pos_emb_head_dim,
            v_head_dim=config.v_head_dim,
            rotary_base=config.mla_rotary_base,
            rms_norm_eps=config.rms_norm_eps,
        ).build(config.hidden_size, float8_cfg=float8_cfg)
        assert config.linear_attention is not None
        gdn = NCPQwen35V2GatedDeltaNetConfig(**config.linear_attention.model_dump()).build(
            config.hidden_size,
            float8_cfg=float8_cfg,
        )

        assert isinstance(mla, NCPQwen35V2MLA)
        assert isinstance(gdn, NCPQwen35V2GatedDeltaNet)
        assert received == [float8_cfg] * 6

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

        padded_ctx, padded_lengths = model._token_context(seq_ctx, [3, 5])

        assert padded_lengths == [8, 8]
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

        token_ctx, padded_lengths = model._token_context(seq_ctx, [8, 12])

        assert token_ctx is seq_ctx
        assert padded_lengths == [8, 12]

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
            ({"linear_attention": None}, "requires the 9B GatedDeltaNet config"),
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

    def test_gdn_delta_uses_dense_path_for_single_document(self):
        seq_ctx = _context([8])

        assert _delta_rule_cu_seqlens(seq_ctx) is None

    def test_gdn_delta_preserves_boundaries_for_multiple_documents(self):
        seq_ctx = _context([5, 3])

        cu_seqlens = _delta_rule_cu_seqlens(seq_ctx)
        assert cu_seqlens is seq_ctx.cu_seq_lens_q
        torch.testing.assert_close(cu_seqlens, torch.tensor([0, 5, 8], dtype=torch.int32), rtol=0, atol=0)


def test_gdn_custom_op_override_is_independent_of_broad_hf_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XTUNER_HF_IMPL", "true")
    monkeypatch.delenv("XTUNER_GDN_CUSTOM_OP", raising=False)
    assert not gdn_ops._custom_op_enabled()

    monkeypatch.setenv("XTUNER_GDN_CUSTOM_OP", "true")
    assert gdn_ops._custom_op_enabled()

    monkeypatch.setenv("XTUNER_GDN_CUSTOM_OP", "false")
    assert not gdn_ops._custom_op_enabled()


def test_gdn_production_features_follow_broad_hf_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    feature = "XTUNER_GDN_FUSED_QK_L2"
    monkeypatch.delenv(feature, raising=False)
    monkeypatch.delenv("XTUNER_HF_IMPL", raising=False)
    assert _production_feature_enabled(feature)

    monkeypatch.setenv("XTUNER_HF_IMPL", "true")
    assert not _production_feature_enabled(feature)

    monkeypatch.setenv(feature, "true")
    assert _production_feature_enabled(feature)

    monkeypatch.setenv(feature, "false")
    assert not _production_feature_enabled(feature)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the CUDA FLA fused norm-gate kernel")
def test_v2_fused_zero_rms_norm_gate_supports_forward_and_backward() -> None:
    pytest.importorskip("fla")
    torch.manual_seed(0)
    eps = 1.0e-6
    x = torch.randn(64, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    weight = torch.randn(16, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    output = _fused_zero_rms_norm_gate(x, gate, weight, eps)
    normalized = (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * (weight.float() + 1.0)
    ).to(x.dtype)
    reference = (normalized.float() * torch.nn.functional.silu(gate.float())).to(x.dtype)
    torch.testing.assert_close(output, reference, rtol=3.0e-2, atol=3.0e-2)

    output.float().square().mean().backward()
    for tensor in (x, gate, weight):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("use_qk_l2norm_in_kernel", [False, True])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the CUDA FLA kernels")
def test_gdn_custom_op_supports_installed_fla_forward_and_backward(use_qk_l2norm_in_kernel: bool) -> None:
    pytest.importorskip("fla")
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
    from xtuner.v1.ops.gated_deltanet.chunk_gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(0)
    shape = (1, 64, 2)
    query = torch.randn(*shape, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(*shape, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(*shape, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    decay = (-torch.rand(*shape, device="cuda", dtype=torch.float32)).requires_grad_()
    beta = torch.rand(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    inputs = (query, key, value, decay, beta)
    reference_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in inputs)

    output, final_state = chunk_gated_delta_rule(
        query,
        key,
        value,
        g=decay,
        beta=beta,
        output_final_state=False,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    assert final_state is None
    assert torch.isfinite(output).all()
    reference, reference_final_state = fla_chunk_gated_delta_rule(
        reference_inputs[0],
        reference_inputs[1],
        reference_inputs[2],
        g=reference_inputs[3],
        beta=reference_inputs[4],
        output_final_state=False,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    assert reference_final_state is None
    torch.testing.assert_close(output, reference, rtol=0, atol=0)

    upstream = torch.randn_like(output)
    gradients = torch.autograd.grad(output, inputs, upstream)
    reference_gradients = torch.autograd.grad(reference, reference_inputs, upstream)
    for gradient, reference_gradient in zip(gradients, reference_gradients, strict=True):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, reference_gradient, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA attention and GDN kernels")
@pytest.mark.parametrize("lengths", [[8], [5, 3], [4, 4, 4]])
def test_v2_default_compile_supports_forward_and_backward(
    monkeypatch: pytest.MonkeyPatch, lengths: list[int]
) -> None:
    pytest.importorskip("causal_conv1d")
    pytest.importorskip("fla")
    from xtuner.v1.utils.compile import is_compiled_function

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


class TestGDNDispatcher:
    @pytest.mark.parametrize("hf_mode", ["true", "false"])
    @pytest.mark.parametrize("override", [None, "true", "false"])
    def test_override_selects_delta_rule_without_changing_conv(
        self, monkeypatch: pytest.MonkeyPatch, hf_mode: str, override: str | None
    ) -> None:
        monkeypatch.setenv("XTUNER_HF_IMPL", hf_mode)
        if override is None:
            monkeypatch.delenv("XTUNER_GDN_CUSTOM_OP", raising=False)
        else:
            monkeypatch.setenv("XTUNER_GDN_CUSTOM_OP", override)
        use_custom_op = override == "true" if override is not None else hf_mode == "false"
        if use_custom_op:
            from xtuner.v1.ops.gated_deltanet.chunk_gated_delta_rule import chunk_gated_delta_rule
        else:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        assert gdn_ops.get_chunk_gated_delta_rule_fn() is chunk_gated_delta_rule
        if hf_mode == "true":
            assert gdn_ops.get_causal_conv1d_fn() is gdn_ops._hf_causal_conv1d_adapter
        else:
            from xtuner.v1.ops.gated_deltanet.causal_conv1d import causal_conv1d_fn

            assert gdn_ops.get_causal_conv1d_fn() is causal_conv1d_fn


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashAttention")
class TestMLAFlashAttention:
    @pytest.mark.parametrize("lengths", [[8], [5, 3]])
    @pytest.mark.parametrize("qk_width,value_width", [(12, 8), (8, 16)])
    def test_matches_sdpa_output_and_gradients(
        self, lengths: list[int], qk_width: int, value_width: int
    ) -> None:
        from xtuner.v1.module.attention.ncp_qwen35_v2 import _causal_attention

        torch.manual_seed(1)
        total = sum(lengths)
        tensors = tuple(
            torch.randn(1, total, 2, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for width in (qk_width, qk_width, value_width)
        )
        reference = tuple(tensor.detach().float().requires_grad_() for tensor in tensors)
        doc_ids = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths)).cuda()
        mask = (doc_ids[:, None] == doc_ids[None, :]) & torch.ones(
            total, total, device="cuda", dtype=torch.bool
        ).tril()
        scale = qk_width**-0.5
        expected = torch.nn.functional.scaled_dot_product_attention(
            *(tensor.transpose(1, 2) for tensor in reference), attn_mask=mask, scale=scale
        ).transpose(1, 2)
        actual = _causal_attention(*tensors, _context(lengths).to("cuda"), scale)
        torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, tensors, upstream)
        expected_grads = torch.autograd.grad(expected, reference, upstream.float())
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad.float(), expected_grad, rtol=0.03, atol=0.02)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton normalization")
class TestGDNZeroCenteredNormGate:
    @pytest.mark.parametrize("rows", [8, 65])
    def test_preserves_te_weight_and_bf16_rounding(self, rows: int) -> None:
        from xtuner.v1.module.attention.ncp_qwen35_v2 import _fused_zero_rms_norm_gate
        from xtuner.v1.ops.rms_norm.ncp_qwen35_v2 import zero_rmsnorm_cuda

        torch.manual_seed(42)
        x = torch.randn(1, rows, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        gate = torch.randn_like(x, requires_grad=True)
        weight = (torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.01).requires_grad_()
        actual = _fused_zero_rms_norm_gate(x, gate, weight, 1e-6)
        normalized = zero_rmsnorm_cuda(x.detach(), weight.detach(), 1e-6)
        expected = (normalized.float() * torch.nn.functional.silu(gate.float())).to(x.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        xf, gf, wf = (tensor.detach().float().requires_grad_() for tensor in (x, gate, weight))
        normalized_reference = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
        normalized_reference = (normalized_reference * (wf + 1.0)).to(x.dtype).float()
        reference = (normalized_reference * torch.nn.functional.silu(gf)).to(x.dtype)
        upstream = torch.randn_like(x)
        actual_grads = torch.autograd.grad(actual, (x, gate, weight), upstream)
        expected_grads = torch.autograd.grad(reference, (xf, gf, wf), upstream)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad.float(), expected_grad, rtol=0.02, atol=0.02)
