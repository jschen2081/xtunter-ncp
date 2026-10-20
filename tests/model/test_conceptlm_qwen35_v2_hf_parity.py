"""Checkpoint-backed Hugging Face parity tests for NCP Qwen3.5 v2."""

import os
from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.dense.conceptlm_qwen35_9b import ConceptLMQwen35V2Config
from xtuner.v1.module.attention import GatedDeltaNetConfig, MHAConfig
from xtuner.v1.utils import HFCheckpointLoader


NCP_QWEN35_V2_PATH = os.environ.get("NCP_QWEN35_V2_PATH")


def _controlled_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "XTUNER_HF_IMPL": "true",
        "NCP_Q35V2_TORCH_RMS": "1",
        "NCP_Q35V2_MUDD_RMS": "1",
        "NCP_Q35V2_TE_RMS": "0",
        "NCP_Q35V2_FLA_L2": "0",
        "NCP_Q35V2_FA3": "0",
        "NCP_Q35V2_TE_MLA_PROJ": "0",
        "NCP_Q35V2_MLA_TE_INPUT_NORM": "0",
        "NCP_Q35V2_GDN_TORCH_INPUT_NORM": "1",
        "NCP_Q35V2_GDN_TE_OUT_NORM": "0",
        "NCP_Q35V2_GDN_PACK_PROJECTION": "0",
        "NCP_Q35V2_TE_MLP_FC1": "0",
        "NCP_Q35V2_TE_FINAL_NORM": "0",
        "NCP_Q35V2_TRITON_CROSS_MUDD": "0",
        "NCP_Q35V2_TRITON_DUAL_CROSS_MUDD": "0",
        "NCP_Q35V2_COMPILE_CROSS_ROUTER": "0",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _use_sdpa_for_xtuner_mla(monkeypatch: pytest.MonkeyPatch) -> None:
    import xtuner.v1.module.attention.concept_qwen35 as v2_attention

    def sdpa(query, key, value, seq_ctx, scale):
        del seq_ctx
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=True,
            scale=scale,
        ).transpose(1, 2)

    monkeypatch.setattr(v2_attention, "_causal_attention", sdpa)


def _materialize_and_load(
    module: nn.Module,
    loader: HFCheckpointLoader,
    key_for: Callable[[str], str],
) -> None:
    module.to_empty(device="cuda")
    module.to(torch.bfloat16)
    for name, parameter in module.named_parameters():
        key = key_for(name)
        tensor = loader.load(key)
        assert tensor is not None, f"checkpoint key not found: {key}"
        parameter.data.copy_(tensor.to(device="cuda", dtype=parameter.dtype))


def _sequence_context(length: int) -> SequenceContext:
    ids = torch.zeros(1, length, dtype=torch.long, device="cuda")
    return SequenceContext.from_input_ids((ids,), device="cuda")


def _tiny_xtuner_config() -> ConceptLMQwen35V2Config:
    return ConceptLMQwen35V2Config(
        compile_cfg=False,
        vocab_size=64,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=1,
        hidden_size=32,
        intermediate_size=64,
        attention=MHAConfig(
            with_gate=True,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            qk_norm=True,
            rms_norm_eps=1.0e-6,
            rms_norm_type="zero_centered",
        ),
        linear_attention=GatedDeltaNetConfig(
            num_value_heads=4,
            num_key_heads=2,
            key_head_dim=4,
            value_head_dim=4,
            conv_kernel_dim=4,
            hidden_act="silu",
            rms_norm_eps=1.0e-6,
        ),
        concept_encoder_layers=4,
        concept_predictor_layers=1,
        concept_decoder_layers=1,
        concept_chunk_size=4,
        concept_codebook_size=8,
        concept_num_codebooks=4,
        mla_num_heads=2,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_head_dim=8,
        qk_pos_emb_head_dim=4,
        v_head_dim=8,
    )


def _capture_embedding_leaf(model: nn.Module, captured: dict[str, torch.Tensor]):
    def make_leaf(_module, _inputs, output):
        leaf = output.detach().requires_grad_(True)
        captured["embedding"] = leaf
        return leaf

    embedding = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else model.embed_tokens
    return embedding.register_forward_hook(make_leaf)


@pytest.mark.skipif(NCP_QWEN35_V2_PATH is None, reason="set NCP_QWEN35_V2_PATH")
class TestConceptLMQwen35V2TinyHFParity:
    def test_whole_model_forward_loss_and_embedding_gradient_are_bitwise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _controlled_environment(monkeypatch)
        from transformers import AutoConfig, AutoModelForCausalLM

        assert NCP_QWEN35_V2_PATH is not None
        hf_config = AutoConfig.from_pretrained(NCP_QWEN35_V2_PATH, trust_remote_code=True)
        values = {
            "vocab_size": 64,
            "max_position_embeddings": 64,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "encoder_layers": 4,
            "predictor_layers": 1,
            "decoder_layers": 1,
            "num_hidden_layers": 6,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 12,
            "q_lora_rank": 16,
            "kv_lora_rank": 8,
            "qk_head_dim": 8,
            "qk_pos_emb_head_dim": 4,
            "v_head_dim": 8,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "chunk_size": 4,
            "codebook_size": 8,
            "num_codebooks": 4,
            "encoder_cross_num_sources": 2,
            "predictor_cross_num_sources": 2,
            "shift_tokens": 3,
        }
        for name, value in values.items():
            setattr(hf_config, name, value)

        torch.manual_seed(5)
        xtuner = _tiny_xtuner_config().build().eval()
        hf = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True).float().eval()
        hf_state = {
            xtuner.to_hf_key_list(name)[0]: xtuner.param_to_safetensor(tensor, xtuner.to_hf_key_list(name)[0])
            for name, tensor in xtuner.state_dict().items()
        }
        incompatible = hf.load_state_dict(hf_state, strict=True)
        assert not incompatible.missing_keys
        assert not incompatible.unexpected_keys
        xtuner.requires_grad_(False)
        hf.requires_grad_(False)

        input_ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
        labels = input_ids[:, 1:].reshape(-1)
        xt_captured: dict[str, torch.Tensor] = {}
        hf_captured: dict[str, torch.Tensor] = {}
        xt_handle = _capture_embedding_leaf(xtuner, xt_captured)
        hf_handle = _capture_embedding_leaf(hf, hf_captured)
        try:
            xt_logits = xtuner(SequenceContext.from_input_ids((input_ids,), device="cpu")).logits
            hf_logits = hf(input_ids=input_ids, use_cache=False).logits
            assert xt_logits is not None
            xt_loss = F.cross_entropy(xt_logits[:, :-1].float().reshape(-1, 64), labels)
            hf_loss = F.cross_entropy(hf_logits[:, :-1].float().reshape(-1, 64), labels)
            (xt_gradient,) = torch.autograd.grad(xt_loss, xt_captured["embedding"])
            (hf_gradient,) = torch.autograd.grad(hf_loss, hf_captured["embedding"])
        finally:
            xt_handle.remove()
            hf_handle.remove()

        torch.testing.assert_close(xt_logits, hf_logits, rtol=0, atol=0)
        torch.testing.assert_close(xt_loss, hf_loss, rtol=0, atol=0)
        torch.testing.assert_close(xt_gradient, hf_gradient, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FLA kernels")
@pytest.mark.skipif(NCP_QWEN35_V2_PATH is None, reason="set NCP_QWEN35_V2_PATH")
class TestConceptLMQwen35V2HFParity:
    @pytest.mark.parametrize("stage,layer_index", [("encoder", 0), ("encoder", 3)])
    def test_decoder_layer_forward_loss_and_input_gradients_are_bitwise(
        self,
        stage: str,
        layer_index: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Compare one GDN and one MLA layer without materializing either full model."""
        pytest.importorskip("fla")
        _controlled_environment(monkeypatch)
        _use_sdpa_for_xtuner_mla(monkeypatch)

        from transformers import AutoConfig, AutoModelForCausalLM

        assert NCP_QWEN35_V2_PATH is not None
        checkpoint = Path(NCP_QWEN35_V2_PATH)
        loader = HFCheckpointLoader(str(checkpoint))

        with torch.device("meta"):
            xtuner = ConceptLMQwen35V2Config(compile_cfg=False).build()
            hf_config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
            hf = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)

        xt_stack = getattr(xtuner, stage)
        hf_stack = getattr(hf.model, stage)
        xt_layer = xt_stack.layers[str(layer_index)]
        hf_layer = hf_stack.layers[layer_index]
        local_prefix = f"{stage}.layers.{layer_index}."
        hf_prefix = f"model.{stage}.layers.{layer_index}."
        _materialize_and_load(
            xt_layer,
            loader,
            lambda name: xtuner.to_hf_key_list(local_prefix + name)[0],
        )
        _materialize_and_load(hf_layer, loader, lambda name: hf_prefix + name)
        _materialize_and_load(
            xt_stack.final_norm,
            loader,
            lambda name: xtuner.to_hf_key_list(f"{stage}.final_norm.{name}")[0],
        )
        _materialize_and_load(hf_stack.final_norm, loader, lambda name: f"model.{stage}.final_norm.{name}")
        _materialize_and_load(xtuner.lm_head, loader, lambda name: f"lm_head.{name}")
        _materialize_and_load(hf.lm_head, loader, lambda name: f"lm_head.{name}")

        torch.manual_seed(7)
        length = 8
        base_streams = tuple(
            torch.randn(1, length, xtuner.config.hidden_size, device="cuda", dtype=torch.bfloat16)
            for _ in range(4)
        )
        hf_streams = tuple(tensor.clone().requires_grad_(True) for tensor in base_streams)
        xt_streams = tuple(tensor.clone().requires_grad_(True) for tensor in base_streams)
        labels = torch.randint(0, xtuner.config.vocab_size, (length,), device="cuda")

        hf_output = hf_layer(hf_streams)
        hf_logits = hf.lm_head(hf_stack.final_norm(hf_output))
        hf_loss = F.cross_entropy(hf_logits.float().reshape(-1, hf.config.vocab_size), labels)
        hf_gradients = torch.autograd.grad(hf_loss, hf_streams, allow_unused=True)

        xt_output = xt_layer(
            xt_streams[3],
            position_embeddings=None,
            seq_ctx=_sequence_context(length),
            mudd_streams=xt_streams,
        )
        _, (xt_logits, _) = xtuner.lm_head(xt_stack.final_norm(xt_output), None)
        xt_logits = xt_logits.to(hf_logits.dtype)
        xt_loss = F.cross_entropy(xt_logits.float().reshape(-1, xtuner.config.vocab_size), labels)
        xt_gradients = torch.autograd.grad(xt_loss, xt_streams, allow_unused=True)

        torch.testing.assert_close(xt_output, hf_output, rtol=0, atol=0)
        torch.testing.assert_close(xt_loss, hf_loss, rtol=0, atol=0)
        for xt_gradient, hf_gradient in zip(xt_gradients, hf_gradients, strict=True):
            assert (xt_gradient is None) == (hf_gradient is None)
            if xt_gradient is not None and hf_gradient is not None:
                torch.testing.assert_close(xt_gradient, hf_gradient, rtol=0, atol=0)

        del hf, xtuner
        torch.cuda.empty_cache()

