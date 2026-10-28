"""FLA delta-rule wrapper numerical and dispatch regressions."""

import pytest
import torch

from xtuner.v1.module.attention.concept_qwen35 import _production_feature_enabled
from xtuner.v1.ops import gated_deltanet as gdn_ops


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
