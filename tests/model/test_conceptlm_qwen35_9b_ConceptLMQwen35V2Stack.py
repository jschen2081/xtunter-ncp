"""Cross-MUDD cache equivalence using real model routes and stack code."""

import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from xtuner.v1.model.dense import conceptlm_qwen35_9b as model
from xtuner.v1.module.rms_norm import concept_qwen35 as norm


ns = vars(model)
Stack = model.ConceptLMQwen35V2Stack


class LayerStub(nn.Module):
    """Small differentiable attention/MLP replacement; real route/stack code is used."""

    def __init__(self, config, index):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden, position_embeddings, seq_ctx, mudd_streams):
        q, k, v, residual = mudd_streams
        return residual + 0.2 * torch.tanh(self.proj(q + k * 0.3 + v * 0.2))


def make_stack(groups=(2, 3), inject=True):
    config = SimpleNamespace(
        self_mudd_block_size=2, rms_norm_eps=1e-6, cross_mudd_epsilon=1e-4, hidden_size=24, concept_route_epsilon=1e-5
    )
    depth = 4
    routes = nn.ModuleList(ns["SelfMUDDRoute"](24, i + 1, depth, 2, 1e-6) for i in range(depth))
    # Scope the tiny layer replacement to construction; other tests keep the real builder.
    with patch.object(Stack, "_build_layer", staticmethod(LayerStub)):
        stack = Stack(config, depth, routes, groups, inject)
    with torch.no_grad():
        for p in list(stack.parameters()) + list(routes.parameters()):
            p.add_(torch.randn_like(p) * 0.1)
    return stack


def params(stack):
    return list(stack.named_parameters()) + [("self_routes." + n, p) for n, p in stack._self_routes.named_parameters()]


class TestCrossSourceCache(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(113)

    def check_pair(
        self, device="cpu", dtype=torch.float32, groups=(3, 5), inject=True, compiled=False, alternate=False
    ):
        with patch.dict(
            os.environ, {"NCP_Q35V2_CACHE_CROSS_SOURCE_RMS": "0", "NCP_Q35V2_MUDD_RMS": "0" if alternate else "1"}
        ):
            baseline = make_stack(groups, inject).to(device=device, dtype=dtype)
            baseline._self_routes.to(device=device, dtype=dtype)
            cached = copy.deepcopy(baseline)
            cached.cache_cross_source_rms = True
            count = 1 + sum(groups) + int(inject)
            inputs = [torch.randn(2, 11, 24, device=device, dtype=dtype, requires_grad=True) for _ in range(count)]
            copies = [x.detach().clone().requires_grad_() for x in inputs]

            def run(stack, tensors):
                banks = []
                start = 1
                for size in groups:
                    banks.append(tuple(tensors[start : start + size]))
                    start += size
                return stack(tensors[0], None, tuple(banks), tensors[-1] if inject else None)

            if compiled:
                run = torch.compile(run, backend="inductor" if device == "cuda" else "aot_eager", fullgraph=True)
            ref, raw_ref = run(baseline, inputs)
            got, raw_got = run(cached, copies)
            tolerance = dict(rtol=4e-5, atol=3e-5) if dtype == torch.float32 else dict(rtol=0.05, atol=0.03)
            for a, b in zip((got, *raw_got), (ref, *raw_ref)):
                torch.testing.assert_close(a, b, **tolerance)
            grads = [torch.randn_like(x) for x in (ref, *raw_ref)]
            torch.autograd.backward((ref, *raw_ref), grads)
            torch.autograd.backward((got, *raw_got), grads)
            for i, (a, b) in enumerate(zip(copies, inputs)):
                torch.testing.assert_close(a.grad, b.grad, **tolerance, msg=lambda m: f"input {i}: {m}")
            for (na, a), (nb, b) in zip(params(cached), params(baseline)):
                self.assertEqual(na, nb)
                torch.testing.assert_close(a.grad, b.grad, **tolerance, msg=lambda m: f"{na}: {m}")
            self.assertEqual(list(cached.state_dict()), list(baseline.state_dict()))
            return baseline, cached

    def test_cpu_forward_backward(self):
        for dtype in [torch.float32, torch.bfloat16]:
            for groups, inject in [((3, 5), True), ((3,), False), ((), False)]:
                with self.subTest(dtype=dtype, groups=groups):
                    self.check_pair(dtype=dtype, groups=groups, inject=inject)

    def test_optimizer_update_and_fresh_inputs(self):
        baseline, cached = self.check_pair()
        for model in (baseline, cached):
            optimizer = torch.optim.SGD([p for _, p in params(model)], lr=0.01)
            optimizer.step()
        for (na, a), (nb, b) in zip(params(baseline), params(cached)):
            self.assertEqual(na, nb)
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)
        banks = tuple(tuple(torch.randn(2, 7, 24) for _ in range(n)) for n in (3, 5))
        hidden, concept = torch.randn(2, 7, 24), torch.randn(2, 7, 24)
        a, _ = baseline(hidden, None, banks, concept)
        b, _ = cached(hidden, None, banks, concept)
        torch.testing.assert_close(a, b, rtol=4e-5, atol=3e-5)

    def test_fullgraph_cpu(self):
        self.check_pair(compiled=True)

    def test_alternate_rms_falls_back(self):
        self.check_pair(alternate=True)

    def test_cache_normalizes_each_source_once(self):
        with patch.dict(os.environ, {"NCP_Q35V2_CACHE_CROSS_SOURCE_RMS": "1", "NCP_Q35V2_MUDD_RMS": "1"}):
            model = make_stack(inject=False)
            original = ns["_cross_source_units"]
            seen = []

            def record(bank, eps):
                seen.append((len(bank), eps))
                return original(bank, eps)

            with patch.dict(ns, {"_cross_source_units": record}):
                # Function globals are ns; cache entries must be rebuilt on every call.
                for _ in range(2):
                    banks = tuple(tuple(torch.randn(1, 7, 24) for _ in range(n)) for n in (2, 3))
                    model(torch.randn(1, 7, 24), None, banks)
            self.assertEqual(seen, [(2, 1e-4), (3, 1e-4)] * 2)
            self.assertFalse(any("cache" in name for name in model.state_dict()))
        with patch.dict(os.environ, {"NCP_Q35V2_CACHE_CROSS_SOURCE_RMS": "0"}):
            self.assertFalse(make_stack().cache_cross_source_rms)

    def test_normalization_dtype_boundaries(self):
        with patch.dict(os.environ, {"NCP_Q35V2_MUDD_RMS": "1"}):
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                for weight_dtype in {dtype, torch.float32}:
                    x = torch.randn(2, 24, 7, dtype=dtype).transpose(1, 2).requires_grad_()
                    w = torch.randn(24, dtype=weight_dtype, requires_grad=True)
                    unit = ns["_cross_source_units"]((x,), 1e-4)[0]
                    self.assertEqual(unit.dtype, torch.float32)
                    # Apply gamma in accumulation precision, then round once as
                    # weighted RMSNorm does (including mixed-dtype weights).
                    actual = (unit * norm.effective_weight(w).float()).to(x.dtype)
                    expected = norm.mudd_rms_norm(x, w, 1e-4)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    grad = torch.randn_like(expected)
                    for a, b in zip(
                        torch.autograd.grad(actual, (x, w), grad, retain_graph=True),
                        torch.autograd.grad(expected, (x, w), grad),
                    ):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_unit_gradients_noncontiguous_and_alias(self):
        x = torch.randn(2, 24, 7).transpose(1, 2).requires_grad_()
        w0, w1 = [torch.randn(24, requires_grad=True) for _ in range(2)]
        unit = ns["_cross_source_units"]((x,), 1e-4)[0]
        actual = unit * w0 + unit * w1
        expected = F.rms_norm(x, (24,), w0, 1e-4) + F.rms_norm(x, (24,), w1, 1e-4)
        for a, b in zip(
            torch.autograd.grad(actual.sum(), (x, w0, w1), retain_graph=True),
            torch.autograd.grad(expected.sum(), (x, w0, w1)),
        ):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)

    @pytest.mark.gpu
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available on this host")
    def test_cuda_and_inductor(self):
        for dtype in [torch.float32, torch.bfloat16]:
            self.check_pair(device="cuda", dtype=dtype)
            self.check_pair(device="cuda", dtype=dtype, compiled=True)


if __name__ == "__main__":
    unittest.main()
