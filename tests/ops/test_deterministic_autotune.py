"""CPU checks for kernel pinning before the model/FLA import chain runs."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("factory", ["decorator", "subclass"])
def test_determinism_pins_direct_autotuner_subclasses(factory: str) -> None:
    # A fresh interpreter verifies initialization order without inheriting the
    # test parent's already-patched Triton/FLA modules. No CUDA launch is needed.
    patch_path = Path(__file__).resolve().parents[2] / "xtuner/v1/__init__.py"
    script = textwrap.dedent(
        """
        import importlib.util
        import os
        import sys
        from types import ModuleType, SimpleNamespace

        import triton
        import triton.runtime.autotuner as autotuner
        from triton.runtime.autotuner import Autotuner

        # Triton 3.2's decorator drops do_bench. Stub only the benchmark
        # provider so constructing a tuner remains a CPU-only operation.
        autotuner.driver = SimpleNamespace(
            active=SimpleNamespace(get_benchmarker=lambda: lambda *a, **k: 0)
        )
        configs = [triton.Config({}, num_warps=value) for value in (1, 2, 4)]

        def kernel(H):
            pass

        kernel.arg_names = ["H"]

        class CachedAutotuner(Autotuner):
            # FLA 0.5 invokes this path instead of triton.autotune.
            pass

        def construct():
            if sys.argv[2] == "decorator":
                return triton.autotune(configs, key=["H"], do_bench=lambda *a, **k: 0)(kernel)
            return CachedAutotuner(kernel, kernel.arg_names, configs, ["H"], None, None,
                                   do_bench=lambda *a, **k: 0)

        assert len(construct().configs) == 3
        os.environ["FLA_CACHE_MODE"] = "full"
        os.environ["XTUNER_DETERMINISTIC"] = "true"
        # Exercise the actual package initializer while isolating unrelated
        # CUDA/distributed patches from this CPU regression.
        sys.modules["xtuner"] = ModuleType("xtuner")
        sys.modules["xtuner.v1.patch"] = ModuleType("xtuner.v1.patch")
        spec = importlib.util.spec_from_file_location("xtuner.v1", sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        patch = module._patch_triton_autotune_for_determinism
        assert os.environ["FLA_CACHE_MODE"] == "disabled"
        tuner = construct()
        assert len(tuner.configs) == 1
        assert tuner.configs[0].num_warps == 1
        assert len(configs) == 3
        initializer, decorator = Autotuner.__init__, triton.autotune
        patch()
        assert Autotuner.__init__ is initializer
        assert triton.autotune is decorator
        assert len(construct().configs) == 1
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(patch_path), factory], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
