"""CPU checks for the Qwen3.5 FLA runtime requirement."""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def runtime_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/v1/scripts/megatron_alignment/runtime_requirements.py"
    )
    spec = importlib.util.spec_from_file_location("qwen35_runtime_requirements", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFlaRequirement:
    @pytest.mark.parametrize("version", ["0.4.2", "0.5.0", "0.5.1rc1", "0.5.1.dev1"])
    def test_rejects_old_or_prerelease_version(
        self, runtime_module: ModuleType, monkeypatch: pytest.MonkeyPatch, version: str
    ) -> None:
        module = SimpleNamespace(__version__=version, __file__="/selected/fla/__init__.py")
        monkeypatch.setattr(runtime_module, "import_module", lambda name: module)
        with pytest.raises(RuntimeError, match=r"requires FLA >= 0\.5\.1"):
            runtime_module.require_fla_version()

    @pytest.mark.parametrize("version", ["0.5.1", "0.5.1.post1", "0.5.1+cu128", "0.5.2", "0.6.0", "0.10.0"])
    def test_accepts_supported_version(
        self, runtime_module: ModuleType, monkeypatch: pytest.MonkeyPatch, version: str
    ) -> None:
        module = SimpleNamespace(__version__=version, __file__="/selected/fla/__init__.py")
        monkeypatch.setattr(runtime_module, "import_module", lambda name: module)
        assert runtime_module.require_fla_version() == version

    @pytest.mark.parametrize("version", [None, "unknown"])
    def test_rejects_unknown_version(
        self, runtime_module: ModuleType, monkeypatch: pytest.MonkeyPatch, version: str | None
    ) -> None:
        module = SimpleNamespace(__file__="/selected/fla/__init__.py")
        if version is not None:
            module.__version__ = version
        monkeypatch.setattr(runtime_module, "import_module", lambda name: module)
        with pytest.raises(RuntimeError, match="cannot determine a valid FLA version"):
            runtime_module.require_fla_version()

    @pytest.mark.parametrize("failure", [ImportError("missing fla"), RuntimeError("broken runtime")])
    def test_rejects_unimportable_runtime(
        self, runtime_module: ModuleType, monkeypatch: pytest.MonkeyPatch, failure: Exception
    ) -> None:
        def fail_import(name: str) -> None:
            raise failure

        monkeypatch.setattr(runtime_module, "import_module", fail_import)
        with pytest.raises(RuntimeError, match="cannot import fla"):
            runtime_module.require_fla_version()
