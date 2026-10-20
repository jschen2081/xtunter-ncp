"""Load real library CPU code without importing GPU-only package initializers.

Cyclopts annotations describe CLI help only. When it is absent on a CPU host,
omit that metadata while retaining Pydantic's actual validation and defaults.
No numerical implementation is replaced here.
"""
import ast
import importlib.util
import logging
import sys
import subprocess
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_ast(relative, namespace, skip_imports=()):
    tree = ast.parse((ROOT / relative).read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and (node.module or "") in skip_imports)]
    namespace = dict(namespace)
    namespace.setdefault("__name__", "cpu_library_test")
    exec(compile(tree, str(ROOT / relative), "exec"), namespace)
    return namespace


def load_libraries():
    load_file("xtuner.v1.utils.dtensor", "xtuner/v1/utils/dtensor.py")
    native = load_file("xtuner.v1.optim.muon", "xtuner/v1/optim/muon.py")
    megatron = load_file("xtuner.v1.optim.concept_qwen35_muon", "xtuner/v1/optim/concept_qwen35_muon.py")
    try:
        from cyclopts import Parameter
    except ImportError:
        Parameter = lambda **kwargs: None
    base = load_ast("xtuner/v1/loss/base_loss_ctx.py", {"Parameter": Parameter}, ("cyclopts",))
    quantizer = load_file("xtuner.v1.module.concept_quantizer", "xtuner/v1/module/concept_quantizer.py")
    sequence = load_ast("xtuner/v1/module/concept_sequence.py", {"SequenceContext": object},
                        ("xtuner.v1.data_proto",))
    base.update(quantize_concepts=quantizer.quantize_concepts, next_concept_mse=sequence["next_concept_mse"])
    loss = load_ast("xtuner/v1/loss/conceptlm_loss.py", base,
                    ("base_loss_ctx", "xtuner.v1.module.concept_quantizer", "xtuner.v1.module.concept_sequence"))
    tree = ast.parse((ROOT / "xtuner/v1/model/dense/conceptlm_qwen35_9b.py").read_text())
    model_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                       and node.name == "ConceptLMQwen35V2")
    metadata = next(node for node in model_class.body if isinstance(node, ast.FunctionDef)
                    and node.name == "get_muon_metadata")
    metadata_ns = {}
    exec(compile(ast.Module(body=[metadata], type_ignores=[]), "model_muon_metadata", "exec"), metadata_ns)
    optim = load_ast("xtuner/v1/config/optim.py", {
        "Parameter": Parameter, "Muon": native.Muon, "MegatronMuon": megatron.MegatronMuon,
        "get_logger": lambda: logging.getLogger("cpu_optimizer_test"),
    }, ("cyclopts", "xtuner.v1.optim", "xtuner.v1.utils"))
    return native, megatron, loss, metadata_ns["get_muon_metadata"], optim


def load_legacy_muon():
    """Independent numerical reference from the unchanged repository commit."""
    source = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", "HEAD:xtuner/v1/optim/muon.py"], text=True,
    )
    module = types.ModuleType("legacy_muon_reference")
    sys.modules[module.__name__] = module
    exec(compile(source, "legacy_muon_reference", "exec"), module.__dict__)
    return module, source
