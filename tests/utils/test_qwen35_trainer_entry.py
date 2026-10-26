"""CPU checks for training configuration, LR timing and loss context devices."""

import ast
import contextlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "examples/v1/scripts/qwen35_9b_alignment"
sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(autouse=True)
def alignment_training_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use shell defaults for direct Python recipe preparation in CPU tests."""
    scripts = Path(__file__).resolve().parents[2] / "examples/v1/scripts"
    module = scripts / "qwen35_9b_alignment/training_environment.py"
    tree = ast.parse(module.read_text())
    names = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "TRAINING_ENV_NAMES" for target in node.targets)
    )
    launcher = (scripts / "run_qwen35_v2_9b_8gpu.sh").read_text()
    block = launcher.split("# Kernel environment: caller values take precedence.\n", 1)[1]
    block = block.split("# Validate before third-party imports.", 1)[0]
    env = {name: value for name, value in os.environ.items() if name not in names}
    command = block + "\n" + "exec \"$1\" -c 'import json,os; print(json.dumps(dict(os.environ)))'"
    result = subprocess.run(
        ["bash", "-c", command, "bash", sys.executable],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    effective = json.loads(result.stdout)
    for name in names:
        monkeypatch.setenv(name, effective[name])


def load_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ast(
    relative: str, namespace: dict[str, Any], skip_imports: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Load real source while omitting selected imports for CPU-only tests."""
    tree = ast.parse((ROOT / relative).read_text())
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and (node.module or "") in skip_imports)
    ]
    namespace = dict(namespace)
    namespace.setdefault("__name__", "cpu_library_test")
    exec(compile(tree, str(ROOT / relative), "exec"), namespace)
    return namespace


ENTRY = load_file("qwen35_cli_entry", SCRIPTS / "train_qwen35.py")
SCHEDULER = load_file("qwen35_native_scheduler", ROOT / "xtuner/v1/utils/lr_scheduler.py")


class TestQwen35TrainerEntry:
    @pytest.mark.parametrize("cli", [False, True])
    def test_loss_chunk_size_environment_and_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli: bool,
    ) -> None:
        monkeypatch.setenv("LOSS_CHUNK_SIZE", "1024")
        output = tmp_path / "run"
        arguments = ["train_qwen35.py", "--run-dir", str(output)]
        if cli:
            arguments += ["--lm-head-chunk-size", "256"]
        monkeypatch.setattr(sys, "argv", arguments)
        ENTRY.main()
        config = json.loads((output / "training_config.json").read_text())
        assert config["lm_head_chunk_size"] == (256 if cli else 1024)

    @pytest.mark.parametrize("style", ["WSD", "cosine"])
    @pytest.mark.parametrize("sp", [1, 8])
    def test_cpu_preparation_and_native_cli_config(
        self, tmp_path: Path, style: str, sp: int, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        output = tmp_path / "run"
        monkeypatch.setattr(sys, "argv", [
            "train_qwen35.py", "--run-dir", str(output), "--sp-size", str(sp), "--lr-decay-style", style,
        ])
        ENTRY.main()
        config = json.loads((output / "training_config.json").read_text())
        assert config["lr_decay_style"] == style.lower()
        assert config["sp_size"] == sp
        assert config["steps"] == 100 and config["decay_iters"] == 143052
        assert not config["profile_trace"]
        ast.parse((output / "trainer_config.py").read_text())

    @pytest.mark.parametrize("sp", [1, 8])
    def test_prepared_recipe_builds_native_trainer_config(
        self, tmp_path: Path, sp: int, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        output = tmp_path / "run"
        arguments = [
            "train_qwen35.py", "--run-dir", str(output), "--sp-size", str(sp),
            "--zero-stage", "2", "--gradient-accumulation-no-sync", "--compile", "--recompute-ratio", "0",
        ]
        monkeypatch.setattr(sys, "argv", arguments)
        monkeypatch.setattr(ENTRY, "require_fla_version", lambda: "0.5.1")
        ENTRY.main()
        recipe = ENTRY.build_trainer_config(json.loads((output / "training_config.json").read_text()))
        assert recipe.grad_norm_dtype == torch.float32
        assert recipe.sp_size == sp
        assert recipe.total_step == 100
        assert not recipe.fsdp_cfg.reshard_after_forward
        assert recipe.fsdp_cfg.gradient_accumulation_no_sync


    @pytest.mark.parametrize("style", ["wsd", "cosine"])
    def test_real_trainer_fit_uses_lr_before_advancing(self, style: str) -> None:
        source = ast.parse((ROOT / "xtuner/v1/train/trainer.py").read_text())
        trainer_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "Trainer")
        fit = next(node for node in trainer_class.body if isinstance(node, ast.FunctionDef) and node.name == "fit")
        prober = SimpleNamespace(set_step=lambda step: None, after_step=lambda: None)
        namespace = {"time": time, "ProberList": prober,
                     "DEVICE_MODULE": SimpleNamespace(reset_peak_memory_stats=lambda: None),
                     "dist": SimpleNamespace(barrier=lambda: None),
                     "log_rank0": logging.getLogger("trainer_fit_cpu"),
                     "HookStage": SimpleNamespace(AFTER_TRAIN_STEP="after_train_step")}
        exec(compile(ast.Module(body=[fit], type_ignores=[]), "native_trainer_fit", "exec"), namespace)
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=8e-4)
        config = SimpleNamespace(lr_type=style, lr_min=8e-5, warmup_steps=4, warmup_ratio=0,
                                 decay_steps=40, wsd_decay_steps=8 if style == "wsd" else None)
        scheduler = SCHEDULER.build_lr_scheduler(optimizer, config, total_steps=43)
        used = []

        class Engine:
            def train_step(self, items: Any) -> dict[str, int]:
                return {"step_consumed_tokens": 8192, "step_seqlen_tokens": 8192}

            def clip_grad_norm(self, **kwargs: Any) -> torch.Tensor:
                return torch.tensor(1.0)

            def step_optimizer(self, grad_norm: torch.Tensor) -> None:
                used.append(optimizer.param_groups[0]["lr"])
                optimizer.step()

        class Harness:
            fit = namespace["fit"]
            cur_step = property(lambda self: self._cur_step)

        runner = Harness()
        runner.__dict__.update(
            _async_hf_export=False, _async_checkpoint=False, _cur_step=0, _cur_epoch=1,
            _local_total_consumed_tokens=0, _offload_optimizer_first_step=False,
            _metrics_recorder=None, _do_clip=True, _grad_norm_dtype=torch.float32,
            _engine=Engine(), _lr_scheduler=scheduler,
            hooks_config=SimpleNamespace(get_hooks=lambda stage: []),
            _data_iter=lambda: iter(range(43)), _prepare_model_input=lambda batch: batch,
            _maybe_profiling=contextlib.nullcontext,
            _maybe_pop_model_internal_metrics=lambda batch: None,
            _compute_performance_metrics=lambda **kwargs: {}, _log_step=lambda **kwargs: None,
            _maybe_check_health=lambda: None, _maybe_save_hf=lambda: None, _maybe_save=lambda **kwargs: True,
            _exp_tracker=SimpleNamespace(close=lambda: None),
        )
        runner.fit()
        assert len(used) == 43
        assert used[0] == 0 and used[4] == 8e-4
        assert used[40:] == [8e-5] * 3
        if style == "wsd":
            assert used[4:33] == [8e-4] * 29
        else:
            assert used[5] < used[4]


class TestConceptLMNativeContext:
    def test_cpu_labels_follow_sequence_device(self) -> None:
        # Load the real loss classes while omitting package initializers only.
        base = load_ast("xtuner/v1/loss/base_loss_ctx.py", {"Parameter": lambda **kwargs: None}, ("cyclopts",))
        loss = load_ast("xtuner/v1/loss/conceptlm_loss.py", base, (
            "base_loss_ctx", "xtuner.v1.module.concept_quantizer", "xtuner.v1.module.concept_sequence",
        ))
        labels = torch.zeros(1, 8, dtype=torch.long)
        context = SimpleNamespace(input_ids=torch.empty(1, 8, device="meta"),
                                  cu_seq_lens_q=torch.tensor([0, 8]))
        result = loss["ConceptLMLossConfig"]().build({"seq_ctx": context, "shifted_labels": labels})
        assert result.loss_kwargs.loss_mask.device.type == "meta"
        assert labels.device.type == "cpu"
