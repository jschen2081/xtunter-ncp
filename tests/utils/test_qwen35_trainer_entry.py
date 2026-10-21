"""CPU checks for CLI preparation, native-loop scheduling and log export."""

import ast
import contextlib
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from test_native_sp1_replay import alignment_training_environment  # noqa: F401
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "examples/v1/scripts/megatron_alignment"
sys.path.insert(0, str(SCRIPTS))


def load_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENTRY = load_file("qwen35_cli_entry", SCRIPTS / "train_qwen35.py")
SCHEDULER = load_file("qwen35_native_scheduler", ROOT / "xtuner/v1/utils/lr_scheduler.py")
OBSERVER = load_file("qwen35_observer", SCRIPTS / "qwen35_training_observer.py")


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
    @pytest.mark.parametrize("trace", [False, True])
    def test_prepared_recipe_builds_native_trainer_config(
        self, tmp_path: Path, sp: int, trace: bool, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        output = tmp_path / "run"
        arguments = [
            "train_qwen35.py", "--run-dir", str(output), "--sp-size", str(sp),
            "--zero-stage", "2", "--gradient-accumulation-no-sync", "--compile", "--recompute-ratio", "0",
        ]
        if trace:
            arguments.append("--profile-trace")
        monkeypatch.setattr(sys, "argv", arguments)
        monkeypatch.setattr(ENTRY, "require_fla_version", lambda: "0.5.1")
        ENTRY.main()
        recipe = ENTRY.build_trainer_config(json.loads((output / "training_config.json").read_text()))
        assert recipe.grad_norm_dtype == torch.float32
        assert recipe.profile_step == ([6, 7] if trace else None)
        assert recipe.sp_size == sp
        assert recipe.total_step == 100
        assert not recipe.fsdp_cfg.reshard_after_forward
        assert recipe.fsdp_cfg.gradient_accumulation_no_sync

    def test_export_retains_full_precision_lr_and_native_tags(self, tmp_path: Path) -> None:
        config = {"steps": 2, "global_batch_size": 128, "sequence_length": 8192, "sp_size": 8}
        (tmp_path / "training_config.json").write_text(json.dumps(config))
        tracker = tmp_path / "logs/exp_tracking/rank0/tracker.jsonl"
        tracker.parent.mkdir(parents=True)
        rows = [{
            "step": index + 1, "lr": index * 8e-4 / 2000, "grad_norm": 2.0, "time/step_time": 1.0,
            "loss/reduced_total_loss": 8.0, "loss/reduced_lm_loss": 5.0,
            "loss/reduced_vq_token_loss": 1.0, "loss/reduced_hlm_token_loss": 2.0,
            "loss/valid_targets": 128 * 8192,
        } for index in range(2)]
        tracker.write_text("".join(json.dumps(row) + "\n" for row in rows))
        ENTRY.collect_metrics(tmp_path)
        result = [json.loads(row) for row in (tmp_path / "train_metrics.jsonl").read_text().splitlines()]
        assert [row["lr"] for row in result] == [row["lr"] for row in rows]
        assert [row["step"] for row in result] == [0, 1]
        assert result[0]["tokens_per_second_per_gpu"] == 128 * 8192 / 8
        assert result[0]["total_loss"] == 8
        assert (tmp_path / "native_training_complete.json").exists()

    def test_incomplete_export_does_not_certify_completion(self, tmp_path: Path) -> None:
        (tmp_path / "training_config.json").write_text(json.dumps({"steps": 2}))
        tracker = tmp_path / "logs/exp_tracking/rank0/tracker.jsonl"
        tracker.parent.mkdir(parents=True)
        tracker.write_text('{"step": 1}\n')
        with pytest.raises(ValueError, match="incomplete"):
            ENTRY.collect_metrics(tmp_path)
        assert not (tmp_path / "native_training_complete.json").exists()

    def test_observer_only_adds_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(OBSERVER.dist, "all_reduce", lambda value: None)
        observer = OBSERVER.ConceptLMTrainingObserver()
        observer.sp_size = 8
        stats = {"concept_vq_loss": torch.tensor([0.5, 0.5]),
                 "concept_hlm_loss": torch.tensor([1.0, 1.0]),
                 "concept_valid_targets": torch.tensor([32, 32])}
        info = {"extra_info": stats, "logs_info": {"reduced_llm_loss": 8.0}}
        before = {key: value.clone() for key, value in stats.items()}
        observer(info, 0, 1, 2, None)
        assert info["logs_info"]["reduced_lm_loss"] == 5.0
        assert info["logs_info"]["valid_targets"] == 8
        for key in stats:
            torch.testing.assert_close(stats[key], before[key], rtol=0, atol=0)

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
        from _cpu_source_loader import load_ast
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
