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
from itertools import accumulate
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
    @pytest.fixture
    def loss_config(self):
        # Load the real loss classes while omitting package initializers only.
        base = load_ast("xtuner/v1/loss/base_loss_ctx.py", {"Parameter": lambda **kwargs: None}, ("cyclopts",))
        seq = load_ast(
            "xtuner/v1/module/concept_sequence.py", {"SequenceContext": object}, ("xtuner.v1.data_proto",)
        )
        base["padded_document_lengths"] = seq["padded_document_lengths"]
        loss = load_ast("xtuner/v1/loss/conceptlm_loss.py", base, (
            "base_loss_ctx", "xtuner.v1.module.concept_quantizer", "xtuner.v1.module.concept_sequence",
        ))
        return loss["ConceptLMLossConfig"]()

    @staticmethod
    def loss_data(lengths: list[int], sp_mesh=None, valid_length: int | None = None):
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        return {
            "seq_ctx": SimpleNamespace(
                input_ids=torch.zeros(1, sum(lengths), dtype=torch.long),
                cu_seq_lens_q=torch.tensor(boundaries, dtype=torch.int32),
                sequence_parallel_mesh=sp_mesh,
                valid_sequence_length=valid_length,
            ),
            "shifted_labels": torch.zeros(1, sum(lengths), dtype=torch.long),
        }

    @pytest.mark.parametrize("chunk_size", [2, 4, 8])
    @pytest.mark.parametrize("chunks_per_document", [(1,), (2, 1), (1, 2)])
    def test_pads_documents_shorter_than_two_chunks(self, loss_config, chunk_size, chunks_per_document):
        # The model pads short documents with pad tokens; the loss grows its mask
        # the same way with zeros so padded concepts carry no valid target.
        loss_config.concept_chunk_size = chunk_size
        lengths = [chunks * chunk_size for chunks in chunks_per_document]
        data = self.loss_data(lengths)
        result = loss_config.build(data)
        padded = [max(2 * chunk_size, length) for length in lengths]
        assert result.loss_kwargs.boundaries.tolist() == [0, *accumulate(padded)]
        assert result.loss_kwargs.loss_mask.shape[1] == sum(padded)
        assert int(result.loss_kwargs.loss_mask.sum()) == sum(lengths)

    @pytest.mark.parametrize("chunks_per_document", [(0, 2), (2, 0)])
    def test_rejects_empty_documents(self, loss_config, chunks_per_document):
        data = self.loss_data([chunks * loss_config.concept_chunk_size for chunks in chunks_per_document])
        with pytest.raises(ValueError, match="empty documents"):
            loss_config.build(data)

    def test_pads_chunk_unaligned_document(self, loss_config):
        data = self.loss_data([10])
        result = loss_config.build(data)
        assert result.loss_kwargs.boundaries.tolist() == [0, 12]
        assert int(result.loss_kwargs.loss_mask.sum()) == 10
        assert int(result.loss_kwargs.loss_mask[:, 10:].sum()) == 0

    @pytest.mark.parametrize("sp_size", [2, 4, 8])
    @pytest.mark.parametrize("from_context", [False, True])
    def test_extends_last_document_for_sp_tail_padding(self, loss_config, sp_size, from_context):
        mesh = SimpleNamespace(size=lambda: sp_size)
        # Each document is at least two chunks, but the global total needs SP padding.
        lengths = [8, sp_size * 4 + 4]
        data = self.loss_data(lengths, mesh if from_context else None)
        result = loss_config.build(data, sp_mesh=None if from_context else mesh)
        total = sum(lengths)
        tail = (-total) % (sp_size * 4)
        assert result.loss_kwargs.boundaries.tolist() == [0, lengths[0], total + tail]
        assert result.loss_kwargs.loss_mask.shape[1] == total + tail
        assert int(result.loss_kwargs.loss_mask.sum()) == total
        assert int(result.loss_kwargs.loss_mask[:, total:].sum()) == 0

    @pytest.mark.parametrize("chunk_size", [2, 4, 8])
    @pytest.mark.parametrize("sp_size", [1, 2, 4, 8])
    def test_accepts_aligned_documents_without_model_padding(self, loss_config, chunk_size, sp_size):
        loss_config.concept_chunk_size = chunk_size
        lengths = [2 * chunk_size * sp_size] * 2
        data = self.loss_data(lengths)
        result = loss_config.build(data, sp_mesh=SimpleNamespace(size=lambda: sp_size))
        assert result is not None
        torch.testing.assert_close(result.loss_kwargs.boundaries, data["seq_ctx"].cu_seq_lens_q)
        torch.testing.assert_close(result.loss_kwargs.loss_mask, data["shifted_labels"].ne(-100))

    @pytest.mark.parametrize("sp_size", [2, 4, 8])
    def test_sp_scopes_loss_to_valid_region_beyond_pack_padding(self, loss_config, sp_size):
        # The collator records pack padding as extra documents beyond the valid
        # region; the SP model path strips them, so the loss must scope to the
        # same real documents instead of crashing on a concept count mismatch.
        mesh = SimpleNamespace(size=lambda: sp_size)
        valid = 2 * 4 * sp_size
        data = self.loss_data([valid, 256], sp_mesh=mesh, valid_length=valid)
        result = loss_config.build(data)
        torch.testing.assert_close(
            result.loss_kwargs.boundaries, torch.tensor([0, valid], dtype=torch.int32)
        )
        assert result.loss_kwargs.loss_mask.shape[1] == valid

    def test_sp_scoping_pads_misaligned_valid_region(self, loss_config):
        # A valid region that is not SP-aligned is padded on its last document,
        # matching the model's tail padding, with the padded concepts masked out.
        mesh = SimpleNamespace(size=lambda: 2)
        data = self.loss_data([12, 8], sp_mesh=mesh, valid_length=12)
        result = loss_config.build(data)
        assert result.loss_kwargs.boundaries.tolist() == [0, 16]
        assert result.loss_kwargs.loss_mask.shape[1] == 16
        assert int(result.loss_kwargs.loss_mask.sum()) == 12
        assert int(result.loss_kwargs.loss_mask[:, 12:].sum()) == 0

    def test_non_sp_keeps_pack_padding_documents(self, loss_config):
        # Without SP the model keeps padding documents; the loss must too.
        data = self.loss_data([16, 8])
        result = loss_config.build(data)
        torch.testing.assert_close(
            result.loss_kwargs.boundaries, torch.tensor([0, 16, 24], dtype=torch.int32)
        )

    def test_cpu_labels_follow_sequence_device(self, loss_config) -> None:
        labels = torch.zeros(1, 8, dtype=torch.long)
        context = SimpleNamespace(input_ids=torch.empty(1, 8, device="meta"),
                                  cu_seq_lens_q=torch.tensor([0, 8]))
        result = loss_config.build({"seq_ctx": context, "shifted_labels": labels})
        assert result.loss_kwargs.loss_mask.device.type == "meta"
        assert labels.device.type == "cpu"
