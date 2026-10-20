"""CPU contract tests for captured inputs entering the native Trainer."""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "examples/v1/scripts/megatron_alignment"
sys.path.insert(0, str(SCRIPTS))
from batch_contract import batch_digest  # noqa: E402
from recorded_batch_source import RecordedBatchSource, batch_path, capture_metadata, collated_update  # noqa: E402
from replay_xtuner import collect_replay, main  # noqa: E402


def make_capture(root: Path, steps: int = 3, micros: int = 2, length: int = 16) -> dict[str, int]:
    (root / "batches").mkdir(parents=True)
    (root / "contract.json").write_text(json.dumps({"hf_path": "/unused/weights", "tokenizer": "/unused/tokenizer"}))
    for rank in range(8):
        completed = {str(step): micros for step in range(steps)}
        (root / f"megatron_complete.rank{rank:02d}.json").write_text(json.dumps(completed))
        for step in range(steps):
            for micro in range(micros):
                tokens = torch.arange(length)[None] + rank * 32 + step * 256 + micro * 1024
                mask = torch.ones(1, length)
                mask[0, 4] = 0
                payload = {
                    "tokens": tokens, "labels": tokens + 1, "loss_mask": mask,
                    "position_ids": torch.arange(length // 2).repeat(2)[None],
                    "cu_seqlens": torch.tensor([0, length // 2, length], dtype=torch.int32),
                    "step": step, "micro": micro, "rank": rank, "world_size": 8,
                    "global_batch_size": 8 * micros, "lr": 123.456 + step, "packed": True,
                }
                payload["sha256"] = batch_digest(payload)
                torch.save(payload, batch_path(root, step, micro, rank))
    return capture_metadata(root)


def native_context_class() -> Any:
    # SequenceContext itself has no GPU dependency. Omit only its SP helper
    # imports: SP1 from_input_ids/to do not call those communication functions.
    path = ROOT / "xtuner/v1/data_proto/sequence_context.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.level and node.module == "utils"
    )]
    namespace: dict[str, Any] = {}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["SequenceContext"]


CPUContext = native_context_class()

def prepare(root: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> dict[str, Any]:
    monkeypatch.setattr(sys, "argv", ["replay_xtuner.py", "--run-dir", str(root), *extra])
    main()
    sp_size = int(extra[extra.index("--sp-size") + 1]) if "--sp-size" in extra else 1
    return json.loads((root / f"xtuner_sp{sp_size}_native/training_config.json").read_text())


def native_outputs(root: Path, config: dict[str, Any]) -> None:
    sp_size = config["sp_size"]
    output = root / f"xtuner_sp{sp_size}_native"
    for rank in range(8):
        source = RecordedBatchSource(root, rank, config["capture_metadata"], sp_size)
        audits, rows = [], []
        for step in range(config["steps"]):
            payloads = source.next_update()
            audits.append({"step": step, "batch_sha256": [p["sha256"] for p in payloads],
                           "valid_targets": (config["sequence_length"] - 1) * config["microbatches_per_update"] * sp_size})
            rows.append({
                "step": step + 1, "lr": 8e-4 * step / 2000, "grad_norm": 2.0, "time/step_time": 0.5,
                "loss/reduced_total_loss": 8.0, "loss/reduced_lm_loss": 5.0,
                "loss/reduced_vq_token_loss": 1.0, "loss/reduced_hlm_token_loss": 2.0,
                "loss/valid_targets": (config["sequence_length"] - 1) * config["global_batch_size"],
            })
        (output / f"replay_inputs.rank{rank:02d}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in audits))
        tracker = output / f"logs/exp_tracking/rank{rank}/tracker.jsonl"
        tracker.parent.mkdir(parents=True)
        tracker.write_text("".join(json.dumps(row) + "\n" for row in rows))


def adapter_namespace() -> dict[str, Any]:
    path = SCRIPTS / "replay_trainer_config.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and (
            node.module.startswith("xtuner.") or node.module in {"train_qwen35", "pydantic"}
        )
    )]
    from pydantic import ConfigDict

    class BaseConfig(BaseModel):
        pack_level: str = "none"
        pack_max_length: int = 17
        pack_to_max_length: bool = True
        group_by_length: bool = False

    namespace = {"DataloaderConfig": BaseConfig, "BaseDataloader": object,
                 "SequenceContext": CPUContext, "TrainerConfig": Any, "ConfigDict": ConfigDict}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class TestNativeSP1Replay:
    def test_order_and_cursor_resume(self, tmp_path: Path) -> None:
        metadata = make_capture(tmp_path)
        source = RecordedBatchSource(tmp_path, 3, metadata)
        assert [p["micro"] for p in source.next_update()] == [0, 1]
        resumed = RecordedBatchSource(tmp_path, 3, metadata)
        resumed.load_state_dict(source.get_state_dict())
        assert [p["step"] for p in resumed.next_update()] == [1, 1]
        resumed.next_update()
        with pytest.raises(StopIteration):
            resumed.next_update()
        wrong_rank = RecordedBatchSource(tmp_path, 4, metadata)
        with pytest.raises(ValueError, match="another capture"):
            wrong_rank.load_state_dict(source.get_state_dict())

    def test_labels_masks_positions_preserved_without_shift(self, tmp_path: Path) -> None:
        source = RecordedBatchSource(tmp_path, 0, make_capture(tmp_path))
        payloads = source.next_update()
        items = collated_update(payloads, CPUContext)
        for payload, item in zip(payloads, items):
            torch.testing.assert_close(item["seq_ctx"].input_ids, payload["tokens"])
            torch.testing.assert_close(item["seq_ctx"].position_ids, payload["position_ids"])
            assert item["shifted_labels"][0, 0] == payload["labels"][0, 0]
            assert item["shifted_labels"].shape == (1, 16)
            assert item["shifted_labels"][0, 4] == -100
            torch.testing.assert_close(item["loss_mask"], payload["loss_mask"])
            assert "lr" not in item
            assert payload["labels"][0, 4] != -100  # Never mutate source tensors.

    @pytest.mark.parametrize("corruption", ["checksum", "identity", "shape"])
    def test_late_corrupt_payload_fails_before_delivery(self, tmp_path: Path, corruption: str) -> None:
        metadata = make_capture(tmp_path)
        path = batch_path(tmp_path, 1, 0, 0)
        payload = torch.load(path, weights_only=True)
        if corruption == "checksum":
            payload["tokens"][0, 0] += 1
        elif corruption == "identity":
            payload["micro"] = 1
            payload["sha256"] = batch_digest(payload)
        else:
            payload["global_batch_size"] *= 2
            payload["sha256"] = batch_digest(payload)
        torch.save(payload, path)
        source = RecordedBatchSource(tmp_path, 0, metadata)
        source.next_update()
        with pytest.raises(ValueError):
            source.next_update()
        assert source.cursor == 1

    def test_missing_file_or_different_completion_rejected(self, tmp_path: Path) -> None:
        make_capture(tmp_path)
        batch_path(tmp_path, 1, 1, 7).unlink()
        with pytest.raises(ValueError, match="batch files"):
            capture_metadata(tmp_path)
        (tmp_path / "megatron_complete.rank07.json").write_text('{"0": 2}')
        with pytest.raises(ValueError, match="incomplete"):
            capture_metadata(tmp_path)

    @pytest.mark.parametrize("style", ["wsd", "cosine"])
    def test_prepare_uses_independent_lr_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               style: str) -> None:
        make_capture(tmp_path)
        config = prepare(tmp_path, monkeypatch, "--lr-decay-style", style, "--lr", "0.0005")
        assert config["lr"] == 5e-4 and config["lr_source"] == "xtuner_scheduler"
        assert config["steps"] == 3 and config["decay_iters"] == 143052
        assert config["lr_decay_style"] == style and not config["profile_trace"]
        assert "lr" not in config["replayed_fields"]
        assert config["tokenizer_path"] is None
        ast.parse((tmp_path / "xtuner_sp1_native/trainer_config.py").read_text())
        with pytest.raises(FileExistsError):
            prepare(tmp_path, monkeypatch)

    def test_native_loader_adapter_and_checkpoint_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        metadata = make_capture(tmp_path)
        namespace = adapter_namespace()
        monkeypatch.setenv("WORLD_SIZE", "8")
        namespace["build_trainer_config"] = lambda config: SimpleNamespace(native_trainer=True)
        namespace["flash_attn_varlen_func"] = SimpleNamespace(__name__="gpu_flash_varlen_attn_v3")
        native_config = namespace["build_replay_config"]({
            "capture_dir": str(tmp_path), "run_dir": str(tmp_path), "capture_metadata": metadata,
            "sp_size": 1, "sequence_length": 16,
        })
        assert native_config.native_trainer
        assert native_config.dataloader_cfg.capture_metadata == metadata
        loader_config = namespace["RecordedBatchDataloaderConfig"](
            capture_dir=tmp_path, audit_dir=tmp_path, capture_metadata=metadata)
        mesh = SimpleNamespace(size=lambda: 8, get_local_rank=lambda: 0)
        loader = loader_config.build(None, mesh, 16, 2, seed=1234, total_step=3)
        item = next(iter(loader))
        assert len(item) == 2
        assert loader.get_state_dict()["cursor"] == 1
        with pytest.raises(ValueError, match="topology"):
            loader_config.build(None, mesh, 16, 1, seed=1234, total_step=3)
        state = loader.get_state_dict()
        loader.load_state_dict(state)
        list(loader)
        with pytest.raises(ValueError, match="exhausted"):
            loader.set_epoch(2)

    def test_export_keeps_fingerprints_and_actual_lr(self, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
        make_capture(tmp_path)
        config = prepare(tmp_path, monkeypatch)
        native_outputs(tmp_path, config)
        collect_replay(tmp_path)
        rows = [json.loads(line) for line in (tmp_path / "xtuner_sp1.jsonl").read_text().splitlines()]
        assert rows[0]["lr"] == 0  # Captured LR was 123.456, never used for training.
        payload = torch.load(batch_path(tmp_path, 0, 0, 3), weights_only=True)
        assert rows[0]["batch_sha256_by_rank"][3][0] == payload["sha256"]
        assert rows[0]["valid_targets"] == 15 * 16
        assert rows[0]["total_loss"] == 8 and rows[0]["accumulation_steps"] == 2
        done = json.loads((tmp_path / "xtuner_complete.json").read_text())
        assert done["steps"] == [0, 1, 2] and done["lr_source"] == "xtuner_scheduler"

    @pytest.mark.parametrize("corruption", ["missing_audit", "different_lr", "targets"])
    def test_incomplete_native_result_not_certified(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  corruption: str) -> None:
        make_capture(tmp_path)
        config = prepare(tmp_path, monkeypatch)
        native_outputs(tmp_path, config)
        output = tmp_path / "xtuner_sp1_native"
        if corruption == "missing_audit":
            path = output / "replay_inputs.rank03.jsonl"
            path.write_text(path.read_text().splitlines()[0] + "\n")
        else:
            path = output / "logs/exp_tracking/rank3/tracker.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["lr"] = 1 if corruption == "different_lr" else rows[0]["lr"]
            if corruption == "targets":
                path = output / "replay_inputs.rank03.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0]["valid_targets"] = 0
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(ValueError):
            collect_replay(tmp_path)
        assert not (tmp_path / "xtuner_complete.json").exists()
