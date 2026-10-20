"""CPU checks for SP8 captured input regrouping and native Trainer routing."""

import ast
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[2] / "examples/v1/scripts/megatron_alignment"
sys.path.insert(0, str(SCRIPTS))

from recorded_batch_source import RecordedBatchSource, batch_path, collated_update  # noqa: E402
from replay_xtuner import collect_replay  # noqa: E402
from test_native_sp1_replay import (  # noqa: E402
    CPUContext,
    adapter_namespace,
    make_capture,
    native_outputs,
    prepare,
)


class TestNativeSP8Replay:
    @pytest.mark.parametrize("micros", [1, 2])
    def test_all_peers_read_same_global_batch_in_original_order(self, tmp_path: Path, micros: int) -> None:
        metadata = make_capture(tmp_path, micros=micros, length=32)
        expected = [(micro, rank) for micro in range(micros) for rank in range(8)]
        reference = None
        for rank in range(8):
            source = RecordedBatchSource(tmp_path, rank, metadata, sp_size=8)
            payloads = source.next_update()
            assert [(p["micro"], p["rank"]) for p in payloads] == expected
            fingerprints = [p["sha256"] for p in payloads]
            if reference is None:
                reference = fingerprints
            assert fingerprints == reference
            items = collated_update(payloads, CPUContext)
            for payload, item in zip(payloads, items):
                torch.testing.assert_close(item["seq_ctx"].input_ids, payload["tokens"])
                torch.testing.assert_close(item["seq_ctx"].position_ids, payload["position_ids"])
                torch.testing.assert_close(
                    item["shifted_labels"], payload["labels"].masked_fill(payload["loss_mask"] == 0, -100)
                )
                assert "lr" not in item
            restored = RecordedBatchSource(tmp_path, rank, metadata, sp_size=8)
            restored.load_state_dict(source.get_state_dict())
            assert {p["step"] for p in restored.next_update()} == {1}
            with pytest.raises(ValueError, match="another capture"):
                RecordedBatchSource(tmp_path, rank, metadata).load_state_dict(source.get_state_dict())

    def test_native_trainer_splits_captured_sequences_without_repacking(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        metadata = make_capture(tmp_path, length=32)
        # Execute the real native input preparation and SP helpers on CPU;
        # only the process mesh/rank and model's loss-context return are stand-ins.
        utils = runpy.run_path(str(root / "xtuner/v1/data_proto/utils.py"))
        for name in ("pad_to_multiple_of", "split_for_sequence_parallel"):
            monkeypatch.setitem(CPUContext.split.__globals__, name, utils[name])
        tree = ast.parse((root / "xtuner/v1/train/trainer.py").read_text())
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Trainer")
        method = next(
            node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_prepare_model_input"
        )
        namespace = {"SequenceContext": CPUContext, "ModelItem": SimpleNamespace, "DEVICE": "cpu"}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "native_sp8_input", "exec"), namespace)
        shards = []
        for rank in range(8):
            monkeypatch.setattr(utils["dist"], "get_rank", lambda group=None, rank=rank: rank)
            mesh = SimpleNamespace(size=lambda: 8, get_local_rank=lambda rank=rank: rank, get_group=lambda: None)
            payloads = RecordedBatchSource(tmp_path, rank, metadata, sp_size=8).next_update()
            batches = collated_update(payloads, CPUContext)

            def loss_contexts(data, sp_mesh):
                assert sp_mesh is mesh
                assert len(data) == 16
                assert all(item["seq_ctx"].input_ids.shape == (1, 32) for item in data)
                return [SimpleNamespace(labels=item["shifted_labels"]) for item in data]

            runner = SimpleNamespace(
                sp_mesh=mesh, _engine=SimpleNamespace(model=SimpleNamespace(build_loss_ctx_batch=loss_contexts))
            )
            items = namespace["_prepare_model_input"](runner, batches)
            for item, payload in zip(items, payloads):
                assert item.seq_ctx.input_ids.shape == (1, 4)
                torch.testing.assert_close(item.seq_ctx.input_ids, payload["tokens"][:, rank * 4 : (rank + 1) * 4])
                torch.testing.assert_close(item.seq_ctx.raw_input_ids, payload["tokens"])
                torch.testing.assert_close(
                    item.loss_ctx.labels, payload["labels"].masked_fill(payload["loss_mask"] == 0, -100)
                )
            shards.append(items)
        for index, payload in enumerate(payloads):
            torch.testing.assert_close(
                torch.cat([items[index].seq_ctx.input_ids for items in shards], dim=1), payload["tokens"]
            )
            torch.testing.assert_close(
                torch.cat([items[index].seq_ctx.position_ids for items in shards], dim=1), payload["position_ids"]
            )

    def test_corruption_in_any_original_rank_rejected_before_update(self, tmp_path: Path) -> None:
        metadata = make_capture(tmp_path, length=32)
        path = batch_path(tmp_path, 1, 1, 7)
        payload = torch.load(path, weights_only=True)
        payload["tokens"][0, 0] += 1
        torch.save(payload, path)
        source = RecordedBatchSource(tmp_path, 0, metadata, sp_size=8)
        source.next_update()
        with pytest.raises(ValueError, match="checksum"):
            source.next_update()
        assert source.cursor == 1

    @pytest.mark.parametrize("style", ["wsd", "cosine"])
    def test_prepare_export_restores_original_rank_fingerprints(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        style: str,
    ) -> None:
        make_capture(tmp_path, length=32)
        config = prepare(tmp_path, monkeypatch, "--sp-size", "8", "--lr-decay-style", style)
        assert config["sp_size"] == 8 and config["tokenizer_path"] is None
        assert config["lr_decay_style"] == style and not config["profile_trace"]
        assert "lr" not in config["replayed_fields"]
        native_outputs(tmp_path, config)
        collect_replay(tmp_path, sp_size=8)
        rows = [json.loads(line) for line in (tmp_path / "xtuner_sp8.jsonl").read_text().splitlines()]
        for row in rows:
            assert row["accumulation_steps"] == 16
            assert row["valid_targets"] == 31 * 16  # Count global targets once, not once per SP peer.
            assert row["sp_size"] == 8 and row["dp_size"] == 1
            assert row["lr"] == 8e-4 * row["step"] / 2000
            assert row["tokens_per_second_per_gpu"] == 16 * 32 / 0.5 / 8
            for rank, fingerprints in enumerate(row["batch_sha256_by_rank"]):
                assert fingerprints == [
                    torch.load(batch_path(tmp_path, row["step"], micro, rank), weights_only=True)["sha256"]
                    for micro in range(2)
                ]
        done = json.loads((tmp_path / "xtuner_complete.json").read_text())
        assert done["training_entry"] == "Trainer.fit" and done["sp_size"] == 8

    @pytest.mark.parametrize("corruption", ["fingerprint", "targets", "missing_micro", "lr"])
    def test_disagreeing_peers_cannot_be_certified(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        corruption: str,
    ) -> None:
        make_capture(tmp_path, length=32)
        config = prepare(tmp_path, monkeypatch, "--sp-size", "8")
        native_outputs(tmp_path, config)
        output = tmp_path / "xtuner_sp8_native"
        path = output / "replay_inputs.rank03.jsonl"
        if corruption == "lr":
            path = output / "logs/exp_tracking/rank3/tracker.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if corruption == "fingerprint":
            rows[0]["batch_sha256"][0] = "different"
        elif corruption == "targets":
            rows[0]["valid_targets"] -= 1
        elif corruption == "missing_micro":
            rows[0]["batch_sha256"].pop()
        else:
            rows[0]["lr"] = 123.456
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(ValueError):
            collect_replay(tmp_path, sp_size=8)
        assert not (tmp_path / "xtuner_complete.json").exists()
        assert not (tmp_path / "xtuner_sp8.jsonl").exists()

    def test_native_loader_uses_dp1_and_audits_every_sp_peer(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        metadata = make_capture(tmp_path, length=32)
        namespace = adapter_namespace()
        namespace["build_trainer_config"] = lambda config, *, dataloader_cfg: SimpleNamespace(native_trainer=True, sp_size=8, dataloader_cfg=dataloader_cfg)
        namespace["flash_attn_varlen_func"] = SimpleNamespace(__name__="gpu_flash_varlen_attn_v3")
        config = namespace["build_replay_config"](
            {
                "capture_dir": str(tmp_path),
                "run_dir": str(tmp_path),
                "capture_metadata": metadata,
                "sp_size": 8,
                "sequence_length": 32,
            }
        )
        loader_config = config.dataloader_cfg
        assert config.native_trainer and config.sp_size == loader_config.sp_size == 8
        mesh = SimpleNamespace(size=lambda: 1, get_local_rank=lambda: 0)
        for rank in range(8):
            monkeypatch.setattr(namespace["dist"], "get_rank", lambda rank=rank: rank)
            loader = loader_config.build(None, mesh, 16, 16, seed=1234, total_step=3)
            assert len(next(iter(loader))) == 16
        audits = [json.loads((tmp_path / f"replay_inputs.rank{rank:02d}.jsonl").read_text()) for rank in range(8)]
        assert all(audit == audits[0] for audit in audits)
        assert audits[0]["valid_targets"] == 31 * 16
        for wrong_dp, wrong_batch in [(8, 16), (1, 2)]:
            with pytest.raises(ValueError, match="topology"):
                loader_config.build(
                    None, SimpleNamespace(size=lambda: wrong_dp), 16, wrong_batch, seed=1234, total_step=3
                )

    def test_incompatible_sequence_length_rejected_during_preparation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        make_capture(tmp_path)  # Length 16 cannot supply chunk-aligned shards for SP8.
        with pytest.raises(ValueError, match="divisible"):
            prepare(tmp_path, monkeypatch, "--sp-size", "8")
        assert not (tmp_path / "xtuner_sp8_native").exists()

    @pytest.mark.parametrize("arguments", [[], ["--sp-size", "8"], ["--sp-size=8"]])
    def test_launcher_routes_both_modes_to_native_cli(self, tmp_path: Path, arguments: list[str]) -> None:
        # Replace only the external interpreter process; no training or GPU commands run.
        recorder = tmp_path / "python-recorder"
        log = tmp_path / "calls.jsonl"
        recorder.write_text(
            f"#!{sys.executable}\nimport json, os, runpy, sys\n"
            "with open(os.environ['CALL_LOG'], 'a') as handle:\n"
            "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if sys.argv[1].endswith('/kernel_config.py'):\n"
            "    runpy.run_path(sys.argv[1], run_name='__main__')\n"
            "else:\n"
            "    assert os.environ['XTUNER_GDN_CUSTOM_OP'] == 'true'\n"
            "    assert os.environ['NCP_Q35V2_TRITON_DUAL_CROSS_MUDD'] == '0'\n"
            "    assert 'XTUNER_HF_IMPL' not in os.environ\n"
        )
        recorder.chmod(0o755)
        subprocess.run(
            ["bash", str(SCRIPTS / "run_xtuner.sh"), *arguments],
            check=True,
            env=dict(os.environ, XTUNER_PYTHON=str(recorder), CALL_LOG=str(log), ALIGNMENT_RUN_DIR=str(tmp_path),
                     XTUNER_HF_IMPL="true", NCP_Q35V2_TRITON_DUAL_CROSS_MUDD="1"),
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        sp_size = 8 if arguments else 1
        assert len(calls) == 5
        assert calls[0] == [str(SCRIPTS / "kernel_config.py")]
        assert calls[2][0] == str(SCRIPTS / "replay_xtuner.py")
        assert calls[3][:2] == ["-m", "torch.distributed.run"]
        assert "xtuner.v1.train.cli.sft" in calls[3]
        assert calls[3][-1] == str(tmp_path / f"xtuner_sp{sp_size}_native/trainer_config.py")
        assert calls[4][-3:] == ["--sp-size", str(sp_size), "--collect-metrics"]
