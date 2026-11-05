import runpy
from pathlib import Path

import pytest


RECIPE = (
    Path(__file__).resolve().parents[2] / "examples/v1/scripts/qwen3p5_40b_a3b_speedtest/speedtest_qwen35_35b_a3b.py"
)


def test_speedtest_uses_unpacked_indexed_tokens_and_scalar_moe_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data_args_path.txt"))
    monkeypatch.setenv("TOKENIZER_PATH", str(tmp_path / "tokenizer"))
    monkeypatch.setenv("SEQ_LEN", "8192")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("GLOBAL_BATCH_SIZE", "8")
    monkeypatch.setenv("TOTAL_STEP", "3")
    monkeypatch.setenv("LR_WARMUP_STEPS", "2")
    monkeypatch.setenv("EXPERT_BACKEND", "grouped_gemm")

    trainer = runpy.run_path(str(RECIPE))["trainer"]
    dataloader = trainer.dataloader_cfg
    dataset = dataloader.dataset_config_list[0]["dataset"]

    assert dataset.class_name == "MegatronIndexedDataset"
    assert dataset.sequence_length == 8193
    assert dataloader.pack_level == "none"
    assert dataloader.pack_max_length == 8193
    assert dataloader.pack_to_max_length
    assert not dataloader.group_by_length
    assert dataloader.collator == "single_sequence_sft_llm_collator"
    assert trainer.model_cfg.ep_size == 1
    assert trainer.model_cfg.dispatcher is None
    assert trainer.model_cfg.expert_backend == "grouped_gemm"
    assert trainer.fsdp_cfg.tp_size == 1
    assert trainer.fsdp_cfg.ep_size == 1
    assert trainer.sp_size == 1
    assert trainer.intra_layer_micro_batch == 1
    assert not trainer.model_cfg.compile_cfg
    assert trainer.fsdp_cfg.recompute_ratio == 0.0
    assert trainer.fsdp_cfg.reshard_after_forward
    assert not trainer.fsdp_cfg.gradient_accumulation_no_sync
    assert trainer.loss_cfg.chunk_size == 128

    source_tokens = list(range(8193))
    batch = dataloader.build_collator()(
        [[{"input_ids": source_tokens, "labels": source_tokens, "num_tokens": len(source_tokens)}]],
        pack_max_length=dataloader.pack_max_length,
        padding_token_idx=0,
        pack_to_max_length=dataloader.pack_to_max_length,
    )

    seq_ctx = batch[0]["seq_ctx"]
    assert seq_ctx.input_ids is not None
    assert seq_ctx.input_ids.shape == (1, 8192)
    assert batch[0]["shifted_labels"].shape == (1, 8192)
    assert seq_ctx.input_ids.tolist() == [source_tokens[:-1]]
    assert batch[0]["shifted_labels"].tolist() == [source_tokens[1:]]
    assert seq_ctx.cu_seq_lens_q.tolist() == [0, 8192]
    assert seq_ctx.num_padding == 0


def test_speedtest_microbatch_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data_args_path.txt"))
    monkeypatch.setenv("TOKENIZER_PATH", str(tmp_path / "tokenizer"))
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("GLOBAL_BATCH_SIZE", "16")
    monkeypatch.setenv("TOTAL_STEP", "3")
    monkeypatch.setenv("LR_WARMUP_STEPS", "2")
    monkeypatch.setenv("INTRA_LAYER_MICRO_BATCH", "2")

    trainer = runpy.run_path(str(RECIPE))["trainer"]

    assert trainer.intra_layer_micro_batch == 2

    monkeypatch.setenv("GLOBAL_BATCH_SIZE", "24")
    with pytest.raises(ValueError, match="divisible by INTRA_LAYER_MICRO_BATCH"):
        runpy.run_path(str(RECIPE))
