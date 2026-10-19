"""Use xTuner's text tokenization, hard packing, sampler and DataLoader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from qwen35_wsd_lr import sample_count_wsd_lr


class XTunerTextSource:
    """Build the native text data pipeline for independent training.

    Args:
        config (dict[str, Any]): Text manifest, tokenizer and training settings.
        rank (int): Data parallel rank.
        world (int): Data parallel world size.
        output (Path): Training output directory.
        dp_mesh: Optional native data-parallel mesh. SP peers use identical
            DP-local ranks and therefore consume identical full samples.
    """

    def __init__(self, config: dict[str, Any], rank: int, world: int, output: Path, *, dp_mesh=None) -> None:
        from torch.distributed.device_mesh import init_device_mesh
        from transformers import AutoTokenizer
        from xtuner.v1.datasets import PretrainTokenizeFunctionConfig
        from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig

        self.config = config
        self.rank = rank
        self.world = world
        self.epoch = 0
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], local_files_only=True)
        indexed = config.get("data_format", "indexed") == "indexed"
        if indexed:
            datasets = [dict(dataset=DatasetConfig(
                anno_path=config["data_path"], class_name="MegatronIndexedDataset", name="tokenized",
                sequence_length=config["sequence_length"] + 1, enable_sequential_sampler=False,
            ), tokenize_fn=PretrainTokenizeFunctionConfig(add_eos_token=False, add_bos_token=False))]
        else:
            files = json.loads(Path(config["text_manifest"]).read_text())
            datasets = [dict(dataset=DatasetConfig(
                anno_path=item["normalized"], class_name="JsonlDataset", name=f"text_{index}",
                cache_dir=config["data_cache"], enable_sequential_sampler=False,
            ), tokenize_fn=PretrainTokenizeFunctionConfig(add_eos_token=True, add_bos_token=False))
                        for index, item in enumerate(files)]
        self.loader_config = DataloaderConfig(
            dataset_config_list=datasets, pack_level="none" if indexed else "hard", global_pack=True,
            pack_max_length=config["sequence_length"] + 1, pack_to_max_length=True,
            collator="single_sequence_sft_llm_collator", shuffle=True, group_by_length=False,
            num_workers=config["num_workers"], pack_workers=config["pack_workers"],
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mesh = dp_mesh if dp_mesh is not None else init_device_mesh(device, (world,), mesh_dim_names=("text_dp",))
        if mesh.size() != world or mesh.get_local_rank() != rank:
            raise ValueError("Data source rank/size must match the DP mesh")
        self.loader = self.loader_config.build(
            tokenizer=tokenizer, dp_mesh=mesh, global_batch_size=config["global_batch_size"],
            micro_batch_size=1, seed=config["seed"], shuffle=True, total_step=config["steps"],
        )
        if len(self.loader) == 0:
            raise ValueError("The supplied text contains no complete hard packs")
        self.iterator = iter(self.loader)
        self.metadata = dict(dataset_class="MegatronIndexedDataset" if indexed else "JsonlDataset",
                             dp_size=world, sp_size=config.get("sp_size", 1),
                             packing_class="fixed_token_stream_blocks" if indexed else "HardPackDataset",
                             sampler_class=type(self.loader.sampler).__name__, shuffle=True,
                             global_pack=not indexed, eos_added=not indexed, model_sequence_length=config["sequence_length"],
                             runtime_megatron_dependency=False, input_format="original_tokenized" if indexed else "original_text")

    def next_step(self, step: int) -> list[dict[str, Any]]:
        """Fetch native sampled microbatches for one optimizer update.

        Args:
            step (int): Number of completed optimizer updates.

        Returns:
            list[dict[str, Any]]: CPU microbatch tensors for the engine loop.
        """
        from batch_contract import batch_digest

        payloads = []
        for micro in range(self.config["global_batch_size"] // self.world):
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.epoch += 1
                self.loader.set_epoch(self.epoch)
                self.iterator = iter(self.loader)
                batch = next(self.iterator)
            if len(batch) != 1:
                raise ValueError("Expected one native collated sample per microbatch")
            context = batch[0]["seq_ctx"]
            labels = batch[0]["shifted_labels"]
            payload = dict(tokens=context.input_ids, position_ids=context.position_ids,
                           labels=labels, loss_mask=labels.ne(-100).float(), cu_seqlens=context.cu_seq_lens_q,
                           step=step, micro=micro, rank=self.rank, world_size=self.world,
                           global_batch_size=self.config["global_batch_size"], packed=False,
                           lr=sample_count_wsd_lr(step, self.config["global_batch_size"], self.config))
            payload["sha256"] = batch_digest(payload)
            payloads.append(payload)
        return payloads

    def close(self) -> None:
        """Release the loader iterator and its workers."""
        self.iterator = None
