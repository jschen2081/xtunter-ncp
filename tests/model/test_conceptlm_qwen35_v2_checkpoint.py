"""Distributed checkpoint round-trip tests for NCP Qwen3.5 v2."""

import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import parametrize
import torch
import torch.distributed as dist
from safetensors import safe_open

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.dense.conceptlm_qwen35_v2 import ConceptLMQwen35V2Config


NCP_QWEN35_V2_PATH = os.environ.get("NCP_QWEN35_V2_PATH")
RUN_ROUND_TRIP = os.environ.get("NCP_QWEN35_V2_RUN_ROUND_TRIP") == "1"


@unittest.skipUnless(NCP_QWEN35_V2_PATH is not None, "set NCP_QWEN35_V2_PATH")
@unittest.skipUnless(RUN_ROUND_TRIP, "set NCP_QWEN35_V2_RUN_ROUND_TRIP=1 for the 9B save test")
class TestConceptLMQwen35V2Checkpoint(DeterministicDDPTestCase):
    @parametrize.parametrize("device", [("cuda",)])
    def test_save_hf_round_trip_is_byte_equal(self, device: str) -> None:
        self.create_pg(device)
        assert NCP_QWEN35_V2_PATH is not None
        origin = Path(NCP_QWEN35_V2_PATH)
        with torch.device("meta"):
            model = ConceptLMQwen35V2Config(compile_cfg=False).build()
        model.fully_shard(FSDPConfig(requires_grad=False, cpu_offload=False))

        with tempfile.TemporaryDirectory() as directory:
            synchronized = [directory]
            dist.broadcast_object_list(synchronized, src=0)
            saved = Path(synchronized[0])
            model.from_hf(origin, strict=True)
            model.save_hf(saved)

            if dist.get_rank() == 0:
                origin_index = json.loads((origin / "model.safetensors.index.json").read_text())["weight_map"]
                saved_index = json.loads((saved / "model.safetensors.index.json").read_text())["weight_map"]
                self.assertEqual(set(saved_index), set(origin_index))
                handles = {}
                with contextlib.ExitStack() as stack:
                    for key, origin_shard in origin_index.items():
                        saved_shard = saved_index[key]
                        origin_path = origin / origin_shard
                        saved_path = saved / saved_shard
                        if origin_path not in handles:
                            handles[origin_path] = stack.enter_context(safe_open(origin_path, framework="pt"))
                        if saved_path not in handles:
                            handles[saved_path] = stack.enter_context(safe_open(saved_path, framework="pt"))
                        self.assertTrue(
                            torch.equal(handles[origin_path].get_tensor(key), handles[saved_path].get_tensor(key)),
                            f"Tensor mismatch for key: {key}",
                        )
            dist.barrier()

