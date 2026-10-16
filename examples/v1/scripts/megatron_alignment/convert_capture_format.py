"""Convert verified historical captures to the current input envelope, on CPU only."""
import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from batch_contract import batch_digest, validate_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    assert source != output and not output.exists(), "Use a new output directory"
    alignment = json.loads((source / "alignment_contract.json").read_text())
    losses = [json.loads(line) for line in (source / "megatron_losses.jsonl").read_text().splitlines()]
    assert [row["step"] for row in losses] == list(range(100))
    log = (source / "console.log").read_text()
    assert "successfully saved checkpoint from iteration     100" in log
    steps = {}
    for line in log.splitlines():
        match = re.search(r"iteration\s+(\d+)/\s+100\s*\|", line)
        if match:
            step = int(match.group(1)) - 1
            assert step not in steps
            assert int(re.search(r"number of skipped iterations:\s+(\d+)", line).group(1)) == 0
            assert int(re.search(r"number of nan iterations:\s+(\d+)", line).group(1)) == 0
            steps[step] = {"step": step, "grad_norm": float(re.search(r"grad norm: ([\d.]+)", line).group(1))}
    assert set(steps) == set(range(100))
    hf = Path(alignment["hf_conversion"])
    conversion = json.loads((hf / "conversion_manifest.json").read_text())
    index = json.loads((hf / "model.safetensors.index.json").read_text())["weight_map"]
    hashes = json.loads((source / "initial_parameter_hashes.json").read_text())
    by_shard = defaultdict(list)
    for name, expected in hashes.items():
        name = re.sub(r"^0\.(?:module\.)+", "", name)
        hf_key = conversion["mapping"][name]
        by_shard[index[hf_key]].append((name, hf_key, expected))
    verified = 0
    for shard, items in by_shard.items():
        with safe_open(hf / shard, framework="pt", device="cpu") as handle:
            for name, key, expected in items:
                dtype = torch.bfloat16 if expected["dtype"] == "torch.bfloat16" else torch.float32
                value = handle.get_tensor(key).to(dtype).reshape(expected["shape"]).contiguous()
                digest = hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                assert digest == expected["sha256"], f"Initial weight mismatch: {name}"
                verified += 1
        print(f"Verified initial weights: {verified}/{len(hashes)}", flush=True)
    assert verified == len(hashes) == 960
    output.mkdir()
    (output / "batches").mkdir()
    manifests = []
    counts = [defaultdict(int) for _ in range(8)]
    targets = defaultdict(float)
    paths = sorted((source / "batches").glob("*.pt"))
    assert len(paths) == 800
    for path in paths:
        old = torch.load(path, map_location="cpu", weights_only=True)
        digest = hashlib.sha256()
        for key in ("tokens", "position_ids", "attention_mask", "labels", "loss_mask", "cu_seqlens"):
            digest.update(key.encode())
            value = old[key]
            digest.update(b"None" if value is None else value.numpy().tobytes())
        assert digest.hexdigest() == old["sha256"], path
        assert old["attention_mask"] is None
        assert old["world_size"] == 8 and old["global_batch_size"] == 8
        new = dict(old)
        if old["cu_seqlens"] is None:
            assert old["packed"] is False
            new["cu_seqlens"] = torch.tensor([0, old["tokens"].shape[1]], dtype=torch.int32)
        else:
            new["cu_seqlens"] = old["cu_seqlens"].reshape(-1).contiguous()
        new["sha256"] = batch_digest(new)
        validate_batch(new, old["rank"], 8)
        target = output / "batches" / path.name
        torch.save(new, target)
        check = torch.load(target, map_location="cpu", weights_only=True)
        validate_batch(check, old["rank"], 8)
        for key in ("tokens", "position_ids", "labels", "loss_mask"):
            assert check[key].dtype == old[key].dtype and torch.equal(check[key], old[key]), (path, key)
        for key in ("step", "micro", "rank", "world_size", "global_batch_size", "lr", "packed"):
            assert check[key] == old[key], (path, key)
        step, rank = old["step"], old["rank"]
        assert old["micro"] == counts[rank][str(step)]
        counts[rank][str(step)] += 1
        targets[step] += old["loss_mask"].sum().item()
        if "lr" in steps[step]:
            assert steps[step]["lr"] == old["lr"]
        steps[step]["lr"] = old["lr"]
        manifests.append({"file": path.name, "step": step, "rank": rank, "micro": old["micro"], "source_sha256": old["sha256"], "converted_sha256": new["sha256"]})
    expected_counts = {str(step): 1 for step in range(100)}
    for rank, count in enumerate(counts):
        assert dict(count) == expected_counts
        (output / f"megatron_complete.rank{rank:02d}.json").write_text(json.dumps(dict(count), sort_keys=True) + "\n")
    for row in losses:
        assert targets[row["step"]] == row["valid_targets"]
    for name in ("alignment_contract.json", "effective_training_config.json", "initial_parameter_hashes.json"):
        shutil.copyfile(source / name, output / name)
    shutil.copyfile(source / "megatron_losses.jsonl", output / "megatron_reference_global_losses.jsonl")
    (output / "megatron_reference_steps.jsonl").write_text("".join(json.dumps(steps[i]) + "\n" for i in range(100)))
    contract = {"commit": alignment["updated_training_commit"], "hf_path": str(hf), "source_capture": str(source), "data_backend": "megatron_indexed", "steps": 100, "world_size": 8, "global_batch_size": 8, "micro_batch_size": 1, "sequence_length": 8192, "original_capture_preserved": True, "initial_weight_tensors_verified": verified, "completion_evidence": "800 validated batches, 100 loss/update log records, zero skipped/nan iterations, final checkpoint save success", "reference_loss_scope": "global; no synthetic per-rank loss records"}
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (output / "conversion_manifest.json").write_text(json.dumps({"source": str(source), "initial_weights_verified": verified, "batch_files": len(manifests), "fields_unchanged": ["tokens", "position_ids", "labels", "loss_mask", "step", "micro", "rank", "world_size", "global_batch_size", "lr", "packed"], "envelope_changes": ["None boundary -> int32 [0,8192]", "new metadata/dtype/shape checksum", "completion markers reconstructed from verified historical records"], "batches": manifests}, indent=2) + "\n")
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
