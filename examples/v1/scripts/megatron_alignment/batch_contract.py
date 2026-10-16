"""Shared, CPU-testable contract for captured Megatron indexed batches."""
import hashlib
import json


TENSOR_FIELDS = ("tokens", "position_ids", "labels", "loss_mask", "cu_seqlens")
META_FIELDS = ("step", "micro", "rank", "world_size", "global_batch_size", "lr", "packed")


def batch_digest(payload):
    digest = hashlib.sha256()
    digest.update(json.dumps({key: payload[key] for key in META_FIELDS}, sort_keys=True).encode())
    for key in TENSOR_FIELDS:
        value = payload[key].detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_batch(payload, rank, world):
    import torch

    if payload["rank"] != rank or payload["world_size"] != world:
        raise ValueError("Captured rank/world size differs")
    if payload["sha256"] != batch_digest(payload):
        raise ValueError("Captured batch checksum differs")
    tokens = payload["tokens"]
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] % 4:
        raise ValueError("Replay requires MBS=1 and chunk-aligned sequences")
    for key in ("position_ids", "labels", "loss_mask"):
        if payload[key].shape != tokens.shape:
            raise ValueError(f"Invalid {key} shape")
    mask = payload["loss_mask"]
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Fractional loss masks require weighted CE; refusing to discard weights")
    if torch.any((payload["labels"] < 0) & (mask != 0)):
        raise ValueError("Active production labels must be nonnegative")
    bounds = payload["cu_seqlens"].tolist()
    if bounds[0] != 0 or bounds[-1] != tokens.shape[1]:
        raise ValueError("Boundaries do not cover tokens")
    if any(b <= a or (b - a) % 4 for a, b in zip(bounds, bounds[1:])):
        raise ValueError("Each sequence must align to concept chunks")
    expected_positions = torch.cat([torch.arange(b - a) for a, b in zip(bounds, bounds[1:])])[None]
    if not torch.equal(payload["position_ids"], expected_positions):
        raise ValueError("Unsupported position resets inside a captured attention sequence")
    if payload["global_batch_size"] % world:
        raise ValueError("GBS must be divisible by eight")
