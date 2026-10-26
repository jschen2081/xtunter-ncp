"""Validate complete batch identity and compare full training objectives on CPU."""
import argparse
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compare(root, sp_size=1):
    done = json.loads((root / "xtuner_complete.json").read_text())
    production = {}
    rank_counts = []
    fingerprints = {}
    for rank in range(8):
        completed = json.loads((root / f"megatron_complete.rank{rank:02d}.json").read_text())
        counts = {}
        for row in rows(root / f"megatron_losses.rank{rank:02d}.jsonl"):
            identities = fingerprints.setdefault(row["step"], [[] for _ in range(8)])[rank]
            if row["micro"] != len(identities):
                raise ValueError("Megatron microbatch loss order differs")
            identities.append(row["batch_sha256"])
            counts[str(row["step"])] = counts.get(str(row["step"]), 0) + 1
            step = production.setdefault(row["step"], dict(lm=0.0, vq=0.0, hlm=0.0, targets=0.0))
            step["lm"] += row["lm_numerator"]
            step["vq"] += row["chunk_size"] * row["vq_weight"] * row["vq_numerator"]
            step["hlm"] += row["chunk_size"] * row["hlm_weight"] * row["hlm_numerator"]
            step["targets"] += row["valid_targets"]
        if counts != completed:
            raise ValueError(f"Megatron loss records are incomplete for rank {rank}")
        rank_counts.append(counts)
    if any(count != rank_counts[0] for count in rank_counts):
        raise ValueError("Megatron rank step counts differ")
    optimizer_rows = rows(root / "megatron_steps.jsonl")
    optimizer = {row["step"]: row for row in optimizer_rows}
    actual = rows(root / f"xtuner_sp{sp_size}.jsonl")
    steps = [row["step"] for row in actual]
    if (steps != sorted(production) or steps != done["steps"] or
            steps != sorted(optimizer) or len(optimizer_rows) != len(steps)):
        raise ValueError("Frameworks have missing/duplicate optimizer steps")
    report = []
    for row in actual:
        step = row["step"]
        reference = production[step]
        if row["valid_targets"] != reference["targets"] or reference["targets"] <= 0:
            raise ValueError(f"Effective target counts differ at step {step}")
        expected_micro = rank_counts[0][str(step)]
        if len(row["batch_sha256_by_rank"]) != 8 or any(
                len(values) != expected_micro for values in row["batch_sha256_by_rank"]):
            raise ValueError("Incomplete xTuner batch fingerprints")
        if row["batch_sha256_by_rank"] != fingerprints[step]:
            raise ValueError(f"Frameworks consumed different batches at step {step}")
        if row["lr"] != optimizer[step]["lr"]:
            raise ValueError(f"Optimizer learning rates differ at step {step}")
        loss = (reference["lm"] + reference["vq"] + reference["hlm"]) / reference["targets"]
        gap = abs(row["total_loss"] - loss)
        report.append(dict(step=step, megatron_total_loss=loss, xtuner_total_loss=row["total_loss"],
                           loss_abs_gap=gap, loss_relative_gap=gap / max(abs(loss), 1e-30),
                           megatron_lm_loss=reference["lm"] / reference["targets"],
                           xtuner_lm_loss=row["lm_loss"],
                           megatron_vq_token_loss=reference["vq"] / reference["targets"],
                           xtuner_vq_token_loss=row["vq_token_loss"],
                           megatron_hlm_token_loss=reference["hlm"] / reference["targets"],
                           xtuner_hlm_token_loss=row["hlm_token_loss"],
                           megatron_grad_norm=optimizer[step]["grad_norm"],
                           xtuner_grad_norm=row["grad_norm"],
                           grad_norm_abs_gap=abs(optimizer[step]["grad_norm"] - row["grad_norm"]),
                           valid_targets=row["valid_targets"], lr=row["lr"],
                           batch_sha256_by_rank=row["batch_sha256_by_rank"]))
    return dict(status="GPU_RESULTS_COMPARED_NO_AUTOMATIC_PARITY_CLAIM", steps=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--sp-size", type=int, choices=(1, 8), default=1)
    args = parser.parse_args()
    report = compare(args.run_dir, args.sp_size)
    (args.run_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
