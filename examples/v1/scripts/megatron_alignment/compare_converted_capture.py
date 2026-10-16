"""Compare a converted capture using its real historical global loss records."""
import argparse
import json
import math
import os
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir
    assert int((root / "exit_code").read_text()) == 0
    contract = json.loads((root / "contract.json").read_text())
    conversion = json.loads((root / "conversion_manifest.json").read_text())
    assert conversion["initial_weights_verified"] == 960 and conversion["batch_files"] == 800
    actual = [json.loads(line) for line in (root / "xtuner_sp1.jsonl").read_text().splitlines()]
    expected = [json.loads(line) for line in (root / "megatron_reference_global_losses.jsonl").read_text().splitlines()]
    done = json.loads((root / "xtuner_complete.json").read_text())
    assert [row["step"] for row in actual] == [row["step"] for row in expected] == done["steps"] == list(range(100))
    updates = {row["step"]: row for row in (json.loads(line) for line in (root / "megatron_reference_steps.jsonl").read_text().splitlines())}
    fingerprints = {}
    for row in conversion["batches"]:
        step = fingerprints.setdefault(row["step"], [[] for _ in range(8)])
        assert row["micro"] == len(step[row["rank"]])
        step[row["rank"]].append(row["converted_sha256"])
    report = []
    for a, e in zip(actual, expected):
        step = a["step"]
        assert a["batch_sha256_by_rank"] == fingerprints[step]
        assert a["lr"] == updates[step]["lr"] and a["valid_targets"] == e["valid_targets"]
        assert all(math.isfinite(a[key]) for key in ["total_loss", "lm_loss", "grad_norm"])
        gap = abs(a["total_loss"] - e["total_loss"])
        targets = e["valid_targets"]
        ref_lm = e["lm_numerator"] / targets
        ref_vq = e["chunk_size"] * e["vq_weight"] * e["vq_numerator"] / targets
        ref_hlm = e["chunk_size"] * e["hlm_weight"] * e["hlm_numerator"] / targets
        grad = updates[step]["grad_norm"]
        report.append({"step": step, "megatron_total_loss": e["total_loss"], "xtuner_total_loss": a["total_loss"], "loss_abs_gap": gap, "loss_relative_gap": gap / abs(e["total_loss"]), "megatron_lm_loss": ref_lm, "xtuner_lm_loss": a["lm_loss"], "megatron_vq_token_loss": ref_vq, "xtuner_vq_token_loss": a["vq_token_loss"], "megatron_hlm_token_loss": ref_hlm, "xtuner_hlm_token_loss": a["hlm_token_loss"], "megatron_grad_norm": grad, "xtuner_grad_norm": a["grad_norm"], "grad_relative_gap": abs(a["grad_norm"] - grad) / max(abs(grad), 1e-30), "valid_targets": targets, "lr": a["lr"], "seconds": a["seconds"]})
    errors = [r["loss_abs_gap"] for r in report]
    summary = {"status": "COMPLETED_NATIVE_ENTRY_REPLAY", "steps": 100, "replayed_batch_files_verified": 800, "initial_weights_verified": 960, "world_size": 8, "sp": 1, "micro_batch_size": 1, "global_batch_size": 8, "source_capture": contract["source_capture"], "run_dir": str(root), "first_step": report[0], "last_step": report[-1], "loss_mae": statistics.mean(errors), "loss_rmse": math.sqrt(statistics.mean(e*e for e in errors)), "mean_loss_relative_error_percent": 100 * statistics.mean(r["loss_relative_gap"] for r in report), "max_loss_relative_error_percent": 100 * max(r["loss_relative_gap"] for r in report), "max_grad_relative_error_percent": 100 * max(r["grad_relative_gap"] for r in report), "steady_mean_seconds": statistics.mean(r["seconds"] for r in report[10:]), "reference_loss_source": "original global loss records; not reconstructed per-rank numerators", "historical_replays": {}}
    reference = Path(contract["source_capture"])
    for name in ["xtuner_replay_20261015_102925", "xtuner_replay_20261016_110718"]:
        p = reference / name / "xtuner_sp1.jsonl"
        if p.exists():
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            assert [r["step"] for r in rows] == list(range(100))
            err = [abs(r["total_loss"] - expected[r["step"]]["total_loss"]) for r in rows]
            summary["historical_replays"][name] = {"loss_mae": statistics.mean(err), "loss_rmse": math.sqrt(statistics.mean(e*e for e in err)), "final_total_loss": rows[-1]["total_loss"]}
    (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    os.environ["MPLCONFIGDIR"] = str(root / "matplotlib-cache")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plotting unavailable; JSON summary saved")
    else:
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        x = [r["step"] + 1 for r in report]
        axes[0].plot(x, [r["megatron_total_loss"] for r in report], label="Megatron")
        axes[0].plot(x, [r["xtuner_total_loss"] for r in report], label="XTuner new entry")
        axes[0].set_ylabel("Total objective"); axes[0].legend(); axes[0].grid(alpha=.2)
        axes[1].plot(x, [100*r["loss_relative_gap"] for r in report], color="#C65A24")
        axes[1].set_ylabel("Loss difference (%)"); axes[1].set_xlabel("Training step"); axes[1].grid(alpha=.2)
        fig.tight_layout(); fig.savefig(root / "loss_comparison.png", dpi=160); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
