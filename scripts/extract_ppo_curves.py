"""Lift the PPO arms' per-step training curves out of outputs/ into results/.

outputs/ is gitignored and multi-GB; results/phase7c-curves.json is committed and tiny. Running
this once after training is what lets plot_phase7c.py stay reproducible from a fresh clone.

Merges into the existing file rather than overwriting it: the GRPO arms' curves came from a
trackio backend that is not reproducible from the repo, so those entries must survive.

Usage: python scripts/extract_ppo_curves.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUTS = Path("outputs")
CURVES = Path("results/phase7c-curves.json")

# Held-out metric names are shared with the GRPO entries already in the file, so a plot can pull
# the same key across algorithms without knowing which trainer produced it.
FIELDS = ("reward", "format_compliance", "mean_tool_turns", "retrieval_fraction")

# ppo's own directory holds the post-collapse continuation; _precollapse is the checkpoint the
# README scores it at, and is the run whose curve is worth plotting.
ARMS = {"ppo": "ppo", "ppo_mr_paper": "ppo_mr_paper", "ppo_mr": "ppo_mr", "mt_ppo": "mt_ppo"}


def main() -> None:
    curves = json.loads(CURVES.read_text()) if CURVES.exists() else {}

    for arm, directory in ARMS.items():
        log = OUTPUTS / directory / "train_log.jsonl"
        if not log.exists():
            print(f"skip {arm}: no {log}")
            continue
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        # Steps skipped for OOM log no metrics, so filter once and index everything off the same
        # rows -- taking step from all rows and metrics from a subset silently misaligns the x-axis.
        rows = [r for r in rows if all(f in r for f in FIELDS)]
        curves[arm] = {f: [r[f] for r in rows] for f in FIELDS}
        curves[arm]["step"] = [r["step"] for r in rows]
        print(f"{arm}: {len(rows)} steps with complete metrics")

    CURVES.write_text(json.dumps(curves, separators=(",", ":")))
    print(f"wrote {CURVES} ({CURVES.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
