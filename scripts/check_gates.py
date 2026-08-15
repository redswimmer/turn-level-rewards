"""Report the Phase 7c §6 gates for a PPO train_log.jsonl.

Kept deliberately small and dependency-free: it reads the log as plain data
(one JSON object per optimizer step) and prints only the four quantities the
phase doc says to stop on. `analyze_train_log.py` remains the tool for
episode-level forensics; this is the at-a-glance gate check.

Usage: python scripts/check_gates.py outputs/<arm>/train_log.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_steps(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarize(path: Path) -> int:
    steps = load_steps(path)
    if not steps:
        print(f"{path}: no steps logged yet")
        return 1

    n = len(steps)
    half = n // 2 or 1

    def mean(key: str, rows: list[dict]) -> float:
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / len(vals) if vals else float("nan")

    skips = sum(r.get("skipped", 0.0) for r in steps)
    ended_on_tool = mean("ended_on_tool_rate", steps)
    peak_gpu = max(r.get("gpu_max_allocated_gb", 0.0) for r in steps)
    distinct_reward = len({round(r["reward"], 6) for r in steps if "reward" in r})
    elapsed_h = steps[-1].get("total_elapsed_seconds", 0.0) / 3600

    print(f"=== {path} ===")
    print(f"steps logged        : {n}")
    print(f"[gate 1] skip rate  : {skips:.0f}/{n} = {skips / n:.1%}   (stop above ~5%)")
    print(f"         peak GPU   : {peak_gpu:.2f} GB  (ceiling ~23.5)")
    print(f"[gate 2] ended_on_tool: {ended_on_tool:.4f}   (must be 0)")
    print(
        f"[gate 3] format compliance: {mean('format_compliance', steps[:half]):.3f}"
        f" -> {mean('format_compliance', steps[half:]):.3f}"
        f"   (stop if still ~0.5 at step 150)"
    )
    print(
        f"[gate 4] reward: {mean('reward', steps[:half]):.3f}"
        f" -> {mean('reward', steps[half:]):.3f}, {distinct_reward} distinct value(s)"
    )
    # PPO-OR's reward is binary, so an all-zero opening stretch is normal rather
    # than dead. Only call it a dead run once there are enough steps for a
    # genuinely learning arm to have scored at least one correct answer.
    dead = distinct_reward <= 1 and n >= 20
    if dead:
        print("         *** DEAD RUN: reward has no variance -- STOP")
    print(
        f"retrieval_fraction  : {mean('retrieval_fraction', steps[:half]):.3f}"
        f" -> {mean('retrieval_fraction', steps[half:]):.3f}"
    )
    print(f"mean tool turns/ep  : {mean('mean_tool_turns', steps):.2f}")
    print(f"elapsed             : {elapsed_h:.2f} h")

    tripped = skips / n > 0.05 or ended_on_tool > 0 or dead
    print(f"GATES: {'*** TRIPPED ***' if tripped else 'ok'}")
    return 2 if tripped else 0


if __name__ == "__main__":
    sys.exit(max(summarize(Path(p)) for p in sys.argv[1:]))
