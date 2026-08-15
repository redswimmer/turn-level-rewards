"""analyze_train_log.py: flag anomalies in a train_log.jsonl without manual grepping.

Built after a real incident (Phase 7b's first full ppo run): a single 2302-action-token episode
(vs. a typical 50-900) left the CUDA allocator fragmented enough to cause a deterministic OOM
wall on every --resume-from-checkpoint replay. Finding that episode required manually grepping
and eyeballing num_action_tokens across the log -- this script makes that a one-command report
instead, both for post-hoc incident analysis and for a quick health check on a run in progress
(safe to run against a train_log.jsonl that's still being appended to).
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_records(path: str) -> list[dict[str, Any]]:
    """Parse a train_log.jsonl file. Tolerates a partially-written last line (a run in progress
    may be mid-write when this is read) by skipping any line that fails to parse.
    """
    records = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def episode_token_stats(records: list[dict[str, Any]]) -> dict[str, float]:
    """Distribution of num_action_tokens across every episode in every non-skipped step.

    Returns an empty dict if no real (non-skipped) episodes exist yet.
    """
    tokens = [
        episode["num_action_tokens"]
        for record in records
        if not record.get("skipped")
        for episode in record.get("episodes", [])
    ]
    if not tokens:
        return {}
    tokens_sorted = sorted(tokens)
    return {
        "count": len(tokens),
        "min": tokens_sorted[0],
        "median": statistics.median(tokens_sorted),
        "p90": tokens_sorted[int(len(tokens_sorted) * 0.9)],
        "p99": tokens_sorted[int(len(tokens_sorted) * 0.99)],
        "max": tokens_sorted[-1],
    }


def find_outlier_episodes(
    records: list[dict[str, Any]], threshold_multiplier: float = 3.0
) -> list[dict[str, Any]]:
    """Episodes whose num_action_tokens exceeds threshold_multiplier times the run's own median --
    the exact signal that would have flagged the 2302-token episode (median ~150-200 in that run,
    so a 3x threshold catches anything past ~450-600) without needing to already know what
    "normal" looks like for this specific run.
    """
    stats = episode_token_stats(records)
    if not stats:
        return []
    threshold = stats["median"] * threshold_multiplier
    outliers = []
    for record in records:
        if record.get("skipped"):
            continue
        for episode in record.get("episodes", []):
            if episode["num_action_tokens"] > threshold:
                outliers.append(
                    {
                        "step": record["step"],
                        "num_action_tokens": episode["num_action_tokens"],
                        "question": episode["question"],
                    }
                )
    return outliers


def find_skips(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All OOM-skipped steps, in order -- the questions_attempted field is exactly what a
    from-scratch incident investigation would otherwise have to reconstruct from raw stdout logs.
    """
    return [record for record in records if record.get("skipped")]


def skip_rate_by_window(records: list[dict[str, Any]], window: int = 20) -> list[tuple[int, float]]:
    """Skip rate (fraction of steps skipped) in consecutive windows of `window` steps -- lets a
    reader see whether skips are spread evenly or clustered in one rough patch of the data,
    without eyeballing the raw sequence.
    """
    windows = []
    for start in range(0, len(records), window):
        chunk = records[start : start + window]
        skipped = sum(1 for r in chunk if r.get("skipped"))
        windows.append((start, skipped / len(chunk)))
    return windows


def reward_variance_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """How much the per-step mean reward actually MOVES -- the check this script was missing.

    Added after the 2026-07-24 incident, in which a broken rollout loop scored every one of 332
    episodes at exactly -0.1. Nothing crashed, no alert fired, and the loss curve looked textbook
    (value_loss 20.5 -> 0.046, which reads as excellent critic convergence and was really the
    critic learning a constant). The run burned 177 steps learning nothing.

    A constant reward is a dead run regardless of its magnitude: PPO's advantage is
    (return - baseline), and a critic fits a constant almost immediately, so advantages decay to
    ~0. Reporting distinct-value count alongside the min/max makes that visible in one line --
    every other statistic in this script would look perfectly healthy while it was happening.
    """
    rewards = [r["reward"] for r in records if not r.get("skipped") and "reward" in r]
    if not rewards:
        return {"steps": 0}
    distinct = sorted(set(rewards))
    return {
        "steps": len(rewards),
        "distinct_values": len(distinct),
        "min": min(rewards),
        "max": max(rewards),
        "mean": sum(rewards) / len(rewards),
        # The headline: one distinct value across many steps is the failure signature itself.
        "is_constant": len(distinct) == 1,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flag anomalies (outlier-length episodes, OOM skips) in a train_log.jsonl."
    )
    parser.add_argument(
        "log_path", help="Path to a train_log.jsonl (e.g. outputs/ppo/train_log.jsonl)"
    )
    parser.add_argument(
        "--outlier-multiplier",
        type=float,
        default=3.0,
        help="Flag episodes whose num_action_tokens exceeds this multiple of the run's median (default: 3.0)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Window size for the skip-rate-by-window report (default: 20)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    records = load_records(args.log_path)
    if not records:
        print(f"No records found in {args.log_path}")
        return

    print(f"{len(records)} total records ({args.log_path})")

    # Printed first, and loudly: a constant reward invalidates every other number below it, so a
    # reader must not have to scroll past healthy-looking token and skip statistics to find it.
    reward = reward_variance_report(records)
    if reward["steps"]:
        print(
            f"\nreward across {reward['steps']} real steps: "
            f"{reward['distinct_values']} distinct value(s), "
            f"min={reward['min']:.3f} mean={reward['mean']:.3f} max={reward['max']:.3f}"
        )
        if reward["is_constant"]:
            print(
                f"  *** DEAD RUN: reward is exactly {reward['min']:.3f} on every step. Zero "
                "variance means near-zero PPO advantage -- nothing is being learned, whatever "
                "the loss curve looks like. Suspect the rollout/parsing path first."
            )

    stats = episode_token_stats(records)
    if stats:
        print(
            f"\nnum_action_tokens distribution ({stats['count']} episodes): "
            f"min={stats['min']} median={stats['median']:.0f} p90={stats['p90']} "
            f"p99={stats['p99']} max={stats['max']}"
        )

    outliers = find_outlier_episodes(records, args.outlier_multiplier)
    if outliers:
        print(f"\n{len(outliers)} outlier episode(s) (> {args.outlier_multiplier}x median):")
        for outlier in outliers:
            print(
                f"  step {outlier['step']}: {outlier['num_action_tokens']} tokens -- "
                f"{outlier['question'][:80]}"
            )

    skips = find_skips(records)
    if skips:
        print(f"\n{len(skips)} OOM-skipped step(s):")
        for skip in skips:
            questions = ", ".join(q[:60] for q in skip.get("questions_attempted", []))
            print(f"  step {skip['step']}: {skip.get('reason', '?')[:80]} -- {questions}")

    windows = skip_rate_by_window(records, args.window)
    rough_windows = [(start, rate) for start, rate in windows if rate > 0]
    if rough_windows:
        print(f"\nSkip rate by {args.window}-step window (only windows with skips shown):")
        for start, rate in rough_windows:
            print(f"  steps {start}-{start + args.window - 1}: {rate:.0%} skipped")


if __name__ == "__main__":
    main()
