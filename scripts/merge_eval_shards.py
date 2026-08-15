"""Merge per-shard eval metrics into one full-set result.

Thin CLI over evaluate_ppo.merge_shard_metrics -- the weighting and the refuse-incomplete
rules live there and are unit-tested; this only finds the files and prints the result.
"""

import argparse
import json
from pathlib import Path

from turn_level_rewards.evaluate_ppo import merge_shard_metrics


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_files", nargs="+", help="Per-shard metrics json files")
    parser.add_argument("--output", default=None, help="Where to write the merged metrics")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    shard_metrics = []
    for path in args.shard_files:
        metrics = json.loads(Path(path).read_text())
        # Surfaced per file rather than only in the aggregate error: with several shards, which
        # ONE failed is the first thing a reader needs, and the shard file records its own error.
        status = "ok" if metrics.get("complete") else f"INCOMPLETE {metrics.get('error', '')}"
        print(f"  {path}: {metrics.get('num_examples', '?')} rows [{status}]")
        shard_metrics.append(metrics)

    merged = merge_shard_metrics(shard_metrics)
    print(json.dumps(merged, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(merged, indent=2))
        print(f"Wrote merged metrics to {args.output}")


if __name__ == "__main__":
    main()
