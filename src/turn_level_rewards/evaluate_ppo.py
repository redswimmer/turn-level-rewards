"""evaluate_ppo.py: run a saved MTPPOTrainer checkpoint over the held-out set.

Unlike evaluate.py (which constructs a GRPOTrainer and calls its standard .evaluate()),
MTPPOTrainer has no equivalent built-in evaluation path -- its rollout loop
(_rollout_episode/_collect_batch) is hand-built, not routed through Trainer's
prediction_step/evaluate() machinery. This reuses _rollout_episode directly under
torch.no_grad(), skipping _ppo_update entirely, against the held-out hotpot_qa validation set.

Takes --checkpoint-dir as a required argument (not hardcoded to a "final" checkpoint)
specifically so it can evaluate whichever checkpoint gets judged best from the trackio curves --
final, or last-before-collapse -- matching the paper's own stated evaluation methodology
(arXiv:2505.11821v2 Section 6.1) rather than assuming every run finishes cleanly. See
docs/superpowers/specs/2026-07-23-phase-7b-full-ppo-runs-design.md.
"""

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from turn_level_rewards import data
from turn_level_rewards.rewards import Completion, format_reward, outcome_reward
from turn_level_rewards.train_ppo import (
    MODEL_NAME,
    MTPPOTrainer,
    _final_retrieval_fraction,
    _PolicyAndCritic,
    build_ppo_config,
)

Condition = Literal["ppo", "ppo_mr", "ppo_mr_paper", "mt_ppo"]


class RolloutSource(Protocol):
    """The single thing run_eval needs from a trainer: turn a dataset row into a rollout.

    Typed as a Protocol rather than as MTPPOTrainer so the seam is described by what is
    actually used, not by a concrete class carrying a model, a critic and a GPU -- which is
    what lets tests/unit/ inject a trivial fake and stay fast and GPU-free (CLAUDE.md's
    Guiding principles, dependency inversion at the slow/external boundary).
    """

    def _rollout_episode(self, row: dict, greedy: bool = False) -> dict: ...


# Rows between progress lines. 50 keeps the log readable over a 7,405-row run while still
# reporting inside the first few minutes of a multi-hour job.
_PROGRESS_INTERVAL = 50


def aggregate_eval_metrics(
    completions: list[Completion],
    golden_answers_list: list[list[str]],
    retrieval_fractions: list[float],
) -> dict[str, float]:
    """Turn a batch of rollout results into summary metrics, reusing outcome_reward's own
    per-example exact_match/f1 selection logic (via its log_metric callback) so eval-time
    scoring is byte-for-byte identical to training-time scoring -- no separate reimplementation
    of the "max over golden_answers" argmax logic to drift out of sync with rewards.py.

    format_compliance_rate comes from format_reward for the same reason, and it matters for
    comparability rather than mere convenience: the paper reports format correctness as a
    headline metric alongside EM (Table 2: MT-PPO 0.998, PPO-OR 0.916, GRPO-OR 0.513), and the
    GRPO track already gets it held-out because TRL's GRPOTrainer logs it. Routing it through
    format_reward means both tracks' figures come from the SAME _extract_answer definition, so
    a PPO number and a GRPO number can be put in one column without a caveat.
    """
    logged: dict[str, list[float]] = {"exact_match": [], "f1": [], "format_compliance_rate": []}

    def log_metric(name: str, value: float) -> None:
        logged[name].append(value)

    outcome_reward(completions, golden_answers_list, log_metric=log_metric)
    # Rewards discarded -- only the logged per-completion compliance flags are wanted here.
    format_reward(completions, log_metric=log_metric)

    num_examples = len(completions)
    return {
        "exact_match": sum(logged["exact_match"]) / num_examples,
        "f1": sum(logged["f1"]) / num_examples,
        "format_compliance_rate": sum(logged["format_compliance_rate"]) / num_examples,
        "retrieval_fraction": sum(retrieval_fractions) / num_examples,
        "num_examples": num_examples,
    }


def shard_rows(rows: list[dict], shard: int, num_shards: int) -> list[dict]:
    """Take shard `shard` of `num_shards` from rows, strided so shards are length-balanced.

    Exists because a single-process eval leaves this machine almost entirely idle. Measured
    mid-run on the full 7,405-row mt_ppo eval: 1 of 32 CPU cores busy (load average 1.20), GPU
    at 39-41% utilization and 17-18% memory bandwidth, 5.1 of 24 GB used. Nothing was saturated
    -- a single Python thread was serially issuing one tiny decode kernel per generated token,
    so the limit was one thread's ability to issue work, not any resource running out. Running
    several shards as independent processes fills that idle capacity.

    Strided (rows[shard::num_shards]) rather than contiguous blocks: episode wall-clock varies
    by an order of magnitude with episode length, so a contiguous split can hand one shard a run
    of long episodes and leave every other process waiting on that straggler. Striding
    interleaves them, and also guarantees shard sizes differ by at most one.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard must be in [0, {num_shards}), got {shard}")
    return rows[shard::num_shards]


def merge_shard_metrics(shard_metrics: list[dict]) -> dict:
    """Combine per-shard metrics into one full-set result, weighted by example count.

    Weighted, not a plain mean of shard means: strided shards differ in size by up to one row,
    so an unweighted average would misreport the full-set number by a small but entirely
    avoidable amount.

    Refuses any shard still marked {"complete": false}. Those files exist by design (run_eval
    writes running metrics so a long job is inspectable and a crash costs only the remaining
    rows), which is exactly why merging must reject them -- silently averaging a partial shard
    would report a full-set number computed from a fraction of the set, with nothing in the
    output to reveal it.
    """
    if not shard_metrics:
        raise ValueError("no shard metrics to merge")
    incomplete = [m for m in shard_metrics if m.get("complete") is False]
    if incomplete:
        raise ValueError(
            f"refusing to merge {len(incomplete)} incomplete shard(s) -- "
            "rerun them before merging, or the merged metrics describe a partial eval"
        )

    total = sum(m["num_examples"] for m in shard_metrics)
    weighted = {
        key: sum(m[key] * m["num_examples"] for m in shard_metrics) / total
        for key in ("exact_match", "f1", "format_compliance_rate", "retrieval_fraction")
    }
    return {**weighted, "num_examples": total, "num_shards": len(shard_metrics)}


def build_eval_trainer(condition: Condition, checkpoint_dir: str, seed: int) -> MTPPOTrainer:
    """Composition root: real policy+critic loaded from checkpoint_dir, real SearchEnv (hits the
    live retrieval server), real tokenizer. Not unit-tested -- this is exactly the integration
    surface the live smoke test validates, matching train_ppo.py's build_ppo_trainer.
    """
    policy = AutoModelForCausalLM.from_pretrained(
        str(Path(checkpoint_dir) / "policy"), dtype=torch.bfloat16
    )
    critic = AutoModelForSequenceClassification.from_pretrained(
        str(Path(checkpoint_dir) / "critic"), num_labels=1, dtype=torch.bfloat16
    )
    model = _PolicyAndCritic(policy, critic)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = build_ppo_config(condition=condition, seed=seed, max_steps=0, num_rollouts_per_step=1)
    # Separate output_dir from the real training run's -- mirrors evaluate.py's own
    # "eval-scratch" convention (Phase 6), so this never touches real training checkpoints even
    # though .train()/_save_full_checkpoint are never actually called from this path.
    config.output_dir = f"outputs/{condition}/ppo-eval-scratch"
    # eval_dataset stands in as train_dataset here -- MTPPOTrainer requires one at construction,
    # but .train() is never called, only _rollout_episode directly (see run_eval below).
    eval_dataset = data.load_eval_dataset(n=1, seed=seed)
    return MTPPOTrainer(
        condition=condition,
        model=model,
        tokenizer=tokenizer,
        train_dataset=eval_dataset,
        args=config,
    )


def run_eval(
    trainer: RolloutSource,
    rows: list[dict],
    report_every: int = _PROGRESS_INTERVAL,
    partial_path: Path | None = None,
) -> dict[str, float]:
    """Roll out every row under torch.no_grad() (no _ppo_update -- pure inference), then
    aggregate. Not unit-tested end-to-end -- calls _rollout_episode (real model/tool-calls);
    validated by the live smoke test instead. The progress/partial-write behaviour added here IS
    unit-tested, via an injected fake trainer.

    Reports progress and writes running metrics every `report_every` rows, because the original
    write-once-at-the-end version was operationally unusable at real scale: mt_ppo's full 7,405-row
    eval ran for over 19 hours emitting nothing at all, so there was no way to tell a working job
    from a hung one, or to know whether it was 10% or 90% done, without inferring it from process
    CPU time and per-episode token counts. A job long enough to need babysitting has to say where
    it is.

    Partial results also mean a crash or a kill costs the remaining rows rather than all of them --
    metrics over the first N rows are a real, usable estimate of the full number.
    """
    completions = []
    golden_answers_list = []
    retrieval_fractions = []
    started = time.monotonic()
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            # greedy: evaluation must be deterministic, so the same checkpoint scores the
            # same number every run. See _rollout_episode's greedy parameter.
            rollout = trainer._rollout_episode(row, greedy=True)
            completions.append(rollout["completion"])
            golden_answers_list.append(row["golden_answers"])
            retrieval_fractions.append(_final_retrieval_fraction(rollout))

            if index % report_every == 0 or index == len(rows):
                running = aggregate_eval_metrics(
                    completions, golden_answers_list, retrieval_fractions
                )
                elapsed = time.monotonic() - started
                remaining = (elapsed / index) * (len(rows) - index)
                print(
                    f"{index}/{len(rows)} rows | em={running['exact_match']:.3f} "
                    f"f1={running['f1']:.3f} fmt={running['format_compliance_rate']:.3f} "
                    f"retr={running['retrieval_fraction']:.3f} | "
                    f"{elapsed / index:.2f}s/row eta={remaining / 3600:.1f}h",
                    flush=True,
                )
                if partial_path is not None:
                    _write_metrics(partial_path, {**running, "complete": index == len(rows)})

    return aggregate_eval_metrics(completions, golden_answers_list, retrieval_fractions)


def _write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved MTPPOTrainer checkpoint on the held-out set (see CLAUDE.md)."
    )
    parser.add_argument(
        "--condition", required=True, choices=["ppo", "ppo_mr", "ppo_mr_paper", "mt_ppo"]
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-size", type=int, default=4)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the eval set across this many independent processes (see shard_rows).",
    )
    parser.add_argument(
        "--shard", type=int, default=0, help="Which shard this process evaluates, 0-indexed."
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output or f"results/{args.condition}_ppo_eval_metrics.json")
    label = f"{args.condition} shard {args.shard}/{args.num_shards}"

    # Everything a post-mortem needs, printed before any slow work starts, so a log tail alone
    # identifies which shard/checkpoint/output a crashed process was working on. Reconstructing
    # that from a bare traceback across several concurrent shards is exactly the guesswork this
    # avoids.
    print(
        f"[{label}] checkpoint={args.checkpoint_dir} eval_size={args.eval_size} "
        f"seed={args.seed} output={output_path} started={datetime.now().isoformat(timespec='seconds')}",
        flush=True,
    )

    try:
        trainer = build_eval_trainer(args.condition, args.checkpoint_dir, args.seed)
        rows = shard_rows(
            list(data.load_eval_dataset(n=args.eval_size, seed=args.seed)),
            shard=args.shard,
            num_shards=args.num_shards,
        )
        print(f"[{label}] evaluating {len(rows)} rows", flush=True)
        # Partial metrics land at the real output path as the run proceeds, marked
        # {"complete": false} until the final row, so a long eval is inspectable mid-flight.
        metrics = run_eval(trainer, rows, partial_path=output_path)
    except BaseException as exc:
        # The failure is recorded in the OUTPUT ARTIFACT, not only on stdout: a shard that dies
        # leaves a file saying so, so the merge step refuses it (complete=false) and a reader
        # inspecting results sees the error without having to find and read the right log.
        # BaseException, not Exception, so a KeyboardInterrupt/SystemExit kill is recorded too --
        # those are the likeliest ways a multi-hour job ends.
        _write_metrics(
            output_path,
            {
                "complete": False,
                "error": f"{type(exc).__name__}: {exc}",
                "shard": args.shard,
                "num_shards": args.num_shards,
                "checkpoint_dir": args.checkpoint_dir,
                "failed_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        print(f"[{label}] FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise

    _write_metrics(output_path, {**metrics, "complete": True, "shard": args.shard})
    print(f"[{label}] done: {metrics}", flush=True)
    print(f"Wrote eval metrics to {output_path}")


if __name__ == "__main__":
    main()
