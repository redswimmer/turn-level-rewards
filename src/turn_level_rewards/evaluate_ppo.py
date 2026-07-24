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
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from turn_level_rewards import data
from turn_level_rewards.rewards import Completion, outcome_reward
from turn_level_rewards.train_ppo import (
    MODEL_NAME,
    MTPPOTrainer,
    _final_retrieval_fraction,
    _PolicyAndCritic,
    build_ppo_config,
)

Condition = Literal["ppo", "mt_ppo"]


def aggregate_eval_metrics(
    completions: list[Completion],
    golden_answers_list: list[list[str]],
    retrieval_fractions: list[float],
) -> dict[str, float]:
    """Turn a batch of rollout results into summary metrics, reusing outcome_reward's own
    per-example exact_match/f1 selection logic (via its log_metric callback) so eval-time
    scoring is byte-for-byte identical to training-time scoring -- no separate reimplementation
    of the "max over golden_answers" argmax logic to drift out of sync with rewards.py.
    """
    logged: dict[str, list[float]] = {"exact_match": [], "f1": []}

    def log_metric(name: str, value: float) -> None:
        logged[name].append(value)

    outcome_reward(completions, golden_answers_list, log_metric=log_metric)

    num_examples = len(completions)
    return {
        "exact_match": sum(logged["exact_match"]) / num_examples,
        "f1": sum(logged["f1"]) / num_examples,
        "retrieval_fraction": sum(retrieval_fractions) / num_examples,
        "num_examples": num_examples,
    }


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


def run_eval(trainer: MTPPOTrainer, rows: list[dict]) -> dict[str, float]:
    """Roll out every row under torch.no_grad() (no _ppo_update -- pure inference), then
    aggregate. Not unit-tested -- calls _rollout_episode (real model/tool-calls); validated by
    the live smoke test instead.
    """
    completions = []
    golden_answers_list = []
    retrieval_fractions = []
    with torch.no_grad():
        for row in rows:
            rollout = trainer._rollout_episode(row)
            completions.append(rollout["completion"])
            golden_answers_list.append(row["golden_answers"])
            retrieval_fractions.append(_final_retrieval_fraction(rollout))
    return aggregate_eval_metrics(completions, golden_answers_list, retrieval_fractions)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved MTPPOTrainer checkpoint on the held-out set (see CLAUDE.md)."
    )
    parser.add_argument("--condition", required=True, choices=["ppo", "mt_ppo"])
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-size", type=int, default=4)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    trainer = build_eval_trainer(args.condition, args.checkpoint_dir, args.seed)
    rows = list(data.load_eval_dataset(n=args.eval_size, seed=args.seed))
    metrics = run_eval(trainer, rows)

    output_path = Path(args.output or f"results/{args.condition}_ppo_eval_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote eval metrics to {output_path}")


if __name__ == "__main__":
    main()
