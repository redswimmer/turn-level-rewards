"""evaluate.py: run a trained checkpoint over the held-out set via GRPOTrainer.evaluate().

See docs/phase-6-evaluation-comparison.md's "evaluate.py's design" section for why this
approach -- construct a GRPOTrainer for evaluation only and call its standard .evaluate() --
is correct: GRPOTrainer already overrides prediction_step to run the exact same
generation-and-score path training uses, confirmed there against a real checkpoint.
"""

import argparse
import json
from pathlib import Path
from typing import Literal

from trl import GRPOConfig, GRPOTrainer

from turn_level_rewards import data
from turn_level_rewards.env import SearchEnv
from turn_level_rewards.rewards import get_paper_reward_funcs, get_reward_funcs

# grpo_or/grpo_mr are the paper-faithful conditions added in Phase 7c; outcome_only/turn_level
# are this repo's original reward design (Phases 5-6). Both are evaluable here.
Condition = Literal["outcome_only", "turn_level", "grpo_or", "grpo_mr"]


def build_eval_config(condition: Condition, eval_batch_size: int) -> GRPOConfig:
    """Build the GRPOConfig for an eval-only GRPOTrainer.

    num_generations is fixed at 2 -- GRPO's hard minimum (>=2, needs group variance for
    advantages); we don't care about advantage quality at eval time, only the per-completion
    reward/metric values GRPOTrainer.evaluate() returns, so the minimum is the cheapest valid
    choice. per_device_train_batch_size is set to eval_batch_size too, even though .train() is
    never called: GRPOConfig enforces the same generation_batch_size % num_generations == 0
    divisibility constraint training does, and per_device_eval_batch_size alone doesn't satisfy
    it. eval_batch_size must therefore be even (a multiple of num_generations=2) -- checked
    explicitly below so a bad CLI value fails fast with a clear message instead of a cryptic
    error deep inside GRPOTrainer's batching logic.
    """
    if eval_batch_size % 2 != 0:
        raise ValueError(f"eval_batch_size must be even (num_generations=2); got {eval_batch_size}")
    return GRPOConfig(
        output_dir=f"outputs/{condition}/eval-scratch",
        num_generations=2,
        per_device_train_batch_size=eval_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        max_tool_calling_iterations=4,
        beta=0.0,
        max_completion_length=2048,
        report_to="none",
    )


def build_eval_trainer(
    condition: Condition, checkpoint: str, eval_size: int | None, config: GRPOConfig
) -> GRPOTrainer:
    """Composition root: real model checkpoint, real SearchEnv, real held-out data.

    Not unit-tested -- this is exactly the integration surface a real canary/full run validates,
    matching train.py's build_trainer.
    """
    return GRPOTrainer(
        model=checkpoint,
        reward_funcs=(
            get_paper_reward_funcs(condition)
            if condition in ("grpo_or", "grpo_mr")
            else get_reward_funcs(condition)
        ),
        args=config,
        train_dataset=data.load_train_dataset(n=2, seed=42),  # unused filler; .train() never called
        eval_dataset=data.load_eval_dataset(n=eval_size, seed=42),
        environment_factory=SearchEnv,  # ty: ignore[invalid-argument-type]
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse evaluate.py's CLI arguments.

    The bare invocation (just --condition/--checkpoint) evaluates a tiny 4-row slice at a small
    batch size -- a smoke-scale default mirroring train.py's. The real full held-out run passes
    --eval-size 7405 explicitly, the same "pass the exact full size" convention train.py's Phase
    5 launch already established (--train-size 90447).
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on the held-out set (see CLAUDE.md)."
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=["outcome_only", "turn_level", "grpo_or", "grpo_mr"],
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--eval-size", type=int, default=4)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    config = build_eval_config(args.condition, args.eval_batch_size)
    trainer = build_eval_trainer(args.condition, args.checkpoint, args.eval_size, config)
    metrics = trainer.evaluate()

    output_path = Path(args.output or f"results/{args.condition}_eval_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote eval metrics to {output_path}")


if __name__ == "__main__":
    main()
