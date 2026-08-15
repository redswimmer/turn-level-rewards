"""train.py: CLI entrypoint wiring GRPOTrainer + SearchEnv + reward funcs + GRPOConfig + trackio.

See docs/superpowers/specs/2026-07-05-phase-4-training-smoke-test-design.md for the full design
rationale (TRL tool-call-loop semantics, NaN-masking gotchas, measured max_completion_length,
and the generation_batch_size/num_generations divisibility constraint).
"""

import argparse
import math
from datetime import datetime
from typing import Literal

import trackio
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from turn_level_rewards import data
from turn_level_rewards.env import SearchEnv
from turn_level_rewards.rewards import get_paper_reward_funcs, get_reward_funcs

# outcome_only/turn_level use THIS REPO's reward design (Phases 5-6, a completed and
# held-out-confirmed experiment). grpo_or/grpo_mr use the PAPER's reward design (Section 6.1),
# added so the GRPO arms can join the PPO arms in one five-arm Table 2 comparison -- holding
# the reward fixed across algorithms is what makes that comparison valid. The original two are
# kept rather than redefined, so existing written-up results stay meaningful.
Condition = Literal["outcome_only", "turn_level", "grpo_or", "grpo_mr"]

# TRL's _get_per_token_logps_and_entropies chunks its forward pass using per_device_train_batch_size
# as the literal chunk size (grpo_trainer.py:2089) -- a real canary run at num_generations=21 with
# per_device_train_batch_size=21 (no chunking) tried to allocate 28.29 GiB for one logits-to-fp32
# conversion and OOMed a 24GB GPU. A chunk size of 3 still OOMed on the next backward pass (a
# smaller, ~1-4 GiB shortfall each retry, confirmed not a fragmentation artifact --
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True made no difference). 1 (fully sequential,
# single-sequence chunks) was the next divisor down and is the smallest possible footprint;
# confirmed by re-running the same canary successfully (all 3 steps) after this fix.
_MAX_TRAIN_MICRO_BATCH_SIZE = 1


def _train_micro_batch_size(num_generations: int, cap: int = _MAX_TRAIN_MICRO_BATCH_SIZE) -> int:
    """Largest divisor of num_generations that is <= cap.

    Must stay an exact divisor: GRPOConfig reconstructs the full num_generations-sized rollout
    group from `per_device_train_batch_size * gradient_accumulation_steps`, so
    generation_batch_size still equals num_generations exactly (one full rollout group per
    optimizer step, unchanged) -- only the per-token-logps forward/backward pass gets chunked
    smaller, with gradient accumulation combining the chunks back into one real update.
    """
    for candidate in range(min(cap, num_generations), 0, -1):
        if num_generations % candidate == 0:
            return candidate
    return 1  # unreachable: 1 always divides evenly


def build_config(
    condition: Condition,
    seed: int,
    max_steps: int,
    num_generations: int,
) -> GRPOConfig:
    """Build the GRPOConfig for a training run.

    per_device_train_batch_size is capped at _MAX_TRAIN_MICRO_BATCH_SIZE (see its docstring) to
    avoid OOMing the per-token-logps forward pass at large num_generations; gradient_accumulation_steps
    makes up the difference so generation_batch_size still equals num_generations exactly (one
    full rollout group per optimizer step) -- **not** steps_per_generation directly. A real
    6300-step run collapsed into a fixed, zero-variance -0.1 reward within ~300 steps: TRL's own
    docstring says steps_per_generation "defaults to gradient_accumulation_steps" when unset,
    meaning the intended lever IS gradient_accumulation_steps -- setting steps_per_generation
    directly while leaving gradient_accumulation_steps at its default of 1 made each of the
    memory-safe micro-batches trigger its own independent optimizer step (no accumulation) instead
    of being combined into one properly-averaged update per rollout group, i.e. 21 (x2 for
    num_iterations) noisy single-sequence gradient steps instead of 1-2 stable ones per group.
    Fixed by setting gradient_accumulation_steps instead and leaving steps_per_generation unset so
    it defaults to match (confirmed by direct construction: same generation_batch_size==21 result,
    but now with real accumulation).

    Periodic in-training eval (eval_strategy="steps") is deliberately NOT enabled here, despite
    being part of Phase 5's original design: GRPOTrainer's `environments` pool (required by
    SearchEnv) is built once at init, sized to train's generation_batch_size, and reused
    unconditionally for both train and eval (installed trl==1.7.1's grpo_trainer.py:544-545,
    1982) -- confirmed by a real canary run raising `ValueError: zip() argument 2 is longer than
    argument 1` the moment eval ran with a smaller per-step batch than train's. Matching eval's
    batch to train's (21) instead reproduces the exact OOM this function's micro-batching already
    fixes, with no eval-side chunking knob to work around it. This is fixed upstream in TRL's
    `main` branch (commit 8b61980d, "Support multiple environments [1/2]: Pool and build
    environment tool dicts at batch time", PR #6001, merged 2026-07-01) via batch-time environment
    pooling -- confirmed NOT in the 1.7.1 PyPI release (`git merge-base --is-ancestor 8b61980d
    v1.7.1` returns false) or any later release as of this writing. Revisit enabling periodic eval
    by pinning trl to a commit that includes 8b61980d if the live eval curve is wanted later;
    until then, Phase 6's evaluate.py (a full post-hoc run over the entire held-out set) is the
    only held-out signal. See docs/superpowers/specs/2026-07-05-phase-5-full-training-runs-design.md
    for the rest of this phase's paper-grounded config (num_iterations,
    save_strategy/save_steps/save_total_limit).
    """
    train_micro_batch_size = _train_micro_batch_size(num_generations)
    return GRPOConfig(
        output_dir=f"outputs/{condition}",
        seed=seed,
        max_steps=max_steps,
        num_generations=num_generations,
        per_device_train_batch_size=train_micro_batch_size,
        gradient_accumulation_steps=num_generations // train_micro_batch_size,
        max_tool_calling_iterations=4,
        beta=0.0,
        # 8-bit AdamW optimizer states, matching the PPO track (train_ppo.py's
        # use_8bit_optimizer). Two reasons, both load-bearing:
        #
        # Consistency: having the PPO arms on 8-bit and the GRPO arms on fp32 would be another
        # silent asymmetry between tracks that are supposed to be comparable -- exactly the
        # class of unexamined difference that cost this project its five-arm comparison once
        # already.
        #
        # Memory: a 100-step verification run at num_generations=8 peaked at 23.3 GB against a
        # ~23.5 GB card. Unlike MTPPOTrainer, TRL's GRPOTrainer has no catch-OOM-and-skip
        # path, so an OOM here is a hard crash mid-run rather than a skipped step. beta=0.0
        # means no reference model, so the optimizer holds state for a single 0.8B policy:
        # AdamW keeps two fp32 states per parameter (~6.4 GB), quantized to ~1.6 GB here.
        # Measured on a real 100-step run: peak 23.3 GB -> 19.7 GB.
        optim="adamw_bnb_8bit",
        max_completion_length=2048,
        logging_steps=1,
        logging_nan_inf_filter=False,
        log_completions=True,
        gradient_checkpointing=True,
        num_iterations=2,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        report_to="trackio",
        # The paper-faithful arms share ONE trackio project with the PPO arms so every arm
        # overlays in a single dashboard -- that combined view IS the deliverable. The
        # original outcome_only/turn_level runs stay in their own project so this repo's
        # earlier, non-paper-reward results are not silently mixed in with the reproduction.
        project=(
            "turn-level-rewards-paper"
            if condition in ("grpo_or", "grpo_mr")
            else "turn-level-rewards"
        ),
        run_name=condition,
    )


_DEAD_REWARD_STEP_THRESHOLD = 20
_ZERO_STD_STREAK_THRESHOLD = 20


class TrackioAlertCallback(TrainerCallback):
    """Fires trackio alerts for silent-failure modes a clean exit code wouldn't catch.

    See docs/superpowers/specs/2026-07-05-phase-4-training-smoke-test-design.md's "Alert
    callback" section for the reasoning behind each threshold and the re-arming behavior.
    Reward-side NaN detection is intentionally not implemented here: GRPOTrainer aggregates the
    logged `reward` metric via torch.nanmean, which silently drops NaN entries rather than
    propagating them, and this repo's own reward functions have no real numerical path to NaN.
    """

    def __init__(self) -> None:
        self._reward_ever_nonzero = False
        self._dead_reward_alerted = False
        self._zero_std_streak = 0
        self._zero_std_alerted = False

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if not logs:
            return

        loss = logs.get("loss")
        if loss is not None and not math.isfinite(loss):
            trackio.alert(
                title="Non-finite loss",
                text=f"Loss is {loss} at step {state.global_step} -- stopping training.",
                level=trackio.AlertLevel.ERROR,
            )
            control.should_training_stop = True
            return

        reward = logs.get("reward")
        if reward is not None:
            if reward != 0.0:
                self._reward_ever_nonzero = True
            if (
                not self._reward_ever_nonzero
                and not self._dead_reward_alerted
                and state.global_step > _DEAD_REWARD_STEP_THRESHOLD
            ):
                trackio.alert(
                    title="Dead reward",
                    text=(
                        f"Reward has been exactly 0.0 for all {state.global_step} steps so far "
                        "-- possible miswired reward function or tool-calling loop."
                    ),
                    level=trackio.AlertLevel.ERROR,
                )
                self._dead_reward_alerted = True

        frac_zero_std = logs.get("frac_reward_zero_std")
        if frac_zero_std is not None:
            if frac_zero_std == 1.0:
                self._zero_std_streak += 1
            else:
                self._zero_std_streak = 0
                self._zero_std_alerted = False
            if self._zero_std_streak >= _ZERO_STD_STREAK_THRESHOLD and not self._zero_std_alerted:
                trackio.alert(
                    title="No learning signal",
                    text=(
                        f"frac_reward_zero_std has been 1.0 for {self._zero_std_streak} "
                        "consecutive logged steps -- every group is scoring identically, so "
                        "the policy gradient is zero."
                    ),
                    level=trackio.AlertLevel.WARN,
                )
                self._zero_std_alerted = True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse train.py's CLI arguments.

    The bare invocation (just --condition) IS Phase 4's smoke test -- see the design spec's
    "CLI" section. Full-scale runs must explicitly override every size/step/generation flag.
    """
    parser = argparse.ArgumentParser(
        description="Train GRPO with outcome-only or turn-level reward (see CLAUDE.md)."
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=["outcome_only", "turn_level", "grpo_or", "grpo_mr"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--eval-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--penalize-length", action="store_true")
    parser.add_argument("--paper-search-penalty", action="store_true")
    parser.add_argument("--remove-search-cap-prompt", action="store_true")
    return parser.parse_args(argv)


def build_trainer(
    condition: Condition,
    train_size: int | None,
    eval_size: int | None,
    config: GRPOConfig,
    penalize_length: bool = False,
    paper_search_penalty: bool = False,
    remove_search_cap_prompt: bool = False,
) -> GRPOTrainer:
    """Composition root: real model, real SearchEnv (hits the live retrieval server), real data.

    paper_search_penalty drops the prompt's "at most 2 searches" instruction (data.py's
    search_cap_in_prompt=False) and adds rewards.py's search_count_penalty in its place -- see
    that function's docstring for why this is a deliberate experiment, not a paper reproduction
    of the GRPO methodology specifically (the paper's GRPO case study has no such term; this
    borrows their PPO-context mechanism).

    remove_search_cap_prompt is the decoupled control condition: drops the same prompt
    instruction WITHOUT adding search_count_penalty. Added after search_count_penalty collapsed
    both conditions' training, to isolate whether the collapse was caused by removing the prompt
    guidance or by the penalty term itself -- the two were previously only testable bundled
    together via paper_search_penalty. Passing both flags together is redundant but harmless
    (the prompt is dropped either way); penalize_search_count only turns on when
    paper_search_penalty is set, never from this flag alone.

    Not unit-tested -- this is exactly the integration surface the live smoke test validates.
    """
    search_cap_in_prompt = not (paper_search_penalty or remove_search_cap_prompt)
    return GRPOTrainer(
        model="Qwen/Qwen3.5-0.8B",
        reward_funcs=(
            get_paper_reward_funcs(condition)
            if condition in ("grpo_or", "grpo_mr")
            else get_reward_funcs(
                condition,
                penalize_length=penalize_length,
                penalize_search_count=paper_search_penalty,
            )
        ),
        args=config,
        train_dataset=data.load_train_dataset(
            n=train_size, seed=config.seed, search_cap_in_prompt=search_cap_in_prompt
        ),
        eval_dataset=data.load_eval_dataset(
            n=eval_size, seed=config.seed, search_cap_in_prompt=search_cap_in_prompt
        ),
        # SearchEnv.reset requires `metadata`, which is stricter than TRL's `_SupportsReset`
        # protocol (bare **kwargs). This is safe in practice: per CLAUDE.md's "TRL mechanics"
        # section, TRL always calls reset() with the entire sampled dataset row as kwargs, and
        # every row (both loaders in data.py) always includes a `metadata` column.
        environment_factory=SearchEnv,  # ty: ignore[invalid-argument-type]
        callbacks=[TrackioAlertCallback()],
    )


def main() -> None:
    args = _parse_args()
    config = build_config(
        condition=args.condition,
        seed=args.seed,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
    )
    # run_name defaults to the bare condition in build_config (kept there so its unit tests stay
    # simple), but every real invocation -- smoke test, canary, or full run -- must get a unique,
    # self-describing name here at the composition root. Without this, repeated invocations of the
    # same --condition silently share one trackio run record: a real full run at num_generations=21
    # found its own reward/exact_match/f1 curve interleaved with every earlier debug invocation's
    # data under the same run name, with no way to cleanly separate them after the fact.
    length_suffix = "-lengthpenalized" if args.penalize_length else ""
    search_suffix = "-searchpenalized" if args.paper_search_penalty else ""
    nocap_suffix = "-nosearchcapprompt" if args.remove_search_cap_prompt else ""
    config.run_name = (
        f"{args.condition}-{args.max_steps}steps{length_suffix}{search_suffix}{nocap_suffix}-"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    trainer = build_trainer(
        args.condition,
        args.train_size,
        args.eval_size,
        config,
        args.penalize_length,
        args.paper_search_penalty,
        args.remove_search_cap_prompt,
    )
    trainer.train()


if __name__ == "__main__":
    main()
