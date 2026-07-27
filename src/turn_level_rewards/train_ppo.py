"""train_ppo.py: custom multi-turn PPO trainer (MTPPOTrainer) for ppo/mt_ppo conditions.

Built directly on transformers.Trainer, not GRPOTrainer/PPOTrainer -- TRL's PPOTrainer has no
multi-turn tool-calling support (confirmed fresh against the installed 1.7.1 and upstream's
dev branch, re-verified 2026-07-23; see
docs/superpowers/specs/2026-07-05-phase-7-mt-ppo-design.md). Reuses SearchEnv/rewards.py/data.py
unmodified. See CLAUDE.md's Goal section and docs/phase-7-mt-ppo.md for the full design.
"""

import argparse
import inspect
import itertools
import json
import math
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import bitsandbytes as bnb
import torch
import torch.nn as nn
import trackio
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    get_constant_schedule,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import get_last_checkpoint, rotate_checkpoints, set_seed
from trl.chat_template_utils import add_response_schema, get_training_chat_template, parse_response

from turn_level_rewards.env import SearchEnv
from turn_level_rewards.rewards import (
    TURN_REWARD_SCALE,
    _extract_answer,
    paper_binary_outcome_reward,
    paper_outcome_reward,
    paper_turn_reward,
)

Condition = Literal["ppo", "ppo_mr", "mt_ppo"]

MODEL_NAME = "Qwen/Qwen3.5-0.8B"

_SAMPLE_COMPLETION_INTERVAL = 10

# Positions per gradient-checkpointed lm_head chunk. Memory-only knob: it changes peak
# memory, never the numbers. See gather_action_logprobs for the measurements behind 256.
LOGPROB_CHUNK_SIZE = 256

_DEAD_REWARD_STEP_THRESHOLD = 20
_FORMAT_COLLAPSE_STREAK_THRESHOLD = 20


def resolve_stop_token_ids(
    tokenizer_eos_token_id: int, model_eos_token_id: int | list[int] | None
) -> list[int]:
    """Token ids that end a single assistant turn, tokenizer's own eos first.

    Load-bearing, and the root cause of the 2026-07-24 flat-reward incident when it was missing.
    A chat model has two different "end" tokens and they are not interchangeable: for
    Qwen/Qwen3.5-0.8B, `tokenizer.eos_token_id` is `<|im_end|>` (248046, ends an assistant TURN)
    while `model.generation_config.eos_token_id` is `<|endoftext|>` (248044, ends a DOCUMENT).
    `generate()` defaults to the model's, so without this a turn never stops at its own
    terminator: the policy sails past `<|im_end|>` and keeps decoding until it happens to emit
    `<|endoftext|>` or hits max_new_tokens.

    Two separate harms followed, both confirmed against real run data, not theorised:
      1. TRL's response template closes the `content` field at `<|im_end|>`, so trailing text
         opened a SECOND content region that overwrote the first -- the model's real
         `<answer>...</answer>` was discarded before format_reward/outcome_reward ever saw it, and
         reward was a constant -0.1 for 100% of 332 episodes (zero variance, so PPO advantages
         collapsed to ~0 and the policy stopped moving by ~step 20).
      2. The overrun decoded into hallucinated `<|im_start|>user`/`<tool_response>` turns -- the
         policy role-playing the environment -- inflating per-turn token counts (p95 1928, max
         3289 action tokens for a task whose correct answer is ~16) and driving the run's ~15%
         per-step CUDA OOM rate.

    TRL's own GRPOTrainer does exactly this override (`grpo_trainer.py`, `"eos_token_id":
    self._tokenizer.eos_token_id`, with a comment citing transformers#42762), which is why the
    Phase 5/6 GRPO runs were unaffected -- this hand-built rollout loop simply did not inherit it.
    Both ids are kept, not just the tokenizer's: a genuine `<|endoftext|>` should still stop
    generation rather than decode into whatever follows it.
    """
    model_ids = (
        []
        if model_eos_token_id is None
        else [model_eos_token_id]
        if isinstance(model_eos_token_id, int)
        else list(model_eos_token_id)
    )
    stop_ids = [tokenizer_eos_token_id]
    stop_ids.extend(token_id for token_id in model_ids if token_id not in stop_ids)
    return stop_ids


def truncate_after_stop_token(token_ids: list[int], stop_token_ids: list[int]) -> list[int]:
    """Cut a generated turn at its first stop token (kept), dropping anything after it.

    Defense in depth behind resolve_stop_token_ids, not a redundant second copy of it: passing
    the right `eos_token_id` to `generate()` is what SHOULD make this a no-op, but this is the
    guarantee that the action mask can never extend past a turn boundary even if that ever
    regresses again. _rollout_episode marks every returned token `action_mask=1`, so an overrun
    does not merely waste tokens -- PPO would compute ratios and GAE over tokens where the policy
    is impersonating the retrieval server, training it to fabricate tool output.

    A turn with no stop token (hit max_new_tokens mid-sentence) is returned whole: that is a real
    truncated turn, and silently emptying it would destroy the episode rather than record it.
    """
    stops = set(stop_token_ids)
    for index, token_id in enumerate(token_ids):
        if token_id in stops:
            return token_ids[: index + 1]
    return token_ids


@dataclass
class _CollapseAlert:
    title: str
    text: str
    level: str  # "ERROR" or "WARN"
    should_stop: bool = False


class CollapseMonitor:
    """Pure, testable collapse-visibility state machine for MTPPOTrainer.train()'s per-step loop.

    Mirrors train.py's TrackioAlertCallback pattern (non-finite loss stops training; reward
    stuck at 0 for too long is a dead-signal alert) but adapted for PPO -- there is no
    frac_reward_zero_std here (a GRPO group-relative-advantage concept this trainer has no
    equivalent of), and a new format-compliance-collapse check is added instead: the paper's own
    PPO-OR baselines are documented as prone to crashing (arXiv:2505.11821v2 Section 6.1,
    "PPO baselines often crash"), and this exists to make that visible live from the trackio
    curves alone, not to auto-rollback -- see
    docs/superpowers/specs/2026-07-23-phase-7b-full-ppo-runs-design.md's "policy collapse"
    decision for why an automated rollback would be inventing a mechanism the paper never
    describes.

    Returns _CollapseAlert instances instead of calling trackio.alert directly, so this class is
    unit-testable without a live trackio backend -- trackio.alert itself is the external seam,
    kept in train()'s caller (cosmicpython dependency-inversion, same principle CLAUDE.md's
    Guiding principles section already applies throughout this repo).
    """

    def __init__(self) -> None:
        self._reward_ever_nonzero = False
        self._dead_reward_alerted = False
        self._distinct_rewards: set[float] = set()
        self._constant_reward_alerted = False
        self._format_ever_compliant = False
        self._format_collapse_streak = 0
        self._format_collapse_alerted = False
        self._format_never_compliant_alerted = False

    def check(
        self, step: int, loss: float, mean_reward: float, mean_format_reward: float
    ) -> list[_CollapseAlert]:
        if not math.isfinite(loss):
            return [
                _CollapseAlert(
                    title="Non-finite loss",
                    text=f"Loss is {loss} at step {step} -- stopping training.",
                    level="ERROR",
                    should_stop=True,
                )
            ]

        alerts: list[_CollapseAlert] = []

        if mean_reward != 0.0:
            self._reward_ever_nonzero = True
        if (
            not self._reward_ever_nonzero
            and not self._dead_reward_alerted
            and step >= _DEAD_REWARD_STEP_THRESHOLD
        ):
            alerts.append(
                _CollapseAlert(
                    title="Dead reward",
                    text=(
                        f"Reward has been exactly 0.0 for all {step + 1} steps so far -- "
                        "possible miswired reward function, tool-calling loop, or policy "
                        "collapse."
                    ),
                    level="ERROR",
                )
            )
            self._dead_reward_alerted = True

        # A reward pinned at ONE value carries no more learning signal than a reward pinned at
        # zero -- PPO's advantage is (return - baseline), and a critic converges onto a constant
        # almost immediately, driving advantages to ~0. The pre-existing dead-reward check above
        # could not see this: it tests `mean_reward != 0.0`, so the 2026-07-24 incident's constant
        # -0.1 marked the reward "alive" on step 0 and suppressed the alert for all 177 steps
        # while nothing was being learned. Distinct-value counting catches the general case.
        self._distinct_rewards.add(mean_reward)
        if (
            self._reward_ever_nonzero
            and len(self._distinct_rewards) == 1
            and not self._constant_reward_alerted
            and step >= _DEAD_REWARD_STEP_THRESHOLD
        ):
            alerts.append(
                _CollapseAlert(
                    title="Constant reward",
                    text=(
                        f"Mean reward has been exactly {mean_reward} on all {step + 1} steps so "
                        "far -- zero variance means near-zero PPO advantage regardless of the "
                        "reward's magnitude. Check that completions are being parsed as the "
                        "reward functions expect (a rollout-loop bug can score every episode "
                        "identically without ever erroring)."
                    ),
                    level="ERROR",
                )
            )
            self._constant_reward_alerted = True

        if mean_format_reward > 0.0:
            self._format_ever_compliant = True
            self._format_collapse_streak = 0
            self._format_collapse_alerted = False
        elif self._format_ever_compliant:
            self._format_collapse_streak += 1
            if (
                self._format_collapse_streak >= _FORMAT_COLLAPSE_STREAK_THRESHOLD
                and not self._format_collapse_alerted
            ):
                alerts.append(
                    _CollapseAlert(
                        title="Format compliance collapsed",
                        text=(
                            f"format_reward has been non-positive for "
                            f"{self._format_collapse_streak} consecutive steps after previously "
                            "being compliant -- likely policy collapse (matches the paper's own "
                            "documented PPO-OR instability). Check trackio curves and consider "
                            "evaluating an earlier checkpoint."
                        ),
                        level="WARN",
                    )
                )
                self._format_collapse_alerted = True
        elif not self._format_never_compliant_alerted and step >= _FORMAT_COLLAPSE_STREAK_THRESHOLD:
            # The branch above only fires on a REGRESSION from a compliant state, so a run that
            # was never once compliant -- the actual 2026-07-24 signature -- fell through both
            # branches silently. Never-compliant is the more urgent diagnosis of the two: a
            # collapse means the policy lost something it had, this means the plumbing between
            # generation and the reward functions may never have worked at all.
            alerts.append(
                _CollapseAlert(
                    title="Format never compliant",
                    text=(
                        f"format_reward has been non-positive on every one of the first "
                        f"{step + 1} steps -- no completion has ever parsed as a well-formed "
                        "<answer>. Suspect the rollout/parsing path before suspecting the policy."
                    ),
                    level="ERROR",
                )
            )
            self._format_never_compliant_alerted = True

        return alerts


@dataclass
class MTPPOConfig(TrainingArguments):
    """Config for MTPPOTrainer. Subclasses TrainingArguments the same way TRL's GRPOConfig does,
    adding this trainer's own fixed hyperparameters (see this plan's Global Constraints section
    for where each value comes from).
    """

    condition: Condition = "ppo"
    n_max: int = 4
    clip_eps: float = 0.2
    kl_beta: float = 0.001
    policy_lr: float = 1e-6
    critic_lr: float = 1e-5
    gamma: float = 1.0
    gae_lambda: float = 1.0
    num_ppo_epochs: int = 4
    value_loss_coef: float = 0.5
    use_8bit_optimizer: bool = True
    num_rollouts_per_step: int = 2
    save_steps: int = 50
    save_total_limit: int = 3
    max_completion_length: int = 2048
    # Shared with the paper-faithful GRPO arms (train.py) so all five overlay in one
    # dashboard. All three PPO arms use the paper's reward design as of Phase 7c.
    project: str = "turn-level-rewards-paper"


def build_ppo_config(
    condition: Condition,
    seed: int,
    max_steps: int,
    num_rollouts_per_step: int,
    save_steps: int = 50,
    save_total_limit: int = 3,
    use_8bit_optimizer: bool = True,
) -> MTPPOConfig:
    """Build the MTPPOConfig for a training run. Mirrors train.py's build_config role from
    Phase 4 -- fixed hyperparameters are baked in here, not exposed as independent CLI flags.
    """
    return MTPPOConfig(
        output_dir=f"outputs/{condition}",
        seed=seed,
        max_steps=max_steps,
        condition=condition,
        n_max=4,
        clip_eps=0.2,
        kl_beta=0.001,
        policy_lr=1e-6,
        critic_lr=1e-5,
        gamma=1.0,
        gae_lambda=1.0,
        num_ppo_epochs=4,
        value_loss_coef=0.5,
        use_8bit_optimizer=use_8bit_optimizer,
        num_rollouts_per_step=num_rollouts_per_step,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        max_completion_length=2048,
        logging_steps=1,
        run_name=condition,
        report_to="none",  # trackio is called directly in MTPPOTrainer.train(), not through
        # transformers' generic report_to integration -- see Task 8.
    )


def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float = 1.0,
    lam: float = 1.0,
    bootstrap_value: float = 0.0,
) -> list[float]:
    """Generalized Advantage Estimation (standard recursive formula).

    len(values) must equal len(rewards) -- values[t] is the critic's estimate at position t.
    bootstrap_value is V for the (terminal) state after the last reward -- 0.0 for an episode
    that truly ends, since there's no further return to bootstrap from. At this repo's fixed
    gamma=1, lambda=1 (paper's own spec), this reduces toward a full-episode
    Monte-Carlo-return-minus-baseline -- no discount/decay tuning needed.
    """
    if len(rewards) != len(values):
        raise ValueError(
            f"rewards ({len(rewards)}) and values ({len(values)}) must be equal length"
        )
    advantages = [0.0] * len(rewards)
    running_gae = 0.0
    next_value = bootstrap_value
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        running_gae = delta + gamma * lam * running_gae
        advantages[t] = running_gae
        next_value = values[t]
    return advantages


def place_paper_rewards(
    num_tokens: int,
    turn_boundary_token_indices: list[int],
    turn_rewards: list[float],
    outcome_reward_value: float,
    condition: Condition,
) -> list[float]:
    """Place the paper's R^I / R^O onto the action-token sequence, per condition.

    The three arms differ ONLY in where reward lands -- the components are computed by the
    caller (rewards.paper_turn_reward / paper_outcome_reward / paper_binary_outcome_reward), so
    this function is the whole of the OR/MR/MT distinction and nothing else:

      ppo     (PPO-OR) : R^O alone, on the last token. Section 6.1: "vanilla PPO trained with
                         only outcome rewards, where the trajectory-level reward is a binary
                         signal indicating final-answer correctness." No R^I at all.
      ppo_mr  (PPO-MR) : R^O + sum(R^I), ALL on the last token. Section 6.1 calls this a
                         "trajectory-level reward [that] combines intermediate rewards
                         (retrieval correctness) and outcome rewards" -- same information as
                         MT-PPO, collapsed into one terminal scalar.
      mt_ppo  (MT-PPO) : R^I at each intermediate turn boundary, R^O on the last token. This is
                         Eq. 9, and the paper's actual contribution.

    PPO-MR is the control that makes the ablation interpretable. ppo -> ppo_mr isolates the
    value of the extra reward signal; ppo_mr -> mt_ppo isolates the value of turn-level
    placement. Comparing only ppo to mt_ppo (as this repo did before) changes both at once, so
    any measured advantage could not be attributed to Eq. 9.

    turn_boundary_token_indices index the compressed ACTION-token sequence (policy-generated
    tokens only) -- see _rollout_episode's docstring.
    """
    if len(turn_boundary_token_indices) != len(turn_rewards):
        raise ValueError(
            f"turn_boundary_token_indices ({len(turn_boundary_token_indices)}) and turn_rewards "
            f"({len(turn_rewards)}) must be equal length"
        )
    if num_tokens < 1:
        raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")

    per_token_rewards = [0.0] * num_tokens
    per_token_rewards[-1] += outcome_reward_value

    if condition == "ppo_mr":
        per_token_rewards[-1] += sum(turn_rewards)
    elif condition == "mt_ppo":
        for token_index, turn_reward_value in zip(
            turn_boundary_token_indices, turn_rewards, strict=True
        ):
            per_token_rewards[token_index] += turn_reward_value
    return per_token_rewards


def place_turn_rewards(
    num_tokens: int,
    turn_boundary_token_indices: list[int],
    retrieval_fraction_after_each_turn: list[float],
    format_and_outcome_reward: float,
    condition: Condition,
    turn_reward_scale: float = TURN_REWARD_SCALE,
) -> list[float]:
    """Eq. 9 turn-boundary reward placement.

    R^O (format_reward + outcome_reward, summed by the caller) always lands on the trajectory's
    last token. R^I -- turn_reward's marginal per-turn contribution -- lands at each intermediate
    turn boundary, mt_ppo only; always 0 for ppo (single lump-sum credit assignment even across a
    multi-turn episode, per the paper's Eq. 9).

    turn_boundary_token_indices and retrieval_fraction_after_each_turn operate over whatever
    token-index space the caller is using (this repo's MTPPOTrainer uses action-token-relative
    indices, i.e. only counting policy-generated tokens -- see _rollout_episode's docstring).
    retrieval_fraction_after_each_turn[i] is SearchEnv.retrieval_fraction sampled immediately
    after intermediate turn i's tool call executed. retrieval_fraction is monotonically
    non-decreasing (SearchEnv only ever adds to its hit set), so each turn's real, marginal
    contribution is that turn's value minus the previous turn's (0.0 before the first turn) --
    not the raw cumulative value, which would double-count every later turn's reward.
    """
    if len(turn_boundary_token_indices) != len(retrieval_fraction_after_each_turn):
        raise ValueError(
            "turn_boundary_token_indices and retrieval_fraction_after_each_turn must be equal "
            "length"
        )
    per_token_rewards = [0.0] * num_tokens
    per_token_rewards[-1] += format_and_outcome_reward
    if condition == "mt_ppo":
        previous_fraction = 0.0
        for token_index, cumulative_fraction in zip(
            turn_boundary_token_indices, retrieval_fraction_after_each_turn, strict=True
        ):
            marginal = cumulative_fraction - previous_fraction
            per_token_rewards[token_index] += turn_reward_scale * marginal
            previous_fraction = cumulative_fraction
    return per_token_rewards


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def compute_ppo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    new_values: torch.Tensor,
    action_mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.001,
    value_loss_coef: float = 0.5,
) -> dict[str, torch.Tensor]:
    """PPO-clip policy loss + value loss + a direct KL penalty term, all masked to action tokens.

    action_mask is 1.0 at positions that are real policy-generated action tokens, 0.0 elsewhere
    (prompt tokens, tool-response tokens injected by the environment, padding) -- none of those
    should ever receive policy gradient. The KL term uses old_logprobs (the rollout-time, frozen
    policy snapshot) as the reference throughout every one of this batch's inner PPO epochs --
    not a separate frozen reference model the way GRPO's beta works (see the design spec's
    stated assumption on this point). Both the clip and the KL term are applied together, not
    either/or, per the paper's spec.
    """
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -_masked_mean(torch.min(unclipped, clipped), action_mask)
    value_loss = _masked_mean((new_values - returns) ** 2, action_mask)
    # Schulman's k3 estimator: exp(d) - d - 1 with d = old - new. Non-negative by construction,
    # low variance, and the same estimator TRL's GRPO uses.
    #
    # The original implementation used mean(new_logprobs - old_logprobs) directly. Over tokens
    # sampled from the OLD policy that expectation is -KL(old||new) <= 0, so adding
    # +kl_beta * it to the loss and minimising MAXIMISED divergence from the rollout policy --
    # the exact opposite of the trust region the term is documented to provide. kl_beta=0.001
    # kept the magnitude small, but the sign was backwards, leaving PPO with no trust region
    # beyond clipping plus a slight push away from it.
    log_ratio_reversed = old_logprobs - new_logprobs
    kl = _masked_mean(torch.exp(log_ratio_reversed) - log_ratio_reversed - 1.0, action_mask)
    loss = policy_loss + value_loss_coef * value_loss + kl_beta * kl

    # Diagnostic-only fields (no gradient contribution) -- mirror TRL's own PPOTrainer metric
    # set (`val/ratio`, `val/ratio_var`, `policy/clipfrac_avg` -- see
    # docs/trl/v1.9.0/ppo_trainer's "Explanation of the logged metrics"), the standard PPO
    # health signals for diagnosing an unstable run from trackio curves alone. clip_fraction
    # uses the raw (pre-clip) ratio against the same clip_eps bounds compute_ppo_loss itself
    # clips to, so it reports exactly what fraction of this batch's surrogate objective was
    # actually clamped. To avoid inf * 0.0 = nan when masking out positions with very large
    # exponents, we replace inf values with a large but finite sentinel (1e6) for the
    # diagnostic computation -- this preserves the "clipped or not" decision (both are way outside
    # bounds) while avoiding nan propagation.
    ratio_safe = torch.where(torch.isinf(ratio), torch.full_like(ratio, 1e6), ratio)
    ratio_mean = _masked_mean(ratio_safe, action_mask)
    ratio_variance = _masked_mean((ratio_safe - ratio_mean) ** 2, action_mask)
    is_clipped = (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)
    clip_fraction = _masked_mean(is_clipped.float(), action_mask)

    return {
        "loss": loss,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "kl": kl.detach(),
        "ratio_mean": ratio_mean.detach(),
        "ratio_variance": ratio_variance.detach(),
        "clip_fraction": clip_fraction.detach(),
    }


class _PolicyAndCritic(nn.Module):
    """Wraps the policy (causal LM) and critic (sequence-classification head) as one nn.Module.

    Lets Trainer's standard model/optimizer/checkpoint plumbing see both submodules through a
    single self.model, while create_optimizer still gives them independent learning rates (the
    paper's own spec: policy_lr=1e-6, critic_lr=1e-5) via separate param groups.
    """

    def __init__(self, policy: PreTrainedModel, critic: PreTrainedModel) -> None:
        super().__init__()
        self.policy = policy
        self.critic = critic


def build_policy_and_critic(model_name: str = MODEL_NAME) -> _PolicyAndCritic:
    """Real policy + real critic, both loaded from the same base checkpoint -- separate models,
    not a shared backbone (the paper's own spec). Not unit-tested: loads real weights: validated
    by the live smoke test (Task 11) instead.

    gradient_checkpointing_enable() is load-bearing, not an optional perf knob: this real bug was
    caught by Task 13's live smoke test. Qwen3.5 is a hybrid architecture with linear-attention
    ("gated delta rule") layers; `flash-linear-attention`/`causal-conv1d` aren't installed in this
    repo's env (see the "fast path is not available" warning printed at model-load time), so those
    layers fall back to a naive pure-PyTorch path whose intermediate activations are far more
    memory-hungry than plain full attention at the same sequence length. Confirmed directly: a
    real forward+backward through both policy and critic without checkpointing hit a masked-OOM
    (this machine's broken NVML/driver mismatch turns a real `CUDA out of memory` into a confusing
    `NVML_SUCCESS ... INTERNAL ASSERT FAILED` -- see docs/phase-7-mt-ppo.md's Handoff notes) at
    just seq_len=2048 -- squarely within this repo's real n_max=4-turn,
    max_completion_length=2048 rollout range. With gradient_checkpointing_enable() on both models,
    the same setup peaked at ~10.7GB even at seq_len=8192. Without this, the live smoke test
    cannot complete a single PPO update on this repo's single RTX 4090 (matches CLAUDE.md Phase
    4's own precedent of needing gradient_checkpointing=True for GRPO's OOM).
    """
    policy = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    critic = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, dtype=torch.bfloat16
    )
    policy.gradient_checkpointing_enable()
    critic.gradient_checkpointing_enable()
    return _PolicyAndCritic(policy, critic)


def _final_retrieval_fraction(rollout: dict) -> float:
    """The episode's final (cumulative) retrieval_fraction -- 0.0 if the episode made no search
    calls at all. Shared by _collect_batch (train-time) and evaluate_ppo.py (eval-time) so both
    extract this the same way.
    """
    fractions = rollout["retrieval_fraction_after_each_turn"]
    return fractions[-1] if fractions else 0.0


def _row_cycle_from_step(
    rows: list[dict], global_step: int, num_rollouts_per_step: int
) -> Iterator[dict]:
    """An infinite cycle over rows, fast-forwarded to the position a from-scratch run would be
    at after global_step completed steps -- so resuming from a checkpoint reproduces the exact
    same data order a run that never stopped would have seen. itertools.cycle over a fixed list
    is deterministic, so replaying the same number of draws always lands in the same place.
    """
    cycle = itertools.cycle(rows)
    for _ in range(global_step * num_rollouts_per_step):
        next(cycle)
    return cycle


class MTPPOTrainer(Trainer):
    """Custom multi-turn PPO trainer with tool-calling, built directly on transformers.Trainer.

    Owns: the rollout loop (render with tools -> generate -> parse_response -> execute
    SearchEnv.search() on a tool call -> append tool message -> repeat up to args.n_max turns ->
    require a final <answer>), the critic forward pass, Eq. 9 reward placement, GAE, and the
    PPO-clip + KL-penalty + value-loss update. Turn-level credit assignment for mt_ppo falls out
    of reward placement + GAE bootstrapping alone -- no MT-GRPO-style extra-rollout advantage
    trick is needed here (PPO already has a real per-token critic). See
    docs/superpowers/specs/2026-07-05-phase-7-mt-ppo-design.md's Context section for why this is
    built on transformers.Trainer directly rather than subclassing GRPOTrainer/PPOTrainer.
    """

    def __init__(
        self,
        condition: Condition,
        model: _PolicyAndCritic,
        tokenizer,
        train_dataset,
        args: MTPPOConfig,
        environment_factory=None,
        callbacks=None,
    ) -> None:
        self.condition = condition
        self.environment_factory = environment_factory or SearchEnv
        super().__init__(model=model, args=args, train_dataset=train_dataset, callbacks=callbacks)
        self.tokenizer = add_response_schema(tokenizer)
        self.training_chat_template = get_training_chat_template(self.tokenizer)
        # Resolved once here rather than per-generate call so the policy's own generation_config
        # is read before any of it is mutated, and so a mismatch is visible in one place. See
        # resolve_stop_token_ids' docstring for why this is not optional.
        self.stop_token_ids = resolve_stop_token_ids(
            self.tokenizer.eos_token_id, model.policy.generation_config.eos_token_id
        )

    # create_optimizer(self) intentionally omits the base transformers.Trainer.create_optimizer's
    # optional `model` parameter. The internal Trainer code path that passes it positionally
    # is only reached for FSDP-XLA, SageMaker MP-DP, or DataParallel; this repo's single-GPU
    # setup (per CLAUDE.md's Hardware section) never triggers it, so the omission is safe.
    def create_optimizer(self) -> torch.optim.Optimizer:  # ty: ignore[invalid-method-override]
        """One AdamW, two param groups -- policy_lr / critic_lr per the paper's spec (10x apart).

        Also sets self.lr_scheduler to a no-op constant schedule (this trainer never decays either
        learning rate -- the paper's own spec fixes policy_lr/critic_lr for the run's duration).
        This is load-bearing, not cosmetic: a real bug found while verifying Task 5's
        checkpoint-resume fix -- transformers.Trainer.__init__ initializes self.lr_scheduler to
        None, and train() never otherwise assigns it, so Trainer's own inherited
        _save_optimizer_and_scheduler/_load_optimizer_and_scheduler (used by
        _save_full_checkpoint/_load_full_checkpoint) unconditionally call
        self.lr_scheduler.state_dict()/.load_state_dict() -- a real crash on the very first
        checkpoint save, `AttributeError: 'NoneType' object has no attribute 'state_dict'`,
        confirmed by direct reproduction. get_constant_schedule's LambdaLR holds only a reference
        to the optimizer plus a step counter (no per-parameter tensor state), so unlike the
        optimizer itself, there is no analogous orphaned-reference risk here across a resume.
        """
        # self.model.policy, self.model.critic, self.args.policy_lr, and self.args.critic_lr
        # are all real attributes defined on this file's _PolicyAndCritic and MTPPOConfig
        # classes respectively. They are safe; ty's inability to see through Trainer's looser
        # base types is not a real issue here.
        # 8-bit optimizer states, for memory not for speed. AdamW keeps two fp32 states per
        # parameter across BOTH the policy and the critic, measured at ~6.4 GB of this setup's
        # ~10 GB static floor on a 24 GB card; 8-bit quantization takes that to ~3.2 GB.
        #
        # This is load-bearing for experimental validity, not just throughput. Without it,
        # mt_ppo skips OOM steps that ppo/ppo_mr do not (its turn reward produces longer
        # episodes), so the arms of the comparison silently receive different numbers of real
        # gradient updates -- the exact confound that invalidated the 2026-07-25 runs, where
        # mt_ppo got 431 updates against ppo's 499. Measured: 13.8% skips before the log-prob
        # chunking fix, 2.8% after it, 0% with this as well.
        #
        # AdamW8bit is a drop-in for torch.optim.AdamW (same update rule, quantized state), so
        # it changes memory rather than optimization semantics.
        optimizer_class = bnb.optim.AdamW8bit if self.args.use_8bit_optimizer else torch.optim.AdamW  # ty: ignore[unresolved-attribute]
        self.optimizer = optimizer_class(
            [
                {"params": self.model.policy.parameters(), "lr": self.args.policy_lr},  # ty: ignore[unresolved-attribute]
                {"params": self.model.critic.parameters(), "lr": self.args.critic_lr},  # ty: ignore[unresolved-attribute]
            ]
        )
        self.lr_scheduler = get_constant_schedule(self.optimizer)
        return self.optimizer

    def _rollout_episode(self, row: dict) -> dict:
        """Run one multi-turn episode for a single dataset row: real generation, real tool calls
        against the real retrieval server (via self.environment_factory, e.g. SearchEnv).

        Returns everything downstream needs, all expressed over a "compressed action-token"
        index space that only counts policy-generated (assistant) tokens -- prompt tokens and
        tool-response tokens (environment-injected, not sampled by the policy) are excluded from
        this space entirely, matching how GAE/PPO treat each policy-generated token as one RL
        timestep:
          - full_token_ids: every token in the final rendered conversation (prompt + all turns),
            in order -- fed to the policy/critic for full context.
          - action_mask: same length as full_token_ids, 1 at positions the policy generated,
            0 elsewhere.
          - turn_boundary_action_indices: for each intermediate turn (one that made a tool call,
            i.e. not the final answering turn), the index INTO THE COMPRESSED ACTION-TOKEN
            SEQUENCE (not full_token_ids) of that turn's last generated token.
          - retrieval_fraction_after_each_turn: SearchEnv.retrieval_fraction sampled immediately
            after that same turn's tool call executed -- one entry per turn_boundary_action_index,
            same order.

        Relies on get_training_chat_template's prefix-preserving guarantee (confirmed supported
        for Qwen3.5): each turn's freshly-rendered prompt is guaranteed to start with exactly the
        tokens already recorded in full_token_ids, so the new suffix at each turn is unambiguous.
        """
        environment = self.environment_factory()
        environment.reset(**row)
        messages = list(row["prompt"])
        # self.model is this file's _PolicyAndCritic and self.args is this file's MTPPOConfig at
        # runtime (both set in __init__/composed by build_ppo_config), but Trainer's base type
        # stubs only know them as the looser `nn.Module | None` / `TrainingArguments` -- same
        # ty-can't-see-through-Trainer's-base-types situation already noted on create_optimizer
        # above. Safe to ignore here for the same reason.
        policy = self.model.policy  # ty: ignore[unresolved-attribute]

        full_token_ids: list[int] = []
        action_mask: list[int] = []
        turn_boundary_action_indices: list[int] = []
        retrieval_fraction_after_each_turn: list[float] = []
        # Per-turn inputs to the paper's R^I (rewards.paper_turn_reward): whether this
        # turn's tool calls were well-formed, and how many searches have been issued in
        # total up to and including this turn (the paper's n_search is cumulative).
        turn_format_ok: list[bool] = []
        cumulative_searches_after_each_turn: list[int] = []
        searches_so_far = 0
        num_action_tokens = 0

        # n_max + 1 generations for n_max tool rounds, so the policy ALWAYS gets one final turn
        # after its last tool result in which to answer.
        #
        # The original loop was `for _turn in range(n_max)` with tool execution at the end of
        # each pass, which meant a model that called a tool on the last pass had its tool
        # executed and the episode then ended on a `role: tool` message -- no generation left in
        # which to produce an <answer>, and a -0.1 format penalty for a turn it was never given.
        # Measured over every logged episode: 58% of ppo and 21% of mt_ppo episodes ended that
        # way, structurally unable to answer.
        #
        # This also made the PPO track incomparable to the GRPO track it is being measured
        # against. TRL's GRPOTrainer._tool_call_loop runs `while tool_calls and iteration_num <
        # max`, executing tools and THEN generating (grpo_trainer.py: tool message appended,
        # then _generate_single_turn, then iteration_num += 1) -- so at the same nominal
        # max_tool_calling_iterations=4, GRPO's conversations always end on a generation and
        # ours could not. Verified in TRL's source, not assumed from documentation.
        for _turn in range(self.args.n_max + 1):  # ty: ignore[unresolved-attribute]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tools=[environment.search],
                add_generation_prompt=True,
                chat_template=self.training_chat_template,
                tokenize=False,
            )
            prompt_token_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

            new_context_tokens = prompt_token_ids[len(full_token_ids) :]
            full_token_ids.extend(new_context_tokens)
            action_mask.extend([0] * len(new_context_tokens))

            # policy.device is a real PreTrainedModel attribute, but ty can't see through the
            # loose Trainer base-class type annotation on self.model -- same root cause as the
            # unresolved-attribute suppressions in create_optimizer above.
            input_ids = torch.tensor([prompt_token_ids], device=policy.device)  # ty: ignore[invalid-argument-type]
            # policy.generate needs use_cache=True for efficient autoregressive decoding, but
            # transformers forces use_cache=False whenever gradient_checkpointing is enabled AND
            # the model is in training mode (see build_policy_and_critic's docstring for why
            # checkpointing itself is required here) -- so generation must run with the model
            # temporarily in eval() mode, restored to train() immediately after. Confirmed via the
            # live smoke test: without this toggle, every generate() call falls back to
            # use_cache=False and pays a large, avoidable slowdown for no memory benefit (rollout
            # generation is always under torch.no_grad() below, so checkpointing's memory savings
            # never applied here in the first place).
            policy.eval()  # ty: ignore[unresolved-attribute]
            with torch.no_grad():
                # policy.generate is a real PreTrainedModel method, but ty can't see it through
                # the loose Trainer base-class type annotation -- policy is provably an
                # AutoModelForCausalLM instance at runtime, always has .generate.
                generation = policy.generate(  # ty: ignore[call-non-callable, unresolved-attribute]
                    input_ids,
                    max_new_tokens=self.args.max_completion_length,  # ty: ignore[unresolved-attribute]
                    do_sample=True,
                    temperature=1.0,
                    # Without this, generate() falls back to the policy's own
                    # generation_config.eos_token_id (<|endoftext|>, a DOCUMENT terminator) and
                    # sails straight past each turn's <|im_end|>. See resolve_stop_token_ids.
                    eos_token_id=self.stop_token_ids,
                )
            policy.train()  # ty: ignore[unresolved-attribute]
            new_token_ids = truncate_after_stop_token(
                generation[0, len(prompt_token_ids) :].tolist(), self.stop_token_ids
            )
            parsed = parse_response(self.tokenizer, new_token_ids, prefix=prompt_token_ids)
            messages.append(parsed)

            full_token_ids.extend(new_token_ids)
            action_mask.extend([1] * len(new_token_ids))
            num_action_tokens += len(new_token_ids)

            tool_calls = parsed.get("tool_calls") or []
            if not tool_calls:
                break  # final answer turn -- episode complete
            if _turn == self.args.n_max:  # ty: ignore[unresolved-attribute]
                # Tool budget exhausted. Stop WITHOUT executing this turn's calls: the episode
                # already ends on a generation (the guarantee above), and running a tool whose
                # result no turn can ever read would only add unattributable tokens. Matches
                # TRL, whose loop likewise leaves the final turn's calls unexecuted.
                break

            turn_had_bad_call = False
            for tool_call in tool_calls:
                # A real failure Task 13's live smoke test caught: an untrained (or
                # early-training) policy can hallucinate a tool call that doesn't match
                # search()'s real signature -- an unexpected/extra/missing argument name
                # (observed directly: "search() got an unexpected keyword argument 'return'"),
                # or a malformed tool_call dict missing "function"/"arguments" entirely. This
                # must not crash the whole training run: it's exactly the kind of
                # self-correctable mistake this episode's own format_reward/outcome_reward
                # should teach the policy out of over time, not a fatal error.
                #
                # This is split into three steps -- extract args, validate the binding, then call
                # -- specifically so that a genuine retrieval-server/infra failure raised from
                # *inside* search()'s own body (e.g. SearchEnv.search()'s `doc["title"]` access
                # raising KeyError on a malformed document from the retrieval server) is never
                # caught here and mislabeled as a model mistake. An earlier version wrapped the
                # entire `environment.search(**tool_call["function"]["arguments"])` call in one
                # try/except(TypeError, KeyError) -- that accidentally also caught KeyErrors
                # raised from inside search()'s body, silently absorbing real infra bugs as if
                # they were hallucinated tool calls. Splitting the extraction, the signature
                # validation, and the actual (uncaught) call into three separate steps closes
                # that gap.

                # Step 1: pull the arguments out of the tool_call dict. A KeyError/TypeError here
                # means the dict itself is malformed (missing "function"/"arguments", or
                # "arguments" isn't a mapping) -- always a model-output problem, never
                # search()'s fault, since search() hasn't been touched yet.
                try:
                    call_args = tool_call["function"]["arguments"]
                except (TypeError, KeyError) as exc:
                    messages.append(
                        {
                            "role": "tool",
                            "name": "search",
                            "content": f"Error: invalid arguments for search(): {exc}",
                        }
                    )
                    continue

                # Step 2: check the extracted arguments are a valid call to search() WITHOUT
                # calling it -- names/arity via inspect binding, plus types against search()'s
                # annotations. Neither check ever executes search()'s body, so anything caught
                # here is guaranteed to be a model-produced argument mistake rather than
                # something that happened inside search(). See validate_tool_call_arguments for
                # why the type half is not optional.
                argument_error = validate_tool_call_arguments(environment.search, call_args)
                if argument_error is not None:
                    turn_had_bad_call = True
                    messages.append(
                        {
                            "role": "tool",
                            "name": "search",
                            "content": f"Error: {argument_error}",
                        }
                    )
                    continue

                # Step 3: only now actually call search(), deliberately OUTSIDE any try/except.
                # Both preceding checks already ruled out a malformed tool_call and an
                # argument-mismatch, so any exception raised here comes from genuinely inside
                # search()'s own body (e.g. a KeyError from a malformed retrieval-server
                # document) -- a real infra failure that must propagate as a visible crash, not
                # be silently absorbed as if it were a model mistake.
                result = environment.search(**call_args)
                messages.append({"role": "tool", "name": "search", "content": result})

            searches_so_far += len(tool_calls)
            turn_boundary_action_indices.append(num_action_tokens - 1)
            retrieval_fraction_after_each_turn.append(environment.retrieval_fraction)
            turn_format_ok.append(not turn_had_bad_call)
            cumulative_searches_after_each_turn.append(searches_so_far)

        completion = messages[len(row["prompt"]) :]
        return {
            "row": row,
            "completion": completion,
            "full_token_ids": full_token_ids,
            "action_mask": action_mask,
            "turn_boundary_action_indices": turn_boundary_action_indices,
            "turn_format_ok": turn_format_ok,
            "cumulative_searches_after_each_turn": cumulative_searches_after_each_turn,
            "retrieval_fraction_after_each_turn": retrieval_fraction_after_each_turn,
        }

    def _action_indices(self, full_token_ids: list[int], action_mask: list[int]) -> torch.Tensor:
        """Shared helper: input_ids tensor + the action-token position indices, used by both
        _forward_policy_logprobs and _forward_critic_values so they stay consistent."""
        # self.model.policy resolves through Trainer's loose base-class type (nn.Module | None),
        # so ty can't see .device as a real attribute -- same root cause as the
        # unresolved-attribute suppressions already used in create_optimizer and
        # _rollout_episode above. Safe: self.model.policy is provably a real PreTrainedModel
        # instance at runtime, always has a genuine .device.
        device = self.model.policy.device  # ty: ignore[unresolved-attribute]
        # device's type is unresolved to ty for the same reason noted immediately above, so it
        # can't confirm device is a valid torch.device-compatible value for this device= kwarg --
        # same root cause as every other `device=device` call site in this class. Safe: device is
        # always a genuine torch.device at runtime.
        mask = torch.tensor(action_mask, device=device, dtype=torch.bool)  # ty: ignore[invalid-argument-type]
        return mask.nonzero(as_tuple=True)[0]

    def _forward_policy_logprobs(
        self, full_token_ids: list[int], action_mask: list[int]
    ) -> torch.Tensor:
        """Policy-only teacher-forced forward pass -- returns action_logprobs, a 1-D tensor of
        length sum(action_mask) (one entry per action/policy-generated token).

        Split out from the critic's own forward (_forward_critic_values) -- a real fix for a real
        bug the live smoke test caught. mt_ppo's PPO update OOM'd intermittently (roughly half the
        time, at episodes as short as ~750 tokens) even after gradient_checkpointing_enable() and
        this method's own selective-lm_head fix (see build_policy_and_critic's and this file's
        git history for that story): the with-grad "new" pass computed BOTH policy's and critic's
        full computational graphs before either backward() call, so both stayed alive
        simultaneously -- roughly double the peak memory of computing+backpropagating each
        model's own loss term sequentially instead. _ppo_update now calls this method, backward()s
        through the policy+KL loss alone (freeing this graph), THEN calls _forward_critic_values
        and backward()s through the value loss alone -- mathematically identical total gradients
        (policy_loss+kl_beta*kl and value_loss_coef*value_loss are independent additive terms over
        disjoint parameters in compute_ppo_loss's formula, so splitting the backward call changes
        nothing about the resulting per-parameter gradients), just without ever holding both
        graphs in memory at once.

        Applies the policy's LM head only at the handful of positions whose log-prob is actually
        needed (one per action token), instead of over the whole sequence -- the OOM-causing bug
        this fixed first. This model's vocab is 248,320 tokens; a naive `policy(input_ids).logits`
        call computes and keeps alive a `[seq_len, vocab]` tensor for EVERY position (prompt
        tokens and environment-injected tool-response tokens included, which this repo's episodes
        have many more of than actual action tokens) -- gradient checkpointing only checkpoints
        the backbone's transformer layers, not this vocab-sized lm_head/log_softmax step sitting
        outside it, so that tensor's memory was still the dominant, unbounded-by-checkpointing
        cost. Computing the backbone's hidden states once (cheap: `[seq_len, hidden_size=1024]`,
        not `[seq_len, vocab]`) and applying `lm_head` only at `predict_positions` (one per action
        token) reduces that tensor from O(seq_len * vocab) to O(num_action_tokens * vocab).
        """
        device = self.model.policy.device  # ty: ignore[unresolved-attribute]
        input_ids = torch.tensor([full_token_ids], device=device)  # ty: ignore[invalid-argument-type]
        action_indices = self._action_indices(full_token_ids, action_mask)
        # An action token at absolute position t was predicted by the model's output at position
        # t-1 (standard next-token-prediction shift).
        predict_positions = action_indices - 1

        # self.model.policy is untyped/ambiguous to ty (see above), so calling `.model(...)` looks
        # like a call-non-callable with an unresolved-attribute layered on top (ty can't see
        # `.model`/`.lm_head` on self.model's loose base type either). Safe: self.model.policy is
        # provably a real Qwen3_5ForCausalLM instance at runtime, whose `.model` (backbone) and
        # `.lm_head` (the vocab projection) are both real, always-callable submodules.
        policy_hidden = self.model.policy.model(input_ids=input_ids).last_hidden_state[0]  # ty: ignore[call-non-callable, unresolved-attribute]  # [seq_len, hidden]
        selected_hidden = policy_hidden[predict_positions]  # [num_action_tokens, hidden]
        next_tokens = input_ids[0, action_indices]
        # Chunked rather than one [num_action_tokens, vocab] pass: identical numbers, but a peak
        # that no longer grows with episode length. See gather_action_logprobs for the measured
        # sizes and for why this was the term behind mt_ppo's 13.8% OOM step-skip rate.
        return gather_action_logprobs(
            self.model.policy.lm_head,  # ty: ignore[unresolved-attribute]
            selected_hidden,
            next_tokens,
        )

    def _forward_critic_values(
        self, full_token_ids: list[int], action_mask: list[int]
    ) -> torch.Tensor:
        """Critic-only teacher-forced forward pass -- returns action_values, a 1-D tensor of
        length sum(action_mask). See _forward_policy_logprobs's docstring for why this is a
        separate method rather than one combined policy+critic forward.

        The critic's head (`.score`) projects to a single scalar per position (num_labels=1), not
        a vocab-sized dimension, so it never needed the selective-position trick
        _forward_policy_logprobs uses -- computed over the whole sequence and indexed afterward is
        fine here.
        """
        device = self.model.critic.device  # ty: ignore[unresolved-attribute]
        input_ids = torch.tensor([full_token_ids], device=device)  # ty: ignore[invalid-argument-type]
        action_indices = self._action_indices(full_token_ids, action_mask)
        # self.model.critic is untyped/ambiguous to ty for the same reason as self.model.policy
        # above -- `.model` resolves to an unresolved-attribute, and calling the result looks
        # like a call-non-callable. Safe: self.model.critic.model is provably the real
        # transformer backbone (a PreTrainedModel) at runtime, always callable.
        critic_hidden = self.model.critic.model(input_ids=input_ids).last_hidden_state  # ty: ignore[call-non-callable, unresolved-attribute]
        # Same root cause one line up: self.model.critic.score is provably a real nn.Linear
        # value head at runtime, always callable, but ty can't see `.score` through self.model's
        # loose base type either.
        critic_values = self.model.critic.score(critic_hidden).squeeze(-1)[0]  # ty: ignore[call-non-callable, unresolved-attribute]  # [seq_len]
        return critic_values[action_indices]

    def _forward_logprobs_and_values(
        self, full_token_ids: list[int], action_mask: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combined policy+critic forward -- thin wrapper used only by _collect_batch's
        torch.no_grad() "old" pass, where no backward ever runs so there's no graph-retention
        memory concern from computing both together (see _forward_policy_logprobs's docstring for
        why _ppo_update's WITH-grad "new" pass calls the two split methods separately instead).
        """
        return (
            self._forward_policy_logprobs(full_token_ids, action_mask),
            self._forward_critic_values(full_token_ids, action_mask),
        )

    def _collect_batch(self, rows: list[dict]) -> list[dict]:
        """Roll out one episode per row, score it, and compute its frozen GAE inputs.

        Not unit-tested -- calls _rollout_episode (real model/tool-calls) and
        _forward_logprobs_and_values (real forward pass) for each row. Validated by the live
        smoke test (Task 11).
        """
        episodes = []
        raw_advantages: list[list[float]] = []
        device = self.model.policy.device  # ty: ignore[unresolved-attribute]
        for row in rows:
            rollout = self._rollout_episode(row)

            with torch.no_grad():
                old_logprobs, old_values = self._forward_logprobs_and_values(
                    rollout["full_token_ids"], rollout["action_mask"]
                )

            completion = rollout["completion"]
            # Paper-faithful rewards (arXiv:2505.11821v2 Sections 5.2/6.1), NOT the GRPO track's
            # format/F1+EM rewards. Section 6.1 defines PPO-OR's reward as "a binary signal
            # indicating final-answer correctness" with no format term at all; PPO-MR and MT-PPO
            # use the graded R^O (+1.0 correct / +0.2 wrong-but-formatted / -1.0 bad format)
            # plus per-turn R^I. See rewards.py's paper_* functions for why carrying the GRPO
            # rewards into this track made the earlier Phase 7b runs a non-reproduction.
            if self.condition == "ppo":
                outcome_r = paper_binary_outcome_reward(completion, row["golden_answers"])
            else:
                outcome_r = paper_outcome_reward(completion, row["golden_answers"])

            # R^I per intermediate turn. found_gold is the MARGINAL retrieval gain, not the
            # cumulative value: SearchEnv.retrieval_fraction only ever grows, so "ground truth in
            # results" for THIS turn means the fraction rose during it.
            previous_fraction = 0.0
            turn_rewards = []
            for fraction, format_ok, n_search in zip(
                rollout["retrieval_fraction_after_each_turn"],
                rollout["turn_format_ok"],
                rollout["cumulative_searches_after_each_turn"],
                strict=True,
            ):
                turn_rewards.append(
                    paper_turn_reward(
                        found_gold=fraction > previous_fraction,
                        format_ok=format_ok,
                        cumulative_searches=n_search,
                    )
                )
                previous_fraction = fraction

            format_r = 1.0 if _extract_answer(completion) is not None else 0.0
            format_and_outcome_reward = outcome_r
            retrieval_fraction = _final_retrieval_fraction(rollout)

            per_token_rewards = place_paper_rewards(
                num_tokens=len(old_values),
                turn_boundary_token_indices=rollout["turn_boundary_action_indices"],
                turn_rewards=turn_rewards,
                outcome_reward_value=outcome_r,
                condition=self.condition,
            )
            advantages = compute_gae(
                rewards=per_token_rewards,
                values=old_values.tolist(),
                # self.args.gamma and self.args.gae_lambda are real fields on this file's
                # MTPPOConfig (set in build_ppo_config), but Trainer's base type stub only
                # knows self.args as the looser `TrainingArguments` -- same ty-can't-see-
                # through-Trainer's-base-types root cause already noted on self.args.n_max in
                # _rollout_episode above. Safe to ignore here for the same reason on both of
                # these adjacent kwargs.
                gamma=self.args.gamma,  # ty: ignore[unresolved-attribute]
                lam=self.args.gae_lambda,  # ty: ignore[unresolved-attribute]
            )
            # returns deliberately use the RAW advantages: returns are the critic's regression
            # target (the actual discounted return), so normalizing them would train the critic
            # to predict a rescaled quantity that no longer matches the rewards it is supposed
            # to estimate. Only the policy-loss advantages get normalized, below, across the
            # whole batch.
            returns = [a + v for a, v in zip(advantages, old_values.tolist(), strict=True)]

            raw_advantages.append(advantages)
            episodes.append(
                {
                    "full_token_ids": rollout["full_token_ids"],
                    "action_mask": rollout["action_mask"],
                    "old_logprobs": old_logprobs,
                    "old_values": old_values,
                    "advantages": torch.tensor(advantages, device=old_values.device),
                    "returns": torch.tensor(returns, device=old_values.device),
                    "format_and_outcome_reward": format_and_outcome_reward,
                    "format_reward": format_r,
                    "retrieval_fraction": retrieval_fraction,
                    "question": row["question"],
                    "completion": completion,
                    "turn_boundary_action_indices": rollout["turn_boundary_action_indices"],
                }
            )

        # Normalized ACROSS THE WHOLE BATCH, not per episode. Per-episode normalization would
        # force every episode's advantages to zero mean, erasing the between-episode difference
        # that says "this rollout was better than that one" -- which is most of the signal when
        # the reward is dominated by a terminal outcome term.
        normalized = normalize_advantages([a for episode in raw_advantages for a in episode])
        offset = 0
        for episode, raw in zip(episodes, raw_advantages, strict=True):
            episode["advantages"] = torch.tensor(
                normalized[offset : offset + len(raw)],
                device=device,  # ty: ignore[invalid-argument-type]
            )
            offset += len(raw)
        return episodes

    def _ppo_update(self, episodes: list[dict]) -> dict[str, float]:
        """Run args.num_ppo_epochs inner passes over the collected batch, gradient-accumulated
        across all episodes in the batch, one optimizer step per inner epoch. Matches this
        repo's existing train.py precedent of per-episode (batch-of-1) forward/backward passes
        with gradient accumulation, avoiding any padding/attention-mask complexity -- consistent
        with the single-RTX-4090, 0.8B-model memory profile the rest of this repo already
        established.
        """
        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "kl": 0.0,
            "ratio_mean": 0.0,
            "ratio_variance": 0.0,
            "clip_fraction": 0.0,
        }
        num_updates = 0
        # self.args.num_ppo_epochs is a real field on this file's MTPPOConfig, but Trainer's
        # base type stub only knows self.args as the looser `TrainingArguments` -- same
        # ty-can't-see-through-Trainer's-base-types root cause already noted throughout this
        # class (see create_optimizer, _rollout_episode, _collect_batch above).
        for _epoch in range(self.args.num_ppo_epochs):  # ty: ignore[unresolved-attribute]
            # self.optimizer is declared Optional (`Optimizer | None | Unknown`) on Trainer's
            # base class, since Trainer only populates it once create_optimizer() has actually
            # been called -- but train() always assigns a real optimizer
            # (self.optimizer = self.create_optimizer()) before ever invoking _ppo_update, so it
            # is provably non-None here at runtime.
            self.optimizer.zero_grad()  # ty: ignore[unresolved-attribute]
            for episode in episodes:
                # Split into two independent forward+backward passes (policy-only, then
                # critic-only) instead of one combined pass -- a real fix for a real OOM the live
                # smoke test caught (see _forward_policy_logprobs's docstring for the full story
                # and why this is mathematically identical to one combined backward call).
                #
                # Pass 1: policy_loss + kl_beta*kl. Uses episode["old_values"] (already detached,
                # computed under torch.no_grad() in _collect_batch) as a placeholder for
                # new_values -- compute_ppo_loss's value_loss term only touches critic parameters
                # through new_values' grad_fn, so a detached stand-in makes that term's
                # contribution to this backward() call exactly zero, without ever running the
                # critic's forward pass at all in this call.
                new_logprobs = self._forward_policy_logprobs(
                    episode["full_token_ids"], episode["action_mask"]
                )
                ones_mask = torch.ones_like(new_logprobs)
                policy_loss_dict = compute_ppo_loss(
                    new_logprobs=new_logprobs,
                    old_logprobs=episode["old_logprobs"],
                    advantages=episode["advantages"],
                    returns=episode["returns"],
                    new_values=episode["old_values"],
                    action_mask=ones_mask,
                    # self.args.clip_eps, self.args.kl_beta, and self.args.value_loss_coef are
                    # real fields on this file's MTPPOConfig -- same ty-can't-see-through-
                    # Trainer's-base-types root cause as self.args.num_ppo_epochs above; these
                    # three are genuinely back-to-back kwargs sharing that identical cause.
                    clip_eps=self.args.clip_eps,  # ty: ignore[unresolved-attribute]
                    kl_beta=self.args.kl_beta,  # ty: ignore[unresolved-attribute]
                    value_loss_coef=self.args.value_loss_coef,  # ty: ignore[unresolved-attribute]
                )
                (policy_loss_dict["loss"] / len(episodes)).backward()

                # Pass 2: value_loss_coef*value_loss. Uses episode["old_logprobs"] (already
                # detached) as a placeholder for new_logprobs, for the same reason in reverse --
                # ratio=exp(old_logprobs-old_logprobs)=1 identically, contributing zero gradient
                # to the policy, and this call never touches the policy's forward/graph at all.
                new_values = self._forward_critic_values(
                    episode["full_token_ids"], episode["action_mask"]
                )
                value_loss_dict = compute_ppo_loss(
                    new_logprobs=episode["old_logprobs"],
                    old_logprobs=episode["old_logprobs"],
                    advantages=episode["advantages"],
                    returns=episode["returns"],
                    new_values=new_values,
                    action_mask=ones_mask,
                    clip_eps=self.args.clip_eps,  # ty: ignore[unresolved-attribute]
                    kl_beta=self.args.kl_beta,  # ty: ignore[unresolved-attribute]
                    value_loss_coef=self.args.value_loss_coef,  # ty: ignore[unresolved-attribute]
                )
                (value_loss_dict["loss"] / len(episodes)).backward()

                # The real, non-placeholder-contaminated metrics: policy_loss/kl come from pass 1
                # (real new_logprobs), value_loss comes from pass 2 (real new_values); the
                # reported "loss" is reconstructed from those three real components rather than
                # taken from either dict's own "loss" field, since each dict's "loss" scalar value
                # (not its gradient) still numerically includes the OTHER pass's placeholder term.
                real_policy_loss = policy_loss_dict["policy_loss"]
                real_kl = policy_loss_dict["kl"]
                real_value_loss = value_loss_dict["value_loss"]
                # ratio_mean/ratio_variance/clip_fraction come from pass 1 (real new_logprobs)
                # for the same reason real_policy_loss/real_kl do -- pass 2's ratio is always
                # exactly 1.0 (old_logprobs vs itself), which would silently dilute these
                # diagnostics toward "everything looks fine" if averaged in.
                totals["policy_loss"] += real_policy_loss.item()
                totals["kl"] += real_kl.item()
                totals["value_loss"] += real_value_loss.item()
                totals["ratio_mean"] += policy_loss_dict["ratio_mean"].item()
                totals["ratio_variance"] += policy_loss_dict["ratio_variance"].item()
                totals["clip_fraction"] += policy_loss_dict["clip_fraction"].item()
                totals["loss"] += (
                    real_policy_loss
                    + self.args.value_loss_coef * real_value_loss  # ty: ignore[unresolved-attribute]
                    + self.args.kl_beta * real_kl  # ty: ignore[unresolved-attribute]
                ).item()
                num_updates += 1

            # Standard PPO practice, and absent until now. Gradients here are accumulated across
            # every episode in the batch with no bound on their norm, so a single outlier episode
            # (this repo's action-token counts reach 3,000+ against a median of ~150) can drive
            # one update far outside the region the clipped surrogate is meant to keep the policy
            # in. That is a textbook cause of exactly the failure observed on 2026-07-25: ppo's
            # format compliance fell from 0.542 at step 399 to 0.120 by step 500.
            # max_grad_norm comes from TrainingArguments (default 1.0), so no new config field.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)  # ty: ignore[unresolved-attribute]
            # self.optimizer is Optional on Trainer's base class for the same reason noted
            # above this method's self.optimizer.zero_grad() call; still provably a real
            # Optimizer at this point in the loop.
            self.optimizer.step()  # ty: ignore[unresolved-attribute]
        return {key: value / num_updates for key, value in totals.items()}

    def _save_policy_and_critic(self, output_dir: str) -> None:
        """Save policy and critic to output_dir/policy and output_dir/critic respectively.

        Deliberately does NOT use the inherited Trainer.save_model()/_save() -- a real bug the
        live smoke test caught. Trainer's generic _save() calls safetensors.torch.save_file() on
        self.model.state_dict() directly, which has no idea self.model (_PolicyAndCritic) is
        wrapping two separate PreTrainedModels with their own tied-weight metadata. Qwen3.5 ties
        `lm_head.weight` to `model.embed_tokens.weight` (a real, standard weight-tying setup, not
        a bug in the model), and safetensors refuses to save two keys that share the same
        underlying storage unless the tied-weight-aware skip/reconstitute logic in
        PreTrainedModel.save_pretrained() runs first -- confirmed directly: the untouched
        Trainer.save_model() path raised exactly this "tensors share memory" RuntimeError.
        Calling each real PreTrainedModel's own .save_pretrained() (which already knows how to
        skip/reconstitute tied weights) instead of Trainer's generic path fixes this.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # self.model.policy/.critic are real PreTrainedModel instances at runtime (see
        # create_optimizer's comment for why ty can't see this through Trainer's loose self.model
        # type), each with a genuine .save_pretrained() that handles tied weights correctly.
        self.model.policy.save_pretrained(output_path / "policy")  # ty: ignore[call-non-callable, unresolved-attribute]
        self.model.critic.save_pretrained(output_path / "critic")  # ty: ignore[call-non-callable, unresolved-attribute]

    def _save_full_checkpoint(self, output_dir: str) -> None:
        """Save everything needed to resume training exactly where it left off: policy+critic
        weights (via _save_policy_and_critic, which already correctly handles Qwen3.5's tied
        lm_head/embed_tokens weights), plus optimizer/RNG/step state via transformers.Trainer's
        own inherited primitives -- the same ones TRL's own PPOTrainer relies on (PPOConfig
        subclasses TrainingArguments directly and exposes resume_from_checkpoint/save_steps,
        confirmed against trl==1.9.0). Not unit-tested: _save_optimizer_and_scheduler and
        _save_rng_state are inherited, untouched Trainer methods that write real optimizer
        state/RNG snapshots to disk -- validated by Task 7's live smoke test instead. See
        docs/superpowers/specs/2026-07-23-phase-7b-full-ppo-runs-design.md.

        Ends by calling transformers' own rotate_checkpoints (the same disk-retention function
        Trainer's built-in _save_checkpoint uses) to delete older checkpoint-N directories beyond
        self.args.save_total_limit -- a full checkpoint here (weights+optimizer+RNG) is far larger
        than the weights-only ~3GB this repo has measured so far (AdamW's two per-parameter
        moment buffers, same bf16 dtype as the params, roughly double each model's own memory),
        so leaving every checkpoint on disk for a 500-step run would exhaust this machine's disk
        -- not a hypothetical, confirmed by direct calculation before this method was written this
        way. rotate_checkpoints always protects the most-recent checkpoint (needed for resume),
        so this never deletes the one train()'s own resume path would need.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._save_policy_and_critic(output_dir)
        # Live smoke testing saw a 6/7 CUDA OOM rate this session, with most failures landing at
        # or near a checkpoint-save step boundary -- but the raw tracebacks never actually showed
        # the save call itself in the stack (see docs/phase-7b-full-ppo-runs.md's "honest caveat"
        # for the specific step-offset counterexamples), so treat the boundary correlation as a
        # plausible, re-test-confirmed contributing factor, not a fully nailed-down root cause.
        # What IS confirmed via direct transformers==5.13.0 source inspection: in this repo's
        # single-GPU config, Trainer._save_optimizer_and_scheduler calls
        # torch.save(self.optimizer.state_dict(), ...) directly on GPU-resident optimizer tensors
        # (AdamW's exp_avg/exp_avg_sq) with no CPU transfer first -- a documented PyTorch/HF
        # community OOM trigger when GPU memory is already this tight (~92-94% utilized here).
        # Freeing cached (allocated-but-unused) blocks before that save gives it room without
        # touching any hyperparameter, gradient, or model output; re-tested once, it took mt_ppo's
        # regression check from 4/4 failures to 1/1 success -- real signal, but only one data
        # point so far.
        torch.cuda.empty_cache()
        self._save_optimizer_and_scheduler(output_dir)
        self._save_rng_state(output_dir)
        self.state.save_to_json(str(Path(output_dir) / "trainer_state.json"))
        rotate_checkpoints(
            output_dir=str(Path(output_dir).parent),
            save_total_limit=self.args.save_total_limit,
            use_mtime=False,
        )

    def _load_full_checkpoint(self, checkpoint_dir: str) -> None:
        """Inverse of _save_full_checkpoint -- reloads policy+critic weights, optimizer/RNG
        state, and global_step, so train() can resume exactly where a prior run left off. Not
        unit-tested: loads real weights and real optimizer/RNG state from disk -- validated by
        Task 7's live smoke test instead, same convention as build_policy_and_critic.

        Loads weights INTO the existing self.model.policy/self.model.critic objects via
        load_state_dict, rather than replacing those objects outright -- a real bug a code
        reviewer caught: train() calls self.optimizer = self.create_optimizer() (which binds
        AdamW's param groups to the *specific tensor objects* of
        self.model.policy.parameters()/self.model.critic.parameters() at that moment) BEFORE this
        method runs. Reassigning self.model.policy/.critic to brand-new model objects loaded from
        the checkpoint would silently orphan the optimizer's param-group references -- forward/
        backward would then populate .grad on the NEW model's parameters, but optimizer.step()
        would only ever update the old, now-untethered tensors nothing else reads from. No
        exception, just a resumed run that silently stops learning. load_state_dict instead
        copies the checkpoint's tensor values into the existing parameter tensors in place, so the
        optimizer's existing references stay valid -- no reordering relative to create_optimizer()
        is needed, and no rebuilding of gradient_checkpointing_enable() either (already enabled on
        the existing self.model.policy/.critic by build_policy_and_critic).
        """
        policy_checkpoint = AutoModelForCausalLM.from_pretrained(
            str(Path(checkpoint_dir) / "policy"), dtype=torch.bfloat16
        )
        critic_checkpoint = AutoModelForSequenceClassification.from_pretrained(
            str(Path(checkpoint_dir) / "critic"), num_labels=1, dtype=torch.bfloat16
        )
        # self.model.policy/.critic are real PreTrainedModel instances at runtime (see
        # create_optimizer's comment for why ty can't see this through Trainer's loose self.model
        # type), each with a genuine .load_state_dict() that copies tensor values in place
        # (handling any cross-device copy automatically), leaving the parameter objects
        # themselves -- and every other reference to them, including the optimizer's -- intact.
        self.model.policy.load_state_dict(policy_checkpoint.state_dict())  # ty: ignore[unresolved-attribute]
        self.model.critic.load_state_dict(critic_checkpoint.state_dict())  # ty: ignore[unresolved-attribute]
        self._load_optimizer_and_scheduler(checkpoint_dir)
        self._load_rng_state(checkpoint_dir)
        self.state = TrainerState.load_from_json(str(Path(checkpoint_dir) / "trainer_state.json"))

    # train(self) intentionally narrows Trainer.train's full signature
    # (resume_from_checkpoint, trial, ignore_keys_for_eval -> TrainOutput) down to train(self)
    # -> None: PPO's collect-then-multi-epoch-update outer loop replaces Trainer's generic
    # dataloader/training_step machinery those parameters exist to support (see this method's
    # own docstring), and no call site in this repo (build_ppo_trainer, the live smoke test)
    # ever needs to pass them.
    def train(self) -> None:  # ty: ignore[invalid-method-override]
        """Overrides Trainer.train() entirely: PPO's collect-then-multi-epoch-update structure
        doesn't fit Trainer's default single-pass-per-batch loop, so this owns the whole outer
        loop instead of relying on get_train_dataloader()/training_step().

        Writes two diagnostic artifacts under self.args.output_dir, so a run can be inspected
        after the fact without rerunning it:
          - train_log.jsonl: one JSON line per step with every metric plus a per-episode
            breakdown (question, reward, retrieval_fraction, action-token count) -- enough detail
            to diagnose a specific step or a specific episode's reward after training has already
            finished, not just an aggregate curve.
          - sample_completions.log: one full example transcript appended every
            _SAMPLE_COMPLETION_INTERVAL steps (plain text), mirroring this repo's existing
            train.py convention of log_completions=True for GRPO -- lets a human spot-check real
            model output during a long run without re-running anything.
        Also prints a one-line progress summary to stdout each step (step/max_steps, key metrics,
        elapsed and estimated-remaining wall-clock) so a long run's progress is visible without
        having trackio's dashboard open.

        set_seed(self.args.seed) is called here explicitly because this override replaces
        Trainer.train() entirely -- the base class's own seeding call never runs, so without this,
        two runs with the same --seed would silently stop reproducing the same rollouts (sampling
        in _rollout_episode uses torch's global RNG).
        """
        set_seed(self.args.seed)
        self.optimizer = self.create_optimizer()
        if self.args.resume_from_checkpoint:
            self._load_full_checkpoint(self.args.resume_from_checkpoint)
        collapse_monitor = CollapseMonitor()
        # self.train_dataset is typed by Trainer's base class as
        # `torch.utils.data.Dataset | datasets.arrow_dataset.Dataset | None`, and
        # torch.utils.data.Dataset's stub doesn't declare __iter__, so ty can't confirm it's
        # Iterable -- but build_ppo_trainer (Task 11) always constructs this trainer with a real
        # datasets.Dataset, which genuinely is iterable at runtime. The same untyped-Dataset root
        # cause is why ty can't confirm the resulting list is `list[dict]` either, needed by
        # _row_cycle_from_step's signature below.
        rows = list(self.train_dataset)  # ty: ignore[invalid-argument-type]
        row_cycle = _row_cycle_from_step(
            rows,  # ty: ignore[invalid-argument-type]
            self.state.global_step,
            self.args.num_rollouts_per_step,  # ty: ignore[unresolved-attribute]
        )
        trackio.init(project=self.args.project, name=self.args.run_name)

        # self.args.output_dir is typed `str | None` on TrainingArguments (None only if a
        # caller explicitly passed output_dir=None), but build_ppo_config always sets a real
        # output_dir string, so this is never actually None at runtime.
        output_dir = Path(self.args.output_dir)  # ty: ignore[invalid-argument-type]
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train_log.jsonl"
        sample_completions_path = output_dir / "sample_completions.log"

        run_start = time.monotonic()
        for step in range(self.state.global_step, self.args.max_steps):
            step_start = time.monotonic()
            # Reset here (not once at the top of train()) so max_memory_allocated below reflects
            # THIS step's own peak, not a running max across the whole training run -- added
            # after real live diagnosis needed exception-message archaeology (parsing "X GiB
            # allocated" out of a caught OOM's str()) to answer "is memory actually growing over
            # time, or just persistently tight?" Real per-step numbers make that a direct read
            # instead of a reconstruction.
            torch.cuda.reset_peak_memory_stats()
            # self.args.num_rollouts_per_step is a real field on this file's MTPPOConfig, but
            # Trainer's base type stub only knows self.args as the looser `TrainingArguments` --
            # same ty-can't-see-through-Trainer's-base-types root cause already noted throughout
            # this class (see create_optimizer, _rollout_episode, _collect_batch, _ppo_update
            # above).
            batch_rows = [
                next(row_cycle)
                for _ in range(self.args.num_rollouts_per_step)  # ty: ignore[unresolved-attribute]
            ]
            # Found live during Phase 7b's first full-scale run: a genuinely long episode (one
            # observed at 2302 action tokens, ~2-20x this run's typical length) can leave the CUDA
            # allocator fragmented enough after its own forward/backward passes that a later
            # step's ordinary-length episode then OOMs -- deterministically reproducible on every
            # --resume-from-checkpoint replay of the same data order, since the same oversized
            # episode recurs at the same step every time (confirmed live: three consecutive
            # resume attempts died at the identical point with byte-identical OOM diagnostics).
            # Researched real-world practice before choosing a fix (not guessed): the standard
            # PyTorch pattern for a per-batch OOM is catch, empty_cache, and move to the next
            # batch rather than crash the whole process (e.g. PyTorch Forums/Databricks
            # community threads on this exact error), and veRL (the framework Search-R1 -- the
            # lineage this repo's own design already builds on -- is itself built on) ships a
            # real `filter_overlong_prompts` feature with the same underlying philosophy: don't
            # let one outlier-length example take down an entire training step. This step
            # therefore catches a CUDA OOM at the per-step level, logs it as a real (not hidden)
            # skipped step, and continues -- rather than crashing the whole process and forcing a
            # resume (which pays a full model-reload cost and, per the finding above, can walk
            # right back into the identical failure on a deterministic replay anyway).
            #
            # Known, accepted limitation: _ppo_update runs args.num_ppo_epochs inner epochs, each
            # ending in a real optimizer.step() -- if the OOM happens partway through (e.g. epoch
            # 3 of 4), the epochs that already completed have already applied real weight updates
            # that are NOT rolled back just because this step is logged as "skipped". This is a
            # deliberate simplicity tradeoff, not an oversight: a fully atomic per-step rollback
            # would need to snapshot and restore optimizer/model state before every step, adding
            # real overhead to every single step to guard against an occasional partial failure.
            failed_at = None
            try:
                failed_at = "_collect_batch"
                episodes = self._collect_batch(batch_rows)
                failed_at = "_ppo_update"
                update_metrics = self._ppo_update(episodes)
                failed_at = None
            except RuntimeError as exc:
                # Catches both torch's own torch.OutOfMemoryError (a RuntimeError subclass) and
                # Triton kernels' plain `RuntimeError: Triton Error [CUDA]: out of memory` (a
                # real, separately-observed failure site in this repo's own live testing, not the
                # same exception class as torch's) -- re-raise anything whose message doesn't
                # actually say "out of memory", so a genuinely different bug still surfaces as a
                # visible crash instead of being silently absorbed here.
                if "out of memory" not in str(exc).lower():
                    raise
                gpu_stats = {
                    "gpu_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "gpu_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                }
                torch.cuda.empty_cache()
                skip_record = {
                    "step": step,
                    "skipped": True,
                    "failed_at": failed_at,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "questions_attempted": [row["question"] for row in batch_rows],
                    **gpu_stats,
                }
                with log_path.open("a") as log_file:
                    log_file.write(json.dumps(skip_record) + "\n")
                print(
                    f"step {step + 1}/{self.args.max_steps} | SKIPPED (CUDA OOM in {failed_at}) "
                    f"-- allocated={gpu_stats['gpu_allocated_gb']:.2f}GB "
                    f"peak={gpu_stats['gpu_max_allocated_gb']:.2f}GB -- see train_log.jsonl for "
                    f"the questions attempted",
                    flush=True,
                )
                # Skips must reach trackio too, or the dashboard silently under-reports how much
                # of a run actually happened -- and an asymmetric skip rate between arms is the
                # confound that invalidated Phase 7b (mt_ppo took 431 real updates to ppo's 499).
                trackio.log({"step": step, "skipped": 1.0, **gpu_stats})
                self.state.global_step = step + 1
                continue
            gpu_stats = {
                "gpu_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "gpu_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
            torch.cuda.empty_cache()

            mean_reward = sum(e["format_and_outcome_reward"] for e in episodes) / len(episodes)
            mean_retrieval_fraction = sum(e["retrieval_fraction"] for e in episodes) / len(episodes)
            total_elapsed = time.monotonic() - run_start
            steps_remaining = self.args.max_steps - (step + 1)
            eta_seconds = (total_elapsed / (step + 1)) * steps_remaining

            metrics = {
                "step": step,
                "loss": update_metrics["loss"],
                "policy_loss": update_metrics["policy_loss"],
                "value_loss": update_metrics["value_loss"],
                "kl": update_metrics["kl"],
                "ratio_mean": update_metrics["ratio_mean"],
                "ratio_variance": update_metrics["ratio_variance"],
                "clip_fraction": update_metrics["clip_fraction"],
                "reward": mean_reward,
                "retrieval_fraction": mean_retrieval_fraction,
                # The four metrics this phase is actually judged on. All were previously
                # computed (or logged to train_log.jsonl) but never reached trackio, so the
                # dashboard could not answer the questions that matter most:
                #   format_compliance -- the anchor. The paper reports 0.998 on HotpotQA; this
                #     repo's pre-fix runs sat near 0.48. It was already being computed for
                #     CollapseMonitor and then discarded.
                #   ended_on_tool_rate -- must stay 0.0. Nonzero means the answer-turn defect
                #     has regressed and episodes are being penalised for a turn they never got.
                #   mean_tool_turns -- the search penalty's observable effect; uncontrolled
                #     growth here is what preceded the earlier OOM cascade.
                #   skipped -- 0.0/1.0 per step, so the dashboard shows the skip RATE. Skips
                #     differing between arms is the confound that invalidated Phase 7b.
                "format_compliance": sum(e["format_reward"] for e in episodes) / len(episodes),
                "ended_on_tool_rate": sum(
                    1.0
                    for e in episodes
                    if e["completion"] and e["completion"][-1].get("role") == "tool"
                )
                / len(episodes),
                "mean_tool_turns": sum(len(e["turn_boundary_action_indices"]) for e in episodes)
                / len(episodes),
                "skipped": 0.0,
                **gpu_stats,
            }
            trackio.log(metrics)

            mean_format_reward = sum(e["format_reward"] for e in episodes) / len(episodes)
            for alert in collapse_monitor.check(
                step=step,
                loss=metrics["loss"],
                mean_reward=mean_reward,
                mean_format_reward=mean_format_reward,
            ):
                trackio.alert(
                    title=alert.title,
                    text=alert.text,
                    level=trackio.AlertLevel.ERROR
                    if alert.level == "ERROR"
                    else trackio.AlertLevel.WARN,
                )
                if alert.should_stop:
                    # Set global_step before saving -- this branch runs before the loop's own
                    # `self.state.global_step = step + 1` line below, so without this, the saved
                    # trainer_state.json would record a stale (one-step-behind) global_step,
                    # off by one from this checkpoint directory's own `checkpoint-{step + 1}`
                    # name.
                    self.state.global_step = step + 1
                    self._save_full_checkpoint(f"{self.args.output_dir}/checkpoint-{step + 1}")
                    print(f"Stopping training early at step {step + 1}: {alert.title}", flush=True)
                    return

            log_record = dict(metrics)
            log_record["step_elapsed_seconds"] = time.monotonic() - step_start
            log_record["total_elapsed_seconds"] = total_elapsed
            log_record["eta_seconds"] = eta_seconds
            log_record["episodes"] = [
                {
                    "question": episode["question"],
                    "format_and_outcome_reward": episode["format_and_outcome_reward"],
                    "retrieval_fraction": episode["retrieval_fraction"],
                    "num_action_tokens": len(episode["old_values"]),
                    # Turn accounting, logged because reconstructing it from the 1-in-10
                    # sampled transcripts was not enough to measure the 2026-07-26 answer-turn
                    # defect -- where an episode that used its whole tool budget ended on a
                    # role:tool message with no generation left in which to answer, and was then
                    # penalised for not answering. ended_on_tool must now be False for every
                    # episode; a nonzero rate means that regression is back.
                    # The format-compliance anchor (paper: 0.998 on HotpotQA), logged
                    # explicitly because it is NOT derivable from format_and_outcome_reward
                    # for the ppo arm: PPO-OR's reward is binary correctness with no format
                    # term, so 0.0 means "wrong answer" OR "no answer" indistinguishably.
                    "format_ok": episode["format_reward"],
                    "num_tool_turns": len(episode["turn_boundary_action_indices"]),
                    "ended_on_tool": episode["completion"][-1].get("role") == "tool"
                    if episode["completion"]
                    else False,
                }
                for episode in episodes
            ]
            with log_path.open("a") as log_file:
                log_file.write(json.dumps(log_record) + "\n")

            if step == 0 or (step + 1) % _SAMPLE_COMPLETION_INTERVAL == 0:
                sample_episode = episodes[0]
                with sample_completions_path.open("a") as sample_file:
                    sample_file.write(
                        f"=== step {step + 1} | reward="
                        f"{sample_episode['format_and_outcome_reward']:.3f} | retrieval_fraction="
                        f"{sample_episode['retrieval_fraction']:.3f} ===\n"
                    )
                    sample_file.write(f"question: {sample_episode['question']}\n")
                    for message in sample_episode["completion"]:
                        sample_file.write(f"[{message.get('role')}] {message.get('content')}\n")
                    sample_file.write("\n")

            print(
                f"step {step + 1}/{self.args.max_steps} | loss={metrics['loss']:.4f} "
                f"reward={mean_reward:.3f} retrieval_fraction={mean_retrieval_fraction:.3f} "
                f"| elapsed={total_elapsed:.0f}s eta={eta_seconds:.0f}s",
                flush=True,
            )

            self.state.global_step = step + 1

            if (step + 1) % self.args.save_steps == 0:
                self._save_full_checkpoint(f"{self.args.output_dir}/checkpoint-{step + 1}")

        self._save_full_checkpoint(f"{self.args.output_dir}/checkpoint-{self.args.max_steps}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse train_ppo.py's CLI arguments. Mirrors train.py's _parse_args pattern from Phase 4 --
    the bare invocation (just --condition) is a tiny smoke-test-scale run; full runs (Phase 7b)
    must explicitly override --train-size/--max-steps/--num-rollouts-per-step.
    """
    parser = argparse.ArgumentParser(
        description="Train multi-turn PPO/MT-PPO (see CLAUDE.md and docs/phase-7-mt-ppo.md)."
    )
    parser.add_argument("--condition", required=True, choices=["ppo", "ppo_mr", "mt_ppo"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--num-rollouts-per-step", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a checkpoint-N directory to resume from, or 'auto' to resume from the "
        "latest checkpoint under this run's output_dir.",
    )
    return parser.parse_args(argv)


def normalize_advantages(advantages: list[float]) -> list[float]:
    """Rescale advantages to zero mean and unit variance.

    Standard PPO practice, and not optional here: this repo takes policy_lr=1e-6 directly from
    the paper's Appendix C.1.3, but the paper runs on veRL, which normalizes advantages by
    default. A learning rate calibrated for unit-scale advantages applied to raw ones -- here
    typically ~0.1-0.3, because most episodes score the -0.1 format floor and gamma=lambda=1
    makes every token's advantage R_total - V(t) -- yields an effective step size several times
    smaller than the paper's. Copying the hyperparameter without its precondition is what made
    the two incomparable.

    Rescales only; it never reorders. The relative ranking of actions IS the learning signal, so
    a transform that changed it would change the algorithm rather than its conditioning.

    Zero variance returns all-zero rather than dividing by ~0. That state is real, not
    hypothetical -- it is exactly what the 2026-07-24 flat-reward bug produced for 177 steps --
    and NaN advantages would poison every subsequent update rather than simply carrying no
    signal, which is the honest representation of a batch in which nothing differed.
    """
    count = len(advantages)
    mean = sum(advantages) / count
    variance = sum((a - mean) ** 2 for a in advantages) / count
    std = math.sqrt(variance)
    if std < 1e-8:
        return [0.0] * count
    return [(a - mean) / std for a in advantages]


def gather_action_logprobs(lm_head, hidden_states, next_tokens, chunk_size=LOGPROB_CHUNK_SIZE):
    """Log-prob of each next_token, applying lm_head in gradient-checkpointed chunks.

    Purely a memory optimisation -- mathematically this is just
    `log_softmax(lm_head(hidden_states)).gather(...)`, since each position's log_softmax is taken
    over its own vocab row and is independent of every other position.

    Agreement is numerical, not bit-exact, and the distinction is worth stating precisely: a
    [chunk, hidden] x [hidden, vocab] matmul can select a different GEMM kernel than the
    full-height one, so lm_head's output shifts by roughly one ULP (measured: 4.77e-07 in
    float32; exactly 0 when one chunk covers every position; ~1e-3 relative in bf16, the same
    caveat Liger-Kernel documents for its own chunked kernels). tests/unit/test_train_ppo.py
    pins values and gradients to a tolerance just above that, at chunk sizes that do and do not
    divide the position count.

    Why it is needed (measured on this repo's own RTX 4090, not assumed). Phase 7 already cut
    the vocab-sized tensor from O(seq_len * vocab) to O(num_action_tokens * vocab) by applying
    lm_head only at action positions. But with a 248,320-token vocab that residual term is still
    the only part of the update that GROWS WITH EPISODE LENGTH -- and mt_ppo's episodes triple in
    length during training (386 -> ~1190 action tokens) as its turn reward teaches the policy to
    search more. Measured peak for this path alone:

        num_action_tokens:   1200      3000
        one pass:           2.32 GB   5.02 GB
        chunked (256):      0.92 GB   0.94 GB      <- flat, at the lm_head-weight floor

    That growth is what drove mt_ppo's 13.8% CUDA-OOM step-skip rate (69 of 500 steps, every one
    of them inside _ppo_update, never in rollout, all peaking at 23.6-24.2 GB against a ~23.5 GB
    card) while ppo -- whose episodes stay short -- skipped 0.2%. An asymmetric skip rate between
    the two conditions is a methodology problem, not just lost throughput: it silently gives the
    two arms of the comparison different numbers of real gradient updates (431 vs 499).

    torch.utils.checkpoint drops each chunk's [chunk_size, vocab] logits after the forward pass
    and recomputes them during backward, trading one extra lm_head matmul per chunk for a peak
    that no longer depends on episode length.

    Liger-Kernel's LigerFusedLinearPPOLoss was evaluated first and deliberately not used: it
    computes the whole PPO objective (it takes advantages/old_per_token_logps/epsilon/beta and a
    loss_type selector), so adopting it would replace this repo's paper-faithful compute_ppo_loss
    and its clip-fraction/ratio diagnostics -- a different algorithm, not a memory optimisation.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    def head(states, targets):
        log_probs = torch.log_softmax(lm_head(states), dim=-1)
        return log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)

    chunks = [
        checkpoint(
            head,
            hidden_states[start : start + chunk_size],
            next_tokens[start : start + chunk_size],
            use_reentrant=False,
        )
        for start in range(0, hidden_states.shape[0], chunk_size)
    ]
    return torch.cat(chunks)


def validate_tool_call_arguments(tool, call_args: dict) -> str | None:
    """Return an error message if call_args is not a valid call to `tool`, else None.

    Checks names/arity via inspect binding AND argument types against the tool's annotations.
    The type half is what a plain bind() misses, and it is load-bearing: Python does not enforce
    annotations, so `search(query=123)` binds perfectly happily and only fails later, out at the
    retrieval server, as `HTTP 422 Unprocessable Content` -- which took down a 500-step run at
    step 166 on 2026-07-24. The policy had emitted a non-string query, which the chat template's
    json value parser (allow_non_json) faithfully passed through as an int.

    A wrongly-TYPED argument is the same class of model mistake as a wrongly-NAMED one -- exactly
    the kind of thing the episode's own format/outcome reward should teach the policy out of --
    so it belongs here, returned as feedback to the model, rather than propagating as a crash.
    That keeps _rollout_episode's deliberate split intact: everything raised from inside search()
    itself still surfaces as a real infra failure and is never swallowed.

    Only simple type annotations are enforced. A parameter annotated with anything else (or not
    annotated at all) is left alone rather than guessed at -- a validator that rejects calls the
    server would have served fine is worse than no validator.
    """
    try:
        bound = inspect.signature(tool).bind(**call_args)
    except TypeError as exc:
        return f"invalid arguments for {tool.__name__}(): {exc}"

    parameters = inspect.signature(tool).parameters
    for name, value in bound.arguments.items():
        annotation = parameters[name].annotation
        if not isinstance(annotation, type) or annotation is inspect.Parameter.empty:
            continue
        if not isinstance(value, annotation):
            return (
                f"invalid arguments for {tool.__name__}(): argument '{name}' must be "
                f"{annotation.__name__}, got {type(value).__name__} ({value!r})"
            )
    return None


def resolve_auto_checkpoint(output_dir: str | None) -> str:
    """Resolve `--resume-from-checkpoint auto` to the latest checkpoint under output_dir.

    Raises rather than silently starting from scratch: `auto` means "continue the run I already
    have", so a missing checkpoint is a mistake worth stopping on, not a reason to quietly begin
    a fresh 500-step run under the same output_dir (which would interleave a new run's logs and
    checkpoints with the old ones).

    An output_dir that does not exist at all (or is None, which TrainingArguments permits) is the
    same "nothing to resume" case, but transformers' get_last_checkpoint raises FileNotFoundError
    from os.listdir before it can return None -- so a first launch with `auto` died on an opaque
    traceback from library internals instead of this module's own message. Normalised here.
    """
    last_checkpoint = (
        get_last_checkpoint(output_dir) if output_dir and os.path.isdir(output_dir) else None
    )
    if last_checkpoint is None:
        raise ValueError(
            f"--resume-from-checkpoint auto: no checkpoint found under {output_dir}. "
            "Omit the flag entirely to start a fresh run."
        )
    return last_checkpoint


def build_ppo_trainer(
    condition: Condition,
    train_size: int | None,
    config: MTPPOConfig,
) -> MTPPOTrainer:
    """Composition root: real policy+critic, real SearchEnv (hits the live retrieval server),
    real data. Not unit-tested -- this is exactly the integration surface the live smoke test
    validates, same principle as train.py's build_trainer.
    """
    from transformers import AutoTokenizer

    from turn_level_rewards import data

    model = build_policy_and_critic(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = data.load_train_dataset(n=train_size, seed=config.seed)
    return MTPPOTrainer(
        condition=condition,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=config,
    )


def main() -> None:
    args = _parse_args()
    config = build_ppo_config(
        condition=args.condition,
        seed=args.seed,
        max_steps=args.max_steps,
        num_rollouts_per_step=args.num_rollouts_per_step,
        save_steps=args.save_steps,
    )
    config.run_name = (
        f"{args.condition}-{args.max_steps}steps-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint == "auto":
        resume_from_checkpoint = resolve_auto_checkpoint(config.output_dir)
    config.resume_from_checkpoint = resume_from_checkpoint
    trainer = build_ppo_trainer(args.condition, args.train_size, config)
    trainer.train()


if __name__ == "__main__":
    main()
