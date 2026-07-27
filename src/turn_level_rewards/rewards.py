"""Reward functions for GRPO training (see CLAUDE.md's "Reward design" section).

turn_reward implements turn-level credit assignment via reward density -- GRPO scores one
scalar per completed trajectory, so there is no per-timestep value function here, and this is
not a literal per-step RL change.
"""

import re
from collections.abc import Callable
from typing import Any, Literal

from turn_level_rewards.metrics import exact_match, f1_score

Completion = list[dict[str, Any]]
LogMetric = Callable[[str, float], None]

TURN_REWARD_SCALE = 0.4

_ANSWER_RE = re.compile(r"<answer>(.+?)</answer>", re.DOTALL)


def _noop_log_metric(name: str, value: float) -> None:
    return None


def _extract_answer(completion: Completion) -> str | None:
    """Return the final answer text if the completion ends in one well-formed <answer> tag.

    Well-formed means: the last message has no unresolved tool_calls, and its content contains
    exactly one non-empty <answer>...</answer> pair.
    """
    if not completion:
        return None
    last = completion[-1]
    if last.get("tool_calls"):
        return None
    content = last.get("content")
    if not isinstance(content, str):
        return None
    matches = _ANSWER_RE.findall(content)
    if len(matches) != 1:
        return None
    answer = matches[0].strip()
    return answer or None


def format_reward(
    completions: list[Completion], log_metric: LogMetric = _noop_log_metric, **kwargs: Any
) -> list[float]:
    """+0.1 for a well-formed single <answer> tag in the final message, -0.1 otherwise.

    Logs format_compliance_rate (1.0/0.0 per completion) -- see CLAUDE.md's "Experiment
    tracking" section.
    """
    rewards = []
    for completion in completions:
        compliant = _extract_answer(completion) is not None
        rewards.append(0.1 if compliant else -0.1)
        log_metric("format_compliance_rate", 1.0 if compliant else 0.0)
    return rewards


def outcome_reward(
    completions: list[Completion],
    golden_answers: list[list[str]],
    log_metric: LogMetric = _noop_log_metric,
    **kwargs: Any,
) -> list[float]:
    """SQuAD F1 + 0.5 exact-match bonus, maxed over each row's golden_answers list.

    Logs the winning answer's raw exact_match and f1 (unblended) -- see CLAUDE.md's "Experiment
    tracking" section.
    """
    rewards = []
    for completion, answers in zip(completions, golden_answers, strict=True):
        prediction = _extract_answer(completion) or ""
        scored = []
        for answer in answers:
            f1 = f1_score(prediction, answer)
            em = exact_match(prediction, answer)
            scored.append((f1 + (0.5 if em else 0.0), f1, em))
        best_reward, best_f1, best_em = max(scored, key=lambda item: item[0])
        rewards.append(best_reward)
        log_metric("exact_match", float(best_em))
        log_metric("f1", best_f1)
    return rewards


def turn_reward(
    environments: list[Any], log_metric: LogMetric = _noop_log_metric, **kwargs: Any
) -> list[float]:
    """0.4 * retrieval_fraction -- dense signal for surfacing gold supporting-fact passages.

    Logs the unscaled retrieval_fraction -- see CLAUDE.md's "Experiment tracking" section.
    """
    rewards = []
    for environment in environments:
        rewards.append(TURN_REWARD_SCALE * environment.retrieval_fraction)
        log_metric("retrieval_fraction", environment.retrieval_fraction)
    return rewards


_LENGTH_PENALTY_CAP = 0.2
_LENGTH_PENALTY_TARGET_CHARS = 2000


def _generated_length(completion: Completion) -> int:
    """Total character length of the model's own generated text (assistant messages only).

    Excludes tool-response content deliberately -- that text is injected by the environment
    (retrieved documents), not written by the model, so it shouldn't count against it.
    """
    return sum(
        len(str(message.get("content") or ""))
        for message in completion
        if message.get("role") == "assistant"
    )


def length_penalty(
    completions: list[Completion], log_metric: LogMetric = _noop_log_metric, **kwargs: Any
) -> list[float]:
    """Small penalty for generated text beyond a target length.

    Added to counter a real, measured drift: Phase 6's symmetric re-run showed completion length
    roughly doubling over training in both conditions, decoupled from correctness (same-length
    rollout groups scored anywhere from 0 to max reward) -- see
    docs/phase-6-evaluation-comparison.md's Handoff notes. Nothing in format_reward/outcome_reward
    penalizes verbosity, so the drift is free under the existing reward; this adds the missing
    pressure. No penalty below _LENGTH_PENALTY_TARGET_CHARS (matching the healthy early-training
    baseline observed in that same run); scales linearly above it, capped at
    -_LENGTH_PENALTY_CAP so it can never dominate outcome_reward (max 1.5) or turn_reward (max 0.4).
    """
    rewards = []
    for completion in completions:
        length = _generated_length(completion)
        excess = max(0, length - _LENGTH_PENALTY_TARGET_CHARS)
        penalty = -_LENGTH_PENALTY_CAP * min(1.0, excess / _LENGTH_PENALTY_TARGET_CHARS)
        rewards.append(penalty)
        log_metric("completion_length", float(length))
    return rewards


_SEARCH_COUNT_PENALTY_COEF = 0.1


def _search_call_count(completion: Completion) -> int:
    """Number of `search` tool calls issued across the completion's assistant turns."""
    count = 0
    for message in completion:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            if tool_call.get("function", {}).get("name") == "search":
                count += 1
    return count


def search_count_penalty(
    completions: list[Completion], log_metric: LogMetric = _noop_log_metric, **kwargs: Any
) -> list[float]:
    """-0.1 * n_search -- the source paper's MT-PPO search-count penalty (Section 5.2/6.1,
    lambda_s=0.1 "by default"), borrowed here for GRPO.

    Confirmed by direct fetch: the paper's own GRPO-OR/GRPO-MR case study (Appendix E) has no
    search-count penalty at all -- this term only exists in their PPO/MT-PPO reward design. So
    this is NOT a paper reproduction of the GRPO methodology; it's a deliberate experiment
    borrowing their PPO-context mechanism and coefficient as the best available grounded
    starting point, paired with dropping the prompt's "at most 2 searches" instruction (see
    data.py's search_cap_in_prompt) -- the reward now does the job the prompt used to do. See
    docs/phase-6-evaluation-comparison.md's Handoff notes for the full reasoning.
    """
    rewards = []
    for completion in completions:
        n_search = _search_call_count(completion)
        rewards.append(-_SEARCH_COUNT_PENALTY_COEF * n_search)
        log_metric("search_call_count", float(n_search))
    return rewards


def get_reward_funcs(
    condition: Literal["outcome_only", "turn_level"],
    penalize_length: bool = False,
    penalize_search_count: bool = False,
) -> list[Any]:
    """Return the reward function list for a training condition (CLAUDE.md's Reward design).

    penalize_length and penalize_search_count are orthogonal toggles (not new condition values)
    so either composes with either condition without duplicating the outcome_only/turn_level
    branch -- see docs/phase-6-evaluation-comparison.md's Handoff notes for why each was added.
    """
    if condition == "outcome_only":
        funcs = [format_reward, outcome_reward]
    elif condition == "turn_level":
        funcs = [format_reward, outcome_reward, turn_reward]
    else:
        raise ValueError(f"Unknown condition: {condition!r}")
    if penalize_length:
        funcs.append(length_penalty)
    if penalize_search_count:
        funcs.append(search_count_penalty)
    return funcs


# ---------------------------------------------------------------------------
# Paper-faithful PPO rewards (arXiv:2505.11821v2 Section 5.2 / Section 6.1).
#
# Used ONLY by the PPO track (train_ppo.py). The GRPO track deliberately keeps the
# format/F1+EM rewards above -- CLAUDE.md documents why that deviation is justified for GRPO
# (its group-relative advantage needs within-group variance, which binary EM does not provide).
# That justification does NOT transfer to PPO, which has a real per-token critic, and carrying
# the GRPO rewards into the PPO track silently is what made the Phase 7b runs a non-reproduction:
# the paper reports MT-PPO at EM 0.453 / format 0.998 on HotpotQA, we measured 0.123 / ~0.48.
#
# Paper's components, quoted from Section 5.2:
#   R^I (per intermediate turn): retrieval existence +0.3 if ground truth in results else 0;
#                                format +0.1 if correct, -0.2 if incorrect;
#                                search penalty -lambda_s * n_search (cumulative), lambda_s=0.1
#   R^O (final turn):            correct answer + correct format  -> +1.0
#                                incorrect answer + correct format -> +0.2
#                                incorrect format                  -> -1.0
# ---------------------------------------------------------------------------

PAPER_RETRIEVAL_BONUS = 0.3
PAPER_TURN_FORMAT_BONUS = 0.1
PAPER_TURN_FORMAT_PENALTY = -0.2
PAPER_SEARCH_PENALTY = 0.1  # lambda_s, the paper's default
PAPER_OUTCOME_CORRECT = 1.0
PAPER_OUTCOME_WRONG_BUT_FORMATTED = 0.2
PAPER_OUTCOME_BAD_FORMAT = -1.0


def paper_outcome_reward(completion: Completion, golden_answers: list[str]) -> float:
    """R^O: +1.0 correct answer with correct format, +0.2 wrong answer with correct format,
    -1.0 incorrect format.

    "Correct" is exact match after SQuAD normalization, maxed over golden_answers -- the paper
    scores answer correctness as a binary signal, not partial credit.

    The -1.0 format penalty is ten times this repo's GRPO-track penalty (-0.1) relative to a
    comparable outcome scale, and that gap is the leading suspect for the format-compliance gap
    between our PPO runs (~0.48) and the paper's (0.998).
    """
    prediction = _extract_answer(completion)
    if prediction is None:
        return PAPER_OUTCOME_BAD_FORMAT
    correct = any(exact_match(prediction, answer) for answer in golden_answers)
    return PAPER_OUTCOME_CORRECT if correct else PAPER_OUTCOME_WRONG_BUT_FORMATTED


def paper_binary_outcome_reward(completion: Completion, golden_answers: list[str]) -> float:
    """PPO-OR's reward: "a binary signal indicating final-answer correctness" (Section 6.1).

    Deliberately carries NO format term -- Section 6.1 defines PPO-OR as "trained with only
    outcome rewards". An unanswerable/unparseable completion scores 0.0, the same as a wrong
    answer, so PPO-OR gets no gradient distinguishing "answered wrongly" from "did not answer".
    That is the baseline's documented weakness, not an implementation shortcut.
    """
    prediction = _extract_answer(completion)
    if prediction is None:
        return 0.0
    return 1.0 if any(exact_match(prediction, answer) for answer in golden_answers) else 0.0


def paper_turn_reward(
    found_gold: bool,
    format_ok: bool,
    cumulative_searches: int,
    search_penalty_weight: float = PAPER_SEARCH_PENALTY,
) -> float:
    """R^I for one intermediate turn: retrieval existence + format + cumulative search penalty.

    The search penalty is cumulative, not per-turn: the paper defines n_search as "the cumulative
    number of search invocations from the first turn up to the current turn", so later turns are
    penalised progressively harder. This is the term whose absence the paper explicitly warns
    about -- "removing this reward term (lambda_s = 0.0) leads to unstable training and degenerate
    behaviors, such as uncontrolled search usage or non-convergent rollouts" -- and its absence is
    exactly what this repo observed: episodes growing from 386 to ~1190 action tokens as the
    policy learned to search without bound, burning its whole turn budget without answering.
    """
    reward = PAPER_RETRIEVAL_BONUS if found_gold else 0.0
    reward += PAPER_TURN_FORMAT_BONUS if format_ok else PAPER_TURN_FORMAT_PENALTY
    return reward - search_penalty_weight * cumulative_searches
