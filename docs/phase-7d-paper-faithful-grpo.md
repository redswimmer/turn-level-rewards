# Phase 7d: Paper-faithful GRPO — completing the five-arm Table 2 comparison

**Status**: not started. **Prerequisite: Phase 7c complete** (`docs/phase-7c-paper-faithful-ppo.md`).
**Must land before Phase 8 (the LLM judge).**

## Why this phase exists

The paper's Table 2 compares GRPO and PPO variants against each other. That comparison is valid
because **the paper holds the reward design fixed across algorithms**. Section 6.1's definitions
are word-for-word identical except for the algorithm name:

> **PPO-OR**: *"vanilla PPO trained with only outcome rewards, where the trajectory-level reward
> is a binary signal indicating final-answer correctness."*
> **GRPO-OR**: *"vanilla GRPO trained with only outcome rewards, where the trajectory-level reward
> is a binary signal indicating final-answer correctness."*

> **PPO-MR** / **GRPO-MR**: *"...merged intermediate and outcome rewards, where the
> trajectory-level reward combines intermediate rewards (retrieval correctness) and outcome
> rewards (answer correctness and format correctness)."*

**This repo currently breaks that property.** After Phase 7c the PPO track uses the paper's
rewards (binary or ±1.0 outcome, per-turn format, λs = 0.1 search penalty), while the GRPO track
(Phases 5-6) still uses this repo's own design: ±0.1 format, F1 + 0.5·EM outcome, 0.4 ×
retrieval_fraction, no search penalty.

That GRPO deviation was deliberate and is justified in `CLAUDE.md` — GRPO's group-relative
advantage needs within-group variance, which a binary EM reward does not provide, and at this
repo's ~300-600 step scale a group of all-zero rewards yields no gradient at all. **That reasoning
is still sound for the GRPO-only comparison.** The problem is only that it makes the two tracks
non-comparable, so the five arms cannot be placed in one table.

Without this phase, the repo can claim:
- PPO-OR vs PPO-MR vs MT-PPO (Phase 7c) ✅
- GRPO-OR vs GRPO-MR (Phases 5-6, existing rewards) ✅
- **PPO vs GRPO** ❌ — conflates algorithm with reward function

## Goal

Re-run **GRPO-OR** and **GRPO-MR** under the paper's reward design, so all five arms share one
reward and the cross-algorithm comparison becomes legitimate.

## Design questions to resolve before implementing

These are real decisions, not formalities. Resolve them explicitly and record the reasoning.

1. **Does binary EM actually kill GRPO's gradient at this scale?** This is the stated reason for
   the original deviation, and it is testable rather than assumed: run a short GRPO-OR job with
   the paper's binary reward and watch `frac_reward_zero_std`. If it pins at 1.0, the deviation
   was correct and this phase should report that as a finding — a real, measured reason this
   repo's scale cannot reproduce the paper's GRPO setup — rather than forcing it.
   `TrackioAlertCallback` already watches exactly this signal.
2. **Keep the existing GRPO runs.** Do not delete or overwrite Phases 5-6's results. They remain
   the valid GRPO-OR-vs-GRPO-MR comparison under this repo's rewards; this phase adds arms, it
   does not replace them.
3. **Reward wiring.** `rewards.py` already has the paper-faithful components
   (`paper_binary_outcome_reward`, `paper_outcome_reward`, `paper_turn_reward`) built in Phase 7c.
   The work is a paper-faithful variant of `get_reward_funcs` plus a CLI flag — the components
   themselves are done and unit-tested.
4. **The search penalty in a GRPO context.** Phase 6 already ran a `search_count_penalty`
   experiment borrowing this mechanism and found it *hurt* both GRPO conditions (see
   `docs/phase-6-evaluation-comparison.md`). That is directly relevant prior evidence and should
   inform expectations here — it may mean paper-faithful GRPO underperforms this repo's GRPO,
   which would itself be a legitimate, reportable result.
5. **Seed and step budget** — match Phase 7c (seed 42) for comparability, and decide whether to
   match Phases 5-6's 600 steps or Phase 7c's 500.

## Exit criteria

- GRPO-OR and GRPO-MR trained under the paper's rewards, evaluated on the held-out set with the
  same eval size used in Phase 7c
- A single five-arm table: PPO-OR, PPO-MR, MT-PPO, GRPO-OR, GRPO-MR — all under one reward design
- Findings recorded either way, including the negative result if binary EM does stall GRPO
- This doc's Handoff notes filled in; `CLAUDE.md`'s roadmap Status updated

## Note on scope

MT-GRPO remains out of scope (see `CLAUDE.md`'s Goal section) — it needs per-turn advantage
estimation that TRL's `GRPOTrainer` exposes no hook for. The five arms above are what this repo
can legitimately produce.
