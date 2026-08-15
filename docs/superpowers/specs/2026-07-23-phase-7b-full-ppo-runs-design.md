# Phase 7b design: full PPO/MT-PPO runs — repeatability, auditability, trackio tracking

## Goal

Design how Phase 7b (`docs/phase-7b-full-ppo-runs.md`) actually runs full `ppo`/`mt_ppo` training on
this repo's single-GPU hardware, in a way that is:

- **Tracked in trackio** — both conditions logged so results are inspectable in the same dashboard
  used by the GRPO track (`turn-level-rewards-ppo` project, already wired in `train_ppo.py`).
- **Auditable** — a run's outcome (or failure) can be diagnosed after the fact from logged
  artifacts alone, without rerunning anything.
- **Repeatable** — a run can survive an infra failure (OOM) and resume, and the same seed
  reproduces the same rollouts.

This doc supplements `docs/phase-7b-full-ppo-runs.md` (which already lists the phase's task
checklist and exit criteria) with the specific mechanisms for these three cross-cutting
requirements, resolved by (a) reading `transformers.Trainer`'s and `trl==1.9.0`'s actual
checkpoint/resume source and docs directly, not assumed, and (b) re-reading the paper's own
PPO-specific sections (arXiv:2505.11821v2, Appendix C.1.3 and Section 6.1) directly for this
brainstorm, not relying solely on CLAUDE.md's existing summary.

## Confirmed facts grounding this design

**TRL still has no multi-turn PPO support, reconfirmed against `trl==1.9.0`** (the version
explicitly checked for this design, not just carried over from the installed `1.7.1` or an earlier
check): `trl.experimental.ppo.PPOTrainer` takes a fixed `train_dataset` of prompts plus separate
`reward_model`/`value_model` and does single-turn (query → response → score) rollouts. No
`environment_factory`, no `tools`, no multi-turn loop — Phase 7's hand-built `MTPPOTrainer` remains
the only path for this repo's tool-calling agent. This is not new, but it is freshly re-verified.

**`PPOConfig` (trl 1.9.0) subclasses `TrainingArguments` directly** — `resume_from_checkpoint`,
`save_steps`, `save_strategy`, `seed`, `data_seed`, `full_determinism`,
`restore_callback_states_from_checkpoint` are all plain `TrainingArguments` passthroughs. TRL's own
PPOTrainer does not invent its own checkpoint format; it relies on the same
`transformers.Trainer` primitives (`_save_optimizer_and_scheduler`, `_save_rng_state`,
`TrainerState.save_to_json`/`load_from_json`) confirmed directly by reading
`transformers/trainer.py` (installed `transformers==5.13.0`). This repo's `MTPPOTrainer` should use
the same primitives for its own resume path — not a custom-invented format — since `Trainer.train()`
already has to be fully overridden anyway (that's the one piece TRL genuinely lacks: multi-turn
tool-calling), but the checkpoint/resume *mechanism* underneath doesn't need to be reinvented.

**The paper's own PPO hyperparameters (Appendix C.1.3, quoted verbatim)**: "Training is performed
for 500 steps over 4 epochs, with warm-up ratios of 0.285 and 0.015 for the policy and critic
models, respectively. The total batch size is 512, with a mini-batch size of 256 and a micro-batch
size of 64 for policy updates, and a micro-batch size of 8 for critic updates. We adopt GAE with
λ=1 and γ=1. The learning rates of the policy and critic models are set to 1e−6 and 1e−5,
respectively." — run on 8×H100. This repo's `MTPPOConfig` already matches the non-batch-size
hyperparameters (`policy_lr=1e-6`, `critic_lr=1e-5`, `gamma=1.0`, `gae_lambda=1.0`,
`num_ppo_epochs=4`, `clip_eps=0.2`). The batch-size figures (512/256/64/8) are not reproducible on
one RTX 4090 — this is a real, deliberate, documented scale deviation (see below), the PPO-track
analog of the GRPO track's already-documented "2 searches vs. the paper's 1" and "F1 vs. binary EM"
deviations.

**The paper's own baselines are unstable (Section 6.1, quoted verbatim)**: "Since PPO baselines
often crash, we evaluate them using either the final checkpoint or the last checkpoint prior to
collapse." This is not a hypothetical concern for this design — it is the paper's own documented
experience with PPO-OR specifically, and it directly shapes the checkpointing requirements below.
Collapse risk is plausibly *higher* in this repo's smaller-batch setting (noisier gradient
estimates from a much smaller batch than the paper's 512), so the mitigation matters more here, not
less.

**Eq. 9 turn-level reward placement (Section 4, quoted verbatim)**: "r_t = R^O if t is the last
token of the entire trajectory; R^I if t is the last token of the intermediate turn; 0 otherwise."
Already correctly implemented by `place_turn_rewards` in `train_ppo.py` and confirmed working via
Phase 7's live smoke test (real nonzero `R^I` at a real gold-title hit). No change needed here —
recorded for completeness since this design touches the same reward-placement code path's
surrounding infrastructure.

## Decisions

### 1. Scale: hardware is the *only* intended deviation from the paper

Every non-batch-size hyperparameter stays exactly as the paper specifies and as `MTPPOConfig`
already has it: `policy_lr=1e-6`, `critic_lr=1e-5`, `gamma=1.0`, `gae_lambda=1.0`,
`num_ppo_epochs=4`, `clip_eps=0.2`. The target step count is the paper's own **500 steps** — not a
number to casually shrink. The one thing this repo's single RTX 4090 cannot match is the paper's
batch scale (512 total / 256 mini-batch / 64+8 micro-batch on 8×H100), so `num_rollouts_per_step`
is the deliberate, hardware-driven deviation — the PPO-track analog of the GRPO track's already-
documented deviations (2 searches vs. the paper's 1, F1 vs. binary EM).

Rather than guessing `num_rollouts_per_step` (and confirming 500 steps is actually reachable in
real wall-clock time on this hardware), the first task of Phase 7b's implementation plan is a
short real timing probe: run ~10-20 steps at a candidate `num_rollouts_per_step`, measure real
wall-clock per step and observed OOM rate. If 500 steps is reachable in reasonable wall-clock time
at some `num_rollouts_per_step`, that's what runs — no further deviation. **Only if the probe
shows 500 steps is genuinely wall-clock-infeasible** does the step count itself get reduced, and if
that happens it gets written up with the same explicit "hardware forced this, here's the tradeoff"
treatment as the batch-size deviation — never silently shrunk. Both the batch-size deviation (real,
expected) and any step-count deviation (only if forced) get documented in this phase's README
section and Handoff notes, the same way CLAUDE.md documents the GRPO track's deviations.

### 2. Repeatability: two separate mechanisms for two separate failure modes

**Infra failure (OOM) → checkpoint-resume.** At each checkpoint interval (chosen from the probe's
measured per-step overhead, so checkpoint I/O stays a small fraction of run time):
- `_save_policy_and_critic` (already correct — handles Qwen3.5's tied `lm_head`/`embed_tokens`
  weights, unlike generic `Trainer._save()`).
- `self._save_optimizer_and_scheduler(checkpoint_dir)` and `self._save_rng_state(checkpoint_dir)`
  (inherited from `Trainer`, unmodified — captures Python/NumPy/CPU/CUDA RNG, which matters because
  `_rollout_episode`'s `do_sample=True` generation draws from torch's global RNG).
- `self.state.save_to_json(checkpoint_dir / "trainer_state.json")` (captures `global_step`).

`train()` gains a `resume_from_checkpoint` path: reload weights, call
`self._load_optimizer_and_scheduler(dir)` and `self._load_rng_state(dir)`, read `global_step` back
from `trainer_state.json`, and resume the step loop from there. `train_log.jsonl` and
`sample_completions.log` are already opened in append mode, so the audit trail stays continuous
across a restart. `row_cycle = itertools.cycle(rows)` is fast-forwarded by
`global_step * num_rollouts_per_step` draws on resume (deterministic given a fixed row list, so
this reproduces the same data order a from-scratch run would have seen).

**Policy collapse → visibility, not auto-rollback.** The paper's own quoted methodology ("the last
checkpoint prior to collapse") reads as a post-hoc human judgment call from inspecting training
curves — not a described automated mechanism. Building automatic rollback would be inventing
something the paper doesn't describe, which is a fidelity risk, not an improvement. Instead: reuse
this repo's existing `TrackioAlertCallback` pattern from `train.py` (same repo-precedented
approach already used for GRPO) — fire a trackio alert when reward/format-compliance craters and
stays low for N consecutive steps, or when loss/KL goes NaN/Inf. This makes collapse visible live;
checkpoint frequency (from decision 2's infra-failure mechanism) ensures a "last good" checkpoint
is actually available to select afterward, exactly mirroring the paper's own approach.

### 3. Diagnosability: extend existing trackio + on-disk audit trail

Already in place (`train_ppo.py`, unchanged by this design): `trackio.log(metrics)` per step,
`train_log.jsonl` (per-step aggregate + per-episode question/reward/retrieval_fraction/token-count
breakdown), `sample_completions.log` every 10 steps.

New: add `ratio` mean/variance and clip fraction (both policy- and value-side) to both
`trackio.log` and `train_log.jsonl` — TRL's own PPOTrainer logs `val/ratio`, `val/ratio_var`,
`policy/clipfrac_avg`, `val/clipfrac_avg` (`docs/trl/v1.9.0/ppo_trainer`'s "Explanation of the
logged metrics" section) as its standard PPO-specific stability diagnostics, on top of the
loss/policy_loss/value_loss/kl/reward/retrieval_fraction this repo already tracks. These are
computed directly from values already available in `compute_ppo_loss` (the `ratio` tensor, the
clip comparison) — no new forward passes needed, just additional aggregation.

### 4. Held-out evaluation path for MTPPOTrainer checkpoints

New `evaluate_ppo.py` (parallel to Phase 6's `evaluate.py`, but for the PPO track): loads a saved
`policy`/`critic` checkpoint directory, reuses `MTPPOTrainer._rollout_episode`/`_collect_batch`
directly under `torch.no_grad()` (skipping `_ppo_update` entirely) against the held-out
`hotpot_qa` validation set, aggregates EM/F1/retrieval_fraction the same way Phase 6's `evaluate.py`
does. Takes `--checkpoint-dir` as a required argument (not hardcoded to "final") specifically so it
can evaluate whichever checkpoint gets judged best from the trackio curves — final, or
last-before-collapse — matching the paper's own stated evaluation methodology rather than assuming
every run finishes cleanly.

### 5. Comparison write-up + charts

Mirrors Phase 6's structure: EM/F1/retrieval_fraction for `ppo` vs `mt_ppo`, reusing Phase 6's
"is more training needed?" validation checklist before treating any result as final. New matplotlib
visuals via the `dataviz` skill. Per this brainstorm's resolution: the new chart system retrofits
*both* the existing GRPO-OR/GRPO-MR charts and the new PPO-OR/MT-PPO comparison, for one consistent
visual system across the whole README rather than two different styles on the same page.

## Out of scope for this design

- The actual full-run budget numbers (exact `max_steps`, `num_rollouts_per_step`, checkpoint
  interval) — these come from the probe in decision 1, not this design doc; recorded in Phase 7b's
  Handoff notes once measured.
- Any change to Eq. 9 reward placement, GAE, or the PPO-clip loss itself — already correct per
  Phase 7's smoke test; this design only touches the surrounding run/checkpoint/logging
  infrastructure.
- MT-GRPO (still explicitly out of scope per CLAUDE.md) and Phase 8's LLM-judge work (separate,
  later phase).

## Handoff to implementation plan

The next step is `superpowers:writing-plans` to turn this into a concrete task sequence, ordered
so the wall-clock probe (decision 1) runs first and informs the exact numbers used everywhere else
(checkpoint interval, full-run budget).
