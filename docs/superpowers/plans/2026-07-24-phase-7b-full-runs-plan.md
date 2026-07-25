# Phase 7b Full Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the paper's real Table 2 headline comparison (`PPO-OR` vs `MT-PPO`) on this
repo's pipeline: run both conditions to the full 500-step budget, evaluate both on the complete
held-out set, write up the comparison, and ship matplotlib charts + a README Results section.

**Architecture:** Two real, long-running training executions (Tasks 1-2, actively monitored
through any crash/resume cycles using the infra already built and verified), a held-out evaluation
pass (Task 3), a new `scripts/compare_ppo_runs.py` mirroring the GRPO track's existing
`scripts/compare_runs.py` shape (Task 4), a written comparison verdict (Task 5), and a chart +
README pass using the `dataviz` skill (Task 6).

**Tech Stack:** Python 3.13, `trackio` CLI, `matplotlib`, `pytest`.

## Global Constraints

- Both training runs use: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--max-steps 500`,
  `--num-rollouts-per-step 2`, `--save-steps 15`, `--seed 42` — see
  `docs/superpowers/specs/2026-07-24-phase-7b-full-runs-design.md` for why `save_steps` is 15,
  not the infra plan's default of 50.
- Runs are sequential, never parallel — this GPU cannot hold two training processes at once
  (each already uses ~92-94% of the RTX 4090's 24GB alone).
- A crash during either run is expected, not exceptional (this session's live smoke test measured
  a ~86% short-run OOM rate). Resume via `--resume-from-checkpoint auto`; do not restart from
  scratch. Only escalate if a failure is NOT a plain `torch.OutOfMemoryError`/Triton OOM (i.e. a
  genuinely new failure mode).
- Held-out evaluation uses the full `hotpot_qa` distractor validation split: `--eval-size 7405`
  (matching the GRPO track's own full-scale eval, not a smoke-scale slice).
- `evaluate_ppo.py` logs EM/F1/retrieval_fraction only from the final held-out pass — unlike the
  GRPO track, `train_ppo.py`'s training loop never logs per-step EM/F1 to trackio (only
  `reward`/`retrieval_fraction`/loss-family metrics; confirmed by inspection — `outcome_reward` is
  called without a `log_metric` callback in `_collect_batch`). Do not assume a training-time
  EM/F1 curve exists for the PPO track; the comparison's headline EM/F1 numbers come only from
  `evaluate_ppo.py`'s output, not a per-step trend.
- Test location: `tests/unit/` only, no new test tiers.
- Use the `dataviz` skill for all new/retrofitted chart work — not ad hoc matplotlib defaults.

---

### Task 1: Launch and complete `ppo`'s full 500-step run

**Files:**
- None (no code changes) — produces `outputs/ppo/checkpoint-500` and its trackio run.

**Interfaces:**
- Consumes: the existing `train_ppo.py` CLI, already real-hardware-verified in the infra plan.
- Produces: a completed 500-step `ppo` training run, `outputs/ppo/checkpoint-500` (plus rotated
  intermediate checkpoints, at most 3 kept at a time), a full `train_log.jsonl`.

- [ ] **Step 1: Confirm a clean starting state**

Run: `ls outputs/ppo/ 2>&1` — if any prior scratch checkpoints/logs exist from earlier smoke
testing, clear them first (`rm -rf outputs/ppo`) so this run's `train_log.jsonl` isn't
contaminated by unrelated earlier data (the same issue Task 7 of the infra plan already hit and
fixed once).

Run: `curl -s -X POST http://localhost:8000/retrieve -H 'Content-Type: application/json' -d '{"queries": ["test"], "topk": 1, "return_scores": false}'` to confirm the retrieval server is up.

- [ ] **Step 2: Launch the run**

Run (in the background, since this takes ~1.4 hours): `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m turn_level_rewards.train_ppo --condition ppo --max-steps 500 --num-rollouts-per-step 2 --save-steps 15 --seed 42 2>&1 | tee /tmp/phase7b_ppo_full_run.log`

- [ ] **Step 3: Monitor to completion, resuming through any crashes**

Watch the log for progress (`step N/500`) and failure signatures (`OutOfMemoryError`,
`Traceback`, `Triton Error`). On a crash:
1. Confirm it's a plain CUDA/Triton OOM (expected, per this phase's Global Constraints) — if it's
   something else, stop and report it as a real finding rather than retrying blind.
2. Resume: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m turn_level_rewards.train_ppo --condition ppo --max-steps 500 --num-rollouts-per-step 2 --save-steps 15 --seed 42 --resume-from-checkpoint auto 2>&1 | tee -a /tmp/phase7b_ppo_full_run.log`
3. Repeat until `step 500/500` is reached.

Expected: eventually reaches `step 500/500`, `outputs/ppo/checkpoint-500` exists with `policy/`,
`critic/`, `optimizer.pt`, `rng_state.pth`, `trainer_state.json`.

- [ ] **Step 4: Record what actually happened**

Note in your task report: total wall-clock (including any crash/resume overhead), how many
crashes occurred, whether `CollapseMonitor` fired any alerts (and if so, whether the run still
finished at reward/format-compliance levels worth using, or whether an earlier checkpoint should
be preferred for evaluation per the paper's own "last checkpoint before collapse" methodology —
this is a real per-run judgment call to make now, not assumed).

- [ ] **Step 5: Commit any documentation of this run's real outcome**

If anything genuinely notable happened (new failure mode, a collapse, a much-longer-than-expected
wall-clock), add a brief note to `docs/phase-7b-full-ppo-runs.md`'s Handoff notes section and
commit it. If nothing beyond ordinary, already-documented OOM/resume behavior occurred, no doc
commit is needed for this task — say so in your report instead of manufacturing a commit.

---

### Task 2: Launch and complete `mt_ppo`'s full 500-step run

**Files:**
- None (no code changes) — produces `outputs/mt_ppo/checkpoint-500` and its trackio run.

**Interfaces:**
- Same as Task 1, condition `mt_ppo` instead of `ppo`.

- [ ] **Step 1-5: identical to Task 1**, substituting `--condition mt_ppo`,
`outputs/mt_ppo/checkpoint-500`, and `/tmp/phase7b_mtppo_full_run.log` throughout. Run this task
only after Task 1's `ppo` run has fully finished (sequential, not parallel — see Global
Constraints).

---

### Task 3: Full held-out evaluation for both conditions

**Files:**
- Produces: `results/ppo_eval_metrics.json`, `results/mt_ppo_eval_metrics.json`.

**Interfaces:**
- Consumes: `evaluate_ppo.py`'s existing CLI (real-hardware-verified in the infra plan's Task 7),
  and whichever checkpoint directories Tasks 1-2 determined are the right ones to evaluate
  (`checkpoint-500`, unless either run's own Step 4 flagged a collapse and recommended an earlier
  checkpoint instead).

- [ ] **Step 1: Evaluate `ppo`**

Run: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m turn_level_rewards.evaluate_ppo --condition ppo --checkpoint-dir outputs/ppo/checkpoint-500 --eval-size 7405 --output results/ppo_eval_metrics.json`

(Substitute the checkpoint directory Task 1 actually recommended, if different from
`checkpoint-500`.)

Expected: completes without error (may take a while at `eval-size 7405` — this is a real,
full-scale run, not a smoke test), `results/ppo_eval_metrics.json` contains `exact_match`, `f1`,
`retrieval_fraction`, `num_examples` (should be 7405) with finite values.

- [ ] **Step 2: Evaluate `mt_ppo`**

Run the same command with `--condition mt_ppo --checkpoint-dir outputs/mt_ppo/checkpoint-500
--output results/mt_ppo_eval_metrics.json` (substituting Task 2's recommended checkpoint if
different).

- [ ] **Step 3: Commit the results files**

```bash
git add results/ppo_eval_metrics.json results/mt_ppo_eval_metrics.json
git commit -m "Add full held-out evaluation results for ppo/mt_ppo"
```

---

### Task 4: `scripts/compare_ppo_runs.py`

**Files:**
- Create: `scripts/compare_ppo_runs.py`
- Test: `tests/unit/test_compare_ppo_runs.py`

**Interfaces:**
- Consumes: `results/ppo_eval_metrics.json`/`results/mt_ppo_eval_metrics.json` (Task 3),
  trackio's `turn-level-rewards-ppo` project (per `MTPPOConfig.project`), run names `ppo`/`mt_ppo`
  (per `build_ppo_config`'s `run_name=condition`).
- Produces: a pure `build_ppo_comparison_data(fetch_metric, run_names, eval_metrics, project) ->
  dict` function (unit-tested with a fake `fetch_metric`, mirroring
  `scripts/compare_runs.py`'s own `build_comparison_data` pattern exactly) and plotting functions
  guided by the `dataviz` skill, plus a CLI `main()`.

- [ ] **Step 1: Load the `dataviz` skill before writing any chart code**

This is a hard requirement from the Global Constraints, not optional.

- [ ] **Step 2: Write the failing test for the pure data-prep function**

Mirror `tests/unit/`'s existing style for `scripts/compare_runs.py` (find and read its test file
if one exists, or design one following the same fake-`fetch_metric`-injection pattern shown in
`scripts/compare_runs.py`'s own `build_comparison_data` docstring). Assert the returned structure
correctly combines fetched trackio metrics (`reward`, `retrieval_fraction` — NOT `exact_match`/`f1`,
per the Global Constraints note on what's actually logged during PPO training) with the loaded
`eval_metrics` dicts.

- [ ] **Step 3: Run the test to verify it fails**

- [ ] **Step 4: Implement `build_ppo_comparison_data`**

Mirror `fetch_trackio_metric`/`load_eval_metrics`/`build_comparison_data` from
`scripts/compare_runs.py` as closely as the different metric set allows — same
dependency-inversion shape (injectable fetch function), same project-name parameter.

- [ ] **Step 5: Run the test to verify it passes**

- [ ] **Step 6: Design and implement the chart set, per the `dataviz` skill's guidance**

At minimum, mirror what `scripts/compare_runs.py` already produces for the GRPO track (training
curves, a final comparison bar/summary chart, retrieval-fraction trend) adapted to the PPO
track's actually-available metrics (reward/retrieval_fraction training curves +
EM/F1/retrieval_fraction final comparison from the held-out eval, per the Global Constraints
note). Follow the `dataviz` skill's palette/style guidance rather than reusing
`scripts/compare_runs.py`'s exact color choices verbatim, if this task's Task 6 sibling (README
retrofit) will unify both under one new system — coordinate with that task's scope rather than
producing two independent style decisions.

- [ ] **Step 7: Add the CLI (`_parse_args`/`main`)**

Mirror `scripts/compare_runs.py`'s CLI shape.

- [ ] **Step 8: Run the full unit test suite**

Run: `uv run pytest tests/unit/ -v`

- [ ] **Step 9: Commit**

```bash
git add scripts/compare_ppo_runs.py tests/unit/test_compare_ppo_runs.py
git commit -m "Add scripts/compare_ppo_runs.py: ppo vs mt_ppo comparison data + charts"
```

---

### Task 5: Comparison verdict write-up

**Files:**
- Modify: `docs/phase-7b-full-ppo-runs.md` (Handoff notes section)

**Interfaces:**
- Consumes: Task 3's eval JSON files, Task 4's comparison charts, `docs/phase-6-evaluation-comparison.md`'s "is more training needed?" checklist (read it — reuse its exact structure, don't reinvent).

- [ ] **Step 1: Apply Phase 6's validation checklist to the real numbers**

Read `docs/phase-6-evaluation-comparison.md`'s validation checklist section in full. Apply each
criterion to this phase's actual `ppo`/`mt_ppo` EM/F1/retrieval_fraction numbers before writing
any verdict.

- [ ] **Step 2: Write the verdict into `docs/phase-7b-full-ppo-runs.md`'s Handoff notes**

State plainly: real advantage for one condition, no meaningful difference, or inconclusive (per
however many of Phase 6's checklist criteria trigger) — the same standard Phase 6 held itself to,
not a softer one. Include the actual numbers, not just a qualitative claim.

- [ ] **Step 3: Update the phase doc's exit criteria checkboxes**

Mark the now-satisfied boxes in `docs/phase-7b-full-ppo-runs.md`'s "Tasks"/"Exit criteria"
sections (full training runs, held-out evaluation, comparison verdict).

- [ ] **Step 4: Commit**

```bash
git add docs/phase-7b-full-ppo-runs.md
git commit -m "docs: record Phase 7b ppo vs mt_ppo comparison verdict"
```

---

### Task 6: README Results section + chart retrofit

**Files:**
- Modify: `README.md`
- Modify: existing GRPO chart-generation code (in `scripts/compare_runs.py` or wherever the
  currently-embedded GRPO PNGs were produced) — retrofit to the same visual system as Task 4's
  new PPO charts, per this phase's design decision to unify both rather than leave two chart
  styles in one README.

**Interfaces:**
- Consumes: Task 4's PPO charts, Task 5's verdict, the existing GRPO charts under `results/`.

- [ ] **Step 1: Load the `dataviz` skill again if not already loaded this session**

- [ ] **Step 2: Decide and implement the retrofit's actual scope**

Read the existing GRPO chart-generation code in `scripts/compare_runs.py` first. Decide whether
"retrofit" means regenerating the existing GRPO PNGs with the new shared style module/palette
(preferred, if Task 4 produced a reusable style module), or a lighter-touch alignment (e.g. shared
color constants only). Do not assume the heavier interpretation without checking what Task 4
actually built — this is a real scope decision to make now, informed by Task 4's actual code, not
pre-decided here.

- [ ] **Step 3: Add the PPO Results section to `README.md`**

Follow the existing GRPO section's structure (self-contained, grounded in the paper's own
numbers, no internal doc-path citations — check `README.md`'s current GRPO section for the exact
tone/format to match). Rename the current "GRPO Results (PPO coming soon)" heading once PPO
content is added (decide on the exact new heading text — e.g. splitting into two clearly labeled
subsections, or one combined "Results" section with GRPO/PPO subsections).

- [ ] **Step 4: Regenerate/verify all embedded chart images actually render**

Confirm every `![...](results/*.png)` reference in `README.md` (both pre-existing GRPO ones and
new PPO ones) points at a real, current file.

- [ ] **Step 5: Commit**

```bash
git add README.md results/
git commit -m "README: add ppo/mt_ppo Results section, unify chart visual system"
```
