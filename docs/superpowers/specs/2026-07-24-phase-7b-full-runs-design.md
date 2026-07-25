# Phase 7b design: full PPO/MT-PPO training runs, evaluation, comparison, charts

## Goal

Execute Phase 7b's actual headline deliverable — the paper's real Table 2 comparison (`PPO-OR`
vs `MT-PPO`), reproduced on this repo's HotpotQA/wiki-18 pipeline — now that the infrastructure
plan (`docs/superpowers/plans/2026-07-23-phase-7b-ppo-infra-plan.md`, all 7 tasks complete,
merged) provides checkpoint-resume, diagnostics, collapse-visibility, and a working
`evaluate_ppo.py`. This is the PPO-track analog of Phases 5+6 combined for the GRPO track: full
runs, then evaluation/comparison/write-up.

## Prerequisites (confirmed already in place)

- `MTPPOTrainer` works for both conditions, live-smoke-tested (Phase 7).
- Checkpoint-resume, `save_total_limit` rotation, PPO diagnostics, `CollapseMonitor`, and
  `evaluate_ppo.py` all real-hardware-verified (Phase 7b infra plan, this session).
- A real, documented ~86% short-run OOM rate this session, mitigated (not eliminated) by a
  `torch.cuda.empty_cache()` fix re-verified once (1/1 success vs. 4/4 failures pre-fix) — see
  `docs/phase-7b-full-ppo-runs.md`'s "Infrastructure verified" section for the full, honestly
  caveated account.
- Retrieval server running and stable.

## Decisions

### 1. Run configuration

Both conditions, launched with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m turn_level_rewards.train_ppo \
    --condition {ppo,mt_ppo} --max-steps 500 --num-rollouts-per-step 2 --save-steps 15 --seed 42
```

`--max-steps 500` and `--num-rollouts-per-step 2` are Task 1's probe results (paper-matched step
count; hardware-driven batch size, the one deliberate deviation from the paper's 8×H100/batch=512
setup). `--save-steps 15` is a **new decision** for this phase, tightened from the infra plan's
default of 50: this session's live smoke test found a materially higher OOM rate (~86%) than
Task 1's probe (~33%), and even with the `empty_cache()` mitigation, treating crash-recovery
granularity conservatively is the right call for a run this long. At `save_steps=15` with
`save_total_limit=3`, a crash loses at most ~150 seconds of work (at the probed ~10s/step), and
disk stays bounded to roughly 3 checkpoints' worth (~30GB) per condition regardless of how many
times a crash/resume cycle repeats over the full run.

**Repeatability plan for this specific execution**: runs are expected to crash and resume one or
more times, per this session's own data. Each crash gets diagnosed (checkpoint present? resumed
cleanly? real OOM or something new?) and resumed via `--resume-from-checkpoint auto`
(`get_last_checkpoint`-based), not restarted from scratch — this is exactly the repeatability
guarantee the infra plan built and real-hardware-verified. If a genuinely new failure mode shows
up (not a plain CUDA OOM), that's treated as a real finding requiring investigation before
blindly retrying, the same standard applied during the infra plan's own Task 7.

### 2. Evaluation

Once each condition's final checkpoint (`checkpoint-500`) exists, run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m turn_level_rewards.evaluate_ppo \
    --condition {ppo,mt_ppo} --checkpoint-dir outputs/{ppo,mt_ppo}/checkpoint-500 \
    --eval-size 7405 --output results/{ppo,mt_ppo}_eval_metrics.json
```

`--eval-size 7405` matches the GRPO track's own full-scale held-out evaluation (Phase 6), the
entire `hotpot_qa` distractor validation split — not a smoke-scale slice.

If a run's own trackio/train_log curves show a real collapse (per `CollapseMonitor`'s alerts) before
step 500, the paper's own methodology applies: evaluate the last checkpoint before the collapse
instead of the final one (`evaluate_ppo.py`'s `--checkpoint-dir` already supports pointing at any
checkpoint for exactly this reason) — decided per-run from the actual curves, not assumed here.

### 3. Comparison write-up

New `scripts/compare_ppo_runs.py` (or an extension of the existing `scripts/compare_runs.py`,
decided during implementation based on how much is actually shareable), mirroring Phase 6's
structure:
- EM/F1/retrieval_fraction for `ppo` vs `mt_ppo`, both training curves (from trackio/train_log)
  and final held-out numbers (from `evaluate_ppo.py`'s output).
- Phase 6's "is more training needed?" validation checklist applied before calling any result
  final — same standard, not a lighter one because this is the PPO track.
- A plainly stated verdict either way (real advantage, no meaningful difference, or inconclusive).

### 4. Matplotlib charts

Using the `dataviz` skill (mandatory per the Phase 7b task list — not ad hoc matplotlib
defaults). Per the earlier brainstorm's resolution: **retrofit both** the existing
`outcome_only`/`turn_level` (GRPO) charts and the new `ppo`/`mt_ppo` comparison into one visually
consistent system, rather than leaving two different chart styles in the same README.

### 5. README

New Results section for `ppo`/`mt_ppo`, following the existing GRPO section's reader-focused
structure (self-contained, grounded in the paper's own numbers, no internal doc-path citations).

## Out of scope

- Phase 8 (LLM-as-judge) and Phase 8b — separate, later phases, unaffected by this one.
- MT-GRPO — still out of scope per CLAUDE.md.
- Any further changes to `MTPPOTrainer`'s core training logic (reward placement, GAE, PPO-clip
  loss) — already correct and verified; this phase only executes and analyzes, it doesn't modify
  the trainer's algorithmic core.

## Handoff to implementation plan

Next: `superpowers:writing-plans`, covering (a) launching and babysitting both full runs through
however many crash/resume cycles occur, (b) evaluation, (c) the comparison write-up, (d) charts,
(e) README. Given the runs alone take ~1.4 hours each and must be actively monitored for
crash/resume (not fire-and-forget), the plan's first task is the launch-and-monitor operation
itself, not a small code change.
