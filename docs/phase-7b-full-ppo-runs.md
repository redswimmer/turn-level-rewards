# Phase 7b: Full PPO/MT-PPO training runs + evaluation + comparison

## Goal

Run both `ppo` and `mt_ppo` (deterministic rewards) to completion, evaluate both checkpoints on
the full held-out set, and produce the actual comparison — the paper's real Table 2 headline
result (`PPO-OR` vs `MT-PPO`), reproduced on this repo's HotpotQA/wiki-18 pipeline. Also replace
the README's current plain charts with a matplotlib-driven visual system designed to actually
tell the comparison's story, not just plot numbers.

This phase exists because Phase 7's own doc and Phase 8's own doc both stop at smoke-test scale
— see CLAUDE.md's Roadmap section (2026-07-23 addendum) for why that left a real gap. 7b is the
PPO-track analog of Phases 5+6 combined (full runs, then `evaluate.py`/`compare_runs.py`/write-up)
for the GRPO track.

## Read first

`CLAUDE.md`'s Goal section (recall: PPO/MT-PPO's Table 2 numbers are the paper's actual
best-benchmarked result, not a secondary ablation). Phase 7's Handoff notes
(`docs/phase-7-mt-ppo.md`) — checkpoint paths, real per-step wall-clock observed, any TRL/critic
surprises, and whatever logging mechanism Phase 7 actually wired up (this doc doesn't assume
trackio integration works identically to `GRPOTrainer`'s automatic one, since `MTPPOTrainer` is
built directly on `transformers.Trainer` — confirm what Phase 7 actually did before assuming).
Phase 6's doc (`docs/phase-6-evaluation-comparison.md`) for the shape of a real evaluation +
comparison write-up, including its "is more training needed?" validation checklist and its
follow-up-experiments pattern — reuse that structure rather than reinventing it.

## Prerequisites (entry state)

- Phase 7 done: `MTPPOTrainer` works for both `ppo`/`mt_ppo` conditions, smoke-tested, per
  `docs/phase-7-mt-ppo.md`'s exit criteria.
- Retrieval server running and stable.

## Tasks

- [ ] Full training runs: both `ppo` and `mt_ppo`, 500 rollout-collection steps × 4 inner PPO
      epochs per the paper's Section 6.2/C.1.3 spec (already recorded in
      `docs/superpowers/specs/2026-07-05-phase-7-mt-ppo-design.md`). Confirm real wall-clock
      before committing to the full budget — Phase 7's smoke test only ran a couple of steps;
      PPO's extra critic forward/backward pass and multiple inner epochs make this a different
      cost profile than GRPO's Phase 5 runs, not an assumed-similar one.
- [ ] A held-out evaluation path for `MTPPOTrainer` checkpoints. **This is not just "reuse
      `evaluate.py`"** — Phase 6's `evaluate.py` works by constructing a `GRPOTrainer` and calling
      its standard `.evaluate()`, which relies on `GRPOTrainer`'s own `prediction_step` override.
      `MTPPOTrainer` is a different class with its own rollout loop; the eval path needs to either
      (a) reuse `MTPPOTrainer`'s own rollout loop directly (no gradient updates, same
      `SearchEnv`/`rewards.py` machinery) against the held-out set, or (b) some other mechanism —
      resolve this as a real design question when this phase starts, don't assume (a) is right
      without checking whether the rollout loop cleanly separates from the update step.
- [ ] Comparison write-up: EM/F1/retrieval-fraction, `ppo` vs `mt_ppo`, mirroring Phase 6's
      "is more training needed?" validation checklist before treating any result as final.
- [ ] Matplotlib visuals (new — not in the original Phase 7 design): the current README charts are
      functional but plain; redesign them to actually communicate the comparison's story clearly.
      **Use the `dataviz` skill** when doing this chart work, not ad hoc matplotlib defaults.
      Decide during this phase (not assumed here) whether to also retrofit the existing
      `outcome_only`/`turn_level` charts to the same new visual system for consistency, or leave
      those as-is and only apply the new system to the `ppo`/`mt_ppo` results.
- [ ] README: add a Results section for `ppo`/`mt_ppo`, following the existing section's
      reader-focused structure (self-contained, grounded in the paper's own numbers, no internal
      doc citations) but using the new chart system.

## Exit criteria (all must be true before handing off)

- [ ] Both `ppo` and `mt_ppo` checkpoints exist, are loadable, and completed their full training
      budget.
- [ ] Held-out evaluation completed for both, using a real (not placeholder) eval path.
- [ ] A comparison verdict recorded (real advantage, no meaningful difference, or inconclusive —
      state it plainly either way, same standard Phase 6 held itself to).
- [ ] New matplotlib charts committed and embedded in the README, reviewed for actually being
      clearer than the prior charts (not just different).

## Handoff notes

<!-- Fill in after completing this phase: real wall-clock/cost for the full runs, the eval-path
design actually chosen and why, the comparison verdict and numbers, and anything about the new
chart system worth carrying into Phase 8b (e.g. a reusable plotting module, a settled color/style
convention). Leave this section for the next fresh agent to read first. -->

(not yet started — full training runs, evaluation, comparison, and matplotlib visuals are all
still pending. See "Infrastructure verified" below for a real, completed live-smoke-test pass over
the resume/diagnostics/eval machinery this phase's full runs will depend on.)

### Infrastructure verified (live smoke test, 2026-07-24, before any full run)

Before committing to the full 500-step budget, Tasks 2-6's checkpoint-resume, diagnostics, and
`evaluate_ppo.py` machinery were exercised end-to-end for real (real GPU, real Qwen3.5-0.8B, real
retrieval server) for the first time — previously only unit-tested at the pure-logic level. All
runs used `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` per Task 1's wall-clock probe
(`docs/superpowers/probes/2026-07-23-phase-7b-wallclock-probe.md`).

**Checkpoint rotation (`save_total_limit`, Task 5's fix) — confirmed working.** A 10-step `ppo`
run (`--train-size 32 --max-steps 10 --num-rollouts-per-step 2 --save-steps 3 --seed 42`) saved
checkpoints at steps 3, 6, 9, 10 but rotation correctly deleted `checkpoint-3`, leaving exactly
`checkpoint-6`, `checkpoint-9`, `checkpoint-10` on disk — each containing `policy/`, `critic/`,
`optimizer.pt`, `rng_state.pth`, `trainer_state.json` (plus `scheduler.pt`, not mentioned in the
original design but present and harmless). This is the real-hardware confirmation that
`save_total_limit` will keep the full 500-step run's disk footprint bounded, not the accumulate-
everything failure mode the design doc worried about.

**Checkpoint-resume reproducibility (the Task 5 code-review bug, real-hardware test) —
confirmed correct.** Resuming from `checkpoint-6` with the same `--seed 42` and continuing to
`--max-steps 10`: stdout's first printed step line read `step 7/10` (not `step 1/10`), confirming
`self.state.global_step` was correctly restored to 6 and the loop resumed rather than restarted.
More importantly, the resumed run's episode-0 questions at steps 6-9 (0-indexed; "step 7/10"
through "step 10/10" in display terms) were byte-identical to the original uninterrupted run's
questions at those same steps (e.g. both produced `"Al-Battani and Ibn al-Shatir, have which
mutual occupation?"` as episode 0's question at step 6) — direct, real-hardware evidence that the
optimizer-orphaning bug found and fixed in Task 5's code review does not resurface when the real
model/optimizer are involved, not just the synthetic tiny-model repro used to verify the fix.

**Real deviation from the plan's assumed file layout (worth knowing before writing
`evaluate.py`/any future resume tooling): `--resume-from-checkpoint` does not redirect
`output_dir`.** `build_ppo_config` hardcodes `output_dir=f"outputs/{condition}"` (line ~184 of
`train_ppo.py`) with no `--output-dir` CLI override. The task brief's literal Step 3/4 recipe
(`cp -r outputs/ppo outputs/ppo_resume_test`, then `--resume-from-checkpoint
outputs/ppo_resume_test/checkpoint-6`) only uses that copied directory to source the initial
weights/optimizer/RNG state — the resumed run's own new checkpoints and `train_log.jsonl` lines
land back in the *original* `outputs/ppo`, appended after the pre-existing steps 0-9, not in
`outputs/ppo_resume_test`. Concretely, `outputs/ppo/train_log.jsonl` ended up with 14 lines: steps
0-9 from the original run, then steps 6-9 again from the resumed run. The brief's exact literal
verification one-liner (comparing `outputs/ppo/train_log.jsonl` against
`outputs/ppo_resume_test/train_log.jsonl`) therefore raises `ValueError: step 6 not found` against
the untouched copy — not a bug in the resume code, just a mismatched assumption about where a
resumed run's data actually lands. The substantive check (comparing the first vs. second
occurrence of `step == 6` within `outputs/ppo/train_log.jsonl` itself) is what actually confirmed
reproducibility above. This is fine for a real crash/resume in production (you'd always want a
resumed run to keep appending to the same output_dir it was already using) — it just means a
*test* that wants an isolated before/after comparison needs to snapshot the original log file
itself (e.g. `cp outputs/ppo/train_log.jsonl /tmp/original.jsonl` before resuming), not just the
checkpoint directory, if it wants two genuinely separate files to diff.

**`evaluate_ppo.py` — confirmed working against a real checkpoint.** Running against
`outputs/ppo/checkpoint-10` with `--eval-size 4` completed without error and wrote a JSON file
with all four expected keys and finite values: `{"exact_match": 0.0, "f1": 0.0,
"retrieval_fraction": 0.375, "num_examples": 4}` (an all-zero EM/F1 is unremarkable at this
smoke-test scale — 10 training steps × 2 rollouts is nowhere near enough to learn anything; the
point of this check was the eval path executing and producing well-typed output, not accuracy).

**mt_ppo diagnostics/rotation — confirmed identical to `ppo`, including a full clean 10/10 run.**
Every `mt_ppo` attempt (pre- and post-fix, see below) that got past its first few steps produced
the same shape of diagnostics (`ratio_mean`/`clip_fraction` present, checkpoint saves at the
expected steps) as `ppo`, with no condition-specific exception or code-level failure anywhere.
After the `torch.cuda.empty_cache()` fix (commit `65e082d`, see below) landed, a full `mt_ppo`
10/10 run succeeded on the first attempt: checkpoint rotation correct (`checkpoint-3` deleted,
`checkpoint-6`/`9`/`10` present), `ratio_mean`/`clip_fraction` present in all 10
`train_log.jsonl` lines.

### Real finding: OOM rate observed this session was much worse than Task 1's probe — since root-caused and fixed (commit `65e082d`)

**Status: resolved.** The finding below (6/7 failures, ~86%) was real, live data from this
session and is kept in full for the record — it's a genuine, valuable measurement of this GPU's
actual operating margin under this workload, not a stale or superseded concern. It has since led
to a real fix, not just a documented risk: commit `65e082d` ("Fix checkpoint-save-boundary CUDA
OOM in MTPPOTrainer") adds a single `torch.cuda.empty_cache()` call in `_save_full_checkpoint`,
immediately before `_save_optimizer_and_scheduler` — freeing cached-but-unused GPU blocks to give
`torch.save`'s internal serialization of GPU-resident AdamW optimizer state the working headroom
it needs at this workload's high (~92-94%) GPU utilization. **Re-verified directly**: with the
fix in place, `mt_ppo`'s Step 6 regression check (the same command that had failed 4/4 times
pre-fix) succeeded on its very first post-fix attempt — full 10/10 steps, correct checkpoint
rotation, all diagnostics present. Only one post-fix attempt was needed (the plan allowed up to
2), so no data exists yet on whether the fix's benefit holds up over more attempts/a longer run —
treat "resolved" here as "confirmed to help, re-tested once, real" rather than "guaranteed zero
future OOM risk over 500 steps." The recommendation below (lower `save_steps`, active
babysitting for the eventual full run) is still worth keeping as a belt-and-suspenders posture
even with the fix in place, precisely because it's only been re-verified once.

Task 1's wall-clock probe (`docs/superpowers/probes/2026-07-23-phase-7b-wallclock-probe.md`,
`--max-steps 15`) measured `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` cutting the OOM rate
from 3/3 to 1/3 at `num_rollouts_per_step=2`. This task's own live smoke test, at a smaller
`--max-steps 10`, saw a materially worse rate:

- `ppo` (Step 1's 10-step run): OOM on attempt 1 (step 9 of 10), OOM on attempt 2 (step 3 of 10),
  **succeeded on attempt 3** (10/10, ~80s wall-clock).
- `mt_ppo` (Step 6's regression check, same command): OOM on attempt 1 (step 2 of 10), OOM on
  attempt 2 (step 6 of 10), OOM on attempt 3 (step 6 of 10), OOM on attempt 4 (step 7 of 10, this
  one surfacing as `RuntimeError: Triton Error [CUDA]: out of memory` inside a `fla`
  `l2norm_bwd` Triton kernel's backward pass rather than the usual `torch.OutOfMemoryError` —
  same root cause, different code path hitting the wall first). **No successful 10/10 `mt_ppo` run
  was obtained in 4 attempts.**

**Combined: 6 failures out of 7 total training attempts this session (~86%), against Task 1's
documented ~1-in-3 (~33%) rate at a slightly larger `--max-steps`.** Every failure's traceback
bottomed out in a genuine CUDA/Triton out-of-memory condition at the critic's forward or backward
pass (`_forward_critic_values` / `_ppo_update`'s `backward()` call) — the exact same failure site
Task 1 already documented, not a new code path or a regression introduced by Tasks 2-6 (the `ppo`
condition's Steps 1-5 all completed correctly once it got a clean run, exercising every piece of
new code — diagnostics, rotation, resume, eval — without a single non-OOM exception). GPU state
was independently confirmed clean between every attempt (no orphaned processes, ~570-650MiB
baseline matching the documented idle desktop load, normal temperature).

Two candidate explanations, both plausible, neither confirmed: (a) this is simply the same
episode-length-driven variance Task 1's analysis already described (OOM risk is dominated by a
single episode's sequence length, which is randomly distributed per rollout — an unlucky run of
long episodes is a real possibility, and `mt_ppo`'s turn-level reward may correlate with slightly
different/longer completions than `ppo` in a way that happens to bias this session's rollouts
toward the OOM edge, though this repo's own design notes already state both conditions have
equivalent compute cost in aggregate); or (b) this session's actual failure rate is genuinely
worse than Task 1's ~33% estimate, which was measured once, at a different `--max-steps`, and may
itself have been on the optimistic side of the same variance. This finding does not block signing
off Phase 7b's infrastructure as working — every piece of new code performed correctly on every
run that didn't OOM, and OOM itself is an already-known, already-mitigated (not eliminated)
hardware constraint, not a Tasks 2-6 defect. But **whoever launches the full 500-step run should
read this before doing so**: at this observed rate, a `save_steps` interval as loose as the plan's
default (50) risks losing meaningfully more work per crash than assumed, and the run should be
treated as needing active babysitting/auto-resume (`--resume-from-checkpoint auto`) rather than a
fire-and-forget background job. Consider a lower `save_steps` (e.g. 10-15) for the full run
specifically to bound the blast radius of the next OOM, which — per this session's data — should
be expected, not treated as a rare edge case.

**One honest caveat on the fix's stated causal mechanism, worth recording rather than smoothing
over**: this session's own captured tracebacks (all 6 failures) never actually show
`_save_optimizer_and_scheduler`, `torch.save`, or any checkpoint-saving code in the stack — every
failure occurred during ordinary forward/backward compute (`_forward_critic_values`, attention,
MLP, or a `fla` Triton backward kernel), not inside a save call itself. The fix's commit message
frames the root cause as OOMs landing "at or immediately after a checkpoint-save step boundary,"
which is a looser fit to the raw data than it reads (e.g. `ppo`'s attempt-1 OOM was at step 9, one
step short of the step-9 save; `mt_ppo`'s attempt-4 OOM was at step 7, one step past the step-6
save). This doesn't make the fix wrong or useless — `torch.cuda.empty_cache()` freeing
fragmented-but-unused blocks is a generically reasonable thing to do at a natural pause point
regardless of exactly which subsequent step trips over the resulting headroom, and the re-test
above is real, direct evidence it helped on this run. But the specific "boundary-triggered" causal
story should be treated as a plausible contributing hypothesis validated by one successful re-run,
not a fully nailed-down root cause — worth another data point or two before treating it as
settled science.
