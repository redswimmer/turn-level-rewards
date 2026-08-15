# Phase 7b prep: wall-clock/OOM scale probe

**Date**: 2026-07-23
**Goal**: empirically choose `num_rollouts_per_step` for the full PPO/MT-PPO training runs
(Phase 7b) on this machine's single RTX 4090, and confirm the paper's target of 500 training
steps is wall-clock-feasible at that scale. Real measurement only — no code changes, no
guessing.

## Step 1: Retrieval server

```
curl -s -X POST http://localhost:8000/retrieve -H 'Content-Type: application/json' -d '{"queries": ["test"], "topk": 1, "return_scores": false}'
```

Returned a real `{"result": [...]}` JSON payload — server already running, no launch needed.

## Steps 2-4: probe runs

All runs: `--train-size 32 --max-steps 15`, `--condition ppo` unless noted. GPU was idle
(~400-500MiB used, 0% util) before every run.

| # | Command (core args) | Outcome | Wall-clock (`time`) | Notes |
|---|---|---|---|---|
| 1 | `--condition ppo --num-rollouts-per-step 2` | **OOM** at step 4/15 | 1m53.4s | `torch.OutOfMemoryError` in `_forward_critic_values` |
| 2 | same, retry | **OOM** at step 8/15 | 2m42.6s | same failure site |
| 3 | same, retry w/ GPU-mem polling | **OOM** at step 10/15 | 2m0.5s | see memory trace below |
| 4 | `--condition ppo --num-rollouts-per-step 4` | **OOM** at step 3/15 (only 2 steps completed) | 1m55.3s | fails faster than r=2, as expected |
| 5 | `--condition ppo --num-rollouts-per-step 1` | **OOM** at step 6/15 | 1m23.1s | reducing batch below the default did **not** fix it (see analysis) |
| 6 | `--condition ppo --num-rollouts-per-step 2` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | **OOM** at step 13/15 (12/15 completed) | 2m2.3s | meaningful improvement over run 1-3 |
| 7 | `--condition mt_ppo --num-rollouts-per-step 2` + `expandable_segments:True` | **clean, 15/15** | 2m26.0s | no OOM |
| 8 | `--condition ppo --num-rollouts-per-step 2` + `expandable_segments:True`, retry | **clean, 15/15** | 2m33.3s | no OOM |

Full logs for all 8 runs are preserved at `/tmp/probe_ppo_r2.log`, `/tmp/probe_ppo_r2_attempt2.log`,
`/tmp/probe_ppo_r2_attempt3.log` (+ `/tmp/gpu_mem_poll_r2_attempt3.log`), `/tmp/probe_ppo_r4.log`,
`/tmp/probe_ppo_r1.log`, `/tmp/probe_ppo_r2_expandable.log`, `/tmp/probe_mt_ppo.log`,
`/tmp/probe_ppo_r2_expandable_attempt2.log`.

### GPU-memory trace during run 3 (r=2, no `expandable_segments`, 3s polling)

Memory rose from a cold ~15GB to a **steady state of ~18.2-18.5GB** (of 23.51GB total) across the
first several steps, held roughly flat for ~9 polls (~27s, corresponding to steps 3-9), then
spiked sharply to ~21.7GB in the poll immediately before the crash. This is consistent with
per-episode sequence-length variance (an unusually long tool-calling trajectory in that step's
batch), not a monotonic memory leak across steps.

### Why reducing `num_rollouts_per_step` below 2 didn't help (run 5)

Read `src/turn_level_rewards/train_ppo.py`'s `_ppo_update`/`_forward_critic_values`: policy and
critic forward/backward passes are done **per-episode in a Python loop**
(`self.model.critic.model(input_ids=input_ids)` called once per episode's own
`full_token_ids`), not batched into a single matmul across the group. This means:
- `num_rollouts_per_step` does **not** change the peak per-forward-pass memory footprint (that's
  driven purely by a single episode's own sequence length).
- It **does** change (a) wall-clock roughly linearly (more sequential per-episode
  forward/backward calls per step), and (b) how many independent "does this one episode's length
  blow the budget" chances occur within a single step — more rollouts per step is strictly more
  such chances, which is exactly why r=4 failed fastest (step 3) and r=1 still failed (step 6):
  the OOM risk is dominated by individual episode length, and even a single long episode at r=1
  is enough on this GPU's thin remaining headroom (base model + critic + both Adam optimizers
  already consume the large majority of the 23.51GB card before any activations).

### `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (env var, zero code changes)

The OOM error message itself suggested this flag to reduce fragmentation. It is a pure
environment variable — no source change — so it was tested within this probe's "no code changes,
only run the existing CLI" scope. Result: a real, measured improvement. Without it, r=2 OOM'd in
3/3 attempts (steps 4, 8, 10). With it, r=2 OOM'd in 1/3 attempts (step 13) and completed cleanly
in 2/3 attempts (both full 15/15 runs, `ppo` and `mt_ppo`).

### Step 4 (`mt_ppo` confirmation)

`mt_ppo` at r=2 + `expandable_segments:True` completed all 15/15 steps cleanly in 146.0s wall-clock
— directly comparable to `ppo`'s clean 153.3s run at the same settings, confirming the two
conditions have equivalent compute cost (they only differ in reward placement, not in how many
forward/backward passes happen), as the design predicted.

## Chosen `num_rollouts_per_step` = 2 (with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)

Reasoning:
- **r=4 is strictly worse** — it OOM'd the fastest (2 successful steps vs. 8-12 for r=2) and the
  brief's explicit stopping criterion ("stop increasing once an OOM appears") was hit immediately.
  No r=8 attempt was made.
- **r=1 is not a real fix** — since forward/backward is per-episode (not batched), reducing
  rollouts below 2 doesn't meaningfully reduce OOM risk (it still failed, at step 6) while halving
  the number of state transitions the group-relative PPO update sees per step. No reason to
  prefer it over 2.
- **r=2 + `expandable_segments:True` is the best measured combination**: 2/3 clean full runs at
  the target 15 steps, one partial failure at step 13/15 (87% of the way through). This is a real,
  substantial, zero-code-change improvement over r=2 without the flag (0/3 clean).

## Per-step wall-clock and 500-step estimate

Using the two **clean, complete** 15-step runs (both at r=2 + `expandable_segments:True`):

| Run | Total wall-clock | Per-step (wall-clock / 15) |
|---|---|---|
| `mt_ppo` | 146.0s | 9.73s/step |
| `ppo` | 153.3s | 10.22s/step |
| **Average** | | **~9.97s/step ≈ 10s/step** |

**Estimated 500-step wall-clock**: `10s/step * 500 steps ≈ 5,000s ≈ 83 minutes ≈ 1.4 hours`, per
condition. This is well within feasible range — nowhere near "multiple days."

**Decision: 500 steps is kept as-is, not reduced.** The wall-clock estimate (~1.4 hours/condition)
gives no reason to reduce it — per the task brief, reduction is only warranted if the estimate is
genuinely impractical, and it isn't.

## A real, separate concern: OOM reliability over a full 500-step run

Wall-clock feasibility and **crash-free reliability** are two different questions, and this probe
found a meaningfully worse answer to the second one than previously documented. CLAUDE.md/Phase 7's
handoff notes recorded a ~12.5% OOM rate at `num_rollouts_per_step=2, max_steps=2` (1 failure in 8
short runs). This probe, running 15 steps instead of 2, found:
- **Without** `expandable_segments:True`: 3 attempts at r=2, **3 failures** (100%), at steps 4, 8,
  and 10.
- **With** `expandable_segments:True`: 3 attempts at r=2 (2 `ppo` + 1 `mt_ppo`), **1 failure**
  (step 13/15).

Extrapolating the with-flag failure rate (roughly 1-in-3 chance of at least one OOM somewhere in a
15-step window) to a per-step failure probability gives ~2.6%/step. Compounded over 500 steps,
that implies a very high probability of hitting **at least one** OOM crash somewhere during a full
500-step run, even with the flag — this is a small-sample extrapolation, not a precise figure, but
the direction is clear and worth taking seriously.

This matters because `train_ppo.py` currently has **no resume-from-checkpoint path**: `train()`
saves checkpoints only every `self.args.save_steps` (defaults to `TrainingArguments`'s default of
500 — i.e., effectively only at the very end of a 500-step run, unless a future config sets it
lower), and there is no CLI flag or code path to resume training from an existing checkpoint after
a crash (confirmed by inspection — `resume_from_checkpoint` appears only in a comment noting the
base `Trainer` signature, never wired into `train()`'s own logic). A mid-run OOM at, say, step 300
would currently mean **losing all progress and starting over**, not resuming.

This is **out of scope for this probe task** (no code changes), but is a real, actionable
finding for whichever phase actually executes the full 500-step runs (Phase 7b): before that
launch, either (a) set `save_steps` much lower (e.g. every 25-50 steps) and add a real
resume-from-checkpoint code path, or (b) accept the risk and be prepared to manually rerun on
crash, or (c) invest in a proper OOM fix beyond the `expandable_segments` mitigation (e.g.
`max_completion_length` reduction, more aggressive gradient checkpointing, or reducing the
critic's own memory footprint). This is flagged for the executing agent, not solved here.

## Recommended launch command for Phase 7b

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m turn_level_rewards.train_ppo \
    --condition {ppo,mt_ppo} --max-steps 500 --num-rollouts-per-step 2 --seed <seed>
```

## Files changed/created

- Created: `docs/superpowers/probes/2026-07-23-phase-7b-wallclock-probe.md` (this file).
- No source code changes.
- Probe logs left at `/tmp/probe_*.log` and `/tmp/gpu_mem_poll_r2_attempt3.log` (not committed —
  scratch artifacts, referenced above for traceability).
- `outputs/ppo/` and `outputs/mt_ppo/` each gained a `checkpoint-15/` directory (~6GB apiece) from
  this probe's clean 15-step runs, plus updated `train_log.jsonl`/`sample_completions.log`.
  `outputs/` is gitignored. **Not cleaned up** — disk is already at 94% used (33GB free) and this
  repo's own history (Phase 6 handoff notes) records that checkpoint deletion should go through
  explicit user approval first, and a pre-existing `checkpoint-2/` in both directories predates
  this session (timestamps ~13:12-13:13 vs. this probe's ~18:40+), suggesting possible concurrent
  work per the "parallel tooling" note in user memory. Flagged here for the user's own cleanup
  decision, not acted on.

## Concerns for follow-up

1. **Disk space**: 33GB free, 94% used. Not this task's problem to fix, but worth the user's
   attention before Phase 7b's real runs (which will also write checkpoints).
2. **OOM reliability over long runs**, detailed above — the single most important finding of this
   probe. Wall-clock is fine; crash-free completion of a full 500-step run is not yet a safe bet
   without either a resume mechanism or a lower-risk memory profile.
3. `expandable_segments:True` should be treated as a **required** part of the Phase 7b launch
   command, not optional — it was the difference between 0/3 and 2/3 clean completions in this
   probe.
