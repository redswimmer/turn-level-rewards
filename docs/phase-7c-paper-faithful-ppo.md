# Phase 7c: Paper-faithful five-arm comparison

**PPO-OR · PPO-MR · MT-PPO · GRPO-OR · GRPO-MR — all under the paper's own reward design.**

**Status**: implementation complete, smoke-tested, verified to 100 steps. **Ready to launch the
five full runs.** Nothing is half-finished; everything described here is committed.

**Read this document first and in full.** It is self-contained: you do not need to read the
Phase 7b history to execute it. Read `CLAUDE.md` for repo-wide context (model, dataset,
retrieval server, guiding principles) and then come back here.

---

## 1. Why this phase exists

Phase 7b ran `ppo` and `mt_ppo` to 500 steps and produced held-out EM of 0.031 and 0.123. Those
numbers are **discarded**. They measured a defective trainer against a reward function that was
not the paper's. Two independent classes of problem were found, both by reading
arXiv:2505.11821v2 directly rather than this repo's paraphrase of it.

### 1a. Four implementation defects (fixed, commit `4d15ff1`)

| Defect | Effect |
|---|---|
| Rollout loop denied the final answer turn | 58% of `ppo` / 21% of `mt_ppo` episodes ended on a `role: tool` message with no generation left in which to answer, then took a format penalty for it |
| No advantage normalization | `policy_lr=1e-6` is the paper's, but the paper runs on veRL which normalizes advantages; raw advantages here are ~0.1-0.3, so the effective step size was several times too small |
| No gradient clipping | Unbounded accumulated gradients; a textbook cause of the collapse observed (format compliance 0.542 → 0.120 between steps 400-500) |
| KL term sign inverted | `mean(new − old)` over old-policy samples estimates `−KL(old‖new)`, so the penalty *maximised* divergence. Replaced with Schulman's k3 estimator |

The answer-turn defect also made the PPO track **incomparable to the GRPO track**. TRL's
`GRPOTrainer._tool_call_loop` runs `while tool_calls and iteration_num < max`, executing tools
and *then* generating, so at the same nominal cap GRPO always ends on a generation. Verified
directly in TRL's source. Our loop now runs `n_max + 1` generations for `n_max` tool rounds.

### 1b. The reward function was not the paper's (fixed, commit `f2e2cba`)

The PPO track had silently inherited the GRPO track's rewards. `CLAUDE.md` justifies the
format/F1+EM design *for GRPO* — its group-relative advantage needs within-group variance, which
binary EM does not provide. **That justification does not transfer to PPO**, which has a real
per-token critic.

Two gaps mapped directly onto observed failures:

- **No search penalty.** The paper: *"removing this reward term (λs = 0.0) leads to unstable
  training and degenerate behaviors, such as uncontrolled search usage or non-convergent
  rollouts."* We ran with λs = 0 and saw exactly that — episodes growing 386 → ~1190 action
  tokens, burning the whole turn budget without answering.
- **Format penalty 10x too weak** — −0.1 against an F1-scale outcome, versus the paper's −1.0
  against +1.0. The paper reports **0.998** format correctness on HotpotQA; we measured ~0.48.

### 1c. PPO-MR was missing, so the ablation was uninterpretable

Section 6.1 defines three PPO arms. We only had two, and the two we had differed in *both* the
reward content and its placement — so no measured advantage could be attributed to Eq. 9, which
is the paper's actual contribution.

---

## 2. The experiment

Five arms, **all at seed 42**, 500 steps each. Within each track every hyperparameter is
identical; arms differ only in which reward components exist and where they land.

| Arm | Trainer | Paper's definition (§6.1, verbatim) | Implementation |
|---|---|---|---|
| `ppo` (**PPO-OR**) | `MTPPOTrainer` | *"vanilla PPO trained with only outcome rewards, where the trajectory-level reward is a binary signal indicating final-answer correctness"* | `paper_binary_outcome_reward` (1.0/0.0, **no format term**) at the last token |
| `ppo_mr` (**PPO-MR**) | `MTPPOTrainer` | *"...merged intermediate and outcome rewards, where the trajectory-level reward combines intermediate rewards (retrieval correctness) and outcome rewards (answer correctness and format correctness)"* | `R^O + sum(R^I)`, **all** at the last token |
| `mt_ppo` (**MT-PPO**) | `MTPPOTrainer` | Same components, turn-level placement via Eq. 9, λs = 0.1 | `R^I` at each intermediate turn boundary, `R^O` at the last token |
| `grpo_or` (**GRPO-OR**) | TRL `GRPOTrainer` | *"vanilla GRPO trained with only outcome rewards, where the trajectory-level reward is a binary signal indicating final-answer correctness"* | `paper_grpo_outcome_reward` (1.0/0.0) |
| `grpo_mr` (**GRPO-MR**) | TRL `GRPOTrainer` | *"...merged intermediate and outcome rewards..."* (same wording as PPO-MR) | `paper_grpo_merged_reward`: `R^O + 0.3 × retrieval_fraction`, one trajectory-level scalar |

**The GRPO arms need no custom trainer.** They are reward-function changes only, so TRL's
`GRPOTrainer` handles them natively — much lower risk than the hand-built `MTPPOTrainer`. (MT-GRPO
*would* need a custom trainer, which is why `CLAUDE.md` scopes it out: TRL exposes no hook for
per-turn advantage estimation.)

**Why five arms and not three**: holding the reward design fixed across algorithms is exactly what
makes the paper's Table 2 cross-algorithm comparison valid — §6.1 defines GRPO-OR/GRPO-MR with
wording identical to PPO-OR/PPO-MR. This repo's earlier GRPO track (Phases 5-6) used its own
reward design, so those runs cannot join this table. They are kept as record, not as a result
here.

**Why PPO-MR matters**: `ppo → ppo_mr` isolates the value of the added reward signal;
`ppo_mr → mt_ppo` isolates the value of turn-level placement. Without MR the comparison changes
two variables at once.

### Reward components (arXiv:2505.11821v2 §5.2, implemented in `rewards.py`)

```
R^I (per intermediate turn):
    retrieval existence   +0.3   if this turn's search surfaced a gold title (MARGINAL gain)
    format                +0.1 correct / -0.2 incorrect
    search penalty        -0.1 * cumulative searches so far   (lambda_s = 0.1, paper default)

R^O (final turn):
    correct answer + correct format    +1.0
    incorrect answer + correct format  +0.2
    incorrect format                   -1.0
```

`found_gold` is the **marginal** retrieval gain, not the cumulative value: `SearchEnv`'s
`retrieval_fraction` only ever grows, so "ground truth in results" for *this* turn means the
fraction rose during it.

### Anchor targets — the paper's Table 2, HotpotQA

| Arm | EM | Format correctness |
|---|---|---|
| GRPO-OR | 0.331 | 0.513 |
| GRPO-MR | 0.416 | (not listed) |
| PPO-OR | 0.435 | 0.916 |
| PPO-MR | 0.436 | (not listed) |
| **MT-PPO** | **0.453** | **0.998** |

Phase 7b's MT-PPO was EM 0.123 / format ~0.48 — far off, which is what prompted this phase.

**Read the effect sizes before interpreting results.** They are very unequal, and this determines
what these runs can and cannot show:

| Comparison | Paper's gap | Realistically detectable here? |
|---|---|---|
| GRPO-OR → GRPO-MR | **+8.5 pts** | Yes |
| PPO-OR vs GRPO-OR | **+10.4 pts** | Yes |
| PPO-OR → PPO-MR | +0.1 pts | No — it is ~zero in the paper too |
| PPO-MR → MT-PPO (**the paper's own claim**) | **+1.7 pts** | **Probably not at n=1 seed** |

At a 2,000-row eval the standard error on EM is ~1 point, so a 1.7-point gap is ~1.2 SE — not
separable from noise with one seed, before even accounting for the extra gradient noise from a
batch of 4 versus the paper's 512. **Do not report a null MT-PPO-vs-PPO-MR result as evidence
against Eq. 9**; at this scale it is an underpowered test, not a refutation. Resolving that gap
honestly needs multiple seeds with paired comparisons — which is the §7 decision.

What these runs CAN establish: that MT-PPO lands in the right range (~0.45, versus 0.123 before),
that the reward-design effect reproduces on the GRPO side, and that the PPO-vs-GRPO gap
reproduces.

Note also that the paper's GRPO-OR format correctness is only **0.513** — roughly half its
episodes never produce a valid answer. That is the expected consequence of a binary reward with
no format term, and it means a low format score for `grpo_or` here is a reproduction, not a bug.

---

## 3. How to run it

**Prerequisite**: the retrieval server must be up. Verify with:

```bash
curl -s -X POST http://localhost:8000/retrieve -H 'Content-Type: application/json' \
  -d '{"queries":["Knox County Regional Airport"],"topk":1,"return_scores":false}' | head -c 200
```

If it is not running, see `docs/phase-1-retrieval-infra.md` for the launch command.

### Training (sequential — the GPU fits one run at a time)

The two tracks use different entry points: PPO arms go through `train_ppo` (the hand-built
`MTPPOTrainer`), GRPO arms through `train` (TRL's `GRPOTrainer`).

```bash
# PPO arms  (~1.7 / 3.5 / 3.0 h respectively, measured at 100 steps and extrapolated)
for C in ppo ppo_mr mt_ppo; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -m turn_level_rewards.train_ppo \
      --condition $C --train-size 90447 --max-steps 500 \
      --num-rollouts-per-step 4 --save-steps 50 --seed 42
done

# GRPO arms
for C in grpo_or grpo_mr; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -m turn_level_rewards.train \
      --condition $C --train-size 90447 --eval-size 8 --max-steps 500 \
      --num-generations 8 --seed 42
done
```

`--num-generations 8` is the GRPO group size, and the verification runs used the same value —
keep them matched. It is *not* the analogue of `--num-rollouts-per-step`: GRPO samples one prompt
many times to build its group baseline, while PPO draws distinct prompts and uses a learned
critic. So the two tracks' batch settings are not directly comparable and should not be forced to
match.

Run it under `systemd-run --user --scope --unit=<name> bash -c "..."`. A transient systemd cgroup
has killed long-running processes on this machine before (documented in
`docs/phase-5-full-training-runs.md`); the scope prevents it.

`--resume-from-checkpoint auto` resumes from the latest checkpoint, but **raises if there is no
checkpoint** — that is deliberate, so pass it only when resuming, never on a first launch.

### Evaluation

`--num-shards N --shard i` runs N independent processes over disjoint strided slices. Measured
speedup on this box is **~1.6x with 3 shards**, not 3x — per-shard rate drops from 2.07 to ~3.9
s/row under contention. Use 3 shards; more will not help.

```bash
for S in 0 1 2; do
  .venv/bin/python -m turn_level_rewards.evaluate_ppo \
    --condition $C --checkpoint-dir outputs/$C/checkpoint-500 --eval-size 2000 \
    --num-shards 3 --shard $S --output results/$C-shard$S.json &
done; wait
.venv/bin/python scripts/merge_eval_shards.py results/$C-shard*.json --output results/$C-eval.json
```

**Eval size**: 2,000 rows is recommended over the full 7,405. At n=2,000 the standard error on EM
is ~1 point, far below any gap that matters, and it turns a ~29 h `mt_ppo` eval into ~8 h. Phase
7b's full-set eval of `mt_ppo` took **29.1 hours** — budget accordingly if you use the full set.

Eval prints progress every 50 rows and writes partial metrics marked `{"complete": false}`.
`merge_eval_shards.py` refuses to merge incomplete shards.

---

## 4. What was verified before handoff (do not re-derive)

All on the real model, real retrieval server, RTX 4090:

| Check | Result |
|---|---|
| Unit tests / ruff / ty | 192 passed, all clean |
| All three arms train | 12 steps each at `rollouts=4`, rewards varying |
| `mt_ppo` 20-step re-verify | **0 skips**, peak GPU **19.75 GB** (was 24.05) |
| Answer-turn fix | **0/80** episodes ended on a tool message (was 21% / 58%) |
| Search penalty working | 0.82 searches/episode (was burning all 4 turns) |
| Eval path, all three arms | train → checkpoint → sharded eval → metrics, exit 0 |
| Hyperparameters identical across arms | verified programmatically, including 8-bit |

### 100-step verification, all three arms (completed before handoff)

```
arm      steps  skip  peakGPU  endedOnTool   format compliance    tools/ep  rewardVals
                                              (steps 0-49 → 50-99)
ppo        100    0    18.2GB     0/400      not measurable*        2.11        3
ppo_mr      99    1    23.5GB     0/396      0.61 → 0.83            1.61       14
mt_ppo     100    0    23.0GB     0/400      0.78 → 0.80            1.69       12
```

Wall-clock: `ppo` 20 min, `ppo_mr` 42 min, `mt_ppo` 36 min per 100 steps — so budget roughly
**1.7 / 3.5 / 3.0 hours** for the 500-step runs.

`*` PPO-OR's reward is binary correctness with no format term, so `format_and_outcome_reward`
cannot distinguish "wrong answer" from "no answer". `format_ok` is now logged per episode
(commit `b94b9c5`) and will be measurable for all three arms in the real runs.

**Read this honestly**: `ended_on_tool` held at 0 across all 1,196 episodes, format compliance is
far above the pre-fix ~0.48 and rising for `ppo_mr`, and reward has genuine variance. But peak
memory climbed back toward the ceiling as episodes grew (23.5 GB on `ppo_mr`, one skip). That is
~1%, and it landed on `ppo_mr` rather than `mt_ppo`, so it is not the asymmetry that invalidated
Phase 7b — but it is not zero. Treat §6's skip gate as a real check, not a formality.

---

## 5. Documented deviations from the paper

These are deliberate and identical across all three arms, so they cannot bias the comparison —
but they must be stated in any write-up, alongside the pre-existing ones in `CLAUDE.md`.

1. **`num_rollouts_per_step=4` vs the paper's total batch size of 512.** Hardware — one RTX 4090
   vs 8×H100. Pre-existing, documented in `docs/phase-7b-full-ppo-runs.md`.
2. **8-bit AdamW optimizer states** (`bitsandbytes`). New in this phase. Not for speed — for
   experimental validity. Without it `mt_ppo` OOM-skips steps the other arms do not, so the arms
   receive different numbers of real gradient updates. That confound is what invalidated the
   2026-07-25 runs (431 updates vs 499). Measured on `mt_ppo` at `rollouts=4`: 13.8% skips
   originally → 2.8% after the chunking and answer-turn fixes → **0%** with 8-bit as well.
3. **2 searches allowed vs the paper's 1** — pre-existing, see `CLAUDE.md`.

---

## 6. Gates — stop and investigate rather than burning 15 hours

1. **Skip rate must stay low and SYMMETRIC.** Check at step ~100 and again at ~300:
   `.venv/bin/python scripts/analyze_train_log.py outputs/<arm>/train_log.jsonl`
   Measured at 100 steps: `ppo` 0%, `ppo_mr` 1%, `mt_ppo` 0%. Peak memory reaches 23.0-23.5 GB
   against a ~23.5 GB ceiling once episodes have grown, so headroom is thin.
   **If the rate climbs above ~5%, or diverges sharply between arms, stop** — unequal real
   gradient-update counts are what invalidated Phase 7b (431 vs 499).
   **The fix is to chunk the critic forward pass** (`_forward_critic_values`, the last remaining
   unchunked memory term, ~1.5 GB — `gather_action_logprobs` is the worked example to copy).
   **Do NOT shrink `num_rollouts_per_step`**: that is the paper's "total batch size" (ours 4, the
   paper's 512), so shrinking it degrades gradient quality and widens the largest documented
   deviation, to fix a peak actually driven by a single long episode.
2. **`ended_on_tool` must stay 0.** It is logged per episode in `train_log.jsonl`. A nonzero rate
   means the answer-turn defect has regressed.
3. **Format compliance must rise.** If `mt_ppo` is still near 0.5 at step 150, stop — the paper
   reports 0.998, and something is still wrong.
4. **Reward must have variance.** `analyze_train_log.py` prints a `*** DEAD RUN` banner if the
   per-step mean reward has exactly one distinct value. That banner means stop immediately; it is
   the signature of a bug that once burned 177 steps silently.

Note: `CollapseMonitor`'s "Format compliance collapsed" alert is **known mis-calibrated** — it
triggers on `mean_format_reward > 0.0`, which requires >50% compliance, so it fires spuriously at
healthy rates. It fired three times on a healthy run. Treat it as noise; use the gates above
instead. Recalibrating it is an open task.

---

## 6a. What this phase can and cannot claim

This matters for the write-up; do not overstate it.

| Comparison | Valid? |
|---|---|
| `ppo` vs `ppo_mr` vs `mt_ppo` | **Yes.** Identical in every hyperparameter; only the reward components and their placement differ. This is the phase's real result. |
| MT-PPO vs the paper's Table 2 (EM 0.453 / format 0.998, HotpotQA) | **Yes** — a genuine reproduction check, subject to the deviations in §5. |
| GRPO-OR vs GRPO-MR (Phases 5-6) | **Yes**, independently. Those numbers stand (EM 0.242 vs 0.307). |
| **PPO track vs GRPO track** | **No — cross-track observation only.** |

The last row is the trap. The PPO track now uses the **paper's** rewards (binary or ±1.0 outcome,
per-turn format, λs search penalty); the GRPO track uses **this repo's** rewards (±0.1 format,
F1 + 0.5·EM outcome, 0.4 × retrieval_fraction). Those are different objectives, so any numeric
difference between the tracks conflates the algorithm with the reward function and cannot be
attributed to PPO-vs-GRPO. Report it as a qualitative observation with that caveat stated, or not
at all.

A genuine PPO-vs-GRPO comparison requires re-running the GRPO track under the paper's rewards
too. **That is now Phase 7d** (`docs/phase-7d-paper-faithful-grpo.md`), scheduled to run after
this phase and BEFORE Phase 8's judge — the paper's own Section 6.1 defines GRPO-OR/GRPO-MR
with wording identical to PPO-OR/PPO-MR, and holding the reward fixed across algorithms is
exactly what makes its Table 2 comparison valid. Do not start 7d as part of this phase.

## 6b. Scope — what this phase is NOT

- **The LLM-as-judge is out of scope.** That is Phase 8 (`docs/phase-8-llm-judge.md`), built on
  top of a working PPO track. This phase uses deterministic rewards only, which is what the
  paper's Table 2 itself uses. Do not add a judge here.
- **MT-GRPO remains out of scope** — see `CLAUDE.md`'s Goal section.
- **Do not start additional seeds.** Stop after one run of each arm plus evaluation and report
  back (§7).

## 6c. Capture this for the write-up

Collect these as you go; reconstructing them afterwards is expensive and some are not
recoverable once checkpoints rotate (`save_total_limit=3`).

**Per arm:**
- Final held-out metrics: EM, F1, retrieval_fraction, `num_examples` (`results/<arm>-eval.json`)
- If the arm collapsed: metrics for **both** the final checkpoint and the last one before
  collapse, per the paper's own methodology
- Training curves from `train_log.jsonl`: per-step reward, `retrieval_fraction`, and format
  compliance (derivable as the fraction of episodes with `format_and_outcome_reward > -1.0`)
- Skip count and peak GPU — needed to state whether the arms were memory-symmetric
- `ended_on_tool` rate — must be 0; state it as evidence the answer-turn defect stayed fixed
- Mean searches per episode (`num_tool_turns`) — the search penalty's observable effect
- Wall-clock per run

**Across arms, for the actual claims:**
- `ppo → ppo_mr` delta = value of the added reward signal
- `ppo_mr → mt_ppo` delta = value of turn-level placement (Eq. 9) — **this is the paper's claim**
- Comparison against the anchor: paper's MT-PPO on HotpotQA = EM 0.453 / format 0.998
- Comparison against this repo's GRPO track (EM 0.242 / 0.307) — note these use *different*
  rewards, so it is a cross-track observation, not a like-for-like comparison

**For the deviations section:** `num_rollouts_per_step=4` vs 512, 8-bit optimizer states, 2
searches vs 1 (§5). Also worth recording honestly: Phase 7b's results were invalidated by four
implementation defects plus a non-paper reward, and that history is part of the story — the repo
documents tradeoffs and pivots explicitly, not just working code.

## 7. After the three runs — the decision point

**Stop here and report to the user.** One run of each arm at seed 42, plus evaluation, is the
whole scope. Do not launch additional seeds — that is a decision to make together, informed by
what these three runs show. Present the results and the inputs below; the user decides.

Inputs to that decision:

- **Is the ablation clean?** Do `ppo → ppo_mr → mt_ppo` show a monotone, interpretable pattern?
- **Does MT-PPO approach the anchor** (EM 0.453 / format 0.998)? If it lands near it, the
  reproduction succeeded and extra seeds mainly tighten error bars. If it is far off, more seeds
  will not fix that — find the remaining discrepancy first.
- **Effect size vs. seed noise.** Single-seed comparisons cannot support algorithmic claims
  (see arXiv:1806.08295); n=3 is the conventional floor. But a *large* effect survives seed
  noise better than a small one, so the measured gap determines how badly seeds are needed.
- **Cost**: ~4-6 h training + ~8 h eval per additional seed across three arms. Two more seeds is
  roughly a day and a half of unattended GPU.

Note in particular that the paper reports PPO instability — *"Since PPO baselines often crash, we
evaluate them using either the final checkpoint or the last checkpoint prior to collapse"* and
*"PPO exhibits high variance and even performance degradation, especially on HotpotQA, while
MT-PPO maintains consistent improvement."* If `ppo` collapses again, that may be a **correct
reproduction**, not a bug. Evaluate both the final checkpoint and the last one before collapse,
as the paper does. `--checkpoint-dir` takes any checkpoint for exactly this reason.

---

## 8. Known-good state of the rest of the repo

- **GRPO track (Phases 5-6) is sound and its numbers stand** (EM 0.242 `outcome_only` vs 0.307
  `turn_level`). It uses TRL's `GRPOTrainer`, and the two failure modes that wrecked the PPO
  track — the eos bug and the answer-turn bug — are both absent from it. Verified in TRL source.
- **Shared code audits clean**: `metrics.py` (canonical SQuAD EM/F1), `rewards.py`, `env.py`
  (`reset()` fully reinitializes; `retrieval_fraction` dedupes correctly).
- **Do not change the GRPO track's rewards.** Its F1+EM deviation is deliberate and justified for
  GRPO specifically; the paper-faithful `paper_*` reward functions are used by the PPO track only.

## 9. Open tasks not required for the three runs

- Recalibrate `CollapseMonitor`'s format-collapse alert (§6).
- The `retrieval_fraction` anomaly: in Phase 7b, `mt_ppo`'s held-out retrieval (0.479) was *lower*
  than `ppo`'s (0.505) despite `turn_reward` rewarding retrieval. Unexplained; likely an artifact
  of turn-exhausted episodes, so **re-check whether it persists** now that the answer-turn defect
  is fixed.
- Phase 7b's remaining deliverables — matplotlib visuals (use the `dataviz` skill), README results
  section — are unchanged and still pending.
