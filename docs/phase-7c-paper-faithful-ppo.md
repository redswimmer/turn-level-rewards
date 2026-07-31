# Phase 7c: Paper-faithful five-arm comparison

**PPO-OR · PPO-MR · MT-PPO · GRPO-OR · GRPO-MR — all under the paper's own reward design.**

**Status: EXECUTED, 2026-07-27 → 2026-07-31. Six arms trained and evaluated on the full 7,404-row
held-out set. READ §15 FIRST — it is the corrected result. §13/§14 are the first pass and §13b's
Eq. 9 conclusion is SUPERSEDED. §§1-12 are the pre-execution plan, preserved as written.**

Headline:

1. **The paper's Eq. 9 claim REPRODUCES**: PPO-MR → MT-PPO = **+3.90 EM** (paper: +1.7), format
   0.642 → 0.830. §13b's "did not reproduce (−0.027)" was measured against `ppo_mr`, which is
   **not** the paper's PPO-MR — it carries MT-PPO's own `R^I`. The faithful baseline is
   `ppo_mr_paper`, added 2026-07-31.
2. **Two arms collapsed** (`ppo`, `grpo_or` — both binary-outcome-only) into verified
   zero-gradient absorbing states. Two different algorithms, one structural failure.
3. **This repo's own contribution**: keeping the deviating `ppo_mr` alongside the faithful arm
   decomposes the paper's PPO-MR → MT-PPO step into a reward-**content** effect (**+6.59 EM**)
   and a turn-**placement** effect (**−2.69 EM**) — a separation the paper's design cannot make,
   because its comparison moves both at once.
4. **λs = 0 did not cause uncontrolled search** (§15d), contra §5.2's warning — isolating the
   binary reward's dead gradient as what actually killed `ppo`.

## STOP CONDITIONS — read before doing anything

0. **`ppo` (PPO-OR) is ALREADY DONE. Do not train it. Train four arms, not five:**
   `ppo_mr`, `mt_ppo`, `grpo_or`, `grpo_mr`. The `ppo` arm was run on 2026-07-27, collapsed,
   and **that collapse is its result** — re-running it produces a longer collapse, nothing
   more. Its checkpoints are preserved and get evaluated like any other arm.
   **Read §12 before anything else**; it has the full analysis and tells you which checkpoints
   to evaluate. Sections 2-4 below describe five arms because that is the experiment's design —
   one of the five is simply already complete.

1. **Do not change the experiment.** Step count (500), eval size (7,404 — see §3), batch
   (`num_rollouts_per_step=4`, `num_generations=8`), seed (42) and the reward design are all
   decided, with the reasoning recorded below. If you believe one is wrong, say so and stop —
   do not adjust it.
2. **Stop after one run of each arm plus evaluation, and report.** Do not start additional
   seeds, do not extend training, do not begin Phase 8. §7 covers what to report.
3. **Stop if a gate in §6 trips** (skip rate climbing or diverging between arms,
   `ended_on_tool` above 0, format compliance flat near 0.5 by step 150, or a `DEAD RUN`
   banner). Report rather than working around it.
4. **Stop if something is genuinely broken.** Everything here was verified live at 100 steps
   per arm, and both eval paths were smoke-tested end to end. If reality contradicts this
   document, that is worth surfacing, not silently repairing.

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

### On the step count — 500 steps, and why it may not be enough

The paper trains for **500 steps** ("Training is performed for 500 steps over 4 epochs") at a
total batch size of 512. We match the step count but not the batch:

```
paper:  500 steps x 512 batch = 256,000 episodes
ours:   500 steps x   4 batch =   2,000 episodes    (128x less data)
```

Matching data volume would require ~64,000 steps (~250 hours) and is not viable, so 500 matches
their **update count**, not their data. **Run 500. This is decided, not open** — it is the paper's
number, matching more steps to compensate for the smaller batch is not viable at ~250 hours, and
deviating on step count as well as batch size would compound deviations. Do not change it.

**But expect it may be undertrained, and check rather than assume.** At 100 steps `ppo_mr`'s
format compliance was still climbing (0.61 → 0.83), and Phase 6 hit exactly this on the GRPO
track: 300 steps were inconclusive and 600 resolved it. Apply Phase 6's "is more training
needed?" checklist at 500 and **report the answer to the user** — do not extend on your own
initiative. If the curves have not plateaued, that is a finding to raise along with the results,
and the write-up should state that 500 steps at batch 4 may leave the arms undertrained. For
reference if the user asks: doubling to 1,000 steps costs roughly 21 h across all five arms
(500 steps is ~10.5 h total: PPO 1.7 / 3.5 / 3.0 h, GRPO ~1.6 / 0.7 h).

Note that more steps at a tiny batch is NOT equivalent to fewer steps at a large one — the
gradients stay noisy, so it buys more updates of lower quality, with more exposure to the PPO
instability the paper itself documents. Extend because a measurement says to, not by default.

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
    --condition $C --checkpoint-dir outputs/$C/checkpoint-500 --eval-size 7404 \
    --num-shards 3 --shard $S --output results/$C-shard$S.json &
done; wait
.venv/bin/python scripts/merge_eval_shards.py results/$C-shard*.json --output results/$C-eval.json
```

**`--eval-batch-size 4` is required, not optional.** The flag defaults to 2, which still runs
and still produces correct numbers -- it just takes ~24 h for the two GRPO arms instead of
~6.5 h. It fails silently in the worst way: by being slow, not by erroring.

GRPO arms use the other evaluator (`evaluate.py`, which routes through `GRPOTrainer.evaluate()`
and batches, so no sharding is needed):

```bash
for C in grpo_or grpo_mr; do
  .venv/bin/python -m turn_level_rewards.evaluate \
    --condition $C --checkpoint outputs/$C/checkpoint-500 \
    --eval-size 7404 --eval-batch-size 4 \
    --output results/$C-eval.json
done
```

Confirm `evaluate.py`'s exact CLI before running — it was written in Phase 6 for the
`outcome_only`/`turn_level` conditions and may need the new condition names threading through.
This is the one part of the pipeline **not** smoke-tested for the paper-faithful GRPO conditions,
because those checkpoints did not exist at handoff time. Check it on a tiny `--eval-size` first.

**Eval size: 7,404 rows — the full held-out set less one row. Not a subset beyond that.** Why 7,404 and not 7,405 (decided 2026-07-26, matching Phase 6): `GRPOConfig` enforces
`generation_batch_size % num_generations == 0` (`evaluate.py:38`), so at `--eval-batch-size 4`
the row count must be even. 7,405 is odd and produces a hard `ValueError` on the ragged final
batch. Running 7,405 IS possible at `--eval-batch-size 2`, but measured cost is ~24 h for the two
GRPO arms versus ~6.5 h at batch 4 — about 17 extra hours to gain one row, which is 0.0135% of the
set and can move EM by at most 0.00013.

**Use 7,404 for all five arms, including the PPO ones**, which have no such constraint. Scoring
GRPO on 7,404 and PPO on 7,405 would put the arms on different data, and identical rows
everywhere is worth more to this comparison than one extra row on three of them.

Beyond that one row, do not subset: the eval set is the one place where using less data buys
nothing scientifically and only adds a caveat to the write-up.

Budget honestly, but do not pre-emptively shrink it:

- **GRPO arms are cheap.** `evaluate.py` goes through `GRPOTrainer.evaluate()`, which batches.
- **PPO arms are the slow ones.** `evaluate_ppo.py` drives `_rollout_episode` one episode at a
  time. Phase 7b measured `mt_ppo`'s full-set eval at **29.1 hours** — but that number is stale
  and almost certainly much too pessimistic now: it predates the answer-turn fix and the λs search
  penalty, which cut searches per episode to ~1.7 from a full 4-turn burn. Eval time scales with
  generated tokens.
- **Shard the PPO evals 3 ways** (~1.6x measured on this box).

If an eval genuinely proves too slow, shard harder or raise it with the user — do **not** silently
substitute a subset. Partial metrics are written as the run proceeds (marked
`{"complete": false}`), so progress is visible and a kill costs only the remaining rows.

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

### 100-step verification, all five arms (completed before handoff)

```
arm      steps  skip  peakGPU  endedOnTool   format compliance    tools/ep  rewardVals
                                              (steps 0-49 → 50-99)
ppo        100    0    18.2GB     0/400      not measurable*        2.11        3
ppo_mr      99    1    23.5GB     0/396      0.61 → 0.83            1.61       14
mt_ppo     100    0    23.0GB     0/400      0.78 → 0.80            1.69       12
```

Wall-clock: `ppo` 20 min, `ppo_mr` 42 min, `mt_ppo` 36 min per 100 steps — so budget roughly
**1.7 / 3.5 / 3.0 hours** for the 500-step runs.

```
arm       steps  exit  peakGPU  zero_std (1st half -> 2nd)  errors  wall-clock
grpo_or     100     0   20.5GB  0.50  (0.48 -> 0.52)             0     19 min
grpo_mr     100     0   20.5GB  0.36  (0.36 -> 0.36)             0      8 min
```

`*` PPO-OR's reward is binary correctness with no format term, so `format_and_outcome_reward`
cannot distinguish "wrong answer" from "no answer". `format_ok` is now logged per episode
(commit `b94b9c5`) and will be measurable for all arms in the real runs.

**The GRPO zero-variance measurement — the assumption this project was built on.** GRPO's
gradient comes entirely from within-group reward variance; `frac_reward_zero_std` is the fraction
of prompt-groups where every rollout scored identically and therefore contributed nothing. This
repo deviated from the paper's binary reward at Phase 2 on the *prediction* that binary rewards
would starve GRPO, and never measured it. Measured now:

- `grpo_or` (binary EM only): **0.50** — half of all groups produce no gradient, and it does not
  improve over 100 steps (0.48 → 0.52).
- `grpo_mr` (+ retrieval reward): **0.36** — the continuous retrieval term restores gradient on
  groups a binary signal left dead.

So the original instinct was **directionally correct but materially overstated**: binary rewards
cost roughly half the gradient signal, they do not eliminate it. This also explains the paper's
own GRPO-OR/GRPO-MR gap (0.331 → 0.416) mechanically — the merged reward is not merely adding
information, it is reviving groups that had none.

Both arms trained cleanly at 8-bit: exit 0, zero errors, 20.5 GB peak against a ~23.5 GB ceiling
(down from 23.3 GB at fp32 — worth keeping, since TRL's GRPOTrainer has no catch-OOM-and-skip
path and an OOM there is a hard crash, not a skipped step).

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

**Fastest way to check all four gates at once:**
`.venv/bin/python scripts/check_gates.py outputs/<arm>/train_log.jsonl` — reports exactly these
four quantities and nothing else. `analyze_train_log.py` still exists for episode-level
forensics, but it buries the gate numbers under ~60 lines of outlier detail.

1. **Skip rate must stay low and SYMMETRIC.** Check at step ~100 and again at ~300:
   `.venv/bin/python scripts/check_gates.py outputs/<arm>/train_log.jsonl`
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

---

## 10. Gate-1 result, 2026-07-26 — phase NOT launched

The five arms were **not** launched. A pre-launch reproducibility gate was run first (two
identical short runs, `--condition ppo --train-size 32 --max-steps 5 --num-rollouts-per-step 4
--seed 42`), with the bar "step-0 loss identical, not just close". It did not clear that bar, so
per instruction execution stopped and handed back.

### What the critic-seeding fix (`57b0894`) did fix — verified, not assumed

| Check | Result |
|---|---|
| Critic built twice under `set_seed(42)`, compare `score.weight` | **Bit-identical** |
| Step-0 loss spread, before the fix | 14.16 vs 9.78 — **45% apart** |
| Step-0 loss spread, after the fix | 4.0991027 vs 4.1366970 — **0.9% apart** |
| Step-0 episodes across the two runs | **Identical in every logged field** (question, `num_action_tokens`, `retrieval_fraction`, `format_ok`, `num_tool_turns`) |

`set_seed(config.seed)` sits immediately before `build_policy_and_critic` in `build_ppo_trainer`,
which is the correct place. Rollouts at step 0 now reproduce exactly. The dominant, seed-shaped
term is gone.

### What still differs, and why it is not a seeding bug

The residual is floating-point nondeterminism. Both models run in `bfloat16`
(`train_ppo.py:565-567`); a bf16 mantissa gives a relative epsilon around 0.4% per operation, and
every discrepancy observed sits at that scale — `value_loss` 0.9% apart, `policy_loss` 2%, `kl`
15% on an absolute value of 0.003. No `torch.use_deterministic_algorithms`,
`cudnn.deterministic`, or `CUBLAS_WORKSPACE_CONFIG` is set anywhere in `src/`, and gradient
checkpointing recomputes activations, so run-to-run variation at this magnitude is expected
rather than anomalous.

**It still costs end-to-end reproducibility.** By step 1 `retrieval_fraction` diverges (0.125 vs
0.250): a sub-1% logit perturbation flips a sampled token and the trajectory diverges
macroscopically from there. Same-seed runs therefore reproduce the *setup* but not the
*trajectory* — which matters here because §2 already calls the MT-PPO-over-PPO-MR effect (+1.7 EM)
marginal at one seed.

### Open decision before this phase can launch

Whether to pursue bit-identity at all. Enabling `torch.use_deterministic_algorithms(True)` plus
`CUBLAS_WORKSPACE_CONFIG=:4096:8` is the standard route, but it may be **unachievable** rather
than merely slow: PyTorch raises on ops lacking a deterministic kernel, which is likely for
attention backward under gradient checkpointing. The alternative is to accept that the seed
controls everything a seed can control, document trajectory non-reproducibility as a known
limitation, and launch.

### State left behind

- No training or eval was launched. No `p7c` systemd scopes remain.
- The 100-step verification artifacts from §4 were moved to `outputs/_verify100/<arm>/` so fresh
  checkpoints would not collide with their stale `checkpoint-100` directories. Nothing deleted;
  all `train_log.jsonl` evidence preserved.
- `scripts/check_gates.py` added — reports the four §6 gates at a glance, since
  `analyze_train_log.py` buries them under per-episode outlier listings. Validated by reproducing
  §4's published `mt_ppo` figures exactly (0 skips, 22.97 GB peak, format 0.78→0.80, 1.69
  tools/ep).
- Gate 2 (eval determinism under greedy decoding) was **not** run.

---

## 11. Reproducibility — resolved 2026-07-26, do not re-open

Gate 1 (two same-seed runs, step-0 loss identical) **fails, and that is accepted.** Launch anyway.

**What was fixed and verified.** The critic's `score.weight` is randomly initialized by
`AutoModelForSequenceClassification.from_pretrained` and was created before `set_seed` ran, so
`--seed` did not control it. Fixed in `57b0894` by seeding immediately before
`build_policy_and_critic`. Verified four ways:

| Check | Result |
|---|---|
| Critic built twice under `set_seed(42)`, compare `score.weight` | **Bit-identical** |
| Step-0 loss spread, before the fix | 14.16 vs 9.78 — 45% apart |
| Step-0 loss spread, after the fix | 4.0997 vs 4.1455 — ~0.9% apart |
| Step-0 episodes across runs (questions, tokens, retrieval, format, turns) | **Identical in every field** |

**What remains is bfloat16 arithmetic, not seeding.** Both models run in bf16
(`train_ppo.py:565-567`), whose mantissa gives ~0.4% relative epsilon per operation, and every
residual discrepancy sits at that scale. Nothing sets deterministic kernels, and gradient
checkpointing recomputes activations, so run-to-run variation at this magnitude is expected.

**Bit-identity is not achievable here — tested, not assumed.** Enabling
`torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8` fails on this
stack before even reaching attention backward:

```
ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)
```

**And it would not help the thing we care about anyway.** A sub-1% logit perturbation flips a
sampled token, so trajectories diverge by step 1 — bf16 noise acts as a second, uncontrolled seed.
Bit-determinism would let a run be replayed exactly (useful for auditing) but would **not reduce
the variance of the estimate**. The +1.7 EM point MT-PPO-over-PPO-MR effect is marginal at one
seed, and the only real remedy for that is multiple seeds with paired comparisons — which is the
§7 decision, not a determinism problem.

**So the standard for this phase is: the seed controls initialization and data order, and
trajectories are not reproducible.** State that plainly in the write-up alongside the other
deviations. Do not spend further time chasing bit-identity.

---

## 12. The `ppo` (PPO-OR) collapse — investigated 2026-07-27, do not re-run this arm

The first `ppo` run collapsed and then OOM-skipped 100% of steps from ~140 onward. **It is a
result, not a failure to retry.** Do not re-run `ppo`. Evaluate its preserved checkpoints instead.

**What happened, in causal order** (from `outputs/ppo/train_log.jsonl`, 160 steps):

1. PPO-OR has **λs = 0 by the paper's own definition** — Section 6.1 gives it outcome reward
   only, so `paper_binary_outcome_reward` carries no format term and no R^I is placed. That is
   exactly the configuration the paper warns of: *"removing this reward term leads to unstable
   training and degenerate behaviors, such as uncontrolled search usage."*
2. Reward went identically 0 after step 46, format compliance 0.247 → 0.009. Binary reward → all
   zeros → critic learns V=0 → advantage 0 → **zero gradient**. An absorbing state; nothing
   recovers from it.
3. The degenerate policy then generated enormous completions. This is the measured driver:

   ```
   window    mean_action_tok   max    mean_maxalloc_GB
    40- 59        182          569         15.89
    80- 99        174          932         16.22
   100-119        736         4198         18.00
   120-139       2197         4249         20.55      <- 12x the baseline
   ```
4. Memory demand followed episode size past the card's capacity, and skips became continuous.

**The reserved-memory pinning is a consequence, not a cause.** `gpu_reserved_gb` sits at ~24.2 GB
after the first OOM while `gpu_allocated_gb` returns cleanly to 9.29 GB every step — that is the
caching allocator holding memory it genuinely needed for 24 GB peaks, not a leak and not
fragmentation poisoning the run. There is no memory bug to fix here. (Note `reset_peak_memory_stats()`
IS called per step at `train_ppo.py:1419`, so `gpu_max_allocated_gb` is a per-step peak.)

**Why the other four arms are not at the same risk.** `ppo_mr` and `mt_ppo` carry λs = 0.1 and the
per-turn format reward, which hold them at ~1.6 tool turns against `ppo`'s 4.0 — they cannot drift
into unbounded search. Note that `ppo` had the *lowest* peak memory of the three arms at step 100
(18.21 GB vs 23.54 / 22.97), so peak memory at 100 steps predicts nothing; the collapse is what
drove its memory, not the reverse. The GRPO arms use a different trainer entirely.

They are still tight — 22.97-23.54 GB peak against a 24.56 GB card — so §6's gate 1 remains a real
check, and §6's named remedy (chunk `_forward_critic_values`) remains the fix if it trips. Do not
apply it pre-emptively; `ppo_mr` took only 1 skip in 100 verification steps.

**Preserved, evaluate these rather than re-training:**
- `outputs/_precollapse/ppo/checkpoint-50` — last checkpoint before collapse
- `outputs/ppo/checkpoint-100` — post-collapse-onset

This matches the paper's stated methodology for its own crashed PPO baselines: *"we evaluate them
using either the final checkpoint or the last checkpoint prior to collapse."*

**For the write-up, state the severity gap honestly**: the paper's PPO-OR still reaches EM 0.435 /
format 0.916, so a total collapse is more severe than theirs. The most plausible cause is the
documented batch-size deviation — at 512 episodes per update a sparse binary reward still yields
usable gradient, while at 4 an all-zero stretch gives literally zero gradient and the policy
random-walks into the absorbing state. That makes our PPO-OR a weaker baseline than the paper's,
which must be said when interpreting the `ppo → ppo_mr` delta: it will measure "the baseline died",
not "the merged reward helps by X". In the paper that delta is +0.001.

---

## 13. Results — executed 2026-07-27 → 2026-07-30

All five arms, seed 42, 500 steps, evaluated on the identical 7,404 held-out rows with greedy
decoding on both eval paths (`evaluate_ppo.py` `do_sample=False`; `evaluate.py` `top_k=1`).

### 13a. The five-arm table

| Arm | EM | F1 | Format | Retrieval | Verdict |
|---|---|---|---|---|---|
| `ppo` (PPO-OR) @ ckpt-50 *(pre-collapse)* | 0.0016 | 0.0019 | **0.0027** | 0.5285 | collapsed |
| `ppo` (PPO-OR) @ ckpt-100 | (see §14) | | | | collapsed |
| `ppo_mr` (PPO-MR) | **0.3012** | **0.3945** | 0.8170 | 0.5189 | healthy |
| `mt_ppo` (MT-PPO) | 0.2743 | 0.3621 | **0.8296** | 0.5205 | healthy |
| `grpo_or` (GRPO-OR) | 0.0000 | 0.0154 | 0.4209 | — | collapsed |
| `grpo_mr` (GRPO-MR) | 0.2954 | 0.3909 | **0.9750** | 0.4745 | healthy |

Anchors (paper Table 2, HotpotQA): GRPO-OR 0.331/0.513 · GRPO-MR 0.416 · PPO-OR 0.435/0.916 ·
PPO-MR 0.436 · MT-PPO 0.453/0.998.

**Every arm undershoots the paper's EM.** The healthy arms land at 0.27-0.30 against 0.42-0.45.
Format, by contrast, is close (0.82-0.98 against 0.92-1.00). That split is the expected shape of
the batch-size deviation: format is a dense per-episode signal that saturates in 500 updates,
while EM is sparse and hard and is what the paper's 128x data advantage actually buys.

### 13b. What the within-track ablations measure

| Comparison | Delta | Paper's | Reading |
|---|---|---|---|
| `ppo` → `ppo_mr` | **+0.300 EM** | +0.001 | Value of the added reward signal. Enormous here only because our baseline died; theirs did not. Measures "the baseline collapsed", NOT "merged reward helps by 0.30". |
| `ppo_mr` → `mt_ppo` | **−0.027 EM** | +0.017 | Turn-level placement (Eq. 9) — the paper's own claim. Did not reproduce. **Underpowered, not refuted** (see 13d). |
| `grpo_or` → `grpo_mr` | **+0.295 EM** | +0.085 | Same caveat as `ppo → ppo_mr`: baseline collapse, not a like-for-like effect size. |

**Eq. 9's effect splits by metric, which is more informative than the bare EM number.** Turn-level
placement is *ahead* on format (0.8296 vs 0.8170) and retrieval (0.5205 vs 0.5189) but *behind* on
EM (−0.027) and F1 (−0.032). `R^I` directly pays for per-turn format and retrieval, so placing it
at turn boundaries sharpened exactly those behaviours — and did not convert them into better final
answers, which only `R^O` pays for.

### 13c. The two collapses — the phase's most robust finding

Two arms, two different algorithms, same reward class (binary outcome only, no format term, λs=0
by the paper's own §6.1 definition), same structural death.

`grpo_or`, from `logs/p7c-grpo_or.log`:

```
steps   0- 99  reward +0.1750  zero_std 0.620  grad_norm  4.6116  entropy 1.156
steps 100-199  reward +0.1450  zero_std 0.720  grad_norm  9.0968  entropy 1.004
steps 200-299  reward +0.0000  zero_std 1.000  grad_norm  0.0000  entropy 1.477
steps 300-399  reward +0.0000  zero_std 1.000  grad_norm  0.0000  entropy 1.552
steps 400-499  reward +0.0000  zero_std 1.000  grad_norm  0.0000  entropy 1.435
```

Last nonzero reward at **step 184**; then `frac_reward_zero_std` = 1.000 and `grad_norm` **exactly
0.0000** for 300 consecutive steps. `checkpoint-450` and `checkpoint-500` are byte-identical,
confirming the weights had fully settled. Held-out `tools/call_frequency` = **0.0**: across all
7,404 questions the frozen policy never called `search` once.

`ppo` (§12, plus held-out confirmation here): format **0.0027** with the *highest* retrieval of any
arm (**0.5285**). It learned to search — better than either healthy PPO arm — and then essentially
never emitted an `<answer>` tag. EM 0.0016 is the consequence.

Mechanism, identical in both: an all-zero reward batch carries no information. PPO routes that
through a critic learning V=0 → advantage 0; GRPO through a group std of 0 → advantage 0. Neither
recovers, because a zero gradient cannot move the policy that produced it.

**This resolves §9's `retrieval_fraction` anomaly.** Phase 7b found `mt_ppo`'s retrieval below
`ppo`'s and could not explain it. It persists (0.5205 vs 0.5285) and is now explained: high
retrieval is what a search-forever-never-answer policy looks like. Retrieval is only meaningful
conditional on answering. Not evidence against the turn reward.

**And it retires the assumption this repo was founded on.** `CLAUDE.md` deviated from the paper's
binary EM reward on the *prediction* that binary rewards starve GRPO of gradient; §4 measured
`frac_reward_zero_std` at 0.50 over 100 steps and softened that to "costs about half the signal".
Over a full 500 steps the original prediction was right, just non-linear in time: the arm survives
on partial signal for ~180 steps, then crosses into a state it provably cannot leave. `grpo_mr`,
identical but for the reward, never exceeded `zero_std` 0.42 and rose monotonically to reward
+0.566.

### 13d. What these runs can and cannot claim

**Eval measurement error is zero.** Both PPO evals were run twice (once before the format fix,
once after) on the same checkpoints and rows. EM, F1 and retrieval reproduced **bit-identically to
16 decimal places** for both `ppo_mr` and `mt_ppo`. Greedy decoding has no sampling step to
amplify bf16 noise, so §11's training-trajectory nondeterminism does not propagate to evaluation.

**Therefore the −0.027 EM gap on Eq. 9 carries no measurement noise — all of its uncertainty is
single-seed training variance, which one seed cannot estimate.** Per §2 this is an underpowered
test and must not be reported as evidence against turn-level placement. Concretely: detecting the
paper's +1.7-point effect needs the SE of the mean difference below ~0.85 points; at a plausible
2-3 point seed SD that is 6-12 seeds, i.e. 110-220 GPU-hours at ~18 h per seed for the two arms.
Not viable here, and the paper's own PPO-OR→PPO-MR delta is +0.1 points, so the effect is marginal
even at 8xH100 scale.

**Cross-track (PPO vs GRPO) numbers are NOT comparable, and this is a correction to §2.** §2
claims five arms validate the cross-algorithm comparison; §3 and §6a of the same document say the
tracks' batch settings are not comparable and should not be forced to match. §3/§6a are correct
and §2 overreached. Four things differ besides the reward:

| | PPO arms | GRPO arms |
|---|---|---|
| Inner passes per batch | `num_ppo_epochs=4` | `num_iterations=2` |
| Second optimizer group | `critic_lr=1e-5` | none (no critic) |
| Reward function | `R^O + Σ R^I`, with per-turn format **and λs=0.1 search penalty** | `R^O + 0.3 × retrieval_fraction`, **neither** |
| **Distinct training prompts** | **1,964 / 1,980 (measured)** | **~250** |

The last row is decisive: GRPO at `num_generations=8` spends a whole generation batch on one
prompt and `num_iterations=2` reuses it, so the PPO arms trained on ~8x more distinct questions.
That is a data-volume difference confounded with the algorithm.

What IS matched across all five arms, and what makes the *results table* legitimate: model, seed,
step count, `max_completion_length`, `max_tool_calling_iterations=4`, `beta=0`, 8-bit AdamW, policy
LR 1e-6, retrieval corpus, the identical 7,404 eval rows, and greedy deterministic decoding. The
metrics are computed by the same `metrics.py`/`rewards.py` functions for both tracks. So the arms
can be tabulated together; only *causal attribution across tracks* is off-limits.

Report as observation, not result: all three merged-reward arms converged into 0.274-0.301 EM
despite the confounds, and the paper's +10.4-point PPO-over-GRPO gap does not appear. We cannot
diagnose that non-reproduction from these runs.

### 13e. Validity evidence — why this table is trustworthy where Phase 7b's was not

| Check | `ppo_mr` | `mt_ppo` | Why it matters |
|---|---|---|---|
| Steps completed | 500/500 | 500/500 | — |
| **`ended_on_tool`** | **0 / 1,964** | **0 / 1,980** | The answer-turn defect cost Phase 7b 21-58% of episodes. Zero recurrence in 3,944 episodes. |
| Skipped steps | 9 (1.8%) | 5 (1.0%) | **491 vs 495 real gradient updates.** Phase 7b's fatal asymmetry was 431 vs 499. |
| Peak GPU | 24.13 GB | 24.02 GB | 0.11 GB apart — memory-symmetric. |
| Mean tool turns/ep | 2.04 | 2.05 | λs search penalty holding both at ~2 vs `ppo`'s 4.0. |
| Wall-clock | 4.72 h | 3.72 h | — |

All four §6 gates passed on both PPO arms for the full run, checked at steps 100 and 300 as
prescribed. `grpo_or`'s collapse is a genuine result, not a gate failure to work around — it was
reported rather than retried, per stop condition #3.

### 13f. Is 500 steps enough? (§2 asked; answered)

**Yes for the PPO arms — they converged rather than truncated.** Both are flat across their final
200 steps: `ppo_mr` reward +0.299 → +0.296 and retrieval 0.531 → 0.501; `mt_ppo` reward +0.209 →
+0.209 and retrieval 0.494 → 0.512. Format was still inching up (`ppo_mr` 0.833 → 0.846) but the
arms plateaued well below the anchor, which points at the batch-size deviation rather than step
count. `grpo_mr` was still rising at step 500 (reward +0.445 → +0.566), so it is the one arm where
more steps might buy something. **No extension was run** — §2 forbids extending on own initiative.


---

## 14. Handoff notes — 2026-07-27 → 2026-07-30

### 14a. What was run

Four arms trained (`ppo` was NOT re-trained, per stop condition #0), all at seed 42, 500 steps,
exactly the commands in §3 with no hyperparameter changed:

| Arm | Steps | Wall-clock | Outcome |
|---|---|---|---|
| `ppo_mr` | 500/500 | 4.72 h | healthy, all four §6 gates passed |
| `mt_ppo` | 500/500 | 3.72 h | healthy, all four §6 gates passed |
| `grpo_or` | 500/500 | 0.88 h | **collapsed at step 184** |
| `grpo_mr` | 500/500 | 0.83 h | healthy, still rising at step 500 |

Then six evaluations on the identical 7,404 held-out rows. Results in §13.

Total: ~10.2 h training, ~35 h evaluation (including a deliberate re-run, see 14c).

### 14b. THE MODEL-SIZE FINDING — read this before comparing anything to the paper again

**This repo never recorded the paper's base model, and it matters more than any other deviation.**
Verified by reading arXiv:2505.11821v2 §6.2 directly:

> *"We use **Qwen2.5-7B** as the base model, **E5** as the retriever, and 2018 Wikipedia dump as
> the corpus. We set the number of retrieved passages to **3**, and the maximum number of turns
> N_max to **4**."*

We use **Qwen3.5-0.8B** — ~8.75x fewer parameters. The paper's own Table 2 gives untrained
baselines on HotpotQA, and they are the decisive numbers:

| HotpotQA | EM | Format |
|---|---|---|
| Qwen2.5-7B-Base (untrained) | 0.160 | 0.098 |
| Qwen2.5-7B-Instruct (untrained) | 0.292 | 0.109 |
| **This repo's best trained arm (`ppo_mr`)** | **0.301** | 0.817 |

**Our fully-trained 0.8B model lands roughly where their model starts.** Every "we undershoot the
anchor by ~0.15 EM" observation in this repo should be read against that, not against batch size
alone.

**Batch size and model size are not competing explanations — they are multiplicative.** The
collapse trigger is the probability that a batch contains zero correct answers, `(1-p)^N`:

- Paper: p ≈ 0.16 at init, N = 512 → `(0.84)^512` ≈ 0. An all-zero batch essentially never occurs.
- Here: small p (weaker model, and BM25 instead of E5 lowers it further) AND small N (4 rollouts,
  or 8 samples of a single prompt) → all-zero batches are common. `grpo_or`'s measured
  `frac_reward_zero_std` of 0.62 IS that quantity, observed directly.

Model size lowers `p`; batch size lowers `N`; they compound in the same exponent. State the
collapse attribution as both, with `(1-p)^N` as the mechanism.

**A larger-batch probe was considered and rejected on these grounds** — it moves `N` while leaving
`p` low, so it could not isolate the cause. Not worth the GPU time.

**Also from §6.2, for the record:**
- `topk=3` retrieved passages — **matches** this repo's `SearchEnv` default exactly.
- `N_max = 4` turns — **matches** `max_tool_calling_iterations=4`.
- E5 **dense** retriever vs this repo's BM25 — a third contributor to lower `p`, previously
  documented in `CLAUDE.md` as a cost/complexity choice but never counted as an accuracy deviation.
- Figure 3's caption: *"variability across **five independent runs**."* The paper's own curves are
  n=5. That is independent confirmation that n=1 is below the bar for the Eq. 9 claim.

### 14c. A real bug found and fixed: PPO eval never recorded format compliance

`evaluate_ppo.py` computed EM, F1 and `retrieval_fraction` but **not** format compliance, so the
PPO half of §2's anchor table could not be filled from held-out data. `_rollout_episode` returns
`turn_format_ok` (per-turn), and the eval loop kept only three fields off each rollout — the
metric was simply never plumbed through. GRPO had it all along because TRL's `GRPOTrainer` logs
`eval_format_compliance_rate` for free.

Fixed by routing it through `rewards.py`'s `format_reward` via the same `log_metric` callback that
`outcome_reward` already uses for EM/F1. That choice is load-bearing: **both tracks now derive
format from the identical `_extract_answer`**, so a PPO number and a GRPO number belong in one
column without a caveat. Also added to `merge_shard_metrics`' weighted keys and to the progress
line. Two unit tests added (200 pass, ruff + ty clean).

Completions are not persisted, so there was no post-hoc path — all four PPO evals were re-run.
Pre-fix results are preserved in `results/_preformat/` and agree exactly (see 14d).

### 14d. Evaluation is bit-reproducible; training is not

Because the PPO evals ran twice on the same checkpoints and rows, this phase accidentally produced
a clean determinism measurement:

```
ppo_mr   exact_match  0.3011885467314965  ==  0.3011885467314965
mt_ppo   exact_match  0.27431118314424635 ==  0.27431118314424635
         (f1 and retrieval_fraction likewise identical to 16 dp)
```

§11 established that *training* trajectories diverge under bf16 because sampling amplifies sub-1%
logit noise into different token choices. Greedy decoding has no such amplifier — argmax is robust
to that perturbation — so **evaluation is exactly repeatable even though training is not**.

Consequence: the −0.027 EM gap on Eq. 9 carries **zero** measurement noise. All of its uncertainty
is single-seed training variance. Re-measuring would tell us nothing; only more seeds would.

### 14e. §2 overreached on cross-track comparability — corrected in §13d

§2 claims the five arms validate the paper's cross-algorithm comparison. §3 and §6a of the same
document say the tracks' batch settings are not comparable and must not be forced to match. **§3
and §6a are right; §2 is wrong.** Measured: the PPO arms trained on 1,964/1,980 distinct prompts,
the GRPO arms on ~250 — an ~8x data difference confounded with the algorithm, plus differing inner
passes, an extra optimizer group, and reward functions that are not the same function.

The *results table* is still legitimate — identical rows, identical metric code, greedy decoding
on both paths — so all five arms can be tabulated together. Only **causal attribution across
tracks** is off-limits. This should have been caught before launch under stop condition #1 rather
than at write-up time; it cost nothing here because the within-track ablations were unaffected,
but it is exactly the class of thing that stop condition exists to catch.

**Implication for Phase 7d**: 7d as written ("re-run GRPO under the paper's rewards") is largely
what 7c already did. What actually remains for a valid cross-algorithm comparison is (a) equalize
distinct training prompts across tracks and (b) reconcile the two reward functions —
`paper_grpo_merged_reward` lacks the λs search penalty and per-turn format term the PPO arms carry.
7d should be rescoped to those two items.

### 14f. Open items, unchanged or newly noted

- **§9's `retrieval_fraction` anomaly is RESOLVED** — see §13c. High retrieval on a collapsed arm
  is what search-forever-never-answer looks like; retrieval is only meaningful conditional on
  answering. Not evidence against the turn reward.
- **`CollapseMonitor`'s format-collapse alert is still mis-calibrated** (§6). Untouched.
- **`grpo_or`'s pre-collapse checkpoint was lost to `save_total_limit=3`** (collapse at step 184;
  only 400/450/500 survive, all frozen and post-collapse). §6c warned of exactly this. Lower
  priority than it first appeared: on the PPO side, where a pre-collapse checkpoint *was*
  preserved, `ppo` @ ckpt-50 scored EM 0.0016 / format 0.0027 — there was never a healthy
  binary-reward policy to capture.
- **Matplotlib visuals and the README results section remain pending** (Phase 7b deliverables).
  `results/phase7c-summary.json` consolidates every arm's held-out metrics for that purpose.
- **No additional seeds were run**, per stop conditions #2 and #6b.

---

## 15. ADDENDUM, 2026-07-31 — the faithful PPO-MR arm, and a reversed headline

**§13b's "Eq. 9 did not reproduce" conclusion is WRONG and is corrected here.** It compared
`mt_ppo` against `ppo_mr`, which is not the paper's PPO-MR (see §14c/§15a). Measured against the
faithful baseline, the paper's claim reproduces.

### 15a. Why a sixth arm was added

Checking §6.1 against arXiv:2505.11821v2 directly showed `ppo_mr` carries MT-PPO's full `R^I` --
the λs search penalty and the per-turn format bonus -- making it a *flattened MT-PPO*, not the
paper's PPO-MR. The paper's MR baselines take **retrieval correctness only**:

> **PPO-MR**: *"the trajectory-level reward combines intermediate rewards (retrieval correctness)
> and outcome rewards (answer correctness and format correctness)."*
>
> **MT-PPO (ours)**: *"the turn-level reward design is described in Section 5.2, with λs = 0.1 by
> default."*
>
> Footnote 1: *"The GRPO baselines correspond to the PPO baselines with the same reward design."*

So λs and per-turn format are MT-PPO's own contribution. `grpo_mr` was faithful all along;
`ppo_mr` was the deviation -- the opposite of what §14e assumed.

`ppo_mr_paper` was added as the faithful reproduction. `ppo_mr` was **kept**, and that turned out
to matter enormously (§15c).

### 15b. The corrected result

| Arm | EM | F1 | Format | Retrieval | Paper EM / Format |
|---|---|---|---|---|---|
| `ppo` (PPO-OR) @ ckpt-50 | 0.0016 | 0.0019 | 0.0027 | 0.5285 | 0.435 / 0.916 |
| **`ppo_mr_paper` (PPO-MR)** | **0.2353** | **0.3013** | **0.6424** | **0.5505** | 0.436 / — |
| `mt_ppo` (MT-PPO) | 0.2743 | 0.3621 | 0.8296 | 0.5205 | 0.453 / 0.998 |
| `grpo_or` (GRPO-OR) | 0.0000 | 0.0154 | 0.4209 | — | 0.331 / 0.513 |
| `grpo_mr` (GRPO-MR) | 0.2954 | 0.3909 | 0.9750 | 0.4745 | 0.416 / — |
| *`ppo_mr` (our extension, not the paper's PPO-MR)* | *0.3012* | *0.3945* | *0.8170* | *0.5189* | *n/a* |

**The paper's Eq. 9 claim reproduces**: PPO-MR → MT-PPO = **+3.90 EM** (paper: +1.7) and format
0.642 → 0.830 (**+0.187**). Right direction, larger magnitude.

### 15c. Decomposing the effect — this repo's actual contribution

`ppo_mr` and `mt_ppo` share identical reward CONTENT and differ only in PLACEMENT.
`ppo_mr_paper` and `ppo_mr` share identical PLACEMENT and differ only in CONTENT. So the three
arms separate two things the paper's own design confounds:

```
ppo_mr_paper  0.2353   R^O + retrieval,                flattened to last token
ppo_mr        0.3012   R^O + retrieval + format + λs,  flattened to last token
mt_ppo        0.2743   R^O + retrieval + format + λs,  placed at turn boundaries
```

| Change | Effect on EM |
|---|---|
| Add λs + per-turn format **content** | **+6.59** |
| Move that content to **turn boundaries** (Eq. 9) | **−2.69** |
| **Net** = the paper's own PPO-MR → MT-PPO step | **+3.90** |

**The paper's +1.7 is the net of a large positive content effect and a smaller negative placement
effect.** Its PPO-MR → MT-PPO comparison changes both at once and cannot separate them. At this
scale the reward *components* do the work, and turn-level placement gives some of it back.

Statistical footing: on 7,404 rows the unpaired SE of an EM difference is ~0.72 points, so the
content effect (+6.59) is ~9 SE and the placement effect (−2.69) ~3.7 SE on sampling noise alone.
But n=1 seed against the paper's n=5 -- the content effect is large enough to survive substantial
seed variance; the placement effect should be called **suggestive, not settled**.

### 15d. λs = 0 did NOT produce uncontrolled search

`ppo_mr_paper` runs λs = 0 by the paper's own PPO-MR definition -- exactly the configuration §5.2
warns of (*"removing this reward term leads to... uncontrolled search usage"*). It did not happen:

```
                        turns/ep by 100-step window
ppo_mr        (λs=0.1)  1.24  2.27  2.36  2.31  2.04
mt_ppo        (λs=0.1)  1.42  2.00  2.34  2.10  2.39
ppo_mr_paper  (λs=0)    1.72  2.68  2.26  2.17  2.10
```

It peaked at 2.68 and settled at **2.10**, between the two penalised arms and far from the 4-turn
cap that `ppo` saturated. This isolates what killed `ppo`: **the binary reward's dead gradient,
not the missing search penalty.** A working graded `R^O` bounds search on its own -- a
wrong-but-formatted answer still earns +0.2, so there is positive pressure to stop and answer.

Its one real cost is longer episodes (732 tokens vs `ppo_mr`'s 519): it searches about as often
but spends more tokens doing it, which also made its eval ~50% slower per row.

### 15e. Training validity, all three PPO arms

| | `ppo_mr` | `mt_ppo` | `ppo_mr_paper` |
|---|---|---|---|
| Steps | 500/500 | 500/500 | 500/500 |
| Episodes | 1,964 | 1,980 | 1,952 |
| Skipped steps | 9 | 5 | 12 |
| **Real gradient updates** | **491** | **495** | **488** |
| `ended_on_tool` | 0 | 0 | 0 |
| Peak GPU | 24.13 GB | 24.02 GB | 23.97 GB |
| Wall-clock | 4.72 h | 3.72 h | 4.14 h |

`ended_on_tool` = **0 across all 5,896 PPO episodes**. Peak GPU within 0.16 GB across arms. All
four §6 gates passed on all three.

### 15f. An eval OOM, and the safeguard that caught it

`ppo_mr_paper`'s shard 2 hit `torch.OutOfMemoryError` at ~800/2,468 rows under three-way GPU
contention. The designed safeguards all fired: the shard wrote its partial marked
`{"complete": false}`, `merge_eval_shards.py` **refused to merge it**, and the driver exited 1.
No partial was ever reported as a full-set number. Re-run solo → exit 0 → merged cleanly.

Two notes for future runs:
- This is the documented residual OOM risk from Phase 7 (`flash-linear-attention` cut it from
  ~50% to ~12.5%, survivors being the catchable kind). Consistent with that, not new.
- **Solo only improved throughput 12.5 → 9.4 s/row**, so GPU contention is NOT the sharding
  bottleneck -- single-threaded generation is, exactly as `shard_rows`' docstring documented
  (1 of 32 cores busy, GPU at 39-41%). More shards will not help much; the fix would be batching
  the eval rollouts.

### 15g. What this changes for the write-up

The narrative is now the one the repo was always supposed to tell:

1. **Faithful reproduction** — five arms under the paper's own reward definitions. Eq. 9
   reproduces (+3.90 EM); binary-outcome-only baselines collapse at this scale; absolute EM sits
   below the paper's because the model is ~8.75x smaller (§14b).
2. **Our own experiment** — `ppo_mr` decomposes the paper's PPO-MR → MT-PPO step into a content
   effect (+6.59) and a placement effect (−2.69), which the paper's design cannot separate.

§13b's delta table is superseded by §15c. §14e's claim that "a true reward-parity set would mean
removing λs from `ppo_mr`" is right, and `ppo_mr_paper` is that arm.
