# Phase 7c: Paper-faithful PPO — PPO-OR vs PPO-MR vs MT-PPO

**Status**: implementation complete, smoke-tested, verified to 100 steps. **Ready to launch the
three full runs.** Nothing is half-finished; everything described here is committed.

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

Three arms, **seed 42**, 500 steps each. Identical in every hyperparameter; they differ only in
which reward components exist and where they land.

| Arm | Paper's definition (§6.1, verbatim) | Implementation |
|---|---|---|
| `ppo` (**PPO-OR**) | *"vanilla PPO trained with only outcome rewards, where the trajectory-level reward is a binary signal indicating final-answer correctness"* | `paper_binary_outcome_reward` (1.0/0.0, **no format term**) at the last token |
| `ppo_mr` (**PPO-MR**) | *"...merged intermediate and outcome rewards, where the trajectory-level reward combines intermediate rewards (retrieval correctness) and outcome rewards (answer correctness and format correctness)"* | `R^O + sum(R^I)`, **all** at the last token |
| `mt_ppo` (**MT-PPO**) | Same components, turn-level placement via Eq. 9, λs = 0.1 | `R^I` at each intermediate turn boundary, `R^O` at the last token |

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

### Anchor target

Paper's MT-PPO on HotpotQA (Table 2): **EM 0.453, format correctness 0.998**. This is the number
to judge success against. Phase 7b's MT-PPO was EM 0.123 / format ~0.48.

---

## 3. How to run it

**Prerequisite**: the retrieval server must be up. Verify with:

```bash
curl -s -X POST http://localhost:8000/retrieve -H 'Content-Type: application/json' \
  -d '{"queries":["Knox County Regional Airport"],"topk":1,"return_scores":false}' | head -c 200
```

If it is not running, see `docs/phase-1-retrieval-infra.md` for the launch command.

### Training (~4-6 h total, sequential — the GPU fits one run at a time)

```bash
for C in ppo ppo_mr mt_ppo; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -m turn_level_rewards.train_ppo \
      --condition $C --train-size 90447 --max-steps 500 \
      --num-rollouts-per-step 4 --save-steps 50 --seed 42
done
```

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

**100-step verification of all three arms was in progress at handoff time.** Check
`outputs/{ppo,ppo_mr,mt_ppo}/train_log.jsonl` — if those directories contain 100-step logs, read
the skip counts before launching. See §6.

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

1. **Skip rate must stay ~0.** Check at step ~100 and again at ~300:
   `.venv/bin/python scripts/analyze_train_log.py outputs/<arm>/train_log.jsonl`
   Any nonzero rate that differs between arms reintroduces the confound that invalidated Phase
   7b. Peak GPU is ~19.75 GB against a ~23.5 GB ceiling, so there is ~3.75 GB of headroom — real,
   but not unlimited. `mt_ppo`'s episodes previously tripled in length during training; the search
   penalty should now prevent that, but verify rather than assume.
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
