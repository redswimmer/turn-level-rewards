# Outcome vs. Turn-Level Reward for Multi-Turn Search Agents

## Goal

Simplified reproduction of one specific ablation from **"Reinforcing Multi-Turn Reasoning in LLM
Agents via Turn-Level Reward Design"** (arXiv:2505.11821) — namely its **Appendix E case study**,
stated in the paper's own terms: **`GRPO-OR` vs `GRPO-MR`**.

- **`GRPO-OR`** (Outcome Reward; this repo's `--condition outcome_only`): reward =
  format-compliance + final-answer correctness (EM/F1). Sparse, terminal-only signal.
- **`GRPO-MR`** (Merged Reward; this repo's `--condition turn_level`): reward = format + outcome
  + a bonus for surfacing a real supporting-fact passage during search. Same trajectory shape,
  denser signal — but still summed into **one trajectory-level scalar**, scored by GRPO's
  standard, unmodified group-relative advantage (Eq. 4 in the paper).

Both conditions use the **same multi-turn agent** (same tool-calling mechanics, same max search
turns) and the **same RL algorithm** (plain GRPO, no modifications). The only variable is which
reward functions are summed. This is *not* a single-turn vs. multi-turn comparison — it's sparse
vs. dense reward on top of an identical multi-turn architecture and identical advantage
estimation, which is exactly what the paper's own `GRPO-OR`/`GRPO-MR` ablation isolates.

### Beyond the GRPO-OR/GRPO-MR comparison: PPO/MT-PPO (Phases 7-8, committed) and MT-GRPO (still out of scope)

The paper proposes two further algorithms beyond `GRPO-OR`/`GRPO-MR`. One of them is now part of
this repo's real, committed roadmap — not a maybe:

- **`PPO` / `MT-PPO`** — the paper's actual best-performing, most-benchmarked method (all of
  Table 2's real numbers are PPO/MT-PPO, not MT-GRPO). **This is Phases 7-8 of this repo's
  roadmap, not out of scope.** Re-checked directly (not assumed stale): TRL's `PPOTrainer`
  (`trl.experimental.ppo`) still has **no multi-turn tool-calling support**, confirmed fresh
  against both the installed version and the current dev branch — recent TRL commits added
  tool-calling to `GRPOTrainer`, `KTOTrainer`, and `RLOOTrainer`, but never `PPOTrainer`. So Phase
  7 hand-builds the multi-turn rollout loop, a critic/value head, and GAE with turn-boundary
  reward placement (Eq. 9) on top of `transformers.Trainer` directly — the two riskiest pieces of
  that build (the critic architecture, and driving the tool-calling loop manually) were verified
  working end-to-end before committing to this plan; see
  `docs/superpowers/specs/2026-07-05-phase-7-mt-ppo-design.md`. Phase 7 reproduces the paper's
  actual Table 2 methodology (deterministic rewards only); Phase 8 adds the paper's separate
  LLM-as-judge exploration (Appendix C.2/C.3, `gpt-oss-120b` via an OpenAI-compatible Bedrock
  endpoint) on top of a working Phase 7. See the Roadmap section below and
  `docs/phase-7-mt-ppo.md`/`docs/phase-8-llm-judge.md`.
- **`MT-GRPO`** — the paper's actual turn-level credit-assignment contribution for GRPO
  specifically: a *separate* advantage computed per turn via extra per-state rollouts (Eq. 5,
  Appendix D), instead of one trajectory-wide advantage. **Still out of scope, unrelated to the
  PPO work above.** TRL's `GRPOTrainer` has no supported hook to override its built-in advantage
  computation with a custom per-turn one — doing this for real means patching non-public trainer
  internals, not a documented extension point. A legitimate follow-on direction, not ruled out
  permanently — if picked up later, it should get its own phase doc under `docs/`.

## Why this design (retrieval backend choice)

Three retrieval options were considered; **Option B (full open-domain retrieval) was chosen**:

| | Closed 10-para pool (per-question) | Pooled BM25 over HotpotQA's own contexts | **Full wiki-18 BM25 (chosen)** |
|---|---|---|---|
| Corpus | That question's 10 given paragraphs | ~482K unique titles pooled across all 90K train rows | Real ~21M-passage 2018 Wikipedia dump |
| Fidelity to paper | Weak (10-way discrimination is too easy) | Medium-high | Highest — this is literally the retrieval corpus the Search-R1 lineage (which this paper builds on) uses |
| Setup cost | ~0 | Build in-process BM25 (`rank_bm25`/`bm25s`) over 482K docs | Download prebuilt corpus+index, install a JDK, run a retrieval server |

Confirmed facts that make Option B tractable (verified directly, not assumed):
- `PeterJinGo/wiki-18-corpus` = 5.12 GB (`wiki-18.jsonl.gz`, raw passages)
- `PeterJinGo/wiki-18-bm25-index` = 2.3 GB, confirmed genuine Lucene/Anserini segment files
  (`.dvm`/`.dvd`). Total download ~7.4 GB — well within the 241GB free on this machine.
- The dense e5 FAISS index was deliberately **not** chosen (would be tens of GB, requires
  reassembling sharded files) — BM25-only avoids that.
- `pyserini==2.3.0` + `pyjnius==1.7.0` resolve cleanly via `uv pip install` on **Python 3.13**
  (this project's version) — verified via a real dry-run install. No separate Python env needed.
- The one dependency pip can't manage: **a JDK must be installed at the OS level**
  (`pyjnius` bridges to a JVM for Lucene). Exact version needed by pyserini 2.3.0 not yet
  confirmed — check at setup time (`apt install openjdk-21-jdk` is the likely candidate).
- Search-R1's reference retrieval server (`search_r1/search/retrieval_server.py` in
  `PeterGriffinJin/Search-R1` on GitHub) exposes a clean FastAPI contract we can reuse/adapt:
  - `POST /retrieve` body: `{"queries": [str, ...], "topk": int, "return_scores": bool}`
  - response: `{"result": [[{"document": {...}, "score": float}, ...], ...]}` (one inner list
    per query)
  - loads the BM25 index via `pyserini.search.lucene.LuceneSearcher`, corpus via
    `datasets.load_dataset('json', data_files=corpus_path)` on the extracted jsonl
  - Run as a **separate process** from training; our `SearchEnv.search()` tool calls it over
    HTTP (`requests.post`), so Pyserini never needs to be imported into the TRL training process
    itself.
- **Confirmed gotcha (found by downloading and inspecting the actual bytes, not assumed)**:
  `PeterJinGo/wiki-18-corpus`'s `wiki-18.jsonl.gz` is **not** a plain gzipped jsonl — it's a
  gzip-compressed **tar archive** with one member,
  `data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl` (21,015,324 passages,
  confirmed by a full scan). Must `tar -xzf wiki-18.jsonl.gz` (or `tarfile.open(path, "r:gz")`)
  to get the real jsonl before pointing `retrieval_server.py` at it — its `load_corpus()` calls
  `datasets.load_dataset('json', data_files=...)`, which expects a plain jsonl(.gz) and would
  choke on the tar wrapper. Re-gzip the extracted file afterward if a smaller corpus file on
  disk is wanted.
- Corpus record schema (confirmed): `{"id": "<str>", "contents": "\"<Title>\"\n<passage text>"}`
  — there is no separate `title` field. Search-R1's own `BM25Retriever` derives it as
  `content.split("\n")[0].strip("\"")`; our `SearchEnv` should extract titles from returned
  documents the same way (this is exactly the logic used to verify alignment below).

**Confirmed (not assumed): wiki-18/HotpotQA title alignment is ~80%, not 100%.** Downloaded the
real 5.12GB corpus and directly scanned all 21,015,324 passages: of 400 unique gold
`supporting_facts` titles sampled from 200 HotpotQA validation questions, 322 (80.5%) exist as
an exact-match title in wiki-18. Verified the shortfall is a genuine corpus/snapshot gap and not
a parsing bug, by full-scanning for one suspicious miss ("Calgary" — a major, stable article
title unlikely to be a real gap) and confirming zero exact-title hits anywhere in the corpus.
**Implication**: even a perfect-retrieval policy will average well under 100% on the turn-level
`retrieval_fraction` signal — expect it to plateau somewhere below ~80%, not near 1.0. This is
fine (GRPO's group-relative advantage only needs meaningful variance within a group, not a
signal that reaches 1.0), but don't mistake a `turn_level` run's retrieval-hit-rate plateauing
at, say, 60-70% for a bug — it may just be hitting the corpus's real ceiling. No fuzzy-matching
fallback was added for this (extra complexity for uncertain benefit); exact-title matching stays
the mechanism, ceiling and all.

## Model

`Qwen/Qwen3.5-0.8B` (the user's original choice — confirmed to exist, apache-2.0, ~0.8B params).
TRL's GRPO docs list the Qwen3.5 family as supported for agent/tool-call training with
auto-patched prefix-preserving chat templates. Tagged `image-text-to-text` (natively
multimodal) but used here as a text-only causal LM.

## Dataset

**Training**: `PeterJinGo/nq_hotpotqa_train`, `default` config, `train` split, filtered to
`data_source == "hotpotqa"` (90,447 rows — confirmed to be the same underlying rows as
`hotpotqa/hotpot_qa`'s distractor train split, just repackaged alongside NQ questions). Use:
- `question` — the prompt
- `golden_answers` — **list** of acceptable answer strings (take max EM/F1 across the list, not
  just the first)
- `metadata.supporting_facts.title` — gold paragraph titles, used for the turn-level reward
- (`metadata.context` is NOT used for retrieval in this design — retrieval goes against the real
  wiki-18 corpus, not the row's own 10-paragraph distractor pool. `context` is only a legacy
  field carried over from the original HotpotQA row.)

Do **not** load this dataset's `test` split — it has a broken/mixed parquet schema (appears to
merge in an unrelated multi-hop QA source with incompatible columns) that throws a
`DatasetGenerationError` in 🤗 `datasets`. Confirmed by direct reproduction.

**Held-out eval**: use `hotpotqa/hotpot_qa`, `distractor` config, `validation` split (7,405 rows)
directly instead — same schema shape (`question`, `answer` (singular; wrap in a list for the
shared metrics code), `supporting_facts`), never touched during training.

Confirmed dataset facts (verified by actually loading and computing, not estimated):
- 90,447 train rows → 899,667 total paragraph slots → 482,021 unique titles (53.6% dedup ratio)
  — this was the basis for evaluating the (rejected) pooled-BM25 option, kept here for reference.
- 100% of rows have their gold `supporting_facts` titles present in their own `context` — no
  data-integrity surprises.
- Avg exactly 2.00 supporting facts per row (clean 2-hop structure).

## Hardware

Single RTX 4090, 24GB VRAM, otherwise idle (~470MiB used by desktop). 241GB free disk. No
multi-GPU, no distributed training. A 0.8B model with GRPO group rollouts should fit without
vLLM for a first pass; vLLM colocate mode available later if generation throughput bottlenecks.

## TRL mechanics being relied on (GRPOTrainer, confirmed via source + docs)

- `environment_factory=SearchEnv`: TRL instantiates environment instances from a **reusable
  pool** (`environment_factory()` called with zero args) — instances are `reset()` many times
  over training, not recreated per episode. **`reset()` must fully reinitialize all mutable
  state** — no leftover state from a prior episode. This is a concrete, testable invariant.
- `reset(self, **kwargs)` receives the **entire sampled dataset row** as kwargs (e.g. `question`,
  `golden_answers`, `metadata`, plus `prompt` itself) — extra columns not in the signature just
  land in `**kwargs` and are ignored.
- Every public method other than `reset` becomes a tool, auto-exposed via type hints + a
  Google-style docstring (same contract as the plain `tools=[...]` list).
- `reward_funcs` callables get an `environments` kwarg (list of the per-rollout instances, one per
  completion) **only when `environment_factory` is set** — this is how turn-level state
  (e.g. `environment.retrieval_fraction`) gets back to the reward function.
- `GRPOConfig(max_tool_calling_iterations=N)` is the hard cutoff on tool-calling turns (currently
  unlimited by default, bounded only by `max_completion_length`). **Finalized: `N=4`.**
  Confirmed directly against TRL's `_tool_call_loop` source
  (`trl/trainer/grpo_trainer.py`): `iteration_num` increments once per *(execute pending tool
  calls → generate the model's next turn)* round; the initial pre-tool-call generation doesn't
  count. So a fully-compliant rollout that does exactly the prompt's soft "at most 2 searches"
  (see Dataset/Reward design sections) consumes **exactly 2 iterations** and would never hit a
  cap set to 2 — it needs to be strictly above 2 or compliant rollouts get truncated before they
  can even answer, silently corrupting the reward signal. `N=4` leaves 2 full rounds of slack
  above that soft limit, so early-training rollouts that haven't yet learned the "at most 2"
  instruction (e.g. search 3-4 times) still complete and receive an honest — probably
  penalized — reward instead of being cut off mid-search with no answer at all. Not set higher
  than 4: each extra iteration is a full extra generation pass per rollout, and `format_reward`'s
  `-0.1` penalty for a missing `<answer>` is the right way to unlearn persistent
  over-searching, not a cutoff patient enough to wait it out.
- **This repo's "at most 2 searches" is a confirmed, deliberate deviation from the paper**, not
  an oversight. The paper's own Appendix E.1 (Task Formulation) states its GRPO case study caps
  the agent at **exactly one** search call before answering ("a simplified two-turn tool-use
  environment... the agent is allowed to call the Wikipedia search engine at most once before
  submitting an answer" — verified by fetching the paper's HTML version directly, not assumed
  from memory). This repo uses 2 instead because HotpotQA is genuinely 2-hop (avg exactly 2.00
  unique gold supporting-fact titles/row, already confirmed above) — a hard 1-search cap would
  make it structurally impossible to ever surface both gold passages, capping `retrieval_fraction`
  at ~50% regardless of policy quality. See
  `docs/superpowers/specs/2026-07-04-phase-3-data-pipeline-design.md` (lines ~29-30) for where
  this was first decided.
- `beta` (KL penalty) defaults to `0.0` in TRL — no reference model needed, saves memory.
- Multiple `reward_funcs` are summed (or weighted via `reward_weights`); returning `None` from a
  reward function lets it abstain per-example (not needed here — no task-mixing).
- `environment_factory` requires `transformers>=5.2.0`.
- **Periodic in-training eval (`GRPOConfig(eval_strategy="steps")`) is incompatible with
  `environment_factory` at large `num_generations`, in the installed `trl==1.7.1`.** Confirmed by
  a real canary run: `GRPOTrainer`'s `environments` pool (required by `SearchEnv`) is built once at
  init, sized to train's `generation_batch_size`, and reused unconditionally for both train and
  eval (`grpo_trainer.py:544-545`, `1982`) — the moment eval ran with a smaller per-step batch than
  train's, this raised `ValueError: zip() argument 2 is longer than argument 1`. Matching eval's
  batch size to train's instead reproduces the exact per-token-logps OOM documented in
  `docs/superpowers/specs/2026-07-05-phase-5-full-training-runs-design.md`, with no eval-side
  chunking knob available to work around it. **Already fixed upstream**, not yet released: TRL's
  `main` branch (commit `8b61980d`, "Support multiple environments [1/2]: Pool and build
  environment tool dicts at batch time", PR #6001, merged 2026-07-01) rebuilds the `environments`
  list at batch time, sized to `len(inputs)` for whichever batch (train or eval) is currently
  running — confirmed NOT an ancestor of the `v1.7.1` tag (`git merge-base --is-ancestor 8b61980d
  v1.7.1` → false) or any later PyPI release as of this writing. **Revisit if the live eval-reward
  curve is wanted later**: pin `trl` to a commit including `8b61980d` (accepting the stability
  tradeoff of an unreleased, unversioned dev build) and re-enable `eval_strategy="steps"` in
  `train.py`'s `build_config`. Until then, Phase 5 does not enable periodic eval — Phase 6's
  `evaluate.py` (one full post-hoc run over the entire held-out set) is the only held-out signal.

## Reward design (the crux decision)

**Terminology**: `outcome_only` below is the paper's `GRPO-OR`; `turn_level` is the paper's
`GRPO-MR`. See the Goal section's "Beyond the GRPO-OR/GRPO-MR comparison" note for why this stops
short of the paper's `MT-GRPO` (no per-turn advantage estimation — both conditions use one
trajectory-level GRPO advantage).

Shared outcome component (identical in both conditions):
- `format_reward`: small ±0.1 nudge for a parseable final-answer tag.
- `outcome_reward`: SQuAD-style F1 (0 to 1) + 0.5 bonus for exact match, maxed over
  `golden_answers`. Range ~[0, 1.5].

**This F1+EM-bonus outcome reward is a confirmed, deliberate deviation from the paper**, not an
oversight — parallel to the "at most 2 searches" deviation above. Fetched the paper's GRPO
Appendix E directly: its outcome reward there is **pure binary exact-match** ("Awards 1.0 if the
model's answer... exactly matches any accepted answer," 0.0 otherwise) — no partial credit. (A
separate, richer LLM-as-judge scoring scheme exists in the paper too, using `gpt-oss-120b` — that's
Phase 8's scope, built on top of Phase 7's PPO/MT-PPO work, not this GRPO comparison; see the Goal
section above.)

This repo uses F1+EM-bonus instead because GRPO's only learning signal is *within-group reward
variance* (the group-relative advantage) — a pure binary EM reward means any group of rollouts
that all miss the exact answer (plausible early in training, before the policy has learned
anything) scores identically 0.0 across the whole group, giving zero variance and thus zero
gradient. That is exactly the `frac_reward_zero_std`-stuck-at-1.0 failure mode
`TrackioAlertCallback` (Phase 4) exists to catch. F1's graduated partial credit gives GRPO
non-identical scores to differentiate rollouts by even when no rollout in a group is exactly
right, which matters more for this repro's compute-constrained, single-GPU, ~300-step scale than
it likely did at whatever scale the paper trained at.

This deviation does **not** confound the actual `GRPO-OR`/`GRPO-MR` comparison this repo is
testing: both conditions share the identical `outcome_reward` formula, so it's orthogonal to the
one variable under test (whether `turn_reward` helps). It does mean this repro's absolute
reward/EM numbers won't be directly comparable to the paper's own Table 5 figures — only the
*relative* outcome_only-vs-turn_level comparison is being reproduced here, which is what the
Goal section above already states as the scope, not the paper's headline absolute numbers.

Turn-level component (`turn_level` condition only):
- `turn_reward = 0.4 * retrieval_fraction`, where `retrieval_fraction` = fraction of the (usually
  2) gold `supporting_facts` titles actually surfaced by a `search()` call during the episode
  (deduped, capped at 1.0).

**Magnitude reasoning**: cap turn_reward at 0.4 vs. outcome_reward's max of 1.5 (~27% ratio).
Keeping it below even a partial-credit answer (F1 ≈ 0.4-0.6) and well below a fully correct
answer (1.5) avoids the real risk in reward shaping — the policy learning to "search well, then
answer carelessly" because the shaping term is easier to farm than the true objective. It's still
~3-4x `format_reward`'s magnitude, so it's a meaningful dense-gradient signal, not a rounding
error. Because GRPO scores one scalar per completed trajectory (no per-timestep value function),
this is turn-level credit assignment *via reward density*, not a literal per-step RL change —
state that explicitly in code comments so it's not mistaken for more than it is.

## Repo layout

```
src/turn_level_rewards/
    env.py             # SearchEnv (environment_factory) - calls the retrieval server over HTTP [done]
    metrics.py         # normalize_answer / exact_match / f1_score (SQuAD-style, stdlib only) [done]
    rewards.py         # format_reward, outcome_reward, turn_reward, get_reward_funcs(condition) [done]
    data.py            # dataset loading/filtering (nq_hotpotqa_train train, hotpot_qa validation) [done]
    train.py           # CLI: python -m turn_level_rewards.train --condition {outcome_only,turn_level} [done]
    evaluate.py         # run a trained checkpoint over held-out eval set, write metrics json [Phase 6, not yet built]
scripts/
    setup_retrieval.sh # download wiki-18 corpus+index, launch the Pyserini retrieval server [done]
    verify_phase{1,2,3,4}.py  # per-phase exit-criteria gates [done]
    compare_runs.py    # plot reward/EM/retrieval curves for both conditions [Phase 6, not yet built]
tests/unit/
    test_env.py / test_rewards.py / test_metrics.py / test_data.py / test_train.py   # no GPU, no live retrieval server needed [done]
```

**Test location is `tests/unit/` only.** Do not add other test tiers (integration tests against
the live retrieval server, GPU smoke-test suites, etc.) without checking with the user first —
this repo intentionally has just one test tier for now.

## Guiding principles for code, tests, and dependencies (cosmicpython, ch. 3 + ch. 5)

These are **general engineering principles for this repo**, from *Architecture Patterns with
Python* (cosmicpython.com) — they govern how *any* module here is designed and tested (not just
`env.py`/`rewards.py`; apply them equally to `train.py`, `evaluate.py`, `data.py`, `scripts/`,
and anything added later):

1. **Depend on abstractions at the seam, not on concrete external systems (dependency
   inversion)**. Wherever code touches something slow, external, or non-deterministic — a
   network/HTTP call, a subprocess, a loaded model/GPU, the filesystem, a logger backend, a real
   clock — that dependency should be received as a parameter/callable from the caller, not
   imported and invoked directly deep inside a function. Concrete implementations (the real
   retrieval HTTP client, the real `GRPOTrainer`/model, the real `trackio.log`) get wired up once,
   at the top-level entrypoint (e.g. `train.py`'s `main()`/CLI) — the "composition root." Tests
   inject a fake at that same seam instead of standing up the real thing. `SearchEnv.search()`'s
   HTTP call and `rewards.py` operating on plain `completions`/`environments` data (never a real
   `GRPOTrainer`) are just the *first two applications* of this rule, not the whole rule.
2. **Keep abstractions thin and honest, not preemptive**. Only introduce a seam where a genuine
   slow/external boundary actually exists (network, GPU, filesystem, randomness, wall-clock
   time) — don't wrap things in interfaces "just in case." Model the boundary as plain data where
   possible (a `{title, text}` dict, a float reward, a string) rather than a class hierarchy — a
   fake should be a trivial, obviously-correct stand-in, not its own mini-framework. This is what
   keeps DI here from turning into over-engineering.
3. **High gear by default, low gear when stuck (test-pyramid shape)**. Prefer a small number of
   fast, high-level tests that exercise real behavior through a stable, primitive-ish interface
   (e.g. call `get_reward_funcs("turn_level")` end-to-end and assert the returned floats; call
   `data.py`'s filtering function and assert on the resulting rows; call
   `scripts/compare_runs.py`'s data-prep function and assert on the resulting series) over many
   brittle tests pinned to internal implementation details. Only drop to low-level/domain-style
   tests where something is genuinely tricky and needs fine-grained design feedback (e.g. the
   SQuAD-style F1/tokenization edge cases in `metrics.py`). This determines test shape everywhere
   in the repo, not only for the files already designed.
4. **Every test in `tests/unit/` must be fast and deterministic** — no GPU, no live retrieval
   server, no network, no real trackio backend. This falls directly out of principle 1: if a test
   can't be made fast by faking the seam, that's a signal the seam is in the wrong place, not a
   reason to accept a slow test.

## Experiment tracking (trackio)

Both conditions log to the **same trackio project** (e.g. `project="turn-level-rewards"`) with
one **run per condition** (`outcome_only`, `turn_level`) so their curves are directly comparable
in one dashboard, rather than separate projects.

- Primary logging path: TRL's built-in integration — `GRPOConfig(report_to="trackio")` — logs
  the standard reward/KL/completion-length metrics automatically per training step.
- Additional custom metrics (via reward functions' `log_metric`/`log_extra` kwargs, see TRL
  mechanics above) should also land in trackio, not a separate side channel: `retrieval_fraction`
  (mean per step), `exact_match`/`f1` (mean per step), and `format_compliance_rate`.
- Use `trackio.alert(...)` (`WARN`/`ERROR` levels) inside `train.py` for autonomous-run
  diagnostics we'd otherwise have to eyeball in a live log: reward staying at exactly 0 past the
  first ~20 steps (dead/miswired reward or tool loop), `frac_reward_zero_std` staying at 1.0 for
  many consecutive steps (no learning signal — every group scoring identically), or loss/reward
  NaN. These let a run be launched in the background and polled with
  `trackio list alerts --project turn-level-rewards --json --since <timestamp>` instead of
  requiring a human/agent to watch stdout continuously.
- Retrieval/comparison after both runs finish: `trackio get metric --project turn-level-rewards
  --run <run> --metric <name> --json` feeds `scripts/compare_runs.py`, rather than re-parsing log
  files by hand.

## Verification approach (before any GPU-blind GRPO run)

1. `tests/unit/test_metrics.py` — EM/F1 sanity pairs.
2. `tests/unit/test_env.py` — fake retrieval-server responses (mocked HTTP), check
   `retrieval_fraction` accounting and that `reset()` fully clears state between episodes
   (exercises the pooled-instance-reuse behavior).
3. `tests/unit/test_rewards.py` — fake completions + duck-typed environments, assert exact reward
   values for both conditions, including the "hit the hard tool-call cap mid-call" edge case.
4. ~~Spot-check wiki-18 title alignment~~ — done (see confirmed ~80% figure above); no longer a
   pending item.
5. Live smoke test: real model, `max_steps=1-2`, tiny batch, `log_completions=True` — confirm the
   model actually calls `search`, the retrieval server responds, `environments` populate.
6. Only then run both full conditions and `evaluate.py` / `compare_runs.py`.

## Roadmap

Design finalized (Option B). Implementation is split into 10 phases, each in its own doc under
`docs/`, so a fresh agent with no memory of this conversation can pick up any single phase. All 10
are committed work, not optional stretch goals — Phases 7-8b (PPO/MT-PPO + LLM judge) are a second,
real comparison alongside the GRPO-OR/GRPO-MR one Phases 1-6 build, not a maybe-someday item.
**Read this file (CLAUDE.md) in full first, then the specific phase doc, then the previous
phase's "Handoff notes" section** — that's the actual handoff mechanism between phases.

**Phases 7b and 8b were added on 2026-07-23**, after noticing a real gap: Phase 7's doc (build
`MTPPOTrainer` + smoke test) and Phase 8's doc (add the LLM judge + smoke test) both stop at
smoke-test scale — neither ever runs a full training comparison. That mirrors Phase 4 (build +
smoke test) without a Phase 5/6 analog (full runs + `evaluate.py`/`compare_runs.py`/write-up),
which would leave this repo without the actual PPO-OR vs MT-PPO comparison — the paper's real
Table 2 headline result — ever getting produced. 7b and 8b close that gap, mirroring Phases 5+6's
shape for the PPO track. 7b also folds in a from-scratch request: the current README's charts are
plain and hard to follow, so 7b adds matplotlib visuals designed to actually tell the comparison's
story (see `docs/phase-7b-full-ppo-runs.md` for detail; use the `dataviz` skill when implementing
that task). 8b is intentionally a thin stub for now — its concrete task list depends on
implementation details (judge combination formula, cost/latency) that don't exist until Phase 8
lands.

| # | Phase | Doc | Status |
|---|---|---|---|
| 1 | Retrieval infra: JDK, wiki-18 download, retrieval server | `docs/phase-1-retrieval-infra.md` | **Done** — server running, `verify_retrieval.py` passes; see phase doc's Handoff notes for the launch command, a `retrieval_server.py` bug fix, and a correction to this file's "confirmed present" title examples |
| 2 | Core library: `env.py`, `rewards.py`, `metrics.py` + `tests/unit/` | `docs/phase-2-core-library.md` | **Done** — merged to `main` via PR #2; `scripts/verify_phase2.py` passes; see phase doc's Handoff notes for the confirmed `reset()` contract and a flagged Phase 3/4 gap (dataset's `prompt` column needs replacing) |
| 3 | Data pipeline: `data.py` | `docs/phase-3-data-pipeline.md` | **Done** — `scripts/verify_phase3.py` passes; real row counts confirmed (90,447 train / 7,405 eval); see phase doc's Handoff notes for the injectable-loader-seam deviation, the exact system prompt location, a `load_train_dataset` bug fix (HF `datasets` was preparing the already-documented-broken `test` split even when only `train` was requested), and a measurement clarification on the "avg supporting facts/row" figure (2.00 holds for unique titles, matching `env.py`'s dedup logic; the raw non-deduped count is 2.385) |
| 4 | `train.py` + live smoke test | `docs/phase-4-training-smoke-test.md` | **Done** — `scripts/verify_phase4.py` passes; live smoke test succeeded for both conditions (real tool calls, real retrieved passages, `turn_reward` confirmed genuinely nonzero, zero trackio alerts); see phase doc's Handoff notes for three real bugs the smoke test caught (a `GRPOConfig` divisibility constraint, two missing runtime dependencies, a docstring-format bug in Phase 2's `env.py`) and a CUDA OOM fixed with `gradient_checkpointing=True` |
| 5 | Full training runs (both conditions) | `docs/phase-5-full-training-runs.md` | **Done** — both conditions completed 300/300 steps, checkpoints saved and loadable, no unresolved alerts; see phase doc's Handoff notes for two real bugs live runs caught (an OOM survived by the first micro-batching fix, then a silent policy collapse from a `steps_per_generation`/`gradient_accumulation_steps` mixup), an unrelated infra fix (transient systemd cgroup killing long-running processes, worked around with `systemd-run`), the resulting 150-distinct-training-prompts figure and why the paper's own text leaves that ambiguous, and first training-batch results (both conditions show real learning; `turn_level` directionally ahead of `outcome_only` on EM/F1, not yet a rigorous claim pending Phase 6's held-out eval) |
| 6 | `evaluate.py` + `compare_runs.py` + write-up | `docs/phase-6-evaluation-comparison.md` | **Fully done, all follow-ups complete** — original run (seed42/300steps) was inconclusive (3 of 4 "more training needed?" criteria triggered); the symmetric re-run (seed123/600steps, both conditions, same larger budget + new seed) resolved it: `turn_level` shows a real, held-out-confirmed advantage over `outcome_only` (EM 0.307 vs 0.242, F1 0.399 vs 0.343), and `retrieval_fraction` rose instead of declining (0.40→0.57). Three follow-up reward-design experiments then ran against this baseline, none of which improved on it: `length_penalty` collapsed `outcome_only` completely and cost `turn_level` a mild real amount of accuracy; `search_count_penalty` (a non-paper-fidelity GRPO experiment, borrowing the paper's PPO-context mechanism) failed even more severely for `outcome_only` and cost `turn_level` real accuracy despite a partial training-time recovery; `remove-search-cap-prompt` (an isolating control, no penalty) confirmed the `search_count_penalty` failures were caused by the penalty term itself, not by losing prompt guidance — that alone costs `outcome_only` a small amount (over-searching) and costs `turn_level` nothing at all. See phase doc's Handoff notes for full numbers per experiment, the `beta=0`-matches-the-paper correction, the cross-experiment synthesis (`outcome_only`'s narrow reward is broadly fragile to added penalties; `turn_level`'s extra `turn_reward` term provides real but incomplete protection), and two infrastructure incidents documented honestly (a disk-space exhaustion resolved with explicit user-approved checkpoint deletion, and an unexplained external process kill that succeeded cleanly on retry) |
| 7 | Multi-turn PPO / MT-PPO (custom trainer, deterministic rewards): build + smoke test | `docs/phase-7-mt-ppo.md` | **Done** — `MTPPOTrainer` built and live-smoke-tested for both `--condition ppo` and `--condition mt_ppo` (real tool calls, real retrieved passages, correct Eq. 9 reward placement confirmed by direct observation — `mt_ppo`'s `R^I` nonzero exactly at a real gold-title hit (now backed by a real `train_log.jsonl` line, not just a removed debug print), `ppo`'s always 0 there — finite non-frozen critic values, `tests/unit/test_train_ppo.py`/`scripts/verify_phase7.py` passing); see phase doc's Handoff notes for five real bugs the smoke test caught, the `tool_call_id`/answerless-completion findings, a same-seed repeatability result (reward/retrieval_fraction reproduce exactly; raw loss internals do not) — **and a real, verified OOM-rate fix**: `flash-linear-attention==0.4.1` is now a pinned dependency (`causal-conv1d` deliberately excluded — its own compiled CUDA extension segfaulted on this machine; the memory win comes from `flash-linear-attention` alone), cutting intermittent OOM from ~50% to ~12.5% across 8 real end-to-end runs, with the residual failures staying the safe/catchable `torch.OutOfMemoryError` kind, not crashes |
| 7b | Full PPO/MT-PPO training runs + evaluation + comparison + matplotlib visuals | `docs/phase-7b-full-ppo-runs.md` | Not started — added 2026-07-23 to close the gap above; mirrors Phases 5+6's shape for the PPO track, plus new matplotlib-driven visuals for the write-up |
| 7c | **Paper-faithful PPO**: PPO-OR / PPO-MR / MT-PPO under the paper's own reward design | `docs/phase-7c-paper-faithful-ppo.md` | **Ready to launch** — added 2026-07-26 after an audit found Phase 7b measured a defective trainer against a reward function that was not the paper's. Four implementation defects fixed, the paper's R^I/R^O adopted, and the missing PPO-MR arm added (without it the ablation could not attribute anything to Eq. 9). Verified to 100 steps on all three arms |
| 7d | **Paper-faithful GRPO**: re-run GRPO-OR / GRPO-MR under the SAME paper reward design | `docs/phase-7d-paper-faithful-grpo.md` | Not started — **must come before Phase 8**. Section 6.1 defines GRPO-OR/GRPO-MR with wording identical to PPO-OR/PPO-MR; holding the reward fixed across algorithms is exactly what makes the paper's Table 2 cross-algorithm comparison valid. This repo's GRPO track currently uses its own (deliberately deviating) rewards, so the five arms are NOT comparable until this lands |
| 8 | LLM-as-judge reward (Bedrock + gpt-oss): build + smoke test | `docs/phase-8-llm-judge.md` | Not started. **Scope clarified 2026-07-26**: confirmed by direct fetch that the paper never benchmarks its judge (Table 2 is all deterministic; the judge appears only as training curves on NQ), so there is no target EM number to reproduce. This phase reproduces the reported training-curve stability only, and must keep the judge reward swappable at the same seam as the deterministic outcome reward so 8b needs no rework |
| 8b | **Judge vs. partial credit** — does an expensive judge beat a cheap graded metric? | `docs/phase-8b-full-judge-runs.md` | Not started — **rewritten 2026-07-26, and the one phase with an original contribution rather than a reproduction**. The paper motivates its judge by exact match being "overly rigid" (Section 5.3); this repo's F1+EM deviation independently addressed the same limitation with a cheap graded metric instead of a model. Nobody has compared them. Three MT-PPO arms (binary EM / judge / F1 partial credit), two of which already exist, evaluated on held-out EM — the metric that disadvantages both treatments, which is what would make a win meaningful |

Each phase doc is self-contained: goal, prerequisites (= previous phase's exit criteria), a task
checklist, exit criteria, and a **Handoff notes** section the executing agent fills in before
finishing — update this table's Status column too when a phase completes.
