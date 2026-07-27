# Phase 8: LLM-as-Judge Reward (Bedrock + gpt-oss)

> **Scope reframed 2026-07-26 — read this before the original Goal below.**
>
> **The paper never benchmarks its judge.** Confirmed by direct fetch: Table 2 is entirely
> deterministic rewards, and the judge appears only as training reward curves (Figure 6) on **NQ**,
> a dataset this repo does not use. So "reproduce the judge" has no target number to hit, and as
> originally scoped this phase could not produce a reproduction result.
>
> **A better framing, and a genuinely original one.** The paper's stated motivation for the judge
> is a specific limitation of verifiable rewards (Section 5.3, verbatim):
>
> > *"Verifiable rewards, such as exact match, provide a strict and objective form of evaluation.
> > However, they can be overly rigid: an agent may produce a correct answer that differs slightly
> > in form from the ground truth but still receives negative feedback."*
>
> **This repo independently identified the same limitation and treated it differently.** The F1 +
> 0.5·EM outcome reward (`rewards.outcome_reward`, Phases 2-6) exists precisely because binary
> exact match is too rigid — the same diagnosis, patched with a cheap graded metric instead of an
> expensive model. Nobody has compared the two treatments.
>
> **Proposed experiment: three MT-PPO arms, differing only in how the outcome reward handles EM's
> rigidity.**
>
> | Arm | Outcome reward | Source |
> |---|---|---|
> | baseline | Binary EM (+1.0 / +0.2 / −1.0) | The paper's Table 2 — comes free from Phase 7c |
> | judge | LLM-as-judge scoring | The paper's Section 5.3 |
> | partial credit | F1-graded outcome | This repo's own deviation |
>
> The question — *does an expensive LLM judge actually beat a cheap graded metric at solving the
> same problem?* — is one the paper raises but never answers, and two of the three arms already
> exist.
>
> **Methodological trap to design around.** Each arm optimizes a different objective, so the
> evaluation metric must not favour one by construction: scoring on F1 advantages the
> partial-credit arm, scoring on judge output advantages the judge arm. Use held-out **exact
> match** — the paper's own reported metric. If either treatment beats the baseline on EM *despite
> not optimizing EM directly*, that is a real result rather than a tautology.
>
> **Also weigh the costs honestly** before committing: a judge call per turn per rollout is real
> latency and spend inside the training loop, and a policy optimizing against a model's opinion can
> learn to produce text the judge likes without being more correct (reward hacking) — a failure
> mode a deterministic check cannot have. Those are worth measuring, not just noting.
>
> **Decide whether this phase earns its place after Phase 7c's five-arm results are in.** It is no
> longer automatically committed work.

## Goal

Add an LLM-as-judge reward signal on top of a working Phase 7 `MTPPOTrainer`, matching the
paper's Appendix C.2/C.3 exploratory judge scheme — via an OpenAI-compatible endpoint on Amazon
Bedrock, using `gpt-oss-120b` (the paper's own choice), with `gpt-oss-20b` as a cheaper
alternative to validate once judge prompts exist.

## Read first

`CLAUDE.md`'s Goal section, and `docs/phase-7-mt-ppo.md`'s Handoff notes (read this first — Phase
8 builds directly on Phase 7's trainer, reward-placement, and critic infrastructure, not a
separate pipeline). Note the paper's own headline Table 2 numbers use deterministic rewards only —
this phase is reproducing a genuine but *secondary* exploration from the paper (Figure 6, shown on
NQ, not HotpotQA), not the paper's primary claim.

## Prerequisites (entry state)

- Phase 7 done: `MTPPOTrainer` works for both `ppo`/`mt_ppo` conditions, smoke-tested.
- An AWS account with Bedrock access to `openai.gpt-oss-120b` (and `openai.gpt-oss-20b` for the
  cost-comparison validation below) in a supported region (confirmed available: us-east-1,
  us-east-2, us-west-2, eu-central-1, and others — see
  `https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-oss-120b.html` for
  the current list).

## Confirmed facts (verified directly, not assumed)

- Bedrock's `bedrock-mantle` endpoint is a genuine, first-class OpenAI-compatible API — usable
  with the unmodified `openai` Python SDK: `OPENAI_BASE_URL="https://bedrock-mantle.<region>.api.aws/v1"`,
  `client.chat.completions.create(model="openai.gpt-oss-120b", ...)`. Not a shim, no LiteLLM or
  similar needed.
- Pricing (Standard tier, per 1M tokens): `gpt-oss-120b` = $0.1545 input / $0.618 output;
  `gpt-oss-20b` = $0.0721 input / $0.309 output (~2x cheaper). Bedrock also offers proprietary
  `GPT-5.4`/`GPT-5.5` (18-55x more expensive) and `GPT OSS Safeguard` variants (similar price to
  gpt-oss but tuned for content moderation, not general quality judging) — neither recommended
  here; stay in the gpt-oss family to match the paper's own choice and avoid paying a frontier-model
  premium for a structured-scoring task.
- The paper's judge uses two separate prompts: an outcome-level prompt ("Score 1.0 if the answer
  matches ground truth, else 0.0") and a turn-level prompt (range `[-1.0, 1.0]`, assessing format
  compliance, content quality, and contribution toward the ground-truth answer) — "used for PPO-OR
  and MT-PPO training respectively" per the paper, though it doesn't give an exact combination
  formula with the deterministic reward. This repo will need to design and document its own
  combination approach (same treatment as other confirmed deviations in CLAUDE.md), not assume one
  from the paper.

## Tasks

- [ ] Design and document the judge-reward combination formula (magnitude, how it adds to/replaces
      the existing deterministic rewards) — an explicit design decision, not a paper-derived
      number; record the reasoning the same way CLAUDE.md documents its other reward-magnitude
      choices.
- [ ] Add a `judge_client` seam (DI, per CLAUDE.md's guiding principles — an HTTP/API call is
      exactly the kind of external boundary that needs an injectable seam, tests fake it) wrapping
      the `openai` SDK pointed at Bedrock.
- [ ] Implement the judge-reward function(s), reusing the paper's two prompt types (outcome-level,
      turn-level).
- [ ] Validate `gpt-oss-20b` vs `gpt-oss-120b` judge-quality on a small sample before committing to
      either for the real runs — an empirical question, not something to assume from pricing alone.
- [ ] `tests/unit/`: judge-reward parsing/combination logic with a fake `judge_client`, no real
      Bedrock calls in `tests/unit/`.
- [ ] Live smoke test against the real Bedrock endpoint (small scale, both conditions).

## Exit criteria (all must be true before handing off)

- [ ] Judge-reward wired into a real `mt_ppo`/`ppo` smoke-test run against the real Bedrock
      endpoint, confirmed working (real judge scores observed, not just no-exception).
- [ ] The 20b-vs-120b judge-quality validation completed and its result recorded.
- [ ] Cost per training run estimated and recorded (given the real token volumes observed during
      the smoke test), so a full run's judge-inference cost is known before committing to it.

## Handoff notes

<!-- Fill in after completing this phase: the actual combination formula chosen and why, the
20b-vs-120b validation result, real observed judge latency/cost per run, and anything about
Bedrock's API (rate limits, auth setup, error handling) that surprised implementation. Leave this
section for the next fresh agent to read first. -->

(not yet started)
