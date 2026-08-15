# Phase 8b: Judge vs. partial credit — does an expensive judge beat a cheap graded metric?

**Status**: not started. **Prerequisite: Phase 8** (judge built and smoke-tested) **and Phase 7c**
(the five-arm deterministic comparison).

**This is the phase with an original contribution in it**, rather than a reproduction.

## The question

The paper motivates its LLM judge with a specific limitation of verifiable rewards
(Section 5.3, verbatim):

> *"Verifiable rewards, such as exact match, provide a strict and objective form of evaluation.
> However, they can be overly rigid: an agent may produce a correct answer that differs slightly
> in form from the ground truth but still receives negative feedback."*

**This repo independently reached the same diagnosis and treated it differently.** The F1 + 0.5·EM
outcome reward (`rewards.outcome_reward`, Phases 2-6) exists precisely because binary exact match
is too rigid — the same problem, patched with a cheap graded metric rather than an expensive
model. That deviation was originally an unvalidated shortcut (see `CLAUDE.md`), but it turns out
to address a limitation the paper's own authors identified.

**Nobody has compared the two treatments.** The paper proposes the judge and never benchmarks it.

## The experiment

Three MT-PPO arms, identical except for how the outcome reward handles EM's rigidity:

| Arm | Outcome reward | Source | Already exists? |
|---|---|---|---|
| `baseline` | Binary EM (+1.0 / +0.2 / −1.0) | The paper's Table 2 | **Yes** — Phase 7c's `mt_ppo` |
| `judge` | LLM-as-judge scoring | The paper's Section 5.3 | Phase 8 |
| `partial_credit` | F1-graded outcome | This repo | **Yes** — `rewards.outcome_reward` |

Two of three already exist, so the marginal cost is one arm plus evaluation.

## Methodological trap — design around it, do not discover it

Each arm optimizes a different objective, so **the evaluation metric must not favour one by
construction**:

- Scoring on F1 advantages `partial_credit` (it optimizes F1 directly)
- Scoring on judge output advantages `judge`
- Scoring on EM advantages `baseline`

**Use held-out exact match.** It is the paper's own reported metric, and it disadvantages the two
treatments — which is what makes a win meaningful. If `judge` or `partial_credit` beats
`baseline` **on EM despite not optimizing EM directly**, that is a real result rather than a
tautology. State this reasoning in the write-up; it is the difference between a finding and a
rigged comparison.

Report F1 alongside EM for completeness, clearly labelled as favouring one arm.

## Also measure, do not merely note

- **Cost and latency**: a judge call per turn per rollout, inside the training loop. Record
  wall-clock and spend against the deterministic arms. If the judge wins but costs 10x, that is
  part of the result.
- **Reward hacking**: a policy optimizing against a model's opinion can learn to produce text the
  judge scores well without being more correct — a failure mode a deterministic check cannot have.
  Watch for judge reward rising while held-out EM stalls or falls. That divergence is itself a
  reportable finding.

## Exit criteria

- Three arms trained and evaluated on held-out EM, same seed, same eval size
- Cost/latency recorded per arm
- An explicit statement of whether the judge justified its cost over F1 partial credit
- Handoff notes filled in; `CLAUDE.md` roadmap Status updated

## Honest framing for the write-up

This is a small, single-seed experiment on one dataset with a batch size far below the paper's.
It cannot settle the question in general. What it can do is provide the first direct comparison of
two treatments for a problem the paper itself named — and if the cheap one holds up, that is worth
saying plainly, with its limitations stated just as plainly.
