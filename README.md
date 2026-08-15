# Training Outcome vs. Turn-Level Reward for Multi-Turn Agents

**Goal**: determine whether rewarding a model's intermediate actions, not just its final answer,
produces a measurably better multi-turn search agent.

**Why this matters.** RL algorithms like GRPO and PPO are the standard way to train the language
models that drive multi-turn agents, and they usually optimize a sparse outcome reward: one number,
right or wrong, at the very end of a long trajectory. That gives the model no signal about which of
its intermediate actions (like a good search) actually helped. The paper this repo is inspired by found that adding
a dense, turn-level reward signal on top of the same algorithms fixes that: more stable training,
faster convergence, and higher accuracy than sparse-reward baselines. This repo tests whether that
holds up at a much smaller scale.

Inspired by ["Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Reward
Design"](https://arxiv.org/abs/2505.11821) (arXiv:2505.11821), specifically its GRPO and PPO case
study. The reward designs below follow the paper's own definitions; experiments of our own are
labelled as such and kept separate from the reproduction.

**Biggest deviation**: a much smaller model on a single consumer GPU — `Qwen3.5-0.8B` on one
NVIDIA RTX 4090, vs. the paper's `Qwen2.5-7B` on 8 NVIDIA H100s — with a batch size 128× smaller.
HotpotQA is one of the paper's own six evaluation datasets; this repo uses it exclusively, for
its genuinely multi-hop questions. Retrieval is BM25 rather than the paper's dense E5. Every arm
here is also a single run (n=1) against the paper's n=5, so seed-to-seed variance is unmeasured.
Smaller deviations are noted inline.

## The Agent

The agent here is a **language model inside a harness**: at each turn the model decides
whether to search a Wikipedia snapshot (a fixed, offline copy, not the live site) or give a final
answer. Different rollouts of the same question can end up searching a different number of times.

What RL updates is only the **model's weights**. The harness around it — the tool-calling loop,
the system prompt, the retrieval backend, the 4-turn cap — is identical in every condition below
and is never trained. Only the reward function differs, so any difference in results is attributable
to the reward design rather than to the agent's architecture.

```mermaid
flowchart LR
    Q(["Question"]) --> D{"Search again,<br/>or answer?"}
    D -- search --> S["Search Wikipedia snapshot"]
    S --> D
    D -- answer --> A(["Final answer"])
```

## Reward approaches explored

GRPO computes **one** advantage per trajectory and applies that identical value to every token in
every turn. A sharp search followed by a garbled answer, and a lazy search followed by a lucky
guess, get scored no differently turn-by-turn. GRPO can't isolate which turn earned the credit.

That holds no matter *where* you attach a bonus. GRPO only ever sees one number per trajectory, so
a bonus placed mid-episode still gets folded into that same number. Actually using *where* a reward
happened requires a critic that can evaluate any point in a trajectory on its own. PPO has one;
GRPO doesn't. That's why the later PPO approaches switch algorithms rather than just rearranging
the reward.

Each approach below uses the paper's own reward definitions (arXiv:2505.11821 §5.2/§6.1).

### 1. `GRPO-OR` — outcome only

```mermaid
flowchart LR
    Q1(["Question"]) --> D1{"Search again,<br/>or answer?"}
    D1 -- search --> S1["Search"]
    S1 --> D1
    D1 -- answer --> A1(["Final answer"])
    A1 ==> R1{{"One score for the<br/>whole trajectory"}}
```

> **Reward:** `R = 1.0 if exact_match else 0.0` — binary, terminal, nothing for search.

### 2. `GRPO-MR` — merged reward

```mermaid
flowchart LR
    Q2(["Question"]) --> D2{"Search again,<br/>or answer?"}
    D2 -- search --> S2["Search"]
    S2 --> D2
    S2 -.->|"+0.3 if a real<br/>supporting passage"| R2
    D2 -- answer --> A2(["Final answer"])
    A2 ==> R2{{"Still ONE combined score"}}
```

> **Reward:** `R = R^O + 0.3 × retrieval_fraction` — one scalar, still terminal, where
> `R^O = +1.0` correct+format / `+0.2` wrong+format / `−1.0` bad format.

The retrieval bonus checks whether a search surfaced one of the specific Wikipedia articles a human
annotator marked as necessary, independent of whether the final answer came out right.

### 3. `PPO-OR` — the same outcome-only reward, scored by a critic

Same reward as `GRPO-OR`, different algorithm: no group comparison, just a learned value function
estimating expected return token-by-token.

```mermaid
flowchart LR
    Q3(["Question"]) --> D3{"Search again,<br/>or answer?"}
    D3 -- search --> S3["Search"]
    S3 --> D3
    D3 -- answer --> A3(["Final answer"])
    A3 ==> R3{{"One score at the<br/>final token"}}
    R3 -.->|"critic spreads credit<br/>backward (GAE)"| C3(("Critic"))
```

> **Reward:** `R = 1.0 if exact_match else 0.0`, placed at the last token.

### 4. `PPO-MR` — merged reward, with a critic

The paper's PPO baseline. The critic spreads credit backward, but the reward itself says nothing
about *when* it was earned.

```mermaid
flowchart LR
    Q4(["Question"]) --> D4{"Search again,<br/>or answer?"}
    D4 -- search --> S4["Search"]
    S4 --> D4
    S4 -.->|"+0.3 if a real<br/>supporting passage"| R4
    D4 -- answer --> A4(["Final answer"])
    A4 ==> R4{{"All rewards summed<br/>at the final token"}}
    R4 -.->|"critic spreads credit<br/>backward (GAE)"| C4(("Critic"))
```

> **Reward:** `R = R^O + Σ(0.3 × retrieval hits)`, **all placed at the last token**.

### 5. `MT-PPO` — turn-level credit assignment

The paper's best method. Placing each turn's reward **at the turn that earned it** is what makes
the critic's value estimate differ turn-by-turn.

```mermaid
flowchart LR
    Q5(["Question"]) --> D5{"Search again,<br/>or answer?"}
    D5 -- search --> S5["Search"]
    S5 --> D5
    S5 ==>|"scored at THIS turn"| R5a{{"R^I: retrieval + format<br/>− search penalty"}}
    D5 -- answer --> A5(["Final answer"])
    A5 ==> R5b{{"R^O at the final token"}}
    R5b -.->|"critic sends credit<br/>backward (GAE)"| R5a
```

> **Reward:** `R^I = 0.3·retrieval_hit + format(±0.1 / −0.2) − 0.1·n_searches` at **each turn
> boundary**, plus `R^O` at the last token.

Note what `R^I` contains that `PPO-MR` does not: a per-turn format term and a search-count penalty.

## Results

Every arm below: same model, same scaffold, same seed, same dataset, 500 training steps (except
`PPO-OR`, stopped once it had collapsed), evaluated on the same 7,404 held-out questions.
Only the reward differs.

![This reproduction against the paper's published results](results/phase7c_vs_paper.png)

Our exact-match scores sit well below the paper's, and they were never going to match: this trains
a model **8.75× smaller** on a batch **128× smaller**. The paper's own *untrained* `Qwen2.5-7B`
scores EM 0.160–0.292 — **roughly where our best trained arm lands.** Their model starts about
where ours finishes.

So absolute accuracy isn't the thing to read here. What survives the shrink is the **structure** —
which rewards produce a working model, which kill it outright, and how large the gaps between them
are. Everything below is about that.

### A binary outcome reward killed both arms that used it

![PPO-OR's format compliance falling to zero while the reward-shaped arms hold](results/phase7c_format.png)

`PPO-OR` stops answering and searches to the cap forever — while posting the **best retrieval score
of any arm** (0.528). The headline metric points the wrong way on a completely broken model.

![GRPO-OR collapsing: reward and reward-variance curves](results/phase7c_collapse.png)

`GRPO-OR` died differently: once every rollout in a group scores the same, GRPO's advantage is zero
by construction, so the gradient is **exactly 0.0000 from step 186 on** and nothing recovers.
Adding the retrieval bonus is the whole difference between that and a working model — `GRPO-MR`
scores 0.295 EM with nothing else changed.

Why here and not in the paper, whose `PPO-OR` reaches 0.435: collapse needs a batch with **zero**
correct answers, `(1−p)^N`. A smaller model lowers `p`, a smaller batch lowers `N`, and they
compound.

### A search penalty is not what prevents runaway searching

![Searches per episode: the unpenalised arm sits near 2 while the collapsed arm pins to the cap](results/phase7c_searches.png)

The paper says removing its search-count penalty causes "uncontrolled search usage." `PPO-MR` runs
that penalty at **zero** and sits near 2 searches per episode, while the arm glued to the cap is the
collapsed one. What bounds search is the +0.2 for a wrong-but-formatted answer — always some
pressure to stop and commit. `PPO-OR` had no such term.

### The paper's headline gain is two effects, not one

**The paper's headline result reproduced** — `PPO-MR → MT-PPO` gained **+0.039** EM here against
their **+0.017**. But that step changes reward *content* and reward *placement* at once. Adding the
missing control separates them:

![Waterfall decomposing the MT-PPO gain into reward content and placement](results/phase7c_decomposition.png)

The content is worth **+0.066**; the turn-level placement the paper's method is named for gives
**−0.027** back. At n=1 the first is large enough to trust, the second isn't.

### Our own experiment: a narrow reward is fragile to added penalties

Separately, we stress-tested a GRPO variant using graded partial credit (F1 + exact-match bonus)
under three added penalties.

![Held-out exact match and completion length across four reward configurations](results/followup_experiments_comparison.png)

No penalty beat the baseline, and the outcome-only arm answered these in **12 and 21 tokens** while
the merged reward degraded and stayed readable. A penalty-free control ruled out the prompt change,
so the penalty itself did it — under GRPO, a penalty with no matching positive incentive is a real
risk, and a denser reward is only partial protection.

## Future work

- **LLM-as-judge reward.** The paper's other turn-level signal — a model scores the turn instead of
  the verifiable retrieval check used here. Not started; the paper reports no benchmark number for
  it either, so there's no published score to check against.

## Running it

Reproducing any of this needs a JDK and a multi-GB local Wikipedia index, since retrieval runs
against the real ~21M-passage wiki-18 dump rather than a per-question pool.
`scripts/setup_retrieval.sh` downloads the index and prints the server command.

Training entrypoints are `turn_level_rewards.train` for the GRPO arms and `turn_level_rewards.train_ppo`
for the PPO arms, each taking `--condition`. `scripts/plot_phase7c.py` regenerates every figure
above from committed data alone, with no GPU and no retrieval server.

## Citation

If you use this repo or build on its experiments, please cite the paper it is based on:

> Quan Wei, Siliang Zeng, Chenliang Li, William Brown, Oana Frunza, Wei Deng, Anderson Schneider, Yuriy Nevmyvaka, Yang Katie Zhao, Alfredo Garcia, and Mingyi Hong. *Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Reward Design.* arXiv:2505.11821, 2025. [https://arxiv.org/abs/2505.11821](https://arxiv.org/abs/2505.11821)

```bibtex
@misc{wei2025reinforcingmultiturnreasoningllm,
  title={Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Reward Design},
  author={Quan Wei and Siliang Zeng and Chenliang Li and William Brown and Oana Frunza and Wei Deng and Anderson Schneider and Yuriy Nevmyvaka and Yang Katie Zhao and Alfredo Garcia and Mingyi Hong},
  year={2025},
  eprint={2505.11821},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.11821},
}
```

## Contributing

See `[CONTRIBUTING.md](CONTRIBUTING.md)` for dev setup, quality gates, and running tests.
