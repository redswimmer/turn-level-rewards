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

**Exact match lands systematically below the paper's; format correctness lands much nearer** — the
signature of training a model **8.75× smaller** (`Qwen3.5-0.8B` vs `Qwen2.5-7B`). Format is dense
and saturates quickly; exact match is sparse and hard, and is what the paper's larger model and
128× larger batch actually buy. For scale, the paper's *untrained* baseline scores EM 0.160–0.292 —
roughly where this repo's best *trained* arm lands.

`PPO-OR` is scored at its last checkpoint before collapse, the paper's own methodology for its
crashed baselines.

### Turn-level reward reproduces, and the effect is larger here

`PPO-MR → MT-PPO` gains **+0.039 exact match** against the paper's +0.017. On the GRPO side the
merged reward is the difference between a working model and a dead one: **0.000 → 0.295** EM.

### A binary outcome reward killed both arms that used it

![PPO arms during training: format compliance and searches per episode](results/phase7c_training.png)

Both binary-outcome-only arms — `PPO-OR` and `GRPO-OR`, two *different algorithms* — were killed by
the same cause, through **two different mechanisms**.

`PPO-OR` is visible above: it stops answering entirely and pins itself to the 4-turn search cap,
ending at a format compliance of **0.003** while still retrieving as well as the arms that worked
(0.528, against 0.52 for `MT-PPO` and 0.55 for `PPO-MR`). It learned to search and never learned to
commit. Its gradient stayed alive the whole
time — `policy_loss` is nonzero in every one of its final 20 steps — so this is a degenerate policy,
not a frozen one.

![GRPO-OR collapsing: reward and reward-variance curves](results/phase7c_collapse.png)

`GRPO-OR` failed harder, and its mechanism is exact. Once every rollout in a group scores
identically, GRPO's advantage — `(rᵢ − group mean) / group std` — is zero, so the gradient is
**exactly 0.0000 from step 186 to 499**. Checkpoints 450 and 500 are byte-identical, and the frozen
policy never called search once across all 7,404 held-out questions. Under GRPO specifically, a
sparse binary reward isn't just slow to learn from; it's an absorbing state you cannot leave. PPO's
critic spares it that — and still produced a model that never answers.

This is a small-scale effect, not a refutation — the paper's `PPO-OR` reaches 0.435 EM. What
triggers it is the chance a batch contains **zero** correct answers, `(1−p)^N`: a smaller model
lowers `p`, a smaller batch lowers `N`, and they compound in the same exponent.

### A search penalty is not what prevents runaway searching

The paper warns that dropping its search-count penalty causes "uncontrolled search usage." `PPO-MR`
runs that penalty at **zero** by the paper's own definition — and, in the right-hand panel above,
sits near 2 searches per episode while the collapsed arm is the one glued to the cap.

What actually bounds search is the **graded** outcome reward: a wrong-but-formatted answer still
earns +0.2, so there is always positive pressure to stop searching and commit. `PPO-OR` had no such
term, and that — not the missing penalty — is what let it search forever.

### The paper's headline gain is two effects, not one

The paper's `PPO-MR → MT-PPO` step changes **two things at once**: it adds reward components (a
per-turn format term and a search penalty) *and* moves them to turn boundaries. Its published gain
cannot say which one earned the credit.

Adding the missing control — the same reward content, flattened back to the final token — separates
them:

![Waterfall decomposing the MT-PPO gain into reward content and placement](results/phase7c_decomposition.png)

**At this scale the reward content does the work (+0.066) and turn-level placement gives some of it
back (−0.027).** The published net gain is real, but it is a large positive plus a smaller negative,
not a single effect. The content effect is big enough to survive a lot of seed variance; the
placement effect is **suggestive, not settled** at n=1.

### Our own experiment: a narrow reward is fragile to added penalties

Beyond the reproduction, we stress-tested a GRPO variant using **graded** partial credit (F1 +
exact-match bonus) instead of the paper's binary reward, under three manufactured pressures.

![Held-out exact match and completion length across four reward configurations](results/followup_experiments_comparison.png)

**None beat the baseline** — but *how* they failed is the finding. Under either penalty the
outcome-only arm collapsed to 12- and 21-token answers, while the merged reward degraded and stayed
coherent. A control with the same prompt change and *no* penalty held up fine, which pins the
collapses on the penalty term itself rather than on the prompt.

If every rollout in a group finds the same cheap trick, GRPO can't see past it — the whole group
looks equally bad, so no gradient says it's wrong. A bare penalty with no matching positive
incentive is genuinely risky under GRPO, and a denser reward is real but **incomplete** protection.

<details>
<summary>Methodology, symmetry checks and caveats</summary>

- **Scale.** All training and evaluation ran on one RTX 4090 — roughly 14 GPU-hours of training
  and 45 of held-out evaluation.
- **Symmetry.** Every arm ran at seed 42 with identical hyperparameters, and no episode in any PPO
  arm ended without a chance to answer. The arms that trained to completion took 488–495 real
  gradient updates out of 500 steps (the rest skipped for OOM); `PPO-OR` is the exception, stopped
  at step 159 once collapsed.
- **Evaluation is deterministic.** Both PPO evaluations were re-run on the same checkpoints and
  reproduced **bit-identically to 16 decimal places** (greedy decoding), so all remaining
  uncertainty is training-run variance, not measurement error. On sampling error alone the two
  decomposition effects are ~9 SE and ~3.7 SE — but that is the wrong error bar to lean on, which
  is why the placement effect is not over-claimed.
- **Seeds.** n=1 against the paper's n=5. Resolving a 1.7-point effect would need ~6–12 seeds
  (110–220 GPU-hours).
- **Cross-algorithm comparison is not valid here.** PPO and GRPO arms saw very different amounts of
  data (~1,964 vs ~250 distinct training prompts) because GRPO must spend its whole generation batch
  on one prompt to form a group baseline. The arms belong in one table; a PPO-vs-GRPO *causal*
  claim does not follow from it.
- **The follow-up experiments are their own track**, not comparable to the arms above: graded
  reward, seed 123, 600 steps, with an internal baseline of 0.242 EM outcome-only vs 0.306 merged.
- **Retrieval ceiling.** ~20% of HotpotQA's gold passage titles aren't in this Wikipedia snapshot,
  so retrieval fraction can't reach 1.0 even with perfect search.

</details>

### Key learnings

1. **A published delta that moves two variables at once is worth decomposing before building on
   it.** The paper's turn-level gain reproduced — but most of the credit belongs to *what* the
   reward contains, not to the turn-level *placement* the paper's contribution is named for.
2. **Reward density is a training-stability property, not just an efficiency one.** A binary reward
   killed both outcome-only arms, but by different routes: under GRPO it was an absorbing state
   (zero within-group variance means a gradient of exactly zero, and nothing recovers), while PPO's
   critic kept the gradient alive and still converged on a model that never answers.
3. **Diagnose the mechanism, not the symptom.** A completely broken model retrieved about as well
   as the arms that worked, and the arm running *without* the paper's anti-search penalty searched
   no more than the penalized ones. Headline metrics pointed the wrong way in both cases; only the
   per-step curves explained what happened.
4. **Reproduce faithfully before improving.** An early pass here measured turn-level placement
   against a "cleaner" baseline of our own design and concluded the paper's central claim *didn't*
   reproduce. It did — our own deviation was hiding it.


## Future work

- **LLM-as-judge reward.** The paper's other turn-level signal — a model scores the turn instead of
  the verifiable retrieval check used here. Not started; the paper reports no benchmark number for
  it either, so there's no published score to check against.

## Project structure

```
.
├── data/       # downloaded wiki-18 retrieval corpus + BM25 index (gitignored, multi-GB)
├── docs/       # phase docs, design specs, roadmap
├── outputs/    # training checkpoints + logs per condition (gitignored)
├── results/    # final held-out metrics + comparison plots (committed)
├── scripts/    # retrieval server, setup/verification, plotting (plot_phase7c.py regenerates
│              #   every README figure from committed data alone)
├── src/        # the turn_level_rewards package (env, rewards, metrics, data, train, evaluate)
└── tests/      # unit tests (fast, no GPU, no live retrieval server)
```



## Getting started



### Prerequisites

- Python 3.13+
- `[uv](https://docs.astral.sh/uv/)`
- JDK 21 (needed by the retrieval server's Lucene bridge)

Two choices worth knowing before you set up: the model is a deliberately small `Qwen3.5-0.8B`
(fits one GPU, no distributed training), and retrieval hits a real ~21M-passage Wikipedia
snapshot rather than a small per-question pool (a closed pool would make retrieval trivially
easy to solve, not a real test).

```bash
uv sync
sudo apt install openjdk-21-jdk
```



### Retrieval server

Training and evaluation search a local BM25 server backed by the real wiki-18
Wikipedia dump (~21M passages). Set it up once:

```bash
bash scripts/setup_retrieval.sh   # downloads the wiki-18 BM25 index (+corpus if needed) into data/wiki18/
```

The script downloads the index, checks whether it also needs the separate
corpus file, and prints the exact command to launch the server, something
like:

```bash
uv run python scripts/retrieval_server.py \
    --index_path data/wiki18/bm25-repo/bm25 \
    --corpus_path data/wiki18/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl \
    --port 8000
```

Run that (in the background or a separate terminal, since it needs to stay up for
the rest of setup and for training/evaluation later), then confirm it's
working:

```bash
uv run python scripts/verify_retrieval.py
```

```
PASS: retrieval server is up, wired correctly, and returns real documents.
```



### Training

```bash
uv run python -m turn_level_rewards.train --condition outcome_only
uv run python -m turn_level_rewards.train --condition turn_level
```

The bare invocation above (no extra flags) runs at smoke-test scale: 8 rows, 2 steps, a real
`Qwen/Qwen3.5-0.8B` model against the retrieval server started above. Pass `--train-size`,
`--max-steps`, `--num-generations`, etc. explicitly for a full-scale run. Both conditions
log to the same [trackio](https://github.com/gradio-app/trackio) project
(`turn-level-rewards`). Run `trackio show --project turn-level-rewards` to view.

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
