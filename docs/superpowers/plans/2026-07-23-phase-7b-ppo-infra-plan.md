# Phase 7b Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the trackio diagnostics, collapse-visibility, and checkpoint-resume
infrastructure `docs/superpowers/specs/2026-07-23-phase-7b-full-ppo-runs-design.md` calls for,
run the wall-clock probe that picks this repo's hardware-driven scale, and verify all of it live
-- everything needed before the actual full 500-step `ppo`/`mt_ppo` training runs can be launched
safely. Mirrors this repo's established Phase 5 boundary: the full runs themselves, and the
eval/comparison/chart write-up, are follow-up work after this plan, once real numbers exist to
plan against (mirroring Phase 6's separate plan).

**Architecture:** Additive changes to the existing `MTPPOTrainer` in
`src/turn_level_rewards/train_ppo.py` (no new trainer class, no restructuring) -- new pure
helper functions and a small `CollapseMonitor` class, all unit-testable without a GPU, plus
extensions to `compute_ppo_loss`/`_collect_batch`/`train()` that reuse `transformers.Trainer`'s
own inherited checkpoint primitives rather than a custom format. One new file,
`src/turn_level_rewards/evaluate_ppo.py`, mirrors Phase 6's `evaluate.py` composition-root
pattern for the PPO track.

**Tech Stack:** Python 3.13, `transformers==5.13.0` (`Trainer`'s checkpoint primitives),
`torch`, `pytest`, `trackio`.

## Global Constraints

- Test location: `tests/unit/` only (CLAUDE.md's "Repo layout" section) -- no new test tiers.
- Every paper-matched hyperparameter in `MTPPOConfig` (`policy_lr=1e-6`, `critic_lr=1e-5`,
  `gamma=1.0`, `gae_lambda=1.0`, `num_ppo_epochs=4`, `clip_eps=0.2`) stays unchanged by this plan.
  The only new config surface is scale (`num_rollouts_per_step`, `save_steps`) and resume
  (`--resume-from-checkpoint`) -- see
  `docs/superpowers/specs/2026-07-23-phase-7b-full-ppo-runs-design.md`'s "hardware is the only
  intended deviation" decision.
- Code paths that touch a real model/GPU/tokenizer (`_rollout_episode`, `_collect_batch`,
  `_save_full_checkpoint`, `_load_full_checkpoint`, `evaluate_ppo.py`'s rollout loop) are **not
  unit-tested** -- this repo's existing, established convention (see every such method's own
  docstring in `train_ppo.py`) is to validate these via the live smoke test instead. Pure logic
  extracted from those paths (helpers, the `CollapseMonitor` class, metric aggregation) **is**
  unit-tested.
- `MTPPOTrainer.__new__(MTPPOTrainer)` (bypassing `__init__`, which needs a real tokenizer) is
  this repo's established pattern for testing trainer methods against fake
  policy/critic/`nn.Linear` stand-ins without a GPU (see
  `test_create_optimizer_uses_two_param_groups_with_paper_learning_rates` in
  `tests/unit/test_train_ppo.py`) -- reuse it, don't invent a different fixture style.
- Run `uv run pytest tests/unit/test_train_ppo.py -v` after every task; all tests (old and new)
  must stay green throughout.

---

### Task 1: Wall-clock/OOM scale probe

**Files:**
- Create: `docs/superpowers/probes/2026-07-23-phase-7b-wallclock-probe.md` (results write-up)

**Interfaces:**
- Consumes: `src/turn_level_rewards/train_ppo.py`'s existing CLI (`python -m
  turn_level_rewards.train_ppo --condition ... --max-steps ... --num-rollouts-per-step ...`) --
  no code changes in this task.
- Produces: a real, measured `num_rollouts_per_step` and confirmation of whether 500 steps is
  wall-clock-feasible, both of which Task 7's live smoke test and this repo's post-plan full-run
  launch command will use.

- [ ] **Step 1: Confirm the retrieval server is running**

Run: `curl -s -X POST http://localhost:8000/retrieve -H 'Content-Type: application/json' -d '{"queries": ["test"], "topk": 1, "return_scores": false}'`
Expected: a JSON response with a `"result"` key (not a connection error). If it fails, start the
server per `docs/phase-1-retrieval-infra.md`'s launch command before continuing.

- [ ] **Step 2: Run a short probe at `num_rollouts_per_step=2` (current default) for `ppo`**

Run: `time uv run python -m turn_level_rewards.train_ppo --condition ppo --train-size 32 --max-steps 15 --num-rollouts-per-step 2 2>&1 | tee /tmp/probe_ppo_r2.log`
Record: wall-clock for the whole command (from `time`), whether any step raised
`torch.OutOfMemoryError` (grep the log for `OutOfMemoryError`), and the per-step `elapsed=`
figures printed by `train()`'s own progress line.

- [ ] **Step 3: Run the same probe at a larger `num_rollouts_per_step` (e.g. 4) for `ppo`**

Run: `time uv run python -m turn_level_rewards.train_ppo --condition ppo --train-size 32 --max-steps 15 --num-rollouts-per-step 4 2>&1 | tee /tmp/probe_ppo_r4.log`
Record the same figures. Repeat with a larger value again (e.g. 8) only if 4 succeeded cleanly
and per-step wall-clock still looks like it leaves headroom -- stop increasing once an OOM
appears or per-step time roughly doubles per doubling of rollouts (diminishing single-GPU
returns).

- [ ] **Step 4: Run the same probe for `mt_ppo` at the `num_rollouts_per_step` chosen in Step 3**

Run: `time uv run python -m turn_level_rewards.train_ppo --condition mt_ppo --train-size 32 --max-steps 15 --num-rollouts-per-step <chosen> 2>&1 | tee /tmp/probe_mt_ppo.log`
Confirm wall-clock and OOM behavior are comparable to `ppo`'s (they should be -- `mt_ppo` only
differs in reward placement, not compute cost).

- [ ] **Step 5: Write up the results**

Create `docs/superpowers/probes/2026-07-23-phase-7b-wallclock-probe.md` containing: the exact
commands run, measured seconds/step at each `num_rollouts_per_step` tried, any OOMs observed, the
chosen `num_rollouts_per_step` for the full run, the resulting estimated wall-clock for 500 steps
at that setting (`seconds_per_step * 500`, converted to hours), and an explicit statement of
whether 500 steps is being kept as-is or reduced -- per the design doc, only reduce if the
estimate is genuinely impractical (e.g. multiple days), and if so say by how much and why.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/probes/2026-07-23-phase-7b-wallclock-probe.md
git commit -m "docs: record Phase 7b wall-clock/OOM probe results"
```

---

### Task 2: PPO diagnostic metrics (ratio mean/variance, clip fraction)

**Files:**
- Modify: `src/turn_level_rewards/train_ppo.py:173-207` (`compute_ppo_loss`)
- Modify: `src/turn_level_rewards/train_ppo.py:639-734` (`_ppo_update`)
- Modify: `src/turn_level_rewards/train_ppo.py:765-881` (`train`)
- Test: `tests/unit/test_train_ppo.py` (append near the existing `compute_ppo_loss` tests)

**Interfaces:**
- Consumes: `_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor` (already
  defined at `train_ppo.py:169`, unchanged).
- Produces: `compute_ppo_loss` now also returns `"ratio_mean"`, `"ratio_variance"`,
  `"clip_fraction"` (all detached `torch.Tensor` scalars) in its result dict. `_ppo_update`'s
  returned dict gains the same three keys (averaged the same way `policy_loss`/`kl` already are).
  `train()`'s `metrics`/`trackio.log`/`train_log.jsonl` gain the same three keys. Task 4's
  `CollapseMonitor` does not depend on these fields -- purely additive, no other task consumes
  them.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_train_ppo.py` (after `test_compute_ppo_loss_value_loss_scales_with_squared_error`):

```python
def test_compute_ppo_loss_reports_clip_fraction_and_ratio_mean():
    """Position 0: new_logprobs=10.0 vs old_logprobs=0.0 -> ratio=exp(10), clipped (clip_eps=0.2).
    Position 1: new_logprobs==old_logprobs -> ratio=1.0, not clipped. clip_fraction should be
    exactly 0.5 (one of two positions clipped); ratio_mean should be the plain mean of the two
    raw (pre-clip) ratios.
    """
    result = compute_ppo_loss(
        new_logprobs=torch.tensor([10.0, 0.0]),
        old_logprobs=torch.tensor([0.0, 0.0]),
        advantages=torch.tensor([1.0, 1.0]),
        returns=torch.tensor([0.0, 0.0]),
        new_values=torch.tensor([0.0, 0.0]),
        action_mask=torch.tensor([1.0, 1.0]),
        clip_eps=0.2,
    )

    expected_ratio_mean = (torch.exp(torch.tensor(10.0)).item() + 1.0) / 2
    assert result["clip_fraction"].item() == pytest.approx(0.5)
    assert result["ratio_mean"].item() == pytest.approx(expected_ratio_mean, rel=1e-4)


def test_compute_ppo_loss_diagnostic_fields_ignore_masked_out_positions():
    """action_mask=0 at position 1 (a wild, would-be-clipped ratio) must not count toward
    clip_fraction or ratio_mean -- only the masked-in position 0 (ratio=1.0, not clipped)
    should be visible.
    """
    result = compute_ppo_loss(
        new_logprobs=torch.tensor([0.0, 999.0]),
        old_logprobs=torch.tensor([0.0, 0.0]),
        advantages=torch.tensor([1.0, 1.0]),
        returns=torch.tensor([0.0, 0.0]),
        new_values=torch.tensor([0.0, 0.0]),
        action_mask=torch.tensor([1.0, 0.0]),
        clip_eps=0.2,
    )

    assert result["clip_fraction"].item() == pytest.approx(0.0)
    assert result["ratio_mean"].item() == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_train_ppo.py -k diagnostic_fields -v`
Expected: FAIL with `KeyError: 'clip_fraction'` (the field doesn't exist yet).

- [ ] **Step 3: Add the three diagnostic fields to `compute_ppo_loss`**

Replace `src/turn_level_rewards/train_ppo.py:194-206`:

```python
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -_masked_mean(torch.min(unclipped, clipped), action_mask)
    value_loss = _masked_mean((new_values - returns) ** 2, action_mask)
    kl = _masked_mean(new_logprobs - old_logprobs, action_mask)
    loss = policy_loss + value_loss_coef * value_loss + kl_beta * kl

    # Diagnostic-only fields (no gradient contribution) -- mirror TRL's own PPOTrainer metric
    # set (`val/ratio`, `val/ratio_var`, `policy/clipfrac_avg` -- see
    # docs/trl/v1.9.0/ppo_trainer's "Explanation of the logged metrics"), the standard PPO
    # health signals for diagnosing an unstable run from trackio curves alone. clip_fraction
    # uses the raw (pre-clip) ratio against the same clip_eps bounds compute_ppo_loss itself
    # clips to, so it reports exactly what fraction of this batch's surrogate objective was
    # actually clamped.
    ratio_mean = _masked_mean(ratio, action_mask)
    ratio_variance = _masked_mean((ratio - ratio_mean) ** 2, action_mask)
    is_clipped = (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)
    clip_fraction = _masked_mean(is_clipped.float(), action_mask)

    return {
        "loss": loss,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "kl": kl.detach(),
        "ratio_mean": ratio_mean.detach(),
        "ratio_variance": ratio_variance.detach(),
        "clip_fraction": clip_fraction.detach(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_train_ppo.py -k "diagnostic_fields or ratio_mean" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full existing `compute_ppo_loss` test suite to confirm no regression**

Run: `uv run pytest tests/unit/test_train_ppo.py -k compute_ppo_loss -v`
Expected: all PASS (the four pre-existing tests plus the two new ones), since the new fields are
purely additive and none of `loss`/`policy_loss`/`value_loss`/`kl`'s existing computation changed.

- [ ] **Step 6: Propagate the new fields through `_ppo_update`**

In `src/turn_level_rewards/train_ppo.py:647`, replace:

```python
        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0}
```

with:

```python
        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "kl": 0.0,
            "ratio_mean": 0.0,
            "ratio_variance": 0.0,
            "clip_fraction": 0.0,
        }
```

Then in `src/turn_level_rewards/train_ppo.py:718-723` (immediately after `real_value_loss =
value_loss_dict["value_loss"]`), replace:

```python
                totals["policy_loss"] += real_policy_loss.item()
                totals["kl"] += real_kl.item()
                totals["value_loss"] += real_value_loss.item()
```

with:

```python
                # ratio_mean/ratio_variance/clip_fraction come from pass 1 (real new_logprobs)
                # for the same reason real_policy_loss/real_kl do -- pass 2's ratio is always
                # exactly 1.0 (old_logprobs vs itself), which would silently dilute these
                # diagnostics toward "everything looks fine" if averaged in.
                totals["policy_loss"] += real_policy_loss.item()
                totals["kl"] += real_kl.item()
                totals["value_loss"] += real_value_loss.item()
                totals["ratio_mean"] += policy_loss_dict["ratio_mean"].item()
                totals["ratio_variance"] += policy_loss_dict["ratio_variance"].item()
                totals["clip_fraction"] += policy_loss_dict["clip_fraction"].item()
```

- [ ] **Step 7: Add trackio/train_log logging for the new metrics in `train()`**

In `src/turn_level_rewards/train_ppo.py:829-837`, replace:

```python
            metrics = {
                "step": step,
                "loss": update_metrics["loss"],
                "policy_loss": update_metrics["policy_loss"],
                "value_loss": update_metrics["value_loss"],
                "kl": update_metrics["kl"],
                "reward": mean_reward,
                "retrieval_fraction": mean_retrieval_fraction,
            }
            trackio.log(metrics)
```

with:

```python
            metrics = {
                "step": step,
                "loss": update_metrics["loss"],
                "policy_loss": update_metrics["policy_loss"],
                "value_loss": update_metrics["value_loss"],
                "kl": update_metrics["kl"],
                "ratio_mean": update_metrics["ratio_mean"],
                "ratio_variance": update_metrics["ratio_variance"],
                "clip_fraction": update_metrics["clip_fraction"],
                "reward": mean_reward,
                "retrieval_fraction": mean_retrieval_fraction,
            }
            trackio.log(metrics)
```

(`log_record = dict(metrics)` on the next line already copies these through into
`train_log.jsonl` unchanged -- no further edit needed there.)

- [ ] **Step 8: Run the full unit test file**

Run: `uv run pytest tests/unit/test_train_ppo.py -v`
Expected: all tests PASS.

- [ ] **Step 9: Commit**

```bash
git add src/turn_level_rewards/train_ppo.py tests/unit/test_train_ppo.py
git commit -m "train_ppo: add ratio/clip-fraction PPO health diagnostics to trackio logging"
```

---

### Task 3: Shared retrieval-fraction helper + per-episode format_reward tracking

**Files:**
- Modify: `src/turn_level_rewards/train_ppo.py:575-637` (`_collect_batch`)
- Test: `tests/unit/test_train_ppo.py`

**Interfaces:**
- Produces: a new module-level function `_final_retrieval_fraction(rollout: dict) -> float` (used
  by `_collect_batch` here, and reused by Task 6's `evaluate_ppo.py` without duplicating the
  extraction logic). Each episode dict in `_collect_batch`'s returned list gains a new
  `"format_reward"` key (the raw `format_r` float already computed inside the loop, previously
  discarded after being summed into `format_and_outcome_reward`). Task 4's `CollapseMonitor`
  consumes this new key from `train()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_train_ppo.py`:

```python
def test_final_retrieval_fraction_returns_zero_for_no_turns():
    assert _final_retrieval_fraction({"retrieval_fraction_after_each_turn": []}) == 0.0


def test_final_retrieval_fraction_returns_last_value():
    rollout = {"retrieval_fraction_after_each_turn": [0.5, 1.0]}
    assert _final_retrieval_fraction(rollout) == 1.0
```

Add `_final_retrieval_fraction` to the existing `from turn_level_rewards.train_ppo import (...)`
block at the top of the file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_train_ppo.py -k final_retrieval_fraction -v`
Expected: FAIL with `ImportError: cannot import name '_final_retrieval_fraction'`.

- [ ] **Step 3: Add `_final_retrieval_fraction` and use it in `_collect_batch`**

In `src/turn_level_rewards/train_ppo.py`, add this function immediately before `class
MTPPOTrainer(Trainer):` (i.e. right after `build_policy_and_critic`, before line 252):

```python
def _final_retrieval_fraction(rollout: dict) -> float:
    """The episode's final (cumulative) retrieval_fraction -- 0.0 if the episode made no search
    calls at all. Shared by _collect_batch (train-time) and evaluate_ppo.py (eval-time) so both
    extract this the same way.
    """
    fractions = rollout["retrieval_fraction_after_each_turn"]
    return fractions[-1] if fractions else 0.0
```

Then in `src/turn_level_rewards/train_ppo.py:596-600`, replace:

```python
            retrieval_fraction = (
                rollout["retrieval_fraction_after_each_turn"][-1]
                if rollout["retrieval_fraction_after_each_turn"]
                else 0.0
            )
```

with:

```python
            retrieval_fraction = _final_retrieval_fraction(rollout)
```

Then in `src/turn_level_rewards/train_ppo.py:623-636`, add `"format_reward": format_r,` to the
appended episode dict (immediately after the `"retrieval_fraction": retrieval_fraction,` line):

```python
            episodes.append(
                {
                    "full_token_ids": rollout["full_token_ids"],
                    "action_mask": rollout["action_mask"],
                    "old_logprobs": old_logprobs,
                    "old_values": old_values,
                    "advantages": torch.tensor(advantages, device=old_values.device),
                    "returns": torch.tensor(returns, device=old_values.device),
                    "format_and_outcome_reward": format_and_outcome_reward,
                    "format_reward": format_r,
                    "retrieval_fraction": retrieval_fraction,
                    "question": row["question"],
                    "completion": completion,
                }
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_train_ppo.py -k final_retrieval_fraction -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full unit test file**

Run: `uv run pytest tests/unit/test_train_ppo.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/turn_level_rewards/train_ppo.py tests/unit/test_train_ppo.py
git commit -m "train_ppo: extract _final_retrieval_fraction helper, track per-episode format_reward"
```

---

### Task 4: Collapse-visibility monitor

**Files:**
- Modify: `src/turn_level_rewards/train_ppo.py:765-881` (`train`, plus a new module-level class
  added before it)
- Test: `tests/unit/test_train_ppo.py`

**Interfaces:**
- Consumes: `episode["format_reward"]` (Task 3), `metrics["loss"]`/`metrics["reward"]` (already
  computed in `train()`).
- Produces: a new `_CollapseAlert` dataclass (`title: str`, `text: str`, `level: str`,
  `should_stop: bool = False`) and a new `CollapseMonitor` class with
  `.check(step: int, loss: float, mean_reward: float, mean_format_reward: float) ->
  list[_CollapseAlert]`. `train()` instantiates one `CollapseMonitor` per run, calls `.check(...)`
  once per step, and fires `trackio.alert(...)` for each returned alert (breaking the training
  loop, after saving a final checkpoint, if any alert has `should_stop=True`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_train_ppo.py`:

```python
def test_collapse_monitor_stops_on_non_finite_loss():
    monitor = CollapseMonitor()

    alerts = monitor.check(step=5, loss=float("nan"), mean_reward=1.0, mean_format_reward=0.1)

    assert len(alerts) == 1
    assert alerts[0].should_stop is True
    assert alerts[0].level == "ERROR"


def test_collapse_monitor_no_alerts_for_healthy_run():
    monitor = CollapseMonitor()

    for step in range(30):
        alerts = monitor.check(step=step, loss=0.5, mean_reward=1.0, mean_format_reward=0.1)
        assert alerts == []


def test_collapse_monitor_fires_dead_reward_alert_after_threshold():
    monitor = CollapseMonitor()

    alerts = []
    for step in range(25):
        alerts = monitor.check(step=step, loss=0.5, mean_reward=0.0, mean_format_reward=0.1)

    assert len(alerts) == 1
    assert alerts[0].should_stop is False
    assert "Dead reward" in alerts[0].title


def test_collapse_monitor_fires_format_collapse_alert_after_regression():
    """format_reward is healthy for the first 10 steps, then craters and stays there -- should
    alert once it's been non-positive for enough consecutive steps, but only because it
    regressed FROM a compliant state (this is a collapse signal, not a "never learned" signal,
    which test_collapse_monitor_fires_dead_reward_alert_after_threshold already covers).
    """
    monitor = CollapseMonitor()

    for step in range(10):
        alerts = monitor.check(step=step, loss=0.5, mean_reward=1.0, mean_format_reward=0.1)
        assert alerts == []

    alerts = []
    for step in range(10, 35):
        alerts = monitor.check(step=step, loss=0.5, mean_reward=-0.1, mean_format_reward=-0.1)

    assert len(alerts) == 1
    assert "Format compliance" in alerts[0].title
    assert alerts[0].should_stop is False


def test_collapse_monitor_each_alert_fires_only_once():
    monitor = CollapseMonitor()

    for step in range(25):
        alerts = monitor.check(step=step, loss=0.5, mean_reward=0.0, mean_format_reward=0.1)

    more_alerts = monitor.check(step=25, loss=0.5, mean_reward=0.0, mean_format_reward=0.1)
    assert more_alerts == []
```

Add `CollapseMonitor` to the existing import block from `turn_level_rewards.train_ppo`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_train_ppo.py -k collapse_monitor -v`
Expected: FAIL with `ImportError: cannot import name 'CollapseMonitor'`.

- [ ] **Step 3: Add `_CollapseAlert` and `CollapseMonitor`**

Add this near the top of `src/turn_level_rewards/train_ppo.py`, immediately after the
`_SAMPLE_COMPLETION_INTERVAL = 10` line:

```python
_DEAD_REWARD_STEP_THRESHOLD = 20
_FORMAT_COLLAPSE_STREAK_THRESHOLD = 20


@dataclass
class _CollapseAlert:
    title: str
    text: str
    level: str  # "ERROR" or "WARN"
    should_stop: bool = False


class CollapseMonitor:
    """Pure, testable collapse-visibility state machine for MTPPOTrainer.train()'s per-step loop.

    Mirrors train.py's TrackioAlertCallback pattern (non-finite loss stops training; reward
    stuck at 0 for too long is a dead-signal alert) but adapted for PPO -- there is no
    frac_reward_zero_std here (a GRPO group-relative-advantage concept this trainer has no
    equivalent of), and a new format-compliance-collapse check is added instead: the paper's own
    PPO-OR baselines are documented as prone to crashing (arXiv:2505.11821v2 Section 6.1,
    "PPO baselines often crash"), and this exists to make that visible live from the trackio
    curves alone, not to auto-rollback -- see
    docs/superpowers/specs/2026-07-23-phase-7b-full-ppo-runs-design.md's "policy collapse"
    decision for why an automated rollback would be inventing a mechanism the paper never
    describes.

    Returns _CollapseAlert instances instead of calling trackio.alert directly, so this class is
    unit-testable without a live trackio backend -- trackio.alert itself is the external seam,
    kept in train()'s caller (cosmicpython dependency-inversion, same principle CLAUDE.md's
    Guiding principles section already applies throughout this repo).
    """

    def __init__(self) -> None:
        self._reward_ever_nonzero = False
        self._dead_reward_alerted = False
        self._format_ever_compliant = False
        self._format_collapse_streak = 0
        self._format_collapse_alerted = False

    def check(
        self, step: int, loss: float, mean_reward: float, mean_format_reward: float
    ) -> list[_CollapseAlert]:
        if not math.isfinite(loss):
            return [
                _CollapseAlert(
                    title="Non-finite loss",
                    text=f"Loss is {loss} at step {step} -- stopping training.",
                    level="ERROR",
                    should_stop=True,
                )
            ]

        alerts: list[_CollapseAlert] = []

        if mean_reward != 0.0:
            self._reward_ever_nonzero = True
        if (
            not self._reward_ever_nonzero
            and not self._dead_reward_alerted
            and step >= _DEAD_REWARD_STEP_THRESHOLD
        ):
            alerts.append(
                _CollapseAlert(
                    title="Dead reward",
                    text=(
                        f"Reward has been exactly 0.0 for all {step + 1} steps so far -- "
                        "possible miswired reward function, tool-calling loop, or policy "
                        "collapse."
                    ),
                    level="ERROR",
                )
            )
            self._dead_reward_alerted = True

        if mean_format_reward > 0.0:
            self._format_ever_compliant = True
            self._format_collapse_streak = 0
            self._format_collapse_alerted = False
        elif self._format_ever_compliant:
            self._format_collapse_streak += 1
            if (
                self._format_collapse_streak >= _FORMAT_COLLAPSE_STREAK_THRESHOLD
                and not self._format_collapse_alerted
            ):
                alerts.append(
                    _CollapseAlert(
                        title="Format compliance collapsed",
                        text=(
                            f"format_reward has been non-positive for "
                            f"{self._format_collapse_streak} consecutive steps after previously "
                            "being compliant -- likely policy collapse (matches the paper's own "
                            "documented PPO-OR instability). Check trackio curves and consider "
                            "evaluating an earlier checkpoint."
                        ),
                        level="WARN",
                    )
                )
                self._format_collapse_alerted = True

        return alerts
```

Add `from dataclasses import dataclass` and `import math` to the top of `train_ppo.py` if not
already present (check the existing import block first -- `dataclass` is already imported for
`MTPPOConfig`; `math` is not yet imported).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_train_ppo.py -k collapse_monitor -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Wire `CollapseMonitor` into `train()`**

In `src/turn_level_rewards/train_ppo.py`, immediately after line 789 (`set_seed(self.args.seed)`),
add:

```python
        collapse_monitor = CollapseMonitor()
```

Then, immediately after the `trackio.log(metrics)` call at line 838, add:

```python
            mean_format_reward = sum(e["format_reward"] for e in episodes) / len(episodes)
            for alert in collapse_monitor.check(
                step=step,
                loss=metrics["loss"],
                mean_reward=mean_reward,
                mean_format_reward=mean_format_reward,
            ):
                trackio.alert(
                    title=alert.title,
                    text=alert.text,
                    level=trackio.AlertLevel.ERROR
                    if alert.level == "ERROR"
                    else trackio.AlertLevel.WARN,
                )
                if alert.should_stop:
                    self._save_full_checkpoint(f"{self.args.output_dir}/checkpoint-{step + 1}")
                    print(f"Stopping training early at step {step + 1}: {alert.title}", flush=True)
                    return
```

(`self._save_full_checkpoint` is added in Task 5 -- this task's edit references it, so Task 5
must land before this code path is exercised end-to-end; the unit tests for `CollapseMonitor`
itself don't depend on it, since they test the class directly, not `train()`.)

- [ ] **Step 6: Run the full unit test file**

Run: `uv run pytest tests/unit/test_train_ppo.py -v`
Expected: all tests PASS. (`train()` itself is not unit-tested per this repo's existing
convention -- this step confirms the new code doesn't break anything importable/collectable.)

- [ ] **Step 7: Commit**

```bash
git add src/turn_level_rewards/train_ppo.py tests/unit/test_train_ppo.py
git commit -m "train_ppo: add CollapseMonitor for live collapse-visibility trackio alerts"
```

---

### Task 5: Checkpoint-resume (infra-failure repeatability)

**Files:**
- Modify: `src/turn_level_rewards/train_ppo.py:44-63` (`MTPPOConfig`, `build_ppo_config`)
- Modify: `src/turn_level_rewards/train_ppo.py:736-881` (`_save_policy_and_critic` area, `train`)
- Modify: `src/turn_level_rewards/train_ppo.py:884-898` (`_parse_args`)
- Test: `tests/unit/test_train_ppo.py`

**Interfaces:**
- Produces: a new module-level function `_row_cycle_from_step(rows: list[dict], global_step:
  int, num_rollouts_per_step: int) -> Iterator[dict]`. Two new `MTPPOTrainer` methods:
  `_save_full_checkpoint(self, output_dir: str) -> None` and `_load_full_checkpoint(self,
  checkpoint_dir: str) -> None`. `MTPPOConfig` gains `save_steps` as an explicit field (was
  previously left at `TrainingArguments`'s own default). `_parse_args` gains
  `--resume-from-checkpoint` (str, default `None`) and `--save-steps` (int, default `50`).
  `train()`'s checkpoint-save call and startup sequence both change to use these.

- [ ] **Step 1: Write the failing test for `_row_cycle_from_step`**

Append to `tests/unit/test_train_ppo.py`:

```python
def test_row_cycle_from_step_starts_at_beginning_when_global_step_is_zero():
    rows = [{"id": 0}, {"id": 1}, {"id": 2}]
    cycle = _row_cycle_from_step(rows, global_step=0, num_rollouts_per_step=2)

    assert [next(cycle)["id"] for _ in range(3)] == [0, 1, 2]


def test_row_cycle_from_step_resumes_at_correct_position():
    rows = [{"id": 0}, {"id": 1}, {"id": 2}]
    # From scratch: step 0 consumes rows 0,1; step 1 consumes rows 2,0. After global_step=2
    # completed steps, 4 rows have been consumed (0,1,2,0) -- the next draw should be row 1.
    cycle = _row_cycle_from_step(rows, global_step=2, num_rollouts_per_step=2)

    assert next(cycle)["id"] == 1
    assert next(cycle)["id"] == 2
```

Add `_row_cycle_from_step` to the existing import block.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_train_ppo.py -k row_cycle_from_step -v`
Expected: FAIL with `ImportError: cannot import name '_row_cycle_from_step'`.

- [ ] **Step 3: Add `_row_cycle_from_step`**

Add this function immediately after `_final_retrieval_fraction` (added in Task 3), before `class
MTPPOTrainer(Trainer):`:

```python
def _row_cycle_from_step(
    rows: list[dict], global_step: int, num_rollouts_per_step: int
) -> Iterator[dict]:
    """An infinite cycle over rows, fast-forwarded to the position a from-scratch run would be
    at after global_step completed steps -- so resuming from a checkpoint reproduces the exact
    same data order a run that never stopped would have seen. itertools.cycle over a fixed list
    is deterministic, so replaying the same number of draws always lands in the same place.
    """
    cycle = itertools.cycle(rows)
    for _ in range(global_step * num_rollouts_per_step):
        next(cycle)
    return cycle
```

Add `from collections.abc import Iterator` to the top of `train_ppo.py`'s import block if not
already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_train_ppo.py -k row_cycle_from_step -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Add `save_steps` to `MTPPOConfig` and `build_ppo_config`**

In `src/turn_level_rewards/train_ppo.py:44-63`, add `save_steps: int = 50` as a field on
`MTPPOConfig` (after `num_rollouts_per_step: int = 2`), and add `save_steps=save_steps,` as a new
parameter/pass-through in `build_ppo_config` (mirroring how `num_rollouts_per_step` is already
threaded through):

```python
    condition: Condition = "ppo"
    n_max: int = 4
    clip_eps: float = 0.2
    kl_beta: float = 0.001
    policy_lr: float = 1e-6
    critic_lr: float = 1e-5
    gamma: float = 1.0
    gae_lambda: float = 1.0
    num_ppo_epochs: int = 4
    value_loss_coef: float = 0.5
    num_rollouts_per_step: int = 2
    save_steps: int = 50
    max_completion_length: int = 2048
    project: str = "turn-level-rewards-ppo"


def build_ppo_config(
    condition: Condition,
    seed: int,
    max_steps: int,
    num_rollouts_per_step: int,
    save_steps: int = 50,
) -> MTPPOConfig:
    """Build the MTPPOConfig for a training run. Mirrors train.py's build_config role from
    Phase 4 -- fixed hyperparameters are baked in here, not exposed as independent CLI flags.
    """
    return MTPPOConfig(
        output_dir=f"outputs/{condition}",
        seed=seed,
        max_steps=max_steps,
        condition=condition,
        n_max=4,
        clip_eps=0.2,
        kl_beta=0.001,
        policy_lr=1e-6,
        critic_lr=1e-5,
        gamma=1.0,
        gae_lambda=1.0,
        num_ppo_epochs=4,
        value_loss_coef=0.5,
        num_rollouts_per_step=num_rollouts_per_step,
        save_steps=save_steps,
        max_completion_length=2048,
        logging_steps=1,
        run_name=condition,
        report_to="none",  # trackio is called directly in MTPPOTrainer.train(), not through
        # transformers' generic report_to integration -- see Task 8.
    )
```

- [ ] **Step 6: Run existing `build_ppo_config` tests to check for regressions**

Run: `uv run pytest tests/unit/test_train_ppo.py -k build_ppo_config -v`
Expected: PASS unchanged -- `save_steps` is additive with a default, no existing call site or
assertion references it.

- [ ] **Step 7: Add `_save_full_checkpoint` and `_load_full_checkpoint`**

Add these two methods to `MTPPOTrainer`, immediately after `_save_policy_and_critic` (currently
ending at line 764, right before `train()`'s `def train(self)` line):

```python
    def _save_full_checkpoint(self, output_dir: str) -> None:
        """Save everything needed to resume training exactly where it left off: policy+critic
        weights (via _save_policy_and_critic, which already correctly handles Qwen3.5's tied
        lm_head/embed_tokens weights), plus optimizer/RNG/step state via transformers.Trainer's
        own inherited primitives -- the same ones TRL's own PPOTrainer relies on (PPOConfig
        subclasses TrainingArguments directly and exposes resume_from_checkpoint/save_steps,
        confirmed against trl==1.9.0). Not unit-tested: _save_optimizer_and_scheduler and
        _save_rng_state are inherited, untouched Trainer methods that write real optimizer
        state/RNG snapshots to disk -- validated by Task 7's live smoke test instead. See
        docs/superpowers/specs/2026-07-23-phase-7b-full-ppo-runs-design.md.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._save_policy_and_critic(output_dir)
        self._save_optimizer_and_scheduler(output_dir)  # ty: ignore[unresolved-attribute]
        self._save_rng_state(output_dir)  # ty: ignore[unresolved-attribute]
        self.state.save_to_json(str(Path(output_dir) / "trainer_state.json"))

    def _load_full_checkpoint(self, checkpoint_dir: str) -> None:
        """Inverse of _save_full_checkpoint -- reloads policy+critic weights, optimizer/RNG
        state, and global_step, so train() can resume exactly where a prior run left off. Not
        unit-tested: loads real weights and real optimizer/RNG state from disk -- validated by
        Task 7's live smoke test instead, same convention as build_policy_and_critic.
        """
        device = self.model.policy.device  # ty: ignore[unresolved-attribute]
        policy = AutoModelForCausalLM.from_pretrained(
            str(Path(checkpoint_dir) / "policy"), dtype=torch.bfloat16
        ).to(device)
        critic = AutoModelForSequenceClassification.from_pretrained(
            str(Path(checkpoint_dir) / "critic"), num_labels=1, dtype=torch.bfloat16
        ).to(device)
        policy.gradient_checkpointing_enable()
        critic.gradient_checkpointing_enable()
        self.model.policy = policy  # ty: ignore[unresolved-attribute]
        self.model.critic = critic  # ty: ignore[unresolved-attribute]
        self._load_optimizer_and_scheduler(checkpoint_dir)  # ty: ignore[unresolved-attribute]
        self._load_rng_state(checkpoint_dir)  # ty: ignore[unresolved-attribute]
        self.state = TrainerState.load_from_json(str(Path(checkpoint_dir) / "trainer_state.json"))
```

Add `from transformers.trainer_callback import TrainerState` and `from transformers.trainer_utils
import get_last_checkpoint` to the top of `train_ppo.py`'s import block.

- [ ] **Step 8: Rewire `train()` to use `_save_full_checkpoint` and support resume**

In `src/turn_level_rewards/train_ppo.py:789-797`, replace:

```python
        set_seed(self.args.seed)
        self.optimizer = self.create_optimizer()
        # self.train_dataset is typed by Trainer's base class as
        # `torch.utils.data.Dataset | datasets.arrow_dataset.Dataset | None`, and
        # torch.utils.data.Dataset's stub doesn't declare __iter__, so ty can't confirm it's
        # Iterable -- but build_ppo_trainer (Task 11) always constructs this trainer with a real
        # datasets.Dataset, which genuinely is iterable at runtime.
        rows = list(self.train_dataset)  # ty: ignore[invalid-argument-type]
        row_cycle = itertools.cycle(rows)
        trackio.init(project=self.args.project, name=self.args.run_name)
```

with:

```python
        set_seed(self.args.seed)
        self.optimizer = self.create_optimizer()
        if self.args.resume_from_checkpoint:  # ty: ignore[unresolved-attribute]
            self._load_full_checkpoint(self.args.resume_from_checkpoint)  # ty: ignore[unresolved-attribute]
        collapse_monitor = CollapseMonitor()
        # self.train_dataset is typed by Trainer's base class as
        # `torch.utils.data.Dataset | datasets.arrow_dataset.Dataset | None`, and
        # torch.utils.data.Dataset's stub doesn't declare __iter__, so ty can't confirm it's
        # Iterable -- but build_ppo_trainer (Task 11) always constructs this trainer with a real
        # datasets.Dataset, which genuinely is iterable at runtime.
        rows = list(self.train_dataset)  # ty: ignore[invalid-argument-type]
        row_cycle = _row_cycle_from_step(
            rows,
            self.state.global_step,
            self.args.num_rollouts_per_step,  # ty: ignore[unresolved-attribute]
        )
        trackio.init(project=self.args.project, name=self.args.run_name)
```

(Note: this replaces the `collapse_monitor = CollapseMonitor()` line Task 4 added right after
`set_seed` -- Task 5 supersedes that exact line by moving it one line later, after the resume
check, so `CollapseMonitor`'s own history isn't reset by a resume. If executing Task 4 and Task 5
in the same session, apply Task 5's version of this block instead of duplicating the line.)

Then in `src/turn_level_rewards/train_ppo.py:809`, replace:

```python
        for step in range(self.args.max_steps):
```

with:

```python
        for step in range(self.state.global_step, self.args.max_steps):
```

Then in `src/turn_level_rewards/train_ppo.py:878-881`, replace:

```python
            if (step + 1) % self.args.save_steps == 0 if self.args.save_steps else False:
                self._save_policy_and_critic(f"{self.args.output_dir}/checkpoint-{step + 1}")

        self._save_policy_and_critic(f"{self.args.output_dir}/checkpoint-{self.args.max_steps}")
```

with:

```python
            if (step + 1) % self.args.save_steps == 0:  # ty: ignore[unresolved-attribute]
                self._save_full_checkpoint(f"{self.args.output_dir}/checkpoint-{step + 1}")

        self._save_full_checkpoint(f"{self.args.output_dir}/checkpoint-{self.args.max_steps}")
```

(`save_steps` is now always a real int, per Step 5's `MTPPOConfig` default of `50` -- the old
`if self.args.save_steps else False` guard existed only because `TrainingArguments`'s own
default could be falsy in edge cases; that's no longer a concern now that `MTPPOConfig` always
sets a concrete value.)

- [ ] **Step 9: Add `--resume-from-checkpoint` and `--save-steps` CLI flags**

In `src/turn_level_rewards/train_ppo.py:884-898` (`_parse_args`), add two new arguments after
`--num-rollouts-per-step`:

```python
    parser.add_argument("--num-rollouts-per-step", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a checkpoint-N directory to resume from, or 'auto' to resume from the "
        "latest checkpoint under this run's output_dir.",
    )
    return parser.parse_args(argv)
```

Then update `main()` (currently at line 925) to thread these through: after `config =
build_ppo_config(...)`, add:

```python
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint == "auto":
        resume_from_checkpoint = get_last_checkpoint(config.output_dir)
        if resume_from_checkpoint is None:
            raise ValueError(f"--resume-from-checkpoint auto: no checkpoint found under {config.output_dir}")
    config.resume_from_checkpoint = resume_from_checkpoint
```

and pass `save_steps=args.save_steps` into the existing `build_ppo_config(...)` call.

- [ ] **Step 10: Write the failing test for the new CLI flags**

Append to `tests/unit/test_train_ppo.py`:

```python
def test_parse_args_resume_and_save_steps_defaults():
    args = _parse_args(["--condition", "ppo"])

    assert args.save_steps == 50
    assert args.resume_from_checkpoint is None


def test_parse_args_resume_and_save_steps_overrides():
    args = _parse_args(
        [
            "--condition",
            "ppo",
            "--save-steps",
            "10",
            "--resume-from-checkpoint",
            "outputs/ppo/checkpoint-100",
        ]
    )

    assert args.save_steps == 10
    assert args.resume_from_checkpoint == "outputs/ppo/checkpoint-100"
```

- [ ] **Step 11: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_train_ppo.py -k "resume_and_save_steps" -v`
Expected: PASS (2 tests).

- [ ] **Step 12: Run the full unit test file**

Run: `uv run pytest tests/unit/test_train_ppo.py -v`
Expected: all tests PASS.

- [ ] **Step 13: Commit**

```bash
git add src/turn_level_rewards/train_ppo.py tests/unit/test_train_ppo.py
git commit -m "train_ppo: add checkpoint-resume via Trainer's own optimizer/RNG/state primitives"
```

---

### Task 6: `evaluate_ppo.py` -- held-out evaluation for MTPPOTrainer checkpoints

**Files:**
- Create: `src/turn_level_rewards/evaluate_ppo.py`
- Test: `tests/unit/test_evaluate_ppo.py`

**Interfaces:**
- Consumes: `turn_level_rewards.rewards.outcome_reward` (existing, unchanged),
  `turn_level_rewards.train_ppo._final_retrieval_fraction` (Task 3),
  `turn_level_rewards.train_ppo._PolicyAndCritic`/`build_ppo_config` (existing).
- Produces: a pure function `aggregate_eval_metrics(completions: list[Completion],
  golden_answers_list: list[list[str]], retrieval_fractions: list[float]) -> dict[str, float]`
  (unit-tested with fakes, no model needed) and a CLI (`python -m
  turn_level_rewards.evaluate_ppo --condition ... --checkpoint-dir ...`) mirroring `evaluate.py`'s
  shape, writing a JSON metrics file.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_evaluate_ppo.py`:

```python
"""Fast, GPU-free tests for evaluate_ppo.py's pure aggregation logic.

The real rollout loop (loading a checkpoint, calling MTPPOTrainer._rollout_episode) requires a
real model/GPU -- not unit-tested here, validated by the live smoke test instead, same
convention as train_ppo.py's own untested methods.
"""

from turn_level_rewards.evaluate_ppo import aggregate_eval_metrics


def test_aggregate_eval_metrics_computes_means():
    completions = [
        [{"role": "assistant", "content": "<answer>Paris</answer>"}],
        [{"role": "assistant", "content": "<answer>wrong</answer>"}],
    ]
    golden_answers_list = [["Paris"], ["London"]]
    retrieval_fractions = [1.0, 0.5]

    metrics = aggregate_eval_metrics(completions, golden_answers_list, retrieval_fractions)

    assert metrics["exact_match"] == 0.5  # first correct, second not
    assert metrics["retrieval_fraction"] == 0.75  # mean of 1.0, 0.5
    assert metrics["num_examples"] == 2
    assert 0.0 <= metrics["f1"] <= 1.0


def test_aggregate_eval_metrics_perfect_run():
    completions = [[{"role": "assistant", "content": "<answer>Paris</answer>"}]]
    golden_answers_list = [["Paris"]]
    retrieval_fractions = [1.0]

    metrics = aggregate_eval_metrics(completions, golden_answers_list, retrieval_fractions)

    assert metrics["exact_match"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["retrieval_fraction"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_evaluate_ppo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'turn_level_rewards.evaluate_ppo'`.

- [ ] **Step 3: Create `evaluate_ppo.py`**

Create `src/turn_level_rewards/evaluate_ppo.py`:

```python
"""evaluate_ppo.py: run a saved MTPPOTrainer checkpoint over the held-out set.

Unlike evaluate.py (which constructs a GRPOTrainer and calls its standard .evaluate()),
MTPPOTrainer has no equivalent built-in evaluation path -- its rollout loop
(_rollout_episode/_collect_batch) is hand-built, not routed through Trainer's
prediction_step/evaluate() machinery. This reuses _rollout_episode directly under
torch.no_grad(), skipping _ppo_update entirely, against the held-out hotpot_qa validation set.

Takes --checkpoint-dir as a required argument (not hardcoded to a "final" checkpoint)
specifically so it can evaluate whichever checkpoint gets judged best from the trackio curves --
final, or last-before-collapse -- matching the paper's own stated evaluation methodology
(arXiv:2505.11821v2 Section 6.1) rather than assuming every run finishes cleanly. See
docs/superpowers/specs/2026-07-23-phase-7b-full-ppo-runs-design.md.
"""

import argparse
import json
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from turn_level_rewards import data
from turn_level_rewards.rewards import Completion, outcome_reward
from turn_level_rewards.train_ppo import (
    MODEL_NAME,
    MTPPOTrainer,
    _final_retrieval_fraction,
    _PolicyAndCritic,
    build_ppo_config,
)

Condition = Literal["ppo", "mt_ppo"]


def aggregate_eval_metrics(
    completions: list[Completion],
    golden_answers_list: list[list[str]],
    retrieval_fractions: list[float],
) -> dict[str, float]:
    """Turn a batch of rollout results into summary metrics, reusing outcome_reward's own
    per-example exact_match/f1 selection logic (via its log_metric callback) so eval-time
    scoring is byte-for-byte identical to training-time scoring -- no separate reimplementation
    of the "max over golden_answers" argmax logic to drift out of sync with rewards.py.
    """
    logged: dict[str, list[float]] = {"exact_match": [], "f1": []}

    def log_metric(name: str, value: float) -> None:
        logged[name].append(value)

    outcome_reward(completions, golden_answers_list, log_metric=log_metric)

    num_examples = len(completions)
    return {
        "exact_match": sum(logged["exact_match"]) / num_examples,
        "f1": sum(logged["f1"]) / num_examples,
        "retrieval_fraction": sum(retrieval_fractions) / num_examples,
        "num_examples": num_examples,
    }


def build_eval_trainer(condition: Condition, checkpoint_dir: str, seed: int) -> MTPPOTrainer:
    """Composition root: real policy+critic loaded from checkpoint_dir, real SearchEnv (hits the
    live retrieval server), real tokenizer. Not unit-tested -- this is exactly the integration
    surface the live smoke test validates, matching train_ppo.py's build_ppo_trainer.
    """
    policy = AutoModelForCausalLM.from_pretrained(
        str(Path(checkpoint_dir) / "policy"), dtype=torch.bfloat16
    )
    critic = AutoModelForSequenceClassification.from_pretrained(
        str(Path(checkpoint_dir) / "critic"), num_labels=1, dtype=torch.bfloat16
    )
    model = _PolicyAndCritic(policy, critic)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = build_ppo_config(condition=condition, seed=seed, max_steps=0, num_rollouts_per_step=1)
    # Separate output_dir from the real training run's -- mirrors evaluate.py's own
    # "eval-scratch" convention (Phase 6), so this never touches real training checkpoints even
    # though .train()/_save_full_checkpoint are never actually called from this path.
    config.output_dir = f"outputs/{condition}/ppo-eval-scratch"
    # eval_dataset stands in as train_dataset here -- MTPPOTrainer requires one at construction,
    # but .train() is never called, only _rollout_episode directly (see run_eval below).
    eval_dataset = data.load_eval_dataset(n=1, seed=seed)
    return MTPPOTrainer(
        condition=condition, model=model, tokenizer=tokenizer, train_dataset=eval_dataset, args=config
    )


def run_eval(trainer: MTPPOTrainer, rows: list[dict]) -> dict[str, float]:
    """Roll out every row under torch.no_grad() (no _ppo_update -- pure inference), then
    aggregate. Not unit-tested -- calls _rollout_episode (real model/tool-calls); validated by
    the live smoke test instead.
    """
    completions = []
    golden_answers_list = []
    retrieval_fractions = []
    with torch.no_grad():
        for row in rows:
            rollout = trainer._rollout_episode(row)
            completions.append(rollout["completion"])
            golden_answers_list.append(row["golden_answers"])
            retrieval_fractions.append(_final_retrieval_fraction(rollout))
    return aggregate_eval_metrics(completions, golden_answers_list, retrieval_fractions)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved MTPPOTrainer checkpoint on the held-out set (see CLAUDE.md)."
    )
    parser.add_argument("--condition", required=True, choices=["ppo", "mt_ppo"])
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-size", type=int, default=4)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    trainer = build_eval_trainer(args.condition, args.checkpoint_dir, args.seed)
    rows = list(data.load_eval_dataset(n=args.eval_size, seed=args.seed))
    metrics = run_eval(trainer, rows)

    output_path = Path(args.output or f"results/{args.condition}_ppo_eval_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote eval metrics to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_evaluate_ppo.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full unit test suite**

Run: `uv run pytest tests/unit/ -v`
Expected: all tests PASS (both `test_train_ppo.py` and the new `test_evaluate_ppo.py`).

- [ ] **Step 6: Commit**

```bash
git add src/turn_level_rewards/evaluate_ppo.py tests/unit/test_evaluate_ppo.py
git commit -m "Add evaluate_ppo.py: held-out evaluation for MTPPOTrainer checkpoints"
```

---

### Task 7: Live smoke test -- verify resume, diagnostics, and collapse-visibility end-to-end

**Files:**
- Modify: `scripts/verify_phase7.py` (extend with new checks) -- if this file doesn't exist or
  doesn't have an obvious extension point, create `scripts/verify_phase7b.py` instead, following
  the same pattern.

**Interfaces:**
- Consumes: everything from Tasks 2-6 (real GPU run of `train_ppo.py` and `evaluate_ppo.py`).
- Produces: a real, observed confirmation (not assumed) that a full run can be interrupted and
  resumed without losing progress or diverging from what an uninterrupted run would have done.

- [ ] **Step 1: Run a short training run to completion, saving checkpoints frequently**

Run: `uv run python -m turn_level_rewards.train_ppo --condition ppo --train-size 32 --max-steps 10 --num-rollouts-per-step 2 --save-steps 3 --seed 42`
Expected: completes all 10 steps, `outputs/ppo/checkpoint-3`, `checkpoint-6`, `checkpoint-9`,
`checkpoint-10` all exist, each containing `policy/`, `critic/`, `optimizer.pt`, `rng_state.pth`,
`trainer_state.json`.

- [ ] **Step 2: Verify the new diagnostic metrics appear in `train_log.jsonl`**

Run: `python -c "import json; lines=[json.loads(l) for l in open('outputs/ppo/train_log.jsonl')]; assert all('ratio_mean' in l and 'clip_fraction' in l for l in lines); print('OK:', len(lines), 'steps')"`
Expected: prints `OK: 10 steps` with no assertion error.

- [ ] **Step 3: Resume from an earlier checkpoint and confirm it continues, not restarts**

Run: `rm -rf outputs/ppo_resume_test && cp -r outputs/ppo outputs/ppo_resume_test`

Truncate the copy's log to simulate a crash after step 6, then resume:

Run: `uv run python -m turn_level_rewards.train_ppo --condition ppo --train-size 32 --max-steps 10 --num-rollouts-per-step 2 --save-steps 3 --seed 42 --resume-from-checkpoint outputs/ppo_resume_test/checkpoint-6`

Expected: stdout's first printed step line reads `step 7/10` (not `step 1/10`) -- confirms
`self.state.global_step` was correctly restored to 6 and the loop range started there, not from
0.

- [ ] **Step 4: Confirm resumed rollouts match a from-scratch run's data order (reproducibility)**

Compare the `question` field of episode 0 at step 7 in both `outputs/ppo/train_log.jsonl` (the
original uninterrupted run) and `outputs/ppo_resume_test/train_log.jsonl` (the resumed run):

Run: `python -c "
import json
def question_at(path, step):
    for line in open(path):
        rec = json.loads(line)
        if rec['step'] == step:
            return rec['episodes'][0]['question']
    raise ValueError(f'step {step} not found')
original = question_at('outputs/ppo/train_log.jsonl', 6)
resumed = question_at('outputs/ppo_resume_test/train_log.jsonl', 6)
assert original == resumed, f'{original!r} != {resumed!r}'
print('OK: resumed run reproduced the same data order')
"`
Expected: prints `OK: resumed run reproduced the same data order`.

- [ ] **Step 5: Run `evaluate_ppo.py` against one of the saved checkpoints**

Run: `uv run python -m turn_level_rewards.evaluate_ppo --condition ppo --checkpoint-dir outputs/ppo/checkpoint-10 --eval-size 4 --output /tmp/ppo_eval_smoke.json`
Expected: completes without error, `/tmp/ppo_eval_smoke.json` contains `exact_match`, `f1`,
`retrieval_fraction`, `num_examples` keys with finite float/int values.

- [ ] **Step 6: Repeat Steps 1-2 for `mt_ppo`**

Run: `uv run python -m turn_level_rewards.train_ppo --condition mt_ppo --train-size 32 --max-steps 10 --num-rollouts-per-step 2 --save-steps 3 --seed 42`
Expected: same checks pass -- `mt_ppo` reward placement is untouched by this plan, so this is a
regression check confirming the diagnostics/resume additions didn't break the `mt_ppo` path.

- [ ] **Step 7: Clean up scratch outputs**

Run: `rm -rf outputs/ppo_resume_test /tmp/ppo_eval_smoke.json`

- [ ] **Step 8: Record results in this repo's phase doc**

Update `docs/phase-7b-full-ppo-runs.md`'s Handoff notes (or add a new "Infrastructure verified"
subsection if the Handoff notes section is still the placeholder) with: confirmation that resume
reproduces identical data order, the checkpoint directory contents observed, and a pointer to
Task 1's wall-clock probe write-up for the numbers the actual full run will use.

- [ ] **Step 9: Commit**

```bash
git add docs/phase-7b-full-ppo-runs.md
git commit -m "docs: record Phase 7b infrastructure live-smoke-test results (resume, diagnostics, eval path)"
```
