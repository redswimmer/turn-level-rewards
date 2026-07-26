"""Fast, GPU-free tests for evaluate_ppo.py's pure aggregation logic.

The real rollout loop (loading a checkpoint, calling MTPPOTrainer._rollout_episode) requires a
real model/GPU -- not unit-tested here, validated by the live smoke test instead, same
convention as train_ppo.py's own untested methods.
"""

import json

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


class _FakeTrainer:
    """Stands in for MTPPOTrainer at run_eval's only seam: _rollout_episode(row) -> rollout dict.

    Injecting here rather than mocking internals keeps this test fast and GPU-free while still
    exercising the real progress/partial-write path (cosmicpython dependency inversion, per
    CLAUDE.md's Guiding principles).
    """

    def __init__(self):
        self.calls = 0

    def _rollout_episode(self, row):
        self.calls += 1
        return {
            "completion": [
                {"role": "assistant", "content": f"<answer>{row['golden_answers'][0]}</answer>"}
            ],
            "retrieval_fraction_after_each_turn": [1.0],
        }


def _rows(n):
    return [{"golden_answers": ["paris"], "prompt": []} for _ in range(n)]


def test_run_eval_writes_partial_metrics_before_finishing(tmp_path):
    """A 19-hour eval that emits nothing until the end is unusable: a crash costs every row, and
    a working job is indistinguishable from a hung one. Partial writes fix both.
    """
    from turn_level_rewards.evaluate_ppo import run_eval

    partial = tmp_path / "eval_metrics.json"
    metrics = run_eval(_FakeTrainer(), _rows(10), report_every=2, partial_path=partial)

    assert partial.exists()
    assert metrics["num_examples"] == 10
    assert json.loads(partial.read_text())["complete"] is True


def test_run_eval_marks_partial_writes_incomplete_until_the_last_row(tmp_path):
    """The `complete` flag is what stops a partial file being mistaken for a finished result."""
    from turn_level_rewards.evaluate_ppo import run_eval

    partial = tmp_path / "eval_metrics.json"
    seen = []

    class _Watching(_FakeTrainer):
        def _rollout_episode(self, row):
            if partial.exists():
                seen.append(json.loads(partial.read_text())["complete"])
            return super()._rollout_episode(row)

    run_eval(_Watching(), _rows(6), report_every=2, partial_path=partial)

    assert seen  # partial file was written and observed mid-run
    assert all(flag is False for flag in seen)


def test_run_eval_without_a_partial_path_writes_nothing(tmp_path):
    """Partial writing is opt-in -- the smoke-eval path should not litter files."""
    from turn_level_rewards.evaluate_ppo import run_eval

    run_eval(_FakeTrainer(), _rows(4), report_every=2, partial_path=None)

    assert list(tmp_path.iterdir()) == []


def test_run_eval_rolls_out_every_row_exactly_once():
    from turn_level_rewards.evaluate_ppo import run_eval

    trainer = _FakeTrainer()
    run_eval(trainer, _rows(7), report_every=3)

    assert trainer.calls == 7
