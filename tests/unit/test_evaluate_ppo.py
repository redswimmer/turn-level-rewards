"""Fast, GPU-free tests for evaluate_ppo.py's pure aggregation logic.

The real rollout loop (loading a checkpoint, calling MTPPOTrainer._rollout_episode) requires a
real model/GPU -- not unit-tested here, validated by the live smoke test instead, same
convention as train_ppo.py's own untested methods.
"""

import json

import pytest
from turn_level_rewards.evaluate_ppo import (
    aggregate_eval_metrics,
    merge_shard_metrics,
    shard_rows,
)


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
    # Both completions carry a well-formed <answer> tag, so both are format-compliant even
    # though only one is correct -- format and correctness are independent axes, which is
    # exactly why the paper reports them as separate columns.
    assert metrics["format_compliance_rate"] == 1.0


def test_aggregate_eval_metrics_counts_missing_answer_tag_as_noncompliant():
    """The collapsed-policy case: a completion with no parseable <answer> scores 0 on format.

    This is the metric that distinguishes "searched and answered wrong" from "never answered",
    which raw EM cannot -- both score EM 0.
    """
    completions = [
        [{"role": "assistant", "content": "<answer>Paris</answer>"}],
        [{"role": "assistant", "content": "I searched but never produced a tag"}],
    ]

    metrics = aggregate_eval_metrics(completions, [["Paris"], ["London"]], [1.0, 1.0])

    assert metrics["format_compliance_rate"] == 0.5
    assert metrics["exact_match"] == 0.5


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

    def _rollout_episode(self, row, greedy=False):
        # greedy is asserted, not merely accepted: evaluation must be deterministic, and a fake
        # that silently swallowed the flag would let a regression to sampled eval pass unnoticed.
        assert greedy is True, "run_eval must evaluate greedily"
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
        def _rollout_episode(self, row, greedy=False):
            if partial.exists():
                seen.append(json.loads(partial.read_text())["complete"])
            return super()._rollout_episode(row, greedy=greedy)

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


def test_shard_rows_partitions_every_row_exactly_once():
    """Sharding must be a partition: no row evaluated twice (which would double-weight it in the
    merged metric) and none dropped (which would silently shrink the held-out set).
    """
    rows = [{"i": i} for i in range(103)]  # deliberately not divisible by num_shards

    shards = [shard_rows(rows, shard=s, num_shards=4) for s in range(4)]

    recovered = sorted(r["i"] for shard in shards for r in shard)
    assert recovered == list(range(103))


def test_shard_rows_balances_length_within_one_row():
    """Strided (not contiguous) so shards get an even mix of short and long episodes -- a
    contiguous split would let one shard draw a run of long ones and straggle.
    """
    rows = [{"i": i} for i in range(103)]

    sizes = [len(shard_rows(rows, shard=s, num_shards=4)) for s in range(4)]

    assert max(sizes) - min(sizes) <= 1


def test_shard_rows_single_shard_is_the_whole_set():
    rows = [{"i": i} for i in range(10)]

    assert shard_rows(rows, shard=0, num_shards=1) == rows


def test_shard_rows_rejects_an_out_of_range_shard_index():
    """Off-by-one here would silently evaluate an empty set and report metrics of 0.0."""
    rows = [{"i": i} for i in range(10)]

    with pytest.raises(ValueError, match="shard"):
        shard_rows(rows, shard=4, num_shards=4)


def test_merge_shard_metrics_weights_by_example_count():
    """Shards can differ in size by one row, so a plain mean of shard means is wrong."""
    merged = merge_shard_metrics(
        [
            {
                "exact_match": 1.0,
                "f1": 1.0,
                "format_compliance_rate": 1.0,
                "retrieval_fraction": 0.5,
                "num_examples": 90,
            },
            {
                "exact_match": 0.0,
                "f1": 0.0,
                "format_compliance_rate": 0.0,
                "retrieval_fraction": 1.0,
                "num_examples": 10,
            },
        ]
    )

    assert merged["num_examples"] == 100
    assert merged["exact_match"] == pytest.approx(0.9)
    assert merged["retrieval_fraction"] == pytest.approx(0.55)
    assert merged["format_compliance_rate"] == pytest.approx(0.9)


def test_merge_shard_metrics_refuses_incomplete_shards():
    """A partial shard file must never be silently merged as if it were a finished result --
    that would report a full-set number computed from a fraction of the set.
    """
    with pytest.raises(ValueError, match="incomplete"):
        merge_shard_metrics(
            [
                {
                    "exact_match": 1.0,
                    "f1": 1.0,
                    "retrieval_fraction": 0.5,
                    "num_examples": 50,
                    "complete": True,
                },
                {
                    "exact_match": 0.0,
                    "f1": 0.0,
                    "retrieval_fraction": 0.5,
                    "num_examples": 10,
                    "complete": False,
                },
            ]
        )


def test_merge_shard_metrics_refuses_an_empty_list():
    with pytest.raises(ValueError, match="no shard"):
        merge_shard_metrics([])
