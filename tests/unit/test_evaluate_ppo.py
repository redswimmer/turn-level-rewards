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
