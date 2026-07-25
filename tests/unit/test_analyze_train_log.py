"""Fast, GPU-free tests for analyze_train_log.py's pure analysis functions."""

from scripts.analyze_train_log import (
    episode_token_stats,
    find_outlier_episodes,
    find_skips,
    load_records,
    skip_rate_by_window,
)


def _step(step: int, tokens: list[int]) -> dict:
    return {
        "step": step,
        "episodes": [{"question": f"q{i}", "num_action_tokens": t} for i, t in enumerate(tokens)],
    }


def _skip(step: int) -> dict:
    return {
        "step": step,
        "skipped": True,
        "reason": "OutOfMemoryError: x",
        "questions_attempted": ["q"],
    }


def test_episode_token_stats_ignores_skipped_records():
    records = [_step(0, [10, 20]), _skip(1), _step(2, [30])]

    stats = episode_token_stats(records)

    assert stats["count"] == 3
    assert stats["min"] == 10
    assert stats["max"] == 30
    assert stats["median"] == 20


def test_episode_token_stats_empty_when_all_skipped():
    assert episode_token_stats([_skip(0), _skip(1)]) == {}


def test_find_outlier_episodes_uses_median_multiplier():
    # median of [10, 10, 10, 300] is 10 -> 3x threshold is 30, so only 300 is an outlier.
    records = [_step(0, [10, 10]), _step(1, [10, 300])]

    outliers = find_outlier_episodes(records, threshold_multiplier=3.0)

    assert len(outliers) == 1
    assert outliers[0]["num_action_tokens"] == 300
    assert outliers[0]["step"] == 1


def test_find_outlier_episodes_empty_when_no_records():
    assert find_outlier_episodes([]) == []


def test_find_skips_returns_only_skipped_records_in_order():
    records = [_step(0, [10]), _skip(1), _step(2, [10]), _skip(3)]

    skips = find_skips(records)

    assert [s["step"] for s in skips] == [1, 3]


def test_skip_rate_by_window_computes_fraction_per_window():
    records = [_step(0, [10]), _skip(1), _step(2, [10]), _step(3, [10])]

    windows = skip_rate_by_window(records, window=2)

    assert windows == [(0, 0.5), (2, 0.0)]


def test_load_records_skips_unparseable_lines(tmp_path):
    log_path = tmp_path / "train_log.jsonl"
    log_path.write_text('{"step": 0}\n{"step": 1, "incomple\n')

    records = load_records(str(log_path))

    assert records == [{"step": 0}]
