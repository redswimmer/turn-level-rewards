import pytest
from turn_level_rewards.rewards import (
    format_reward,
    get_reward_funcs,
    length_penalty,
    outcome_reward,
    search_count_penalty,
    turn_reward,
)


class FakeEnvironment:
    def __init__(self, retrieval_fraction: float) -> None:
        self.retrieval_fraction = retrieval_fraction


class _FakeLogMetric:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def __call__(self, name: str, value: float) -> None:
        self.calls.append((name, value))


def _search_tool_call(query: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "search", "arguments": {"query": query}}}
        ],
    }


def _tool_response(content: str) -> dict:
    return {"role": "tool", "name": "search", "content": content}


def _answer(text: str) -> dict:
    return {"role": "assistant", "content": f"<answer>{text}</answer>"}


def test_well_formed_correct_answer_full_retrieval():
    completions = [
        [
            _search_tool_call("127 hours"),
            _tool_response('Doc 1 (Title: "127 Hours"): A 2010 film.'),
            _answer("127 Hours"),
        ]
    ]
    golden_answers = [["127 Hours"]]
    environments = [FakeEnvironment(retrieval_fraction=1.0)]

    format_reward, outcome_reward, turn_reward = get_reward_funcs("turn_level")

    assert format_reward(completions=completions) == pytest.approx([0.1])
    assert outcome_reward(completions=completions, golden_answers=golden_answers) == pytest.approx(
        [1.5]
    )
    assert turn_reward(completions=completions, environments=environments) == pytest.approx([0.4])


def test_well_formed_correct_answer_zero_retrieval():
    completions = [[_answer("127 Hours")]]
    golden_answers = [["127 Hours"]]
    environments = [FakeEnvironment(retrieval_fraction=0.0)]

    format_reward, outcome_reward, turn_reward = get_reward_funcs("turn_level")

    assert format_reward(completions=completions) == pytest.approx([0.1])
    assert outcome_reward(completions=completions, golden_answers=golden_answers) == pytest.approx(
        [1.5]
    )
    assert turn_reward(completions=completions, environments=environments) == pytest.approx([0.0])


def test_well_formed_wrong_answer():
    completions = [[_answer("Peter Schmeichel")]]
    golden_answers = [["127 Hours"]]
    environments = [FakeEnvironment(retrieval_fraction=1.0)]

    format_reward, outcome_reward, turn_reward = get_reward_funcs("turn_level")

    assert format_reward(completions=completions) == pytest.approx([0.1])
    assert outcome_reward(completions=completions, golden_answers=golden_answers) == pytest.approx(
        [0.0]
    )
    assert turn_reward(completions=completions, environments=environments) == pytest.approx([0.4])


def test_malformed_missing_answer_tag():
    completions = [[{"role": "assistant", "content": "I believe it is 127 Hours."}]]
    golden_answers = [["127 Hours"]]
    environments = [FakeEnvironment(retrieval_fraction=0.5)]

    format_reward, outcome_reward, turn_reward = get_reward_funcs("turn_level")

    assert format_reward(completions=completions) == pytest.approx([-0.1])
    assert outcome_reward(completions=completions, golden_answers=golden_answers) == pytest.approx(
        [0.0]
    )
    assert turn_reward(completions=completions, environments=environments) == pytest.approx([0.2])


def test_hard_tool_call_cap_mid_call_unresolved_tool_calls_no_answer():
    completions = [[_search_tool_call("127 hours")]]  # cap hit: no trailing tool/answer message
    golden_answers = [["127 Hours"]]
    environments = [FakeEnvironment(retrieval_fraction=0.0)]

    format_reward, outcome_reward, turn_reward = get_reward_funcs("turn_level")

    assert format_reward(completions=completions) == pytest.approx([-0.1])
    assert outcome_reward(completions=completions, golden_answers=golden_answers) == pytest.approx(
        [0.0]
    )
    assert turn_reward(completions=completions, environments=environments) == pytest.approx([0.0])


def test_get_reward_funcs_outcome_only_excludes_turn_reward():
    funcs = get_reward_funcs("outcome_only")
    assert [f.__name__ for f in funcs] == ["format_reward", "outcome_reward"]


def test_get_reward_funcs_turn_level_includes_turn_reward():
    funcs = get_reward_funcs("turn_level")
    assert [f.__name__ for f in funcs] == ["format_reward", "outcome_reward", "turn_reward"]


def test_get_reward_funcs_rejects_unknown_condition():
    with pytest.raises(ValueError):
        get_reward_funcs("bogus")  # type: ignore


def test_format_reward_logs_format_compliance_rate():
    log_metric = _FakeLogMetric()
    completions = [
        [_answer("127 Hours")],
        [{"role": "assistant", "content": "no tag here"}],
    ]

    format_reward(completions=completions, log_metric=log_metric)

    assert log_metric.calls == [
        ("format_compliance_rate", 1.0),
        ("format_compliance_rate", 0.0),
    ]


def test_outcome_reward_logs_exact_match_and_f1_per_completion():
    log_metric = _FakeLogMetric()
    completions = [[_answer("127 Hours")], [_answer("Peter Schmeichel")]]
    golden_answers = [["127 Hours"], ["127 Hours"]]

    outcome_reward(completions=completions, golden_answers=golden_answers, log_metric=log_metric)

    assert log_metric.calls == [
        ("exact_match", 1.0),
        ("f1", 1.0),
        ("exact_match", 0.0),
        ("f1", 0.0),
    ]


def test_turn_reward_logs_unscaled_retrieval_fraction():
    log_metric = _FakeLogMetric()
    environments = [
        FakeEnvironment(retrieval_fraction=1.0),
        FakeEnvironment(retrieval_fraction=0.5),
    ]

    turn_reward(environments=environments, log_metric=log_metric)

    assert log_metric.calls == [
        ("retrieval_fraction", 1.0),
        ("retrieval_fraction", 0.5),
    ]


def test_length_penalty_zero_below_target():
    completions = [[{"role": "assistant", "content": "x" * 100}]]

    assert length_penalty(completions=completions) == pytest.approx([0.0])


def test_length_penalty_scales_linearly_above_target():
    # target=2000, excess=1000 -> half of one target-width -> -0.2 * 0.5 = -0.1
    completions = [[{"role": "assistant", "content": "x" * 3000}]]

    assert length_penalty(completions=completions) == pytest.approx([-0.1])


def test_length_penalty_caps_at_max_magnitude():
    # excess=4000 -> 2x target-width, but capped at -0.2
    completions = [[{"role": "assistant", "content": "x" * 6000}]]

    assert length_penalty(completions=completions) == pytest.approx([-0.2])


def test_length_penalty_sums_across_multiple_assistant_turns():
    completions = [
        [
            _search_tool_call("query"),  # assistant, content="" -- contributes 0
            _tool_response("y" * 10000),  # tool response -- must NOT count toward the penalty
            {"role": "assistant", "content": "x" * 2500},  # final answer turn
        ]
    ]

    # total assistant content = 0 + 2500 = 2500, excess = 500 -> -0.2 * (500/2000) = -0.05
    assert length_penalty(completions=completions) == pytest.approx([-0.05])


def test_length_penalty_logs_completion_length():
    log_metric = _FakeLogMetric()
    completions = [[{"role": "assistant", "content": "x" * 500}]]

    length_penalty(completions=completions, log_metric=log_metric)

    assert log_metric.calls == [("completion_length", 500.0)]


def test_get_reward_funcs_penalize_length_appends_length_penalty_for_outcome_only():
    funcs = get_reward_funcs("outcome_only", penalize_length=True)
    assert [f.__name__ for f in funcs] == ["format_reward", "outcome_reward", "length_penalty"]


def test_get_reward_funcs_penalize_length_appends_length_penalty_for_turn_level():
    funcs = get_reward_funcs("turn_level", penalize_length=True)
    assert [f.__name__ for f in funcs] == [
        "format_reward",
        "outcome_reward",
        "turn_reward",
        "length_penalty",
    ]


def test_get_reward_funcs_penalize_length_defaults_to_false():
    funcs = get_reward_funcs("outcome_only")
    assert [f.__name__ for f in funcs] == ["format_reward", "outcome_reward"]


def test_search_count_penalty_zero_calls():
    completions = [[_answer("127 Hours")]]  # no tool calls at all

    assert search_count_penalty(completions=completions) == pytest.approx([0.0])


def test_search_count_penalty_scales_with_call_count():
    completions = [
        [
            _search_tool_call("query 1"),
            _tool_response("doc"),
            _search_tool_call("query 2"),
            _tool_response("doc"),
            _answer("127 Hours"),
        ]
    ]

    # 2 search calls * -0.1 (paper's lambda_s for MT-PPO's search-count penalty, Section 5.2/6.1 --
    # borrowed here since the paper's own GRPO case study has no equivalent term)
    assert search_count_penalty(completions=completions) == pytest.approx([-0.2])


def test_search_count_penalty_ignores_non_search_tool_calls():
    other_tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "not_search", "arguments": {}}}],
    }
    completions = [[other_tool_call, _answer("127 Hours")]]

    assert search_count_penalty(completions=completions) == pytest.approx([0.0])


def test_search_count_penalty_logs_search_call_count():
    log_metric = _FakeLogMetric()
    completions = [[_search_tool_call("q"), _tool_response("doc"), _answer("127 Hours")]]

    search_count_penalty(completions=completions, log_metric=log_metric)

    assert log_metric.calls == [("search_call_count", 1.0)]


def test_get_reward_funcs_penalize_search_count_appends_for_outcome_only():
    funcs = get_reward_funcs("outcome_only", penalize_search_count=True)
    assert [f.__name__ for f in funcs] == [
        "format_reward",
        "outcome_reward",
        "search_count_penalty",
    ]


def test_get_reward_funcs_penalize_search_count_appends_for_turn_level():
    funcs = get_reward_funcs("turn_level", penalize_search_count=True)
    assert [f.__name__ for f in funcs] == [
        "format_reward",
        "outcome_reward",
        "turn_reward",
        "search_count_penalty",
    ]


def test_get_reward_funcs_penalize_search_count_defaults_to_false():
    funcs = get_reward_funcs("outcome_only")
    assert "search_count_penalty" not in [f.__name__ for f in funcs]


def test_get_reward_funcs_both_penalties_composable():
    funcs = get_reward_funcs("outcome_only", penalize_length=True, penalize_search_count=True)
    assert [f.__name__ for f in funcs] == [
        "format_reward",
        "outcome_reward",
        "length_penalty",
        "search_count_penalty",
    ]


def test_turn_reward_scale_constant_matches_turn_reward_behavior():
    from turn_level_rewards.rewards import TURN_REWARD_SCALE

    assert TURN_REWARD_SCALE == 0.4


# --- Paper-faithful GRPO rewards (Section 6.1), for the five-arm comparison ---


def test_paper_grpo_outcome_reward_is_binary_correctness():
    """GRPO-OR is defined word-for-word identically to PPO-OR: "a binary signal indicating
    final-answer correctness". No format term, no partial credit.
    """
    from turn_level_rewards.rewards import paper_grpo_outcome_reward

    completions = [
        [{"role": "assistant", "content": "<answer>Paris</answer>"}],
        [{"role": "assistant", "content": "<answer>Berlin</answer>"}],
        [{"role": "tool", "name": "search", "content": "some retrieved text"}],
    ]

    rewards = paper_grpo_outcome_reward(completions, [["Paris"], ["Paris"], ["Paris"]])

    assert rewards == [1.0, 0.0, 0.0]


def test_paper_grpo_merged_reward_adds_retrieval_to_the_graded_outcome():
    """GRPO-MR combines retrieval correctness with the graded R^O, as ONE trajectory-level
    scalar -- GRPO has no per-timestep value function, so there is nowhere to place a per-turn
    reward. That is the structural reason GRPO-MR is the analogue of PPO-MR, not of MT-PPO.
    """
    from turn_level_rewards.rewards import paper_grpo_merged_reward

    class _Env:
        retrieval_fraction = 0.5

    completions = [
        [{"role": "assistant", "content": "<answer>Paris</answer>"}],
        [{"role": "tool", "name": "search", "content": "docs"}],
    ]

    rewards = paper_grpo_merged_reward(
        completions, [["Paris"], ["Paris"]], environments=[_Env(), _Env()]
    )

    assert rewards[0] == pytest.approx(1.0 + 0.3 * 0.5)
    assert rewards[1] == pytest.approx(-1.0 + 0.3 * 0.5)


def test_paper_grpo_merged_reward_tolerates_missing_environments():
    """environments is only supplied when environment_factory is set; a missing one must score
    retrieval as 0 rather than crash the run.
    """
    from turn_level_rewards.rewards import paper_grpo_merged_reward

    rewards = paper_grpo_merged_reward(
        [[{"role": "assistant", "content": "<answer>Paris</answer>"}]], [["Paris"]]
    )

    assert rewards == [1.0]


def test_get_paper_reward_funcs_selects_one_function_per_condition():
    from turn_level_rewards.rewards import get_paper_reward_funcs

    assert [f.__name__ for f in get_paper_reward_funcs("grpo_or")] == ["paper_grpo_outcome_reward"]
    assert [f.__name__ for f in get_paper_reward_funcs("grpo_mr")] == ["paper_grpo_merged_reward"]


def test_get_paper_reward_funcs_rejects_an_unknown_condition():
    from turn_level_rewards.rewards import get_paper_reward_funcs

    with pytest.raises(ValueError, match="unknown paper GRPO condition"):
        get_paper_reward_funcs("turn_level")  # ty: ignore[invalid-argument-type]


def test_paper_grpo_rewards_have_within_group_variance_when_answers_differ():
    """GRPO's gradient comes ENTIRELY from within-group reward variance. This pins the known
    risk of the binary reward: a group whose rollouts all miss scores identically, giving zero
    variance and therefore zero gradient. Watch frac_reward_zero_std during the real runs.
    """
    from turn_level_rewards.rewards import paper_grpo_outcome_reward

    all_wrong = paper_grpo_outcome_reward(
        [[{"role": "assistant", "content": "<answer>Berlin</answer>"}]] * 4, [["Paris"]] * 4
    )
    mixed = paper_grpo_outcome_reward(
        [
            [{"role": "assistant", "content": "<answer>Paris</answer>"}],
            [{"role": "assistant", "content": "<answer>Berlin</answer>"}],
        ],
        [["Paris"], ["Paris"]],
    )

    assert len(set(all_wrong)) == 1  # zero variance -> no gradient from this group
    assert len(set(mixed)) == 2


def test_paper_grpo_rewards_log_every_metric_the_comparison_needs():
    """Both GRPO arms must log the same metric set, or the five-arm table has holes.

    paper_grpo_merged_reward originally logged retrieval_fraction and format_compliance_rate but
    NOT exact_match -- so the grpo_mr arm would have finished a full 500-step run reporting no
    exact match at all, the headline metric of the entire comparison. Caught by the eval
    determinism gate, where grpo_or emitted eval_exact_match and grpo_mr would not have.
    """
    from turn_level_rewards.rewards import paper_grpo_merged_reward, paper_grpo_outcome_reward

    class _Env:
        retrieval_fraction = 0.5

    completions = [[{"role": "assistant", "content": "<answer>Knox County Airport</answer>"}]]
    golden = [["Knox County Regional Airport"]]

    for reward_fn, kwargs in (
        (paper_grpo_outcome_reward, {}),
        (paper_grpo_merged_reward, {"environments": [_Env()]}),
    ):
        logged: dict[str, list[float]] = {}

        def record(name: str, value: float, sink: dict = logged) -> None:
            sink.setdefault(name, []).append(value)

        reward_fn(completions, golden, log_metric=record, **kwargs)
        assert "exact_match" in logged, f"{reward_fn.__name__} must log exact_match"
        assert "f1" in logged, f"{reward_fn.__name__} must log f1"
        assert "format_compliance_rate" in logged, f"{reward_fn.__name__} must log format rate"
        # A near-miss answer: EM is 0 but F1 is high -- the exact case that makes reporting both
        # worthwhile, and the rigidity the paper cites as its reason for adding a judge.
        assert logged["exact_match"] == [0.0]
        assert logged["f1"][0] == pytest.approx(0.857, abs=0.01)


def test_paper_baseline_turn_reward_is_retrieval_only():
    """The paper's PPO-MR/GRPO-MR baselines take retrieval correctness as their ONLY intermediate
    reward. The per-turn format bonus and the lambda_s search penalty are MT-PPO's own Section 5.2
    contribution -- giving them to an MR baseline turns it into a flattened MT-PPO.

    Pinned as a test because this exact confusion made Phase 7c's `ppo_mr` a non-reproduction of
    the paper's PPO-MR, and the two arms now differ only by which of these functions they call.
    """
    from turn_level_rewards.rewards import PAPER_RETRIEVAL_BONUS, paper_baseline_turn_reward

    assert paper_baseline_turn_reward(found_gold=True) == PAPER_RETRIEVAL_BONUS
    assert paper_baseline_turn_reward(found_gold=False) == 0.0


def test_paper_baseline_turn_reward_ignores_format_and_search_count():
    """Same inputs that move paper_turn_reward must NOT move the baseline variant."""
    from turn_level_rewards.rewards import paper_baseline_turn_reward, paper_turn_reward

    # paper_turn_reward is sensitive to format and cumulative searches...
    assert paper_turn_reward(
        found_gold=True, format_ok=True, cumulative_searches=1
    ) != paper_turn_reward(found_gold=True, format_ok=False, cumulative_searches=3)

    # ...the baseline has no such terms to be sensitive to, so retrieval alone determines it.
    assert paper_baseline_turn_reward(found_gold=True) == paper_baseline_turn_reward(
        found_gold=True
    )
    # And it is strictly the retrieval term: never negative, unlike a search-penalised turn.
    assert paper_baseline_turn_reward(found_gold=False) == 0.0
    assert paper_turn_reward(found_gold=False, format_ok=False, cumulative_searches=4) < 0.0
