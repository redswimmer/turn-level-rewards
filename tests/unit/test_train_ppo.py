"""Fast, GPU-free tests for train_ppo.py's pure functions and config builder.

No real MTPPOTrainer, model, or GPU is constructed here -- the rollout loop and critic
construction require a real model/chat-template, which is exactly what the live smoke test
(not tests/unit/) validates instead, per CLAUDE.md's Guiding principles.
"""

import pytest
import torch
import torch.nn as nn
from turn_level_rewards.train_ppo import (
    CollapseMonitor,
    MTPPOTrainer,
    _final_retrieval_fraction,
    _parse_args,
    _PolicyAndCritic,
    _row_cycle_from_step,
    build_ppo_config,
    compute_gae,
    compute_ppo_loss,
    gather_action_logprobs,
    place_turn_rewards,
    resolve_auto_checkpoint,
    resolve_stop_token_ids,
    truncate_after_stop_token,
    validate_tool_call_arguments,
)


def test_compute_gae_matches_hand_computed_returns_minus_baseline_at_gamma_lambda_one():
    """At gamma=1, lambda=1 (this repo's fixed values), GAE reduces to
    (full-episode Monte-Carlo return from t) - V_t -- hand-computed here, not just re-deriving
    the recursive formula back at itself.

    rewards=[1.0, 0.0, 2.0], values=[0.5, 0.5, 0.5], bootstrap_value=0.0:
      return_2 = 2.0 + 0.0        = 2.0  -> A_2 = 2.0 - 0.5 = 1.5
      return_1 = 0.0 + return_2   = 2.0  -> A_1 = 2.0 - 0.5 = 1.5
      return_0 = 1.0 + return_1   = 3.0  -> A_0 = 3.0 - 0.5 = 2.5
    """
    advantages = compute_gae(rewards=[1.0, 0.0, 2.0], values=[0.5, 0.5, 0.5])

    assert advantages == [2.5, 1.5, 1.5]


def test_compute_gae_single_step_episode():
    advantages = compute_gae(rewards=[1.5], values=[0.2])

    assert advantages == [1.3]


def test_compute_gae_nonzero_bootstrap_value_feeds_into_final_step():
    advantages = compute_gae(rewards=[1.0], values=[0.5], bootstrap_value=2.0)

    # delta = r + gamma*bootstrap - V = 1.0 + 2.0 - 0.5 = 2.5
    assert advantages == [2.5]


def test_compute_gae_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        compute_gae(rewards=[1.0, 2.0], values=[0.5])


def test_place_turn_rewards_ppo_condition_never_places_turn_reward():
    """ppo: R^I is always 0 -- single lump-sum credit assignment even across a multi-turn
    episode. Only R^O (format_and_outcome_reward) lands, at the last token.
    """
    rewards = place_turn_rewards(
        num_tokens=10,
        turn_boundary_token_indices=[2, 5],
        retrieval_fraction_after_each_turn=[0.5, 1.0],
        format_and_outcome_reward=1.2,
        condition="ppo",
    )

    assert rewards == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2]


def test_place_turn_rewards_mt_ppo_places_marginal_retrieval_gain_at_each_turn_boundary():
    """mt_ppo: R^I at each intermediate turn boundary is the MARGINAL gain in retrieval_fraction
    that specific turn caused (0.5 at turn 1, then 1.0-0.5=0.5 at turn 2) -- not the raw
    cumulative value, which would double-count every later turn's contribution.
    """
    rewards = place_turn_rewards(
        num_tokens=10,
        turn_boundary_token_indices=[2, 5],
        retrieval_fraction_after_each_turn=[0.5, 1.0],
        format_and_outcome_reward=1.2,
        condition="mt_ppo",
        turn_reward_scale=0.4,
    )

    assert rewards[2] == 0.4 * 0.5
    assert rewards[5] == 0.4 * (1.0 - 0.5)
    assert rewards[-1] == 1.2
    assert rewards[0] == 0.0
    assert rewards[1] == 0.0
    assert rewards[3] == 0.0
    assert rewards[4] == 0.0


def test_place_turn_rewards_mt_ppo_with_zero_intermediate_turns_matches_ppo():
    """An episode that answers without ever calling search (no intermediate turns) should score
    identically in both conditions -- there's nothing for the turn_reward term to differentiate.
    """
    ppo_rewards = place_turn_rewards(
        num_tokens=4,
        turn_boundary_token_indices=[],
        retrieval_fraction_after_each_turn=[],
        format_and_outcome_reward=0.9,
        condition="ppo",
    )
    mt_ppo_rewards = place_turn_rewards(
        num_tokens=4,
        turn_boundary_token_indices=[],
        retrieval_fraction_after_each_turn=[],
        format_and_outcome_reward=0.9,
        condition="mt_ppo",
    )

    assert ppo_rewards == mt_ppo_rewards == [0.0, 0.0, 0.0, 0.9]


def test_place_turn_rewards_defaults_turn_reward_scale_to_the_shared_constant():
    from turn_level_rewards.rewards import TURN_REWARD_SCALE

    rewards = place_turn_rewards(
        num_tokens=3,
        turn_boundary_token_indices=[0],
        retrieval_fraction_after_each_turn=[1.0],
        format_and_outcome_reward=0.0,
        condition="mt_ppo",
    )

    assert rewards[0] == TURN_REWARD_SCALE


def test_place_turn_rewards_rejects_mismatched_boundary_and_fraction_lengths():
    with pytest.raises(ValueError, match="equal length"):
        place_turn_rewards(
            num_tokens=5,
            turn_boundary_token_indices=[1, 2],
            retrieval_fraction_after_each_turn=[0.5],
            format_and_outcome_reward=0.0,
            condition="mt_ppo",
        )


def test_compute_ppo_loss_zero_advantage_and_matched_values_gives_only_kl_term():
    """advantages=0 -> policy_loss term is 0 regardless of ratio; new_values==returns ->
    value_loss is 0; new_logprobs==old_logprobs -> ratio==1 and kl==0. Total loss should be
    exactly 0.0 in this fully-matched case.
    """
    result = compute_ppo_loss(
        new_logprobs=torch.tensor([0.1, 0.2, 0.3]),
        old_logprobs=torch.tensor([0.1, 0.2, 0.3]),
        advantages=torch.tensor([0.0, 0.0, 0.0]),
        returns=torch.tensor([1.0, 1.0, 1.0]),
        new_values=torch.tensor([1.0, 1.0, 1.0]),
        action_mask=torch.tensor([1.0, 1.0, 1.0]),
    )

    assert result["loss"].item() == 0.0
    assert result["policy_loss"].item() == 0.0
    assert result["value_loss"].item() == 0.0
    assert result["kl"].item() == 0.0


def test_compute_ppo_loss_clips_large_positive_ratio_on_positive_advantage():
    """A large ratio (new much more likely than old) on positive advantage should be clipped to
    (1+clip_eps), not allowed to blow up the policy objective unbounded.
    """
    result = compute_ppo_loss(
        new_logprobs=torch.tensor([10.0]),  # ratio = exp(10) >> 1 + clip_eps
        old_logprobs=torch.tensor([0.0]),
        advantages=torch.tensor([1.0]),
        returns=torch.tensor([0.0]),
        new_values=torch.tensor([0.0]),
        action_mask=torch.tensor([1.0]),
        clip_eps=0.2,
        kl_beta=0.0,
        value_loss_coef=0.0,
    )

    # unclipped would be -(exp(10) * 1.0); clipped surrogate must use min(unclipped, clipped) --
    # since advantage is positive, clipping caps the objective at 1.2 * 1.0, so policy_loss is
    # exactly -1.2, not some huge negative number.
    assert result["policy_loss"].item() == pytest.approx(-1.2, abs=1e-4)


def test_compute_ppo_loss_masks_out_non_action_positions():
    """A masked-out position (action_mask=0) with wildly wrong values must not affect the loss at
    all -- only masked-in (action_mask=1) positions should contribute.
    """
    masked_result = compute_ppo_loss(
        new_logprobs=torch.tensor([0.0, 999.0]),
        old_logprobs=torch.tensor([0.0, -999.0]),
        advantages=torch.tensor([0.0, 999.0]),
        returns=torch.tensor([1.0, -999.0]),
        new_values=torch.tensor([1.0, 999.0]),
        action_mask=torch.tensor([1.0, 0.0]),
    )
    unmasked_result = compute_ppo_loss(
        new_logprobs=torch.tensor([0.0]),
        old_logprobs=torch.tensor([0.0]),
        advantages=torch.tensor([0.0]),
        returns=torch.tensor([1.0]),
        new_values=torch.tensor([1.0]),
        action_mask=torch.tensor([1.0]),
    )

    assert masked_result["loss"].item() == pytest.approx(unmasked_result["loss"].item())


def test_compute_ppo_loss_value_loss_scales_with_squared_error():
    result = compute_ppo_loss(
        new_logprobs=torch.tensor([0.0]),
        old_logprobs=torch.tensor([0.0]),
        advantages=torch.tensor([0.0]),
        returns=torch.tensor([3.0]),
        new_values=torch.tensor([1.0]),
        action_mask=torch.tensor([1.0]),
        kl_beta=0.0,
    )

    assert result["value_loss"].item() == pytest.approx(4.0)  # (1.0 - 3.0)**2
    assert result["loss"].item() == pytest.approx(0.5 * 4.0)  # value_loss_coef defaults to 0.5


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


def test_build_ppo_config_fixed_hyperparameters_identical_across_conditions():
    """These come from the paper (Section 6.2/C.1.3) or the design spec's stated assumptions --
    every one must hold for BOTH conditions, since ppo/mt_ppo differ only in reward placement
    (Eq. 9), not in any of these hyperparameters.
    """
    ppo_config = build_ppo_config("ppo", seed=42, max_steps=2, num_rollouts_per_step=2)
    mt_ppo_config = build_ppo_config("mt_ppo", seed=42, max_steps=2, num_rollouts_per_step=2)

    for config in (ppo_config, mt_ppo_config):
        assert config.n_max == 4
        assert config.clip_eps == 0.2
        assert config.kl_beta == 0.001
        assert config.policy_lr == 1e-6
        assert config.critic_lr == 1e-5
        assert config.gamma == 1.0
        assert config.gae_lambda == 1.0
        assert config.num_ppo_epochs == 4
        assert config.value_loss_coef == 0.5
        assert config.max_completion_length == 2048


def test_build_ppo_config_condition_and_derived_fields_differ():
    ppo_config = build_ppo_config("ppo", seed=42, max_steps=2, num_rollouts_per_step=2)
    mt_ppo_config = build_ppo_config("mt_ppo", seed=42, max_steps=2, num_rollouts_per_step=2)

    assert ppo_config.condition == "ppo"
    assert mt_ppo_config.condition == "mt_ppo"
    assert ppo_config.output_dir == "outputs/ppo"
    assert mt_ppo_config.output_dir == "outputs/mt_ppo"
    assert ppo_config.run_name == "ppo"
    assert mt_ppo_config.run_name == "mt_ppo"


def test_build_ppo_config_passes_through_seed_max_steps_and_rollout_count():
    config = build_ppo_config("ppo", seed=7, max_steps=500, num_rollouts_per_step=8)

    assert config.seed == 7
    assert config.max_steps == 500
    assert config.num_rollouts_per_step == 8


class _FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


class _FakeCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)


def test_create_optimizer_uses_two_param_groups_with_paper_learning_rates(tmp_path):
    config = build_ppo_config("ppo", seed=42, max_steps=2, num_rollouts_per_step=2)
    config.output_dir = str(tmp_path)
    model = _PolicyAndCritic(_FakePolicy(), _FakeCritic())
    trainer = MTPPOTrainer.__new__(MTPPOTrainer)  # bypass __init__ (needs a real tokenizer)
    trainer.model = model
    trainer.args = config

    optimizer = trainer.create_optimizer()

    assert len(optimizer.param_groups) == 2
    policy_group, critic_group = optimizer.param_groups
    assert policy_group["lr"] == 1e-6
    assert critic_group["lr"] == 1e-5
    assert list(policy_group["params"]) == list(model.policy.parameters())
    assert list(critic_group["params"]) == list(model.critic.parameters())


def test_parse_args_defaults():
    args = _parse_args(["--condition", "ppo"])

    assert args.condition == "ppo"
    assert args.seed == 42
    assert args.train_size == 8
    assert args.max_steps == 2
    assert args.num_rollouts_per_step == 2


def test_parse_args_condition_required():
    with pytest.raises(SystemExit):
        _parse_args([])


def test_parse_args_condition_choices_enforced():
    with pytest.raises(SystemExit):
        _parse_args(["--condition", "not_a_real_condition"])


def test_parse_args_overrides():
    args = _parse_args(
        [
            "--condition",
            "mt_ppo",
            "--seed",
            "7",
            "--train-size",
            "90447",
            "--max-steps",
            "500",
            "--num-rollouts-per-step",
            "8",
        ]
    )

    assert args.condition == "mt_ppo"
    assert args.seed == 7
    assert args.train_size == 90447
    assert args.max_steps == 500
    assert args.num_rollouts_per_step == 8


def test_final_retrieval_fraction_returns_zero_for_no_turns():
    assert _final_retrieval_fraction({"retrieval_fraction_after_each_turn": []}) == 0.0


def test_final_retrieval_fraction_returns_last_value():
    rollout = {"retrieval_fraction_after_each_turn": [0.5, 1.0]}
    assert _final_retrieval_fraction(rollout) == 1.0


def test_collapse_monitor_stops_on_non_finite_loss():
    monitor = CollapseMonitor()

    alerts = monitor.check(step=5, loss=float("nan"), mean_reward=1.0, mean_format_reward=0.1)

    assert len(alerts) == 1
    assert alerts[0].should_stop is True
    assert alerts[0].level == "ERROR"


def test_collapse_monitor_no_alerts_for_healthy_run():
    """A healthy run's mean reward MOVES between steps.

    This originally passed a fixed mean_reward=1.0 for all 30 steps. That encoded an assumption
    the 2026-07-24 flat-reward incident disproved: a bit-identical mean reward step after step is
    itself the failure signature (see test_collapse_monitor_fires_constant_reward_alert), not a
    picture of health, so the healthy case has to vary to still mean anything.
    """
    monitor = CollapseMonitor()

    for step in range(30):
        alerts = monitor.check(
            step=step, loss=0.5, mean_reward=1.0 + 0.01 * step, mean_format_reward=0.1
        )
        assert alerts == []


def test_collapse_monitor_fires_dead_reward_alert_after_threshold():
    """The alert fires exactly once, part-way through this loop (once step reaches the
    threshold), then stays suppressed for the remaining iterations (see
    test_collapse_monitor_each_alert_fires_only_once) -- so alerts must be accumulated across
    the whole loop, not read off only the final iteration's return value, which would always be
    empty by the fire-once design.
    """
    monitor = CollapseMonitor()

    fired: list = []
    for step in range(25):
        fired.extend(monitor.check(step=step, loss=0.5, mean_reward=0.0, mean_format_reward=0.1))

    assert len(fired) == 1
    assert fired[0].should_stop is False
    assert "Dead reward" in fired[0].title


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

    # Accumulated across the loop, not read off only the final iteration -- see the analogous
    # note on test_collapse_monitor_fires_dead_reward_alert_after_threshold above; the alert
    # fires part-way through this range and is then suppressed for the rest of it.
    fired: list = []
    for step in range(10, 35):
        fired.extend(monitor.check(step=step, loss=0.5, mean_reward=-0.1, mean_format_reward=-0.1))

    assert len(fired) == 1
    assert "Format compliance" in fired[0].title
    assert fired[0].should_stop is False


def test_collapse_monitor_each_alert_fires_only_once():
    monitor = CollapseMonitor()

    for step in range(25):
        monitor.check(step=step, loss=0.5, mean_reward=0.0, mean_format_reward=0.1)

    more_alerts = monitor.check(step=25, loss=0.5, mean_reward=0.0, mean_format_reward=0.1)
    assert more_alerts == []


def test_collapse_monitor_fires_constant_reward_alert():
    """Reward pinned at one nonzero value for many steps is a dead signal, not a healthy one.

    This is the exact 2026-07-24 failure the old monitor could not see: mean_reward was -0.1 on
    every one of 158 steps. The pre-existing "Dead reward" check only tested `mean_reward != 0.0`,
    so a constant -0.1 marked the reward "alive" at step 0 and permanently suppressed the alert.
    """
    monitor = CollapseMonitor()

    fired: list = []
    for step in range(25):
        fired.extend(monitor.check(step=step, loss=0.5, mean_reward=-0.1, mean_format_reward=0.1))

    assert len(fired) == 1
    assert "Constant reward" in fired[0].title
    assert fired[0].should_stop is False


def test_collapse_monitor_fires_format_never_compliant_alert():
    """format_reward non-positive from the very first step must alert too.

    The pre-existing collapse check was gated on `elif self._format_ever_compliant`, so a run that
    was NEVER compliant -- which is what a broken rollout loop produces -- could never trip it.
    A never-compliant run is a more urgent signal than a regression, not a lesser one.
    """
    monitor = CollapseMonitor()

    fired: list = []
    for step in range(25):
        fired.extend(monitor.check(step=step, loss=0.5, mean_reward=-0.1, mean_format_reward=-0.1))

    titles = [alert.title for alert in fired]
    assert "Format never compliant" in titles
    assert all(alert.should_stop is False for alert in fired)


def test_collapse_monitor_no_format_alert_once_compliance_is_seen():
    monitor = CollapseMonitor()

    fired: list = []
    for step in range(25):
        fired.extend(
            monitor.check(
                step=step, loss=0.5, mean_reward=0.5 + 0.01 * step, mean_format_reward=0.1
            )
        )

    assert fired == []


def test_resolve_stop_token_ids_puts_the_chat_turn_terminator_first():
    """The tokenizer's eos (the chat turn terminator) must be a stop id, even when the model's own
    generation_config names a different one.

    This is the root cause of the 2026-07-24 flat-reward incident, pinned as a test: for
    Qwen/Qwen3.5-0.8B, tokenizer.eos_token_id is <|im_end|> (248046) but
    model.generation_config.eos_token_id is <|endoftext|> (248044). Generating with only the
    latter runs every turn past its own terminator.
    """
    assert resolve_stop_token_ids(248046, 248044) == [248046, 248044]


def test_resolve_stop_token_ids_deduplicates_and_accepts_a_list_or_none():
    assert resolve_stop_token_ids(248046, 248046) == [248046]
    assert resolve_stop_token_ids(248046, [248044, 248046]) == [248046, 248044]
    assert resolve_stop_token_ids(248046, None) == [248046]


def test_truncate_after_stop_token_keeps_the_terminator_and_drops_everything_after():
    """A turn ends at its terminator; anything decoded past it is not part of this turn.

    Real observed tail from the incident: after emitting <|im_end|>, the policy kept decoding
    `\\n<|im_start|>user\\n<tool_response>{"results": ...}` -- hallucinating the environment's next
    turn. Those tokens were being marked action_mask=1 and trained on.
    """
    assert truncate_after_stop_token([1, 2, 99, 3, 4], stop_token_ids=[99]) == [1, 2, 99]


def test_truncate_after_stop_token_is_a_noop_when_no_stop_token_is_present():
    """A turn that hit max_new_tokens without terminating is returned whole, not silently emptied."""
    assert truncate_after_stop_token([1, 2, 3], stop_token_ids=[99]) == [1, 2, 3]


def test_truncate_after_stop_token_cuts_at_the_first_of_several_stop_ids():
    assert truncate_after_stop_token([1, 88, 2, 99], stop_token_ids=[99, 88]) == [1, 88]


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


def test_resolve_auto_checkpoint_returns_the_latest_checkpoint(tmp_path):
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-300").mkdir()
    (tmp_path / "checkpoint-200").mkdir()

    assert resolve_auto_checkpoint(str(tmp_path)).endswith("checkpoint-300")


def test_resolve_auto_checkpoint_raises_a_clear_error_on_a_missing_output_dir(tmp_path):
    """A first launch with `auto` used to die on a FileNotFoundError from inside transformers'
    os.listdir, not on this module's own message -- the output dir does not exist yet, which is
    the same "nothing to resume" case as an empty one.
    """
    with pytest.raises(ValueError, match="no checkpoint found"):
        resolve_auto_checkpoint(str(tmp_path / "does-not-exist"))


def test_resolve_auto_checkpoint_raises_on_an_existing_but_empty_output_dir(tmp_path):
    with pytest.raises(ValueError, match="no checkpoint found"):
        resolve_auto_checkpoint(str(tmp_path))


def _search(query: str, topk: int = 3) -> str:
    """Stand-in with the same annotation shape as SearchEnv.search."""
    return ""


def test_validate_tool_call_arguments_accepts_a_well_formed_call():
    assert validate_tool_call_arguments(_search, {"query": "who is X"}) is None
    assert validate_tool_call_arguments(_search, {"query": "x", "topk": 5}) is None


def test_validate_tool_call_arguments_rejects_a_non_string_query():
    """The 2026-07-24 crash: the policy emitted a numeric/null query, which bind() accepts
    (Python does not enforce annotations) and the retrieval server then rejected with HTTP 422,
    taking the whole run down at step 166.

    A wrongly-TYPED argument is the same class of model mistake as a wrongly-NAMED one, so it
    has to be caught here and fed back to the policy, not raised out of search().
    """
    assert "query" in (validate_tool_call_arguments(_search, {"query": 123}) or "")
    assert "query" in (validate_tool_call_arguments(_search, {"query": None}) or "")


def test_validate_tool_call_arguments_rejects_unknown_and_missing_arguments():
    assert validate_tool_call_arguments(_search, {"return": "x"}) is not None
    assert validate_tool_call_arguments(_search, {}) is not None


def test_validate_tool_call_arguments_allows_a_bool_where_int_is_annotated():
    """bool is a subclass of int, so isinstance already accepts it -- pinned so a future
    tightening does not start rejecting calls the retrieval server would have served fine.
    """
    assert validate_tool_call_arguments(_search, {"query": "x", "topk": True}) is None


def test_validate_tool_call_arguments_ignores_parameters_without_a_simple_annotation():
    def tool(payload, flag: str = "a"):
        return ""

    assert validate_tool_call_arguments(tool, {"payload": {"anything": 1}}) is None


def _logprob_fixture(num_positions: int, vocab: int = 64, hidden: int = 8):
    """Small float32 CPU stand-in for the policy's lm_head + selected hidden states."""
    torch.manual_seed(0)
    lm_head = nn.Linear(hidden, vocab, bias=False)
    states = torch.randn(num_positions, hidden, requires_grad=True)
    targets = torch.randint(0, vocab, (num_positions,))
    return lm_head, states, targets


def _unchunked_reference(lm_head, states, targets):
    """What _forward_policy_logprobs computed before chunking: one [n, vocab] tensor."""
    log_probs = torch.log_softmax(lm_head(states), dim=-1)
    return log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize("chunk_size", [1, 3, 4, 16, 1000])
def test_gather_action_logprobs_matches_the_unchunked_computation(chunk_size):
    """Chunking is a memory optimisation, so it must not change the numbers meaningfully.

    Each position's log_softmax is over its own vocab row and is mathematically independent of
    every other position. The agreement is not bit-exact, though, and the first version of this
    test wrongly asserted torch.equal: a [chunk, hidden] x [hidden, vocab] matmul can select a
    different GEMM kernel than the full-height one, so lm_head's own output moves by about one
    ULP (measured: 4.77e-07 in float32, and exactly 0 when chunk_size covers every position).
    Tolerance is set just above that, tight enough that any real algorithmic change fails.
    """
    lm_head, states, targets = _logprob_fixture(10)
    expected = _unchunked_reference(lm_head, states, targets)

    actual = gather_action_logprobs(lm_head, states, targets, chunk_size=chunk_size)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)


def test_gather_action_logprobs_produces_identical_gradients():
    """The value being right is not enough -- this sits inside a backward pass, so the gradient
    it feeds to the policy has to match too.
    """
    lm_head, states, targets = _logprob_fixture(10)
    _unchunked_reference(lm_head, states, targets).sum().backward()
    assert states.grad is not None
    expected_grad = states.grad.clone()

    states.grad = None
    gather_action_logprobs(lm_head, states, targets, chunk_size=3).sum().backward()
    actual_grad = states.grad

    assert actual_grad is not None
    assert torch.allclose(actual_grad, expected_grad, atol=1e-6)


def test_gather_action_logprobs_handles_a_single_action_token():
    lm_head, states, targets = _logprob_fixture(1)

    actual = gather_action_logprobs(lm_head, states, targets, chunk_size=256)

    assert actual.shape == (1,)
    assert torch.allclose(actual, _unchunked_reference(lm_head, states, targets), atol=1e-6)


def test_gather_action_logprobs_rejects_a_non_positive_chunk_size():
    """A chunk_size of 0 would loop forever building empty chunks -- fail loudly instead."""
    lm_head, states, targets = _logprob_fixture(4)

    with pytest.raises(ValueError, match="chunk_size"):
        gather_action_logprobs(lm_head, states, targets, chunk_size=0)
