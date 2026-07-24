"""Fast, GPU-free tests for train_ppo.py's pure functions and config builder.

No real MTPPOTrainer, model, or GPU is constructed here -- the rollout loop and critic
construction require a real model/chat-template, which is exactly what the live smoke test
(not tests/unit/) validates instead, per CLAUDE.md's Guiding principles.
"""

import pytest
import torch
import torch.nn as nn
from turn_level_rewards.train_ppo import (
    MTPPOTrainer,
    _parse_args,
    _PolicyAndCritic,
    build_ppo_config,
    compute_gae,
    compute_ppo_loss,
    place_turn_rewards,
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
