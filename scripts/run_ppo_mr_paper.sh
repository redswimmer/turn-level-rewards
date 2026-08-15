#!/usr/bin/env bash
# Phase 7c addendum: the paper-faithful PPO-MR arm.
#
# `ppo_mr` carries MT-PPO's full R^I (lambda_s search penalty + per-turn format), which makes it
# a flattened MT-PPO rather than the paper's PPO-MR -- a cleaner isolation of Eq. 9, but NOT a
# reproduction. `ppo_mr_paper` uses the paper's Section 6.1 baseline R^I: retrieval correctness
# only. Both are kept so the cost of that deviation is measurable rather than assumed.
set -u
REPO=/home/asavala/Development/grpo/turn-level-rewards
cd "$REPO" || exit 1
mkdir -p logs results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== START train ppo_mr_paper $(date -Is) ==="
.venv/bin/python -m turn_level_rewards.train_ppo \
  --condition ppo_mr_paper --train-size 90447 --max-steps 500 \
  --num-rollouts-per-step 4 --save-steps 50 --seed 42 \
  > logs/ppo_mr_paper-train.log 2>&1
echo "=== DONE train ppo_mr_paper exit=$? $(date -Is) ==="

echo "=== START eval ppo_mr_paper $(date -Is) ==="
for S in 0 1 2; do
  .venv/bin/python -m turn_level_rewards.evaluate_ppo \
    --condition ppo_mr_paper --checkpoint-dir outputs/ppo_mr_paper/checkpoint-500 \
    --eval-size 7404 --num-shards 3 --shard "$S" \
    --output "results/ppo_mr_paper-shard$S.json" \
    > "logs/ppo_mr_paper-eval-shard$S.log" 2>&1 &
done
wait
.venv/bin/python scripts/merge_eval_shards.py results/ppo_mr_paper-shard*.json \
  --output results/ppo_mr_paper-eval.json >> logs/ppo_mr_paper-eval.log 2>&1
echo "=== DONE eval ppo_mr_paper exit=$? $(date -Is) ==="
