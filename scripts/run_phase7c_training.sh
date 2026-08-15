#!/usr/bin/env bash
# Phase 7c training driver: four arms, sequential, one GPU.
# `ppo` (PPO-OR) is deliberately absent -- it is already complete (see
# docs/phase-7c-paper-faithful-ppo.md section 12); re-running it would only
# produce a longer collapse.
set -u

REPO=/home/asavala/Development/grpo/turn-level-rewards
cd "$REPO" || exit 1
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for C in ppo_mr mt_ppo; do
  echo "=== START $C $(date -Is) ==="
  .venv/bin/python -m turn_level_rewards.train_ppo \
    --condition "$C" --train-size 90447 --max-steps 500 \
    --num-rollouts-per-step 4 --save-steps 50 --seed 42 \
    > "logs/p7c-$C.log" 2>&1
  echo "=== DONE $C exit=$? $(date -Is) ==="
done

for C in grpo_or grpo_mr; do
  echo "=== START $C $(date -Is) ==="
  .venv/bin/python -m turn_level_rewards.train \
    --condition "$C" --train-size 90447 --eval-size 8 --max-steps 500 \
    --num-generations 8 --seed 42 \
    > "logs/p7c-$C.log" 2>&1
  echo "=== DONE $C exit=$? $(date -Is) ==="
done

echo "=== ALL ARMS FINISHED $(date -Is) ==="
