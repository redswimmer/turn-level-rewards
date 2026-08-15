#!/usr/bin/env bash
# Phase 7c evaluation driver: six evaluations across five arms, 7,404 held-out rows each.
#
# `ppo` contributes TWO evaluations rather than one -- the last checkpoint before
# its collapse and one after -- which is the paper's own stated methodology for
# its crashed PPO baselines ("we evaluate them using either the final checkpoint
# or the last checkpoint prior to collapse").
#
# GRPO arms run first: they go through GRPOTrainer.evaluate(), which batches, so
# they are cheap and bank two of the five arms before the slow per-episode PPO
# evals start.
set -u

REPO=/home/asavala/Development/grpo/turn-level-rewards
cd "$REPO" || exit 1
mkdir -p logs results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVAL_SIZE=7404

# --- GRPO arms: batched, no sharding needed ---------------------------------
# --eval-batch-size 4 is required, not cosmetic: the default of 2 still produces
# correct numbers but takes ~24 h for the two arms instead of ~6.5 h.
for C in grpo_or grpo_mr; do
  echo "=== START eval $C $(date -Is) ==="
  .venv/bin/python -m turn_level_rewards.evaluate \
    --condition "$C" --checkpoint "outputs/$C/checkpoint-500" \
    --eval-size $EVAL_SIZE --eval-batch-size 4 \
    --output "results/$C-eval.json" > "logs/p7c-eval-$C.log" 2>&1
  echo "=== DONE eval $C exit=$? $(date -Is) ==="
done

# --- PPO arms: one episode at a time, so shard 3 ways (~1.6x measured) -------
# Fields: <label> <condition> <checkpoint-dir>
run_ppo_eval() {
  local label=$1 cond=$2 ckpt=$3
  echo "=== START eval $label $(date -Is) ==="
  for S in 0 1 2; do
    .venv/bin/python -m turn_level_rewards.evaluate_ppo \
      --condition "$cond" --checkpoint-dir "$ckpt" --eval-size $EVAL_SIZE \
      --num-shards 3 --shard "$S" --output "results/$label-shard$S.json" \
      > "logs/p7c-eval-$label-shard$S.log" 2>&1 &
  done
  wait
  .venv/bin/python scripts/merge_eval_shards.py results/$label-shard*.json \
    --output "results/$label-eval.json" >> "logs/p7c-eval-$label.log" 2>&1
  echo "=== DONE eval $label exit=$? $(date -Is) ==="
}

run_ppo_eval mt_ppo          mt_ppo outputs/mt_ppo/checkpoint-500
run_ppo_eval ppo_mr          ppo_mr outputs/ppo_mr/checkpoint-500
run_ppo_eval ppo-precollapse ppo    outputs/_precollapse/ppo/checkpoint-50
run_ppo_eval ppo-final       ppo    outputs/ppo/checkpoint-100

echo "=== ALL EVALS FINISHED $(date -Is) ==="
