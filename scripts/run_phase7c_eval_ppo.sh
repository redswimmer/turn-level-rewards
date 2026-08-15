#!/usr/bin/env bash
# Phase 7c: re-run the four PPO evaluations now that evaluate_ppo.py records
# format_compliance_rate.
#
# Why re-run rather than patch the numbers: format compliance is derived from the
# completion text, and completions are not persisted -- only aggregate metrics are.
# So there is no post-hoc path to the metric.
#
# The GRPO arms are NOT re-run: TRL's GRPOTrainer already logged
# eval_format_compliance_rate for them (grpo_or 0.421, grpo_mr 0.975), and both
# tracks now derive it from the same rewards.py _extract_answer.
#
# Ordering: the two healthy arms first. The collapsed `ppo` checkpoints generate
# enormous completions (~23 s/row measured, vs ~2 s/row healthy) so they are the
# expensive ones, and they carry the least information -- run them last so a kill
# costs the least.
set -u

REPO=/home/asavala/Development/grpo/turn-level-rewards
cd "$REPO" || exit 1
mkdir -p logs results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVAL_SIZE=7404

run_ppo_eval() {
  local label=$1 cond=$2 ckpt=$3
  echo "=== START eval $label $(date -Is) ==="
  for S in 0 1 2; do
    .venv/bin/python -m turn_level_rewards.evaluate_ppo \
      --condition "$cond" --checkpoint-dir "$ckpt" --eval-size $EVAL_SIZE \
      --num-shards 3 --shard "$S" --output "results/$label-shard$S.json" \
      > "logs/p7c-eval2-$label-shard$S.log" 2>&1 &
  done
  wait
  .venv/bin/python scripts/merge_eval_shards.py results/$label-shard*.json \
    --output "results/$label-eval.json" >> "logs/p7c-eval2-$label.log" 2>&1
  echo "=== DONE eval $label exit=$? $(date -Is) ==="
}

run_ppo_eval mt_ppo          mt_ppo outputs/mt_ppo/checkpoint-500
run_ppo_eval ppo_mr          ppo_mr outputs/ppo_mr/checkpoint-500
run_ppo_eval ppo-precollapse ppo    outputs/_precollapse/ppo/checkpoint-50
run_ppo_eval ppo-final       ppo    outputs/ppo/checkpoint-100

echo "=== ALL PPO EVALS FINISHED $(date -Is) ==="
