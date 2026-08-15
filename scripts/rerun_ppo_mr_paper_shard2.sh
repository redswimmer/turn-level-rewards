#!/usr/bin/env bash
# Re-run ppo_mr_paper eval shard 2 only. Shards 0 and 1 completed; shard 2 OOM'd at ~800
# rows under three-way GPU contention. Run alone, then merge all three.
set -u
REPO=/home/asavala/Development/grpo/turn-level-rewards
cd "$REPO" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== START eval ppo_mr_paper shard2 (solo) $(date -Is) ==="
.venv/bin/python -m turn_level_rewards.evaluate_ppo \
  --condition ppo_mr_paper --checkpoint-dir outputs/ppo_mr_paper/checkpoint-500 \
  --eval-size 7404 --num-shards 3 --shard 2 \
  --output results/ppo_mr_paper-shard2.json \
  > logs/ppo_mr_paper-eval-shard2-rerun.log 2>&1
echo "=== DONE eval ppo_mr_paper shard2 exit=$? $(date -Is) ==="

.venv/bin/python scripts/merge_eval_shards.py results/ppo_mr_paper-shard*.json \
  --output results/ppo_mr_paper-eval.json > logs/ppo_mr_paper-merge.log 2>&1
echo "=== MERGE exit=$? $(date -Is) ==="
cat results/ppo_mr_paper-eval.json 2>/dev/null
