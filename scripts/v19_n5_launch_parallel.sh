#!/usr/bin/env bash
# Launch 5 parallel --trials 1 --seed {42..46} jobs per model. Each job
# writes its own jsonl. Output filenames mirror what the serial run
# would have written, but with `_seed{N}` appended. The merge step
# below stitches them into the canonical _100x5_v19.jsonl files,
# remapping trial_index by seed offset (42→0, 43→1, ..., 46→4).
#
# Usage: bash scripts/v19_n5_launch_parallel.sh
#   Prints "DS pids: ..." and "QW pids: ..." and writes them to
#   results/v19_n5_pids.txt
#
# After completion, results/v19_n5_DONE will appear.

set -u
cd "$(dirname "$0")/.."
mkdir -p results

SEEDS=(42 43 44 45 46)

DS_MODEL="deepseek-ai/DeepSeek-V4-Pro"
DS_EFFORT="medium"
QW_MODEL="Qwen/Qwen3.5-35B-A3B"
QW_EFFORT="low"

DS_PIDS=()
QW_PIDS=()

for seed in "${SEEDS[@]}"; do
  out="results/deepseek_midcap_100x5_v19_seed${seed}.jsonl"
  log="results/deepseek_midcap_100x5_v19_seed${seed}.log"
  python3 -m tnsbench.cli run \
    --agent llm_agent --agent-provider deepinfra --agent-model "$DS_MODEL" \
    --agent-reasoning-effort "$DS_EFFORT" \
    --simulator scripted \
    --judge-provider mock --judge-model mock \
    --tasks all --trials 1 --seed "$seed" \
    --out "$out" > "$log" 2>&1 &
  DS_PIDS+=($!)
done

for seed in "${SEEDS[@]}"; do
  out="results/qwen_lowcap_100x5_v19_seed${seed}.jsonl"
  log="results/qwen_lowcap_100x5_v19_seed${seed}.log"
  python3 -m tnsbench.cli run \
    --agent llm_agent --agent-provider deepinfra --agent-model "$QW_MODEL" \
    --agent-reasoning-effort "$QW_EFFORT" \
    --simulator scripted \
    --judge-provider mock --judge-model mock \
    --tasks all --trials 1 --seed "$seed" \
    --out "$out" > "$log" 2>&1 &
  QW_PIDS+=($!)
done

{
  echo "ds_pids=${DS_PIDS[@]}"
  echo "qw_pids=${QW_PIDS[@]}"
} > results/v19_n5_pids.txt
echo "DS pids: ${DS_PIDS[@]}"
echo "QW pids: ${QW_PIDS[@]}"
