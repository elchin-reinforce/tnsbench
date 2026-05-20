#!/usr/bin/env bash
# TnSBench full benchmark: three agents × N=5 trials × Grok-cascade simulator.
#
# Prereqs (in .env):
#   OPENAI_API_KEY=sk-...
#   DEEPINFRA_API_KEY=...
#   XAI_API_KEY=...        # required for Grok primary; without it the
#                          # cascading simulator falls back per-task to
#                          # UNI-AI/qwen3.5-35b-a3b-abliterated-no-reasoning.
#
# Cost: ~$25-40 total across all three agents at N=5 (see README).
# Wall-clock: ~3-6h wall, parallelised below.

set -euo pipefail

mkdir -p results

SIM_FLAGS=(
  --simulator cascading
  --sim-provider xai --sim-model grok-4-fast-non-reasoning
  --sim-fallback-provider deepinfra
  --sim-fallback-model "UNI-AI/qwen3.5-35b-a3b-abliterated-no-reasoning"
)

# Optional: enable LLM judge by exporting JUDGE_FLAGS before invoking.
JUDGE_FLAGS=${JUDGE_FLAGS:-"--judge-provider mock --judge-model mock"}

SEEDS=(42 43 44 45 46)
AGENTS=(
  "gpt55:openai:gpt-5.5"
  "deepseek:deepinfra:deepseek-ai/DeepSeek-V4-Pro"
  "qwen:deepinfra:Qwen/Qwen3.5-35B-A3B"
)

PIDS=()
for entry in "${AGENTS[@]}"; do
  short="${entry%%:*}"; rest="${entry#*:}"
  prov="${rest%%:*}"; model="${rest#*:}"
  for seed in "${SEEDS[@]}"; do
    out="results/run_${short}_seed${seed}.jsonl"
    log="results/run_${short}_seed${seed}.log"
    python3 -m tnsbench.cli run \
      --agent llm_agent --agent-provider "$prov" --agent-model "$model" \
      "${SIM_FLAGS[@]}" $JUDGE_FLAGS \
      --tasks all --trials 1 --seed "$seed" \
      --out "$out" 2> "$log" &
    PIDS+=($!)
    echo "launched $short seed=$seed pid=$!"
  done
done

echo "Waiting on ${#PIDS[@]} background jobs..."
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo "All runs complete. Merging seeds and aggregating reports..."
for short in gpt55 deepseek qwen; do
  merged="results/run_${short}.jsonl"
  : > "$merged"
  for seed in "${SEEDS[@]}"; do
    cat "results/run_${short}_seed${seed}.jsonl" >> "$merged"
  done
  python3 -m tnsbench.cli report "$merged" --out "results/report_${short}.md"
done

echo
echo "Reports written:"
ls -1 results/report_*.md
