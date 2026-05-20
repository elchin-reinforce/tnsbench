#!/usr/bin/env bash
# TnSBench-Hard v0.7 — three-model benchmark
#   GPT-5.5            (High capability) via OpenAI
#   DeepSeek-V4-Pro    (Mid)             via DeepInfra
#   Qwen3.5-35B-A3B    (Low)             via DeepInfra
#
# Simulator: Grok-4-fast (primary) with Qwen3.5 fallback when xAI errors
# Judge:     mock (state + transcript checks do the heavy lifting in v0.7)
#
# 100 adversarial tasks × 5 trials = 500 episodes per model = 1500 total.
# Each model runs as ONE process (with --trials 5 internally) so the JSONL
# row count is the live progress indicator.
#
# Run: bash scripts/run_v07.sh
# Track in another terminal: bash scripts/track_v07.sh

set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR="results/v07_run"
mkdir -p "$RUN_DIR"
: > "$RUN_DIR/.pids"

SIM_FLAGS=(
  --simulator cascading
  --sim-provider xai --sim-model grok-4-fast-non-reasoning
  --sim-fallback-provider deepinfra
  --sim-fallback-model "UNI-AI/qwen3.5-35b-a3b-abliterated-no-reasoning"
)

# Mock judge — v0.7 grading is state + transcript predicates; LLM judge would
# add cost without changing the leaderboard signal.
JUDGE_FLAGS=(--judge-provider mock --judge-model mock)

# short : provider : model : reasoning_effort (capability tier)
#   high   = strongest reasoning budget (GPT-5.5 at the top end)
#   medium = standard reasoning budget (DeepSeek-V4-Pro)
#   low    = minimal reasoning budget  (Qwen3.5-35B-A3B as the floor)
AGENTS=(
  "gpt55:openai:gpt-5.5:high"
  "deepseek:deepinfra:deepseek-ai/DeepSeek-V4-Pro:medium"
  "qwen:deepinfra:Qwen/Qwen3.5-35B-A3B:low"
)

echo "TnSBench-Hard v0.7 launch — 100 tasks × 5 trials × 3 models"
echo "RUN_DIR=$RUN_DIR"
echo "Capability tiers:"
echo "  gpt55     reasoning_effort=high     (High capability tier)"
echo "  deepseek  reasoning_effort=medium   (Mid capability tier)"
echo "  qwen      reasoning_effort=low      (Low capability tier)"
echo

for entry in "${AGENTS[@]}"; do
  IFS=':' read -r short prov model effort <<< "$entry"
  out="$RUN_DIR/run_${short}.jsonl"
  log="$RUN_DIR/run_${short}.log"
  : > "$out"
  : > "$log"

  # caffeinate -is keeps the laptop awake AND prevents idle sleep while the
  # child python process runs. Each child is a single `cli run` that drives
  # all 5 trials × 100 tasks sequentially. The JSONL grows by one line per
  # finished episode, which the tracker watches.
  caffeinate -is python3 -m tnsbench.cli run \
    --agent llm_agent --agent-provider "$prov" --agent-model "$model" \
    --agent-reasoning-effort "$effort" \
    "${SIM_FLAGS[@]}" "${JUDGE_FLAGS[@]}" \
    --tasks all --trials 5 --seed 42 \
    --out "$out" >> "$log" 2>&1 &
  pid=$!
  echo "$short $pid $effort" >> "$RUN_DIR/.pids"
  echo "  launched $short ($prov / $model, effort=$effort) pid=$pid"
  echo "    out: $out"
  echo "    log: $log"
done

echo
echo "All three jobs running in background."
echo "Live tracker:  bash scripts/track_v07.sh"
echo "Stop all:      kill \$(awk '{print \$2}' $RUN_DIR/.pids)"
