#!/usr/bin/env bash
# Usage: bash scripts/progress.sh
# Prints progress + live metrics for the in-flight three-agent sweep.

cd "$(dirname "$0")/.."

echo "=========================================="
echo "TnSBench live progress  ($(date '+%H:%M:%S'))"
echo "=========================================="

echo
echo "Episodes completed per agent (target: 500 each):"
for a in gpt55 deepseek qwen; do
  n=$(cat results/run_${a}_seed*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  bar_n=$(( n / 25 ))                     # 1 block per 25 episodes (20 blocks at 500)
  bar=$(printf "%${bar_n}s" "" | tr ' ' '#')
  printf "  %-9s  %4d / 500   [%-20s]\n" "$a" "$n" "$bar"
done

echo
echo "Per-agent live metrics + cost:"
for a in gpt55 deepseek qwen; do
  tmpfile=/tmp/_tnsb_${a}.jsonl
  cat results/run_${a}_seed*.jsonl 2>/dev/null > "$tmpfile"
  if [ -s "$tmpfile" ]; then
    python3 - "$tmpfile" "$a" <<'PY'
import sys
from tnsbench.runner.aggregate import aggregate, load_results
path, label = sys.argv[1], sys.argv[2]
o = aggregate(load_results(path))
ov, c = o["overall"], o["cost"]
print(f"  === {label} ===")
print(f"    episodes:    {ov['episodes']}")
print(f"    safety:      {ov['safety_pass_rate']}")
print(f"    helpful:     {ov['helpfulness_rate']}")
print(f"    strict:      {ov['strict_success_rate']}")
print(f"    overrefusal: {ov['overrefusal_rate']}")
print(f"    unsafe_act:  {ov['unsafe_action_rate']}    "
      f"pii_leak: {ov['pii_leak_rate']}    "
      f"pi_susc: {ov['prompt_injection_susceptibility_rate']}")
print(f"    cost:        ${ov['total_cost_usd']:.4f}   "
      f"(agent ${c['totals']['agent']['estimated_usd']:.4f}, "
      f"sim ${c['totals']['simulator']['estimated_usd']:.4f}, "
      f"judge ${c['totals']['judge']['estimated_usd']:.4f})")
print(f"    tokens:      {ov['total_tokens']:,}   provider_errors: {c['provider_errors']}")
PY
  else
    echo "  === $a ===   (no episodes yet)"
  fi
done

echo
echo "Watch loop:   while sleep 30; do clear; bash scripts/progress.sh; done"
echo "Stop loop:    Ctrl+C (does not affect the running benchmark)"
