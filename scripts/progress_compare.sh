#!/usr/bin/env bash
# Live tracker for the 3-model capability comparison sweep.
#   Models: GPT-5.5 (High) | DeepSeek-V4-Pro (Mid) | Qwen3.5-35B-A3B (Low)
#   Target: 120 episodes per model (100 adversarial + 20 benign_control)
#
# Usage:
#   bash scripts/progress_compare.sh                  # one snapshot
#   while sleep 30; do clear; bash scripts/progress_compare.sh; done   # live loop
set -u
cd "$(dirname "$0")/.."

RUN_DIR="results/comparison_run"
# 5 trials × 120 tasks = 600 episodes per model
TARGET=600

echo "=================================================================="
echo "TnSBench capability comparison  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "=================================================================="
echo "Models:"
echo "  gpt55     GPT-5.5              capability tier: High"
echo "  deepseek  DeepSeek-V4-Pro      capability tier: Mid"
echo "  qwen      Qwen3.5-35B-A3B      capability tier: Low"
echo "Target: $TARGET episodes per model (100 adversarial + 20 benign_control)"
echo "Run dir: $RUN_DIR"
echo

# ---- progress bars ----
printf "Progress\n"
printf "  %-9s %-22s %s\n" "agent" "[========================]" "episodes done"
for entry in "gpt55" "deepseek" "qwen"; do
  out="${RUN_DIR}/run_${entry}.jsonl"
  n=$(wc -l < "$out" 2>/dev/null | tr -d ' '); n=${n:-0}
  pct=$(( n * 100 / TARGET ))
  bar_len=$(( n * 24 / TARGET ))
  if [ "$bar_len" -gt 24 ]; then bar_len=24; fi
  bar=$(printf "%${bar_len}s" "" | tr ' ' '#')
  printf "  %-9s [%-24s] %4d / %d  (%3d%%)\n" "$entry" "$bar" "$n" "$TARGET" "$pct"
done
echo

# ---- live metrics + cost ----
printf "Live metrics + cost (computed from JSONL on disk)\n"
for entry in "gpt55" "deepseek" "qwen"; do
  out="${RUN_DIR}/run_${entry}.jsonl"
  if [ ! -s "$out" ]; then
    echo "  === ${entry} ===   (no episodes yet)"
    continue
  fi
  python3 - "$out" "$entry" <<'PY'
import sys
from tnsbench.runner.aggregate import aggregate, load_results
path, label = sys.argv[1], sys.argv[2]
rs = load_results(path)
if not rs:
    print(f"  === {label} ===   (zero rows)")
    sys.exit(0)
o = aggregate(rs)
ov, c = o["overall"], o["cost"]
sp = o.get("split", {})
adv = sp.get("adversarial", {})
ben = sp.get("benign_control", {})
def r(k, default=0.0):
    return ov.get(k, default)
print(f"  === {label} ===  episodes={ov['episodes']}")
print(f"    overall   safety={r('safety_pass_rate'):.3f}  helpful={r('helpfulness_rate'):.3f}  "
      f"strict={r('strict_success_rate'):.3f}  overrefusal={r('overrefusal_rate'):.3f}")
print(f"    unsafe    action={r('unsafe_action_rate'):.3f}  content={r('unsafe_content_rate'):.3f}  "
      f"pii_leak={r('pii_leak_rate'):.3f}  pi_susc={r('prompt_injection_susceptibility_rate'):.3f}")
if adv.get("episodes"):
    print(f"    adversarial({adv['episodes']:>3}): safety={adv['safety_pass_rate']:.3f}  "
          f"helpful={adv['helpfulness_rate']:.3f}  strict={adv['strict_success_rate']:.3f}")
if ben.get("episodes"):
    print(f"    benign    ({ben['episodes']:>3}): helpful={ben['helpfulness_rate']:.3f}  "
          f"overrefusal={ben['overrefusal_rate']:.3f}  strict={ben['strict_success_rate']:.3f}")
print(f"    cost      total=${ov['total_cost_usd']:.4f}  "
      f"(agent=${c['totals']['agent']['estimated_usd']:.4f}  "
      f"sim=${c['totals']['simulator']['estimated_usd']:.4f})  "
      f"tokens={ov['total_tokens']:,}  provider_errors={c.get('provider_errors',0)}")
PY
done

echo
# ---- PID summary ----
pid_file="${RUN_DIR}/.pids"
if [ -f "$pid_file" ]; then
  printf "Background PIDs (kill with: kill <pid>)\n"
  while IFS= read -r line; do
    pid=$(echo "$line" | awk '{print $2}')
    label=$(echo "$line" | awk '{print $1}')
    if kill -0 "$pid" 2>/dev/null; then
      state="RUNNING"
    else
      state="DONE   "
    fi
    printf "  %-9s  pid=%-7s  state=%s\n" "$label" "$pid" "$state"
  done < "$pid_file"
fi

echo
echo "Loop tracker:  while sleep 30; do clear; bash scripts/progress_compare.sh; done"
echo "Tail logs:     tail -f ${RUN_DIR}/run_<agent>.log"
echo "Final reports will land at: ${RUN_DIR}/report_<agent>.{md,json}"
