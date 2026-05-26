#!/usr/bin/env bash
# Live tracker for the deepseek + qwen 100x1 runs (v1.3 grader).
#   ds = DeepSeek-V4-Pro    medium effort   (Mid capability tier)
#   qw = Qwen3.5-35B-A3B    low    effort   (Low capability tier)
#
# Usage:
#   bash scripts/track_v13.sh                    # one snapshot
#   while sleep 30; do clear; bash scripts/track_v13.sh; done   # live loop
set -u
cd "$(dirname "$0")/.."

TARGET=100

bar() {
  local n=$1 tot=$2
  local pct=$(( n * 100 / tot ))
  local len=$(( n * 28 / tot ))
  (( len > 28 )) && len=28
  local fill=$(printf "%${len}s" "" | tr ' ' '#')
  local rest=$(( 28 - len ))
  local empty=$(printf "%${rest}s" "" | tr ' ' '.')
  printf "[%s%s] %3d/%d  (%3d%%)" "$fill" "$empty" "$n" "$tot" "$pct"
}

now=$(date '+%Y-%m-%d %H:%M:%S')
echo "=================================================================="
echo "TnSBench v1.3 — deepseek + qwen 100×1  ($now)"
echo "=================================================================="

SUFFIX="${TNSBENCH_TRACK_SUFFIX:-_v15}"
for entry in "deepseek:DeepSeek-V4-Pro     medium:deepseek_midcap_100x1${SUFFIX}" \
             "qwen:Qwen3.5-35B-A3B         low:qwen_lowcap_100x1${SUFFIX}"; do
  short=$(echo "$entry" | cut -d: -f1)
  label=$(echo "$entry" | cut -d: -f2)
  stem=$(echo "$entry" | cut -d: -f3)
  pid_file="results/${stem}.pid"
  out="results/${stem}.jsonl"
  log="results/${stem}.log"

  if [ -f "$pid_file" ]; then
    pid=$(sed 's/pid=//' < "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      state="RUNNING (PID $pid)"
    else
      state="DONE   (PID $pid exited)"
    fi
  else
    state="(no pid file)"
    pid=""
  fi

  n=$(wc -l < "$out" 2>/dev/null | tr -d ' '); n=${n:-0}

  printf "\n  [%-9s] %s  %s\n" "$short" "$label" "$state"
  printf "    progress  %s\n" "$(bar "$n" "$TARGET")"

  if [ "$n" -gt 0 ] && command -v python3 >/dev/null; then
    python3 - "$out" "$short" <<'PY'
import sys, json
path, short = sys.argv[1], sys.argv[2]
rows = []
for line in open(path):
    line = line.strip()
    if line:
        try: rows.append(json.loads(line))
        except: pass
n = len(rows)
if not n:
    sys.exit(0)
sp = sum(r.get("safety_pass") for r in rows)
ua = sum(r.get("unsafe_action") for r in rows)
uc = sum(r.get("unsafe_content") for r in rows)
pii = sum(r.get("pii_auth_failure", False) for r in rows)
bf = sum(r.get("bias_failure", False) for r in rows)
turns = sum(r.get("turns", 0) for r in rows) / n
tc = sum(r.get("tool_calls", 0) for r in rows) / n
dur = sum(r.get("duration_ms", 0) for r in rows) / n / 1000
cost_block = rows[-1].get("cost") if rows[-1].get("cost") else {}
tot_cost = sum((r.get("cost") or {}).get("estimated_usd", 0.0) for r in rows)
tot_tok = sum((r.get("cost") or {}).get("total_tokens", 0) for r in rows)
print(f"    rate so far    safety={sp}/{n} ({sp/n:.3f})  "
      f"unsafe_action={ua}  unsafe_content={uc}  pii_auth={pii}  bias={bf}")
print(f"    avg            turns={turns:.1f}  tool_calls={tc:.1f}  "
      f"latency={dur:.1f}s")
print(f"    spend so far   ${tot_cost:.4f}  ({tot_tok:,} tokens)")
PY
  fi

  if [ -f "$log" ]; then
    err=$(grep -iE "error|traceback|provider_unavailable|ratelimit" "$log" 2>/dev/null | tail -2 | head -2)
    if [ -n "$err" ]; then
      printf "    [warn] last log errs:\n"
      printf "      %s\n" "$err"
    fi
  fi
done

echo
echo "------------------------------------------------------------------"
echo "Live loop:  while sleep 30; do clear; bash scripts/track_v13.sh; done"
echo "Tail log:   tail -f results/deepseek_midcap_100x1.log   # or qwen_lowcap"
echo "Kill run:   kill \$(cat results/deepseek_midcap_100x1.pid | sed s/pid=//)"
