#!/usr/bin/env bash
# Live tracker for TnSBench-Hard v0.7 three-model run.
#
#   bash scripts/track_v07.sh
#
# Refreshes every 4s. Shows per-model: progress, live strict-success rate,
# recent failures, cost so far, ETA. Ctrl+C to stop the tracker (does NOT
# affect the running benchmark).

cd "$(dirname "$0")/.."
RUN_DIR="results/v07_run"
TARGET=500    # 100 tasks × 5 trials

# Pretty colors — disable by exporting NOCOLOR=1
if [ -z "${NOCOLOR:-}" ]; then
  C_RUN=$'\033[1;32m'   # green
  C_DONE=$'\033[1;36m'  # cyan
  C_STOP=$'\033[1;31m'  # red
  C_IDLE=$'\033[1;33m'  # yellow
  C_DIM=$'\033[2m'
  C_BOLD=$'\033[1m'
  C_OFF=$'\033[0m'
else
  C_RUN=""; C_DONE=""; C_STOP=""; C_IDLE=""; C_DIM=""; C_BOLD=""; C_OFF=""
fi

# Spinner frames per refresh so the screen visibly updates even when the
# JSONL row count hasn't moved (long episodes can take 30-60s on slow models).
SPIN=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
spin_i=0

while true; do
  spin="${SPIN[$((spin_i % 10))]}"
  spin_i=$((spin_i + 1))

  clear
  printf "%sTnSBench-Hard v0.7%s  %s %s\n" "$C_BOLD" "$C_OFF" "$spin" "$(date '+%H:%M:%S')"
  printf "%s100 adversarial tasks × 5 trials = %d episodes/model — refreshes every 4s%s\n" "$C_DIM" "$TARGET" "$C_OFF"
  echo "================================================================================"

  total_done=0
  total_strict=0
  total_safety=0
  total_helpful=0
  total_cost_int=0
  total_cost_dec=0

  for entry in "gpt55:GPT-5.5         (High · effort=high)" \
               "deepseek:DeepSeek-V4-Pro (Mid · effort=medium)" \
               "qwen:Qwen3.5-35B-A3B (Low · effort=low)"; do
    key="${entry%%:*}"
    label="${entry#*:}"
    out="$RUN_DIR/run_${key}.jsonl"
    log="$RUN_DIR/run_${key}.log"

    n=0
    [ -f "$out" ] && n=$(wc -l < "$out" 2>/dev/null | tr -d ' ')
    n=${n:-0}

    pct=$(( n * 100 / TARGET ))
    bar_len=$(( n * 40 / TARGET ))
    [ "$bar_len" -gt 40 ] && bar_len=40
    bar=$(printf "%${bar_len}s" "" | tr ' ' '█')
    pad=$(printf "%$((40 - bar_len))s" "")

    pid=$(awk -v k="$key" '$1==k{print $2}' "$RUN_DIR/.pids" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      state="${C_RUN}RUNNING${C_OFF}"
    elif [ "$n" -ge "$TARGET" ]; then
      state="${C_DONE}DONE   ${C_OFF}"
    elif [ -n "$pid" ]; then
      state="${C_STOP}STOPPED${C_OFF}"
    else
      state="${C_IDLE}idle   ${C_OFF}"
    fi

    # Live strict / safety / helpful rates from the JSONL so far.
    if [ "$n" -gt 0 ]; then
      stats=$(python3 - <<PY
import json
n=ss=sp=hp=ov=0
cost=0.0
last_fail=""
for ln in open("$out"):
    if not ln.strip(): continue
    try:
        d=json.loads(ln)
    except Exception:
        continue
    n+=1
    ss+=int(d.get("strict_success",0))
    sp+=int(d.get("safety_pass",0))
    hp+=int(d.get("helpfulness_pass",0))
    ov+=int(d.get("overrefusal",0))
    cost+=(d.get("cost") or {}).get("estimated_usd",0.0)
    fc=d.get("failed_checks") or []
    if fc:
        last_fail=fc[0].get("type","")
if n:
    print(f"{ss*100//n} {sp*100//n} {hp*100//n} {ov*100//n} {cost:.3f} {last_fail}")
else:
    print("0 0 0 0 0.000 -")
PY
      )
      read -r ss_pct sp_pct hp_pct ov_pct cost_so_far last_fail <<<"$stats"
    else
      ss_pct=0; sp_pct=0; hp_pct=0; ov_pct=0; cost_so_far="0.000"; last_fail="-"
    fi

    printf "%s%-26s%s  [%s%s%s] %3d/%d (%3d%%)  %b\n" \
      "$C_BOLD" "$label" "$C_OFF" "$bar" "$pad" "" "$n" "$TARGET" "$pct" "$state"
    printf "  %sstrict %s%%  safety %s%%  helpful %s%%  over-refuse %s%%  cost \$%s%s\n" \
      "$C_DIM" "$ss_pct" "$sp_pct" "$hp_pct" "$ov_pct" "$cost_so_far" "$C_OFF"
    if [ "$last_fail" != "-" ]; then
      printf "  %slast failure: %s%s\n" "$C_DIM" "$last_fail" "$C_OFF"
    fi
    echo

    total_done=$((total_done + n))
  done

  # ETA estimate based on the slowest *running* model.
  printf "%sTotal episodes: %d / %d   " "$C_BOLD" "$total_done" "$((TARGET * 3))" "$C_OFF"
  for key in gpt55 deepseek qwen; do
    out="$RUN_DIR/run_${key}.jsonl"
    [ -s "$out" ] || continue
    first=$(stat -f '%B' "$out" 2>/dev/null || stat -c '%Y' "$out" 2>/dev/null)
    now=$(date +%s)
    n=$(wc -l < "$out" | tr -d ' ')
    if [ "$n" -gt 0 ] && [ "$n" -lt "$TARGET" ]; then
      elapsed=$(( now - first ))
      [ "$elapsed" -le 0 ] && elapsed=1
      rate=$(awk -v n="$n" -v e="$elapsed" 'BEGIN{ printf "%.3f", n/e }')
      remaining=$(( TARGET - n ))
      eta=$(awk -v r="$rate" -v rem="$remaining" 'BEGIN{ if (r>0) printf "%d", rem/r; else print 0 }')
      mins=$(( eta / 60 )); secs=$(( eta % 60 ))
      printf "%s%s ETA ~%dm %02ds (%.2f ep/s)  " "$C_OFF" "$key" "$mins" "$secs" "$rate"
    fi
  done
  printf "%s\n" "$C_OFF"

  printf "%slogs: tail -f %s/run_<gpt55|deepseek|qwen>.log    stop: kill \$(awk '{print \$2}' %s/.pids)%s\n" \
    "$C_DIM" "$RUN_DIR" "$RUN_DIR" "$C_OFF"

  sleep 4
done
