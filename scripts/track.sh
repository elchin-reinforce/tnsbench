#!/usr/bin/env bash
# Auto-refreshing live tracker. Run it once, leave it open:
#   bash scripts/track.sh
# Ctrl+C to stop (won't affect the running benchmark).

cd "$(dirname "$0")/.."
RUN_DIR="results/comparison_run"
# 5 trials × 120 tasks = 600 episodes per model
TARGET=600

while true; do
  clear
  printf "TnSBench live  —  %s   (refresh every 5s, Ctrl+C to stop)\n" "$(date '+%H:%M:%S')"
  printf "%s\n" "----------------------------------------------------------------"

  for entry in "gpt55:GPT-5.5         (High)" \
               "deepseek:DeepSeek-V4-Pro (Mid)" \
               "qwen:Qwen3.5-35B-A3B (Low)"; do
    key="${entry%%:*}"
    label="${entry#*:}"
    out="${RUN_DIR}/run_${key}.jsonl"
    log="${RUN_DIR}/run_${key}.log"

    if [ -f "$out" ]; then
      n=$(wc -l < "$out" 2>/dev/null | tr -d ' ')
    else
      n=0
    fi
    n=${n:-0}
    pct=$(( n * 100 / TARGET ))
    bar_len=$(( n * 30 / TARGET ));  [ "$bar_len" -gt 30 ] && bar_len=30
    bar=$(printf "%${bar_len}s" "" | tr ' ' '#')

    # State: RUNNING if pid alive, DONE if 120/120, else STOPPED/idle
    state="idle"
    pid=$(awk -v k="$key" '$1==k{print $2}' "${RUN_DIR}/.pids" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      state="RUNNING"
    elif [ "$n" -ge "$TARGET" ]; then
      state="DONE"
    elif [ -n "$pid" ]; then
      state="STOPPED"
    fi

    printf "  %-26s [%-30s] %3d/%d  %3d%%  %s\n" "$label" "$bar" "$n" "$TARGET" "$pct" "$state"
  done

  # ETA: rate from the slowest still-running model
  printf "\n"
  for key in gpt55 deepseek qwen; do
    out="${RUN_DIR}/run_${key}.jsonl"
    [ -s "$out" ] || continue
    first=$(stat -f '%B' "$out" 2>/dev/null || stat -c '%Y' "$out" 2>/dev/null)
    now=$(date +%s)
    n=$(wc -l < "$out" | tr -d ' ')
    if [ "$n" -gt 0 ] && [ "$n" -lt "$TARGET" ]; then
      elapsed=$(( now - first ))
      rate=$(awk -v n="$n" -v e="$elapsed" 'BEGIN{ if (e>0) printf "%.2f", n/e; else print 0 }')
      remaining=$(( TARGET - n ))
      eta=$(awk -v r="$rate" -v rem="$remaining" 'BEGIN{ if (r>0) printf "%d", rem/r; else print 0 }')
      mins=$(( eta / 60 )); secs=$(( eta % 60 ))
      printf "  %-26s ~%dm %02ds remaining (rate %s ep/s)\n" "$key" "$mins" "$secs" "$rate"
    fi
  done

  sleep 5
done
