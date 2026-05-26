#!/usr/bin/env bash
# Wait for the DeepSeek + Qwen N=5 v1.9 runs to finish, then grade each
# through gpt-4o-mini and produce a Markdown report. Self-contained so it
# can be launched with nohup and outlive the spawning shell.

set -u
cd "$(dirname "$0")/.."

DS_PID="$(sed 's/pid=//' < results/deepseek_midcap_100x5_v19.pid 2>/dev/null || true)"
QW_PID="$(sed 's/pid=//' < results/qwen_lowcap_100x5_v19.pid 2>/dev/null || true)"

echo "[$(date '+%F %T')] post-process watcher starting"
echo "  DS_PID=${DS_PID:-<none>}"
echo "  QW_PID=${QW_PID:-<none>}"

is_alive() {
  local pid="$1"
  [ -z "$pid" ] && return 1
  kill -0 "$pid" 2>/dev/null
}

while is_alive "$DS_PID" || is_alive "$QW_PID"; do
  ds_state="dead"
  qw_state="dead"
  is_alive "$DS_PID" && ds_state="alive"
  is_alive "$QW_PID" && qw_state="alive"
  ds_n=$(wc -l < results/deepseek_midcap_100x5_v19.jsonl 2>/dev/null | tr -d ' ')
  qw_n=$(wc -l < results/qwen_lowcap_100x5_v19.jsonl 2>/dev/null | tr -d ' ')
  echo "[$(date '+%F %T')]  ds=$ds_state ($ds_n/500)   qw=$qw_state ($qw_n/500)"
  sleep 60
done
echo "[$(date '+%F %T')] both runs finished -- grading + reporting"

# Grade + report DeepSeek
python3 -m tnsbench.cli report \
  results/deepseek_midcap_100x5_v19.jsonl \
  --judge-model gpt-4o-mini \
  --out results/deepseek_midcap_100x5_v19_report.md \
  >> results/deepseek_midcap_100x5_v19.log 2>&1
echo "[$(date '+%F %T')] deepseek report done"

# Grade + report Qwen
python3 -m tnsbench.cli report \
  results/qwen_lowcap_100x5_v19.jsonl \
  --judge-model gpt-4o-mini \
  --out results/qwen_lowcap_100x5_v19_report.md \
  >> results/qwen_lowcap_100x5_v19.log 2>&1
echo "[$(date '+%F %T')] qwen report done"

date '+%F %T' > results/v19_n5_DONE
echo "[$(date '+%F %T')] watcher exiting"
