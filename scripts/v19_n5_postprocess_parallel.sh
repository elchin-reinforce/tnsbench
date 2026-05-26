#!/usr/bin/env bash
# Wait for all 10 seed-x-model runs to finish, merge them into the
# canonical _100x5_v19.jsonl files (remapping trial_index = seed - 42),
# then grade with gpt-4o-mini and emit Markdown reports.
#
# Designed to be launched with nohup once.

set -u
cd "$(dirname "$0")/.."

SEEDS=(42 43 44 45 46)

# read pids file written by the launcher
DS_PIDS=()
QW_PIDS=()
if [ -f results/v19_n5_pids.txt ]; then
  while read -r line; do
    case "$line" in
      ds_pids=*) read -ra DS_PIDS <<< "${line#ds_pids=}";;
      qw_pids=*) read -ra QW_PIDS <<< "${line#qw_pids=}";;
    esac
  done < results/v19_n5_pids.txt
fi

echo "[$(date '+%F %T')] watcher (parallel) starting"
echo "  DS_PIDS=${DS_PIDS[@]:-<none>}"
echo "  QW_PIDS=${QW_PIDS[@]:-<none>}"

any_alive() {
  for pid in "$@"; do
    [ -z "$pid" ] && continue
    if kill -0 "$pid" 2>/dev/null; then return 0; fi
  done
  return 1
}

while any_alive "${DS_PIDS[@]}" "${QW_PIDS[@]}"; do
  ds_total=0; qw_total=0
  for s in "${SEEDS[@]}"; do
    n=$(wc -l < results/deepseek_midcap_100x5_v19_seed${s}.jsonl 2>/dev/null | tr -d ' ')
    ds_total=$(( ds_total + ${n:-0} ))
    n=$(wc -l < results/qwen_lowcap_100x5_v19_seed${s}.jsonl 2>/dev/null | tr -d ' ')
    qw_total=$(( qw_total + ${n:-0} ))
  done
  echo "[$(date '+%F %T')]  ds_total=$ds_total/500  qw_total=$qw_total/500"
  sleep 60
done

echo "[$(date '+%F %T')] all 10 seed-runs finished -- merging"

python3 - <<'PY'
import json
from pathlib import Path

ROOT = Path("results")
SEEDS = [42, 43, 44, 45, 46]

for prefix, out_name in [
    ("deepseek_midcap_100x5_v19", "deepseek_midcap_100x5_v19.jsonl"),
    ("qwen_lowcap_100x5_v19",     "qwen_lowcap_100x5_v19.jsonl"),
]:
    out_path = ROOT / out_name
    with out_path.open("w", encoding="utf-8") as fout:
        for seed in SEEDS:
            seed_path = ROOT / f"{prefix}_seed{seed}.jsonl"
            if not seed_path.exists():
                print(f"  WARN: missing {seed_path}")
                continue
            for line in seed_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # Remap trial_index = seed - 42 so the analysis script
                # treats each seed as a distinct "trial".
                row["trial_index"] = seed - 42
                row["seed"] = seed
                fout.write(json.dumps(row) + "\n")
    n = sum(1 for _ in out_path.open())
    print(f"  merged {n} episodes -> {out_path}")
PY

echo "[$(date '+%F %T')] grading deepseek with gpt-4o-mini"
python3 -m tnsbench.cli report \
  results/deepseek_midcap_100x5_v19.jsonl \
  --judge-model gpt-4o-mini \
  --out results/deepseek_midcap_100x5_v19_report.md \
  >> results/v19_n5_postprocess.log 2>&1
echo "[$(date '+%F %T')] grading qwen with gpt-4o-mini"
python3 -m tnsbench.cli report \
  results/qwen_lowcap_100x5_v19.jsonl \
  --judge-model gpt-4o-mini \
  --out results/qwen_lowcap_100x5_v19_report.md \
  >> results/v19_n5_postprocess.log 2>&1

date '+%F %T' > results/v19_n5_DONE
echo "[$(date '+%F %T')] watcher exiting"
