"""Select v0.8 task slice from pilot-calibration JSON output.

The pilot's `keep` list is the authoritative safety-separating set. This
script:
  1. Reads the calibration JSON.
  2. Loads the current task pool (defaults to adversarial_tasks.jsonl).
  3. Emits a selected set respecting the v0.8 category distribution.
  4. Reports which tasks must be REWRITTEN (drop_reasons explain why).

Usage:
    python -m tnsbench.scripts.select_v08 \\
        --calibration results/v07_run_partial/safety_calibration.json \\
        --pool tnsbench/tasks/adversarial_tasks.jsonl \\
        --out  results/v08_selection.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..tasks.task_loader import load_tasks


# Target distribution (v0.7 spec, carried forward).
TARGET_DIST = {
    "fraud": 12, "account_takeover": 12, "pii": 12, "prompt_injection": 12,
    "mixed": 12, "edge_case": 10, "jailbreak": 10, "bias": 20,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", type=Path, required=True,
                    help="JSON output from pilot_calibration.py")
    ap.add_argument("--pool", type=Path, default=None,
                    help="Task pool JSONL (defaults to the configured adversarial set).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Where to write the selection report (JSON).")
    args = ap.parse_args()

    calib = json.loads(args.calibration.read_text(encoding="utf-8"))
    metric = calib.get("metric", "safety_pass")
    keep_ids = set(calib.get("keep", []))
    drop_reasons = calib.get("drop_reasons", {})

    tasks = load_tasks(args.pool) if args.pool else load_tasks()
    by_id = {t.id: t for t in tasks}

    cat_keep = defaultdict(list)
    cat_drop = defaultdict(list)
    for t in tasks:
        if t.id in keep_ids:
            cat_keep[t.primary_category].append(t.id)
        else:
            cat_drop[t.primary_category].append(t.id)

    # Decide selection per category: take all KEEP, fill the remainder from
    # tasks for which we have NO pilot signal yet (unseen by the calibration
    # run because the run was cut short). Those tasks are "unknown" — not
    # known to be saturated. Tasks explicitly in `drop` are excluded.
    seen_ids = set(calib.get("per_task", {}).keys())
    selection: Dict[str, List[str]] = {}
    rewrite: Dict[str, List[str]] = {}
    for cat, target_n in TARGET_DIST.items():
        keep_for_cat = cat_keep.get(cat, [])
        all_for_cat = [t.id for t in tasks if t.primary_category == cat]
        unseen_for_cat = [tid for tid in all_for_cat
                          if tid not in seen_ids and tid not in keep_for_cat]
        chosen = list(keep_for_cat) + unseen_for_cat
        chosen = chosen[:target_n]
        selection[cat] = chosen
        # Anything we couldn't fill is a rewrite need.
        if len(chosen) < target_n:
            rewrite[cat] = [
                f"NEEDS_NEW_TASK_{cat}_{i+1}" for i in range(target_n - len(chosen))
            ]

    flat_selected = [tid for ids in selection.values() for tid in ids]
    out = {
        "metric": metric,
        "selected_count": len(flat_selected),
        "selected_by_category": selection,
        "rewrite_needed_by_category": rewrite,
        "keep_from_calibration": sorted(keep_ids),
        "drop_from_calibration": [
            {"task_id": tid, "reason": reason}
            for tid, reason in drop_reasons.items()
        ],
        "tasks_not_yet_piloted": sorted(set(by_id) - seen_ids),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"Selected {len(flat_selected)} tasks. Wrote {args.out}.")
    print("By category:")
    for cat, ids in selection.items():
        n_keep = sum(1 for x in ids if x in keep_ids)
        print(f"  {cat:<18s} {len(ids):>3d}  (calibrated-keep: {n_keep})")
    if rewrite:
        print()
        print("Categories needing new tasks (pilot showed saturation):")
        for cat, missing in rewrite.items():
            print(f"  {cat:<18s} needs {len(missing)} replacement(s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
