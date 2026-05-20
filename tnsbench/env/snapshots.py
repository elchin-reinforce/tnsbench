"""Snapshot and diff utilities for the retail DB."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from .database import RetailDB


def snapshot(db: RetailDB) -> Dict[str, Any]:
    """Take a JSON-serializable snapshot of the DB."""
    return {
        "users": {k: v.model_dump() for k, v in db.users.items()},
        "products": {k: v.model_dump() for k, v in db.products.items()},
        "orders": {k: v.model_dump() for k, v in db.orders.items()},
        "promos": {k: v.model_dump() for k, v in db.promos.items()},
        "support_cases": {k: v.model_dump() for k, v in db.support_cases.items()},
        "returns": {k: v.model_dump() for k, v in db.returns.items()},
        "refunds": {k: v.model_dump() for k, v in db.refunds.items()},
    }


def snapshot_hash(snap: Dict[str, Any]) -> str:
    payload = json.dumps(snap, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def diff_snapshots(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Return a structured diff between two snapshots."""
    diff: Dict[str, Any] = {}
    keys = set(before.keys()) | set(after.keys())
    for k in keys:
        b = before.get(k, {})
        a = after.get(k, {})
        diff[k] = _diff_dict(b, a)
    return diff


def _diff_dict(b: Dict[str, Any], a: Dict[str, Any]) -> Dict[str, Any]:
    added: Dict[str, Any] = {}
    removed: Dict[str, Any] = {}
    changed: Dict[str, Any] = {}
    keys = set(b.keys()) | set(a.keys())
    for k in keys:
        if k in a and k not in b:
            added[k] = a[k]
        elif k in b and k not in a:
            removed[k] = b[k]
        elif b[k] != a[k]:
            changed[k] = {"before": b[k], "after": a[k]}
    return {"added": added, "removed": removed, "changed": changed}


def _flatten_changes(diff: Dict[str, Any]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for kind, sections in diff.items():
        for op in ("added", "removed", "changed"):
            for key, val in sections.get(op, {}).items():
                flat.append({"kind": kind, "op": op, "key": key, "value": val})
    return flat


def changed_kinds(diff: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for kind, sections in diff.items():
        if any(sections.get(op) for op in ("added", "removed", "changed")):
            out.append(kind)
    return out
