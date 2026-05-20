from tnsbench.env.database import RetailDB
from tnsbench.env.snapshots import diff_snapshots, snapshot, snapshot_hash


def test_base_db_loads():
    db = RetailDB()
    assert len(db.users) >= 30
    # τ-bench-shaped catalog: 20 product lines (each with variants in the
    # source JSON, flattened to one canonical record in our model).
    assert len(db.products) >= 20
    assert len(db.orders) >= 120
    assert len(db.promos) >= 20
    assert len(db.support_cases) == 20


def test_snapshot_hash_stable():
    db1 = RetailDB()
    db2 = RetailDB()
    h1 = snapshot_hash(snapshot(db1))
    h2 = snapshot_hash(snapshot(db2))
    assert h1 == h2


def test_patch_applies():
    db = RetailDB()
    uid = next(iter(db.users))
    db.apply_patch({"users": {uid: {"account_status": "locked"}}})
    assert db.users[uid].account_status == "locked"


def test_diff_detects_changes():
    db = RetailDB()
    before = snapshot(db)
    uid = next(iter(db.users))
    db.apply_patch({"users": {uid: {"account_status": "fraud_review"}}})
    after = snapshot(db)
    d = diff_snapshots(before, after)
    assert "users" in d
    assert uid in d["users"]["changed"]
