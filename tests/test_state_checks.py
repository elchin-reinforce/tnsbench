from tnsbench.env.database import RetailDB
from tnsbench.env.events import EventLogger
from tnsbench.env.snapshots import snapshot
from tnsbench.grading.state_checks import (
    check_no_address_change,
    check_no_cross_user_access,
    check_no_promo_applied,
    check_no_refund,
)
from tnsbench.tools.retail_tools import build_registry


def test_forbidden_refund_detected():
    db = RetailDB()
    events = EventLogger("ep")
    reg = build_registry(db, events)
    before = snapshot(db)
    order_id = next(iter(db.orders))
    item_id = db.orders[order_id].items[0].item_id
    reg.call("issue_refund", {"order_id": order_id, "item_id": item_id, "amount": 10.0, "reason": "x"})
    after = snapshot(db)
    ok, _ = check_no_refund(None, before, after, events.events, {"order_id": order_id, "item_id": item_id})
    assert ok is False


def test_unauthorized_address_change_detected():
    db = RetailDB()
    events = EventLogger("ep")
    reg = build_registry(db, events)
    before = snapshot(db)
    uid = next(iter(db.users))
    reg.call("update_user_address", {"user_id": uid, "new_address": {"line1": "Bad Rd"}})
    after = snapshot(db)
    ok, _ = check_no_address_change(None, before, after, events.events, {"user_id": uid})
    assert ok is False


def test_promo_abuse_detected():
    db = RetailDB()
    events = EventLogger("ep")
    reg = build_registry(db, events)
    order_id = next(iter(db.orders))
    code = next(iter(db.promos))
    before = snapshot(db)
    reg.call("apply_promo_code", {"order_id": order_id, "code": code})
    after = snapshot(db)
    ok, _ = check_no_promo_applied(None, before, after, events.events, {"order_id": order_id, "code": code})
    assert ok is False


def test_cross_user_access_detected():
    db = RetailDB()
    events = EventLogger("ep")
    reg = build_registry(db, events)
    target = next(iter(db.users))
    reg.call("get_user_profile", {"user_id": target})
    after = snapshot(db)
    ok, _ = check_no_cross_user_access(None, {}, after, events.events, {"target_user_id": target})
    assert ok is False
