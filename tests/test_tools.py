from tnsbench.core.types import ToolError, ToolResult
from tnsbench.env.database import RetailDB
from tnsbench.env.events import EventLogger
from tnsbench.tools.retail_tools import build_registry


def _setup():
    db = RetailDB()
    events = EventLogger("ep_test")
    reg = build_registry(db, events)
    return db, events, reg


def test_issue_refund_logs_event_and_changes_state():
    db, events, reg = _setup()
    order_id = next(iter(db.orders))
    item_id = db.orders[order_id].items[0].item_id
    out = reg.call("issue_refund", {"order_id": order_id, "item_id": item_id, "amount": 25.0, "reason": "test"})
    assert isinstance(out, ToolResult)
    assert any(e.event_type == "tool_call" and e.tool_name == "issue_refund" for e in events.events)
    assert len(db.refunds) == 1


def test_update_address_logs_event():
    db, events, reg = _setup()
    uid = next(iter(db.users))
    out = reg.call("update_user_address", {"user_id": uid, "new_address": {"line1": "999 New Rd"}})
    assert isinstance(out, ToolResult)
    assert db.users[uid].address.line1 == "999 New Rd"


def test_invalid_ids_return_tool_error():
    db, events, reg = _setup()
    err = reg.call("get_order", {"order_id": "does_not_exist"})
    assert isinstance(err, ToolError)
    assert err.code == "NOT_FOUND"


def test_evaluator_only_not_in_registry():
    db, events, reg = _setup()
    names = set(reg.names())
    for forbidden in [
        "inspect_db", "diff_snapshots", "get_events", "get_protected_attributes",
        "check_refund_issued", "check_promo_applied", "check_address_changed",
    ]:
        assert forbidden not in names


def test_invalid_refund_amount_rejected():
    db, events, reg = _setup()
    order_id = next(iter(db.orders))
    item_id = db.orders[order_id].items[0].item_id
    out = reg.call("issue_refund", {"order_id": order_id, "item_id": item_id, "amount": -10, "reason": "bad"})
    assert isinstance(out, ToolError)
    assert out.code == "INVALID_ARGUMENT"
