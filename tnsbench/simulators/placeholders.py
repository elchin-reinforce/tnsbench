"""Auth-credential placeholder resolution for scripted simulator turns.

Tasks may embed templated credentials in their `scripted_turns[*].user_message`
so the simulator emits realistic auth values at runtime instead of vague
placeholders like "email + zip provided". The resolver looks them up against
the *current* RetailDB (the same DB the agent will see), so the agent
genuinely has to validate them against tool output.

Supported placeholders:
    {user.email}              -> task.user_profile_id's email
    {user.name}               -> name
    {user.zip}                -> zip_code
    {user.phone_last4}        -> last 4 digits of phone
    {user.address.line1}      -> shipping address line1
    {user.address.city}       -> city
    {user.address.zip}        -> address zip
    {user.payment_last4}      -> auth_secret_last4

    {target_user.email}       -> task.target_user_id's email  (cross-user fields)
    {target_user.zip}         -> ...

    {order.id}                -> task.order_id (echo)
    {order.item_id}           -> first item_id of task.order_id
    {order.product_name}      -> first item.name of task.order_id

    {wrong_zip}               -> a deterministically-different zip from the
                                 real one (off-by-one digit) — used to seed
                                 partial-auth confusion traps.
    {wrong_email}             -> the email of a *different* user, sampled
                                 deterministically by task.id

Tasks may use either `{...}` or `<<...>>` delimiters. The resolver is
non-strict on unknown keys (it leaves them alone) so older tasks without
placeholders are unaffected.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Optional

# Recognised by the linter as auth-resolved placeholders. Kept in sync with
# `RESOLVABLE_PLACEHOLDERS` so a scripted turn that only uses these (vs the
# vague "email + zip provided" sentence) passes the lint rule.
RESOLVABLE_PLACEHOLDERS = {
    "{user.email}", "{user.name}", "{user.zip}", "{user.phone_last4}",
    "{user.payment_last4}", "{user.address.line1}", "{user.address.city}",
    "{user.address.zip}",
    "{target_user.email}", "{target_user.name}", "{target_user.zip}",
    "{target_user.phone_last4}", "{target_user.payment_last4}",
    "{order.id}", "{order.item_id}", "{order.product_name}",
    "{wrong_zip}", "{wrong_email}",
}

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_.]*)\}")


def _wrong_zip(real_zip: str) -> str:
    if not real_zip or not real_zip.isdigit():
        return "00000"
    # Flip the last digit deterministically.
    last = int(real_zip[-1])
    new_last = (last + 5) % 10
    return real_zip[:-1] + str(new_last)


def _wrong_email_for(task_id: str, db) -> str:
    # Pick a deterministically-different user's email.
    h = int(hashlib.md5((task_id or "x").encode()).hexdigest(), 16)
    users = list(db.users.values()) if db is not None else []
    if not users:
        return "unknown@example.com"
    return users[h % len(users)].email


def _user_field(user, attr: str) -> str:
    if user is None:
        return ""
    if attr == "email":
        return user.email
    if attr == "name":
        return user.name
    if attr == "zip":
        return user.zip_code
    if attr == "phone_last4":
        return (user.phone or "").replace("-", "").replace(" ", "")[-4:]
    if attr == "payment_last4":
        return user.auth_secret_last4 or ""
    if attr.startswith("address."):
        sub = attr.split(".", 1)[1]
        addr = user.address
        if sub == "line1":
            return addr.line1
        if sub == "city":
            return addr.city
        if sub == "zip":
            return addr.zip_code
        if sub == "state":
            return addr.state
    return ""


def _order_field(order, attr: str) -> str:
    if order is None:
        return ""
    if attr == "id":
        return order.order_id
    if attr == "item_id":
        return order.items[0].item_id if order.items else ""
    if attr == "product_name":
        return order.items[0].name if (order.items and order.items[0].name) else ""
    return ""


def resolve(text: str, task, db) -> str:
    """Resolve placeholders in ``text`` against ``db`` for ``task``.

    Unknown keys are left intact so a missing FK doesn't silently corrupt
    the scripted turn — the linter is responsible for flagging missing FKs.
    """
    if not text or "{" not in text:
        return text
    user = db.users.get(getattr(task, "user_profile_id", None)) if db is not None else None
    target = db.users.get(getattr(task, "target_user_id", None)) if (db is not None and getattr(task, "target_user_id", None)) else None
    order = db.orders.get(getattr(task, "order_id", None)) if (db is not None and getattr(task, "order_id", None)) else None

    def _sub(m: "re.Match[str]") -> str:
        key = m.group(1)
        if key == "wrong_zip":
            return _wrong_zip(user.zip_code if user else "")
        if key == "wrong_email":
            return _wrong_email_for(task.id, db)
        if key.startswith("user."):
            return _user_field(user, key.split(".", 1)[1])
        if key.startswith("target_user."):
            return _user_field(target, key.split(".", 1)[1])
        if key.startswith("order."):
            return _order_field(order, key.split(".", 1)[1])
        return m.group(0)
    return _PLACEHOLDER_RE.sub(_sub, text)


__all__ = ["resolve", "RESOLVABLE_PLACEHOLDERS"]
