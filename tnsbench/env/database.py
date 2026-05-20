"""Mutable in-memory retail database used during an episode."""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional

from .models import (
    Address,
    Order,
    OrderItem,
    PromoCode,
    Product,
    Refund,
    ReturnRequest,
    SupportCase,
    User,
)
from .retail_data import build_base_data


class RetailDB:
    """Holds all retail state for a single episode.

    Tasks may apply an initial_state_patch to deviate from the base.
    Tools mutate the DB; the snapshot system captures before/after diffs.
    """

    def __init__(self, base_seed: int = 42) -> None:
        data = build_base_data(seed=base_seed)
        self.users: Dict[str, User] = {u.user_id: u for u in data["users"]}
        self.products: Dict[str, Product] = {p.product_id: p for p in data["products"]}
        self.orders: Dict[str, Order] = {o.order_id: o for o in data["orders"]}
        self.promos: Dict[str, PromoCode] = {c.code: c for c in data["promos"]}
        self.support_cases: Dict[str, SupportCase] = {c.case_id: c for c in data["support_cases"]}
        self.returns: Dict[str, ReturnRequest] = {}
        self.refunds: Dict[str, Refund] = {}

    # ---------- lookups ----------
    def get_user(self, user_id: str) -> Optional[User]:
        return self.users.get(user_id)

    def find_user_by_email(self, email: str) -> Optional[User]:
        email_l = (email or "").strip().lower()
        for u in self.users.values():
            if u.email.lower() == email_l:
                return u
        return None

    def find_user_by_name_zip(self, name: str, zip_code: str) -> Optional[User]:
        n = (name or "").strip().lower()
        z = (zip_code or "").strip()
        for u in self.users.values():
            if u.name.lower() == n and u.zip_code == z:
                return u
        return None

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)

    def list_user_orders(self, user_id: str) -> List[Order]:
        return [o for o in self.orders.values() if o.user_id == user_id]

    def get_product(self, product_id: str) -> Optional[Product]:
        return self.products.get(product_id)

    def search_products(self, query: str, category: Optional[str] = None) -> List[Product]:
        q = (query or "").lower()
        results: List[Product] = []
        for p in self.products.values():
            if category and p.category != category:
                continue
            if q in p.name.lower() or q in p.category.lower() or q in p.description.lower():
                results.append(p)
        return results[:20]

    def get_promo(self, code: str) -> Optional[PromoCode]:
        return self.promos.get(code.upper())

    def get_support_case(self, case_id: str) -> Optional[SupportCase]:
        return self.support_cases.get(case_id)

    # ---------- mutations ----------
    def apply_patch(self, patch: Dict[str, Any]) -> None:
        """Apply a task-defined initial_state_patch to the DB.

        Patch shape:
          {
            "users": {"u_001": {"account_status": "fraud_review"}},
            "products": {"p_005": {"injection_payload": "..."}},
            "orders": {"o_0010": {"status": "delivered", "delivered_at": "..."}},
            "promos": {"PROMO00": {"expires_at": "2020-01-01"}},
            "support_cases": {"sc_004": {"notes": "..."}}
          }
        """
        for kind, items in (patch or {}).items():
            target = getattr(self, kind, None)
            if not isinstance(target, dict):
                continue
            for key, updates in items.items():
                if key not in target:
                    continue
                obj = target[key]
                obj_dict = obj.model_dump()
                deep_merge(obj_dict, updates)
                target[key] = obj.__class__(**obj_dict)

    def update_user_address(self, user_id: str, new_address: Dict[str, Any]) -> None:
        u = self.users[user_id]
        d = u.model_dump()
        d["address"].update(new_address)
        if "zip_code" in new_address:
            d["zip_code"] = new_address["zip_code"]
        self.users[user_id] = User(**d)

    def cancel_order(self, order_id: str) -> None:
        o = self.orders[order_id]
        d = o.model_dump()
        d["status"] = "cancelled"
        for it in d["items"]:
            it["status"] = "cancelled"
        self.orders[order_id] = Order(**d)

    def add_refund(self, refund: Refund) -> None:
        self.refunds[refund.refund_id] = refund

    def add_return(self, ret: ReturnRequest) -> None:
        self.returns[ret.return_id] = ret

    def mark_item_returned(self, order_id: str, item_id: str) -> None:
        o = self.orders[order_id]
        d = o.model_dump()
        for it in d["items"]:
            if it["item_id"] == item_id:
                it["status"] = "returned"
        # transition order status
        statuses = {it["status"] for it in d["items"]}
        if statuses == {"returned"}:
            d["status"] = "returned"
        elif "returned" in statuses:
            d["status"] = "partially_returned"
        self.orders[order_id] = Order(**d)

    def apply_promo_to_order(self, order_id: str, code: str) -> None:
        o = self.orders[order_id]
        d = o.model_dump()
        if code.upper() not in d["promo_codes"]:
            d["promo_codes"].append(code.upper())
        self.orders[order_id] = Order(**d)
        # mark code used
        promo = self.promos[code.upper()]
        pd = promo.model_dump()
        pd["used_by"].append(o.user_id)
        self.promos[code.upper()] = PromoCode(**pd)

    def add_fraud_flag(self, user_id: Optional[str], order_id: Optional[str], reason: str) -> None:
        if order_id and order_id in self.orders:
            o = self.orders[order_id]
            d = o.model_dump()
            if reason not in d["fraud_flags"]:
                d["fraud_flags"].append(reason)
            self.orders[order_id] = Order(**d)
        if user_id and user_id in self.users:
            u = self.users[user_id]
            d = u.model_dump()
            d["notes"].append(f"flag:{reason}")
            self.users[user_id] = User(**d)

    def add_support_case(self, case: SupportCase) -> None:
        self.support_cases[case.case_id] = case

    def escalate_case(self, case_id: str, reason: str) -> None:
        if case_id not in self.support_cases:
            return
        c = self.support_cases[case_id]
        d = c.model_dump()
        d["status"] = "escalated"
        d["flags"].append(f"escalated:{reason}")
        self.support_cases[case_id] = SupportCase(**d)


def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst
