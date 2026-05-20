"""Loader for τ-bench-shaped retail data.

Reads `env/vendor/tau_bench_retail/{users,products,orders}.json` (τ-bench
field names) and adapts records into TnSBench pydantic models. The base
records are then augmented by `overlays.apply_overlays`.

τ-bench upstream layout (we mirror it):

  users.json
    [{user_id, name:{first_name,last_name}, address:{address1,address2,city,country,state,zip},
      email, payment_methods:{<id>:{source,brand,last_four,id}}, orders:[order_id]}]

  products.json
    [{name, product_id, variants:{<item_id>:{item_id, options, available, price}}}]

  orders.json
    [{order_id, user_id, address, items:[{name,product_id,item_id,price,options}],
      fulfillments, status, payment_history}]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .models import Address, Order, OrderItem, Product, User

VENDOR_DIR = Path(__file__).parent / "vendor" / "tau_bench_retail"


def _addr(a: Dict[str, Any]) -> Address:
    return Address(
        line1=a.get("address1") or a.get("line1") or "",
        line2=a.get("address2") or a.get("line2"),
        city=a.get("city", ""),
        state=a.get("state", ""),
        zip_code=str(a.get("zip") or a.get("zip_code") or ""),
        country=a.get("country", "US"),
    )


def load_users() -> List[User]:
    raw = json.loads((VENDOR_DIR / "users.json").read_text())
    out: List[User] = []
    for u in raw:
        name = u.get("name", {})
        if isinstance(name, dict):
            full = f"{name.get('first_name','')} {name.get('last_name','')}".strip()
        else:
            full = str(name)
        addr = _addr(u.get("address", {}))
        # Payment last4 — pull from the first payment method when available.
        last4 = ""
        for pm in (u.get("payment_methods") or {}).values():
            last4 = str(pm.get("last_four", ""))
            break
        # Synthetic phone derived deterministically from user_id so phone last-4
        # auth path still works.
        digits = "".join(c for c in u["user_id"] if c.isdigit()) or "0000"
        phone = f"555-{int(digits[-3:]) % 1000:03d}-{int(digits[-4:]) % 10000:04d}"
        phone_last4 = phone.replace("-", "")[-4:]
        out.append(
            User(
                user_id=u["user_id"],
                name=full,
                email=u.get("email", ""),
                phone=phone,
                zip_code=addr.zip_code,
                address=addr,
                loyalty_tier=u.get("loyalty_tier", "none"),
                account_status=u.get("account_status", "active"),
                created_at=u.get("created_at", "2024-01-01"),
                protected_attributes={},
                auth_secret_last4=phone_last4,
                notes=list(u.get("notes", [])),
            )
        )
    return out


def load_products() -> List[Product]:
    raw = json.loads((VENDOR_DIR / "products.json").read_text())
    out: List[Product] = []
    for p in raw:
        # τ-bench has variants; we flatten by picking the first variant as the
        # canonical product. The variant's item_id is *not* the product_id.
        variants = p.get("variants") or {}
        first_var = next(iter(variants.values()), {})
        out.append(
            Product(
                product_id=p["product_id"],
                name=p.get("name", ""),
                category=p.get("category", "general"),
                price=float(first_var.get("price", p.get("price", 0.0))),
                is_final_sale=bool(p.get("is_final_sale", False)),
                is_hazard_restricted=bool(p.get("is_hazard_restricted", False)),
                return_window_days=int(p.get("return_window_days", 30)),
                warranty_days=int(p.get("warranty_days", 0)),
                description=p.get("description", ""),
                vendor_note=p.get("vendor_note", ""),
                support_note=p.get("support_note", ""),
                injection_payload=p.get("injection_payload"),
            )
        )
    return out


def load_orders() -> List[Order]:
    raw = json.loads((VENDOR_DIR / "orders.json").read_text())
    out: List[Order] = []
    for o in raw:
        items = [
            OrderItem(
                item_id=it["item_id"],
                product_id=it["product_id"],
                quantity=int(it.get("quantity", 1)),
                unit_price=float(it.get("price", 0.0)),
                status=it.get("status", "active"),
                options=dict(it.get("options") or {}),
                name=it.get("name"),
            )
            for it in o.get("items", [])
        ]
        total = sum(i.unit_price * i.quantity for i in items)
        # Use the first payment's last 4 as payment_last4 if present.
        last4 = ""
        for pay in o.get("payment_history", []):
            pmid = pay.get("payment_method_id", "")
            if "_" in pmid:
                last4 = pmid.rsplit("_", 1)[-1]
                break
        out.append(
            Order(
                order_id=o["order_id"],
                user_id=o["user_id"],
                status=o.get("status", "delivered"),
                created_at=o.get("created_at", "2026-04-01"),
                delivered_at=o.get("delivered_at"),
                items=items,
                shipping_address=_addr(o.get("address", {})),
                payment_last4=last4 or "0000",
                total=round(total, 2),
                promo_codes=list(o.get("promo_codes", [])),
                fraud_flags=list(o.get("fraud_flags", [])),
                note=o.get("note", ""),
            )
        )
    return out


def vendor_files_present() -> bool:
    return all((VENDOR_DIR / fn).exists() for fn in ("users.json", "products.json", "orders.json"))
