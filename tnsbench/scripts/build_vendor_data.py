"""Emit τ-bench-shaped retail JSON into env/vendor/tau_bench_retail/.

This builder produces a realistic catalog modeled on τ-bench's retail
products: each product line has options (size, color, switches, ventilation,
etc.) so task scenarios can reference concrete specs ("clicky switches",
"RGB backlight", "Google Home–compatible", "size 8").

The output is deterministic and intended to be checked in. Re-run to refresh.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from tnsbench.env.tau_bench_loader import VENDOR_DIR


SEED = 42
N_USERS = 30
N_ORDERS = 120


# ---------------------------------------------------------------------------
# Personas-for-base-data (the catalog of base names; the *task* personas live
# in tnsbench/tasks/personas.py)
# ---------------------------------------------------------------------------

NAMES = [
    ("Yusuf", "Rossi"),    ("Mia", "Garcia"),     ("Fatima", "Johnson"),
    ("Mei", "Davis"),      ("Ethan", "Garcia"),   ("Sofia", "Hernandez"),
    ("Arjun", "Patel"),    ("Naoko", "Tanaka"),   ("Liam", "OBrien"),
    ("Priya", "Kumar"),    ("Kenji", "Ito"),      ("Aaliyah", "Brown"),
    ("Omar", "Hassan"),    ("Hana", "Lee"),       ("Niko", "Petrov"),
    ("Sara", "Schwartz"),  ("Diego", "Reyes"),    ("Aisha", "Khan"),
    ("Lucas", "Becker"),   ("MeiLin", "Chen"),    ("Jorge", "Costa"),
    ("Yuki", "Park"),      ("Talia", "Mendes"),   ("Hiroshi", "Sato"),
    ("Beatriz", "Lopez"),  ("Owen", "Singh"),     ("Asha", "Williams"),
    ("Henri", "Dubois"),   ("Gabriela", "Costa"), ("Davi", "Nguyen"),
]
CITIES = [
    ("Philadelphia", "PA", "19122"), ("Brooklyn", "NY", "11215"),
    ("Cambridge", "MA", "02139"),    ("Austin", "TX", "78704"),
    ("Seattle", "WA", "98101"),      ("Denver", "CO", "80202"),
    ("Portland", "OR", "97214"),     ("Chicago", "IL", "60607"),
    ("Oakland", "CA", "94612"),      ("Miami", "FL", "33139"),
    ("Madison", "WI", "53703"),      ("Ann Arbor", "MI", "48104"),
    ("Boulder", "CO", "80302"),      ("Pittsburgh", "PA", "15213"),
    ("Tucson", "AZ", "85705"),       ("Honolulu", "HI", "96813"),
    ("Burlington", "VT", "05401"),   ("Nashville", "TN", "37203"),
    ("New York", "NY", "10001"),     ("Asheville", "NC", "28801"),
]


# ---------------------------------------------------------------------------
# Realistic product catalog — 20 lines × ~4 variants each ≈ 80 SKUs
# Each product carries a stable product_id (10-digit) and variants keyed by
# item_id (10-digit), exactly the shape τ-bench uses.
# ---------------------------------------------------------------------------

PRODUCT_LINES: List[Dict[str, Any]] = [
    {
        "name": "Mechanical Keyboard", "category": "electronics",
        "base_price": 159.99, "warranty_days": 365, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"switch": "linear",  "backlight": "RGB",  "size": "full"},
            {"switch": "clicky",  "backlight": "RGB",  "size": "full"},
            {"switch": "tactile", "backlight": "none", "size": "TKL"},
            {"switch": "clicky",  "backlight": "none", "size": "60%"},
        ],
    },
    {
        "name": "Smart Thermostat", "category": "home",
        "base_price": 219.00, "warranty_days": 730, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"compatibility": "Apple HomeKit",  "color": "white"},
            {"compatibility": "Google Home",    "color": "white"},
            {"compatibility": "Amazon Alexa",   "color": "black"},
            {"compatibility": "Matter",         "color": "black"},
        ],
    },
    {
        "name": "Wireless Headphones", "category": "electronics",
        "base_price": 189.00, "warranty_days": 365, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"type": "over-ear",  "wireless": True,  "color": "black"},
            {"type": "over-ear",  "wireless": True,  "color": "silver"},
            {"type": "in-ear",    "wireless": True,  "color": "white"},
            {"type": "on-ear",    "wireless": True,  "color": "navy"},
        ],
    },
    {
        "name": "Smart Watch", "category": "electronics",
        "base_price": 329.00, "warranty_days": 365, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"size": "38mm", "band": "leather",  "color": "black"},
            {"size": "42mm", "band": "metal",    "color": "silver"},
            {"size": "46mm", "band": "silicone", "color": "navy"},
            {"size": "42mm", "band": "silicone", "color": "rose"},
        ],
    },
    {
        "name": "Office Chair", "category": "home",
        "base_price": 279.00, "warranty_days": 730, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"material": "mesh",    "lumbar": True,  "color": "black"},
            {"material": "leather", "lumbar": True,  "color": "black"},
            {"material": "mesh",    "lumbar": False, "color": "grey"},
            {"material": "fabric",  "lumbar": True,  "color": "navy"},
        ],
    },
    {
        "name": "Cycling Helmet", "category": "outdoor",
        "base_price": 129.00, "warranty_days": 365, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"size": "small",  "ventilation": "high", "color": "black"},
            {"size": "medium", "ventilation": "high", "color": "red"},
            {"size": "large",  "ventilation": "low",  "color": "white"},
            {"size": "medium", "ventilation": "low",  "color": "matte black"},
        ],
    },
    {
        "name": "Water Bottle", "category": "outdoor",
        "base_price": 32.00, "warranty_days": 0, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"capacity": "16oz", "material": "steel",   "color": "black"},
            {"capacity": "24oz", "material": "steel",   "color": "ocean"},
            {"capacity": "32oz", "material": "steel",   "color": "sand"},
            {"capacity": "20oz", "material": "glass",   "color": "clear"},
        ],
    },
    {
        "name": "Cotton T-Shirt", "category": "apparel",
        "base_price": 29.00, "warranty_days": 0, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"size": "S",  "color": "black",  "neck": "crew",   "material": "cotton"},
            {"size": "M",  "color": "white",  "neck": "v-neck", "material": "cotton"},
            {"size": "L",  "color": "navy",   "neck": "crew",   "material": "cotton"},
            {"size": "XL", "color": "olive",  "neck": "crew",   "material": "cotton"},
        ],
    },
    {
        "name": "Hiking Boots", "category": "outdoor",
        "base_price": 189.00, "warranty_days": 365, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"size": "8",  "waterproof": True,  "material": "leather"},
            {"size": "9",  "waterproof": True,  "material": "leather"},
            {"size": "10", "waterproof": False, "material": "synthetic"},
            {"size": "11", "waterproof": True,  "material": "leather"},
        ],
    },
    {
        "name": "Travel Backpack", "category": "outdoor",
        "base_price": 119.00, "warranty_days": 365, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"capacity": "20L", "color": "black"},
            {"capacity": "30L", "color": "olive"},
            {"capacity": "45L", "color": "navy"},
            {"capacity": "30L", "color": "grey"},
        ],
    },
    {
        "name": "Bluetooth Speaker", "category": "electronics",
        "base_price": 99.00, "warranty_days": 365, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"size": "portable", "water_resistant": True,  "color": "black"},
            {"size": "medium",   "water_resistant": True,  "color": "blue"},
            {"size": "large",    "water_resistant": False, "color": "grey"},
            {"size": "portable", "water_resistant": False, "color": "white"},
        ],
    },
    {
        "name": "Coffee Maker", "category": "kitchen",
        "base_price": 159.00, "warranty_days": 365, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"capacity": "4-cup",  "color": "black",     "milk_frother": False},
            {"capacity": "8-cup",  "color": "stainless", "milk_frother": True},
            {"capacity": "12-cup", "color": "black",     "milk_frother": False},
            {"capacity": "8-cup",  "color": "white",     "milk_frother": True},
        ],
    },
    {
        "name": "Yoga Mat", "category": "fitness",
        "base_price": 49.00, "warranty_days": 0, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"thickness": "4mm", "color": "purple"},
            {"thickness": "6mm", "color": "blue"},
            {"thickness": "8mm", "color": "black"},
            {"thickness": "6mm", "color": "teal"},
        ],
    },
    {
        "name": "Air Purifier", "category": "home",
        "base_price": 299.00, "warranty_days": 730, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"room_size": "small",  "hepa": True,  "color": "white"},
            {"room_size": "medium", "hepa": True,  "color": "white"},
            {"room_size": "large",  "hepa": True,  "color": "black"},
            {"room_size": "small",  "hepa": False, "color": "white"},
        ],
    },
    {
        "name": "Robot Vacuum", "category": "home",
        "base_price": 449.00, "warranty_days": 365, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"with_mop": True,  "color": "black"},
            {"with_mop": False, "color": "white"},
            {"with_mop": True,  "color": "grey"},
            {"with_mop": False, "color": "black"},
        ],
    },
    {
        "name": "E-Reader", "category": "electronics",
        "base_price": 149.00, "warranty_days": 365, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"storage": "8GB",  "frontlight": True,  "color": "black"},
            {"storage": "16GB", "frontlight": True,  "color": "black"},
            {"storage": "32GB", "frontlight": True,  "color": "white"},
            {"storage": "16GB", "frontlight": False, "color": "black"},
        ],
    },
    {
        "name": "Pet Bed", "category": "home",
        "base_price": 79.00, "warranty_days": 0, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"size": "small",  "color": "grey"},
            {"size": "medium", "color": "brown"},
            {"size": "large",  "color": "navy"},
            {"size": "medium", "color": "olive"},
        ],
    },
    {
        "name": "Standing Desk", "category": "home",
        "base_price": 599.00, "warranty_days": 1095, "return_window_days": 30,
        "is_final_sale": False,
        "option_grid": [
            {"width_in": 48, "color": "oak"},
            {"width_in": 60, "color": "walnut"},
            {"width_in": 72, "color": "white"},
            {"width_in": 60, "color": "black"},
        ],
    },
    {
        "name": "Garden Tool Set", "category": "outdoor",
        "base_price": 89.00, "warranty_days": 365, "return_window_days": 15,
        "is_final_sale": False,
        "option_grid": [
            {"pieces": 4, "handle": "wood"},
            {"pieces": 6, "handle": "wood"},
            {"pieces": 8, "handle": "fiberglass"},
            {"pieces": 6, "handle": "fiberglass"},
        ],
    },
    {
        "name": "Limited-Edition Vinyl Record", "category": "books",
        "base_price": 39.00, "warranty_days": 0, "return_window_days": 0,
        "is_final_sale": True,  # this product line is final-sale
        "option_grid": [
            {"color": "splatter", "weight": "180g"},
            {"color": "black",    "weight": "180g"},
            {"color": "clear",    "weight": "150g"},
            {"color": "marble",   "weight": "180g"},
        ],
    },
]


def _id_from(prefix: str, key: str, modulo: int = 9_000_000_000, offset: int = 1_000_000_000) -> str:
    n = int(hashlib.sha256(f"{prefix}|{key}".encode()).hexdigest()[:10], 16) % modulo + offset
    return f"{n}"


def build_users() -> List[Dict[str, Any]]:
    out = []
    for i in range(N_USERS):
        first, last = NAMES[i % len(NAMES)]
        city, state, zipc = CITIES[i % len(CITIES)]
        last4 = f"{(1000 + i * 7) % 10000:04d}"
        pmid = f"credit_card_{last4}"
        out.append(
            {
                "user_id": f"user_id_{i + 1}",
                "name": {"first_name": first, "last_name": last},
                "address": {
                    "address1": f"{100 + i} Main St",
                    "address2": None,
                    "city": city,
                    "country": "USA",
                    "state": state,
                    "zip": zipc,
                },
                "email": f"{first.lower()}.{last.lower()}{i}@example.com",
                "payment_methods": {
                    pmid: {
                        "source": "credit_card",
                        "brand": "visa" if i % 2 == 0 else "mastercard",
                        "last_four": last4,
                        "id": pmid,
                    }
                },
                "orders": [],
                "loyalty_tier": ["none", "silver", "gold", "platinum"][i % 4],
            }
        )
    return out


def build_products() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for li, line in enumerate(PRODUCT_LINES):
        product_id = _id_from("p", line["name"])
        variants: Dict[str, Any] = {}
        for vi, opts in enumerate(line["option_grid"]):
            item_id = _id_from("i", f"{line['name']}|{vi}")
            # Slight price spread by variant.
            price = round(line["base_price"] * (1.0 + 0.05 * vi), 2)
            variants[item_id] = {
                "item_id": item_id,
                "options": dict(opts),
                "available": True,
                "price": price,
            }
        out.append(
            {
                "name": line["name"],
                "product_id": product_id,
                "category": line["category"],
                "is_final_sale": line["is_final_sale"],
                "is_hazard_restricted": False,
                "return_window_days": line["return_window_days"],
                "warranty_days": line["warranty_days"],
                "description": f"{line['name']} — available in multiple options.",
                "vendor_note": "",
                "support_note": "",
                "variants": variants,
            }
        )
    return out


def build_orders(users: List[Dict[str, Any]], products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """120 orders distributed across statuses + dates so tasks can target
    specific situations (delivered just inside window, just outside, pending,
    shipped, returned, etc.)."""
    out: List[Dict[str, Any]] = []
    statuses = ["pending", "shipped", "delivered", "cancelled", "returned", "partially_returned"]
    # benchmark "today" is 2026-05-19; we use it to date delivered orders.
    for i in range(N_ORDERS):
        user = users[i % len(users)]
        # Mix products: each order has 1-3 items pulled from different lines.
        item_count = 1 + (i % 3)
        items: List[Dict[str, Any]] = []
        total = 0.0
        for j in range(item_count):
            p = products[(i * 5 + j * 7) % len(products)]
            v = list(p["variants"].values())[(i + j) % len(p["variants"])]
            qty = 1 + (j % 2)
            items.append(
                {
                    "name": p["name"],
                    "product_id": p["product_id"],
                    "item_id": v["item_id"],
                    "price": v["price"],
                    "options": v["options"],
                    "quantity": qty,
                    "status": "active",
                }
            )
            total += v["price"] * qty
        status = statuses[i % len(statuses)]
        delivered_at = None
        if status in ("delivered", "returned", "partially_returned"):
            # Spread delivery dates from 5 days ago to 90 days ago so tasks
            # can target inside-window vs outside-window.
            day_offset = 5 + (i * 4) % 90
            # Anchor: 2026-05-19; subtract day_offset days.
            from datetime import date, timedelta
            d = date(2026, 5, 19) - timedelta(days=day_offset)
            delivered_at = d.isoformat()
        pmid = next(iter(user["payment_methods"].keys()))
        order_id = "#W" + _id_from("o", str(i), modulo=9_000_000, offset=1_000_000)
        out.append(
            {
                "order_id": order_id,
                "user_id": user["user_id"],
                "address": dict(user["address"]),
                "items": items,
                "fulfillments": [
                    {"tracking_id": [f"TRK{(i * 7) % 1_000_000:06d}"], "item_ids": [it["item_id"] for it in items]}
                ],
                "status": status,
                "payment_history": [
                    {"transaction_type": "payment", "amount": round(total, 2), "payment_method_id": pmid}
                ],
                "created_at": "2026-04-01",
                "delivered_at": delivered_at,
            }
        )
        user["orders"].append(order_id)
    return out


def main() -> int:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    users = build_users()
    products = build_products()
    orders = build_orders(users, products)
    (VENDOR_DIR / "users.json").write_text(json.dumps(users, indent=2))
    (VENDOR_DIR / "products.json").write_text(json.dumps(products, indent=2))
    (VENDOR_DIR / "orders.json").write_text(json.dumps(orders, indent=2))
    print(f"Wrote {len(users)} users, {len(products)} products, {len(orders)} orders to {VENDOR_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
