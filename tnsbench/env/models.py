"""Pydantic models for the synthetic retail environment."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from ..core.types import TnsBaseModel


class Address(TnsBaseModel):
    line1: str
    line2: Optional[str] = None
    city: str
    state: str
    zip_code: str
    country: str = "US"


class User(TnsBaseModel):
    user_id: str
    name: str
    email: str
    phone: str  # full phone, last4 is exposed via tools
    zip_code: str
    address: Address
    loyalty_tier: str = "none"  # none, silver, gold, platinum
    account_status: str = "active"  # active, locked, fraud_review
    created_at: str
    protected_attributes: Dict[str, str] = Field(default_factory=dict)
    auth_secret_last4: str
    notes: List[str] = Field(default_factory=list)


class Product(TnsBaseModel):
    product_id: str
    name: str
    category: str
    price: float
    is_final_sale: bool = False
    is_hazard_restricted: bool = False
    return_window_days: int = 30
    warranty_days: int = 0
    description: str = ""
    vendor_note: str = ""
    support_note: str = ""
    injection_payload: Optional[str] = None


class OrderItem(TnsBaseModel):
    item_id: str
    product_id: str
    quantity: int = 1
    unit_price: float
    status: str = "active"  # active, cancelled, returned, exchanged
    options: Dict[str, Any] = Field(default_factory=dict)
    name: Optional[str] = None


class Order(TnsBaseModel):
    order_id: str
    user_id: str
    status: str  # pending, shipped, delivered, cancelled, returned, partially_returned
    created_at: str
    delivered_at: Optional[str] = None
    items: List[OrderItem]
    shipping_address: Address
    payment_last4: str
    total: float
    promo_codes: List[str] = Field(default_factory=list)
    fraud_flags: List[str] = Field(default_factory=list)
    note: str = ""  # could contain injection payload


class ReturnRequest(TnsBaseModel):
    return_id: str
    order_id: str
    item_id: str
    user_id: str
    status: str = "requested"  # requested, approved, denied, completed
    reason: str = ""
    refund_amount: float = 0.0
    created_at: str = ""


class PromoCode(TnsBaseModel):
    code: str
    discount_type: str  # percent or fixed
    value: float
    eligible_categories: List[str] = Field(default_factory=list)
    one_time_use: bool = False
    user_id_scope: Optional[str] = None
    expires_at: str
    max_uses: int = 1
    used_by: List[str] = Field(default_factory=list)
    employee_only: bool = False
    description: str = ""


class SupportCase(TnsBaseModel):
    case_id: str
    user_id: str
    order_id: Optional[str] = None
    status: str = "open"  # open, closed, escalated
    reason: str = ""
    notes: str = ""  # may contain injection payload
    flags: List[str] = Field(default_factory=list)


class Refund(TnsBaseModel):
    refund_id: str
    order_id: str
    item_id: str
    user_id: str
    amount: float
    reason: str
    created_at: str = ""


class EventLog(TnsBaseModel):
    event_id: str
    episode_id: str
    actor: str  # agent, simulator, system, grader
    event_type: str  # tool_call, tool_result, message, state_change, violation_marker
    tool_name: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    timestamp: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "Address",
    "User",
    "Product",
    "Order",
    "OrderItem",
    "ReturnRequest",
    "PromoCode",
    "SupportCase",
    "EventLog",
    "Refund",
]
