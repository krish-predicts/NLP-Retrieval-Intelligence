"""Built-in evaluation queries with domain-adapted terms."""

from __future__ import annotations

EVAL_QUERIES: dict[str, list[str]] = {
    "battery issues": ["battery", "power", "charge"],
    "shipping problems": ["shipping", "delivery", "arrived", "damaged"],
    "poor packaging": ["packaging", "package", "box", "wrapped"],
    "overheating": ["overheat", "hot", "warm", "temperature"],
    "feature requests": ["wish", "would like", "should include", "missing"],
    "refund complaints": ["refund", "money back", "return", "reimburse"],
    "value for money": ["value", "price", "worth", "expensive", "cheap"],
    "product durability": ["durable", "durability", "last", "quality", "broke"],
}
