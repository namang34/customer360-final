"""PII redaction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Fields in an event payload whose VALUES are free text that may name people.
FREE_TEXT_PAYLOAD_FIELDS = ("raw_text", "search_text", "counterparty_name", "merchant_name")

# Generic patterns worth masking wherever they appear, independent of the customer's own
# identity. ORDER MATTERS, and it is the opposite of the obvious one.
GENERIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("<EMAIL>", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("<SSN>", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("<CARD_NUMBER>", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("<PHONE>", re.compile(r"\b(?:\+?\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}\b")),
]


@dataclass
class Redactor:
    """Built once per customer from entities.json, then reused everywhere."""

    customer_id: str | None = None
    name: str | None = None
    account_ids: tuple[str, ...] = ()

    @classmethod
    def from_entities(cls, entities: dict[str, Any]) -> "Redactor":
        profile = entities.get("profile") or {}
        accounts = entities.get("accounts") or []
        return cls(
            customer_id=entities.get("customer_id"),
            name=profile.get("name"),
            account_ids=tuple(a.get("account_id") for a in accounts if a.get("account_id")),
        )


    def _name_patterns(self) -> list[tuple[str, re.Pattern[str]]]:
        """Match the full name and each of its parts, longest first."""
        if not self.name:
            return []
        parts = [p for p in self.name.split() if len(p) > 1]
        out = [("<CUSTOMER_NAME>", re.compile(re.escape(self.name), re.IGNORECASE))]
        out += [
            ("<CUSTOMER_NAME>", re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE))
            for p in sorted(parts, key=len, reverse=True)
        ]
        return out

    def scrub(self, text: str) -> str:
        """Redact a single string. Safe to call on anything, including None-ish."""
        if not text or not isinstance(text, str):
            return text
        result = text
        # Structured identifiers first -- see the note on GENERIC_PATTERNS.
        for token, pattern in GENERIC_PATTERNS:
            result = pattern.sub(token, result)
        for token, pattern in self._name_patterns():
            result = pattern.sub(token, result)
        return result

    def scrub_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a redacted COPY of an event payload."""
        scrubbed = dict(payload)
        for field in FREE_TEXT_PAYLOAD_FIELDS:
            if field in scrubbed and isinstance(scrubbed[field], str):
                scrubbed[field] = self.scrub(scrubbed[field])
        return scrubbed

    def counterparty_is_customer(self, payload: dict[str, Any]) -> bool:
        """Did the customer send money to THEMSELVES at another institution?"""
        raw = payload.get("counterparty_name")
        if not raw or not isinstance(raw, str):
            return False
        return "<CUSTOMER_NAME>" in self.scrub(raw)

    def contains_pii(self, text: str) -> list[str]:
        """Report any un-redacted PII found in `text`. Used by the output writer to check
        its own work before anything reaches disk.
        """
        if not text or not isinstance(text, str):
            return []
        found: list[str] = []
        for token, pattern in self._name_patterns():
            if pattern.search(text):
                found.append(f"customer name -> {token}")
                break
        for token, pattern in GENERIC_PATTERNS:
            if pattern.search(text):
                found.append(f"{token} pattern")
        return found
