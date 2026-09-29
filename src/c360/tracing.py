"""Tracing -- LangSmith when configured, a local JSONL trace always."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .pii import Redactor


def langsmith_enabled() -> bool:
    return bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))


def configure_langsmith(project: str = "customer360") -> bool:
    """Turn on LangChain's tracing if a key is present. Safe to call always."""
    if not langsmith_enabled():
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
    return True


class RunTrace:
    """One JSONL record per traced moment, flushed per line and redacted on the way out."""

    def __init__(
        self,
        path: str | Path | None,
        redactor: Redactor | None = None,
        tags: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.redactor = redactor
        # customer_id + scenario + as_of on every line, as the brief asks.
        self.tags = tags or {}
        self.entries: list[dict[str, Any]] = []
        self._handle = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", encoding="utf-8")

    def record(self, as_of: datetime, kind: str, **fields: Any) -> dict[str, Any]:
        entry = {
            "as_of": as_of.isoformat(),
            "kind": kind,
            **self.tags,
            **{k: _plain(v) for k, v in fields.items()},
        }
        if self.redactor is not None:
            entry = _scrub(entry, self.redactor)
        self.entries.append(entry)
        if self._handle:
            self._handle.write(json.dumps(entry, default=str) + "\n")
            self._handle.flush()  # flushed per line so a crashed run still leaves a usable trace
        return entry

    def close(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "RunTrace":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def decisions(self) -> list[dict[str, Any]]:
        return [e for e in self.entries if e.get("kind") == "checkpoint"]


def _plain(value: Any) -> Any:
    """Make anything JSON-serialisable without losing its shape."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value") and type(value).__name__ in (
        "Action", "InferredState", "ConfidenceBand", "HitlStatus", "SignalStrength"
    ):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _scrub(value: Any, redactor: Redactor) -> Any:
    if isinstance(value, str):
        return redactor.scrub(value)
    if isinstance(value, dict):
        return {k: _scrub(v, redactor) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, redactor) for v in value]
    return value
