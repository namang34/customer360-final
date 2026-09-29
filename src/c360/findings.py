"""Findings -- what a perception agent publishes to the state board."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SignalStrength(str, Enum):
    """How loudly one agent is shouting."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"

    @property
    def score(self) -> int:
        return {"weak": 1, "moderate": 2, "strong": 3}[self.value]


@dataclass(frozen=True)
class Finding:
    """One observation by one perception agent at one moment in simulated time."""

    agent: str
    as_of: datetime
    signal: str
    strength: SignalStrength
    event_ids: tuple[str, ...] = ()
    source_systems: tuple[str, ...] = ()
    detail: str = ""
    window_start: datetime | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("Finding.as_of must be timezone-aware")
        if isinstance(self.strength, str):
            object.__setattr__(self, "strength", SignalStrength(self.strength))
        object.__setattr__(self, "event_ids", tuple(self.event_ids))
        # Sorted + de-duplicated so the guardrail's "how many independent
        # systems?" count can never be inflated by the same system listed twice.
        object.__setattr__(self, "source_systems", tuple(sorted(set(self.source_systems))))

    @property
    def independent_source_count(self) -> int:
        return len(self.source_systems)

    def to_row(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "as_of": self.as_of.isoformat(),
            "signal": self.signal,
            "strength": self.strength.value,
            "event_ids": json.dumps(list(self.event_ids)),
            "source_systems": json.dumps(list(self.source_systems)),
            "detail": self.detail,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "metrics": json.dumps(self.metrics, default=str),
        }

    def __repr__(self) -> str:
        return (
            f"Finding({self.agent}/{self.signal} {self.strength.value} "
            f"@{self.as_of:%Y-%m-%d} <- {len(self.event_ids)} event(s))"
        )
