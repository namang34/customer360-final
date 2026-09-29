"""Event schema, parsing, and the fixed output enums."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Output enums -- fixed by README_dataset_schema.md. Do not invent values.
# ---------------------------------------------------------------------------

class InferredState(str, Enum):
    NO_SIGNIFICANT_EVENT = "no_significant_event"
    NEW_CHILD_LIFE_EVENT = "new_child_life_event"
    MARRIAGE_OR_RELATIONSHIP_CHANGE = "marriage_or_relationship_change"
    JOB_CHANGE_OR_PROMOTION = "job_change_or_promotion"
    JOB_LOSS_OR_INCOME_DISRUPTION = "job_loss_or_income_disruption"
    MEDICAL_HARDSHIP = "medical_hardship"
    FINANCIAL_DISTRESS_GENERAL = "financial_distress_general"
    RELOCATION = "relocation"
    RETIREMENT_TRANSITION = "retirement_transition"
    WEALTH_GROWTH_OR_WINDFALL = "wealth_growth_or_windfall"
    POTENTIAL_FRAUD_OR_TAKEOVER = "potential_fraud_or_takeover"
    ELDER_VULNERABILITY_OR_SCAM_RISK = "elder_vulnerability_or_scam_risk"
    CHURN_RISK = "churn_risk"
    SMALL_BUSINESS_CASHFLOW_EVENT = "small_business_cashflow_event"


class Action(str, Enum):
    NO_ACTION = "no_action"
    PROACTIVE_RETENTION_OUTREACH = "proactive_retention_outreach"
    RELATIONSHIP_MANAGER_ESCALATION = "relationship_manager_escalation"
    PERSONALIZED_OFFER = "personalized_offer"
    SUPPORT_INTERVENTION = "support_intervention"
    COMPLIANCE_FRAUD_HOLD = "compliance_fraud_hold"


class ConfidenceBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class HitlStatus(str, Enum):
    AUTO_APPROVED = "auto_approved"
    ESCALATED = "escalated"
    HUMAN_APPROVED = "human_approved"
    HUMAN_REJECTED = "human_rejected"
    HUMAN_MODIFIED = "human_modified"


# The nine source systems in the dataset. The README's payload table has eight rows only
# because instant_payments and ach_wire share a payload shape.
KNOWN_SOURCE_SYSTEMS = frozenset({
    "card_payments",
    "instant_payments",
    "ach_wire",
    "core_banking_ledger",
    "trading_brokerage",
    "loan_kyc",
    "web_app_events",
    "support_logs",
    "social_signal_consented",
})

# Which perception agent owns which source system. Defined here so the routing
# table lives next to the schema it routes on, and so a source system can never
# be silently dropped: anything not listed maps to None and gets logged.
PERCEPTION_ROUTING = {
    "card_payments": "transaction",
    "instant_payments": "transaction",
    "ach_wire": "transaction",
    "core_banking_ledger": "transaction",
    "trading_brokerage": "transaction",
    "web_app_events": "usage",
    "support_logs": "support",
    "loan_kyc": "life_signal",
    "social_signal_consented": "life_signal",
}


class EventParseError(ValueError):
    """Raised for a row that cannot be turned into a usable Event."""


def parse_timestamp(value: str, field_name: str) -> datetime:
    """Parse an ISO-8601 timestamp into a timezone-AWARE UTC datetime."""
    if not isinstance(value, str):
        raise EventParseError(f"{field_name} is not a string: {value!r}")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise EventParseError(f"{field_name} is not valid ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class Event:
    """One event, parsed and immutable."""

    event_id: str
    event_time: datetime
    ingestion_time: datetime
    customer_id: str
    account_id: str | None
    source_system: str
    event_type: str
    schema_version: str
    payload: dict[str, Any] = field(default_factory=dict)

    # ---- derived properties -------------------------------------------------

    @property
    def release_time(self) -> datetime:
        """The moment this event becomes VISIBLE to the system."""
        return max(self.ingestion_time, self.event_time)

    @property
    def is_late_arrival(self) -> bool:
        """True if this event reached the system after it occurred."""
        return self.ingestion_time > self.event_time

    @property
    def arrival_lag_seconds(self) -> float:
        return (self.ingestion_time - self.event_time).total_seconds()

    @property
    def perception_agent(self) -> str | None:
        """Which perception agent should see this event, or None if unroutable."""
        return PERCEPTION_ROUTING.get(self.source_system)

    @property
    def has_known_source_system(self) -> bool:
        return self.source_system in KNOWN_SOURCE_SYSTEMS

    def __repr__(self) -> str:  # keeps debug output readable
        return (
            f"Event({self.event_id} {self.event_time:%Y-%m-%d %H:%M} "
            f"{self.source_system}/{self.event_type})"
        )


REQUIRED_FIELDS = ("event_id", "event_time", "customer_id", "source_system", "event_type")


def event_from_dict(row: dict[str, Any]) -> Event:
    """Turn one decoded JSON object into an Event, or raise EventParseError."""
    if not isinstance(row, dict):
        raise EventParseError(f"row is not a JSON object: {type(row).__name__}")

    missing = [f for f in REQUIRED_FIELDS if row.get(f) in (None, "")]
    if missing:
        raise EventParseError(f"missing required field(s): {', '.join(missing)}")

    event_time = parse_timestamp(row["event_time"], "event_time")

    # ingestion_time is optional in the envelope. If it is absent we assume the
    # event was received the instant it happened -- the least-surprising default,
    # and one that can never make an event visible EARLIER than it should be.
    raw_ingestion = row.get("ingestion_time")
    ingestion_time = (
        parse_timestamp(raw_ingestion, "ingestion_time") if raw_ingestion else event_time
    )

    payload = row.get("payload") or {}
    if not isinstance(payload, dict):
        raise EventParseError(f"payload is not an object: {type(payload).__name__}")

    return Event(
        event_id=str(row["event_id"]),
        event_time=event_time,
        ingestion_time=ingestion_time,
        customer_id=str(row["customer_id"]),
        account_id=row.get("account_id"),
        source_system=str(row["source_system"]),
        event_type=str(row["event_type"]),
        schema_version=str(row.get("schema_version", "1.0")),
        payload=payload,
    )


@dataclass
class LoadReport:
    """What happened while reading a .jsonl file. Surfaced, never swallowed."""

    path: Path
    events: list[Event] = field(default_factory=list)
    rejected: list[tuple[int, str, str]] = field(default_factory=list)  # (lineno, reason, raw)
    unknown_source_systems: set[str] = field(default_factory=set)
    late_arrivals: int = 0
    duplicate_event_ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected

    def summary(self) -> str:
        bits = [f"{len(self.events)} events from {self.path.name}"]
        if self.rejected:
            bits.append(f"{len(self.rejected)} REJECTED")
        if self.duplicate_event_ids:
            bits.append(f"{len(self.duplicate_event_ids)} duplicate event_id(s)")
        if self.late_arrivals:
            bits.append(f"{self.late_arrivals} late arrival(s)")
        if self.unknown_source_systems:
            bits.append(f"unknown source_system(s): {sorted(self.unknown_source_systems)}")
        return ", ".join(bits)


def iter_jsonl(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (1-based line number, raw line) for each non-blank line."""
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped:
                yield lineno, stripped


def load_events(path: Path, strict: bool = False) -> LoadReport:
    """Read a .jsonl event file into a LoadReport."""
    path = Path(path)
    report = LoadReport(path=path)
    seen_ids: set[str] = set()

    if not path.exists():
        # A missing file is reported, not raised (unless strict).
        if strict:
            raise FileNotFoundError(path)
        report.rejected.append((0, f"file not found: {path}", ""))
        return report

    for lineno, raw in iter_jsonl(path):
        try:
            row = json.loads(raw)
            event = event_from_dict(row)
        except (json.JSONDecodeError, EventParseError) as exc:
            if strict:
                raise EventParseError(f"{path.name}:{lineno}: {exc}") from exc
            report.rejected.append((lineno, str(exc), raw[:200]))
            continue

        if event.event_id in seen_ids:
            # Duplicates are recorded but kept -- dropping one could remove a
            # signal event. Deduplication, if needed, belongs in episodic memory
            # where there is an explicit primary key, not silently at load time.
            report.duplicate_event_ids.append(event.event_id)
        seen_ids.add(event.event_id)

        if not event.has_known_source_system:
            report.unknown_source_systems.add(event.source_system)
        if event.is_late_arrival:
            report.late_arrivals += 1

        report.events.append(event)

    return report


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
