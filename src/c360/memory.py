"""Episodic memory -- the per-customer SQLite store."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .findings import Finding, SignalStrength
from .output import Checkpoint
from .schema import Event

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    event_time      TEXT NOT NULL,
    ingestion_time  TEXT NOT NULL,
    release_time    TEXT NOT NULL,
    customer_id     TEXT NOT NULL,
    account_id      TEXT,
    source_system   TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload         TEXT NOT NULL
);
-- The composite index matches the visibility predicate exactly, so the filter
-- that protects the hard rule is also the fast path.
CREATE INDEX IF NOT EXISTS idx_events_visibility
    ON events (customer_id, event_time, release_time);
CREATE INDEX IF NOT EXISTS idx_events_source
    ON events (customer_id, source_system, event_time);

CREATE TABLE IF NOT EXISTS findings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     TEXT NOT NULL,
    agent           TEXT NOT NULL,
    as_of           TEXT NOT NULL,
    signal          TEXT NOT NULL,
    strength        TEXT NOT NULL,
    event_ids       TEXT NOT NULL,
    source_systems  TEXT NOT NULL,
    detail          TEXT,
    window_start    TEXT,
    metrics         TEXT
);
CREATE INDEX IF NOT EXISTS idx_findings_time ON findings (customer_id, as_of);

CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     TEXT NOT NULL,
    as_of_time      TEXT NOT NULL,
    inferred_state  TEXT NOT NULL,
    confidence_band TEXT NOT NULL,
    action          TEXT NOT NULL,
    action_subtype  TEXT,
    hitl_status     TEXT NOT NULL,
    notes           TEXT,
    citations       TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_time ON decisions (customer_id, as_of_time);
"""


class TemporalLeakError(RuntimeError):
    """Raised when a query would expose information from the future."""


def _iso(moment: datetime) -> str:
    """Serialise a datetime for storage."""
    if moment.tzinfo is None:
        raise ValueError("refusing to store a naive datetime")
    return moment.astimezone(timezone.utc).isoformat()


def _parse(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def _require_as_of(as_of: Any) -> datetime:
    if not isinstance(as_of, datetime):
        raise TypeError(f"as_of must be a datetime, got {type(as_of).__name__}")
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware -- a naive 'now' is a leak waiting to happen")
    return as_of.astimezone(timezone.utc)


class EpisodicMemory:
    """Per-customer event, finding and decision history."""

    def __init__(self, path: str | Path = ":memory:", customer_id: str | None = None) -> None:
        self.path = str(path)
        self.customer_id = customer_id
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "EpisodicMemory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # Writing

    def record_event(self, event: Event) -> None:
        self.record_events([event])

    def record_events(self, events: Iterable[Event]) -> int:
        """Store events. Re-recording the same event_id is a harmless no-op."""
        rows = [
            (
                e.event_id,
                _iso(e.event_time),
                _iso(e.ingestion_time),
                _iso(e.release_time),
                e.customer_id,
                e.account_id,
                e.source_system,
                e.event_type,
                json.dumps(e.payload),
            )
            for e in events
        ]
        if not rows:
            return 0
        cursor = self.conn.executemany(
            "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        self.conn.commit()
        return cursor.rowcount

    def record_finding(self, finding: Finding) -> None:
        row = finding.to_row()
        self.conn.execute(
            "INSERT INTO findings (customer_id, agent, as_of, signal, strength, event_ids, "
            "source_systems, detail, window_start, metrics) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                self.customer_id,
                row["agent"],
                row["as_of"],
                row["signal"],
                row["strength"],
                row["event_ids"],
                row["source_systems"],
                row["detail"],
                row["window_start"],
                row["metrics"],
            ),
        )
        self.conn.commit()

    def record_findings(self, findings: Iterable[Finding]) -> None:
        for finding in findings:
            self.record_finding(finding)

    def record_decision(self, checkpoint: Checkpoint) -> None:
        self.conn.execute(
            "INSERT INTO decisions (customer_id, as_of_time, inferred_state, confidence_band, "
            "action, action_subtype, hitl_status, notes, citations) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                self.customer_id,
                _iso(checkpoint.as_of_time),
                checkpoint.inferred_state.value,
                checkpoint.confidence_band.value,
                checkpoint.action.value,
                checkpoint.action_subtype,
                checkpoint.hitl_status.value,
                checkpoint.notes,
                json.dumps(list(checkpoint.citations)),
            ),
        )
        self.conn.commit()

    # Reading -- every query below takes as_of FIRST and REQUIRED (stats() is not a query)

    def events_as_of(
        self,
        as_of: datetime,
        *,
        since_days: float | None = None,
        source_systems: Sequence[str] | None = None,
        event_types: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Every event visible at `as_of`, oldest first."""
        as_of = _require_as_of(as_of)
        clauses = ["event_time <= :as_of", "release_time <= :as_of"]
        params: dict[str, Any] = {"as_of": _iso(as_of)}

        if self.customer_id:
            clauses.append("customer_id = :customer_id")
            params["customer_id"] = self.customer_id

        if since_days is not None:
            # A lookback window, not an escape hatch: it can only ever narrow the
            # set, never extend it past as_of.
            clauses.append("event_time >= :since")
            params["since"] = _iso(as_of - timedelta(days=since_days))

        if source_systems:
            placeholders = ",".join(f":ss{i}" for i in range(len(source_systems)))
            clauses.append(f"source_system IN ({placeholders})")
            params.update({f"ss{i}": s for i, s in enumerate(source_systems)})

        if event_types:
            placeholders = ",".join(f":et{i}" for i in range(len(event_types)))
            clauses.append(f"event_type IN ({placeholders})")
            params.update({f"et{i}": t for i, t in enumerate(event_types)})

        sql = f"SELECT * FROM events WHERE {' AND '.join(clauses)} ORDER BY event_time, event_id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        rows = self.conn.execute(sql, params).fetchall()
        events = [self._row_to_event(r) for r in rows]

        # Belt and braces.
        for event in events:
            if event.event_time > as_of or event.release_time > as_of:
                raise TemporalLeakError(
                    f"{event.event_id} leaked: event_time={event.event_time.isoformat()} "
                    f"release_time={event.release_time.isoformat()} as_of={as_of.isoformat()}"
                )
        return events

    def count_as_of(self, as_of: datetime, **kwargs: Any) -> int:
        return len(self.events_as_of(as_of, **kwargs))

    def findings_as_of(
        self,
        as_of: datetime,
        *,
        since_days: float | None = None,
        agents: Sequence[str] | None = None,
        signals: Sequence[str] | None = None,
    ) -> list[Finding]:
        """Findings published at or before `as_of`."""
        as_of = _require_as_of(as_of)
        clauses = ["as_of <= :as_of"]
        params: dict[str, Any] = {"as_of": _iso(as_of)}

        if self.customer_id:
            clauses.append("customer_id = :customer_id")
            params["customer_id"] = self.customer_id
        if since_days is not None:
            clauses.append("as_of >= :since")
            params["since"] = _iso(as_of - timedelta(days=since_days))
        if agents:
            placeholders = ",".join(f":a{i}" for i in range(len(agents)))
            clauses.append(f"agent IN ({placeholders})")
            params.update({f"a{i}": a for i, a in enumerate(agents)})
        if signals:
            placeholders = ",".join(f":s{i}" for i in range(len(signals)))
            clauses.append(f"signal IN ({placeholders})")
            params.update({f"s{i}": s for i, s in enumerate(signals)})

        rows = self.conn.execute(
            f"SELECT * FROM findings WHERE {' AND '.join(clauses)} ORDER BY as_of, id", params
        ).fetchall()
        return [self._row_to_finding(r) for r in rows]

    def decisions_as_of(self, as_of: datetime) -> list[dict[str, Any]]:
        as_of = _require_as_of(as_of)
        clauses = ["as_of_time <= :as_of"]
        params: dict[str, Any] = {"as_of": _iso(as_of)}
        if self.customer_id:
            clauses.append("customer_id = :customer_id")
            params["customer_id"] = self.customer_id
        rows = self.conn.execute(
            f"SELECT * FROM decisions WHERE {' AND '.join(clauses)} ORDER BY as_of_time, id", params
        ).fetchall()
        return [dict(r) for r in rows]

    def latest_decision_as_of(self, as_of: datetime) -> dict[str, Any] | None:
        """The most recent committed decision at or before `as_of`."""
        decisions = self.decisions_as_of(as_of)
        return decisions[-1] if decisions else None

    # Derived views the perception agents lean on

    def source_systems_active(
        self, as_of: datetime, *, since_days: float = 14
    ) -> dict[str, int]:
        """How many events per source system in the trailing window."""
        counts: dict[str, int] = {}
        for event in self.events_as_of(as_of, since_days=since_days):
            counts[event.source_system] = counts.get(event.source_system, 0) + 1
        return counts

    def daily_rate(
        self,
        as_of: datetime,
        *,
        source_systems: Sequence[str] | None = None,
        event_types: Sequence[str] | None = None,
        window_days: float = 14,
    ) -> float:
        """Events per day over the trailing window."""
        events = self.events_as_of(
            as_of,
            since_days=window_days,
            source_systems=source_systems,
            event_types=event_types,
        )
        return len(events) / window_days if window_days else 0.0

    def baseline_daily_rate(
        self,
        as_of: datetime,
        *,
        source_systems: Sequence[str] | None = None,
        event_types: Sequence[str] | None = None,
        baseline_days: float = 90,
        exclude_recent_days: float = 14,
    ) -> float:
        """The customer's OWN normal rate, measured before the recent window."""
        as_of = _require_as_of(as_of)
        cutoff = as_of - timedelta(days=exclude_recent_days)
        window = baseline_days - exclude_recent_days
        if window <= 0:
            return 0.0
        events = [
            e
            for e in self.events_as_of(
                cutoff, since_days=window, source_systems=source_systems, event_types=event_types
            )
        ]
        return len(events) / window


    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        return Event(
            event_id=row["event_id"],
            event_time=_parse(row["event_time"]),
            ingestion_time=_parse(row["ingestion_time"]),
            customer_id=row["customer_id"],
            account_id=row["account_id"],
            source_system=row["source_system"],
            event_type=row["event_type"],
            schema_version="1.0",
            payload=json.loads(row["payload"]),
        )

    @staticmethod
    def _row_to_finding(row: sqlite3.Row) -> Finding:
        return Finding(
            agent=row["agent"],
            as_of=_parse(row["as_of"]),
            signal=row["signal"],
            strength=SignalStrength(row["strength"]),
            event_ids=tuple(json.loads(row["event_ids"])),
            source_systems=tuple(json.loads(row["source_systems"])),
            detail=row["detail"] or "",
            window_start=_parse(row["window_start"]) if row["window_start"] else None,
            metrics=json.loads(row["metrics"]) if row["metrics"] else {},
        )

    def stats(self) -> dict[str, int]:
        return {
            "events": self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "findings": self.conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0],
            "decisions": self.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0],
        }
