"""The perception swarm."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from ..findings import Finding
from .base import SIGNALS, PerceptionAgent, PerceptionContext
from .life_signal import LifeSignalAgent
from .support import SupportAgent
from .transaction import TransactionAgent
from .usage import UsageAgent

__all__ = [
    "SIGNALS",
    "LifeSignalAgent",
    "PerceptionAgent",
    "PerceptionContext",
    "SupportAgent",
    "TransactionAgent",
    "UsageAgent",
    "default_swarm",
    "run_swarm_on_event",
    "run_swarm_on_tick",
]


def default_swarm() -> list[PerceptionAgent]:
    return [TransactionAgent(), UsageAgent(), SupportAgent(), LifeSignalAgent()]


def run_swarm_on_event(
    agents: Sequence[PerceptionAgent], event, as_of: datetime, ctx: PerceptionContext
) -> list[Finding]:
    """Event-based trigger. Only agents that own the event's source system run."""
    findings: list[Finding] = []
    for agent in agents:
        if agent.handles(event):
            findings.extend(agent.on_event(event, as_of, ctx))
    return findings


def run_swarm_on_tick(
    agents: Sequence[PerceptionAgent], as_of: datetime, ctx: PerceptionContext
) -> list[Finding]:
    """Time-based trigger. Every agent gets a chance to notice an absence."""
    findings: list[Finding] = []
    for agent in agents:
        findings.extend(agent.on_tick(as_of, ctx))
    return findings
