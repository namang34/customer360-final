"""The state board -- the shared per-customer surface the perception swarm writes to and
the Synthesis Agent reads from.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from .findings import Finding, SignalStrength
from .memory import EpisodicMemory
from .schema import Action, ConfidenceBand, HitlStatus, InferredState

# How far back two findings can sit and still count as "the same window". 14 days is
# chosen from the data, not plucked out of the air.
DEFAULT_WINDOW_DAYS = 30

# Perception agents, named once so a typo in an agent name cannot silently create
# a fifth "agent" that never corroborates anything.
PERCEPTION_AGENTS = ("transaction", "usage", "support", "life_signal")


@dataclass(frozen=True)
class Corroboration:
    """The evidence picture in one window. This is the guardrail's input."""

    as_of: datetime
    window_days: float
    findings: tuple[Finding, ...]
    source_systems: tuple[str, ...]
    agents: tuple[str, ...]
    event_ids: tuple[str, ...]

    @property
    def independent_source_count(self) -> int:
        return len(self.source_systems)

    @property
    def agent_count(self) -> int:
        return len(self.agents)

    @property
    def is_corroborated(self) -> bool:
        """THE GUARDRAIL PREDICATE."""
        return self.independent_source_count >= 2

    @property
    def strongest(self) -> SignalStrength | None:
        if not self.findings:
            return None
        return max((f.strength for f in self.findings), key=lambda s: s.score)

    def explain(self) -> str:
        if not self.findings:
            return "no findings in window"
        verdict = "CORROBORATED" if self.is_corroborated else "UNCORROBORATED"
        return (
            f"{verdict}: {self.independent_source_count} independent source system(s) "
            f"({', '.join(self.source_systems)}) across {self.agent_count} agent(s) "
            f"in the {self.window_days:.0f}-day window"
        )


@dataclass(frozen=True)
class BoardState:
    """A snapshot of what the system believes at one moment."""

    as_of: datetime
    inferred_state: InferredState
    confidence_band: ConfidenceBand
    action: Action
    action_subtype: str | None
    hitl_status: HitlStatus
    decided_at: datetime | None
    notes: str = ""

    @property
    def is_default(self) -> bool:
        """True if nothing has ever been decided -- the cold-start state."""
        return self.decided_at is None

    @property
    def age_days(self) -> float | None:
        if self.decided_at is None:
            return None
        return (self.as_of - self.decided_at).total_seconds() / 86400


class StateBoard:
    """Usage: board = StateBoard(memory) board.publish(finding)                      #
    perception agents write board.findings(now)                         # synthesis
    reads board.corroboration(now)                    # the guardrail reads
    board.current_state(now)                    # persisted belief
    """

    def __init__(self, memory: EpisodicMemory, window_days: float = DEFAULT_WINDOW_DAYS) -> None:
        self.memory = memory
        self.window_days = window_days

    # writing

    def publish(self, finding: Finding) -> Finding:
        if finding.agent not in PERCEPTION_AGENTS:
            raise ValueError(
                f"unknown agent {finding.agent!r}; expected one of {PERCEPTION_AGENTS}. "
                "A typo here would create an agent that never corroborates anything."
            )
        self.memory.record_finding(finding)
        return finding

    def publish_all(self, findings: Iterable[Finding]) -> list[Finding]:
        return [self.publish(f) for f in findings]

    # reading

    def findings(
        self,
        as_of: datetime,
        *,
        window_days: float | None = None,
        agents: Sequence[str] | None = None,
        signals: Sequence[str] | None = None,
    ) -> list[Finding]:
        """Findings inside the trailing window. `as_of` mandatory, as everywhere."""
        return self.memory.findings_as_of(
            as_of,
            since_days=self.window_days if window_days is None else window_days,
            agents=agents,
            signals=signals,
        )

    def corroboration(
        self,
        as_of: datetime,
        *,
        window_days: float | None = None,
        signals: Sequence[str] | None = None,
    ) -> Corroboration:
        """Assemble the evidence picture the guardrail will rule on."""
        window = self.window_days if window_days is None else window_days
        found = self.findings(as_of, window_days=window, signals=signals)

        source_systems: set[str] = set()
        agents: set[str] = set()
        event_ids: list[str] = []
        for finding in found:
            source_systems.update(finding.source_systems)
            agents.add(finding.agent)
            for event_id in finding.event_ids:
                if event_id not in event_ids:
                    event_ids.append(event_id)

        return Corroboration(
            as_of=as_of,
            window_days=window,
            findings=tuple(found),
            source_systems=tuple(sorted(source_systems)),
            agents=tuple(sorted(agents)),
            event_ids=tuple(event_ids),
        )

    def should_synthesise(self, as_of: datetime, *, window_days: float | None = None) -> bool:
        """THE AGENT-DEPENDENT TRIGGER."""
        found = self.findings(as_of, window_days=window_days)
        return len({f.agent for f in found}) >= 2

    def current_state(self, as_of: datetime) -> BoardState:
        """What the system believes right now, carried forward from the last decision."""
        last = self.memory.latest_decision_as_of(as_of)
        if last is None:
            return BoardState(
                as_of=as_of,
                inferred_state=InferredState.NO_SIGNIFICANT_EVENT,
                confidence_band=ConfidenceBand.LOW,
                action=Action.NO_ACTION,
                action_subtype=None,
                hitl_status=HitlStatus.AUTO_APPROVED,
                decided_at=None,
            )
        from .memory import _parse  # local import keeps the module surface small

        return BoardState(
            as_of=as_of,
            inferred_state=InferredState(last["inferred_state"]),
            confidence_band=ConfidenceBand(last["confidence_band"]),
            action=Action(last["action"]),
            action_subtype=last["action_subtype"],
            hitl_status=HitlStatus(last["hitl_status"]),
            decided_at=_parse(last["as_of_time"]),
            notes=last["notes"] or "",
        )

    def summary(self, as_of: datetime) -> str:
        state = self.current_state(as_of)
        corroboration = self.corroboration(as_of)
        signals = ", ".join(sorted({f"{f.agent}/{f.signal}" for f in corroboration.findings})) or "none"
        return (
            f"[{as_of:%Y-%m-%d}] believes={state.inferred_state.value}/"
            f"{state.confidence_band.value} | signals={signals} | {corroboration.explain()}"
        )
