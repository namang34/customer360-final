"""The guardrail -- a hard, code-level check on every proposed action."""

from __future__ import annotations

from dataclasses import dataclass, field

from .action import ActionProposal
from .schema import Action
from .state_board import Corroboration
from .synthesis import Synthesis

# Actions serious enough to require corroboration before they may fire.
GUARDED_ACTIONS = frozenset({
    Action.COMPLIANCE_FRAUD_HOLD,
    Action.RELATIONSHIP_MANAGER_ESCALATION,
    Action.PERSONALIZED_OFFER,
})

MIN_INDEPENDENT_SOURCES = 2

# An action must not rest on one event wearing two hats.
MIN_DISTINCT_EVENTS = 2


@dataclass
class GuardrailVerdict:
    passed: bool
    reason: str
    checks: dict[str, bool] = field(default_factory=dict)
    independent_sources: tuple[str, ...] = ()
    event_ids: tuple[str, ...] = ()

    def explain(self) -> str:
        return self.reason


def check(proposal: ActionProposal, synthesis: Synthesis) -> GuardrailVerdict:
    """Run the guardrail. Deterministic, cheap, and called on EVERY proposal."""
    corroboration: Corroboration = synthesis.corroboration
    sources = corroboration.source_systems
    events = corroboration.event_ids

    checks = {
        "action_is_guarded": proposal.action in GUARDED_ACTIONS,
        "independent_sources_ok": len(sources) >= MIN_INDEPENDENT_SOURCES,
        "distinct_events_ok": len(set(events)) >= MIN_DISTINCT_EVENTS,
    }

    if proposal.action is Action.NO_ACTION:
        return GuardrailVerdict(
            True, "no_action requires no corroboration", checks, sources, events
        )

    if proposal.action not in GUARDED_ACTIONS:
        return GuardrailVerdict(
            True,
            f"{proposal.action.value} is not a guarded action (low cost of being wrong); "
            f"corroboration was {len(sources)} source system(s)",
            checks,
            sources,
            events,
        )

    if not checks["independent_sources_ok"]:
        return GuardrailVerdict(
            False,
            f"BLOCKED: {proposal.action.value} requires >= {MIN_INDEPENDENT_SOURCES} independent "
            f"source systems but the evidence comes from {len(sources)} "
            f"({', '.join(sources) or 'none'}). This is the isolated-anomaly pattern.",
            checks,
            sources,
            events,
        )

    if not checks["distinct_events_ok"]:
        return GuardrailVerdict(
            False,
            f"BLOCKED: {proposal.action.value} rests on {len(set(events))} distinct event(s). "
            "A single event cannot corroborate itself, however many systems it touches.",
            checks,
            sources,
            events,
        )

    return GuardrailVerdict(
        True,
        f"PASSED: {len(sources)} independent source systems ({', '.join(sources)}) across "
        f"{len(set(events))} distinct events",
        checks,
        sources,
        events,
    )
