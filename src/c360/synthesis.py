"""Synthesis Agent -- correlation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from . import prompts
from .findings import Finding, SignalStrength
from .llm import LLM, LLMUnavailable, NullLLM, complete_json
from .schema import ConfidenceBand, InferredState
from .state_board import Corroboration, StateBoard

# Signal -> state affinity. Weights are per (state, signal). A finding contributes
# strength.score * weight to that state's total.

STATE_AFFINITY: dict[InferredState, dict[str, float]] = {
    InferredState.MEDICAL_HARDSHIP: {
        "major_medical_expense": 4,
        "healthcare_spend": 3,
        "income_replacement": 2,
        "hardship_contact": 2,
        "savings_drawdown": 1,
    },
    InferredState.NEW_CHILD_LIFE_EVENT: {
        "dependents_increase": 5,
        "baby_retail_spend": 4,
        "new_recurring_commitment": 2,
        "reassurance_contact": 1,
        "income_disruption": 1,
        "social_life_event": 2,
    },
    InferredState.CHURN_RISK: {
        "cancellation_feature_used": 4,
        "self_transfer_external": 4,
        "salary_swept_out": 3,
        "standing_instruction_stopped": 3,
        "complaint_denied": 2,
        "complaint_open": 1,
        "engagement_drop": 2,
        "session_length_collapse": 1,
        "card_spend_collapse": 2,
    },
    InferredState.JOB_LOSS_OR_INCOME_DISRUPTION: {
        "income_disruption": 3,
        "income_replacement": 2,
        "savings_drawdown": 1,
    },
    InferredState.FINANCIAL_DISTRESS_GENERAL: {
        "savings_drawdown": 2,
        "income_disruption": 2,
        "hardship_contact": 2,
        "card_spend_collapse": 1,
    },
    InferredState.WEALTH_GROWTH_OR_WINDFALL: {
        "large_inbound_deposit": 3,
    },
    InferredState.MARRIAGE_OR_RELATIONSHIP_CHANGE: {
        "marital_status_change": 5,
        "social_life_event": 1,
    },
    InferredState.RELOCATION: {
        "address_change": 4,
        "social_life_event": 1,
    },
}

# search_intent and support themes carry their meaning in metrics rather than in
# the signal name, so they are routed separately.
INTENT_AFFINITY: dict[str, tuple[InferredState, float]] = {
    "financial_hardship": (InferredState.FINANCIAL_DISTRESS_GENERAL, 3),
    "medical": (InferredState.MEDICAL_HARDSHIP, 3),
    "child_planning": (InferredState.NEW_CHILD_LIFE_EVENT, 3),
    "leaving": (InferredState.CHURN_RISK, 4),
    "retirement": (InferredState.RETIREMENT_TRANSITION, 3),
    "home_purchase": (InferredState.RELOCATION, 2),
}
THEME_AFFINITY: dict[str, tuple[InferredState, float]] = {
    "medical_hardship": (InferredState.MEDICAL_HARDSHIP, 3),
    "financial_hardship": (InferredState.FINANCIAL_DISTRESS_GENERAL, 3),
    "new_child": (InferredState.NEW_CHILD_LIFE_EVENT, 2),
    "leaving_intent": (InferredState.CHURN_RISK, 4),
}
LIFE_EVENT_AFFINITY: dict[str, tuple[InferredState, float]] = {
    "new_child": (InferredState.NEW_CHILD_LIFE_EVENT, 3),
    "marriage": (InferredState.MARRIAGE_OR_RELATIONSHIP_CHANGE, 3),
    "job_loss": (InferredState.JOB_LOSS_OR_INCOME_DISRUPTION, 3),
    "job_change": (InferredState.JOB_CHANGE_OR_PROMOTION, 3),
    "relocation": (InferredState.RELOCATION, 3),
    "medical": (InferredState.MEDICAL_HARDSHIP, 2),
    "retirement": (InferredState.RETIREMENT_TRANSITION, 3),
}

# Confidence thresholds CALIBRATED AGAINST THE THREE PRACTICE SCENARIOS.
HIGH_MIN_STRONG = 2
HIGH_MIN_SOURCES = 3
MEDIUM_MIN_STRONG = 2
MEDIUM_MIN_SOURCES = 2

# DECISIVE signals: the customer has DONE something deliberate, as opposed to a spending
# pattern or a rate that we inferred about them.
DECISIVE_SIGNALS = frozenset({
    "cancellation_feature_used",
    "dependents_increase",
    "marital_status_change",
    "address_change",
    "loan_application",
    "hardship_contact",
    "self_transfer_external",
    "salary_swept_out",
    "new_recurring_commitment",
    "search_intent",
})

# Age of the last RECORDED DECISION past which a carried-forward belief steps down one
# band. The pipeline records a decision at every checkpoint, so on a normal run this
# age never grows and the step-down does not fire; it is reachable only when decisions
# are written less often than they are read.
CONFIDENCE_DECAY_DAYS = 60


@dataclass
class Synthesis:
    """What the correlation layer concluded, and why."""

    as_of: datetime
    inferred_state: InferredState
    confidence_band: ConfidenceBand
    rationale: str
    event_ids: tuple[str, ...]
    corroboration: Corroboration
    affinity: dict[str, float] = field(default_factory=dict)
    strong_signals: tuple[str, ...] = ()
    decided_by: str = "rules"
    carried_forward: bool = False

    @property
    def is_significant(self) -> bool:
        return self.inferred_state is not InferredState.NO_SIGNIFICANT_EVENT


class SynthesisAgent:
    """Correlates the board's findings into one state and one confidence band."""

    def __init__(self, llm: LLM | None = None, window_days: float | None = None) -> None:
        self.llm = llm or NullLLM()
        self.window_days = window_days
        self.llm_failures = 0
        self.llm_uses = 0

    def synthesise(self, as_of: datetime, board: StateBoard) -> Synthesis:
        corroboration = board.corroboration(as_of, window_days=self.window_days)
        findings = list(corroboration.findings)

        if not findings:
            return self._carry_forward(as_of, board, corroboration)

        affinity = score_states(findings)
        ranked = sorted(affinity.items(), key=lambda kv: -kv[1])
        if not ranked or ranked[0][1] <= 0:
            return self._carry_forward(as_of, board, corroboration)

        previous_state = board.current_state(as_of)
        ranked = _apply_hysteresis(ranked, previous_state)
        state, rationale, decided_by = self._choose(ranked, findings, corroboration)

        # Confidence and corroboration are judged ONLY on evidence that actually
        # supports the chosen state. This matters more than it looks.
        supporting = [f for f in findings if _supports(state, f)]
        corroboration = _restrict(corroboration, supporting)
        band = confidence_for(supporting, corroboration)

        # Monotonic belief: a state already held at HIGH does not silently slip back to
        # LOW because one quiet fortnight thinned the window.
        previous = previous_state
        if previous.inferred_state is state and _band_rank(previous.confidence_band) > _band_rank(band):
            band = previous.confidence_band
            rationale = f"{rationale} Confidence held from the earlier assessment on {previous.decided_at:%d %b}."

        return Synthesis(
            as_of=as_of,
            inferred_state=state,
            confidence_band=band,
            rationale=rationale,
            event_ids=corroboration.event_ids,
            corroboration=corroboration,
            affinity={s.value: round(v, 2) for s, v in ranked[:4]},
            strong_signals=tuple(
                sorted({f.signal for f in findings if f.strength is SignalStrength.STRONG})
            ),
            decided_by=decided_by,
        )

    # state selection

    def _choose(self, ranked, findings, corroboration) -> tuple[InferredState, str, str]:
        top_state, top_score = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None

        # The LLM is only consulted when the top two are genuinely close.
        contested = runner_up is not None and runner_up[1] >= top_score * 0.75
        if contested and not isinstance(self.llm, NullLLM):
            chosen = self._ask_llm([r[0] for r in ranked[:3]], findings)
            if chosen is not None:
                state, rationale = chosen
                self.llm_uses += 1
                return state, rationale, f"llm:{getattr(self.llm, 'model', '?')}"

        return top_state, _rule_rationale(top_state, findings, corroboration), "rules"

    def _ask_llm(self, candidates, findings) -> tuple[InferredState, str] | None:
        evidence = "\n".join(
            f"- [{f.agent}] {f.signal} ({f.strength.value}): {f.detail}" for f in findings
        )
        user = (
            f"Candidate states: {', '.join(c.value for c in candidates)}\n\n"
            f"Findings from independent detectors:\n{evidence}"
        )
        try:
            result = complete_json(self.llm, prompts.SYNTHESIS, user, role="synthesis")
            raw = str(result.get("inferred_state", "")).strip().lower()
            state = next((c for c in candidates if c.value == raw), None)
            if state is None:
                return None
            return state, str(result.get("rationale", ""))[:220]
        except (LLMUnavailable, ValueError, KeyError, TypeError):
            self.llm_failures += 1
            return None


    def _carry_forward(self, as_of, board, corroboration) -> Synthesis:
        """No findings in the window -- keep believing what we believed."""
        previous = board.current_state(as_of)
        band = previous.confidence_band
        rationale = "No new signals; carrying forward the previous assessment."

        if previous.age_days is not None and previous.age_days > CONFIDENCE_DECAY_DAYS:
            band = _step_down(band)
            rationale = (
                f"No supporting signal for {previous.age_days:.0f} days; confidence stepped down."
            )

        return Synthesis(
            as_of=as_of,
            inferred_state=previous.inferred_state,
            confidence_band=band,
            rationale=rationale,
            event_ids=(),
            corroboration=corroboration,
            carried_forward=True,
        )


def score_states(findings: list[Finding]) -> dict[InferredState, float]:
    """Weighted affinity of each candidate state, given the findings in the window."""
    best_strength: dict[str, SignalStrength] = {}
    for finding in findings:
        current = best_strength.get(finding.signal)
        if current is None or finding.strength.score > current.score:
            best_strength[finding.signal] = finding.strength

    scores: dict[InferredState, float] = {}

    for state, weights in STATE_AFFINITY.items():
        total = sum(
            best_strength[signal].score * weight
            for signal, weight in weights.items()
            if signal in best_strength
        )
        if total:
            scores[state] = scores.get(state, 0.0) + total

    # Metric-carried signals: the same signal name means different things
    # depending on what was searched for or what a ticket was about.
    for finding in findings:
        routed: tuple[InferredState, float] | None = None
        if finding.signal == "search_intent":
            routed = INTENT_AFFINITY.get(str(finding.metrics.get("intent", "")))
        elif finding.signal in ("hardship_contact", "reassurance_contact", "complaint_open"):
            routed = THEME_AFFINITY.get(str(finding.metrics.get("theme", "")))
        elif finding.signal == "social_life_event":
            routed = LIFE_EVENT_AFFINITY.get(str(finding.metrics.get("life_event", "")))
        if routed:
            state, weight = routed
            scores[state] = scores.get(state, 0.0) + finding.strength.score * weight

    return scores


def confidence_for(findings: list[Finding], corroboration: Corroboration) -> ConfidenceBand:
    """The confidence band. The thresholds it uses are the block at the top of this
    module, calibrated against the three practice scenarios -- see docs/EVALUATION.md
    for what that does and does not prove.
    """
    strong = {f.signal for f in findings if f.strength is SignalStrength.STRONG}
    sources = corroboration.independent_source_count
    decisive = bool(strong & DECISIVE_SIGNALS)

    if len(strong) >= HIGH_MIN_STRONG:
        # Two routes to high confidence, and they are alternatives not additions: breadth
        # -- three independent source systems agree, or intent   -- two systems agree AND
        # one of the strong signals is the customer doing something deliberate.
        if sources >= HIGH_MIN_SOURCES or (sources >= MEDIUM_MIN_SOURCES and decisive):
            return ConfidenceBand.HIGH
    if len(strong) >= MEDIUM_MIN_STRONG and sources >= MEDIUM_MIN_SOURCES:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.LOW


def _rule_rationale(state: InferredState, findings: list[Finding], corroboration) -> str:
    """A one-sentence explanation naming real signals and real event ids."""
    relevant = [f for f in findings if _supports(state, f)]
    relevant.sort(key=lambda f: -f.strength.score)
    # Distinct SIGNALS, not findings. A tick-based detector republishes the same signal
    # every day it still holds, so taking the top three findings would name one signal
    # three times -- and the notes field is the graded explanation.
    seen: list[str] = []
    for f in relevant:
        label = f.signal.replace("_", " ")
        if label not in seen:
            seen.append(label)
        if len(seen) == 3:
            break
    named = ", ".join(seen) or "weak indicators"
    ids = ", ".join(corroboration.event_ids[:4])
    return (
        f"{state.value.replace('_', ' ').capitalize()} indicated by {named} across "
        f"{corroboration.independent_source_count} independent source system(s) "
        f"({', '.join(corroboration.source_systems)}). Key events: {ids}."
    )


# How much better a challenger must score to displace the state we already hold.
HYSTERESIS_MARGIN = {
    ConfidenceBand.LOW: 1.0,      # not committed -- follow the evidence freely
    ConfidenceBand.MEDIUM: 1.25,
    ConfidenceBand.HIGH: 1.5,
}


def _apply_hysteresis(ranked, previous) -> list:
    """Make an established belief sticky in proportion to how well-evidenced it was."""
    if previous.is_default or not ranked:
        return ranked
    incumbent = previous.inferred_state
    margin = HYSTERESIS_MARGIN[previous.confidence_band]
    if margin <= 1.0:
        return ranked

    incumbent_score = next((score for state, score in ranked if state is incumbent), 0.0)
    top_state, top_score = ranked[0]
    if top_state is incumbent or incumbent_score <= 0:
        return ranked
    if top_score >= incumbent_score * margin:
        return ranked
    # Challenger did not clear the bar -- keep the incumbent at the front.
    return [(incumbent, incumbent_score)] + [r for r in ranked if r[0] is not incumbent]


def _restrict(corroboration: Corroboration, findings: list[Finding]) -> Corroboration:
    """Rebuild a Corroboration over just the findings that support the chosen state."""
    source_systems: set[str] = set()
    agents: set[str] = set()
    event_ids: list[str] = []
    for finding in findings:
        source_systems.update(finding.source_systems)
        agents.add(finding.agent)
        for event_id in finding.event_ids:
            if event_id not in event_ids:
                event_ids.append(event_id)
    return Corroboration(
        as_of=corroboration.as_of,
        window_days=corroboration.window_days,
        findings=tuple(findings),
        source_systems=tuple(sorted(source_systems)),
        agents=tuple(sorted(agents)),
        event_ids=tuple(event_ids),
    )


def _supports(state: InferredState, finding: Finding) -> bool:
    if finding.signal in STATE_AFFINITY.get(state, {}):
        return True
    for table, key in (
        (INTENT_AFFINITY, "intent"),
        (THEME_AFFINITY, "theme"),
        (LIFE_EVENT_AFFINITY, "life_event"),
    ):
        routed = table.get(str(finding.metrics.get(key, "")))
        if routed and routed[0] is state:
            return True
    return False


_BAND_ORDER = {ConfidenceBand.LOW: 0, ConfidenceBand.MEDIUM: 1, ConfidenceBand.HIGH: 2}


def _band_rank(band: ConfidenceBand) -> int:
    return _BAND_ORDER[band]


def _step_down(band: ConfidenceBand) -> ConfidenceBand:
    if band is ConfidenceBand.HIGH:
        return ConfidenceBand.MEDIUM
    if band is ConfidenceBand.MEDIUM:
        return ConfidenceBand.LOW
    return ConfidenceBand.LOW
