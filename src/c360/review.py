"""The ambiguity review queue -- escalation for uncertainty rather than for cost."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .action import ActionProposal
from .schema import Action, ConfidenceBand, InferredState
from .synthesis import Synthesis

# Corroboration breadth at which weak evidence becomes worth a human's attention.
MIN_SOURCES_FOR_REVIEW = 2

# How close the top two candidate states must be before the choice between them is
# treated as unresolved rather than decided.
CONTESTED_MARGIN = 0.25


@dataclass(frozen=True)
class ReviewItem:
    """One checkpoint worth a human glance, despite no action being taken."""

    as_of: datetime
    reason: str                      # weak_but_corroborated | contested
    inferred_state: InferredState
    confidence_band: ConfidenceBand
    source_systems: tuple[str, ...]
    event_ids: tuple[str, ...]
    detail: str

    def to_row(self) -> dict:
        return {
            "as_of_time": self.as_of.isoformat().replace("+00:00", "Z"),
            "reason": self.reason,
            "inferred_state": self.inferred_state.value,
            "confidence_band": self.confidence_band.value,
            "source_systems": list(self.source_systems),
            "citations": list(self.event_ids),
            "detail": self.detail,
        }


def assess(synthesis: Synthesis, proposal: ActionProposal) -> ReviewItem | None:
    """Decide whether this checkpoint deserves a human look. Returns None for the
    overwhelming majority of them.
    """
    if proposal.action is not Action.NO_ACTION:
        return None
    if synthesis.inferred_state is InferredState.NO_SIGNIFICANT_EVENT:
        return None
    if synthesis.confidence_band is ConfidenceBand.HIGH:
        # High confidence with no action means a policy or guardrail deliberately
        # declined. That is a considered decision, not an unresolved one.
        return None

    sources = synthesis.corroboration.source_systems

    # --- contested: the arithmetic had to break a near-tie ------------------
    ranked = sorted(synthesis.affinity.items(), key=lambda kv: -kv[1])
    if len(ranked) >= 2 and ranked[0][1] > 0:
        (top_state, top_score), (second_state, second_score) = ranked[0], ranked[1]
        if second_score >= top_score * (1 - CONTESTED_MARGIN):
            return ReviewItem(
                as_of=synthesis.as_of,
                reason="contested",
                inferred_state=synthesis.inferred_state,
                confidence_band=synthesis.confidence_band,
                source_systems=sources,
                event_ids=synthesis.event_ids,
                detail=(
                    f"{top_state} ({top_score:.1f}) and {second_state} ({second_score:.1f}) "
                    f"score within {CONTESTED_MARGIN:.0%} of each other; the choice between "
                    f"them is not well separated by the evidence."
                ),
            )

    # --- weak but corroborated: breadth without strength --------------------
    if len(sources) >= MIN_SOURCES_FOR_REVIEW:
        return ReviewItem(
            as_of=synthesis.as_of,
            reason="weak_but_corroborated",
            inferred_state=synthesis.inferred_state,
            confidence_band=synthesis.confidence_band,
            source_systems=sources,
            event_ids=synthesis.event_ids,
            detail=(
                f"{len(sources)} independent source systems ({', '.join(sources)}) agree on "
                f"{synthesis.inferred_state.value}, but only at "
                f"{synthesis.confidence_band.value} confidence — below the bar for any action."
            ),
        )

    return None


class ReviewQueue:
    """Collects the checkpoints a human might want to look at."""

    def __init__(self) -> None:
        self.items: list[ReviewItem] = []
        self._last_signature: tuple | None = None

    @staticmethod
    def _signature(item: ReviewItem) -> tuple:
        return (item.reason, item.inferred_state, item.confidence_band, item.source_systems)

    def consider(self, synthesis: Synthesis, proposal: ActionProposal) -> ReviewItem | None:
        item = assess(synthesis, proposal)
        if item is None:
            # The situation resolved -- the next time it qualifies it is news again.
            self._last_signature = None
            return None
        signature = self._signature(item)
        if signature == self._last_signature:
            return None
        self._last_signature = signature
        self.items.append(item)
        return item

    def __len__(self) -> int:
        return len(self.items)

    def to_rows(self) -> list[dict]:
        return [item.to_row() for item in self.items]

    def summary(self) -> str:
        if not self.items:
            return "review queue: empty"
        by_reason: dict[str, int] = {}
        for item in self.items:
            by_reason[item.reason] = by_reason.get(item.reason, 0) + 1
        parts = ", ".join(f"{n} {reason}" for reason, n in sorted(by_reason.items()))
        return f"review queue: {len(self.items)} checkpoint(s) — {parts}"
