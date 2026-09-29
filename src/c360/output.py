"""The inferred-events output writer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .pii import Redactor
from .schema import Action, ConfidenceBand, HitlStatus, InferredState

# Every event_id in this dataset looks like EVT_000123. The notes field must
# cite at least one for any row that proposes an action -- see CheckpointError.
EVENT_ID_PATTERN = re.compile(r"\bEVT_\d{4,}\b")

# Actions that the mid-term architecture requires to pass the corroboration guardrail
# before they may fire.
GUARDED_ACTIONS = frozenset({
    Action.COMPLIANCE_FRAUD_HOLD,
    Action.RELATIONSHIP_MANAGER_ESCALATION,
    Action.PERSONALIZED_OFFER,
})


class CheckpointError(ValueError):
    """A checkpoint that would produce an invalid or ungradeable output row."""


def _iso_z(moment: datetime) -> str:
    """Format as ...Z, matching the dataset's own style exactly."""
    if moment.tzinfo is None:
        raise CheckpointError(f"as_of_time must be timezone-aware: {moment!r}")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Checkpoint:
    """One row of the graded output."""

    as_of_time: datetime
    inferred_state: InferredState
    confidence_band: ConfidenceBand
    action: Action
    action_subtype: str | None = None
    hitl_status: HitlStatus = HitlStatus.AUTO_APPROVED
    notes: str = ""

    # Not serialised -- the graded file carries citations inside notes. Held on the
    # object so the run log and the tests can see WHICH events drove a decision
    # without parsing the notes prose.
    citations: tuple[str, ...] = field(default=(), compare=False)
    guardrail_checked: bool = field(default=False, compare=False)

    def __post_init__(self) -> None:
        # Coerce plain strings into enums so an agent may return either. An
        # unrecognised spelling raises here rather than silently reaching disk.
        object.__setattr__(self, "inferred_state", _coerce(InferredState, self.inferred_state, "inferred_state"))
        object.__setattr__(self, "confidence_band", _coerce(ConfidenceBand, self.confidence_band, "confidence_band"))
        object.__setattr__(self, "action", _coerce(Action, self.action, "action"))
        object.__setattr__(self, "hitl_status", _coerce(HitlStatus, self.hitl_status, "hitl_status"))

        if self.as_of_time.tzinfo is None:
            raise CheckpointError("as_of_time must be timezone-aware")

        # rules that are ours, not the dataset's

        # 1. Explainability is graded. A row that proposes an intervention must say
        # which events justify it.
        if self.action is not Action.NO_ACTION and not EVENT_ID_PATTERN.search(self.notes):
            raise CheckpointError(
                f"action={self.action.value} at {self.as_of_time.isoformat()} but notes cite no "
                f"event_id. Explainability is a graded requirement; notes were: {self.notes!r}"
            )

        # 2. The mid-term design says any action other than no_action defaults to
        # escalated.
        if self.action is not Action.NO_ACTION and self.hitl_status is HitlStatus.AUTO_APPROVED:
            raise CheckpointError(
                f"action={self.action.value} must not be auto_approved -- the architecture routes "
                "every non-no_action decision to HITL. Use escalated (or a human_* outcome)."
            )

        # 3. no_action with a subtype is incoherent: there is no action to
        #    subtype. Cheap check, catches a copy-paste slip in agent code.
        if self.action is Action.NO_ACTION and self.action_subtype:
            raise CheckpointError(
                f"action=no_action cannot carry action_subtype={self.action_subtype!r}"
            )

        # 4. A guarded action must have been through the corroboration check.
        if self.action in GUARDED_ACTIONS and not self.guardrail_checked:
            raise CheckpointError(
                f"action={self.action.value} requires the >=2 independent source_systems "
                "corroboration guardrail, but guardrail_checked is False"
            )

    def to_row(self) -> dict[str, Any]:
        """The exact JSON object shape the README specifies. Nothing extra."""
        return {
            "as_of_time": _iso_z(self.as_of_time),
            "inferred_state": self.inferred_state.value,
            "confidence_band": self.confidence_band.value,
            "action": self.action.value,
            "action_subtype": self.action_subtype,
            "hitl_status": self.hitl_status.value,
            "notes": self.notes,
        }


def _coerce(enum_cls, value, field_name: str):
    """Accept an enum member or its exact string value; reject anything else."""
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError:
        allowed = ", ".join(m.value for m in enum_cls)
        raise CheckpointError(
            f"{field_name}={value!r} is not a permitted value. "
            f"The enums are fixed by the dataset README; allowed: {allowed}"
        ) from None


class InferredEventsWriter:
    """Collects checkpoints during a run and writes the graded JSON array."""

    def __init__(self, redactor: Redactor | None = None, scenario_id: str | None = None) -> None:
        self.redactor = redactor
        self.scenario_id = scenario_id
        self._checkpoints: list[Checkpoint] = []

    def __len__(self) -> int:
        return len(self._checkpoints)

    @property
    def checkpoints(self) -> list[Checkpoint]:
        return list(self._checkpoints)

    def add(self, checkpoint: Checkpoint) -> Checkpoint:
        """Append one checkpoint, enforcing ordering and PII rules."""
        if self._checkpoints:
            previous = self._checkpoints[-1]
            if checkpoint.as_of_time < previous.as_of_time:
                raise CheckpointError(
                    f"checkpoints must be chronological: {checkpoint.as_of_time.isoformat()} "
                    f"follows {previous.as_of_time.isoformat()}"
                )
            if checkpoint.as_of_time == previous.as_of_time:
                raise CheckpointError(
                    f"duplicate checkpoint at {checkpoint.as_of_time.isoformat()} -- a scorer "
                    "matching on as_of_time could not tell which row is the answer"
                )

        if self.redactor is not None:
            leaks = self.redactor.contains_pii(checkpoint.notes)
            if leaks:
                raise CheckpointError(
                    f"notes at {checkpoint.as_of_time.isoformat()} contain un-redacted PII "
                    f"({'; '.join(leaks)}). Scrub before constructing the Checkpoint."
                )

        self._checkpoints.append(checkpoint)
        return checkpoint

    def extend(self, checkpoints: Iterable[Checkpoint]) -> None:
        for checkpoint in checkpoints:
            self.add(checkpoint)

    def to_rows(self) -> list[dict[str, Any]]:
        return [c.to_row() for c in self._checkpoints]

    def write(self, path: str | Path) -> Path:
        """Serialise to disk and immediately read it back to prove it round-trips."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_rows()
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

        reloaded = json.loads(path.read_text(encoding="utf-8"))
        if reloaded != rows:
            raise CheckpointError(f"{path} did not round-trip through JSON")
        return path

    # self-checks the run script calls before declaring success

    def covers(self, required: Iterable[datetime]) -> list[datetime]:
        """Return any required as_of_times that have NO row in this file."""
        present = {_iso_z(c.as_of_time) for c in self._checkpoints}
        return [when for when in required if _iso_z(when) not in present]

    def summary(self) -> str:
        if not self._checkpoints:
            return "no checkpoints"
        actions: dict[str, int] = {}
        states: dict[str, int] = {}
        for checkpoint in self._checkpoints:
            actions[checkpoint.action.value] = actions.get(checkpoint.action.value, 0) + 1
            states[checkpoint.inferred_state.value] = states.get(checkpoint.inferred_state.value, 0) + 1
        span = (
            f"{self._checkpoints[0].as_of_time:%Y-%m-%d} -> "
            f"{self._checkpoints[-1].as_of_time:%Y-%m-%d}"
        )
        return (
            f"{len(self._checkpoints)} checkpoints ({span})\n"
            f"  states : {dict(sorted(states.items(), key=lambda kv: -kv[1]))}\n"
            f"  actions: {dict(sorted(actions.items(), key=lambda kv: -kv[1]))}"
        )


def load_checkpoints(path: str | Path) -> list[dict[str, Any]]:
    """Read an inferred-events file back. Used by the scoring harness in step 9."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
