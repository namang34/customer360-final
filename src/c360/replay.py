"""The replay engine.

History is ordered by event_time; live events are ordered by RELEASE time,
max(ingestion_time, event_time), so an event can never be released before it
happened and a late arrival surfaces on the day the bank learned of it. Within a
single timestamp, sort_rank orders events (0) ahead of the clock tick (1), so a
checkpoint always sees everything released up to and including its own moment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Literal

from .clock import SimClock, daily_boundaries
from .schema import Event, LoadReport, load_events, load_json, parse_timestamp


@dataclass(frozen=True)
class ReplayConfig:
    scenario_id: str
    simulated_start: datetime
    simulated_end: datetime
    seconds_per_sim_day: float

    @classmethod
    def from_file(cls, path: Path) -> "ReplayConfig":
        raw = load_json(path)
        start = parse_timestamp(raw["simulated_start"], "simulated_start")
        end = parse_timestamp(raw["simulated_end"], "simulated_end")
        if end < start:
            raise ValueError(f"simulated_end {end} precedes simulated_start {start}")
        return cls(
            scenario_id=str(raw.get("scenario_id", path.parent.name)),
            simulated_start=start,
            simulated_end=end,
            seconds_per_sim_day=float(raw.get("replay_speed_seconds_per_simulated_day", 0)),
        )

    @property
    def span_days(self) -> int:
        return (self.simulated_end - self.simulated_start).days


# Ticks -- the two things the stream can yield

@dataclass(frozen=True)
class EventTick:
    """An event arrived. Fires the event-based trigger (perception agents)."""

    as_of: datetime
    event: Event
    kind: Literal["event"] = "event"

    @property
    def event_id(self) -> str:
        return self.event.event_id


@dataclass(frozen=True)
class ClockTick:
    """A simulated day boundary passed. Fires the time-based trigger."""

    as_of: datetime
    events_since_last_tick: int = 0
    kind: Literal["clock"] = "clock"


Tick = EventTick | ClockTick


class ReplayEngine:
    """Replays a scenario as a stream: history loaded up front, live events one at a time."""

    def __init__(
        self,
        scenario_dir: str | Path,
        speed: float | None = None,
        tick_interval: timedelta = timedelta(days=1),
        strict: bool = False,
        sleep_fn=None,
    ) -> None:
        self.dir = Path(scenario_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"scenario directory not found: {self.dir}")

        self.config = ReplayConfig.from_file(self.dir / "replay_config.json")
        # An explicit speed overrides the config. Default is the config value, so
        # a plain run reproduces the intended pacing; tests pass speed=0.
        self.speed = self.config.seconds_per_sim_day if speed is None else float(speed)
        self.tick_interval = tick_interval
        self.strict = strict
        self._sleep_fn = sleep_fn

        self.entities: dict[str, Any] = {}
        self.history: list[Event] = []
        self.live: list[Event] = []
        self.history_report: LoadReport | None = None
        self.live_report: LoadReport | None = None
        self.warnings: list[str] = []
        self.clock = SimClock(
            self.config.simulated_start, seconds_per_sim_day=self.speed, sleep_fn=sleep_fn
        )
        self._loaded = False


    def load(self) -> "ReplayEngine":
        entities_path = self.dir / "entities.json"
        if entities_path.exists():
            self.entities = load_json(entities_path)

        self.history_report = load_events(self.dir / "history_seed.jsonl", strict=self.strict)
        self.live_report = load_events(self.dir / "live_stream.jsonl", strict=self.strict)

        # History is the backstory: it is loaded in one go by design, because all of it
        # predates simulated_start and so none of it can leak the future.
        self.history = sorted(self.history_report.events, key=lambda e: (e.event_time, e.event_id))

        # Live events are sorted by RELEASE time -- see the module docstring above.
        self.live = sorted(self.live_report.events, key=lambda e: (e.release_time, e.event_id))

        self._validate()
        self._loaded = True
        return self

    def _validate(self) -> None:
        """Check the assumptions the rest of the system is entitled to make."""
        cfg = self.config

        # 1. History must genuinely be history.
        late_history = [e for e in self.history if e.event_time >= cfg.simulated_start]
        if late_history:
            self.warnings.append(
                f"{len(late_history)} history_seed event(s) at or after simulated_start "
                f"{cfg.simulated_start.isoformat()} (first: {late_history[0].event_id}). "
                "These are NOT pre-loaded as backstory; they are held back and released "
                "by the live stream instead, so they cannot leak into an early checkpoint."
            )
            keep = {e.event_id for e in late_history}
            self.history = [e for e in self.history if e.event_id not in keep]
            self.live = sorted(
                self.live + late_history, key=lambda e: (e.release_time, e.event_id)
            )

        # 2. Live events should fall inside the configured window.
        # Released before the window even opened means the bank already knew when the
        # replay starts, so it is backstory. Fold it into history -- the mirror of the
        # case above. Without this the clock is asked to run backwards and the run dies.
        pre_window = [e for e in self.live if e.release_time < cfg.simulated_start]
        if pre_window:
            self.warnings.append(
                f"{len(pre_window)} live event(s) released before simulated_start "
                f"(first: {pre_window[0].event_id}). Already known when the window "
                "opened, so they are folded into history as backstory rather than replayed."
            )
            drop = {e.event_id for e in pre_window}
            self.live = [e for e in self.live if e.event_id not in drop]
            self.history = sorted(
                self.history + pre_window, key=lambda e: (e.event_time, e.event_id)
            )

        early = [e for e in self.live if e.event_time < cfg.simulated_start]
        if early:
            self.warnings.append(
                f"{len(early)} live event(s) with event_time before simulated_start "
                f"(first: {early[0].event_id}). Released at their ingestion time as normal."
            )
        beyond = [e for e in self.live if e.release_time > cfg.simulated_end]
        if beyond:
            self.warnings.append(
                f"{len(beyond)} live event(s) release after simulated_end "
                f"(first: {beyond[0].event_id}). They are still streamed; the clock runs "
                "past simulated_end to reach them."
            )

        # 3. Impossible timestamps: received before they happened.
        impossible = [e for e in self.live + self.history if e.ingestion_time < e.event_time]
        if impossible:
            self.warnings.append(
                f"{len(impossible)} event(s) have ingestion_time BEFORE event_time "
                f"(first: {impossible[0].event_id}). Release is floored at event_time so "
                "the no-future-leakage rule still holds."
            )

        # 4. Single-customer scenarios: more than one id is worth knowing about.
        customers = {e.customer_id for e in self.history + self.live}
        if len(customers) > 1:
            self.warnings.append(f"multiple customer_ids present: {sorted(customers)}")

        for report in (self.history_report, self.live_report):
            if report and not report.ok:
                self.warnings.append(f"{report.path.name}: {len(report.rejected)} row(s) rejected")


    def stream(self) -> Iterator[Tick]:
        """Yield ticks in simulated-time order, advancing the clock before each one."""
        if not self._loaded:
            self.load()

        # sort_rank: 0 = event, 1 = clock tick, so a tick sees the day's events first.
        schedule: list[tuple[datetime, int, Event | None]] = [
            (e.release_time, 0, e) for e in self.live
        ]
        schedule += [(ts, 1, None) for ts in self.checkpoint_times]
        # Third sort key keeps events with the same release_time deterministic.
        schedule.sort(key=lambda item: (item[0], item[1], item[2].event_id if item[2] else ""))

        events_since_tick = 0
        for timestamp, rank, event in schedule:
            self.clock.advance_to(timestamp)
            if rank == 0 and event is not None:
                events_since_tick += 1
                yield EventTick(as_of=self.clock.now, event=event)
            else:
                yield ClockTick(as_of=self.clock.now, events_since_last_tick=events_since_tick)
                events_since_tick = 0

    # introspection helpers (used heavily by the tests)

    def visible_events(self, as_of: datetime) -> list[Event]:
        """Every event the system is ALLOWED to know about at `as_of`."""
        if not self._loaded:
            self.load()
        return [
            e
            for e in self.history + self.live
            if e.event_time <= as_of and e.release_time <= as_of
        ]

    @property
    def checkpoint_times(self) -> list[datetime]:
        """Every timestamp at which a checkpoint row will be emitted."""
        return list(
            daily_boundaries(
                self.config.simulated_start, self.config.simulated_end, self.tick_interval
            )
        )

    def describe(self) -> str:
        if not self._loaded:
            self.load()
        lines = [
            f"scenario         : {self.config.scenario_id}",
            f"window           : {self.config.simulated_start:%Y-%m-%d} -> "
            f"{self.config.simulated_end:%Y-%m-%d}  ({self.config.span_days} days)",
            f"pacing           : {self.speed}s per simulated day "
            f"({'instant' if self.speed == 0 else f'~{self.config.span_days * self.speed / 60:.1f} min run'})",
            f"history events   : {len(self.history)}",
            f"live events      : {len(self.live)}",
            f"checkpoints      : {len(self.checkpoint_times)} daily boundaries",
        ]
        if self.history:
            lines.append(
                f"history spans    : {self.history[0].event_time:%Y-%m-%d} -> "
                f"{self.history[-1].event_time:%Y-%m-%d}"
            )
        if self.live_report:
            lines.append(f"live load        : {self.live_report.summary()}")
        if self.history_report:
            lines.append(f"history load     : {self.history_report.summary()}")
        for warning in self.warnings:
            lines.append(f"WARNING          : {warning}")
        return "\n".join(lines)
