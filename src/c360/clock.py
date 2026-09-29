"""The simulated clock."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Callable

SECONDS_PER_DAY = 86_400.0


class ClockRegressionError(RuntimeError):
    """Raised when something tries to move simulated time backwards."""


class SimClock:
    """A monotonic clock over simulated time, with optional wall-clock pacing."""

    def __init__(
        self,
        start: datetime,
        seconds_per_sim_day: float = 0.0,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        if start.tzinfo is None:
            raise ValueError("SimClock requires a timezone-aware start time")
        self._now = start
        self._start = start
        self.seconds_per_sim_day = float(seconds_per_sim_day)
        self._sleep = sleep_fn if sleep_fn is not None else time.sleep
        self.total_slept_seconds = 0.0
        self.advance_count = 0

    @property
    def now(self) -> datetime:
        """Current simulated time. The only definition of 'now' in the system."""
        return self._now

    @property
    def start(self) -> datetime:
        return self._start

    @property
    def elapsed(self) -> timedelta:
        return self._now - self._start

    def advance_to(self, target: datetime) -> None:
        """Move simulated time forward to `target`, pacing in wall-clock if asked."""
        if target.tzinfo is None:
            raise ValueError("advance_to requires a timezone-aware datetime")
        if target < self._now:
            raise ClockRegressionError(
                f"refusing to move simulated time backwards: "
                f"now={self._now.isoformat()} target={target.isoformat()}"
            )

        delta = (target - self._now).total_seconds()
        if delta > 0 and self.seconds_per_sim_day > 0:
            sleep_for = (delta / SECONDS_PER_DAY) * self.seconds_per_sim_day
            self.total_slept_seconds += sleep_for
            self._sleep(sleep_for)

        self._now = target
        self.advance_count += 1

    def __repr__(self) -> str:
        return f"SimClock(now={self._now.isoformat()}, speed={self.seconds_per_sim_day}s/day)"


def daily_boundaries(start: datetime, end: datetime, step: timedelta = timedelta(days=1)):
    """Yield every simulated midnight in [start, end], inclusive of both ends."""
    if start > end:
        return
    current = start
    if step == timedelta(days=1):
        # Ground truth timestamps every checkpoint at 00:00Z, so a daily cycle has to
        # land on midnight even if the replay window opens mid-day. All three practice
        # configs start at midnight, which would hide the difference.
        midnight = start.replace(hour=0, minute=0, second=0, microsecond=0)
        current = midnight if midnight >= start else midnight + step
    while current <= end:
        yield current
        current += step
