"""Synchronization helpers for timeline alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, TypeVar

T = TypeVar("T")


@dataclass
class TimedEvent:
    timestamp: float
    payload: object


def find_nearest_with_tolerance(
    target_ts: float,
    events: Sequence[T],
    get_ts=lambda x: x.timestamp,
    tolerance: float = 1.0,
) -> Optional[T]:
    """Find nearest event by timestamp within tolerance in seconds."""
    nearest = None
    nearest_dt = float("inf")
    for event in events:
        dt = abs(get_ts(event) - target_ts)
        if dt <= tolerance and dt < nearest_dt:
            nearest = event
            nearest_dt = dt
    return nearest
