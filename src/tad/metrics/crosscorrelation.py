from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CrossCorrelationResult:
    """Placeholder result for future cross-correlation metrics."""

    value: Optional[float] = None


def compute_crosscorrelation() -> CrossCorrelationResult:
    """Return a placeholder cross-correlation result."""
    return CrossCorrelationResult()
