"""Convenience helpers for feasibility sweeps."""
from __future__ import annotations

from typing import Any, Iterable

from .bounds import ParameterBoundReport, feasible_range


def sweep_parameters(
    sequence: Any,
    parameters: Iterable[str],
    **kwargs: Any,
) -> dict[str, ParameterBoundReport]:
    """Return feasible-range reports for several protocol parameters."""

    return {name: feasible_range(sequence, name, **kwargs) for name in parameters}
