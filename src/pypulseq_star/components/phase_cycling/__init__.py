"""Reusable phase-cycling components for PyPulseq-Star.

This package provides sequence-independent abstractions for defining and
using RF / receiver phase cycles.

The public API includes:

- ``PhaseCycleState``:
  One complete RF / receiver phase-cycle state.

- ``PhaseCycle``:
  An immutable ordered collection of phase-cycle states.

- ``make_phase_cycle``:
  Construct a phase cycle from independent RF phase sets.

- ``wrap_phase``:
  Normalize phases to the interval ``[0, 2*pi)``.

- ``QUADRATURE_PHASES_4``:
  Canonical four-step quadrature phases.

- ``EXORCYCLE_PHASES_4``:
  Four-step RF phase set used for EXORCYCLE-style cycling.

- ``CYCLOPS_4``:
  Canonical four-step transmit / receiver CYCLOPS phase cycle.
"""

from .phase_cycling import (
    CYCLOPS_4,
    EXORCYCLE_PHASES_4,
    QUADRATURE_PHASES_4,
    PhaseCycle,
    PhaseCycleState,
    make_phase_cycle,
    wrap_phase,
)

__all__ = [
    "CYCLOPS_4",
    "EXORCYCLE_PHASES_4",
    "QUADRATURE_PHASES_4",
    "PhaseCycle",
    "PhaseCycleState",
    "make_phase_cycle",
    "wrap_phase",
]