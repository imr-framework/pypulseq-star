"""Reusable RF / receiver phase-cycling utilities.

This module provides small, sequence-independent abstractions for defining and
using RF / receiver phase cycles.

The initial API intentionally separates:

1. phase sets
   Reusable RF phase values such as four-step EXORCYCLE.

2. phase-cycle states
   One complete combination of RF phases and receiver phase.

3. phase cycles
   Ordered immutable collections of phase-cycle states.

4. phase-cycle construction
   Construction from independent RF phase sets using ``make_phase_cycle()`` or
   direct construction from an explicit state table.

The module does not contain PRESS-specific coherence-pathway logic. Sequence-
specific receiver-phase relationships should remain with the corresponding
sequence implementation and can be passed to ``make_phase_cycle()``.

Examples
--------
A simple two-RF independent phase cycle::

    cycle = make_phase_cycle(
        rf_phases=(
            EXORCYCLE_PHASES_4,
            EXORCYCLE_PHASES_4,
        ),
        receiver_phase=lambda rf1, rf2: rf1 - rf2,
        name="example",
    )

Retrieve a cyclic state::

    state = cycle.state(17)

which is equivalent to::

    state = cycle.state(17 % len(cycle))

Explicit phase tables can be represented directly::

    cycle = PhaseCycle.from_states(
        name="custom",
        states=(
            PhaseCycleState(
                rf_phases_rad=(0.0,),
                receiver_phase_rad=0.0,
            ),
            PhaseCycleState(
                rf_phases_rad=(math.pi,),
                receiver_phase_rad=math.pi,
            ),
        ),
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Callable, Iterable, Sequence


# =============================================================================
# PHASE HELPERS
# =============================================================================


TWO_PI = 2.0 * math.pi

QUADRATURE_PHASES_4: tuple[float, ...] = (
    0.0,
    0.5 * math.pi,
    math.pi,
    1.5 * math.pi,
)

EXORCYCLE_PHASES_4 = QUADRATURE_PHASES_4

def wrap_phase(
    phase_rad: float,
) -> float:
    """Wrap a phase angle to the interval ``[0, 2*pi)``.

    Parameters
    ----------
    phase_rad
        Phase in radians.

    Returns
    -------
    float
        Wrapped phase in radians.

    Raises
    ------
    ValueError
        If ``phase_rad`` is not finite.
    """

    phase = float(phase_rad)

    if not math.isfinite(phase):
        raise ValueError(
            "Phase must be finite."
        )

    wrapped = phase % TWO_PI

    # Avoid returning a floating-point value effectively equal to 2*pi.
    if math.isclose(
        wrapped,
        TWO_PI,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        return 0.0

    return wrapped


# =============================================================================
# REUSABLE RF PHASE SETS
# =============================================================================


EXORCYCLE_PHASES_4: tuple[float, ...] = (
    0.0,
    0.5 * math.pi,
    math.pi,
    1.5 * math.pi,
)
"""Canonical four-step EXORCYCLE RF phase set.

Equivalent phases in degrees:

    0, 90, 180, 270

This is a reusable RF phase set rather than a complete ``PhaseCycle`` because
the same four phases may be applied independently to different RF pulses.
"""


# =============================================================================
# PHASE-CYCLE STATE
# =============================================================================


@dataclass(frozen=True, slots=True)
class PhaseCycleState:
    """One complete RF / receiver phase-cycle state.

    Parameters
    ----------
    rf_phases_rad
        RF phases, in radians, in event order.

        For example, a PRESS state may contain::

            (
                excitation_phase,
                refocusing_1_phase,
                refocusing_2_phase,
            )

    receiver_phase_rad
        Receiver / ADC phase in radians.

    Notes
    -----
    Values are normalized to ``[0, 2*pi)`` during construction.
    """

    rf_phases_rad: tuple[float, ...]
    receiver_phase_rad: float

    def __post_init__(self) -> None:
        rf_phases = tuple(
            wrap_phase(phase)
            for phase in self.rf_phases_rad
        )

        if not rf_phases:
            raise ValueError(
                "PhaseCycleState requires at least one RF phase."
            )

        object.__setattr__(
            self,
            "rf_phases_rad",
            rf_phases,
        )

        object.__setattr__(
            self,
            "receiver_phase_rad",
            wrap_phase(
                self.receiver_phase_rad
            ),
        )

    @property
    def num_rf_events(self) -> int:
        """Return the number of RF events represented by this state."""

        return len(
            self.rf_phases_rad
        )

    def rf_phase(
        self,
        rf_index: int,
    ) -> float:
        """Return the RF phase for one RF event."""

        try:
            return self.rf_phases_rad[
                rf_index
            ]
        except IndexError as exc:
            raise IndexError(
                f"RF phase index {rf_index} is out of range for "
                f"{self.num_rf_events} RF events."
            ) from exc


# =============================================================================
# PHASE CYCLE
# =============================================================================


@dataclass(frozen=True, slots=True)
class PhaseCycle:
    """Immutable ordered RF / receiver phase cycle.

    All states must contain the same number of RF phases.
    """

    name: str
    states: tuple[PhaseCycleState, ...]

    def __post_init__(self) -> None:
        name = str(
            self.name
        ).strip()

        if not name:
            raise ValueError(
                "PhaseCycle name must not be empty."
            )

        states = tuple(
            self.states
        )

        if not states:
            raise ValueError(
                "PhaseCycle requires at least one state."
            )

        expected_rf_events = (
            states[0].num_rf_events
        )

        for index, state in enumerate(
            states
        ):
            if not isinstance(
                state,
                PhaseCycleState,
            ):
                raise TypeError(
                    "All PhaseCycle states must be PhaseCycleState "
                    f"instances; state {index} has type "
                    f"{type(state).__name__!r}."
                )

            if (
                state.num_rf_events
                != expected_rf_events
            ):
                raise ValueError(
                    "All PhaseCycle states must contain the same "
                    "number of RF phases."
                )

        object.__setattr__(
            self,
            "name",
            name,
        )

        object.__setattr__(
            self,
            "states",
            states,
        )

    @classmethod
    def from_states(
        cls,
        *,
        name: str,
        states: Iterable[PhaseCycleState],
    ) -> "PhaseCycle":
        """Create a phase cycle from an explicit state table."""

        return cls(
            name=name,
            states=tuple(states),
        )

    def __len__(self) -> int:
        return len(
            self.states
        )

    def __iter__(self):
        return iter(
            self.states
        )

    @property
    def num_rf_events(self) -> int:
        """Return the number of RF events represented by each state."""

        return self.states[
            0
        ].num_rf_events

    def state(
        self,
        index: int,
    ) -> PhaseCycleState:
        """Return a state using cyclic indexing.

        Examples
        --------
        For a 4-state cycle::

            state(0) == states[0]
            state(4) == states[0]
            state(5) == states[1]

        Negative indices also wrap cyclically.
        """

        if not isinstance(
            index,
            int,
        ):
            raise TypeError(
                "Phase-cycle index must be an integer."
            )

        return self.states[
            index % len(self.states)
        ]

    def rf_phase(
        self,
        index: int,
        rf_index: int,
    ) -> float:
        """Return one RF phase from a cyclic phase-cycle state."""

        return self.state(
            index
        ).rf_phase(
            rf_index
        )

    def receiver_phase(
        self,
        index: int,
    ) -> float:
        """Return the receiver phase from a cyclic phase-cycle state."""

        return self.state(
            index
        ).receiver_phase_rad


# =============================================================================
# PHASE-CYCLE CONSTRUCTION
# =============================================================================


ReceiverPhaseFunction = Callable[..., float]


def make_phase_cycle(
    *,
    rf_phases: Sequence[Sequence[float]],
    receiver_phase: ReceiverPhaseFunction,
    name: str = "phase_cycle",
) -> PhaseCycle:
    """Construct a phase cycle from independent RF phase sets.

    Parameters
    ----------
    rf_phases
        Sequence of RF phase sets.

        Each entry corresponds to one RF event.

        For example::

            rf_phases=(
                (0.0,),
                EXORCYCLE_PHASES_4,
                EXORCYCLE_PHASES_4,
            )

        represents one fixed RF phase followed by two independently cycled
        four-step RF phases, resulting in ``1 * 4 * 4 = 16`` states.

    receiver_phase
        Callable that receives the RF phases of one state as positional
        arguments and returns the corresponding receiver phase.

        For example::

            def receiver_phase(
                excitation,
                refocusing_1,
                refocusing_2,
            ):
                return (
                    excitation
                    - 2.0 * refocusing_1
                    + 2.0 * refocusing_2
                )

    name
        Human-readable phase-cycle name.

    Returns
    -------
    PhaseCycle
        Immutable phase-cycle definition.

    Notes
    -----
    Internally this function evaluates the Cartesian product of the supplied
    independent RF phase sets. The mathematical detail is intentionally hidden
    behind the spectroscopy-facing ``make_phase_cycle()`` API.
    """

    if not callable(
        receiver_phase
    ):
        raise TypeError(
            "receiver_phase must be callable."
        )

    normalized_phase_sets: list[
        tuple[float, ...]
    ] = []

    for rf_index, phase_set in enumerate(
        rf_phases
    ):
        phases = tuple(
            wrap_phase(phase)
            for phase in phase_set
        )

        if not phases:
            raise ValueError(
                f"RF phase set {rf_index} must contain at least one phase."
            )

        normalized_phase_sets.append(
            phases
        )

    if not normalized_phase_sets:
        raise ValueError(
            "At least one RF phase set must be provided."
        )

    states: list[
        PhaseCycleState
    ] = []

    for phase_combination in product(
        *normalized_phase_sets
    ):
        receiver_phase_rad = (
            receiver_phase(
                *phase_combination
            )
        )

        states.append(
            PhaseCycleState(
                rf_phases_rad=tuple(
                    phase_combination
                ),
                receiver_phase_rad=(
                    receiver_phase_rad
                ),
            )
        )

    return PhaseCycle(
        name=name,
        states=tuple(states),
    )


# =============================================================================
# CYCLOPS
# =============================================================================
#
# CYCLOPS is represented as a COMPLETE PhaseCycle because transmit and receiver
# phases are coupled.
#
# This initial preset uses matched transmit / receiver phase progression:
#
#     RF       Receiver
#
#      0°          0°
#     90°         90°
#    180°        180°
#    270°        270°
#
# The exact receiver sign convention used by a sequence/backend must remain
# explicit during integration. If ppstar's ADC phase convention requires the
# opposite sign for a particular use case, define a corresponding preset
# rather than silently changing this canonical table.
# =============================================================================


CYCLOPS_4 = PhaseCycle.from_states(
    name="cyclops_4",
    states=tuple(
        PhaseCycleState(
            rf_phases_rad=(
                phase,
            ),
            receiver_phase_rad=phase,
        )
        for phase in EXORCYCLE_PHASES_4
    ),
)
"""Canonical four-step CYCLOPS transmit/receiver phase cycle."""


# =============================================================================
# PUBLIC EXPORTS
# =============================================================================


__all__ = [
    "CYCLOPS_4",
    "EXORCYCLE_PHASES_4",
    "PhaseCycle",
    "PhaseCycleState",
    "make_phase_cycle",
    "wrap_phase",
]