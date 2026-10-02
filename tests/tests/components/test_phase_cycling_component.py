"""Tests for reusable PyPulseq-Star phase-cycling components."""

from __future__ import annotations

import math

import pytest

from pypulseq_star.components.phase_cycling import (
    CYCLOPS_4,
    EXORCYCLE_PHASES_4,
    QUADRATURE_PHASES_4,
    PhaseCycle,
    PhaseCycleState,
    make_phase_cycle,
    wrap_phase,
)


# =============================================================================
# wrap_phase()
# =============================================================================


@pytest.mark.parametrize(
    ("input_phase", "expected"),
    [
        (0.0, 0.0),
        (0.5 * math.pi, 0.5 * math.pi),
        (math.pi, math.pi),
        (1.5 * math.pi, 1.5 * math.pi),
        (2.0 * math.pi, 0.0),
        (4.0 * math.pi, 0.0),
        (-2.0 * math.pi, 0.0),
        (-0.5 * math.pi, 1.5 * math.pi),
        (-math.pi, math.pi),
        (2.5 * math.pi, 0.5 * math.pi),
    ],
)
def test_wrap_phase_maps_to_zero_two_pi(
    input_phase: float,
    expected: float,
) -> None:
    assert wrap_phase(input_phase) == pytest.approx(expected)


@pytest.mark.parametrize(
    "input_phase",
    [
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_wrap_phase_rejects_nonfinite_values(
    input_phase: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Phase must be finite",
    ):
        wrap_phase(input_phase)


def test_wrap_phase_returns_float() -> None:
    result = wrap_phase(1)

    assert isinstance(result, float)


# =============================================================================
# Canonical phase sets
# =============================================================================


def test_quadrature_phases_4_are_canonical() -> None:
    assert QUADRATURE_PHASES_4 == pytest.approx(
        (
            0.0,
            0.5 * math.pi,
            math.pi,
            1.5 * math.pi,
        )
    )


def test_exorcycle_phases_4_match_quadrature_set() -> None:
    assert EXORCYCLE_PHASES_4 == pytest.approx(
        QUADRATURE_PHASES_4
    )


# =============================================================================
# PhaseCycleState
# =============================================================================


def test_phase_cycle_state_normalizes_all_phases() -> None:
    state = PhaseCycleState(
        rf_phases_rad=(
            -0.5 * math.pi,
            2.5 * math.pi,
        ),
        receiver_phase_rad=-math.pi,
    )

    assert state.rf_phases_rad == pytest.approx(
        (
            1.5 * math.pi,
            0.5 * math.pi,
        )
    )

    assert state.receiver_phase_rad == pytest.approx(
        math.pi
    )


def test_phase_cycle_state_requires_at_least_one_rf_phase() -> None:
    with pytest.raises(
        ValueError,
        match="requires at least one RF phase",
    ):
        PhaseCycleState(
            rf_phases_rad=(),
            receiver_phase_rad=0.0,
        )


def test_phase_cycle_state_rejects_nonfinite_rf_phase() -> None:
    with pytest.raises(
        ValueError,
        match="Phase must be finite",
    ):
        PhaseCycleState(
            rf_phases_rad=(
                0.0,
                math.nan,
            ),
            receiver_phase_rad=0.0,
        )


def test_phase_cycle_state_rejects_nonfinite_receiver_phase() -> None:
    with pytest.raises(
        ValueError,
        match="Phase must be finite",
    ):
        PhaseCycleState(
            rf_phases_rad=(
                0.0,
            ),
            receiver_phase_rad=math.inf,
        )


def test_phase_cycle_state_num_rf_events() -> None:
    state = PhaseCycleState(
        rf_phases_rad=(
            0.0,
            math.pi,
            0.5 * math.pi,
        ),
        receiver_phase_rad=0.0,
    )

    assert state.num_rf_events == 3


def test_phase_cycle_state_rf_phase_lookup() -> None:
    state = PhaseCycleState(
        rf_phases_rad=(
            0.0,
            0.5 * math.pi,
            math.pi,
        ),
        receiver_phase_rad=0.0,
    )

    assert state.rf_phase(0) == pytest.approx(0.0)
    assert state.rf_phase(1) == pytest.approx(0.5 * math.pi)
    assert state.rf_phase(2) == pytest.approx(math.pi)


def test_phase_cycle_state_rf_phase_rejects_out_of_range_index() -> None:
    state = PhaseCycleState(
        rf_phases_rad=(
            0.0,
            math.pi,
        ),
        receiver_phase_rad=0.0,
    )

    with pytest.raises(
        IndexError,
        match="out of range",
    ):
        state.rf_phase(2)


# =============================================================================
# PhaseCycle construction and validation
# =============================================================================


def test_phase_cycle_from_states() -> None:
    states = (
        PhaseCycleState(
            rf_phases_rad=(0.0,),
            receiver_phase_rad=0.0,
        ),
        PhaseCycleState(
            rf_phases_rad=(math.pi,),
            receiver_phase_rad=math.pi,
        ),
    )

    cycle = PhaseCycle.from_states(
        name="two_step",
        states=states,
    )

    assert cycle.name == "two_step"
    assert len(cycle) == 2
    assert cycle.states == states


def test_phase_cycle_strips_name_whitespace() -> None:
    cycle = PhaseCycle.from_states(
        name="  test_cycle  ",
        states=(
            PhaseCycleState(
                rf_phases_rad=(0.0,),
                receiver_phase_rad=0.0,
            ),
        ),
    )

    assert cycle.name == "test_cycle"


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "\t",
    ],
)
def test_phase_cycle_rejects_empty_name(
    name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="name must not be empty",
    ):
        PhaseCycle.from_states(
            name=name,
            states=(
                PhaseCycleState(
                    rf_phases_rad=(0.0,),
                    receiver_phase_rad=0.0,
                ),
            ),
        )


def test_phase_cycle_requires_at_least_one_state() -> None:
    with pytest.raises(
        ValueError,
        match="requires at least one state",
    ):
        PhaseCycle.from_states(
            name="empty",
            states=(),
        )


def test_phase_cycle_requires_consistent_number_of_rf_events() -> None:
    states = (
        PhaseCycleState(
            rf_phases_rad=(0.0,),
            receiver_phase_rad=0.0,
        ),
        PhaseCycleState(
            rf_phases_rad=(
                0.0,
                math.pi,
            ),
            receiver_phase_rad=0.0,
        ),
    )

    with pytest.raises(
        ValueError,
        match="same number of RF phases",
    ):
        PhaseCycle.from_states(
            name="inconsistent",
            states=states,
        )


def test_phase_cycle_rejects_non_state_objects() -> None:
    with pytest.raises(
        TypeError,
        match="must be PhaseCycleState",
    ):
        PhaseCycle(
            name="bad",
            states=(
                PhaseCycleState(
                    rf_phases_rad=(0.0,),
                    receiver_phase_rad=0.0,
                ),
                "not_a_state",  # type: ignore[arg-type]
            ),
        )


def test_phase_cycle_num_rf_events() -> None:
    cycle = PhaseCycle.from_states(
        name="three_rf",
        states=(
            PhaseCycleState(
                rf_phases_rad=(
                    0.0,
                    math.pi,
                    0.5 * math.pi,
                ),
                receiver_phase_rad=0.0,
            ),
        ),
    )

    assert cycle.num_rf_events == 3


# =============================================================================
# Cyclic indexing
# =============================================================================


def test_phase_cycle_state_wraps_forward() -> None:
    cycle = PhaseCycle.from_states(
        name="cycle",
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

    assert cycle.state(0) == cycle.states[0]
    assert cycle.state(1) == cycle.states[1]
    assert cycle.state(2) == cycle.states[0]
    assert cycle.state(3) == cycle.states[1]


def test_phase_cycle_state_wraps_negative_indices() -> None:
    cycle = PhaseCycle.from_states(
        name="cycle",
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

    assert cycle.state(-1) == cycle.states[1]
    assert cycle.state(-2) == cycle.states[0]


def test_phase_cycle_state_requires_integer_index() -> None:
    cycle = PhaseCycle.from_states(
        name="cycle",
        states=(
            PhaseCycleState(
                rf_phases_rad=(0.0,),
                receiver_phase_rad=0.0,
            ),
        ),
    )

    with pytest.raises(
        TypeError,
        match="index must be an integer",
    ):
        cycle.state(1.5)  # type: ignore[arg-type]


def test_phase_cycle_rf_phase_accessor() -> None:
    cycle = PhaseCycle.from_states(
        name="cycle",
        states=(
            PhaseCycleState(
                rf_phases_rad=(
                    0.0,
                    math.pi,
                ),
                receiver_phase_rad=0.5 * math.pi,
            ),
        ),
    )

    assert cycle.rf_phase(0, 0) == pytest.approx(0.0)
    assert cycle.rf_phase(0, 1) == pytest.approx(math.pi)


def test_phase_cycle_receiver_phase_accessor() -> None:
    cycle = PhaseCycle.from_states(
        name="cycle",
        states=(
            PhaseCycleState(
                rf_phases_rad=(0.0,),
                receiver_phase_rad=0.5 * math.pi,
            ),
        ),
    )

    assert cycle.receiver_phase(0) == pytest.approx(
        0.5 * math.pi
    )


# =============================================================================
# Iteration behavior
# =============================================================================


def test_phase_cycle_is_iterable() -> None:
    states = (
        PhaseCycleState(
            rf_phases_rad=(0.0,),
            receiver_phase_rad=0.0,
        ),
        PhaseCycleState(
            rf_phases_rad=(math.pi,),
            receiver_phase_rad=math.pi,
        ),
    )

    cycle = PhaseCycle.from_states(
        name="iterable",
        states=states,
    )

    assert tuple(cycle) == states


# =============================================================================
# make_phase_cycle()
# =============================================================================


def test_make_phase_cycle_single_rf_phase_set() -> None:
    cycle = make_phase_cycle(
        rf_phases=(
            EXORCYCLE_PHASES_4,
        ),
        receiver_phase=lambda rf1: rf1,
        name="single_rf",
    )

    assert len(cycle) == 4
    assert cycle.num_rf_events == 1

    for index, phase in enumerate(
        EXORCYCLE_PHASES_4
    ):
        assert cycle.rf_phase(
            index,
            0,
        ) == pytest.approx(phase)

        assert cycle.receiver_phase(
            index
        ) == pytest.approx(phase)


def test_make_phase_cycle_two_independent_four_step_sets_gives_16_states() -> None:
    cycle = make_phase_cycle(
        rf_phases=(
            EXORCYCLE_PHASES_4,
            EXORCYCLE_PHASES_4,
        ),
        receiver_phase=lambda rf1, rf2: (
            -2.0 * rf1
            + 2.0 * rf2
        ),
        name="two_exorcycles",
    )

    assert len(cycle) == 16
    assert cycle.num_rf_events == 2


def test_make_phase_cycle_contains_all_independent_combinations() -> None:
    cycle = make_phase_cycle(
        rf_phases=(
            EXORCYCLE_PHASES_4,
            EXORCYCLE_PHASES_4,
        ),
        receiver_phase=lambda rf1, rf2: 0.0,
        name="all_combinations",
    )

    realized = {
        state.rf_phases_rad
        for state in cycle
    }

    expected = {
        (
            rf1,
            rf2,
        )
        for rf1 in EXORCYCLE_PHASES_4
        for rf2 in EXORCYCLE_PHASES_4
    }

    assert realized == expected


def test_make_phase_cycle_three_rf_press_style_gives_16_states() -> None:
    cycle = make_phase_cycle(
        rf_phases=(
            (0.0,),
            EXORCYCLE_PHASES_4,
            EXORCYCLE_PHASES_4,
        ),
        receiver_phase=lambda excitation, rf2, rf3: (
            excitation
            - 2.0 * rf2
            + 2.0 * rf3
        ),
        name="press_style",
    )

    assert len(cycle) == 16
    assert cycle.num_rf_events == 3

    assert all(
        state.rf_phases_rad[0]
        == pytest.approx(0.0)
        for state in cycle
    )


def test_make_phase_cycle_preserves_product_order() -> None:
    cycle = make_phase_cycle(
        rf_phases=(
            (
                0.0,
                math.pi,
            ),
            (
                0.0,
                0.5 * math.pi,
            ),
        ),
        receiver_phase=lambda rf1, rf2: 0.0,
        name="order",
    )

    realized = [
        state.rf_phases_rad
        for state in cycle
    ]

    expected = [
        (
            0.0,
            0.0,
        ),
        (
            0.0,
            0.5 * math.pi,
        ),
        (
            math.pi,
            0.0,
        ),
        (
            math.pi,
            0.5 * math.pi,
        ),
    ]

    assert realized == pytest.approx(expected)


def test_make_phase_cycle_wraps_input_phases() -> None:
    cycle = make_phase_cycle(
        rf_phases=(
            (
                -0.5 * math.pi,
                2.5 * math.pi,
            ),
        ),
        receiver_phase=lambda rf1: rf1,
        name="wrapped",
    )

    assert cycle.states[0].rf_phases_rad == pytest.approx(
        (
            1.5 * math.pi,
        )
    )

    assert cycle.states[1].rf_phases_rad == pytest.approx(
        (
            0.5 * math.pi,
        )
    )


def test_make_phase_cycle_wraps_receiver_phase() -> None:
    cycle = make_phase_cycle(
        rf_phases=(
            (
                0.0,
            ),
        ),
        receiver_phase=lambda rf1: -0.5 * math.pi,
        name="receiver_wrap",
    )

    assert cycle.receiver_phase(
        0
    ) == pytest.approx(
        1.5 * math.pi
    )


def test_make_phase_cycle_requires_receiver_callable() -> None:
    with pytest.raises(
        TypeError,
        match="receiver_phase must be callable",
    ):
        make_phase_cycle(
            rf_phases=(
                EXORCYCLE_PHASES_4,
            ),
            receiver_phase=0.0,  # type: ignore[arg-type]
            name="bad_receiver",
        )


def test_make_phase_cycle_requires_at_least_one_rf_phase_set() -> None:
    with pytest.raises(
        ValueError,
        match="At least one RF phase set",
    ):
        make_phase_cycle(
            rf_phases=(),
            receiver_phase=lambda: 0.0,
            name="empty",
        )


def test_make_phase_cycle_rejects_empty_rf_phase_set() -> None:
    with pytest.raises(
        ValueError,
        match="must contain at least one phase",
    ):
        make_phase_cycle(
            rf_phases=(
                EXORCYCLE_PHASES_4,
                (),
            ),
            receiver_phase=lambda rf1, rf2: 0.0,
            name="empty_set",
        )


def test_make_phase_cycle_propagates_receiver_function_error() -> None:
    def bad_receiver_phase(
        rf1: float,
    ) -> float:
        raise RuntimeError(
            "receiver calculation failed"
        )

    with pytest.raises(
        RuntimeError,
        match="receiver calculation failed",
    ):
        make_phase_cycle(
            rf_phases=(
                (
                    0.0,
                ),
            ),
            receiver_phase=bad_receiver_phase,
            name="bad_receiver_function",
        )


# =============================================================================
# CYCLOPS_4
# =============================================================================


def test_cyclops_4_has_four_states() -> None:
    assert len(CYCLOPS_4) == 4


def test_cyclops_4_has_one_rf_event() -> None:
    assert CYCLOPS_4.num_rf_events == 1


def test_cyclops_4_uses_quadrature_rf_phases() -> None:
    realized = tuple(
        state.rf_phases_rad[0]
        for state in CYCLOPS_4
    )

    assert realized == pytest.approx(
        QUADRATURE_PHASES_4
    )


def test_cyclops_4_receiver_phase_matches_rf_phase() -> None:
    for state in CYCLOPS_4:
        assert state.receiver_phase_rad == pytest.approx(
            state.rf_phases_rad[0]
        )


def test_cyclops_4_cycles_correctly() -> None:
    assert CYCLOPS_4.state(4) == CYCLOPS_4.state(0)
    assert CYCLOPS_4.state(5) == CYCLOPS_4.state(1)
    assert CYCLOPS_4.state(8) == CYCLOPS_4.state(0)


# =============================================================================
# Immutability
# =============================================================================


def test_phase_cycle_state_is_immutable() -> None:
    state = PhaseCycleState(
        rf_phases_rad=(0.0,),
        receiver_phase_rad=0.0,
    )

    with pytest.raises(
        AttributeError,
    ):
        state.receiver_phase_rad = math.pi  # type: ignore[misc]


def test_phase_cycle_is_immutable() -> None:
    cycle = PhaseCycle.from_states(
        name="immutable",
        states=(
            PhaseCycleState(
                rf_phases_rad=(0.0,),
                receiver_phase_rad=0.0,
            ),
        ),
    )

    with pytest.raises(
        AttributeError,
    ):
        cycle.name = "changed"  # type: ignore[misc]