"""Unit tests for PyPulseq-Star voxel geometry."""

from __future__ import annotations

import math

import pytest

from pypulseq_star.geometry import (
    DEFAULT_MAX_ABS_POSITION_M,
    Voxel,
)


# =============================================================================
# Basic construction
# =============================================================================


def test_voxel_constructs_with_valid_geometry() -> None:
    voxel = Voxel(
        size_read_m=20e-3,
        size_phase_m=30e-3,
        size_slice_m=40e-3,
        position_read_m=10e-3,
        position_phase_m=-15e-3,
        position_slice_m=0.0,
    )

    assert voxel.size_read_m == pytest.approx(20e-3)
    assert voxel.size_phase_m == pytest.approx(30e-3)
    assert voxel.size_slice_m == pytest.approx(40e-3)

    assert voxel.position_read_m == pytest.approx(10e-3)
    assert voxel.position_phase_m == pytest.approx(-15e-3)
    assert voxel.position_slice_m == pytest.approx(0.0)


def test_voxel_defaults_to_isocenter() -> None:
    voxel = Voxel(
        size_read_m=20e-3,
        size_phase_m=20e-3,
        size_slice_m=20e-3,
    )

    assert voxel.position_m == pytest.approx(
        (
            0.0,
            0.0,
            0.0,
        )
    )

    assert voxel.is_at_isocenter


def test_negative_voxel_positions_are_allowed() -> None:
    voxel = Voxel(
        size_read_m=20e-3,
        size_phase_m=20e-3,
        size_slice_m=20e-3,
        position_read_m=-0.10,
        position_phase_m=-0.05,
        position_slice_m=-0.15,
    )

    assert voxel.position_read_m == pytest.approx(-0.10)
    assert voxel.position_phase_m == pytest.approx(-0.05)
    assert voxel.position_slice_m == pytest.approx(-0.15)


# =============================================================================
# Default position guardrail
# =============================================================================


def test_default_position_limit_is_20_cm() -> None:
    assert DEFAULT_MAX_ABS_POSITION_M == pytest.approx(
        0.20
    )


@pytest.mark.parametrize(
    "position",
    [
        -0.20,
        0.20,
    ],
)
def test_voxel_position_exactly_at_default_limit_is_allowed(
    position: float,
) -> None:
    voxel = Voxel(
        size_read_m=20e-3,
        size_phase_m=20e-3,
        size_slice_m=20e-3,
        position_read_m=position,
    )

    assert voxel.position_read_m == pytest.approx(position)


@pytest.mark.parametrize(
    "position",
    [
        -0.200001,
        0.200001,
    ],
)
def test_voxel_position_beyond_default_limit_is_rejected(
    position: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="exceeds the allowed",
    ):
        Voxel(
            size_read_m=20e-3,
            size_phase_m=20e-3,
            size_slice_m=20e-3,
            position_read_m=position,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "position_read_m",
        "position_phase_m",
        "position_slice_m",
    ],
)
def test_position_limit_applies_to_each_logical_axis(
    field_name: str,
) -> None:
    kwargs = {
        "size_read_m": 20e-3,
        "size_phase_m": 20e-3,
        "size_slice_m": 20e-3,
        field_name: 0.25,
    }

    with pytest.raises(
        ValueError,
        match="exceeds the allowed",
    ):
        Voxel(
            **kwargs
        )


def test_custom_position_limit_is_respected() -> None:
    voxel = Voxel(
        size_read_m=20e-3,
        size_phase_m=20e-3,
        size_slice_m=20e-3,
        position_read_m=0.25,
        max_abs_position_m=0.30,
    )

    assert voxel.position_read_m == pytest.approx(0.25)
    assert voxel.max_abs_position_m == pytest.approx(0.30)


def test_custom_position_limit_rejects_out_of_range_position() -> None:
    with pytest.raises(
        ValueError,
        match="exceeds the allowed",
    ):
        Voxel(
            size_read_m=20e-3,
            size_phase_m=20e-3,
            size_slice_m=20e-3,
            position_slice_m=-0.31,
            max_abs_position_m=0.30,
        )


def test_position_guardrail_can_be_disabled() -> None:
    voxel = Voxel(
        size_read_m=20e-3,
        size_phase_m=20e-3,
        size_slice_m=20e-3,
        position_read_m=1.0,
        position_phase_m=-2.0,
        position_slice_m=3.0,
        max_abs_position_m=None,
    )

    assert voxel.position_m == pytest.approx(
        (
            1.0,
            -2.0,
            3.0,
        )
    )

    assert voxel.max_abs_position_m is None


# =============================================================================
# Dimension validation
# =============================================================================


@pytest.mark.parametrize(
    "field_name",
    [
        "size_read_m",
        "size_phase_m",
        "size_slice_m",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        0.0,
        -1e-3,
        -1.0,
    ],
)
def test_voxel_rejects_nonpositive_dimensions(
    field_name: str,
    invalid_value: float,
) -> None:
    kwargs = {
        "size_read_m": 20e-3,
        "size_phase_m": 20e-3,
        "size_slice_m": 20e-3,
    }

    kwargs[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match="must be > 0",
    ):
        Voxel(
            **kwargs
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "size_read_m",
        "size_phase_m",
        "size_slice_m",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_voxel_rejects_nonfinite_dimensions(
    field_name: str,
    invalid_value: float,
) -> None:
    kwargs = {
        "size_read_m": 20e-3,
        "size_phase_m": 20e-3,
        "size_slice_m": 20e-3,
    }

    kwargs[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        Voxel(
            **kwargs
        )


# =============================================================================
# Position validation
# =============================================================================


@pytest.mark.parametrize(
    "field_name",
    [
        "position_read_m",
        "position_phase_m",
        "position_slice_m",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_voxel_rejects_nonfinite_positions(
    field_name: str,
    invalid_value: float,
) -> None:
    kwargs = {
        "size_read_m": 20e-3,
        "size_phase_m": 20e-3,
        "size_slice_m": 20e-3,
        field_name: invalid_value,
    }

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        Voxel(
            **kwargs
        )


@pytest.mark.parametrize(
    "invalid_limit",
    [
        0.0,
        -0.20,
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_voxel_rejects_invalid_position_guardrail(
    invalid_limit: float,
) -> None:
    with pytest.raises(
        ValueError,
    ):
        Voxel(
            size_read_m=20e-3,
            size_phase_m=20e-3,
            size_slice_m=20e-3,
            max_abs_position_m=invalid_limit,
        )


# =============================================================================
# Compact representations
# =============================================================================


def test_size_m_returns_read_phase_slice_order() -> None:
    voxel = Voxel(
        size_read_m=10e-3,
        size_phase_m=20e-3,
        size_slice_m=30e-3,
    )

    assert voxel.size_m == pytest.approx(
        (
            10e-3,
            20e-3,
            30e-3,
        )
    )


def test_position_m_returns_read_phase_slice_order() -> None:
    voxel = Voxel(
        size_read_m=20e-3,
        size_phase_m=20e-3,
        size_slice_m=20e-3,
        position_read_m=10e-3,
        position_phase_m=-20e-3,
        position_slice_m=30e-3,
    )

    assert voxel.position_m == pytest.approx(
        (
            10e-3,
            -20e-3,
            30e-3,
        )
    )


# =============================================================================
# Axis-based access
# =============================================================================


def test_size_along_returns_correct_dimension() -> None:
    voxel = Voxel(
        size_read_m=10e-3,
        size_phase_m=20e-3,
        size_slice_m=30e-3,
    )

    assert voxel.size_along("read") == pytest.approx(10e-3)
    assert voxel.size_along("phase") == pytest.approx(20e-3)
    assert voxel.size_along("slice") == pytest.approx(30e-3)


def test_position_along_returns_correct_position() -> None:
    voxel = Voxel(
        size_read_m=20e-3,
        size_phase_m=20e-3,
        size_slice_m=20e-3,
        position_read_m=-10e-3,
        position_phase_m=20e-3,
        position_slice_m=-30e-3,
    )

    assert voxel.position_along("read") == pytest.approx(-10e-3)
    assert voxel.position_along("phase") == pytest.approx(20e-3)
    assert voxel.position_along("slice") == pytest.approx(-30e-3)


@pytest.mark.parametrize(
    "axis",
    [
        "READ",
        " Read ",
        "PHASE",
        " slice ",
    ],
)
def test_axis_names_are_normalized(
    axis: str,
) -> None:
    voxel = Voxel(
        size_read_m=10e-3,
        size_phase_m=20e-3,
        size_slice_m=30e-3,
    )

    # The call itself should succeed.
    value = voxel.size_along(axis)

    assert value > 0.0


@pytest.mark.parametrize(
    "axis",
    [
        "",
        "x",
        "y",
        "z",
        "frequency",
        "foobar",
    ],
)
def test_invalid_logical_axis_is_rejected(
    axis: str,
) -> None:
    voxel = Voxel(
        size_read_m=20e-3,
        size_phase_m=20e-3,
        size_slice_m=20e-3,
    )

    with pytest.raises(
        ValueError,
        match="Logical axis must be one of",
    ):
        voxel.size_along(axis)


# =============================================================================
# Bounds
# =============================================================================


def test_bounds_along_centered_voxel() -> None:
    voxel = Voxel(
        size_read_m=20e-3,
        size_phase_m=40e-3,
        size_slice_m=60e-3,
    )

    assert voxel.bounds_along("read") == pytest.approx(
        (
            -10e-3,
            10e-3,
        )
    )

    assert voxel.bounds_along("phase") == pytest.approx(
        (
            -20e-3,
            20e-3,
        )
    )

    assert voxel.bounds_along("slice") == pytest.approx(
        (
            -30e-3,
            30e-3,
        )
    )


def test_bounds_along_shifted_voxel() -> None:
    voxel = Voxel(
        size_read_m=20e-3,
        size_phase_m=20e-3,
        size_slice_m=20e-3,
        position_read_m=-50e-3,
    )

    assert voxel.read_bounds_m == pytest.approx(
        (
            -60e-3,
            -40e-3,
        )
    )


def test_named_bound_properties_match_bounds_along() -> None:
    voxel = Voxel(
        size_read_m=10e-3,
        size_phase_m=20e-3,
        size_slice_m=30e-3,
        position_read_m=5e-3,
        position_phase_m=-5e-3,
        position_slice_m=10e-3,
    )

    assert voxel.read_bounds_m == pytest.approx(
        voxel.bounds_along("read")
    )

    assert voxel.phase_bounds_m == pytest.approx(
        voxel.bounds_along("phase")
    )

    assert voxel.slice_bounds_m == pytest.approx(
        voxel.bounds_along("slice")
    )


# =============================================================================
# Derived geometry
# =============================================================================


def test_voxel_volume() -> None:
    voxel = Voxel(
        size_read_m=10e-3,
        size_phase_m=20e-3,
        size_slice_m=30e-3,
    )

    expected = (
        10e-3
        * 20e-3
        * 30e-3
    )

    assert voxel.volume_m3 == pytest.approx(expected)


def test_is_at_isocenter_false_when_any_position_is_nonzero() -> None:
    voxel = Voxel(
        size_read_m=20e-3,
        size_phase_m=20e-3,
        size_slice_m=20e-3,
        position_slice_m=-1e-3,
    )

    assert not voxel.is_at_isocenter


def test_max_center_displacement_m() -> None:
    voxel = Voxel(
        size_read_m=20e-3,
        size_phase_m=20e-3,
        size_slice_m=20e-3,
        position_read_m=-0.03,
        position_phase_m=0.08,
        position_slice_m=-0.05,
    )

    assert voxel.max_center_displacement_m == pytest.approx(
        0.08
    )


# =============================================================================
# Convenience constructor
# =============================================================================


def test_from_size_and_position() -> None:
    voxel = Voxel.from_size_and_position(
        size_m=(
            20e-3,
            30e-3,
            40e-3,
        ),
        position_m=(
            -10e-3,
            15e-3,
            -20e-3,
        ),
    )

    assert voxel.size_m == pytest.approx(
        (
            20e-3,
            30e-3,
            40e-3,
        )
    )

    assert voxel.position_m == pytest.approx(
        (
            -10e-3,
            15e-3,
            -20e-3,
        )
    )


def test_from_size_and_position_defaults_to_isocenter() -> None:
    voxel = Voxel.from_size_and_position(
        size_m=(
            20e-3,
            20e-3,
            20e-3,
        )
    )

    assert voxel.is_at_isocenter


def test_from_size_and_position_uses_custom_position_limit() -> None:
    voxel = Voxel.from_size_and_position(
        size_m=(
            20e-3,
            20e-3,
            20e-3,
        ),
        position_m=(
            0.25,
            0.0,
            0.0,
        ),
        max_abs_position_m=0.30,
    )

    assert voxel.position_read_m == pytest.approx(0.25)


def test_from_size_and_position_can_disable_position_limit() -> None:
    voxel = Voxel.from_size_and_position(
        size_m=(
            20e-3,
            20e-3,
            20e-3,
        ),
        position_m=(
            1.0,
            -1.0,
            2.0,
        ),
        max_abs_position_m=None,
    )

    assert voxel.position_m == pytest.approx(
        (
            1.0,
            -1.0,
            2.0,
        )
    )


@pytest.mark.parametrize(
    "size_m",
    [
        (),
        (1.0,),
        (1.0, 2.0),
        (1.0, 2.0, 3.0, 4.0),
    ],
)
def test_from_size_and_position_requires_three_size_values(
    size_m: tuple[float, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match="size_m must contain exactly three values",
    ):
        Voxel.from_size_and_position(
            size_m=size_m,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "position_m",
    [
        (),
        (0.0,),
        (0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    ],
)
def test_from_size_and_position_requires_three_position_values(
    position_m: tuple[float, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match="position_m must contain exactly three values",
    ):
        Voxel.from_size_and_position(
            size_m=(
                20e-3,
                20e-3,
                20e-3,
            ),
            position_m=position_m,  # type: ignore[arg-type]
        )


# =============================================================================
# Human-readable description
# =============================================================================


def test_describe_reports_millimeter_geometry() -> None:
    voxel = Voxel(
        size_read_m=20e-3,
        size_phase_m=30e-3,
        size_slice_m=40e-3,
        position_read_m=10e-3,
        position_phase_m=-5e-3,
        position_slice_m=0.0,
    )

    description = voxel.describe()

    assert "Voxel(" in description
    assert "size_mm=(20, 30, 40)" in description
    assert "position_from_isocenter_mm=(10, -5, 0)" in description
    assert "center_position_limit=+/-200 mm" in description


def test_describe_reports_unbounded_position_limit() -> None:
    voxel = Voxel(
        size_read_m=20e-3,
        size_phase_m=20e-3,
        size_slice_m=20e-3,
        max_abs_position_m=None,
    )

    assert "center_position_limit=unbounded" in voxel.describe()


# =============================================================================
# Immutability
# =============================================================================


def test_voxel_is_immutable() -> None:
    voxel = Voxel(
        size_read_m=20e-3,
        size_phase_m=20e-3,
        size_slice_m=20e-3,
    )

    with pytest.raises(
        AttributeError,
    ):
        voxel.position_read_m = 10e-3  # type: ignore[misc]