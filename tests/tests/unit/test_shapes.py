"""Unit tests for waveform shape construction."""

from __future__ import annotations

import math

import numpy as np
import pytest

import pypulseq_star as ppstar

pytestmark = pytest.mark.unit


def test_block_rf_has_expected_duration_and_samples() -> None:
    rf = ppstar.make_block_pulse(
        math.pi / 2.0,
        300e-6,
        use="excitation",
    )

    assert rf.shape is not None
    assert rf.shape.duration == pytest.approx(300e-6)
    assert rf.shape.kind == "rf_block"
    assert rf.shape.rf_raster_time > 0


def test_arbitrary_gradient_preserves_waveform_physical_channel(
    system: ppstar.Opts,
) -> None:
    """Legacy physical-channel construction remains supported."""

    waveform = np.array([0.0, 1.0, 0.5, 0.0])

    gradient = ppstar.make_arbitrary_grad(
        channel="x",
        waveform=waveform,
        system=system,
        name="gx_arbitrary",
    )

    stored = np.asarray(
        getattr(
            gradient,
            "waveform",
            gradient.parameters.get("waveform"),
        )
    )

    np.testing.assert_allclose(stored, waveform)

    assert gradient.channel == "x"
    assert gradient.axis_role == "read"
    assert gradient.encoding_role == "read"
    assert gradient.metadata["logical_axis"] == "read"
    assert gradient.metadata["gradient_coordinate_mode"] == "physical"


def test_arbitrary_gradient_preserves_waveform_logical_axis(
    system: ppstar.Opts,
) -> None:
    """Logical-axis-only construction is supported."""

    waveform = np.array([0.0, 1.0, 0.5, 0.0])

    gradient = ppstar.make_arbitrary_grad(
        axis_role="phase",
        waveform=waveform,
        system=system,
        name="gphase_arbitrary",
    )

    stored = np.asarray(
        getattr(
            gradient,
            "waveform",
            gradient.parameters.get("waveform"),
        )
    )

    np.testing.assert_allclose(stored, waveform)

    # Internal construction channel only.
    assert gradient.channel == "y"

    # Scientific/logical identity.
    assert gradient.axis_role == "phase"
    assert gradient.encoding_role == "phase"
    assert gradient.metadata["logical_axis"] == "phase"
    assert gradient.metadata["gradient_coordinate_mode"] == "logical"


def test_trapezoid_area_matches_requested_area_physical_channel(
    system: ppstar.Opts,
) -> None:
    """Legacy PyPulseq-style physical-channel call remains supported."""

    gradient = ppstar.make_trapezoid(
        channel="y",
        area=4.0,
        duration=2e-3,
        system=system,
        name="gy_area",
    )

    assert float(gradient.area) == pytest.approx(
        4.0,
        rel=1e-6,
    )

    assert gradient.channel == "y"
    assert gradient.axis_role == "phase"
    assert gradient.encoding_role == "phase"
    assert gradient.metadata["logical_axis"] == "phase"
    assert gradient.metadata["gradient_coordinate_mode"] == "physical"


def test_trapezoid_area_matches_requested_area_logical_axis(
    system: ppstar.Opts,
) -> None:
    """Logical-axis-only trapezoid construction is supported."""

    gradient = ppstar.make_trapezoid(
        axis_role="phase",
        area=4.0,
        duration=2e-3,
        system=system,
        name="gphase_area",
    )

    assert float(gradient.area) == pytest.approx(
        4.0,
        rel=1e-6,
    )

    # Internal construction channel only.
    assert gradient.channel == "y"

    # Scientific/logical identity.
    assert gradient.axis_role == "phase"
    assert gradient.encoding_role == "phase"
    assert gradient.metadata["logical_axis"] == "phase"
    assert gradient.metadata["gradient_coordinate_mode"] == "logical"


def test_trapezoid_accepts_explicit_physical_and_logical_axes(
    system: ppstar.Opts,
) -> None:
    """Explicit channel + logical role remains supported."""

    gradient = ppstar.make_trapezoid(
        channel="x",
        axis_role="slice",
        area=4.0,
        duration=2e-3,
        system=system,
        name="gx_logical_slice",
    )

    assert float(gradient.area) == pytest.approx(
        4.0,
        rel=1e-6,
    )

    # Explicit construction channel is preserved.
    assert gradient.channel == "x"

    # Logical identity is independent of construction channel.
    assert gradient.axis_role == "slice"
    assert gradient.encoding_role == "slice"
    assert gradient.metadata["logical_axis"] == "slice"
    assert gradient.metadata["gradient_coordinate_mode"] == "physical"


def test_invalid_gradient_channel_is_rejected(
    system: ppstar.Opts,
) -> None:
    with pytest.raises((ValueError, AssertionError)):
        ppstar.make_trapezoid(
            channel="q",
            area=1.0,
            duration=1e-3,
            system=system,
        )


def test_invalid_logical_gradient_axis_is_rejected(
    system: ppstar.Opts,
) -> None:
    with pytest.raises(ValueError):
        ppstar.make_trapezoid(
            axis_role="measurement",
            area=1.0,
            duration=1e-3,
            system=system,
        )


def test_gradient_requires_channel_or_logical_axis(
    system: ppstar.Opts,
) -> None:
    with pytest.raises(ValueError):
        ppstar.make_trapezoid(
            area=1.0,
            duration=1e-3,
            system=system,
        )


def test_axis_role_and_encoding_role_must_agree(
    system: ppstar.Opts,
) -> None:
    with pytest.raises(ValueError):
        ppstar.make_trapezoid(
            axis_role="read",
            encoding_role="slice",
            area=1.0,
            duration=1e-3,
            system=system,
        )


def test_split_gradient_preserves_logical_axis(
    system: ppstar.Opts,
) -> None:
    gradient = ppstar.make_trapezoid(
        axis_role="slice",
        area=4.0,
        duration=2e-3,
        system=system,
        name="gslice_parent",
    )

    ramp_up, flat_top, ramp_down = ppstar.split_gradient(
        gradient,
        system=system,
        name="gslice_split",
    )

    for part in (
        ramp_up,
        flat_top,
        ramp_down,
    ):
        assert part.axis_role == "slice"
        assert part.encoding_role == "slice"
        assert part.metadata["logical_axis"] == "slice"
        assert part.metadata["gradient_coordinate_mode"] == "logical"