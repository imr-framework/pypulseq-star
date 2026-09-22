from __future__ import annotations

import math

import pytest

from pypulseq_star.make.rf import make_sinc_pulse
from pypulseq_star.opts import Opts
from pypulseq_star.shapes.rf import (
    SeqStarRFArbitraryShape,
    SeqStarRFBlockShape,
    SeqStarRFGaussShape,
    SeqStarRFShape,
    SeqStarRFSincShape,
    _sinc,
)

pytestmark = pytest.mark.unit

GAMMA = 42.575575e6


# =============================================================================
# Helpers
# =============================================================================


def integrated_flip(samples, raster):
    return 2 * math.pi * GAMMA * sum(samples["v"][0]["am"]) * raster


def make_test_system() -> Opts:
    """Return a system permissive enough for RF slice-selection unit tests."""

    return Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_dead_time=100e-6,
        rf_ringdown_time=20e-6,
        rf_raster_time=1e-6,
        grad_raster_time=10e-6,
    )


# =============================================================================
# Base RF shapes
# =============================================================================


def test_base_rf_shape_and_common_validation():
    with pytest.raises(NotImplementedError):
        SeqStarRFShape(
            name="base",
            duration=1e-3,
        ).to_gammastar_samples(
            flip_angle=1.0,
            gamma_hz_per_t=GAMMA,
        )

    block = SeqStarRFBlockShape(
        name="block",
        duration=1e-3,
        rf_raster_time=1e-6,
    )

    with pytest.raises(ValueError, match="flip_angle"):
        block.to_gammastar_samples(
            flip_angle=0,
            gamma_hz_per_t=GAMMA,
        )

    with pytest.raises(ValueError, match="gamma_hz_per_t"):
        block.to_gammastar_samples(
            flip_angle=1,
            gamma_hz_per_t=0,
        )

    with pytest.raises(ValueError, match="not aligned"):
        SeqStarRFBlockShape(
            name="block",
            duration=1.5e-6,
            rf_raster_time=1e-6,
        ).to_gammastar_samples(
            flip_angle=1,
            gamma_hz_per_t=GAMMA,
        )

    with pytest.raises(ValueError, match="max_rf must be non-negative"):
        block.to_gammastar_samples(
            flip_angle=1,
            gamma_hz_per_t=GAMMA,
            max_rf=-1,
        )

    with pytest.raises(ValueError, match="exceeds max_rf"):
        block.to_gammastar_samples(
            flip_angle=math.pi,
            gamma_hz_per_t=GAMMA,
            max_rf=1e-9,
        )


def test_block_rf_samples_have_expected_amplitude():
    shape = SeqStarRFBlockShape(
        name="block",
        duration=1e-3,
    )

    result = shape.to_gammastar_samples(
        flip_angle=math.pi / 2,
        gamma_hz_per_t=GAMMA,
    )

    amp = result["v"][0]["am"][0]

    assert amp == pytest.approx(
        (math.pi / 2)
        / (
            2
            * math.pi
            * GAMMA
            * 1e-3
        )
    )

    assert result["t"] == [
        0.0,
        1e-3,
    ]


@pytest.mark.parametrize(
    "shape_cls",
    [
        SeqStarRFSincShape,
        SeqStarRFGaussShape,
    ],
)
def test_rastered_rf_shapes_normalize_and_preserve_flip(
    shape_cls,
):
    shape = shape_cls(
        name="shape",
        duration=20e-6,
        rf_raster_time=1e-6,
    )

    t, envelope = shape.normalized_envelope()

    assert len(t) == len(envelope) == 20
    assert max(abs(x) for x in envelope) == pytest.approx(1.0)

    samples = shape.to_gammastar_samples(
        flip_angle=math.pi / 3,
        gamma_hz_per_t=GAMMA,
    )

    assert integrated_flip(
        samples,
        1e-6,
    ) == pytest.approx(
        math.pi / 3
    )


@pytest.mark.parametrize(
    "shape_cls",
    [
        SeqStarRFSincShape,
        SeqStarRFGaussShape,
    ],
)
def test_rastered_rf_shape_validation(
    shape_cls,
):
    with pytest.raises(
        ValueError,
        match="rf_raster_time",
    ):
        shape_cls(
            name="shape",
            duration=1e-3,
        ).normalized_envelope()

    with pytest.raises(
        ValueError,
        match="at least two",
    ):
        shape_cls(
            name="shape",
            duration=1e-6,
            rf_raster_time=1e-6,
        ).normalized_envelope()

    with pytest.raises(
        ValueError,
        match="time_bw_product",
    ):
        shape_cls(
            name="shape",
            duration=10e-6,
            rf_raster_time=1e-6,
            time_bw_product=0,
        ).normalized_envelope()

    with pytest.raises(
        ValueError,
        match="apodization",
    ):
        shape_cls(
            name="shape",
            duration=10e-6,
            rf_raster_time=1e-6,
            apodization=2,
        ).normalized_envelope()

    with pytest.raises(
        ValueError,
        match="center_pos",
    ):
        shape_cls(
            name="shape",
            duration=10e-6,
            rf_raster_time=1e-6,
            center_pos=2,
        ).normalized_envelope()


# =============================================================================
# Arbitrary RF shapes
# =============================================================================



def test_arbitrary_rf_happy_path_and_validation():
    shape = SeqStarRFArbitraryShape(
        name="arb",
        duration=4e-6,
        rf_raster_time=1e-6,
        signal=[
            1.0,
            0.5,
            0.5,
            1.0,
        ],
    )

    t, envelope = shape.normalized_envelope()

    assert t == pytest.approx(
        [
            0.5e-6,
            1.5e-6,
            2.5e-6,
            3.5e-6,
        ]
    )
    assert envelope == [
        1.0,
        0.5,
        0.5,
        1.0,
    ]

    samples = shape.to_gammastar_samples(
        flip_angle=1.0,
        gamma_hz_per_t=GAMMA,
    )

    assert integrated_flip(
        samples,
        1e-6,
    ) == pytest.approx(
        1.0
    )

    invalids = [
        SeqStarRFArbitraryShape(
            name="arb",
            duration=1e-6,
            rf_raster_time=None,
            signal=[1],
        ),
        SeqStarRFArbitraryShape(
            name="arb",
            duration=1e-6,
            rf_raster_time=0,
            signal=[1],
        ),
        SeqStarRFArbitraryShape(
            name="arb",
            duration=1e-6,
            rf_raster_time=1e-6,
            signal=[],
        ),
        SeqStarRFArbitraryShape(
            name="arb",
            duration=1e-6,
            rf_raster_time=1e-6,
            signal=[0],
        ),
        SeqStarRFArbitraryShape(
            name="arb",
            duration=1e-6,
            rf_raster_time=1e-6,
            signal=[float("nan")],
        ),
    ]

    for invalid in invalids:
        with pytest.raises(
            (
                ValueError,
                TypeError,
            )
        ):
            invalid.normalized_envelope()

    # Complex arbitrary RF is now supported. Validate AM/FM export instead
    # of retaining the historical NotImplementedError expectation.
    complex_shape = SeqStarRFArbitraryShape(
        name="arb_complex",
        duration=4e-6,
        rf_raster_time=1e-6,
        signal=[
            1.0 + 0.0j,
            1.0 + 1.0j,
            1.0 + 0.0j,
            1.0 - 1.0j,
        ],
    )

    complex_t, complex_envelope = complex_shape.normalized_envelope()
    assert len(complex_t) == 4
    assert len(complex_envelope) == 4
    assert any(isinstance(value, complex) for value in complex_envelope)

    complex_samples = complex_shape.to_gammastar_samples(
        flip_angle=1.0,
        gamma_hz_per_t=GAMMA,
    )

    assert len(complex_samples["t"]) == 4
    assert len(complex_samples["v"]) == 1

    channel = complex_samples["v"][0]
    assert len(channel["am"]) == 4
    assert len(channel["fm"]) == 4
    assert all(value >= 0.0 for value in channel["am"])
    assert any(abs(value) > 0.0 for value in channel["fm"])



# =============================================================================
# Sinc helper
# =============================================================================


def test_sinc_helper():
    assert _sinc(
        0.0
    ) == pytest.approx(
        1.0
    )

    assert _sinc(
        1.0
    ) == pytest.approx(
        0.0,
        abs=1e-12,
    )


# =============================================================================
# Selective sinc RF: backward compatibility
# =============================================================================


def test_sinc_pulse_legacy_selection_defaults_to_physical_z_and_logical_slice():
    """Old make_sinc_pulse calls must retain their historical behavior."""

    system = make_test_system()

    rf, gz, gzr = make_sinc_pulse(
        flip_angle=math.pi / 2,
        duration=3e-3,
        time_bw_product=4.0,
        slice_thickness=20e-3,
        system=system,
        return_gz=True,
        name="legacy_rf",
    )

    # Historical behavior:
    #
    #     physical construction channel = z
    #     logical axis role             = slice
    #
    assert gz.channel == "z"
    assert gzr.channel == "z"

    assert gz.axis_role == "slice"
    assert gzr.axis_role == "slice"

    assert gz.encoding_role == "slice"
    assert gzr.encoding_role == "slice"

    assert rf.parameters["slice_select_channel"] == "z"
    assert rf.parameters["slice_select_axis_role"] == "slice"
    assert rf.parameters["logical_axis"] == "slice"


# =============================================================================
# Selective sinc RF: logical-axis authoring
# =============================================================================


@pytest.mark.parametrize(
    (
        "axis_role",
        "expected_internal_channel",
    ),
    [
        (
            "read",
            "x",
        ),
        (
            "phase",
            "y",
        ),
        (
            "slice",
            "z",
        ),
    ],
)
def test_sinc_pulse_supports_logical_selection_axes(
    axis_role: str,
    expected_internal_channel: str,
):
    """Logical RF localization should propagate to both selective gradients."""

    system = make_test_system()

    rf, gsel, greph = make_sinc_pulse(
        flip_angle=math.pi / 2,
        duration=3e-3,
        time_bw_product=4.0,
        slice_thickness=20e-3,
        axis_role=axis_role,
        system=system,
        return_gz=True,
        name=f"rf_{axis_role}",
    )

    # -------------------------------------------------------------------------
    # Logical geometry must remain authoritative.
    # -------------------------------------------------------------------------

    assert gsel.axis_role == axis_role
    assert greph.axis_role == axis_role

    assert gsel.encoding_role == axis_role
    assert greph.encoding_role == axis_role

    # -------------------------------------------------------------------------
    # make_trapezoid() still needs an internal physical construction channel.
    #
    # Logical authoring maps:
    #
    #     read  -> x
    #     phase -> y
    #     slice -> z
    #
    # but gradient metadata should record that this was logically authored.
    # -------------------------------------------------------------------------

    assert gsel.channel == expected_internal_channel
    assert greph.channel == expected_internal_channel

    assert (
        gsel.metadata["gradient_coordinate_mode"]
        == "logical"
    )

    assert (
        greph.metadata["gradient_coordinate_mode"]
        == "logical"
    )

    # -------------------------------------------------------------------------
    # RF metadata / parameters should also retain the localization axis.
    # -------------------------------------------------------------------------

    assert (
        rf.parameters["slice_select_axis_role"]
        == axis_role
    )

    assert (
        rf.parameters["logical_axis"]
        == axis_role
    )

    # In logical-authoring mode no physical channel was prescribed by the RF
    # caller. The construction channel is therefore intentionally stored as
    # None on the RF-side bookkeeping.
    assert (
        rf.parameters["slice_select_channel"]
        is None
    )


@pytest.mark.parametrize(
    "axis_role",
    [
        "READ",
        " Phase ",
        "SLICE",
    ],
)
def test_sinc_pulse_normalizes_logical_axis_role(
    axis_role: str,
):
    system = make_test_system()

    _, gsel, greph = make_sinc_pulse(
        flip_angle=math.pi / 2,
        duration=3e-3,
        time_bw_product=4.0,
        slice_thickness=20e-3,
        axis_role=axis_role,
        system=system,
        return_gz=True,
        name="normalized_axis_rf",
    )

    expected = axis_role.strip().lower()

    assert gsel.axis_role == expected
    assert greph.axis_role == expected


# =============================================================================
# Selective sinc RF: explicit physical + logical authoring
# =============================================================================


def test_sinc_pulse_allows_explicit_physical_channel_and_logical_axis():
    """Low-level callers may specify physical construction and logical role."""

    system = make_test_system()

    rf, gsel, greph = make_sinc_pulse(
        flip_angle=math.pi / 2,
        duration=3e-3,
        time_bw_product=4.0,
        slice_thickness=20e-3,
        channel="x",
        axis_role="slice",
        system=system,
        return_gz=True,
        name="explicit_channel_rf",
    )

    assert gsel.channel == "x"
    assert greph.channel == "x"

    assert gsel.axis_role == "slice"
    assert greph.axis_role == "slice"

    # Because the user explicitly supplied a physical channel, this should be
    # represented as physically authored rather than logically authored.
    assert (
        gsel.metadata["gradient_coordinate_mode"]
        == "physical"
    )

    assert (
        greph.metadata["gradient_coordinate_mode"]
        == "physical"
    )

    assert (
        rf.parameters["slice_select_channel"]
        == "x"
    )

    assert (
        rf.parameters["slice_select_axis_role"]
        == "slice"
    )


# =============================================================================
# Selective sinc RF: validation
# =============================================================================


@pytest.mark.parametrize(
    "axis_role",
    [
        "",
        "x",
        "y",
        "z",
        "frequency",
        "foobar",
    ],
)
def test_sinc_pulse_rejects_invalid_axis_role(
    axis_role: str,
):
    system = make_test_system()

    with pytest.raises(
        ValueError,
        match="axis_role must be one of",
    ):
        make_sinc_pulse(
            flip_angle=math.pi / 2,
            duration=3e-3,
            time_bw_product=4.0,
            slice_thickness=20e-3,
            axis_role=axis_role,
            system=system,
            return_gz=True,
            name="invalid_axis_rf",
        )


@pytest.mark.parametrize(
    "channel",
    [
        "",
        "read",
        "phase",
        "slice",
        "a",
    ],
)
def test_sinc_pulse_rejects_invalid_explicit_physical_channel(
    channel: str,
):
    system = make_test_system()

    with pytest.raises(
        ValueError,
        match="channel must be one of",
    ):
        make_sinc_pulse(
            flip_angle=math.pi / 2,
            duration=3e-3,
            time_bw_product=4.0,
            slice_thickness=20e-3,
            channel=channel,
            axis_role="slice",
            system=system,
            return_gz=True,
            name="invalid_channel_rf",
        )


# =============================================================================
# Selection-gradient relationships
# =============================================================================


@pytest.mark.parametrize(
    "axis_role",
    [
        "read",
        "phase",
        "slice",
    ],
)
def test_sinc_selection_rephaser_matches_selection_axis_and_half_area(
    axis_role: str,
):
    """The rephasing lobe should remain paired with its selection gradient."""

    system = make_test_system()

    _, gsel, greph = make_sinc_pulse(
        flip_angle=math.pi / 2,
        duration=3e-3,
        time_bw_product=4.0,
        slice_thickness=20e-3,
        axis_role=axis_role,
        system=system,
        return_gz=True,
        name=f"relationship_{axis_role}",
    )

    assert gsel.axis_role == axis_role
    assert greph.axis_role == axis_role

    assert greph.area == pytest.approx(
        -0.5 * gsel.area
    )

    assert (
        greph.metadata["refocuses"]
        == gsel.name
    )

    assert (
        gsel.metadata["slice_refocus_gradient"]
        == greph.name
    )


# =============================================================================
# Metadata
# =============================================================================


@pytest.mark.parametrize(
    "axis_role",
    [
        "read",
        "phase",
        "slice",
    ],
)
def test_sinc_selection_metadata_records_logical_axis(
    axis_role: str,
):
    system = make_test_system()

    rf, gsel, greph = make_sinc_pulse(
        flip_angle=math.pi / 2,
        duration=3e-3,
        time_bw_product=4.0,
        slice_thickness=20e-3,
        axis_role=axis_role,
        system=system,
        return_gz=True,
        name=f"metadata_{axis_role}",
    )

    assert (
        gsel.metadata["logical_axis"]
        == axis_role
    )

    assert (
        gsel.metadata["axis_role"]
        == axis_role
    )

    assert (
        greph.metadata["logical_axis"]
        == axis_role
    )

    assert (
        greph.metadata["axis_role"]
        == axis_role
    )

    assert (
        rf.parameters["logical_axis"]
        == axis_role
    )