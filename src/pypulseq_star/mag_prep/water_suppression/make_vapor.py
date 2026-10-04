from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeAlias

import numpy as np

import pypulseq_star as ppstar

logging.basicConfig(level=logging.INFO)

# =============================================================================
# DEFAULT VAPOR IMPLEMENTATION
# =============================================================================
# Default VAPOR RF and timing parameters follow the 3 T implementation
# described in:
#
#   Öz et al., "Advanced single voxel 1H magnetic resonance spectroscopy
#   techniques in humans: Experts' consensus recommendations",
#   NMR in Biomedicine (2021).
#
#   https://doi.org/10.1002/nbm.4236
#
# The reference 3 T implementation uses:
#
#   - 8 water-suppression RF pulses
#   - relative flip-angle schedule:
#       α, α, 1.78α, α, 1.59α, α, 1.78α, 1.86α
#   - 7 inter-pulse intervals:
#       160, 110, 132, 115, 112, 71, 88 ms
#   - final recovery interval:
#       24 ms
#   - 30 ms SLR water-suppression RF pulse
#   - 42 Hz Mz bandwidth
#
# Published "inter-pulse delays" are interpreted in PyPulseq-Star as
# RF timing-reference to RF timing-reference intervals.
#
# For the default symmetric RF pulse, the timing reference corresponds to
# the temporal midpoint. Arbitrary or asymmetric RF waveforms may provide
# a different effective timing reference.
#
# The default 3 T crusher prescription is based on:
#
#   Zhu H, Ouwerkerk R, Barker PB.
#   "Dual-band water and lipid suppression for MR spectroscopic imaging
#   at 3 Tesla."
#   Magnetic Resonance in Medicine. 2010;63(6):1486-1492.
#
#   https://doi.org/10.1002/mrm.22324
#
# In that implementation, the crusher gradients following VAPOR pulses
# 1 through 7 were 10 ms in duration, while the crusher following pulse 8
# was 5 ms. The reported gradient amplitudes in x/y/z were:
#
#       (10,  0,  0) mT/m
#       ( 0, 10,  0) mT/m
#       ( 0,  0, 10) mT/m
#       ( 9,  0,  0) mT/m
#       ( 0,  8,  0) mT/m
#       ( 0,  0,  7) mT/m
#       ( 8,  0,  0) mT/m
#       ( 0, 10, 10) mT/m
#
# PyPulseq-Star represents crusher schemes by gradient AREA rather than
# gradient amplitude or duration. Therefore, the reference_3T scheme is
# converted to relative gradient-area vectors using:
#
#       1.0 = 10 mT/m * 10 ms
#           = 100 mT/m·ms
#           = 1e-4 T/m·s
#
# giving:
#
#       (1.00, 0.00, 0.00)
#       (0.00, 1.00, 0.00)
#       (0.00, 0.00, 1.00)
#       (0.90, 0.00, 0.00)
#       (0.00, 0.80, 0.00)
#       (0.00, 0.00, 0.70)
#       (0.80, 0.00, 0.00)
#       (0.00, 0.50, 0.50)
#
# The final crusher has relative area 0.5 on each active axis because its
# reported duration is 5 ms rather than 10 ms.
#
# Each crusher is therefore represented as a relative 3-vector in logical
# crusher coordinates:
#
#       (area_read, area_phase, area_slice)
#
# multiplied by a scheme-specific reference area.
#
# Gradient amplitude and duration are realization details. make_vapor()
# should construct the shortest feasible gradient waveform that preserves
# the requested area subject to system Gmax, slew-rate, and gradient-raster
# constraints.
#
# The generic default crusher scheme is "reference_3T", providing a fully
# literature-traceable physical area prescription.
#
# "optimized_3T" contains the relative crusher-area pattern intended for
# the initial PRESS demonstration. Its source values were supplied in M/P/S
# coordinates and are represented using absolute, normalized component
# magnitudes. Its absolute physical area scale remains intentionally
# unspecified until independently confirmed from the source implementation.

Vec3: TypeAlias = tuple[float, float, float]
CrusherAreaVectors: TypeAlias = tuple[Vec3, ...]
CrusherSchemeName: TypeAlias = Literal["reference_3T", "optimized_3T"]


# =============================================================================
# CRUSHER SCHEMES
# =============================================================================


@dataclass(frozen=True, slots=True)
class VaporCrusherScheme:
    """Relative crusher-area prescription for a VAPOR preparation."""

    name: CrusherSchemeName

    # Eight relative gradient-area vectors in logical crusher coordinates.
    relative_areas: CrusherAreaVectors

    # Physical gradient area corresponding to a relative component of 1.0.
    #
    # SI units:
    #     T/m * s
    #
    # None means that the relative pattern is known but its absolute
    # physical scale must be provided explicitly by the caller.
    reference_area_t_per_m_s: float | None

    # Short provenance label for inspection and serialization.
    source: str


# Published 3 T VAPOR crusher scheme.
#
# Original implementation:
#
#   crushers 1–7:
#       maximum amplitude = 10 mT/m
#       duration          = 10 ms
#
#   crusher 8:
#       maximum amplitude = 10 mT/m
#       duration          = 5 ms
#
# Therefore:
#
#       1.0 = 10 mT/m * 10 ms
#           = 100 mT/m·ms
#           = 1e-4 T/m·s
#
# Crusher 8 consequently has relative area components of 0.5 rather than
# 1.0 because its duration is half as long.
REFERENCE_3T_CRUSHERS = VaporCrusherScheme(
    name="reference_3T",
    relative_areas=(
        (1.00, 0.00, 0.00),
        (0.00, 1.00, 0.00),
        (0.00, 0.00, 1.00),
        (0.90, 0.00, 0.00),
        (0.00, 0.80, 0.00),
        (0.00, 0.00, 0.70),
        (0.80, 0.00, 0.00),
        (0.00, 0.50, 0.50),
    ),
    reference_area_t_per_m_s=1e-4,
    source="published_3T_vapor",
)


# Optimized crusher pattern intended for the initial PRESS demonstration.
#
# Source values were supplied in M/P/S coordinates. Their absolute
# component magnitudes are normalized here to the largest supplied value,
# 179.22.
#
# The values are treated as relative crusher-area weights. Their absolute
# physical area scale remains unspecified until confirmed independently.
OPTIMIZED_3T_CRUSHERS = VaporCrusherScheme(
    name="optimized_3T",
    relative_areas=(
        (1.000, 0.000, 0.000),
        (0.000, 0.000, 1.000),
        (0.000, 1.000, 0.000),
        (0.468, 0.000, 0.000),
        (0.000, 0.000, 0.468),
        (0.000, 0.468, 0.000),
        (0.575, 0.000, 0.411),
        (0.179, 0.198, 0.113),
    ),
    reference_area_t_per_m_s=None,
    source="optimized_3T",
)


CRUSHER_SCHEMES: dict[CrusherSchemeName, VaporCrusherScheme] = {
    "reference_3T": REFERENCE_3T_CRUSHERS,
    "optimized_3T": OPTIMIZED_3T_CRUSHERS,
}


# =============================================================================
# RF DEFAULTS
# =============================================================================


@dataclass(frozen=True, slots=True)
class VaporRFDefaults:
    """Reference RF specification for the default 3 T VAPOR implementation."""

    # Consensus 3 T water-suppression pulse specification.
    duration_s: float = 30e-3
    bandwidth_hz: float = 42.0

    # The reference implementation uses an SLR saturation pulse.
    design: str = "slr"

    # rf_waveform=None means that make_vapor() should generate the default
    # consensus-style 3 T waveform.
    source: Literal["consensus_3T"] = "consensus_3T"

    # Design backend used when rf_waveform=None.
    backend: Literal["sigpy"] = "sigpy"

    # Timing reference used by VAPOR timing relationships.
    timing_reference: Literal["rf_center"] = "rf_center"

    # Base pulse should represent the nominal alpha pulse;
    # individual VAPOR RF events scale this waveform according to
    # flip_angle_scale.
    pulse_type: Literal["sat"] = "sat"

    # Number of samples used for the generated default RF waveform.
    # This can be revised once we compare against the reference pulse.
    n_samples: int = 512

    # filter type used for the SLR design. The reference implementation uses a
    # least squares filter. This can be revised once we compare against the reference pulse.
    filter_type: Literal["ls"] = "ls"

    # Passband ripple for the SLR design. The reference implementation uses 0.01. This can be revised once we compare against the reference pulse.
    passband_ripple: float = 0.04

    # stopband ripple for the SLR design. The reference implementation uses 0.01. This can be revised once we compare against the reference pulse.  
    stopband_ripple: float = 0.04


# =============================================================================
# VAPOR DEFAULTS
# =============================================================================


@dataclass(frozen=True, slots=True)
class VaporDefaults:
    """Scientific defaults for the reference 3 T VAPOR preparation."""

    # Nominal flip angle. Individual RF pulses are alpha multiplied by the
    # corresponding relative scale below.
    alpha_deg: float = 90.0

    # Relative flip-angle multipliers for RF pulses 1 through 8.
    flip_angle_scale: tuple[float, ...] = (
        1.00,
        1.00,
        1.78,
        1.00,
        1.59,
        1.00,
        1.78,
        1.86,
    )

    # RF-reference-to-RF-reference intervals:
    #
    #   RF1→RF2, RF2→RF3, ..., RF7→RF8
    #
    # SI units: seconds.
    inter_pulse_delays_s: tuple[float, ...] = (
        160e-3,
        110e-3,
        132e-3,
        115e-3,
        112e-3,
        71e-3,
        88e-3,
    )

    # Interval from the timing reference of RF8 to the end of the VAPOR
    # preparation.
    final_delay_s: float = 24e-3

    # Default RF specification.
    rf: VaporRFDefaults = field(default_factory=VaporRFDefaults)

    # Generic VAPOR users receive the literature-reference crusher scheme.
    crusher_scheme: CrusherSchemeName = "reference_3T"

    @property
    def crushers(self) -> VaporCrusherScheme:
        """Return the selected crusher-area prescription."""
        return CRUSHER_SCHEMES[self.crusher_scheme]


DEFAULT_VAPOR = VaporDefaults()


# =============================================================================
# PROTOCOL PARAMETERS
# =============================================================================
#
# VAPOR parameters are added to the existing sequence Protocol using the
# "vapor_" namespace so that the preparation can be composed with PRESS,
# sLASER, or other sequence kernels without parameter-name collisions.
#
# Scientific schedules that may participate in symbolic relationships are
# stored as individual scalar protocol parameters rather than opaque arrays.
#
# The crusher pattern itself is selected through crusher_scheme and remains
# a structural property of the VAPOR module. The protocol stores only its
# absolute reference area so that crusher strength can remain editable
# without exposing all 8 x 3 crusher coefficients.
#
# rf_waveform is intentionally NOT stored in the Protocol. It is a runtime
# waveform realization supplied to make_vapor(), or generated from the
# default RF specification when rf_waveform=None.


def _add_vapor_protocol_parameters(
    protocol: ppstar.Protocol,
    defaults: VaporDefaults,
    *,
    crusher_reference_area_t_per_m_s: float | None = None,
    rf_frequency_offset_hz: float = 0.0,
) -> None:
    """Add VAPOR-specific parameters to an existing ppstar Protocol."""

    crusher = defaults.crushers

    if crusher_reference_area_t_per_m_s is None:
        crusher_reference_area_t_per_m_s = (
            crusher.reference_area_t_per_m_s
        )

    # For schemes such as optimized_3T, the relative crusher pattern may be
    # known before its absolute physical area scale has been established.
    if crusher_reference_area_t_per_m_s is None:
        raise ValueError(
            f"Crusher scheme {crusher.name!r} does not define an absolute "
            "reference area. Supply crusher_reference_area_t_per_m_s."
        )

    params = protocol.parameters

    # -------------------------------------------------------------------------
    # RF / flip-angle parameters
    # -------------------------------------------------------------------------

    params.setdefault(
        "vapor_alpha_deg",
        defaults.alpha_deg,
    )

    for i, scale in enumerate(defaults.flip_angle_scale, start=1):
        params.setdefault(
            f"vapor_flip_scale_{i}",
            scale,
        )

    # -------------------------------------------------------------------------
    # VAPOR timing parameters
    # -------------------------------------------------------------------------
    #
    # These represent RF timing-reference to RF timing-reference intervals:
    #
    #   RF1→RF2 ... RF7→RF8
    #

    for i, delay_s in enumerate(defaults.inter_pulse_delays_s, start=1):
        params.setdefault(
            f"vapor_interval_{i}_{i + 1}_s",
            delay_s,
        )

    params.setdefault(
        "vapor_final_delay_s",
        defaults.final_delay_s,
    )

    # -------------------------------------------------------------------------
    # Crusher parameters
    # -------------------------------------------------------------------------
    #
    # Relative 8 x 3 crusher areas remain part of the selected CrusherScheme.
    # This parameter establishes the physical area represented by a relative
    # component of 1.0.
    #

    params.setdefault(
        "vapor_crusher_reference_area_t_per_m_s",
        crusher_reference_area_t_per_m_s,
    )

    # -------------------------------------------------------------------------
    # RF specification
    # -------------------------------------------------------------------------
    #
    # These parameters describe the requested RF realization. The actual
    # waveform is generated or loaded separately.
    #

    params.setdefault(
        "vapor_rf_duration_s",
        defaults.rf.duration_s,
    )

    params.setdefault(
        "vapor_rf_bandwidth_hz",
        defaults.rf.bandwidth_hz,
    )

    params.setdefault(
        "vapor_rf_frequency_offset_hz",
        rf_frequency_offset_hz,
    )




# =============================================================================
# RF PULSE CONSTRUCTION
# =============================================================================
#
# VAPOR accepts either:
#
#   1. rf_waveform=None
#      Generate the default 3 T SLR saturation pulse using SigPy.
#
#   2. rf_waveform=<RFWaveform>
#      Use a preconstructed waveform directly.
#
#   3. rf_waveform=<array-like>
#      Interpret the supplied complex samples using rf_dwell_s.
#
# The VAPOR implementation does not contain an SLR design algorithm.
# SigPy is used only when the default waveform must be generated.


@dataclass(frozen=True, slots=True)
class RFWaveform:
    """Backend-neutral RF waveform used by VAPOR."""

    signal: np.ndarray
    dwell_s: float
    time_ref_s: float
    source: str


def _generate_default_vapor_rf(
    rf_defaults: VaporRFDefaults,
) -> RFWaveform:
    """Generate the default 3 T VAPOR SLR saturation pulse."""

    try:
        from sigpy.mri.rf.slr import dzrf
    except ImportError as exc:
        raise ImportError(
            "Generating the default VAPOR RF pulse requires SigPy. "
            "Install PyPulseq-Star with the RF extra:\n\n"
            "    python -m pip install -e '.[rf]'\n\n"
            "or provide rf_waveform explicitly."
        ) from exc

    # Time-bandwidth product for the requested RF specification.
    tb = rf_defaults.duration_s * rf_defaults.bandwidth_hz

    signal = dzrf(
        n=rf_defaults.n_samples,
        tb=tb,
        ptype="sat",
        ftype=rf_defaults.filter_type,
        d1=rf_defaults.passband_ripple,
        d2=rf_defaults.stopband_ripple,
    )

    signal = np.asarray(signal, dtype=np.complex128)

    dwell_s = rf_defaults.duration_s / signal.size

    # Default generated pulse is currently treated as symmetric.
    time_ref_s = 0.5 * rf_defaults.duration_s

    return RFWaveform(
        signal=signal,
        dwell_s=dwell_s,
        time_ref_s=time_ref_s,
        source="sigpy_slr_consensus_3T",
    )


def _resolve_vapor_rf_waveform(
    defaults: VaporDefaults,
    *,
    rf_waveform: RFWaveform | np.ndarray | None = None,
    rf_dwell_s: float | None = None,
    rf_time_ref_s: float | None = None,
) -> RFWaveform:
    """Resolve and validate the RF waveform used by the VAPOR preparation.

    The resolved waveform must be consistent with the requested VAPOR RF
    duration stored in ``defaults.rf.duration_s``.

    Supported inputs are:

    1. ``rf_waveform=None``
       Generate the default consensus-style 3 T SLR waveform.

    2. ``rf_waveform=RFWaveform(...)``
       Use an already normalized backend-neutral waveform.

    3. ``rf_waveform=<array-like>``
       Interpret the supplied complex samples using ``rf_dwell_s``. If
       ``rf_time_ref_s`` is omitted, the temporal midpoint is used.

    In every case, the resolved waveform is validated for:

    - one-dimensional, non-empty signal;
    - finite RF samples;
    - finite, positive dwell time;
    - waveform duration matching the requested VAPOR RF duration;
    - timing reference lying within the realized RF waveform.

    Returns
    -------
    RFWaveform
        Validated backend-neutral RF waveform.

    Raises
    ------
    ValueError
        If the supplied or generated waveform is structurally invalid or is
        inconsistent with the requested VAPOR RF duration/timing.
    """

    # -------------------------------------------------------------------------
    # Default path: generate the reference 3 T waveform with SigPy
    # -------------------------------------------------------------------------

    if rf_waveform is None:
        rf = _generate_default_vapor_rf(
            defaults.rf
        )

    # -------------------------------------------------------------------------
    # Already normalized to our internal representation
    # -------------------------------------------------------------------------

    elif isinstance(
        rf_waveform,
        RFWaveform,
    ):
        rf = rf_waveform

    # -------------------------------------------------------------------------
    # Raw user-supplied complex waveform
    # -------------------------------------------------------------------------

    else:
        signal = np.asarray(
            rf_waveform,
            dtype=np.complex128,
        )

        if signal.ndim != 1:
            raise ValueError(
                "rf_waveform must be a one-dimensional RF waveform."
            )

        if signal.size == 0:
            raise ValueError(
                "rf_waveform must contain at least one sample."
            )

        if rf_dwell_s is None:
            raise ValueError(
                "rf_dwell_s must be provided when rf_waveform is supplied "
                "as an array."
            )

        rf_dwell_s = float(
            rf_dwell_s
        )

        if (
            not math.isfinite(rf_dwell_s)
            or rf_dwell_s <= 0.0
        ):
            raise ValueError(
                "rf_dwell_s must be finite and positive."
            )

        duration_s = (
            signal.size
            * rf_dwell_s
        )

        # For an arbitrary waveform, use the temporal midpoint only when the
        # caller has not supplied a more appropriate effective timing reference.
        if rf_time_ref_s is None:
            rf_time_ref_s = (
                0.5
                * duration_s
            )

        rf = RFWaveform(
            signal=signal,
            dwell_s=rf_dwell_s,
            time_ref_s=float(
                rf_time_ref_s
            ),
            source="user_supplied",
        )

    # =========================================================================
    # COMMON VALIDATION
    # =========================================================================
    #
    # All three input paths must satisfy the same realization contract.
    #
    # In particular:
    #
    #       len(signal) * dwell
    #
    # must describe the same RF duration that make_vapor() subsequently passes
    # to make_arbitrary_rf(). Otherwise the VAPOR timing reference would refer
    # to a different waveform duration than the realized RF event.
    #

    signal = np.asarray(
        rf.signal,
        dtype=np.complex128,
    )

    # -------------------------------------------------------------------------
    # Signal structure
    # -------------------------------------------------------------------------

    if signal.ndim != 1:
        raise ValueError(
            "Resolved VAPOR RF waveform must be one-dimensional."
        )

    if signal.size == 0:
        raise ValueError(
            "Resolved VAPOR RF waveform must contain at least one sample."
        )

    if not np.all(
        np.isfinite(signal)
    ):
        raise ValueError(
            "Resolved VAPOR RF waveform contains non-finite samples."
        )

    # -------------------------------------------------------------------------
    # Dwell time
    # -------------------------------------------------------------------------

    dwell_s = float(
        rf.dwell_s
    )

    if (
        not math.isfinite(dwell_s)
        or dwell_s <= 0.0
    ):
        raise ValueError(
            "Resolved VAPOR RF dwell time must be finite and positive."
        )

    # -------------------------------------------------------------------------
    # Waveform duration
    # -------------------------------------------------------------------------

    waveform_duration_s = (
        signal.size
        * dwell_s
    )

    requested_duration_s = float(
        defaults.rf.duration_s
    )

    if (
        not math.isfinite(requested_duration_s)
        or requested_duration_s <= 0.0
    ):
        raise ValueError(
            "Requested VAPOR RF duration must be finite and positive."
        )

    if not math.isclose(
        waveform_duration_s,
        requested_duration_s,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "VAPOR RF waveform duration does not match the requested "
            "VAPOR RF duration: "
            f"waveform={waveform_duration_s * 1e3:.6f} ms, "
            f"requested={requested_duration_s * 1e3:.6f} ms."
        )

    # -------------------------------------------------------------------------
    # Effective RF timing reference
    # -------------------------------------------------------------------------

    time_ref_s = float(
        rf.time_ref_s
    )

    if not math.isfinite(
        time_ref_s
    ):
        raise ValueError(
            "VAPOR RF timing reference must be finite."
        )

    if not (
        0.0
        <= time_ref_s
        <= waveform_duration_s
    ):
        raise ValueError(
            "VAPOR RF timing reference must lie within the RF waveform: "
            f"time_ref={time_ref_s * 1e3:.6f} ms, "
            f"waveform_duration={waveform_duration_s * 1e3:.6f} ms."
        )

    # -------------------------------------------------------------------------
    # Return normalized representation
    # -------------------------------------------------------------------------

    return RFWaveform(
        signal=signal,
        dwell_s=dwell_s,
        time_ref_s=time_ref_s,
        source=rf.source,
    )
# =============================================================================
# CRUSHER CONSTRUCTION
# =============================================================================
#
# Each VAPOR crusher is specified by gradient AREA:
#
#       reference_area * relative_area[i, :]
#
# rather than by gradient amplitude or duration.
#
# Crusher-area vectors are expressed in PyPulseq-Star logical coordinates:
#
#       (read, phase, slice)
#
# For crusher prescriptions reported in M/P/S coordinates, the convention
# used here is:
#
#       measurement -> read
#       phase       -> phase
#       slice       -> slice
#
# These are NOT scanner x/y/z coordinates. The active EncodingFrame is
# responsible for mapping logical read/phase/slice directions onto physical
# scanner axes during realization/export.
#
# Gradient amplitude and duration are scanner-realization details.
# For each 3-axis crusher, PyPulseq-Star constructs the shortest feasible
# trapezoidal gradient that preserves the requested area subject to:
#
#   - system maximum gradient amplitude, Gmax
#   - system maximum slew rate, Smax
#   - gradient raster time
#
# The three logical components of a crusher share a common duration. The
# component requiring the longest hardware-feasible duration determines that
# duration; the remaining components are realized at amplitudes required to
# preserve their respective gradient areas.
#
# Consequently, crusher gradients are NOT forced to Gmax:
#
#   - small areas may produce triangular gradients with peak < Gmax
#   - larger areas may reach Gmax and include a flat-top interval
#
# Inter-pulse timing is handled separately. The VAPOR timing relationships
# subsequently verify that each realized crusher fits within the available
# RF-reference-to-RF-reference timing interval.
#
# This common area-based representation supports:
#
#   - literature/reference crusher schemes
#   - DOTCOPS-derived schemes
#   - future optimized crusher schemes
#
# without changing the VAPOR sequence structure or tying the crusher
# prescription to a particular scanner orientation.


LOGICAL_CRUSHER_AXES = ("read", "phase", "slice")


def _minimum_trapezoid_duration(
    area_t_per_m_s: float,
    *,
    max_grad_t_per_m: float,
    max_slew_t_per_m_per_s: float,
) -> float:
    """Return the shortest continuous-time trapezoid for a gradient area."""

    area = abs(area_t_per_m_s)

    if area == 0.0:
        return 0.0

    # Largest area achievable by a triangular gradient whose peak amplitude
    # is exactly Gmax:
    #
    #       A_limit = Gmax^2 / Smax
    #
    triangular_limit = (
        max_grad_t_per_m**2
        / max_slew_t_per_m_per_s
    )

    if area <= triangular_limit:
        # Triangular gradient:
        #
        #       A      = G_peak^2 / Smax
        #       G_peak = sqrt(A * Smax)
        #       T      = 2 * G_peak / Smax
        #
        peak_grad = (
            area * max_slew_t_per_m_per_s
        ) ** 0.5

        return (
            2.0
            * peak_grad
            / max_slew_t_per_m_per_s
        )

    # Trapezoidal gradient reaching Gmax.
    rise_time_s = (
        max_grad_t_per_m
        / max_slew_t_per_m_per_s
    )

    # Combined area contributed by the rising and falling ramps.
    ramp_area_t_per_m_s = (
        max_grad_t_per_m
        * rise_time_s
    )

    flat_time_s = (
        area - ramp_area_t_per_m_s
    ) / max_grad_t_per_m

    return (
        2.0 * rise_time_s
        + flat_time_s
    )


def _round_up_to_raster(
    duration_s: float,
    raster_s: float,
) -> float:
    """Round a duration upward to the gradient raster."""

    if duration_s <= 0.0:
        return 0.0

    return (
        math.ceil(duration_s / raster_s)
        * raster_s
    )


def _crusher_area_vectors(
    crusher: VaporCrusherScheme,
    *,
    reference_area_t_per_m_s: float,
) -> tuple[Vec3, ...]:
    """Return physical crusher areas in logical read/phase/slice coordinates.

    The returned vectors have the ordering:

        (area_read, area_phase, area_slice)

    and remain independent of scanner x/y/z orientation.
    """

    return tuple(
        tuple(
            reference_area_t_per_m_s * component
            for component in relative_area
        )
        for relative_area in crusher.relative_areas
    )


def _crusher_duration(
    area_logical_t_per_m_s: Vec3,
    *,
    max_grad_t_per_m: float,
    max_slew_t_per_m_per_s: float,
    grad_raster_s: float,
) -> float:
    """Return the common minimum duration for one logical 3-axis crusher.

    Parameters
    ----------
    area_logical_t_per_m_s
        Gradient-area vector ordered as:

            (read, phase, slice)

        These are logical PyPulseq-Star axes, not scanner x/y/z axes.
    """

    if len(area_logical_t_per_m_s) != len(LOGICAL_CRUSHER_AXES):
        raise ValueError(
            "A VAPOR crusher must contain exactly three logical components "
            "(read, phase, slice)."
        )

    axis_durations_s = (
        _minimum_trapezoid_duration(
            area,
            max_grad_t_per_m=max_grad_t_per_m,
            max_slew_t_per_m_per_s=max_slew_t_per_m_per_s,
        )
        for area in area_logical_t_per_m_s
    )

    duration_s = max(axis_durations_s)

    duration_s = _round_up_to_raster(
        duration_s,
        grad_raster_s,
    )

    # Leave one raster of realization margin so downstream trapezoid
    # construction cannot exceed Gmax because of independent rasterization
    # of rise/flat/fall times.
    if duration_s > 0.0:
        duration_s += grad_raster_s

    return duration_s
# =============================================================================
# VAPOR TIMING RELATIONSHIPS
# =============================================================================
#
# VAPOR timing is defined using RF timing-reference to RF timing-reference
# intervals.
#
# For the default symmetric RF waveform, the timing reference is the RF
# midpoint. Arbitrary/asymmetric RF waveforms may define a different
# effective timing reference.
#
# The published VAPOR intervals therefore establish the invariant:
#
#       RF_ref[i + 1] - RF_ref[i] = interval[i]
#
# rather than an RF-end-to-RF-start delay.
#
# Once the RF-reference positions are fixed:
#
#       RF_start[i] = RF_ref[i] - rf_time_ref
#       RF_end[i]   = RF_start[i] + rf_duration
#
# Each crusher is placed immediately after its associated RF event:
#
#       crusher_start[i] = RF_end[i]
#       crusher_end[i]   = crusher_start[i] + crusher_duration[i]
#
# The remaining interval before the next RF is therefore:
#
#       residual_delay[i]
#           = RF_start[i + 1] - crusher_end[i]
#
# For RF8:
#
#       vapor_end = RF_ref[8] + final_delay
#
# and:
#
#       final_residual_delay = vapor_end - crusher_end[8]
#
# Any negative residual delay means that the requested RF/crusher
# realization cannot fit within the published VAPOR timing schedule and
# construction must fail rather than silently alter the timing.
#
# IMPORTANT:
# rf_duration_s and rf_time_ref_s must describe the ACTUAL realized RF
# event timing used by the sequence. If the RF event includes additional
# delay/dead/ringdown time, its timing reference must be expressed relative
# to that realized event start so those contributions are not omitted or
# counted twice.


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VaporPulseTiming:
    """Resolved timing for one RF/crusher pair in the VAPOR train."""

    index: int

    rf_start_s: float
    rf_ref_s: float
    rf_end_s: float

    crusher_start_s: float
    crusher_end_s: float
    crusher_duration_s: float

    # Delay after this crusher.
    #
    # Pulses 1-7:
    #     crusher end -> next RF start
    #
    # Pulse 8:
    #     crusher end -> end of VAPOR preparation
    residual_delay_s: float


@dataclass(frozen=True, slots=True)
class VaporTimingPlan:
    """Fully resolved timing plan for a VAPOR preparation."""

    pulses: tuple[VaporPulseTiming, ...]

    preparation_start_s: float
    preparation_end_s: float

    @property
    def duration_s(self) -> float:
        """Total VAPOR preparation duration."""
        return self.preparation_end_s - self.preparation_start_s


def _build_vapor_timing_plan(
    *,
    rf_duration_s: float,
    rf_time_ref_s: float,
    crusher_durations_s: tuple[float, ...],
    inter_pulse_intervals_s: tuple[float, ...],
    final_delay_s: float,
    tolerance_s: float = 1e-9,
) -> VaporTimingPlan:
    """Resolve and validate the complete VAPOR timing schedule.

    Parameters
    ----------
    rf_duration_s
        Duration of the realized RF event.

    rf_time_ref_s
        Effective RF timing reference measured from the start of the
        realized RF event.

    crusher_durations_s
        Realized durations of crushers 1 through 8.

    inter_pulse_intervals_s
        RF-reference-to-RF-reference intervals for RF1→RF2 through RF7→RF8.

    final_delay_s
        Interval from the RF8 timing reference to the end of VAPOR.

    tolerance_s
        Numerical tolerance used when determining whether an event exceeds
        the available timing interval.

    Returns
    -------
    VaporTimingPlan
        Fully resolved VAPOR timing plan.

    Raises
    ------
    ValueError
        If the RF timing reference is invalid, the number of timing
        parameters is incorrect, or any crusher cannot fit within the
        requested VAPOR timing schedule.
    """

    # -------------------------------------------------------------------------
    # Validate input structure
    # -------------------------------------------------------------------------

    if rf_duration_s <= 0.0:
        raise ValueError("rf_duration_s must be positive.")

    if not 0.0 <= rf_time_ref_s <= rf_duration_s:
        raise ValueError(
            "rf_time_ref_s must lie within the realized RF event."
        )

    if len(inter_pulse_intervals_s) != 7:
        raise ValueError(
            "VAPOR requires exactly 7 RF-reference-to-RF-reference intervals."
        )

    if len(crusher_durations_s) != 8:
        raise ValueError(
            "VAPOR requires exactly 8 crusher durations."
        )

    if any(delay <= 0.0 for delay in inter_pulse_intervals_s):
        raise ValueError(
            "All VAPOR inter-pulse intervals must be positive."
        )

    if final_delay_s <= 0.0:
        raise ValueError("final_delay_s must be positive.")

    if any(duration < 0.0 for duration in crusher_durations_s):
        raise ValueError(
            "Crusher durations must be non-negative."
        )

    # -------------------------------------------------------------------------
    # Establish RF timing-reference positions
    # -------------------------------------------------------------------------
    #
    # RF1 begins at t = 0.
    #
    # Its timing reference therefore occurs at rf_time_ref_s.
    # All subsequent timing references are determined exclusively by the
    # requested VAPOR inter-pulse schedule.
    #

    rf_refs_s = [rf_time_ref_s]

    for interval_s in inter_pulse_intervals_s:
        rf_refs_s.append(
            rf_refs_s[-1] + interval_s
        )

    rf_starts_s = [
        rf_ref_s - rf_time_ref_s
        for rf_ref_s in rf_refs_s
    ]

    rf_ends_s = [
        rf_start_s + rf_duration_s
        for rf_start_s in rf_starts_s
    ]

    preparation_start_s = 0.0

    # Published final delay is interpreted relative to the RF8 timing
    # reference, not relative to RF8 end.
    preparation_end_s = rf_refs_s[-1] + final_delay_s

    # -------------------------------------------------------------------------
    # Resolve crusher placement and residual delays
    # -------------------------------------------------------------------------

    pulse_timings: list[VaporPulseTiming] = []

    for i in range(8):
        crusher_start_s = rf_ends_s[i]
        crusher_end_s = (
            crusher_start_s + crusher_durations_s[i]
        )

        if i < 7:
            available_end_s = rf_starts_s[i + 1]
        else:
            available_end_s = preparation_end_s

        residual_delay_s = available_end_s - crusher_end_s

        # Permit only tiny floating-point excursions around zero.
        if residual_delay_s < -tolerance_s:
            if i < 7:
                target = f"RF{i + 2}"
            else:
                target = "end of VAPOR"

            raise ValueError(
                f"VAPOR crusher {i + 1} does not fit before {target}. "
                f"Crusher ends at {crusher_end_s * 1e3:.6f} ms, "
                f"but the available interval ends at "
                f"{available_end_s * 1e3:.6f} ms "
                f"({-residual_delay_s * 1e3:.6f} ms overrun)."
            )

        # Remove numerically insignificant negative zero/slack.
        if abs(residual_delay_s) <= tolerance_s:
            residual_delay_s = 0.0

        pulse_timings.append(
            VaporPulseTiming(
                index=i + 1,
                rf_start_s=rf_starts_s[i],
                rf_ref_s=rf_refs_s[i],
                rf_end_s=rf_ends_s[i],
                crusher_start_s=crusher_start_s,
                crusher_end_s=crusher_end_s,
                crusher_duration_s=crusher_durations_s[i],
                residual_delay_s=residual_delay_s,
            )
        )

    plan = VaporTimingPlan(
        pulses=tuple(pulse_timings),
        preparation_start_s=preparation_start_s,
        preparation_end_s=preparation_end_s,
    )

    _log_vapor_timing_plan(
        plan,
        inter_pulse_intervals_s=inter_pulse_intervals_s,
        final_delay_s=final_delay_s,
    )

    return plan


def _log_vapor_timing_plan(
    plan: VaporTimingPlan,
    *,
    inter_pulse_intervals_s: tuple[float, ...],
    final_delay_s: float,
) -> None:
    """Emit detailed VAPOR timing diagnostics at DEBUG level."""

    if not logger.isEnabledFor(logging.DEBUG):
        return

    logger.debug(
        "Resolved VAPOR timing plan: total duration = %.6f ms",
        plan.duration_s * 1e3,
    )

    logger.debug(
        "VAPOR timing convention: RF timing-reference -> "
        "RF timing-reference"
    )

    for pulse in plan.pulses:
        logger.debug(
            (
                "VAPOR RF%d: "
                "start=%10.6f ms | "
                "ref=%10.6f ms | "
                "end=%10.6f ms | "
                "crusher=%10.6f -> %10.6f ms | "
                "crusher_duration=%9.6f ms | "
                "residual=%9.6f ms"
            ),
            pulse.index,
            pulse.rf_start_s * 1e3,
            pulse.rf_ref_s * 1e3,
            pulse.rf_end_s * 1e3,
            pulse.crusher_start_s * 1e3,
            pulse.crusher_end_s * 1e3,
            pulse.crusher_duration_s * 1e3,
            pulse.residual_delay_s * 1e3,
        )

    # -------------------------------------------------------------------------
    # Independently reconstruct and verify every requested RF-reference
    # interval. This is intentionally redundant because timing errors here
    # directly affect VAPOR simulation and sequence behavior.
    # -------------------------------------------------------------------------

    for i, requested_s in enumerate(
        inter_pulse_intervals_s
    ):
        actual_s = (
            plan.pulses[i + 1].rf_ref_s
            - plan.pulses[i].rf_ref_s
        )

        error_s = actual_s - requested_s

        logger.debug(
            (
                "VAPOR interval RF%d->RF%d: "
                "requested=%10.6f ms | "
                "realized=%10.6f ms | "
                "error=%+.3f us"
            ),
            i + 1,
            i + 2,
            requested_s * 1e3,
            actual_s * 1e3,
            error_s * 1e6,
        )

    actual_final_delay_s = (
        plan.preparation_end_s
        - plan.pulses[-1].rf_ref_s
    )

    logger.debug(
        (
            "VAPOR final interval: "
            "requested=%10.6f ms | "
            "realized=%10.6f ms | "
            "error=%+.3f us"
        ),
        final_delay_s * 1e3,
        actual_final_delay_s * 1e3,
        (actual_final_delay_s - final_delay_s) * 1e6,
    )


# =============================================================================
# VAPOR NODE CONSTRUCTION
# =============================================================================
#
# Construct the VAPOR preparation directly on the PyPulseq-Star timeline:
#
#   vapor
#       pulse_1
#       crusher_1
#       delay_1
#       pulse_2
#       crusher_2
#       delay_2
#       ...
#       pulse_8
#       crusher_8
#       final_fill
#
# PyPulseq-Star hierarchy is represented through dotted node paths supplied
# to seq.add_block(..., node=...). No separate seq.add_node(...) call is
# required.
#
# Each RF pulse occupies one block.
#
# Each crusher occupies one block containing its simultaneously executed
# logical read / phase / slice gradient components. The node-construction
# code is intentionally agnostic to scanner x/y/z geometry; logical-axis
# semantics are carried by the gradient events themselves.
#
# For pulses 1-7, the following delay block contains the residual timing
# required to place the NEXT RF timing reference at the requested
# RF-reference-to-RF-reference interval.
#
# After RF8, final_fill contains the residual timing required to satisfy:
#
#       VAPOR_end - RF8_reference = vapor_final_delay_s
#
# Therefore vapor.final_fill is a scanner realization detail and is
# generally NOT numerically equal to vapor_final_delay_s.
#
# The explicit sequence timeline is:
#
#       RF1 -> crusher1 -> delay1
#       RF2 -> crusher2 -> delay2
#       ...
#       RF8 -> crusher8 -> final_fill
#
# This hierarchy is retained for inspection, validation, and backend
# lowering, including future gammaSTAR .seq.json export.


def _add_vapor_node(
    seq,
    *,
    node: str,
    rf_events: tuple,
    crusher_events: tuple[tuple, ...],
    timing_plan: VaporTimingPlan,
    system,
) -> tuple:
    """Add a resolved VAPOR preparation to a PyPulseq-Star sequence.

    Parameters
    ----------
    seq
        Existing PyPulseq-Star sequence.

    rf_events
        Eight realized RF events, already scaled to the requested VAPOR
        flip-angle schedule.

    crusher_events
        Eight tuples of logical gradient events. Each tuple corresponds to
        one crusher and may contain read, phase, and/or slice components.

        Components with zero area may be represented as None and are omitted
        from the executable block.

    timing_plan
        Fully resolved VAPOR timing plan produced by
        _build_vapor_timing_plan().

    system
        Sequence hardware specification used when constructing delay events.

    Returns
    -------
    tuple
        Sequence blocks belonging to the VAPOR preparation, in timeline
        order. Returning the blocks is useful for validation and debugging;
        the blocks are already part of ``seq``.

    Notes
    -----
    This function establishes semantic hierarchy and block order only.

    Scientific timing is defined by ``VaporTimingPlan`` and should be
    independently checked against the realized sequence after relationship
    resolution and rasterization.

    Crusher gradients are assumed to already carry their logical
    ``axis_role`` metadata. This function never assigns physical x/y/z
    channels.
    """

    # -------------------------------------------------------------------------
    # Validate construction inputs
    # -------------------------------------------------------------------------

    if len(rf_events) != 8:
        raise ValueError(
            "VAPOR requires exactly 8 RF events; "
            f"got {len(rf_events)}."
        )

    if len(crusher_events) != 8:
        raise ValueError(
            "VAPOR requires exactly 8 crusher event groups; "
            f"got {len(crusher_events)}."
        )

    if len(timing_plan.pulses) != 8:
        raise ValueError(
            "VAPOR timing plan must contain exactly 8 pulse timings; "
            f"got {len(timing_plan.pulses)}."
        )

    logger.debug(
        "Constructing VAPOR hierarchy with 8 RF/crusher pairs."
    )

    vapor_blocks = []

    # -------------------------------------------------------------------------
    # RF1 -> crusher1 -> delay1 ... RF8 -> crusher8 -> final_fill
    # -------------------------------------------------------------------------

    for i in range(8):
        pulse_number = i + 1
        timing = timing_plan.pulses[i]

        # ---------------------------------------------------------------------
        # RF pulse
        # ---------------------------------------------------------------------

        rf_event = rf_events[i]

        pulse_block = seq.add_block(
            rf_event,
            name=f"vapor_pulse_{pulse_number}",
            role="water_suppression_rf",
            node=f"{node}.pulse_{pulse_number}",
        )

        vapor_blocks.append(pulse_block)

        logger.debug(
            "Added vapor.pulse_%d: "
            "start=%.6f ms | ref=%.6f ms | end=%.6f ms",
            pulse_number,
            timing.rf_start_s * 1e3,
            timing.rf_ref_s * 1e3,
            timing.rf_end_s * 1e3,
        )

        # ---------------------------------------------------------------------
        # Crusher
        # ---------------------------------------------------------------------
        #
        # The crusher tuple contains logical read/phase/slice gradient events.
        # Zero-area components are represented as None and are omitted.
        #
        # All nonzero logical components are added to ONE block so they execute
        # simultaneously and remain one semantic crusher operation.
        #

        crusher_axis_events = tuple(
            event
            for event in crusher_events[i]
            if event is not None
        )

        if crusher_axis_events:
            crusher_block = seq.add_block(
                *crusher_axis_events,
                name=f"vapor_crusher_{pulse_number}",
                role="water_suppression_crusher",
                node=f"{node}.crusher_{pulse_number}",
            )

            vapor_blocks.append(crusher_block)

            logger.debug(
                "Added vapor.crusher_%d: "
                "components=%d | "
                "start=%.6f ms | end=%.6f ms | duration=%.6f ms",
                pulse_number,
                len(crusher_axis_events),
                timing.crusher_start_s * 1e3,
                timing.crusher_end_s * 1e3,
                timing.crusher_duration_s * 1e3,
            )

            if logger.isEnabledFor(logging.DEBUG):
                logical_axes = tuple(
                    getattr(event, "axis_role", None)
                    for event in crusher_axis_events
                )

                logger.debug(
                    "VAPOR crusher %d logical axes: %s",
                    pulse_number,
                    logical_axes,
                )

        else:
            logger.debug(
                "VAPOR crusher %d has zero area on all logical axes; "
                "no crusher block added.",
                pulse_number,
            )

        # ---------------------------------------------------------------------
        # Residual timing fill
        # ---------------------------------------------------------------------

        residual_delay_s = timing.residual_delay_s

        if residual_delay_s < 0.0:
            # _build_vapor_timing_plan() should already prevent this.
            # Keep the check here so invalid timing can never enter the
            # executable sequence silently.
            raise ValueError(
                f"VAPOR pulse {pulse_number} has negative residual delay: "
                f"{residual_delay_s * 1e3:.6f} ms."
            )

        if residual_delay_s == 0.0:
            logger.debug(
                "VAPOR pulse %d requires no residual timing-fill block.",
                pulse_number,
            )
            continue

        if pulse_number < 8:
            delay_name = f"vapor_delay_{pulse_number}"
            delay_role = "water_suppression_timing_fill"
            delay_node = f"{node}.delay_{pulse_number}"
        else:
            delay_name = "vapor_final_fill"
            delay_role = "water_suppression_final_fill"
            delay_node = f"{node}.final_fill"

        delay_event = ppstar.make_delay(
            residual_delay_s,
            system=system,
            name=delay_name,
            role=delay_role,
        )

        delay_block = seq.add_block(
            delay_event,
            name=delay_name,
            role=delay_role,
            node=delay_node,
        )

        vapor_blocks.append(delay_block)

        if pulse_number < 8:
            logger.debug(
                "Added vapor.delay_%d: %.6f ms "
                "(crusher%d end -> RF%d start)",
                pulse_number,
                residual_delay_s * 1e3,
                pulse_number,
                pulse_number + 1,
            )
        else:
            logger.debug(
                "Added vapor.final_fill: %.6f ms "
                "(crusher8 end -> VAPOR end)",
                residual_delay_s * 1e3,
            )

    logger.debug(
        "VAPOR hierarchy construction complete: "
        "blocks=%d | requested duration=%.6f ms",
        len(vapor_blocks),
        timing_plan.duration_s * 1e3,
    )

    return tuple(vapor_blocks)

# =============================================================================
# VALIDATION
# =============================================================================
#
# VAPOR validation deliberately compares:
#
#   scientific request
#       ↓
#   realized PyPulseq-Star events
#
# rather than silently modifying the requested preparation to make it fit.
#
# Validation covers:
#
#   - exactly 8 RF pulses
#   - exactly 8 crusher prescriptions
#   - exactly 8 timing entries:
#         7 RF-reference-to-RF-reference intervals
#         1 RF8-reference-to-VAPOR-end interval
#   - crusher vector shape == (8, 3)
#   - non-negative / physically meaningful timing
#   - RF raster compatibility
#   - gradient raster compatibility
#   - gradient amplitude / slew feasibility
#   - RF peak amplitude feasibility
#   - requested versus realized RF flip-angle consistency
#   - requested versus realized RF duration consistency
#   - crusher duration consistency across logical components
#
# Validation is intentionally non-destructive. Invalid protocol
# relationships are reported; requested values are never silently changed.


def _is_on_raster(
    value_s: float,
    raster_s: float,
    *,
    atol_s: float = 1e-12,
) -> bool:
    """Return True when a duration lies on the requested raster."""

    if raster_s <= 0.0:
        return False

    n = value_s / raster_s

    return math.isclose(
        n,
        round(n),
        rel_tol=0.0,
        abs_tol=atol_s / raster_s,
    )


def _event_float(
    event,
    key: str,
) -> float | None:
    """Return a numeric value from an event or its parameter dictionary."""

    value = getattr(
        event,
        key,
        None,
    )

    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    parameters = getattr(
        event,
        "parameters",
        None,
    )

    if isinstance(parameters, dict):
        value = parameters.get(
            key
        )

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return None


def _rf_peak_amplitude_t(
    rf_event,
) -> float | None:
    """Return realized RF peak B1 amplitude in tesla, when available."""

    amplitude_t = _event_float(
        rf_event,
        "amplitude_t",
    )

    if amplitude_t is not None:
        return amplitude_t

    metadata = getattr(
        rf_event,
        "metadata",
        None,
    )

    if not isinstance(metadata, dict):
        return None

    rf_shape = metadata.get(
        "rf_shape"
    )

    if isinstance(rf_shape, dict):
        value = rf_shape.get(
            "amplitude_t"
        )

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return None


def _validate_vapor(
    *,
    system,
    crusher: VaporCrusherScheme,
    crusher_areas: tuple[Vec3, ...],
    rf_events: tuple,
    crusher_events: tuple[tuple, ...],
    timing_plan: VaporTimingPlan,
    alpha_deg: float,
    flip_angle_scale: tuple[float, ...],
    rf_duration_s: float,
    inter_pulse_intervals_s: tuple[float, ...],
    final_delay_s: float,
) -> None:
    """Validate a fully constructed VAPOR preparation.

    Validation compares requested scientific parameters with the realized
    PyPulseq-Star RF and gradient events.

    All detected problems are accumulated and reported together. No protocol
    value or event is modified by this function.
    """

    errors: list[str] = []

    rf_raster_s = float(
        system.rf_raster_time
    )

    grad_raster_s = float(
        system.grad_raster_time
    )

    max_rf_t = float(
        system.max_rf
    )

    max_grad = float(
        system.max_grad
    )

    max_slew = float(
        system.max_slew
    )

    # -------------------------------------------------------------------------
    # Structural validation
    # -------------------------------------------------------------------------

    if len(rf_events) != 8:
        errors.append(
            "VAPOR requires exactly 8 RF events; "
            f"got {len(rf_events)}."
        )

    if len(flip_angle_scale) != 8:
        errors.append(
            "VAPOR requires exactly 8 RF flip-angle scale factors; "
            f"got {len(flip_angle_scale)}."
        )

    if len(crusher.relative_areas) != 8:
        errors.append(
            "VAPOR crusher scheme requires exactly 8 relative-area vectors; "
            f"got {len(crusher.relative_areas)}."
        )

    if len(crusher_areas) != 8:
        errors.append(
            "VAPOR requires exactly 8 realized crusher-area vectors; "
            f"got {len(crusher_areas)}."
        )

    if len(crusher_events) != 8:
        errors.append(
            "VAPOR requires exactly 8 crusher event groups; "
            f"got {len(crusher_events)}."
        )

    if len(inter_pulse_intervals_s) != 7:
        errors.append(
            "VAPOR requires exactly 7 RF-reference-to-RF-reference "
            "intervals; "
            f"got {len(inter_pulse_intervals_s)}."
        )

    if len(timing_plan.pulses) != 8:
        errors.append(
            "VAPOR timing plan requires exactly 8 RF/crusher timing entries; "
            f"got {len(timing_plan.pulses)}."
        )

    # Seven inter-pulse intervals + one final RF8-reference interval.
    requested_timing_entries = (
        len(inter_pulse_intervals_s)
        + 1
    )

    if requested_timing_entries != 8:
        errors.append(
            "VAPOR requires exactly 8 requested timing entries "
            "(7 inter-pulse intervals + 1 final interval); "
            f"got {requested_timing_entries}."
        )

    # -------------------------------------------------------------------------
    # Crusher vector shape
    # -------------------------------------------------------------------------

    for i, relative_area in enumerate(
        crusher.relative_areas,
        start=1,
    ):
        if len(relative_area) != 3:
            errors.append(
                f"VAPOR crusher {i} relative-area vector must contain "
                "exactly 3 logical components "
                "(read, phase, slice); "
                f"got {len(relative_area)}."
            )

    for i, area_vector in enumerate(
        crusher_areas,
        start=1,
    ):
        if len(area_vector) != 3:
            errors.append(
                f"VAPOR crusher {i} physical-area vector must contain "
                "exactly 3 logical components "
                "(read, phase, slice); "
                f"got {len(area_vector)}."
            )

    # -------------------------------------------------------------------------
    # Requested timing relationships
    # -------------------------------------------------------------------------

    if rf_duration_s <= 0.0:
        errors.append(
            "VAPOR RF duration must be positive; "
            f"got {rf_duration_s:.9g} s."
        )

    for i, interval_s in enumerate(
        inter_pulse_intervals_s,
        start=1,
    ):
        if interval_s <= 0.0:
            errors.append(
                f"VAPOR RF-reference interval {i}->{i + 1} must "
                f"be positive; got {interval_s:.9g} s."
            )

    if final_delay_s <= 0.0:
        errors.append(
            "VAPOR final RF8-reference-to-end interval must be positive; "
            f"got {final_delay_s:.9g} s."
        )

    for timing in timing_plan.pulses:
        if timing.rf_start_s < 0.0:
            errors.append(
                f"VAPOR RF{timing.index} starts before preparation time zero: "
                f"{timing.rf_start_s:.9g} s."
            )

        if timing.rf_ref_s < timing.rf_start_s:
            errors.append(
                f"VAPOR RF{timing.index} timing reference occurs before "
                "RF start."
            )

        if timing.rf_ref_s > timing.rf_end_s:
            errors.append(
                f"VAPOR RF{timing.index} timing reference occurs after "
                "RF end."
            )

        if timing.crusher_start_s < timing.rf_end_s:
            errors.append(
                f"VAPOR crusher {timing.index} starts before RF"
                f"{timing.index} has ended."
            )

        if timing.crusher_end_s < timing.crusher_start_s:
            errors.append(
                f"VAPOR crusher {timing.index} has negative duration."
            )

        if timing.residual_delay_s < 0.0:
            errors.append(
                f"VAPOR timing after crusher {timing.index} is infeasible: "
                f"residual delay = "
                f"{timing.residual_delay_s * 1e3:.6f} ms."
            )

    if timing_plan.preparation_end_s <= timing_plan.preparation_start_s:
        errors.append(
            "VAPOR preparation duration must be positive."
        )

    # -------------------------------------------------------------------------
    # RF raster and flip-angle validation
    # -------------------------------------------------------------------------

    if rf_raster_s <= 0.0:
        errors.append(
            f"RF raster time must be positive; got {rf_raster_s:.9g} s."
        )

    for i, rf_event in enumerate(
        rf_events,
        start=1,
    ):
        realized_duration_s = _event_float(
            rf_event,
            "duration",
        )

        if realized_duration_s is None:
            errors.append(
                f"VAPOR RF{i} does not expose a realized duration."
            )
        else:
            if not _is_on_raster(
                realized_duration_s,
                rf_raster_s,
            ):
                errors.append(
                    f"VAPOR RF{i} duration "
                    f"{realized_duration_s * 1e3:.6f} ms is not aligned "
                    f"to the RF raster "
                    f"{rf_raster_s * 1e6:.6f} us."
                )

            # Do not silently accept constructor rasterization that changes
            # the scientifically requested RF duration.
            if not math.isclose(
                realized_duration_s,
                rf_duration_s,
                rel_tol=0.0,
                abs_tol=max(
                    1e-12,
                    rf_raster_s * 1e-6,
                ),
            ):
                errors.append(
                    f"VAPOR RF{i} requested duration "
                    f"{rf_duration_s * 1e3:.6f} ms but realized "
                    f"{realized_duration_s * 1e3:.6f} ms. "
                    "Requested RF duration is not RF-raster compatible."
                )

        if i <= len(flip_angle_scale):
            requested_flip_rad = math.radians(
                alpha_deg
                * flip_angle_scale[i - 1]
            )

            realized_flip_rad = _event_float(
                rf_event,
                "flip_angle",
            )

            if realized_flip_rad is None:
                errors.append(
                    f"VAPOR RF{i} does not expose a realized flip angle."
                )

            elif not math.isclose(
                realized_flip_rad,
                requested_flip_rad,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                errors.append(
                    f"VAPOR RF{i} flip-angle mismatch: "
                    f"requested "
                    f"{math.degrees(requested_flip_rad):.6f} deg, "
                    f"realized "
                    f"{math.degrees(realized_flip_rad):.6f} deg."
                )

        peak_b1_t = _rf_peak_amplitude_t(
            rf_event
        )

        if peak_b1_t is None:
            errors.append(
                f"VAPOR RF{i} does not expose realized peak B1 amplitude."
            )

        elif peak_b1_t > max_rf_t * (
            1.0 + 1e-12
        ):
            errors.append(
                f"VAPOR RF{i} exceeds system.max_rf: "
                f"peak B1={peak_b1_t:.9g} T, "
                f"max_rf={max_rf_t:.9g} T."
            )

        # Ask the RF shape itself to validate its raster and amplitude
        # representation as an independent consistency check.
        shape = getattr(
            rf_event,
            "shape",
            None,
        )

        if shape is None:
            errors.append(
                f"VAPOR RF{i} has no RF shape."
            )

        elif hasattr(
            shape,
            "to_gammastar_samples",
        ):
            realized_flip_rad = _event_float(
                rf_event,
                "flip_angle",
            )

            if realized_flip_rad is not None:
                try:
                    shape.to_gammastar_samples(
                        flip_angle=realized_flip_rad,
                        gamma_hz_per_t=float(
                            system.gamma
                        ),
                        max_rf=max_rf_t,
                    )
                except (ValueError, TypeError) as exc:
                    errors.append(
                        f"VAPOR RF{i} waveform validation failed: {exc}"
                    )

    # -------------------------------------------------------------------------
    # Gradient raster / amplitude / slew validation
    # -------------------------------------------------------------------------

    if grad_raster_s <= 0.0:
        errors.append(
            "Gradient raster time must be positive; "
            f"got {grad_raster_s:.9g} s."
        )

    for crusher_index, event_group in enumerate(
        crusher_events,
        start=1,
    ):
        nonzero_events = tuple(
            event
            for event in event_group
            if event is not None
        )

        # A crusher may theoretically have zero area on every logical axis,
        # although the standard VAPOR schemes normally do not.
        if not nonzero_events:
            continue

        realized_durations = []

        for gradient in nonzero_events:
            logical_axis = getattr(
                gradient,
                "axis_role",
                None,
            )

            if logical_axis not in {
                "read",
                "phase",
                "slice",
            }:
                errors.append(
                    f"VAPOR crusher {crusher_index} contains a gradient "
                    f"with invalid logical axis {logical_axis!r}."
                )

            duration_s = _event_float(
                gradient,
                "duration",
            )

            if duration_s is None:
                errors.append(
                    f"VAPOR crusher {crusher_index} "
                    f"{logical_axis!r} gradient does not expose duration."
                )
            else:
                realized_durations.append(
                    duration_s
                )

                if not _is_on_raster(
                    duration_s,
                    grad_raster_s,
                ):
                    errors.append(
                        f"VAPOR crusher {crusher_index} "
                        f"{logical_axis!r} duration "
                        f"{duration_s * 1e3:.6f} ms is not aligned "
                        f"to gradient raster "
                        f"{grad_raster_s * 1e6:.6f} us."
                    )

            shape = getattr(
                gradient,
                "shape",
                None,
            )

            if shape is None:
                errors.append(
                    f"VAPOR crusher {crusher_index} "
                    f"{logical_axis!r} gradient has no shape."
                )
                continue

            # SeqStar gradient shapes already implement scanner-limit
            # validation. Calling it here makes VAPOR validation explicit
            # rather than relying only on constructor-time checks.
            if hasattr(
                shape,
                "validate",
            ):
                try:
                    shape.validate(
                        max_grad=max_grad,
                        max_slew=max_slew,
                    )
                except (ValueError, TypeError) as exc:
                    errors.append(
                        f"VAPOR crusher {crusher_index} "
                        f"{logical_axis!r} gradient violates hardware "
                        f"limits: {exc}"
                    )

        # Every nonzero logical component of one crusher must have exactly
        # the same realized duration so the vector executes simultaneously.
        if realized_durations:
            reference_duration_s = (
                realized_durations[0]
            )

            for duration_s in realized_durations[1:]:
                if not math.isclose(
                    duration_s,
                    reference_duration_s,
                    rel_tol=0.0,
                    abs_tol=max(
                        1e-12,
                        grad_raster_s * 1e-6,
                    ),
                ):
                    errors.append(
                        f"VAPOR crusher {crusher_index} logical components "
                        "do not share a common realized duration."
                    )
                    break

            # The timing plan must use the same duration as the actual
            # realized gradient block.
            if crusher_index <= len(
                timing_plan.pulses
            ):
                planned_duration_s = (
                    timing_plan
                    .pulses[crusher_index - 1]
                    .crusher_duration_s
                )

                if not math.isclose(
                    reference_duration_s,
                    planned_duration_s,
                    rel_tol=0.0,
                    abs_tol=max(
                        1e-12,
                        grad_raster_s * 1e-6,
                    ),
                ):
                    errors.append(
                        f"VAPOR crusher {crusher_index} duration mismatch: "
                        f"timing plan="
                        f"{planned_duration_s * 1e3:.6f} ms, "
                        f"realized="
                        f"{reference_duration_s * 1e3:.6f} ms."
                    )

    # -------------------------------------------------------------------------
    # Requested RF-reference timing verification
    # -------------------------------------------------------------------------

    if len(timing_plan.pulses) == 8:
        for i, requested_interval_s in enumerate(
            inter_pulse_intervals_s,
            start=1,
        ):
            if i >= len(
                timing_plan.pulses
            ):
                break

            realized_interval_s = (
                timing_plan.pulses[i].rf_ref_s
                - timing_plan.pulses[i - 1].rf_ref_s
            )

            error_s = (
                realized_interval_s
                - requested_interval_s
            )

            if not math.isclose(
                realized_interval_s,
                requested_interval_s,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                errors.append(
                    f"VAPOR RF-reference interval {i}->{i + 1} mismatch: "
                    f"requested="
                    f"{requested_interval_s * 1e3:.6f} ms, "
                    f"planned="
                    f"{realized_interval_s * 1e3:.6f} ms, "
                    f"error={error_s * 1e6:.3f} us."
                )

        realized_final_delay_s = (
            timing_plan.preparation_end_s
            - timing_plan.pulses[-1].rf_ref_s
        )

        final_error_s = (
            realized_final_delay_s
            - final_delay_s
        )

        if not math.isclose(
            realized_final_delay_s,
            final_delay_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append(
                "VAPOR final RF8-reference interval mismatch: "
                f"requested={final_delay_s * 1e3:.6f} ms, "
                f"planned={realized_final_delay_s * 1e3:.6f} ms, "
                f"error={final_error_s * 1e6:.3f} us."
            )

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    if errors:
        message = (
            "Invalid VAPOR preparation:\n  - "
            + "\n  - ".join(
                errors
            )
        )

        logger.error(
            "%s",
            message,
        )

        raise ValueError(
            message
        )

    logger.debug(
        "VAPOR validation passed: "
        "8 RF pulses | "
        "8 crushers | "
        "8 timing entries | "
        "RF/gradient hardware checks passed."
    )


@dataclass(frozen=True, slots=True)
class VaporBuildResult:
    """References to a VAPOR module already inserted into a sequence.

    ``make_vapor`` mutates the supplied sequence by inserting the VAPOR
    preparation. This result object gives the parent sequence convenient
    handles to the semantic node, concrete blocks, and resolved timing.
    """

    node: str
    node_record: dict[str, Any]
    blocks: tuple
    timing: VaporTimingPlan

    @property
    def duration_s(self) -> float:
        """Return the resolved VAPOR preparation duration."""
        return self.timing.duration_s



# =============================================================================
# PUBLIC CONSTRUCTOR
# =============================================================================
def make_vapor(
    seq,
    protocol,
    *,
    node: str = "vapor",
    defaults: VaporDefaults = DEFAULT_VAPOR,
    crusher_scheme: CrusherSchemeName | None = None,
    crusher_reference_area_t_per_m_s: float | None = None,
    rf_waveform: RFWaveform | np.ndarray | None = None,
    rf_dwell_s: float | None = None,
    rf_time_ref_s: float | None = None,
    rf_frequency_offset_hz: float | None = None,
    display_rf: bool = False,
) -> VaporBuildResult:
    """Construct and insert an eight-pulse VAPOR preparation.

    ``make_vapor`` is a reusable PyPulseq-Star sequence-module constructor.

    It performs four high-level operations:

    1. resolve the VAPOR protocol and scientific prescription;
    2. construct the eight RF pulses;
    3. construct the eight logical-axis crusher groups;
    4. validate, register, and insert the complete VAPOR module.

    The supplied sequence is modified directly. The returned
    :class:`VaporBuildResult` provides convenient references to the resulting
    semantic node, concrete timeline blocks, and resolved timing plan.

    A parent sequence controls where VAPOR appears in the semantic hierarchy.
    For example::

        vapor = make_vapor(
            seq,
            protocol,
            node="kernel.mag_prep.vapor",
        )

    produces a hierarchy of the form::

        kernel
            mag_prep
                vapor
                    pulse_1
                    crusher_1
                    delay_1
                    pulse_2
                    crusher_2
                    delay_2
                    ...
                    pulse_8
                    crusher_8
                    final_fill

    The blocks are already part of ``seq`` when this function returns.
    Therefore the caller must NOT subsequently call ``seq.add_block`` or
    ``seq.add_node`` on the returned VAPOR result.

    Parameters
    ----------
    seq
        Existing PyPulseq-Star sequence into which VAPOR will be inserted.

    protocol
        Sequence Protocol. Scanner-facing VAPOR parameters are added under
        the ``vapor_`` namespace while preserving values already supplied by
        the parent sequence or scanner.

    node
        Semantic hierarchy path that owns the complete VAPOR module.

        The default is ``"vapor"``. Parent sequences should normally provide
        a context-specific path, for example::

            kernel.mag_prep.vapor

        VAPOR owns everything below this path; the parent sequence owns the
        hierarchy above it.

    defaults
        Scientific defaults for the VAPOR preparation.

    crusher_scheme
        Optional named crusher prescription. If omitted,
        ``defaults.crusher_scheme`` is used.

    crusher_reference_area_t_per_m_s
        Physical gradient area corresponding to a relative crusher component
        of 1.0, in T/m*s.

        If omitted, the selected crusher scheme's reference area is used.

    rf_waveform
        Optional externally supplied RF waveform.

        ``None`` generates the default VAPOR RF waveform. An ``RFWaveform``
        may be supplied directly, or a raw numpy array may be supplied
        together with ``rf_dwell_s``.

    rf_dwell_s
        RF sample dwell time in seconds for a raw-array RF waveform.

    rf_time_ref_s
        Effective RF timing reference measured from waveform start.

        For the default symmetric pulse this is the RF midpoint. Asymmetric
        externally supplied pulses should provide their appropriate timing
        reference explicitly.

    rf_frequency_offset_hz
        VAPOR RF frequency offset in Hz.

        If omitted, the value stored in the Protocol is used.

    display_rf
        If True, display the resolved base RF waveform before constructing
        the eight scaled VAPOR RF events.

    Returns
    -------
    VaporBuildResult
        Handle to the VAPOR module that has already been inserted into
        ``seq``.

        ``result.node``
            Semantic hierarchy path.

        ``result.node_record``
            PyPulseq-Star hierarchy-node record registered by
            ``seq.set_node()``.

        ``result.blocks``
            Concrete VAPOR timeline blocks, in execution order.

        ``result.timing``
            Resolved scientific VAPOR timing plan.

        ``result.duration_s``
            Convenience access to the resolved preparation duration.

    Notes
    -----
    Scientific specification, semantic hierarchy, and scanner realization
    remain separate:

    - crusher vectors are specified in logical read/phase/slice coordinates;
    - the EncodingFrame determines final physical scanner orientation;
    - hardware limits determine realizable gradient amplitudes and durations;
    - RF-reference intervals define the scientific VAPOR timing;
    - the parent sequence determines where the VAPOR subtree belongs.

    Validation occurs before VAPOR is inserted into ``seq``. Invalid protocol
    relationships therefore do not leave a partially constructed preparation
    on the sequence timeline.
    """

    # =========================================================================
    # STEP 1 — RESOLVE PROTOCOL AND SCIENTIFIC PRESCRIPTION
    # =========================================================================

    system = seq.system

    if system is None:
        raise ValueError(
            "make_vapor requires seq.system to define scanner hardware limits."
        )

    # -------------------------------------------------------------------------
    # Validate semantic hierarchy path
    # -------------------------------------------------------------------------
    #
    # Do not register the node yet. We wait until the complete VAPOR
    # preparation has passed validation so a failed construction cannot leave
    # an empty or partially populated semantic node in the parent sequence.
    #

    if not isinstance(node, str):
        raise TypeError(
            "VAPOR node path must be a string."
        )

    node = node.strip()

    if not node:
        raise ValueError(
            "VAPOR node path must not be empty."
        )

    if node.startswith(".") or node.endswith("."):
        raise ValueError(
            "VAPOR node path must not begin or end with '.'."
        )

    if ".." in node:
        raise ValueError(
            "VAPOR node path must not contain empty hierarchy components."
        )

    logger.debug(
        "Starting VAPOR construction for node %r.",
        node,
    )

    # -------------------------------------------------------------------------
    # Select crusher prescription
    # -------------------------------------------------------------------------

    selected_scheme_name = (
        defaults.crusher_scheme
        if crusher_scheme is None
        else crusher_scheme
    )

    try:
        crusher = CRUSHER_SCHEMES[
            selected_scheme_name
        ]
    except KeyError as exc:
        available = ", ".join(
            sorted(CRUSHER_SCHEMES)
        )

        raise ValueError(
            f"Unknown VAPOR crusher scheme {selected_scheme_name!r}. "
            f"Available schemes: {available}."
        ) from exc

    if crusher_reference_area_t_per_m_s is None:
        crusher_reference_area_t_per_m_s = (
            crusher.reference_area_t_per_m_s
        )

    if crusher_reference_area_t_per_m_s is None:
        raise ValueError(
            f"VAPOR crusher scheme {selected_scheme_name!r} contains "
            "relative crusher areas but does not define an absolute "
            "reference area. Supply "
            "crusher_reference_area_t_per_m_s explicitly."
        )

    crusher_reference_area_t_per_m_s = float(
        crusher_reference_area_t_per_m_s
    )

    if crusher_reference_area_t_per_m_s <= 0.0:
        raise ValueError(
            "crusher_reference_area_t_per_m_s must be positive."
        )

    # -------------------------------------------------------------------------
    # Initialize scanner-facing protocol parameters
    # -------------------------------------------------------------------------
    #
    # The helper uses setdefault-style behavior:
    #
    #     existing protocol values are preserved
    #     missing VAPOR values receive defaults
    #
    # After initialization, Protocol is the authoritative source for
    # scanner-facing VAPOR parameters.
    #

    if rf_frequency_offset_hz is not None:
        protocol.parameters["vapor_rf_frequency_offset_hz"] = float(
        rf_frequency_offset_hz
        )

    if crusher_reference_area_t_per_m_s is not None:
        protocol.parameters[
            "vapor_crusher_reference_area_t_per_m_s"
        ] = float(crusher_reference_area_t_per_m_s)


    _add_vapor_protocol_parameters(
        protocol,
        defaults,
        crusher_reference_area_t_per_m_s=(
            crusher_reference_area_t_per_m_s
        ),
        rf_frequency_offset_hz=(
            0.0
            if rf_frequency_offset_hz is None
            else float(rf_frequency_offset_hz)
        ),
    )

    alpha_deg = float(
        protocol.get_parameter(
            "vapor_alpha_deg"
        )
    )

    flip_angle_scale = tuple(
        float(
            protocol.get_parameter(
                f"vapor_flip_scale_{i}"
            )
        )
        for i in range(1, 9)
    )

    inter_pulse_intervals_s = tuple(
        float(
            protocol.get_parameter(
                f"vapor_interval_{i}_{i + 1}_s"
            )
        )
        for i in range(1, 8)
    )

    final_delay_s = float(
        protocol.get_parameter(
            "vapor_final_delay_s"
        )
    )

    crusher_reference_area_t_per_m_s = float(
        protocol.get_parameter(
            "vapor_crusher_reference_area_t_per_m_s"
        )
    )

    rf_duration_s = float(
        protocol.get_parameter(
            "vapor_rf_duration_s"
        )
    )

    rf_bandwidth_hz = float(
        protocol.get_parameter(
            "vapor_rf_bandwidth_hz"
        )
    )

    if rf_frequency_offset_hz is None:
        rf_frequency_offset_hz = float(
            protocol.get_parameter(
                "vapor_rf_frequency_offset_hz"
            )
        )
    else:
        rf_frequency_offset_hz = float(
            rf_frequency_offset_hz
        )

    logger.debug(
        "VAPOR protocol resolved: "
        "alpha=%.6f deg | "
        "RF duration=%.6f ms | "
        "RF bandwidth=%.6f Hz | "
        "RF offset=%.6f Hz | "
        "crusher scheme=%s",
        alpha_deg,
        rf_duration_s * 1e3,
        rf_bandwidth_hz,
        rf_frequency_offset_hz,
        selected_scheme_name,
    )

    # The RF specification is immutable. Construct an active specification
    # containing the current Protocol values rather than blindly using values
    # frozen into DEFAULT_VAPOR.
    active_rf_defaults = replace(
        defaults.rf,
        duration_s=rf_duration_s,
        bandwidth_hz=rf_bandwidth_hz,
    )

    active_defaults = replace(
        defaults,
        rf=active_rf_defaults,
    )

    # =========================================================================
    # STEP 2 — CONSTRUCT THE EIGHT RF PULSES
    # =========================================================================
    #
    # First resolve one backend-neutral base RF waveform. The same shape is
    # reused for all eight pulses; only the requested flip angle changes.
    #

    rf = _resolve_vapor_rf_waveform(
        active_defaults,
        rf_waveform=rf_waveform,
        rf_dwell_s=rf_dwell_s,
        rf_time_ref_s=rf_time_ref_s,
    )

    if display_rf:
        from pypulseq_star.plotting import plot_rf_waveform

        plot_rf_waveform(
            rf,
            title="VAPOR base RF waveform",
        )

    rf_waveform_duration_s = (
        len(rf.signal)
        * rf.dwell_s
    )

    logger.debug(
        "Resolved VAPOR RF waveform: "
        "source=%s | "
        "samples=%d | "
        "dwell=%.6f us | "
        "duration=%.6f ms | "
        "time_ref=%.6f ms",
        rf.source,
        len(rf.signal),
        rf.dwell_s * 1e6,
        rf_waveform_duration_s * 1e3,
        rf.time_ref_s * 1e3,
    )

    rf_events = []

    for pulse_index, scale in enumerate(
        flip_angle_scale,
        start=1,
    ):
        flip_angle_deg = (
            alpha_deg
            * scale
        )

        flip_angle_rad = math.radians(
            flip_angle_deg
        )

        rf_event = ppstar.make_arbitrary_rf(
            signal=rf.signal,
            flip_angle=flip_angle_rad,
            duration=rf_duration_s,
            dwell=rf.dwell_s,
            freq_offset=rf_frequency_offset_hz,
            phase_offset=0.0,
            system=system,
            use="saturation",
            name=f"vapor_rf_{pulse_index}",
        )

        rf_events.append(
            rf_event
        )

        logger.debug(
            "Constructed VAPOR RF%d: "
            "scale=%.6f | "
            "flip angle=%.6f deg",
            pulse_index,
            scale,
            flip_angle_deg,
        )

    rf_events = tuple(
        rf_events
    )

    # =========================================================================
    # STEP 3 — CONSTRUCT THE EIGHT LOGICAL-AXIS CRUSHERS
    # =========================================================================
    #
    # Crusher prescriptions are stored as logical:
    #
    #       (read, phase, slice)
    #
    # gradient areas in physical T/m*s. No scanner x/y/z geometry is authored
    # here.
    #

    crusher_areas = _crusher_area_vectors(
        crusher,
        reference_area_t_per_m_s=(
            crusher_reference_area_t_per_m_s
        ),
    )

    if len(crusher_areas) != 8:
        raise ValueError(
            f"VAPOR crusher scheme {selected_scheme_name!r} must "
            f"contain exactly 8 crusher vectors; "
            f"got {len(crusher_areas)}."
        )

    # PyPulseq-Star/Pulseq gradient limits are gamma-scaled internally.
    # Convert them to physical units for the hardware-feasibility calculation.
    gamma_hz_per_t = float(
        system.gamma
    )

    if gamma_hz_per_t <= 0.0:
        raise ValueError(
            "system.gamma must be positive."
        )

    max_grad_t_per_m = (
        float(system.max_grad)
        / gamma_hz_per_t
    )

    max_slew_t_per_m_per_s = (
        float(system.max_slew)
        / gamma_hz_per_t
    )

    grad_raster_s = float(
        system.grad_raster_time
    )

    crusher_events = []
    crusher_durations_s = []

    for crusher_index, area_logical in enumerate(
        crusher_areas,
        start=1,
    ):
        # Every nonzero logical component of one crusher must execute with the
        # same duration. The most demanding component determines that common
        # hardware-feasible duration.
        crusher_duration_s = _crusher_duration(
            area_logical,
            max_grad_t_per_m=max_grad_t_per_m,
            max_slew_t_per_m_per_s=(
                max_slew_t_per_m_per_s
            ),
            grad_raster_s=grad_raster_s,
        )

        crusher_durations_s.append(
            crusher_duration_s
        )

        logical_gradient_events = []

        for logical_axis, area_t_per_m_s in zip(
            LOGICAL_CRUSHER_AXES,
            area_logical,
            strict=True,
        ):
            if area_t_per_m_s == 0.0:
                logical_gradient_events.append(
                    None
                )
                continue

            # Convert physical gradient area:
            #
            #       T/m*s * Hz/T = cycles/m
            #
            # into the gamma-scaled gradient-area convention used internally
            # by Pulseq/PyPulseq-Star.
            area_pulseq = (
                area_t_per_m_s
                * gamma_hz_per_t
            )

            gradient = ppstar.make_trapezoid(
                axis_role=logical_axis,
                area=area_pulseq,
                duration=crusher_duration_s,
                system=system,
                name=(
                    f"vapor_crusher_"
                    f"{crusher_index}_"
                    f"{logical_axis}"
                ),
                role="water_suppression_crusher",
                metadata={
                    "vapor_crusher_index": crusher_index,
                    "crusher_scheme": selected_scheme_name,
                    "crusher_area_t_per_m_s": (
                        area_t_per_m_s
                    ),
                    "reference_area_t_per_m_s": (
                        crusher_reference_area_t_per_m_s
                    ),
                },
            )

            logical_gradient_events.append(
                gradient
            )

        crusher_events.append(
            tuple(logical_gradient_events)
        )

        logger.debug(
            "Constructed VAPOR crusher %d: "
            "(read, phase, slice)="
            "(%.9g, %.9g, %.9g) T/m*s | "
            "duration=%.6f ms",
            crusher_index,
            area_logical[0],
            area_logical[1],
            area_logical[2],
            crusher_duration_s * 1e3,
        )

    crusher_events = tuple(
        crusher_events
    )

    crusher_durations_s = tuple(
        crusher_durations_s
    )

    # =========================================================================
    # STEP 4 — RESOLVE TIMING, VALIDATE, AND COMPOSE THE VAPOR MODULE
    # =========================================================================
    #
    # The requested timing convention is:
    #
    #       RF reference -> RF reference
    #
    # for pulses 1-7, followed by:
    #
    #       RF8 reference -> VAPOR end
    #
    # for the final interval.
    #

    timing_plan = _build_vapor_timing_plan(
        rf_duration_s=rf_duration_s,
        rf_time_ref_s=rf.time_ref_s,
        crusher_durations_s=crusher_durations_s,
        inter_pulse_intervals_s=(
            inter_pulse_intervals_s
        ),
        final_delay_s=final_delay_s,
    )

    # Validate the complete requested/realized module BEFORE mutating seq.
    #
    # This ensures an invalid VAPOR protocol cannot leave half of a module on
    # the parent sequence timeline.
    _validate_vapor(
        system=system,
        crusher=crusher,
        crusher_areas=crusher_areas,
        rf_events=rf_events,
        crusher_events=crusher_events,
        timing_plan=timing_plan,
        alpha_deg=alpha_deg,
        flip_angle_scale=flip_angle_scale,
        rf_duration_s=rf_duration_s,
        inter_pulse_intervals_s=(
            inter_pulse_intervals_s
        ),
        final_delay_s=final_delay_s,
    )

    # -------------------------------------------------------------------------
    # Register the VAPOR module root
    # -------------------------------------------------------------------------
    #
    # set_node() gives VAPOR a first-class semantic identity independent of
    # its individual blocks.
    #
    # For:
    #
    #       node="kernel.mag_prep.vapor"
    #
    # dotted-path parentage places all subsequent VAPOR child blocks beneath
    # the same module root for relationship visualization and .seq.json.
    #

    vapor_node = seq.set_node(
        node,
        role="water_suppression",
        metadata={
            "module": "vapor",
            "preparation": "water_suppression",
            "crusher_scheme": selected_scheme_name,
        },
    )

    # -------------------------------------------------------------------------
    # Insert concrete timeline blocks
    # -------------------------------------------------------------------------
    #
    # _add_vapor_node() adds the already constructed events directly to seq:
    #
    #       <node>.pulse_1
    #       <node>.crusher_1
    #       <node>.delay_1
    #       ...
    #       <node>.pulse_8
    #       <node>.crusher_8
    #       <node>.final_fill
    #
    # The returned tuple preserves concrete timeline order and is useful to
    # parent sequences, tests, simulation code, and post-realization timing
    # validation.
    #

    vapor_blocks = _add_vapor_node(
        seq,
        node=node,
        rf_events=rf_events,
        crusher_events=crusher_events,
        timing_plan=timing_plan,
        system=system,
    )

    # -------------------------------------------------------------------------
    # Return a reusable module handle
    # -------------------------------------------------------------------------
    #
    # Nothing needs to be re-added by demo_press.py or another parent
    # sequence. The module is already part of seq.
    #
    # The returned object simply exposes convenient references:
    #
    #       vapor.node
    #       vapor.node_record
    #       vapor.blocks
    #       vapor.timing
    #       vapor.duration_s
    #

    result = VaporBuildResult(
        node=node,
        node_record=vapor_node,
        blocks=vapor_blocks,
        timing=timing_plan,
    )

    logger.debug(
        "VAPOR construction complete: "
        "node=%s | "
        "%d blocks | "
        "duration=%.6f ms",
        result.node,
        len(result.blocks),
        result.duration_s * 1e3,
    )

    return result