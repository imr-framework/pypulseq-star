"""
PyPulseq-Star demo: relationship-aware Point RESolved Spectroscopy (PRESS).

Mission
-------
Demonstrate how a scientifically defined PRESS experiment can remain editable
at the protocol level while PyPulseq-Star preserves the relationships required
for scanner realization.

The demo intentionally keeps the PRESS localization kernel visible.

Generic sequence-programming mechanics are delegated to reusable ppstar APIs,
while PRESS-specific timing, localization order, phase cycling, and acquisition
composition remain explicit in this file.

Core experiment
---------------

Metabolite acquisition:

    VAPOR
        ->
    PRESS
        ->
    spectroscopy ADC
        ->
    TR recovery

Water reference:

    PRESS
        ->
    spectroscopy ADC
        ->
    TR recovery

The same make_press() implementation is used for both acquisition families.
"""


from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import pypulseq_star as ppstar
from pypulseq_star.mag_prep.water_suppression import make_vapor
from pypulseq_star.geometry import Voxel
from pypulseq_star.components.phase_cycling import PhaseCycle, make_phase_cycle, wrap_phase, EXORCYCLE_PHASES_4
from pypulseq_star.sequence import vary

from pypulseq_star.writers import (
    GammaStarWriter,
    PulseqWriter,
)


# =============================================================================
# 1. PRESS-SPECIFIC PHASE CYCLE
# =============================================================================
#
# Generic phase-cycle mechanics live in:
#
#     pypulseq_star.components.phase_cycling
#
# including:
#
#     PhaseCycle
#     PhaseCycleState
#     make_phase_cycle()
#     wrap_phase()
#     EXORCYCLE_PHASES_4
#     CYCLOPS_4
#
# This demo owns the PRESS-specific phase-cycle composition:
#
#     RF1 excitation      -> fixed phase
#     RF2 refocusing #1   -> EXORCYCLE_PHASES_4
#     RF3 refocusing #2   -> EXORCYCLE_PHASES_4
#
# giving:
#
#     1 x 4 x 4 = 16 phase-cycle states
#
# The receiver phase is derived from the desired PRESS coherence pathway.
#
# NOTE:
# The receiver-phase sign convention must be validated against the phase
# convention used by the ppstar/Pulseq ADC implementation before treating
# this table as scanner-validated.
# =============================================================================


def _press_receiver_phase(
    excitation_phase_rad: float,
    refocusing_1_phase_rad: float,
    refocusing_2_phase_rad: float,
) -> float:
    """Return receiver phase for the desired PRESS coherence pathway."""

    return wrap_phase(
        excitation_phase_rad
        - 2.0 * refocusing_1_phase_rad
        + 2.0 * refocusing_2_phase_rad
    )


PRESS_EXORCYCLE_16 = make_phase_cycle(
    rf_phases=(
        (0.0,),                 # RF1: 90° excitation, fixed phase
        EXORCYCLE_PHASES_4,     # RF2: first 180° refocusing pulse
        EXORCYCLE_PHASES_4,     # RF3: second 180° refocusing pulse
    ),
    receiver_phase=_press_receiver_phase,
    name="press_exorcycle_16",
)

# =============================================================================
# 2. SYSTEM
# =============================================================================


def define_system() -> ppstar.Opts:
    """Return representative generic scanner limits."""

    return ppstar.Opts(
        max_grad=28.0,
        grad_unit="mT/m",
        max_slew=120.0,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=100e-6,
        rf_raster_time=2e-6,
        grad_raster_time=10e-6,
        block_duration_raster=10e-6,
        adc_raster_time=1e-6,
        gamma=42.575575e6,
        max_rf = 15e-6,
    )


# =============================================================================
# 3. SCIENTIFIC PRESS PROTOCOL
# =============================================================================
#
# Only scientific / editable quantities belong here.
#
# PRESS:
#     press_*
#
# Spectroscopy acquisition:
#     spectroscopy_*
#
# Water reference:
#     water_reference_*
#
# VAPOR:
#     vapor_*
#     owned by make_vapor()
# =============================================================================


def define_protocol(
    overrides: dict[str, object] | None = None,
) -> ppstar.Protocol:

    protocol = ppstar.Protocol(
        name="press",
        description=(
            "Relationship-aware single-voxel PRESS spectroscopy."
        ),
        parameters={
            # -------------------------------------------------------------
            # Identity
            # -------------------------------------------------------------
            "sequence_name": "press",

            # -------------------------------------------------------------
            # PRESS timing
            # -------------------------------------------------------------
            #
            # Units:
            #     seconds
            #
            # Total echo time is derived as:
            #
            #     TE = TE1 + TE2
            #
            "press_te1_s": 32e-3,                 # s
            "press_te2_s": 65e-3,                 # s
            "press_repetition_time_s": 2.0,        # s

            # -------------------------------------------------------------
            # Voxel geometry
            # -------------------------------------------------------------
            #
            # Voxel dimensions are specified in meters in the logical
            # read / phase / slice frame.
            #
            # Voxel position is specified relative to scanner isocenter:
            #
            #     0.0 m = centered at isocenter along that logical axis
            #
            # Positive / negative displacement follows the corresponding
            # logical encoding-frame direction.
            #
            "press_orientation": "axial",

            "press_voxel_size_read_m": 20e-3,      # m
            "press_voxel_size_phase_m": 20e-3,     # m
            "press_voxel_size_slice_m": 20e-3,     # m

            "press_voxel_position_read_m": 0.0,     # m from isocenter
            "press_voxel_position_phase_m": 0.0,    # m from isocenter
            "press_voxel_position_slice_m": 0.0,    # m from isocenter

            # -------------------------------------------------------------
            # PRESS RF
            # -------------------------------------------------------------
            #
            # Flip angles:
            #     degrees
            #
            # RF durations:
            #     seconds
            #
            # RF bandwidths:
            #     Hz
            #
            "press_excitation_flip_angle_deg": 90.0,       # deg
            "press_refocusing_flip_angle_deg": 180.0,      # deg

            "press_excitation_duration_s": 3e-3,            # s
            "press_refocusing_duration_s": 5e-3,           # s

            # Preserve pulse-shape selectivity with fixed time-bandwidth
            # product while allowing the RF duration to stretch to satisfy
            # scanner max-B1 constraints. The realized bandwidth is therefore
            # TBW / realized_duration.
            "press_excitation_time_bw_product": 6.0,
            "press_refocusing_time_bw_product": 10.0,

            "press_excitation_bandwidth_hz": 2000.0,       # requested Hz
            "press_refocusing_bandwidth_hz": 2000.0,       # requested Hz
            
            # PRESS RF transmit frequency offset relative to the scanner reference [Hz].
            # For an NAA-centered acquisition, set this to the desired NAA carrier offset.
            "press_transmit_frequency_offset_hz": 0.0,      # Hz

            # -------------------------------------------------------------
            # PRESS crushers
            # -------------------------------------------------------------
            #
            # Scientific prescription:
            #     gradient area
            #
            # Units:
            #     T/m * s
            #
            # Crusher amplitude and duration are scanner-realization
            # quantities derived from this area and the hardware limits.
            #
            "press_crusher_area_t_per_m_s": 1e-4,           # T/m * s

            # -------------------------------------------------------------
            # Spectroscopy ADC
            # -------------------------------------------------------------
            #
            # spectroscopy_num_samples:
            #     number of complex ADC samples
            #
            # spectroscopy_dwell_s:
            #     seconds per complex sample
            #
            # Derived:
            #
            #     spectral bandwidth = 1 / dwell
            #
            #     acquisition duration = num_samples * dwell
            #
            "spectroscopy_num_samples": 2048,               # samples
            "spectroscopy_dwell_s": 500e-6,                 # s/sample

            # -------------------------------------------------------------
            # Number of signal averages
            # -------------------------------------------------------------
            #
            # Dimensionless repetition counts.
            #
            "spectroscopy_metabolite_nsa": 16,              # acquisitions
            "water_reference_nsa": 2,                       # acquisitions

            # -------------------------------------------------------------
            # Acquisition-family controls
            # -------------------------------------------------------------
            #
            # Boolean protocol controls.
            #
            "spectroscopy_water_suppression_enabled": True,
            "water_reference_enabled": True,
        },
        aliases={
            "TE1": "press_te1_s",
            "TE2": "press_te2_s",
            "TR": "press_repetition_time_s",
            "NSA": "spectroscopy_metabolite_nsa",
        },
    )

    if overrides:
        protocol.parameters.update(
            overrides
        )

    return protocol


# =============================================================================
# 4. VOXEL GEOMETRY
# =============================================================================
#
# Generic voxel geometry lives in:
#
#     pypulseq_star.geometry.Voxel
#
# The Voxel owns:
#
#     size in logical read / phase / slice coordinates
#     position relative to scanner isocenter
#
# Orientation is intentionally NOT stored in Voxel.
#
# The sequence EncodingFrame owns the mapping of logical:
#
#     read
#     phase
#     slice
#
# onto physical scanner axes.
#
# PRESS declares the localization roles:
#
#     RF1 -> logical slice
#     RF2 -> logical phase
#     RF3 -> logical read
# =============================================================================


def make_press_voxel(
    protocol: ppstar.Protocol,
) -> Voxel:
    """Construct the PRESS voxel geometry from the scientific Protocol."""

    p = protocol.symbols

    return Voxel(
        size_read_m=p.press_voxel_size_read_m,
        size_phase_m=p.press_voxel_size_phase_m,
        size_slice_m=p.press_voxel_size_slice_m,

        position_read_m=p.press_voxel_position_read_m,
        position_phase_m=p.press_voxel_position_phase_m,
        position_slice_m=p.press_voxel_position_slice_m,
    )

# =============================================================================
# 5. PRESS TIMING CONTRACT
# =============================================================================
#
# PRESS timing is defined from RF-center to RF-center / echo-center.
#
# Let:
#
#     RF1 = center of the 90° excitation pulse
#     RF2 = center of the first 180° refocusing pulse
#     RF3 = center of the second 180° refocusing pulse
#     E   = nominal PRESS spin echo
#
# The protocol-facing subecho times are:
#
#     TE1 = press_te1_s
#     TE2 = press_te2_s
#
# and the PRESS timing relationships are:
#
#     RF2 - RF1 = TE1 / 2
#
#     RF3 - RF2 = (TE1 + TE2) / 2
#
#     E - RF3   = TE2 / 2
#
# Therefore:
#
#     TE = TE1 + TE2
#
# Each requested interval contains fixed event occupancy from RF pulses,
# gradients, rephasers, and crushers. The remaining time is represented by
# a symbolic delay:
#
#     fill = requested interval - fixed occupied interval
#
# Because the protocol quantities remain symbolic, changing TE1 or TE2 changes
# these fills without changing the PRESS construction logic.
#
# Negative fills indicate that the requested TE1 / TE2 combination is not
# physically realizable with the selected RF / gradient realization and must
# be rejected during resolution / validation.
# =============================================================================


@dataclass(frozen=True, slots=True)
class PressTimingExpressions:
    """Symbolic timing relationships for one PRESS localization kernel."""

    # -------------------------------------------------------------------------
    # Requested center-to-center / center-to-echo intervals
    # -------------------------------------------------------------------------

    rf1_to_rf2_target: ppstar.Expression
    rf2_to_rf3_target: ppstar.Expression
    rf3_to_echo_target: ppstar.Expression

    # -------------------------------------------------------------------------
    # Symbolic non-negative timing fills
    # -------------------------------------------------------------------------

    rf1_to_rf2_fill: ppstar.Expression
    rf2_to_rf3_fill: ppstar.Expression
    rf3_to_echo_fill: ppstar.Expression

    # -------------------------------------------------------------------------
    # Derived total PRESS echo time
    # -------------------------------------------------------------------------

    total_te: ppstar.Expression


def make_press_timing(
    protocol: ppstar.Protocol,
    *,
    rf1_to_rf2_occupied: ppstar.Expression,
    rf2_to_rf3_occupied: ppstar.Expression,
    rf3_to_echo_occupied: ppstar.Expression,
) -> PressTimingExpressions:
    """Construct the symbolic PRESS TE1 / TE2 timing relationships.

    Parameters
    ----------
    protocol
        PRESS protocol containing ``press_te1_s`` and ``press_te2_s``.

    rf1_to_rf2_occupied
        Fixed duration already occupied between the CENTER of RF1 and the
        CENTER of RF2.

    rf2_to_rf3_occupied
        Fixed duration already occupied between the CENTER of RF2 and the
        CENTER of RF3.

    rf3_to_echo_occupied
        Fixed duration already occupied between the CENTER of RF3 and the
        nominal PRESS echo.

    Returns
    -------
    PressTimingExpressions
        Symbolic target intervals, timing fills, and total PRESS TE.

    Notes
    -----
    The three ``occupied`` inputs should be derived from the actual RF,
    gradient, rephasing, and crusher events constructed by ``make_press()``.

    This function does not know how those events are realized. It only
    expresses the PRESS timing contract.
    """

    p = protocol.symbols

    # -------------------------------------------------------------------------
    # Derived total PRESS echo time
    # -------------------------------------------------------------------------
    #
    # TE is not an independent protocol parameter:
    #
    #     TE = TE1 + TE2
    # -------------------------------------------------------------------------

    total_te = (
        p.press_te1_s
        + p.press_te2_s
    )

    # -------------------------------------------------------------------------
    # Requested PRESS intervals
    # -------------------------------------------------------------------------

    rf1_to_rf2_target = (
        0.5
        * p.press_te1_s
    )

    rf2_to_rf3_target = (
        0.5
        * (
            p.press_te1_s
            + p.press_te2_s
        )
    )

    rf3_to_echo_target = (
        0.5
        * p.press_te2_s
    )

    # -------------------------------------------------------------------------
    # Remaining symbolic timing fills
    # -------------------------------------------------------------------------
    #
    # These expressions remain tied to TE1 / TE2. If the protocol changes,
    # resolution recomputes them automatically.
    #
    # A negative resolved value means that the requested timing is infeasible.
    # -------------------------------------------------------------------------

    rf1_to_rf2_fill = (
        rf1_to_rf2_target
        - rf1_to_rf2_occupied
    )

    rf2_to_rf3_fill = (
        rf2_to_rf3_target
        - rf2_to_rf3_occupied
    )

    rf3_to_echo_fill = (
        rf3_to_echo_target
        - rf3_to_echo_occupied
    )

    return PressTimingExpressions(
        rf1_to_rf2_target=rf1_to_rf2_target,
        rf2_to_rf3_target=rf2_to_rf3_target,
        rf3_to_echo_target=rf3_to_echo_target,
        rf1_to_rf2_fill=rf1_to_rf2_fill,
        rf2_to_rf3_fill=rf2_to_rf3_fill,
        rf3_to_echo_fill=rf3_to_echo_fill,
        total_te=total_te,
    )


# =============================================================================
# 6. PRESS KERNEL
# =============================================================================
#
# Scientific PRESS structure:
#
#     RF1:  90° excitation       -> logical slice
#     RF2: 180° refocusing #1    -> logical phase
#     RF3: 180° refocusing #2    -> logical read
#
# Timing:
#
#     RF2 - RF1 = TE1 / 2
#     RF3 - RF2 = (TE1 + TE2) / 2
#     E   - RF3 = TE2 / 2
#
# Spatial directions remain logical through axis_role. EncodingFrame performs
# the later mapping to physical scanner x/y/z.
#
# Phase cycling belongs to the enclosing NSA loop. The PRESS node itself is
# the localization/acquisition motif and is not independently repeated.
# =============================================================================


@dataclass(frozen=True, slots=True)
class PressBuildResult:
    """Events and relationships created for one PRESS kernel."""

    node: str

    excitation: tuple[object, object, object]
    refocusing_1: tuple[object, object, object]
    refocusing_2: tuple[object, object, object]

    crushers: tuple[object, object, object, object]

    readout: object
    timing: PressTimingExpressions

    blocks: tuple[str, ...]
    duration_s: ppstar.Expression


def make_press(
    seq: ppstar.Sequence,
    protocol: ppstar.Protocol,
    *,
    voxel: Voxel,
    node: str,
    repeat_node,
    phase_cycle: PhaseCycle,
) -> PressBuildResult:
    """Add one relationship-aware PRESS localization/acquisition kernel.

    Parameters
    ----------
    seq
        Sequence receiving the PRESS kernel.

    protocol
        PRESS scientific protocol.

    voxel
        PRESS localization voxel in logical read/phase/slice coordinates.

    node
        Hierarchical node for this PRESS motif, for example
        ``"root.metabolite.averages_metabolite.kernel_press"``.

    repeat_node
        Enclosing NSA loop handle returned by ``seq.set_node(...)``.
        Phase cycling is attached to this node.

    phase_cycle
        RF1/RF2/RF3/receiver phase-cycle definition.
    """

    p = protocol.symbols
    system = seq.system
    prefix = node.replace(".", "_")

    # Register the PRESS hierarchy node. It is a child of the enclosing NSA
    # loop but is not itself repeated.
    seq.set_node(
        node,
        role="press_localization",
    )

    # -------------------------------------------------------------------------
    # Selective PRESS RF
    # -------------------------------------------------------------------------

    def make_press_rf(
        *,
        axis_role: str,
        flip_angle_deg,
        duration_s,
        bandwidth_hz,
        time_bw_product,
        use: str,
        name: str,
    ):
        """Create one voxel-selective PRESS RF pulse."""

        thickness_m = voxel.size_along(axis_role)
        position_m = voxel.position_along(axis_role)

        # Gradient amplitude is in Hz/m.
        selection_gradient_hz_per_m = (
            bandwidth_hz / thickness_m
        )

        # Scientific transmit offset plus the localization offset required for
        # an off-isocenter voxel.
        frequency_offset_hz = (
            p.press_transmit_frequency_offset_hz
            + selection_gradient_hz_per_m * position_m
        )

        return ppstar.make_sinc_pulse(
            flip_angle=flip_angle_deg * math.pi / 180.0,
            duration=duration_s,
            time_bw_product=time_bw_product,
            rf_constraint_policy="stretch",
            slice_thickness=thickness_m,
            freq_offset=frequency_offset_hz,
            axis_role=axis_role,
            use=use,
            system=system,
            return_gz=True,
            name=name,
            gz_name=f"{name}_select",
            gzr_name=f"{name}_rephaser",
        )

    excitation_rf, excitation_g, excitation_rephaser = make_press_rf(
        axis_role="slice",
        flip_angle_deg=p.press_excitation_flip_angle_deg,
        duration_s=p.press_excitation_duration_s,
        bandwidth_hz=p.press_excitation_bandwidth_hz,
        time_bw_product=p.press_excitation_time_bw_product,
        use="excitation",
        name=f"{prefix}_rf1",
    )

    refocusing_1_rf, refocusing_1_g, refocusing_1_rephaser = make_press_rf(
        axis_role="phase",
        flip_angle_deg=p.press_refocusing_flip_angle_deg,
        duration_s=p.press_refocusing_duration_s,
        bandwidth_hz=p.press_refocusing_bandwidth_hz,
        time_bw_product=p.press_refocusing_time_bw_product,
        use="refocusing",
        name=f"{prefix}_rf2",
    )

    refocusing_2_rf, refocusing_2_g, refocusing_2_rephaser = make_press_rf(
        axis_role="read",
        flip_angle_deg=p.press_refocusing_flip_angle_deg,
        duration_s=p.press_refocusing_duration_s,
        bandwidth_hz=p.press_refocusing_bandwidth_hz,
        time_bw_product=p.press_refocusing_time_bw_product,
        use="refocusing",
        name=f"{prefix}_rf3",
    )

    # -------------------------------------------------------------------------
    # Crushers
    # -------------------------------------------------------------------------
    #
    # Protocol crusher area is stored in T/m*s.
    # Pulseq gradient area is expressed in cycles/m, so multiply by gamma.
    # -------------------------------------------------------------------------

    crusher_area = (
        p.press_crusher_area_t_per_m_s
        * system.gamma
    )

    crusher_before_rf2 = ppstar.make_trapezoid(
        area=crusher_area,
        axis_role="phase",
        system=system,
        name=f"{prefix}_crusher_before_rf2",
        role="crusher",
    )

    crusher_after_rf2 = ppstar.make_trapezoid(
        area=crusher_area,
        axis_role="phase",
        system=system,
        name=f"{prefix}_crusher_after_rf2",
        role="crusher",
    )

    crusher_before_rf3 = ppstar.make_trapezoid(
        area=crusher_area,
        axis_role="read",
        system=system,
        name=f"{prefix}_crusher_before_rf3",
        role="crusher",
    )

    crusher_after_rf3 = ppstar.make_trapezoid(
        area=crusher_area,
        axis_role="read",
        system=system,
        name=f"{prefix}_crusher_after_rf3",
        role="crusher",
    )

    # -------------------------------------------------------------------------
    # Spectroscopy ADC
    # -------------------------------------------------------------------------

    readout = ppstar.make_adc(
        num_samples=p.spectroscopy_num_samples,
        dwell=p.spectroscopy_dwell_s,
        freq_offset=p.press_transmit_frequency_offset_hz,
        system=system,
        name=f"{prefix}_adc",
        role="spectroscopy",
    )

    # -------------------------------------------------------------------------
    # Fixed RF-center occupancy
    # -------------------------------------------------------------------------
    #
    # The RF constructors have now realized hardware-rasterized events. These
    # quantities describe the fixed event occupancy around each RF center;
    # make_press_timing() uses them to derive the remaining TE fills.
    # -------------------------------------------------------------------------

    def rf_center_occupancy(
        rf,
        selection_gradient,
    ) -> tuple[float, float]:
        """Return fixed time before and after the RF center."""

        rf_delay = float(
            rf.parameters.get(
                "delay",
                0.0,
            )
        )

        rf_duration = float(
            rf.duration
        )

        rf_ringdown = float(
            rf.parameters.get(
                "rf_ringdown_time",
                getattr(
                    system,
                    "rf_ringdown_time",
                    0.0,
                ),
            )
        )

        rf_center = (
            rf_delay
            + 0.5 * rf_duration
        )

        rf_end = (
            rf_delay
            + rf_duration
            + rf_ringdown
        )

        gradient_end = (
            float(
                getattr(
                    selection_gradient,
                    "delay",
                    0.0,
                )
            )
            + float(
                getattr(
                    selection_gradient,
                    "rise_time",
                    0.0,
                )
            )
            + float(
                getattr(
                    selection_gradient,
                    "flat_time",
                    0.0,
                )
            )
            + float(
                getattr(
                    selection_gradient,
                    "fall_time",
                    0.0,
                )
            )
        )

        block_end = max(
            rf_end,
            gradient_end,
        )

        return (
            rf_center,
            block_end - rf_center,
        )

    _, excitation_post_center = rf_center_occupancy(
        excitation_rf,
        excitation_g,
    )

    (
        refocusing_1_pre_center,
        refocusing_1_post_center,
    ) = rf_center_occupancy(
        refocusing_1_rf,
        refocusing_1_g,
    )

    (
        refocusing_2_pre_center,
        refocusing_2_post_center,
    ) = rf_center_occupancy(
        refocusing_2_rf,
        refocusing_2_g,
    )

    # -------------------------------------------------------------------------
    # PRESS TE relationships
    # -------------------------------------------------------------------------

    timing = make_press_timing(
        protocol,
        rf1_to_rf2_occupied=(
            excitation_post_center
            + excitation_rephaser.duration
            + crusher_before_rf2.duration
            + refocusing_1_pre_center
        ),
        rf2_to_rf3_occupied=(
            refocusing_1_post_center
            + crusher_after_rf2.duration
            + refocusing_1_rephaser.duration
            + crusher_before_rf3.duration
            + refocusing_2_pre_center
        ),
        rf3_to_echo_occupied=(
            refocusing_2_post_center
            + crusher_after_rf3.duration
            + refocusing_2_rephaser.duration
        ),
    )

    rf1_to_rf2_fill = ppstar.make_delay(
        timing.rf1_to_rf2_fill,
        system=system,
        name=f"{prefix}_rf1_to_rf2_fill",
        role="echo_time_fill",
    )

    rf2_to_rf3_fill = ppstar.make_delay(
        timing.rf2_to_rf3_fill,
        system=system,
        name=f"{prefix}_rf2_to_rf3_fill",
        role="echo_time_fill",
    )

    rf3_to_echo_fill = ppstar.make_delay(
        timing.rf3_to_echo_fill,
        system=system,
        name=f"{prefix}_rf3_to_echo_fill",
        role="echo_time_fill",
    )

    # -------------------------------------------------------------------------
    # PRESS timeline
    # -------------------------------------------------------------------------

    rf1_block = f"{prefix}_rf1"
    rf1_rephaser_block = f"{prefix}_rf1_rephaser"
    rf1_fill_block = f"{prefix}_rf1_to_rf2_fill"

    crusher_before_rf2_block = f"{prefix}_crusher_before_rf2"
    rf2_block = f"{prefix}_rf2"
    crusher_after_rf2_block = f"{prefix}_crusher_after_rf2"
    rf2_rephaser_block = f"{prefix}_rf2_rephaser"
    rf2_fill_block = f"{prefix}_rf2_to_rf3_fill"

    crusher_before_rf3_block = f"{prefix}_crusher_before_rf3"
    rf3_block = f"{prefix}_rf3"
    crusher_after_rf3_block = f"{prefix}_crusher_after_rf3"
    rf3_rephaser_block = f"{prefix}_rf3_rephaser"
    rf3_fill_block = f"{prefix}_rf3_to_echo_fill"

    adc_block = f"{prefix}_adc"

    seq.add_block(
        excitation_rf,
        excitation_g,
        name=rf1_block,
        role="excitation",
        node=node,
    )

    seq.add_block(
        excitation_rephaser,
        name=rf1_rephaser_block,
        role="rephasing",
        node=node,
    )

    seq.add_block(
        rf1_to_rf2_fill,
        name=rf1_fill_block,
        role="echo_time_fill",
        node=node,
    )

    seq.add_block(
        crusher_before_rf2,
        name=crusher_before_rf2_block,
        role="crusher",
        node=node,
    )

    seq.add_block(
        refocusing_1_rf,
        refocusing_1_g,
        name=rf2_block,
        role="refocusing",
        node=node,
    )

    seq.add_block(
        crusher_after_rf2,
        name=crusher_after_rf2_block,
        role="crusher",
        node=node,
    )

    seq.add_block(
        refocusing_1_rephaser,
        name=rf2_rephaser_block,
        role="rephasing",
        node=node,
    )

    seq.add_block(
        rf2_to_rf3_fill,
        name=rf2_fill_block,
        role="echo_time_fill",
        node=node,
    )

    seq.add_block(
        crusher_before_rf3,
        name=crusher_before_rf3_block,
        role="crusher",
        node=node,
    )

    seq.add_block(
        refocusing_2_rf,
        refocusing_2_g,
        name=rf3_block,
        role="refocusing",
        node=node,
    )

    seq.add_block(
        crusher_after_rf3,
        name=crusher_after_rf3_block,
        role="crusher",
        node=node,
    )

    seq.add_block(
        refocusing_2_rephaser,
        name=rf3_rephaser_block,
        role="rephasing",
        node=node,
    )

    seq.add_block(
        rf3_to_echo_fill,
        name=rf3_fill_block,
        role="echo_time_fill",
        node=node,
    )

    seq.add_block(
        readout,
        name=adc_block,
        role="acquisition",
        node=node,
    )

    block_names = (
        rf1_block,
        rf1_rephaser_block,
        rf1_fill_block,
        crusher_before_rf2_block,
        rf2_block,
        crusher_after_rf2_block,
        rf2_rephaser_block,
        rf2_fill_block,
        crusher_before_rf3_block,
        rf3_block,
        crusher_after_rf3_block,
        rf3_rephaser_block,
        rf3_fill_block,
        adc_block,
    )

    # -------------------------------------------------------------------------
    # PRESS phase cycle
    # -------------------------------------------------------------------------
    #
    # PhaseCycle owns the scientific RF / receiver phase table.
    # The enclosing NSA repeat node owns the loop-dependent variation.
    # Writers lower these table relationships to gammaSTAR or Pulseq.
    # -------------------------------------------------------------------------

    if phase_cycle.num_rf_events != 3:
        raise ValueError(
            "PRESS phase cycle must define exactly three RF phases per state; "
            f"got {phase_cycle.num_rf_events}."
        )

    repeat_node.vary(
        vary(
            excitation_rf,
            "phase_offset",
            values=tuple(
                state.rf_phases_rad[0]
                for state in phase_cycle.states
            ),
            mode="table",
            wrap=2.0 * math.pi,
        ),
        vary(
            refocusing_1_rf,
            "phase_offset",
            values=tuple(
                state.rf_phases_rad[1]
                for state in phase_cycle.states
            ),
            mode="table",
            wrap=2.0 * math.pi,
        ),
        vary(
            refocusing_2_rf,
            "phase_offset",
            values=tuple(
                state.rf_phases_rad[2]
                for state in phase_cycle.states
            ),
            mode="table",
            wrap=2.0 * math.pi,
        ),
        vary(
            readout,
            "phase_offset",
            values=tuple(
                state.receiver_phase_rad
                for state in phase_cycle.states
            ),
            mode="table",
            wrap=2.0 * math.pi,
        ),
    )

    # -------------------------------------------------------------------------
    # Kernel duration
    # -------------------------------------------------------------------------

    duration_s = seq.duration(
        start_block_name=rf1_block,
        end_block_name=adc_block,
    )

    return PressBuildResult(
        node=node,
        excitation=(
            excitation_rf,
            excitation_g,
            excitation_rephaser,
        ),
        refocusing_1=(
            refocusing_1_rf,
            refocusing_1_g,
            refocusing_1_rephaser,
        ),
        refocusing_2=(
            refocusing_2_rf,
            refocusing_2_g,
            refocusing_2_rephaser,
        ),
        crushers=(
            crusher_before_rf2,
            crusher_after_rf2,
            crusher_before_rf3,
            crusher_after_rf3,
        ),
        readout=readout,
        timing=timing,
        blocks=block_names,
        duration_s=duration_s,
    )

# =============================================================================
# 7. COMPLETE SEQUENCE
# =============================================================================
#
# Explicit execution hierarchy:
#
#     root
#         metabolite
#             averages_metabolite          [loop: NSA, repeat_every = TR]
#                 vapor
#                 kernel_press
#
#         water
#             averages_water               [loop: NSA, repeat_every = TR]
#                 kernel_press
#
# The acquisition-family nodes (metabolite / water) are semantic containers.
# Only the averages_* nodes repeat. Each repeated node owns:
#
#     repeat_count
#     repeat_every = TR
#     loop counter
#     PRESS phase-cycle variation
#
# VAPOR is inside averages_metabolite, so it executes once per metabolite TR.
# kernel_press is the same PRESS implementation in both acquisition families.
# =============================================================================


def build_sequence(
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
    *,
    debug: bool = False,
) -> ppstar.Sequence:
    """Build the complete PRESS spectroscopy experiment."""

    p = protocol.symbols

    seq = ppstar.Sequence(
        system=system,
        protocol=protocol,
        name=p.sequence_name,
        debug=debug,
    )

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------

    voxel = make_press_voxel(
        protocol
    )

    seq.set_encoding_frame(
        protocol.parameters["press_orientation"]
    )

    # =========================================================================
    # Explicit hierarchy root
    # =========================================================================

    seq.set_node(
        "root",
        role="press_spectroscopy",
    )

    # =========================================================================
    # Metabolite acquisition family
    # =========================================================================
    #
    # root
    #     metabolite
    #         averages_metabolite          [NSA loop, period = TR]
    #             vapor
    #             kernel_press
    #
    # The family node is semantic only. The averages node is the executable
    # repetition owner and therefore also owns the PRESS phase-cycle variation.
    # =========================================================================

    seq.set_node(
        "root.metabolite",
        role="spectroscopy_metabolite",
    )

    metabolite_averages = seq.set_node(
        "root.metabolite.averages_metabolite",
        role="spectroscopy_metabolite_averages",
        factor=p.spectroscopy_metabolite_nsa,
        repeat_every=p.press_repetition_time_s,
        counter="metabolite_nsa_index",
        repeat_mode="loop",
    )

    if protocol.parameters["spectroscopy_water_suppression_enabled"]:
        make_vapor(
            seq,
            protocol,
            node="root.metabolite.averages_metabolite.vapor",
        )

    make_press(
        seq,
        protocol,
        voxel=voxel,
        node="root.metabolite.averages_metabolite.kernel_press",
        repeat_node=metabolite_averages,
        phase_cycle=PRESS_EXORCYCLE_16,
    )

    # =========================================================================
    # Unsuppressed water-reference acquisition family
    # =========================================================================
    #
    # root
    #     water
    #         averages_water               [NSA loop, period = TR]
    #             kernel_press
    #
    # The water family is independent of the metabolite family but uses the
    # identical PRESS kernel implementation. No VAPOR node is present.
    # =========================================================================

    if protocol.parameters["water_reference_enabled"]:

        seq.set_node(
            "root.water",
            role="spectroscopy_water_reference",
        )

        water_averages = seq.set_node(
            "root.water.averages_water",
            role="spectroscopy_water_averages",
            factor=p.water_reference_nsa,
            repeat_every=p.press_repetition_time_s,
            counter="water_reference_nsa_index",
            repeat_mode="loop",
        )

        make_press(
            seq,
            protocol,
            voxel=voxel,
            node="root.water.averages_water.kernel_press",
            repeat_node=water_averages,
            phase_cycle=PRESS_EXORCYCLE_16,
        )

    return seq


# =============================================================================
# 9. MAIN / RESOLVE / PLOT / EXPORT
# =============================================================================


def main(
    *,
    orientation: str = "axial",
    plot: bool = True,
    write_seq: bool = True,
    write_json: bool = True,
    debug: bool = False,
    protocol_overrides: dict[str, object] | None = None,
) -> None:
    """Build, resolve, visualize, and export the PRESS sequence."""

    system = define_system()

    overrides = dict(protocol_overrides or {})
    overrides["press_orientation"] = orientation

    protocol = define_protocol(
        overrides=overrides,
    )
    
    seq = build_sequence(
        system,
        protocol,
        debug=debug,
    )

    resolved = seq.resolve()

    ok, report = resolved.check_timing()

    print(
        "Timing check passed successfully"
        if ok
        else "Timing check failed"
    )

    for error in report:
        print(error)

    if not ok:
        raise RuntimeError(
            "PRESS sequence timing validation failed."
        )

    print(f"TE1: {resolved.protocol.press_te1_s * 1e3:.3f} ms")
    print(f"TE2: {resolved.protocol.press_te2_s * 1e3:.3f} ms")

    print(
    f"TE:  "
    f"{(resolved.protocol.press_te1_s + resolved.protocol.press_te2_s) * 1e3:.3f} ms"
)
    
    print(f"TR:  {resolved.protocol.press_repetition_time_s * 1e3:.3f} ms")
    print(
        f"Metabolite NSA: "
        f"{resolved.protocol.spectroscopy_metabolite_nsa}"
    )

    if resolved.protocol.water_reference_enabled:
        print(
            f"Water-reference NSA: "
            f"{resolved.protocol.water_reference_nsa}"
        )

    if plot:
        seq.plot(
            realization=resolved,
            one_tr=True,
            # time_range=(0.0, 3),
            title=f"PyPulseq-Star PRESS ({orientation})",
            gradient_scale="mt_per_m",
            debug=debug,
        )

    output_dir = Path(
        "out/press"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if write_seq:
        seq_path = PulseqWriter(
            seq
        ).write(
            output_dir / f"press_{orientation}.seq",
            realization=resolved,
        )

        print(
            f"Wrote Pulseq: {seq_path}"
        )

    
    if write_json:
        json_path = GammaStarWriter(
            seq
        ).write(
            output_dir / f"press_{orientation}.seq.json",
            defaults=resolved,
        )

        print(
            f"Wrote gammaSTAR JSON: {json_path}"
        )


# =============================================================================
# 10. COMMAND-LINE INTERFACE
# =============================================================================


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Protocol-centered relationship-aware PRESS spectroscopy demo."
        )
    )

    parser.add_argument(
        "--te1",
        type=float,
        help="Override PRESS TE1 in seconds.",
    )

    parser.add_argument(
        "--te2",
        type=float,
        help="Override PRESS TE2 in seconds.",
    )

    parser.add_argument(
        "--tr",
        type=float,
        help="Override repetition time in seconds.",
    )

    parser.add_argument(
        "--nsa",
        type=int,
        help="Override metabolite number of signal averages.",
    )

    parser.add_argument(
        "--water-reference-nsa",
        type=int,
        help="Override water-reference number of signal averages.",
    )

    parser.add_argument(
        "--voxel-size-read",
        type=float,
        help="Override voxel size along logical read axis in metres.",
    )

    parser.add_argument(
        "--voxel-size-phase",
        type=float,
        help="Override voxel size along logical phase axis in metres.",
    )

    parser.add_argument(
        "--voxel-size-slice",
        type=float,
        help="Override voxel size along logical slice axis in metres.",
    )

    parser.add_argument(
        "--voxel-position-read",
        type=float,
        help="Override voxel position along logical read axis in metres.",
    )

    parser.add_argument(
        "--voxel-position-phase",
        type=float,
        help="Override voxel position along logical phase axis in metres.",
    )

    parser.add_argument(
        "--voxel-position-slice",
        type=float,
        help="Override voxel position along logical slice axis in metres.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        help="Override number of spectroscopy samples.",
    )

    parser.add_argument(
        "--dwell",
        type=float,
        help="Override spectroscopy dwell time in seconds.",
    )

    parser.add_argument(
        "--orientation",
        choices=("axial", "coronal", "sagittal"),
        default="axial",
    )

    parser.add_argument(
        "--no-water-suppression",
        action="store_true",
        help="Disable VAPOR water suppression.",
    )

    parser.add_argument(
        "--no-water-reference",
        action="store_true",
        help="Disable unsuppressed water-reference acquisition.",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
    )

    parser.add_argument(
        "--no-seq",
        action="store_true",
    )

    parser.add_argument(
        "--no-json",
        action="store_true",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    args = parser.parse_args()

    overrides: dict[str, object] = {}

    if args.te1 is not None:
        overrides["press_te1_s"] = args.te1

    if args.te2 is not None:
        overrides["press_te2_s"] = args.te2

    if args.tr is not None:
        overrides["press_repetition_time_s"] = args.tr

    if args.nsa is not None:
        overrides["spectroscopy_metabolite_nsa"] = args.nsa

    if args.water_reference_nsa is not None:
        overrides["water_reference_nsa"] = args.water_reference_nsa

    if args.voxel_size_read is not None:
        overrides["press_voxel_size_read_m"] = args.voxel_size_read

    if args.voxel_size_phase is not None:
        overrides["press_voxel_size_phase_m"] = args.voxel_size_phase

    if args.voxel_size_slice is not None:
        overrides["press_voxel_size_slice_m"] = args.voxel_size_slice

    if args.voxel_position_read is not None:
        overrides["press_voxel_position_read_m"] = args.voxel_position_read

    if args.voxel_position_phase is not None:
        overrides["press_voxel_position_phase_m"] = args.voxel_position_phase

    if args.voxel_position_slice is not None:
        overrides["press_voxel_position_slice_m"] = args.voxel_position_slice

    if args.num_samples is not None:
        overrides["spectroscopy_num_samples"] = args.num_samples

    if args.dwell is not None:
        overrides["spectroscopy_dwell_s"] = args.dwell

    if args.no_water_suppression:
        overrides["spectroscopy_water_suppression_enabled"] = False

    if args.no_water_reference:
        overrides["water_reference_enabled"] = False

    main(
        orientation=args.orientation,
        plot=not args.no_plot,
        write_seq=not args.no_seq,
        write_json=not args.no_json,
        debug=args.debug,
        protocol_overrides=overrides,
    )