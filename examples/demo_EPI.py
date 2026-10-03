"""Phase 2 demo: protocol-centered, relationship-aware segmented EPI.

This example follows the same developer workflow and visual organization as
``demo_GRE.py``:

* protocol-derived scalar values remain symbolic while the sequence is built;
* one representative EPI shot is retained in the symbolic sequence;
* the shot is repeated through a logical segment loop;
* phase-prephasing and phase-blip scaling remain protocol relationships;
* ``seq.resolve()`` creates the immutable numeric realization used by Pulseq;
* gammaSTAR receives the symbolic sequence plus validated defaults;
* interactive echo-spacing/readout-duration reshaping of an already sampled
  arbitrary train is intentionally deferred to 0.3.0-alpha1.

Important protocol relationships
--------------------------------

    segment_count = n_y / echo_train_length
    phase_blip_area = 1 / fov_phase
    gradient_flat_duration = readout_duration + 2*adc_guard_time
    echo_spacing = gradient_flat_duration + 2*readout_ramp_time + blip_duration

The phase-blip relationship is defined per readout interval, not for the
complete train. This keeps each blip independent of ETL. Changes to
``readout_duration`` are supported when the Python sequence is regenerated.
Interactive website-side resizing of the already sampled arbitrary train is
not claimed in this release.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import pypulseq_star as ppstar
from pypulseq_star.sequence import vary
from pypulseq_star.writers import GammaStarWriter, PulseqWriter

ORIENTATION = "axial"  # "axial", "coronal", or "sagittal"



def _launch_relationship_dashboard(
    seq: ppstar.Sequence,
    output_dir: Path,
    *,
    enabled: bool,
    graph_name: str,
    title: str,
) -> None:
    """Write and optionally open the Streamlit relationship dashboard.

    The dashboard is opt-in so routine demo execution never opens a browser.
    Install the optional dependencies with ``pip install -e ".[dashboard]"``.
    """

    if not enabled:
        return

    import sys

    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from apps.relationship_graph_spec import (
            launch_relationship_dashboard,
            relationship_dashboard_command,
            write_relationship_dashboard_json,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Dashboard support is optional. Install it with "
            "`python -m pip install -e \".[dashboard]\"`."
        ) from exc

    graph_path = write_relationship_dashboard_json(
        seq,
        output_dir / graph_name,
        title=title,
        include_protocol=True,
        include_system=True,
        include_relationships=True,
        include_derived=False,
        show_all_protocol_parameters=False,
    )
    print(f"Wrote relationship graph: {graph_path}")
    print("Dashboard command:")
    print(relationship_dashboard_command(graph_path))
    launch_relationship_dashboard(graph_path, wait=False)


def _ceil_to_raster(value: float, raster: float) -> float:
    return math.ceil(value / raster - 1e-12) * raster


def define_system_limits() -> ppstar.Opts:
    return ppstar.Opts(
        max_grad=28.0,
        grad_unit="mT/m",
        max_slew=120.0,
        slew_unit="mT/m/ms",
        rf_ringdown_time=20e-6,
        rf_dead_time=500e-6,
        adc_dead_time=120e-6,
        rf_raster_time=2e-6,
        grad_raster_time=10e-6,
        block_duration_raster=10e-6,
        adc_raster_time=1e-6,
        gamma=42.575575e6,
        max_rf=15e-6,
    )


def define_protocol(*, orientation: str = ORIENTATION, overrides: dict[str, object] | None = None) -> ppstar.Protocol:
    protocol = ppstar.Protocol(
        name="epi",
        description="Protocol-centered segmented 2D EPI sequence.",
        parameters={
            "sequence_name": "epi",
            "sequence_type": "EPI",
            "trajectory": "cartesian_epi",
            "orientation": orientation,
            "n_x": 64,
            "n_y": 64,
            "echo_train_length": 16,
            "fov_read": 0.220,
            "fov_phase": 0.220,
            "slice_thickness": 5e-3,
            "flip_angle": 90.0,
            "rf_duration": 3e-3,
            "echo_time": 30e-3,
            "repetition_time": 400e-3,
            "readout_duration": 640e-6,
            "adc_guard_time": 10e-6,
            "readout_ramp_time": 90e-6,
            "blip_duration": 80e-6,
            "prephaser_duration": 1.0e-3,
            "adc_dwell": 10e-6,
            "apodization": 0.5,
            "time_bw_product": 4.0,
            "rf_phase_offset": 0.0,
            "rf_frequency_offset": 0.0,
            "adc_phase_offset": 0.0,
            "adc_frequency_offset": 0.0,
            "compact_gamma_export": True,
        },
        aliases={
            "Name": "sequence_name",
            "TE": "echo_time",
            "TR": "repetition_time",
            "Orientation": "orientation",
        },
    )

    if overrides:
        protocol.parameters.update(overrides)

    p = protocol.symbols

    # Keep all geometry- and loop-dependent values as named protocol
    # relationships so gammaSTAR Protocol Control can update the plotted
    # sequence without stale Python-side constants.
    protocol.parameters["segment_count"] = p.n_y / p.echo_train_length
    protocol.parameters["readout_delta_k"] = 1.0 / p.fov_read
    protocol.parameters["phase_delta_k"] = 1.0 / p.fov_phase
    protocol.parameters["readout_area"] = p.n_x / p.fov_read
    protocol.parameters["readout_amplitude"] = (
        p.readout_area / p.readout_duration
    )
    protocol.parameters["gradient_flat_duration"] = (
        p.readout_duration + 2.0 * p.adc_guard_time
    )
    protocol.parameters["phase_blip_area"] = 1.0 / p.fov_phase
    protocol.parameters["phase_blip_amplitude"] = (
        2.0 * p.phase_blip_area / p.blip_duration
    )
    protocol.parameters["echo_spacing"] = (
        p.gradient_flat_duration
        + 2.0 * p.readout_ramp_time
        + p.blip_duration
    )
    protocol.parameters["phase_encode_start"] = -0.5 * p.n_y / p.fov_phase
    protocol.parameters["phase_encode_segment_step"] = (
        p.echo_train_length / p.fov_phase
    )

    return protocol


def _build_epi_train_waveforms(
    *,
    system: ppstar.Opts,
    echo_train_length: int,
    echo_spacing: float,
    readout_duration: float,
    gradient_flat_duration: float,
    readout_ramp_time: float,
    blip_duration: float,
    readout_amplitude: float,
    phase_blip_area: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build one numeric default EPI train for realization/export.

    The Gy waveform is later registered with a symbolic per-segment ``area``
    relationship. The gammaSTAR writer retains the normalized waveform shape
    and scales each selected segment from ``phase_blip_area``.
    """

    raster = float(system.grad_raster_time)
    ramp_time = _ceil_to_raster(readout_ramp_time, raster)
    blip_time = _ceil_to_raster(blip_duration, raster)
    ramp_samples = max(1, int(round(ramp_time / raster)))
    flat_samples = max(1, int(round(gradient_flat_duration / raster)))
    blip_samples = max(2, int(round(blip_time / raster)))
    lobe_samples = 2 * ramp_samples + flat_samples
    spacing_samples = int(round(echo_spacing / raster))

    if spacing_samples < lobe_samples:
        raise ValueError(
            "echo_spacing is too short for the requested readout lobe: "
            f"echo_spacing={echo_spacing:g} s, "
            f"minimum={lobe_samples * raster:g} s"
        )

    gap_samples = spacing_samples - lobe_samples
    if abs(gap_samples - blip_samples) > 1:
        raise ValueError(
            "echo_spacing must equal gradient_flat_duration + 2*readout_ramp_time "
            "+ blip_duration on the gradient raster. "
            f"Computed gap={gap_samples * raster:g} s, "
            f"requested blip_duration={blip_time:g} s."
        )
    if echo_train_length > 1 and gap_samples < 2:
        raise ValueError(
            "At least two gradient-raster samples are required between "
            "readout lobes to form a phase-encoding blip."
        )

    total_samples = echo_train_length * spacing_samples
    gx = np.zeros(total_samples, dtype=float)
    gy = np.zeros(total_samples, dtype=float)

    # Arbitrary-gradient values are center sampled.  Build the ramps at the
    # centers of the gradient-raster cells rather than at their right edges.
    # For a 90 us ramp on a 10 us raster, the samples therefore represent
    # 5, 15, ..., 85 us.  Linear extrapolation then gives exactly zero at
    # the event boundary and exactly the requested amplitude at 90 us.
    ramp_fraction = (np.arange(ramp_samples, dtype=float) + 0.5) / ramp_samples
    ramp_up = ramp_fraction
    ramp_down = 1.0 - ramp_fraction

    for echo_index in range(echo_train_length):
        start = echo_index * spacing_samples
        polarity = 1.0 if echo_index % 2 == 0 else -1.0

        gx[start : start + ramp_samples] = (
            polarity * readout_amplitude * ramp_up
        )
        flat_start = start + ramp_samples
        gx[flat_start : flat_start + flat_samples] = (
            polarity * readout_amplitude
        )
        fall_start = flat_start + flat_samples
        gx[fall_start : fall_start + ramp_samples] = (
            polarity * readout_amplitude * ramp_down
        )

        if echo_index >= echo_train_length - 1:
            continue

        blip_start = start + lobe_samples
        half = max(1, gap_samples // 2)
        triangle = np.concatenate(
            [
                np.linspace(0.0, 1.0, half + 1, endpoint=True)[1:],
                np.linspace(
                    1.0,
                    0.0,
                    gap_samples - half + 1,
                    endpoint=True,
                )[1:],
            ]
        )[:gap_samples]

        normalized_area = float(np.sum(triangle) * raster)
        gy[blip_start : blip_start + gap_samples] = (
            phase_blip_area / normalized_area
        ) * triangle


    max_grad_hz_per_m = float(system.max_grad)
    max_slew_hz_per_m_per_s = float(system.max_slew)
    readout_slew = abs(readout_amplitude) / ramp_time
    blip_peak = abs(phase_blip_area) / max(float(np.sum(triangle) * raster), 1e-20) if echo_train_length > 1 else 0.0
    blip_slew = blip_peak / max(0.5 * blip_time, raster)
    if abs(readout_amplitude) > max_grad_hz_per_m + 1e-9:
        raise ValueError("Readout gradient amplitude exceeds system.max_grad.")
    if readout_slew > max_slew_hz_per_m_per_s + 1e-6:
        raise ValueError("Readout gradient slew exceeds system.max_slew.")
    if blip_peak > max_grad_hz_per_m + 1e-9:
        raise ValueError("Phase blip amplitude exceeds system.max_grad.")
    if blip_slew > max_slew_hz_per_m_per_s + 1e-6:
        raise ValueError("Phase blip slew exceeds system.max_slew.")

    # Keep the ADC strictly inside the gradient flat using one explicit
    # guard interval on each side. This avoids depending on how a renderer
    # interpolates center-sampled arbitrary-gradient values at ramp/flat
    # boundaries. adc_dead_time remains a lower bound, not an added offset.
    adc_guard_time = 0.5 * max(0.0, gradient_flat_duration - readout_duration)
    adc_flat_delay = max(
        float(system.adc_dead_time),
        ramp_time + adc_guard_time,
    )

    return gx, gy, adc_flat_delay


def build_sequence(
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
    *,
    orientation: str = ORIENTATION,
    debug: bool = False,
) -> ppstar.Sequence:
    p = protocol.symbols

    n_y_default = int(protocol.parameters["n_y"])
    etl_default = int(protocol.parameters["echo_train_length"])
    if n_y_default < 1 or etl_default < 1:
        raise ValueError("n_y and echo_train_length must be positive integers.")
    if n_y_default % etl_default != 0:
        raise ValueError(
            "This compact segmented-EPI demo currently requires n_y to be "
            "divisible by echo_train_length. "
            f"Received n_y={n_y_default}, ETL={etl_default}."
        )

    seq = ppstar.Sequence(
        system=system,
        protocol=protocol,
        name=p.sequence_name,
        debug=debug,
    )

    protocol_orientation = protocol.get_parameter("slice_orientation", orientation)
    seq.set_encoding_frame(protocol_orientation)

    seq.set_definition(
        "FOV",
        [p.fov_read, p.fov_phase, p.slice_thickness],
    )
    seq.set_definition("Name", p.sequence_name)

    # One representative shot is retained. Numeric realization/writers own
    # materialization of the outer segment loop.
    segment = seq.set_node(
        "segment",
        role="shot",
        repeat_count=p.segment_count,
        repeat_every=p.repetition_time,
        counter="segment_index",
        repeat_mode="loop",
    )
    seq.set_node("segment.kernel", role="kernel")

    # ==================
    # CREATE BASE EVENTS
    # ==================
    rf, gz, gz_reph = ppstar.make_sinc_pulse(
        flip_angle=p.flip_angle * math.pi / 180.0,
        duration=p.rf_duration,
        slice_thickness=p.slice_thickness,
        apodization=p.apodization,
        time_bw_product=p.time_bw_product,
        phase_offset=p.rf_phase_offset,
        freq_offset=p.rf_frequency_offset,
        system=system,
        return_gz=True,
        delay=system.rf_dead_time,
        use="excitation",
        name="rf_excitation",
        gz_name="gz_slice_select",
        gzr_name="gz_rephaser",
    )
    gz.axis_role = "slice"
    gz_reph.axis_role = "slice"

    n_x = int(protocol.parameters["n_x"])
    fov_read = float(protocol.parameters["fov_read"])
    fov_phase = float(protocol.parameters["fov_phase"])
    readout_duration = float(protocol.parameters["readout_duration"])
    adc_guard_time = float(protocol.parameters["adc_guard_time"])
    gradient_flat_duration = readout_duration + 2.0 * adc_guard_time
    readout_ramp_time = float(protocol.parameters["readout_ramp_time"])
    blip_duration = float(protocol.parameters["blip_duration"])
    echo_spacing = (
        gradient_flat_duration
        + 2.0 * readout_ramp_time
        + blip_duration
    )

    readout_area_default = n_x / fov_read
    readout_amplitude_default = readout_area_default / readout_duration
    phase_blip_area_default = 1.0 / fov_phase

    gx_waveform, gy_waveform, adc_flat_delay = _build_epi_train_waveforms(
        system=system,
        echo_train_length=etl_default,
        echo_spacing=echo_spacing,
        readout_duration=readout_duration,
        gradient_flat_duration=gradient_flat_duration,
        readout_ramp_time=readout_ramp_time,
        blip_duration=blip_duration,
        readout_amplitude=readout_amplitude_default,
        phase_blip_area=phase_blip_area_default,
    )

    gx_train = ppstar.make_arbitrary_grad(
        channel="x",
        waveform=gx_waveform,
        system=system,
        name="gx_readout_train",
        role="readout_train",
        axis_role="read",
    )
    gy_train = ppstar.make_arbitrary_grad(
        channel="y",
        waveform=gy_waveform,
        system=system,
        name="gy_phase_blip_train",
        role="encoding_train",
        axis_role="phase",
    )

    gx_pre = ppstar.make_trapezoid(
        channel="x",
        axis_role="read",
        area=-0.5 * p.readout_area,
        duration=p.prephaser_duration,
        system=system,
        name="gx_readout_prephaser",
        role="prephasing",
    )

    # Unit area is intentional: segment.vary() supplies the protocol-driven
    # starting ky area for each segment.
    gy_pre = ppstar.make_trapezoid(
        channel="y",
        axis_role="phase",
        area=1.0,
        duration=p.prephaser_duration,
        system=system,
        name="gy_phase_prephaser",
        role="phase_encode",
    )

    adc = ppstar.make_adc_train(
        # The current ADC-train constructor validates a concrete default
        # window layout. Symbolic protocol values remain available in the
        # protocol graph, while the resolved defaults define this realization.
        num_samples=n_x,
        dwell=float(protocol.parameters["adc_dwell"]),
        duration=readout_duration,
        num_echoes=etl_default,
        first_delay=adc_flat_delay,
        echo_spacing=echo_spacing,
        mode="regular",
        polarity="alternating",
        trajectory="cartesian_epi",
        phase_offset=p.adc_phase_offset,
        freq_offset=p.adc_frequency_offset,
        name="adc_train",
        role="acquisition_train",
        dead_time=float(system.adc_dead_time),
        dead_time_policy="train_level_once",
        pulseq_export_mode="windows",
        parameters={
            # Generic writer contract: this regular ADC train shares one
            # segment clock with the gradients in the same logical block.
            # The writer therefore lowers each gradient segment together with
            # its matching ADC window rather than serializing the gradients
            # first and the ADC windows afterwards.
            "synchronized_with_block_gradients": True,
            "segment_duration": echo_spacing,
        },
        system=system,
    )

    # ================================================================
    # LOOP VARIATIONS: THE CORE RELATIONSHIP-CONTRACT DEMONSTRATION
    # ================================================================
    # 1) Segment-dependent phase prephasing:
    #       gy_pre.area = phase_encode_start
    #                     + segment_index * phase_encode_segment_step
    #
    # 2) Protocol-dependent phase-blip scaling:
    #       selected Gy segment area = 1 / fov_phase
    #
    # The relationship is per inner readout interval. ETL changes the number
    # of selected segments but does not rescale an individual blip.
    #
    # 3) Protocol-dependent readout timing:
    #       gradient_flat_duration = readout_duration + 2*adc_guard_time
    echo_spacing = gradient_flat_duration + 2*readout_ramp_time + blip_duration
    #
    # This keeps each ADC window aligned to the readout flat when the readout
    # duration changes.
    # ================================================================
    segment.vary(
        vary(
            gy_pre,
            "area",
            strength=p.phase_encode_start,
            step=p.phase_encode_segment_step,
        ),
        vary(
            gy_train,
            "amplitude",
            strength=p.phase_blip_amplitude,
            step=0.0,
        ),
        vary(
            gx_train,
            "amplitude",
            strength=p.readout_amplitude,
            step=0.0,
        ),
    )

    # ==================
    # ADD KERNEL BLOCKS
    # ==================
    seq.add_block(
        rf,
        gz,
        name="excitation",
        role="excitation",
        node="segment.kernel.excitation",
    )
    seq.add_block(
        gx_pre,
        gy_pre,
        gz_reph,
        name="prephase",
        role="prephase",
        node="segment.kernel.prephase",
    )
    seq.add_block(
        gx_train,
        gy_train,
        adc,
        name="echo_train",
        role="echo_train",
        node="segment.kernel.echo_train",
    )

    return seq


def main(
    *,
    orientation: str = ORIENTATION,
    plot: bool = True,
    write_seq: bool = True,
    write_json: bool = True,
    debug: bool = False,
    dashboard: bool = False,
    seq_filename: str | None = None,
    json_filename: str | None = None,
    protocol_overrides: dict[str, object] | None = None,
) -> None:
    # =============================
    # DEFINE SYSTEM AND PROTOCOL
    # =============================
    system = define_system_limits()
    protocol = define_protocol(
        orientation=orientation,
        overrides=protocol_overrides,
    )

    # ==================
    # BUILD THE SEQUENCE
    # ==================
    seq = build_sequence(
        system,
        protocol,
        orientation=orientation,
        debug=debug,
    )

    print(f"Orientation: {protocol.parameters.get('orientation', orientation)}")
    print(f"Read direction:  {seq.encoding_frame.read_dir}")
    print(f"Phase direction: {seq.encoding_frame.phase_dir}")
    print(f"Slice direction: {seq.encoding_frame.slice_dir}")

    # ===========================
    # RESOLVE SYMBOLIC RELATIONSHIPS
    # ===========================
    resolved = seq.resolve()

    # ==================
    # VALIDATE TIMING
    # ==================
    ok, report = resolved.check_timing()
    print("Timing check passed successfully" if ok else "Timing check failed")
    for error in report:
        print(error)

    print(f"n_x: {resolved.protocol.n_x}")
    print(f"n_y: {resolved.protocol.n_y}")
    print(f"ETL: {resolved.protocol.echo_train_length}")
    print(f"Segments: {resolved.protocol.segment_count:g}")
    print(f"FOV read:  {resolved.protocol.fov_read * 1e3:.1f} mm")
    print(f"FOV phase: {resolved.protocol.fov_phase * 1e3:.1f} mm")
    print(
        "Phase-blip area: "
        f"{resolved.protocol.phase_blip_area:.6g} 1/m"
    )
    print(
        "Readout ramp time: "
        f"{resolved.protocol.readout_ramp_time * 1e6:.1f} us"
    )
    print(
        "ADC guard time: "
        f"{resolved.protocol.adc_guard_time * 1e6:.1f} us per side"
    )
    print(
        "Gradient flat duration: "
        f"{resolved.protocol.gradient_flat_duration * 1e6:.1f} us"
    )
    print(
        "Blip duration: "
        f"{resolved.protocol.blip_duration * 1e6:.1f} us"
    )
    print(
        "Echo spacing: "
        f"{resolved.protocol.echo_spacing * 1e6:.1f} us"
    )

    # ==================
    # PLOT THE REALIZATION
    # ==================
    if plot:
        seq.plot(
            realization=resolved,
            repetitions=min(2, int(resolved.protocol.segment_count)),
            title=f"PyPulseq-Star Segmented EPI ({orientation})",
            gradient_scale="mt_per_m",
            debug=debug,
            one_tr=True,
        )

    # =================
    # WRITE DEMO OUTPUTS
    # =================
    # Match the FID/GRE demo convention: each demo owns one subdirectory.
    output_dir = Path("out") / "epi"
    output_dir.mkdir(parents=True, exist_ok=True)

    orientation_name = str(
        protocol.get_parameter("slice_orientation", orientation)
    ).lower()
    seq_filename = seq_filename or f"epi_{orientation_name}.seq"
    json_filename = json_filename or f"epi_{orientation_name}.seq.json"

    if write_seq:
        seq_path = PulseqWriter(
            seq,
            adc_train_policy="windows",
        ).write(
            output_dir / seq_filename,
            realization=resolved,
        )
        print(f"Wrote Pulseq: {seq_path}")

    if write_json:
        json_path = GammaStarWriter(seq).write(
            output_dir / json_filename,
            defaults=resolved,
        )
        print(f"Wrote gammaSTAR JSON: {json_path}")

    _launch_relationship_dashboard(
        seq,
        output_dir,
        enabled=dashboard,
        graph_name="epi.relationship_graph.json",
        title=f"PyPulseq-Star segmented EPI ({orientation_name}) relationship graph",
    )



def run_relationship_cases() -> None:
    """Generate focused protocol-edit cases without changing default paths."""

    cases = [
        ("ny128", {"n_y": 128}, "axial"),
        # Regeneration test only; interactive gammaSTAR resizing is deferred.
        ("readout960us_regenerated", {"readout_duration": 960e-6}, "axial"),
        ("etl8", {"echo_train_length": 8}, "axial"),
        ("coronal", {}, "coronal"),
        ("sagittal", {}, "sagittal"),
    ]
    for label, overrides, orientation in cases:
        print(f"\n=== RELATIONSHIP CASE: {label} ===")
        main(
            orientation=orientation,
            plot=False,
            seq_filename=f"epi_axial_{label}.seq",
            json_filename=f"epi_axial_{label}.seq.json",
            protocol_overrides=overrides,
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Protocol-centered segmented EPI demo."
    )
    parser.add_argument("--n-y", type=int, help="Override phase-encode matrix size.")
    parser.add_argument("--etl", type=int, help="Override echo-train length.")
    parser.add_argument(
        "--fov-phase",
        type=float,
        help="Override phase-direction FOV in metres.",
    )
    parser.add_argument(
        "--readout-duration",
        type=float,
        help="Override ADC readout duration in seconds.",
    )
    parser.add_argument("--te", type=float, help="Override echo time in seconds.")
    parser.add_argument("--tr", type=float, help="Override repetition time in seconds.")
    parser.add_argument(
        "--orientation",
        choices=("axial", "coronal", "sagittal"),
        default=ORIENTATION,
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable waveform plotting.")
    parser.add_argument("--no-seq", action="store_true", help="Disable Pulseq export.")
    parser.add_argument("--no-json", action="store_true", help="Disable gammaSTAR export.")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Generate relationship-graph JSON and open the Streamlit dashboard.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--relationship-cases", action="store_true")
    args = parser.parse_args()

    if args.relationship_cases:
        run_relationship_cases()
    else:
        overrides: dict[str, object] = {}
        if args.n_y is not None:
            overrides["n_y"] = args.n_y
        if args.etl is not None:
            overrides["echo_train_length"] = args.etl
        if args.fov_phase is not None:
            overrides["fov_phase"] = args.fov_phase
        if args.readout_duration is not None:
            overrides["readout_duration"] = args.readout_duration
        if args.te is not None:
            overrides["echo_time"] = args.te
        if args.tr is not None:
            overrides["repetition_time"] = args.tr

        main(
            orientation=args.orientation,
            plot=not args.no_plot,
            write_seq=not args.no_seq,
            write_json=not args.no_json,
            debug=args.debug,
            dashboard=args.dashboard,
            protocol_overrides=overrides,
        )
