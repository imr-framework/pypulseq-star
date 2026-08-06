"""Phase 2 vision demo: protocol-centered, relationship-aware FID.

This file defines the intended PyPulseq-Star v0.2.0 developer contract.

Protocol-derived scalar values remain symbolic while the sequence is built.
Events retain both a symbolic specification and a resolved numerical
realization. The developer writes familiar PyPulseq-style arithmetic; the
expression graph is resolved numerically for Pulseq and preserved for
gammaSTAR.

This is a vision-level API target and is not expected to run until the Phase 2
protocol/expression contract is implemented.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pypulseq_star as ppstar
from pypulseq_star.writers import GammaStarWriter, PulseqWriter


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


def define_system_limits() -> ppstar.Opts:
    return ppstar.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=100,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=500e-6,
        adc_dead_time=50e-6,
        rf_raster_time=2e-6,
        grad_raster_time=10e-6,
    )


def define_protocol(
    *,
    overrides: dict[str, object] | None = None,
) -> ppstar.Protocol:
    protocol = ppstar.Protocol(
        name="fid",
        description="Protocol-centered free-induction-decay sequence.",
        parameters={
            "sequence_name": "fid",
            "sequence_type": "FID",
            # Scanner/export geometry remains valid even though an FID has no
            # spatial-encoding gradients. The generic gammaSTAR writer derives
            # the read matrix from ADC samples and uses these neutral geometry
            # values for acquisition headers.
            "seq_dim": "2D",
            "fov": 1.0,
            "slice_thickness": 1.0,
            "flip_angle": 90.0,
            "rf_duration": 300e-6,
            "echo_time": 20e-3,
            "repetition_time": 1.0,
            "average": 4,
            "num_samples": 2048,
            "dwell": 20e-6,
            "rf_phase_offset": 0.0,
            "rf_frequency_offset": 0.0,
            "adc_phase_offset": 0.0,
            "adc_frequency_offset": 0.0,
        },
        aliases={
            "Name": "sequence_name",
            "TE": "echo_time",
            "TR": "repetition_time",
            "averages": "average",
            "num_averages": "average",
        },
    )
    if overrides:
        protocol.parameters.update(overrides)
    return protocol


def build_sequence(
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
    *,
    debug: bool = False,
) -> tuple[ppstar.Sequence, ppstar.Expression, ppstar.Expression]:
    # Symbolic protocol references behave like scalars while retaining their
    # canonical name, source, units, dependencies, and default value.
    p = protocol.symbols

    seq = ppstar.Sequence(
        system=system,
        protocol=protocol,
        name=p.sequence_name,
        debug=debug,
    )

    seq.set_definition("Name", p.sequence_name)
    seq.set_node(
        "kernel",
        role="kernel",
        repeat_count=p.average,
        repeat_every=p.repetition_time,
        counter="average_index",
        repeat_mode="loop",
    )

    # ==================
    # CREATE BASE EVENTS
    # ==================
    rf = ppstar.make_block_pulse(
        flip_angle=p.flip_angle * math.pi / 180.0,
        duration=p.rf_duration,
        phase_offset=p.rf_phase_offset,
        freq_offset=p.rf_frequency_offset,
        use="excitation",
        system=system,
        name="rf_excitation",
    )

    adc = ppstar.make_adc(
        num_samples=p.num_samples,
        dwell=p.dwell,
        delay=system.adc_dead_time,
        phase_offset=p.adc_phase_offset,
        freq_offset=p.adc_frequency_offset,
        system=system,
        name="fid_adc",
        role="acquisition",
    )

    # =========================
    # COMPUTE THE SYMBOLIC FILLS
    # =========================
    # seq.duration(...) is the single duration API:
    #
    #   seq.duration(event)
    #   seq.duration([event_1, event_2])
    #   seq.duration(block)
    #   seq.duration([block_1, block_2])
    #   seq.duration(start_block_name="...", end_block_name="...")
    #
    # Optional start/end anchors select a partial duration within a single
    # event. Here:
    #
    #   rf, start="center"  -> RF center through the end of the RF event
    #   adc, end="start"    -> ADC event origin through active ADC start
    #
    # Therefore the remaining TE budget is explicit and readable.

    te_fixed_occupancy = seq.duration(
        start=rf.anchor("center"),
        end=adc.anchor("start"),
    )

    te_fill_duration = p.echo_time - te_fixed_occupancy

    te_fill = ppstar.make_delay(
        te_fill_duration,
        system=system,
        name="te_fill",
        role="echo_time_fill",
    )

    
    # ==================
    # ADD KERNEL BLOCKS
    # ==================
    seq.add_block(
        rf,
        name="excitation",
        role="excitation",
        node="kernel.excitation",
    )
    seq.add_block(
        te_fill,
        name="echo_delay",
        role="echo_time_fill",
        node="kernel.echo_delay",
    )
    seq.add_block(
        adc,
        name="readout",
        role="readout",
        node="kernel.readout",
    )

    # The TR budget is equally explicit. A named block range is inclusive.
    kernel_occupancy = seq.duration(
        start_block_name="excitation",
        end_block_name="readout",
    )
    tr_fill_duration = p.repetition_time - kernel_occupancy

    tr_fill = ppstar.make_delay(
        tr_fill_duration,
        system=system,
        name="tr_fill",
        role="repetition_time_fill",
    )
    seq.add_block(
        tr_fill,
        name="repetition_delay",
        role="repetition_time_fill",
        node="kernel.repetition_delay",
    )

    return seq, te_fill_duration, tr_fill_duration


def main(
    *,
    plot: bool = True,
    write_seq: bool = True,
    write_json: bool = True,
    debug: bool = False,
    dashboard: bool = False,
    protocol_overrides: dict[str, object] | None = None,
    seq_filename: str = "fid.seq",
    json_filename: str = "fid.seq.json",
) -> None:
    system = define_system_limits()
    protocol = define_protocol(overrides=protocol_overrides)
    seq, te_fill_duration, tr_fill_duration = build_sequence(
        system,
        protocol,
        debug=debug,
    )

    # eval(seq) evaluates against the sequence's current protocol defaults.
    # It is useful for inspection before a full immutable realization is made.
    print(f"Requested TE fill: {te_fill_duration.eval(seq) * 1e3:.3f} ms")
    print(f"Requested TR fill: {tr_fill_duration.eval(seq) * 1e3:.3f} ms")

    # Resolution evaluates all symbolic bindings, applies raster rules, checks
    # ownership and constraints, and returns an immutable numeric realization.
    resolved = seq.resolve()

    ok, report = resolved.check_timing()
    print("Timing check passed successfully" if ok else "Timing check failed")
    for error in report:
        print(error)

    print(f"TE: {resolved.protocol.echo_time * 1e3:.3f} ms")
    print(f"TR: {resolved.protocol.repetition_time * 1e3:.3f} ms")
    print(f"Averages: {resolved.protocol.num_averages}")
    print(f"Resolved TE fill: {te_fill_duration.eval(resolved) * 1e3:.3f} ms")
    print(f"Resolved TR fill: {tr_fill_duration.eval(resolved) * 1e3:.3f} ms")
    print(f"Kernel duration: {resolved.node('kernel').duration * 1e3:.3f} ms")

    if plot:
        seq.plot(
            realization=resolved,
            repetitions=2,
            title="PyPulseq-Star FID",
            debug=debug,
        )

    output_dir = Path("out/fid")
    output_dir.mkdir(parents=True, exist_ok=True)

    if write_seq:
        seq_path = PulseqWriter(seq).write(
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
        graph_name="fid.relationship_graph.json",
        title="PyPulseq-Star FID relationship graph",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Protocol-centered relationship-aware FID demo."
    )
    parser.add_argument("--averages", type=int, help="Override number of averages.")
    parser.add_argument("--num-samples", type=int, help="Override ADC sample count.")
    parser.add_argument("--flip-angle", type=float, help="Override RF flip angle in degrees.")
    parser.add_argument("--te", type=float, help="Override echo time in seconds.")
    parser.add_argument("--tr", type=float, help="Override repetition time in seconds.")
    parser.add_argument("--no-plot", action="store_true", help="Disable waveform plotting.")
    parser.add_argument("--no-seq", action="store_true", help="Disable Pulseq export.")
    parser.add_argument("--no-json", action="store_true", help="Disable gammaSTAR export.")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Generate relationship-graph JSON and open the Streamlit dashboard.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    overrides: dict[str, object] = {}
    if args.averages is not None:
        overrides["average"] = args.averages
    if args.num_samples is not None:
        overrides["num_samples"] = args.num_samples
    if args.flip_angle is not None:
        overrides["flip_angle"] = args.flip_angle
    if args.te is not None:
        overrides["echo_time"] = args.te
    if args.tr is not None:
        overrides["repetition_time"] = args.tr

    main(
        plot=not args.no_plot,
        write_seq=not args.no_seq,
        write_json=not args.no_json,
        debug=args.debug,
        dashboard=args.dashboard,
        protocol_overrides=overrides,
    )
