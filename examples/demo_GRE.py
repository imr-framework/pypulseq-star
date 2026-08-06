"""Phase 2 demo: protocol-centered, relationship-aware spoiled GRE.

This example follows the same developer contract as ``demo_FID.py``:

* protocol-derived scalar values remain symbolic while the sequence is built;
* one representative GRE kernel is retained in the symbolic sequence;
* the kernel is repeated through a logical phase-encoding loop;
* TE and TR fills are explicit symbolic expressions;
* ``seq.resolve()`` creates the immutable numeric realization used by Pulseq;
* gammaSTAR receives the symbolic sequence plus validated defaults.

Orientation is intentionally simple for developers: set ``ORIENTATION`` to
``"axial"``, ``"coronal"``, or ``"sagittal"``. Gradients are written in
logical imaging axes via ``axis_role="read"|"phase"|"slice"``; the library
maps them to scanner x/y/z using the active EncodingFrame.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pypulseq_star as ppstar
from pypulseq_star.writers import GammaStarWriter, PulseqWriter
from pypulseq_star.sequence import vary


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


def define_system_limits() -> ppstar.Opts:
    return ppstar.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
        rf_raster_time=2e-6,
        grad_raster_time=10e-6,
    )


def define_protocol(
    *,
    orientation: str = "axial",
    overrides: dict[str, object] | None = None,
) -> ppstar.Protocol:
    fov = 256e-3
    n_y = 64

    protocol = ppstar.Protocol(
        name="gre",
        description="Protocol-centered spoiled gradient-echo sequence.",
        parameters={
            "sequence_name": "gre",
            "sequence_type": "GRE",
            "orientation": orientation,
            "fov": fov,
            "n_x": 64,
            "n_y": n_y,
            "slice_thickness": 3e-3,
            "flip_angle": 10.0,
            "rf_duration": 3e-3,
            "readout_duration": 3.2e-3,
            "echo_time": 5e-3,
            "repetition_time": 12e-3,
            "apodization": 0.42,
            "time_bw_product": 4.0,
            "rf_spoiling_increment": 117.0,
            "prephaser_duration": 1e-3,
            "rf_phase_offset": 0.0,
            "rf_frequency_offset": 0.0,
            "adc_phase_offset": 0.0,
            "adc_frequency_offset": 0.0,
        },
        aliases={
            "Name": "sequence_name",
            "TE": "echo_time",
            "TR": "repetition_time",
            "Orientation": "orientation",
            "rf_spoiling_inc": "rf_spoiling_increment",
            "phase_encode_start": "phase_encode_start",
            "phase_encode_step": "phase_encode_step",
        },
    )

    if overrides:
        protocol.parameters.update(overrides)

    # Keep phase-encode limits as named protocol-level derived relationships.
    # The variation below then references p.phase_encode_start and
    # p.phase_encode_step directly. This is important for gammaSTAR Protocol
    # Control: if n_y is changed on the website, root.prot.phase_encode_start
    # must update with it rather than retaining the original 64-line default.
    #
    # Convention used here:
    #     ky(i) = -0.5 * n_y / fov + i / fov,  i = 0 ... n_y - 1
    #
    # This matches the usual even-matrix DFT-style phase-encode grid
    # [-Ny/2, ..., Ny/2 - 1] * Δk.
    p = protocol.symbols
    protocol.parameters["phase_encode_start"] = -0.5 * p.n_y / p.fov
    protocol.parameters["phase_encode_step"] = 1.0 / p.fov

    return protocol


def build_sequence(
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
    *,
    orientation: str = "axial",
    debug: bool = False,
) -> tuple[
    ppstar.Sequence,
    ppstar.Expression,
    ppstar.Expression,
]:
    p = protocol.symbols

    seq = ppstar.Sequence(
        system=system,
        protocol=protocol,
        name=p.sequence_name,
        debug=debug,
    )

    # Static Python-side orientation used for plotting and Pulseq export.
    # The gammaSTAR JSON also exposes root.prot.orientation for live switching.
    protocol_orientation = protocol.get_parameter("slice_orientation", orientation)
    if isinstance(getattr(protocol, "parameters", None), dict):
        protocol.parameters.setdefault("orientation", protocol_orientation)
    seq.set_encoding_frame(protocol_orientation)

    seq.set_definition(
        "FOV",
        [p.fov, p.fov, p.slice_thickness],
    )
    seq.set_definition("Name", p.sequence_name)

    # One representative kernel is retained. The numeric realization/writers
    # own materialization of the phase-encoding loop.
    kernel = seq.set_node(
        "kernel",
        role="kernel",
        factor=p.n_y,
        repeat_every=p.repetition_time,
        counter="ky_index",
        repeat_mode="loop",
    )

    delta_k = p.phase_encode_step

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

    # The RF helper currently returns slice gradients as conventional z-channel
    # gradients. Mark them as logical slice gradients so the orientation frame
    # can rotate them during plotting/export.
    gz.axis_role = "slice"
    gz_reph.axis_role = "slice"

    gx = ppstar.make_trapezoid(
        channel="x",
        axis_role="read",
        flat_area=p.n_x * delta_k,
        flat_time=p.readout_duration,
        system=system,
        name="gx_readout",
        role="readout",
    )

    adc = ppstar.make_adc(
        num_samples=p.n_x,
        duration=p.readout_duration,
        delay=system.adc_dead_time,
        phase_offset=p.adc_phase_offset,
        freq_offset=p.adc_frequency_offset,
        system=system,
        name="adc_readout",
        role="acquisition",
    )

    gx_pre = ppstar.make_trapezoid(
        channel="x",
        axis_role="read",
        area=-0.5 * gx.area,
        duration=p.prephaser_duration,
        system=system,
        name="gx_prephaser",
        role="prephasing",
    )

    # The representative phase encode uses unit area. kernel.vary() supplies
    # the live strength/step progression for every phase-encoding line.
    gy_pre = ppstar.make_trapezoid(
        channel="y",
        axis_role="phase",
        area=1.0,
        duration=p.prephaser_duration,
        system=system,
        name="gy_phase_encode",
        role="phase_encode",
    )

    # ================================================================
    # LOOP VARIATIONS: THE CORE RELATIONSHIP-CONTRACT DEMONSTRATION
    # ================================================================
    # This GRE demo intentionally keeps one representative kernel in the
    # symbolic sequence and declares how selected event properties vary across
    # the logical ky loop.  The writer is responsible for lowering these
    # semantic relationships to backend-consumed fields in Pulseq/gammaSTAR.
    #
    # 1) Phase encode gradient area
    #    gy_pre.area = phase_encode_start + ky_index * phase_encode_step
    #    with phase_encode_start and phase_encode_step defined as protocol
    #    relationships.  This is the k-space loop demonstration.
    #
    # 2) RF phase spoiling
    #    rf.phase_offset accumulates by rf_spoiling_increment each ky line and
    #    wraps at 2π.  In gammaSTAR this semantic phase_offset must drive the
    #    backend-consumed rf.phase field.
    #
    # 3) ADC phase alignment
    #    adc.phase_offset uses the same accumulated phase relationship so the
    #    receiver phase tracks the RF spoiling state.  In gammaSTAR this must
    #    drive the backend-consumed adc.phase field.
    #
    # Review-level expectation: node.vary(...) is generic. It is not a GRE
    # special case; developers can use the same abstraction for custom kernels,
    # RF/ADC phase/frequency changes, and gradient area/amplitude progressions.
    # ================================================================
    kernel.vary(
        vary(
            gy_pre,
            "area",
            strength=p.phase_encode_start,
            step=p.phase_encode_step,
        ),
        vary(
            [rf, adc],
            "phase_offset",
            strength=0.0,
            step=p.rf_spoiling_increment * math.pi / 180.0,
            mode="accumulated",
            wrap=2.0 * math.pi,
        ),
    )

    gx_spoil = ppstar.make_trapezoid(
        channel="x",
        axis_role="read",
        area=2.0 * p.n_x * delta_k,
        system=system,
        name="gx_spoil",
        role="spoiling",
    )
    gy_spoil = ppstar.make_trapezoid(
        channel="y",
        axis_role="phase",
        area=2.0 * p.n_y * delta_k,
        # Let the phase spoiler choose its own shortest feasible duration.
        # Coupling it to gx_spoil is invalid when n_y > n_x because the
        # phase-spoiler area is larger while the forced duration remains
        # sized for the read-spoiler area. All spoilers still begin in the
        # same block; the block duration is set by the longest event.
        system=system,
        name="gy_spoil",
        role="spoiling",
    )
    gz_spoil = ppstar.make_trapezoid(
        channel="z",
        axis_role="slice",
        area=4.0 / p.slice_thickness,
        system=system,
        name="gz_spoil",
        role="spoiling",
    )

    # =========================
    # COMPUTE THE SYMBOLIC FILLS
    # =========================
    # TE is defined from RF center to ADC center for GRE.
    excitation_tail = seq.duration([rf, gz]) - rf.anchor("center")
    prephase_occupancy = seq.duration(
        [gx_pre, gy_pre, gz_reph]
    )
    readout_to_echo = adc.anchor("center")

    te_fixed_occupancy = (
        excitation_tail
        + prephase_occupancy
        + readout_to_echo
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
        gz,
        name="excitation",
        role="excitation",
        node="kernel.excitation",
    )
    seq.add_block(
        gx_pre,
        gy_pre,
        gz_reph,
        name="prephase",
        role="prephase",
        node="kernel.prephase",
    )
    seq.add_block(
        te_fill,
        name="echo_delay",
        role="echo_time_fill",
        node="kernel.echo_delay",
    )
    seq.add_block(
        gx,
        adc,
        name="readout",
        role="readout",
        node="kernel.readout",
    )

    # Spoilers occur before the final repetition fill so that the complete
    # occupied kernel, not merely the readout, is budgeted against TR.
    seq.add_block(
        gx_spoil,
        gy_spoil,
        gz_spoil,
        name="spoiling",
        role="spoiling",
        node="kernel.spoiling",
    )

    kernel_occupancy = seq.duration(
        start_block_name="excitation",
        end_block_name="spoiling",
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
    orientation: str = ORIENTATION,
    plot: bool = True,
    write_seq: bool = True,
    write_json: bool = True,
    debug: bool = False,
    dashboard: bool = False,
    protocol_overrides: dict[str, object] | None = None,
    seq_filename: str | None = None,
    json_filename: str | None = None,
) -> None:
    system = define_system_limits()
    protocol = define_protocol(
        orientation=orientation,
        overrides=protocol_overrides,
    )

    seq, te_fill_duration, tr_fill_duration = build_sequence(
        system,
        protocol,
        orientation=orientation,
        debug=debug,
    )

    print(f"Orientation: {protocol.get_parameter("slice_orientation", orientation)}")
    print(f"Read direction:  {seq.encoding_frame.read_dir}")
    print(f"Phase direction: {seq.encoding_frame.phase_dir}")
    print(f"Slice direction: {seq.encoding_frame.slice_dir}")

    print(
        f"Requested TE fill: "
        f"{te_fill_duration.eval(seq) * 1e3:.3f} ms"
    )
    print(
        f"Requested TR fill: "
        f"{tr_fill_duration.eval(seq) * 1e3:.3f} ms"
    )

    resolved = seq.resolve()

    ok, report = resolved.check_timing()
    print("Timing check passed successfully" if ok else "Timing check failed")
    for error in report:
        print(error)

    print(f"TE: {resolved.protocol.echo_time * 1e3:.3f} ms")
    print(f"TR: {resolved.protocol.repetition_time * 1e3:.3f} ms")
    print(f"Phase encodes: {resolved.protocol.n_y}")
    print(f"Phase-encode start: {resolved.protocol.phase_encode_start:.6g} 1/m")
    print(f"Phase-encode step:  {resolved.protocol.phase_encode_step:.6g} 1/m")
    print(
        f"Resolved TE fill: "
        f"{te_fill_duration.eval(resolved) * 1e3:.3f} ms"
    )
    print(
        f"Resolved TR fill: "
        f"{tr_fill_duration.eval(resolved) * 1e3:.3f} ms"
    )
    print(
        f"Kernel duration: "
        f"{resolved.node('kernel').duration * 1e3:.3f} ms"
    )

    if plot:
        seq.plot(
            realization=resolved,
            repetitions=2,
            title=f"PyPulseq-Star Spoiled GRE ({orientation})",
            gradient_scale="mt_per_m",
            debug=debug,
            one_tr=True,
        )



    output_dir = Path("out/gre")
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_filename = seq_filename or f"gre_{orientation}.seq"
    json_filename = json_filename or f"gre_{orientation}.seq.json"

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
        graph_name="gre.relationship_graph.json",
        title=f"PyPulseq-Star spoiled GRE ({orientation}) relationship graph",
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Protocol-centered relationship-aware spoiled GRE demo."
    )
    parser.add_argument("--n-y", type=int, help="Override phase-encode matrix size.")
    parser.add_argument("--fov", type=float, help="Override in-plane FOV in metres.")
    parser.add_argument(
        "--rf-spoiling-increment",
        type=float,
        help="Override RF/ADC spoiling increment in degrees.",
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
    args = parser.parse_args()

    overrides: dict[str, object] = {}
    if args.n_y is not None:
        overrides["n_y"] = args.n_y
    if args.fov is not None:
        overrides["fov"] = args.fov
    if args.rf_spoiling_increment is not None:
        overrides["rf_spoiling_increment"] = args.rf_spoiling_increment
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
