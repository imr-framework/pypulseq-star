"""Protocol-centered, loop-native turbo spin echo (TSE) demo.

The demo follows the same developer workflow as ``demo_FID.py`` and
``demo_GRE.py`` while highlighting PyPulseq-Star's logical echo-train model:

* one representative echo motif is retained in the symbolic sequence;
* shot and echo loops are explicit protocol relationships;
* refocusing RF, crushers, phase encoding, readout, and rewind events are
  preserved beneath the echo loop;
* a one-window ``ADC train`` is used as the repeatable acquisition motif with
  ``dead_time_policy='train_level_once'``;
* ``echo_spacing`` is owned only by the repeated echo node; the Pulseq writer
  adds the residual inter-echo delay after concrete ADC lowering;
* gammaSTAR receives the symbolic sequence and a validated realization;
* Pulseq receives the concrete realization.

Release-level protocol tests
----------------------------

* ETL: ``echo_train_length`` changes the inner echo-loop length and shot count.
* Effective TE: for linear/reverse ordering,
  ``center_echo_number = TE_effective / echo_spacing`` places k-space center at
  the selected echo. Strict centric ordering places k-space center at echo 1.
* Ordering: ``linear``, strict center-out ``centric``, and ``reverse``
  regenerate the phase table without changing echo-train timing.
* Refocusing flip: all repeated refocusing pulses depend on one protocol value.
* Crushers: area and duration are protocol controls and remain rastered.
* Orientation: logical read/phase/slice axes are mapped by the EncodingFrame.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

import pypulseq_star as ppstar
from pypulseq_star.bindings import bind_gradient_area_to_loop_table
from pypulseq_star.writers import GammaStarWriter, PulseqWriter

ORIENTATION = "sagittal"  # "axial", "coronal", or "sagittal"



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
    return math.ceil(float(value) / float(raster) - 1e-12) * float(raster)


def _phase_encode_order(
    n_y: int,
    echo_train_length: int,
    *,
    order: str,
    center_echo_index: int,
) -> list[list[float]]:
    """Return a shot-major ky-index table.

    Values are dimensionless ky indices. The gradient binding multiplies the
    table by ``delta_k = 1/FOV``.

    ``linear`` and ``reverse`` place ky=0 at ``center_echo_index`` so the
    selected effective TE controls the k-space-center echo. ``centric`` is a
    strict center-out ordering: ky=0 is acquired at the first echo of the first
    shot and the remaining central lines are distributed across the first echo
    column before progressively more peripheral lines are assigned to later
    echoes. Consequently, strict centric ordering has an effective TE equal to
    one echo spacing.
    """

    if n_y < 1 or echo_train_length < 1:
        raise ValueError("n_y and echo_train_length must be positive.")
    if n_y % echo_train_length != 0:
        raise ValueError(
            "This compact TSE demo requires n_y to be divisible by ETL. "
            f"Received n_y={n_y}, ETL={echo_train_length}."
        )
    if not 0 <= center_echo_index < echo_train_length:
        raise ValueError(
            "The effective-TE echo must lie inside the echo train. "
            f"center_echo_index={center_echo_index}, ETL={echo_train_length}."
        )

    n_shots = n_y // echo_train_length
    order_key = str(order).strip().lower()

    if order_key == "linear":
        acquisition = np.arange(-n_y // 2, n_y // 2, dtype=float)
    elif order_key == "reverse":
        acquisition = np.arange(n_y // 2 - 1, -n_y // 2 - 1, -1, dtype=float)
    elif order_key == "centric":
        values: list[float] = [0.0]
        for radius in range(1, n_y):
            values.extend([-float(radius), float(radius)])
            if len(values) >= n_y:
                break
        acquisition = np.asarray(values[:n_y], dtype=float)
    else:
        raise ValueError(
            f"Unsupported phase_encode_order={order!r}; expected linear, "
            "centric, or reverse."
        )

    table = acquisition.reshape((n_shots, echo_train_length), order="F")

    # A strict center-out train starts at k-space center. Do not rotate the
    # centric table to a later effective-TE echo; doing so would turn it into a
    # reordered/interleaved trajectory rather than true centric acquisition.
    if order_key != "centric":
        zero_locations = np.argwhere(table == 0.0)
        zero_column = int(zero_locations[0, 1]) if len(zero_locations) else 0
        table = np.roll(table, center_echo_index - zero_column, axis=1)

    return table.tolist()


def define_system_limits() -> ppstar.Opts:
    return ppstar.Opts(
        max_grad=28.0,
        grad_unit="mT/m",
        max_slew=120.0,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
        rf_raster_time=2e-6,
        grad_raster_time=10e-6,
        block_duration_raster=10e-6,
        adc_raster_time=1e-6,
        gamma=42.575575e6,
    )


def define_protocol(
    *,
    orientation: str = ORIENTATION,
    overrides: dict[str, Any] | None = None,
) -> ppstar.Protocol:
    protocol = ppstar.Protocol(
        name="tse",
        description="Protocol-centered loop-native 2D turbo spin echo.",
        parameters={
            "sequence_name": "tse",
            "sequence_type": "TSE",
            "trajectory": "cartesian_tse",
            "orientation": orientation,
            "fov_read": 0.256,
            "fov_phase": 0.256,
            "n_x": 128,
            "n_y": 128,
            "slice_thickness": 5e-3,
            "echo_train_length": 8,
            "echo_spacing": 16e-3,
            "TE_effective": 64e-3,
            "repetition_time": 2.0,
            "phase_encode_order": "centric",
            "excitation_flip_angle": 90.0,
            "refocusing_flip_angle": 180.0,
            "excitation_duration": 2.5e-3,
            "refocusing_duration": 2.5e-3,
            "excitation_phase": math.pi / 2.0,
            "refocusing_phase": 0.0,
            "readout_duration": 6.4e-3,
            "readout_guard_time": 20e-6,
            "readout_rise_time": 250e-6,
            "crusher_area": 24.0,
            "crusher_duration": 1.0e-3,
            "spoiler_area": 48.0,
            "prephaser_duration": 1.0e-3,
            "apodization": 0.5,
            "time_bw_product": 4.0,
            "adc_phase_offset": 0.0,
            "adc_frequency_offset": 0.0,
            "compact_gamma_export": True,
        },
        aliases={
            "Name": "sequence_name",
            "ETL": "echo_train_length",
            "TE": "TE_effective",
            "TR": "repetition_time",
            "Orientation": "orientation",
        },
    )

    if overrides:
        protocol.parameters.update(overrides)

    # Some Protocol implementations normalize aliases and may not retain the
    # canonical orientation key. Preserve it explicitly for plotting, export,
    # and output naming.
    protocol.parameters.setdefault("orientation", orientation)

    p = protocol.symbols
    protocol.parameters["shot_count"] = p.n_y / p.echo_train_length
    protocol.parameters["delta_k_read"] = 1.0 / p.fov_read
    protocol.parameters["delta_k_phase"] = 1.0 / p.fov_phase
    protocol.parameters["readout_area"] = p.n_x / p.fov_read
    protocol.parameters["center_echo_number"] = p.TE_effective / p.echo_spacing
    protocol.parameters["center_echo_index"] = p.center_echo_number - 1.0
    protocol.parameters["center_echo_time"] = (
        p.center_echo_number * p.echo_spacing
    )

    n_y = int(protocol.parameters["n_y"])
    etl = int(protocol.parameters["echo_train_length"])
    effective_te = float(protocol.parameters["TE_effective"])
    echo_spacing = float(protocol.parameters["echo_spacing"])
    center_echo_number = int(round(effective_te / echo_spacing))
    if not math.isclose(
        effective_te,
        center_echo_number * echo_spacing,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "TE_effective must be an integer multiple of echo_spacing."
        )
    if center_echo_number < 1 or center_echo_number > etl:
        raise ValueError(
            "TE_effective selects an echo outside the train: "
            f"echo={center_echo_number}, ETL={etl}."
        )

    protocol.parameters["phase_encode_steps"] = _phase_encode_order(
        n_y,
        etl,
        order=str(protocol.parameters["phase_encode_order"]),
        center_echo_index=center_echo_number - 1,
    )

    # Declare how the literal default table is recomputed by gammaSTAR when
    # its source protocol controls change. The writer consumes this generic
    # metadata contract; the TSE-specific ordering policy remains in the demo.
    protocol.metadata.setdefault("computed_protocol_tables", {})[
        "phase_encode_steps"
    ] = {
        "generator": "phase_encode_order_table",
        "inputs": {
            "n_y": "n_y",
            "echo_train_length": "echo_train_length",
            "order": "phase_encode_order",
            "center_echo_index": "center_echo_index",
        },
    }
    return protocol


def build_sequence(
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
    *,
    orientation: str = ORIENTATION,
    debug: bool = False,
) -> tuple[ppstar.Sequence, ppstar.Expression, ppstar.Expression, ppstar.Expression]:
    p = protocol.symbols
    seq = ppstar.Sequence(
        system=system,
        protocol=protocol,
        name=p.sequence_name,
        debug=debug,
    )

    seq.set_encoding_frame(protocol.get_parameter("slice_orientation", orientation))
    seq.set_definition(
        "FOV",
        [p.fov_read, p.fov_phase, p.slice_thickness],
    )
    seq.set_definition("Name", p.sequence_name)

    seq.set_node(
        "shot",
        role="shot",
        repeat_count=p.shot_count,
        repeat_every=p.repetition_time,
        counter="shot_index",
        repeat_mode="loop",
    )
    seq.set_node("shot.kernel", role="kernel")
    seq.set_node(
        "shot.kernel.echo_train",
        role="echo_train",
        repeat_count=p.echo_train_length,
        repeat_every=p.echo_spacing,
        counter="echo_index",
        repeat_mode="loop",
    )

    # ==================
    # CREATE BASE EVENTS
    # ==================
    rf_exc, gz_exc, gz_exc_reph = ppstar.make_sinc_pulse(
        flip_angle=p.excitation_flip_angle * math.pi / 180.0,
        duration=p.excitation_duration,
        slice_thickness=p.slice_thickness,
        apodization=p.apodization,
        time_bw_product=p.time_bw_product,
        phase_offset=p.excitation_phase,
        system=system,
        return_gz=True,
        delay=system.rf_dead_time,
        use="excitation",
        name="rf_excitation",
        gz_name="gz_excitation",
        gzr_name="gz_excitation_rephaser",
    )
    rf_ref, gz_ref, _ = ppstar.make_sinc_pulse(
        flip_angle=p.refocusing_flip_angle * math.pi / 180.0,
        duration=p.refocusing_duration,
        slice_thickness=p.slice_thickness,
        apodization=p.apodization,
        time_bw_product=p.time_bw_product,
        phase_offset=p.refocusing_phase,
        system=system,
        return_gz=True,
        delay=system.rf_dead_time,
        use="refocusing",
        name="rf_refocusing",
        gz_name="gz_refocusing",
    )
    for gradient in (gz_exc, gz_exc_reph, gz_ref):
        gradient.axis_role = "slice"

    crusher_x_pre = ppstar.make_trapezoid(
        channel="x",
        axis_role="read",
        area=p.crusher_area,
        duration=p.crusher_duration,
        system=system,
        name="gx_crusher_pre",
        role="crusher",
    )
    crusher_z_pre = ppstar.make_trapezoid(
        channel="z",
        axis_role="slice",
        area=p.crusher_area,
        duration=p.crusher_duration,
        system=system,
        name="gz_crusher_pre",
        role="crusher",
    )
    crusher_x_post = ppstar.make_trapezoid(
        channel="x",
        axis_role="read",
        area=p.crusher_area,
        duration=p.crusher_duration,
        system=system,
        name="gx_crusher_post",
        role="crusher",
    )
    crusher_z_post = ppstar.make_trapezoid(
        channel="z",
        axis_role="slice",
        area=p.crusher_area,
        duration=p.crusher_duration,
        system=system,
        name="gz_crusher_post",
        role="crusher",
    )

    gy_phase = ppstar.make_trapezoid(
        channel="y",
        axis_role="phase",
        area=1.0,
        duration=p.crusher_duration,
        system=system,
        name="gy_phase_encode",
        role="phase_encode",
    )
    gy_rewind = ppstar.make_trapezoid(
        channel="y",
        axis_role="phase",
        area=-1.0,
        duration=p.crusher_duration,
        system=system,
        name="gy_phase_rewind",
        role="phase_rewind",
    )

    gx_readout = ppstar.make_trapezoid(
        channel="x",
        axis_role="read",
        flat_area=p.readout_area,
        flat_time=p.readout_duration + 2.0 * p.readout_guard_time,
        rise_time=p.readout_rise_time,
        system=system,
        name="gx_readout",
        role="readout",
    )

    n_x_default = int(protocol.parameters["n_x"])
    readout_duration_default = float(protocol.parameters["readout_duration"])
    guard_default = float(protocol.parameters["readout_guard_time"])
    rise_default = _ceil_to_raster(
        float(protocol.parameters["readout_rise_time"]),
        float(system.grad_raster_time),
    )
    adc_train = ppstar.make_adc_train(
        num_samples=n_x_default,
        duration=readout_duration_default,
        num_echoes=1,
        first_delay=max(
            float(system.adc_dead_time),
            rise_default + guard_default,
        ),
        echo_spacing=float(protocol.parameters["echo_spacing"]),
        mode="regular",
        polarity="positive",
        trajectory="cartesian_tse",
        phase_offset=p.adc_phase_offset,
        freq_offset=p.adc_frequency_offset,
        name="adc_echo_train_window",
        role="acquisition_train",
        dead_time=float(system.adc_dead_time),
        dead_time_policy="train_level_once",
        pulseq_export_mode="windows",
        system=system,
    )

    gx_spoiler = ppstar.make_trapezoid(
        channel="x",
        axis_role="read",
        area=p.spoiler_area,
        duration=p.crusher_duration,
        system=system,
        name="gx_echo_spoiler",
        role="spoiling",
    )

    # =========================
    # COMPUTE SYMBOLIC TIMING
    # =========================
    excitation_tail = seq.duration([rf_exc, gz_exc]) - rf_exc.anchor("center")
    excitation_rephase_occupancy = seq.duration(gz_exc_reph)
    crusher_pre_occupancy = seq.duration([crusher_x_pre, crusher_z_pre])
    refocusing_to_center = rf_ref.anchor("center")
    first_refocus_fill_duration = (
        0.5 * p.echo_spacing
        - excitation_tail
        - excitation_rephase_occupancy
        - crusher_pre_occupancy
        - refocusing_to_center
    )
    first_refocus_fill = ppstar.make_delay(
        first_refocus_fill_duration,
        system=system,
        name="first_refocus_fill",
        role="echo_time_fill",
    )

    refocusing_tail = seq.duration([rf_ref, gz_ref]) - rf_ref.anchor("center")
    post_crusher_occupancy = seq.duration(
        [crusher_x_post, crusher_z_post, gy_phase]
    )
    readout_to_center = adc_train.anchor("center")
    pre_readout_fill_duration = (
        0.5 * p.echo_spacing
        - refocusing_tail
        - post_crusher_occupancy
        - readout_to_center
    )
    pre_readout_fill = ppstar.make_delay(
        pre_readout_fill_duration,
        system=system,
        name="pre_readout_fill",
        role="echo_time_fill",
    )

    # ==================
    # ADD KERNEL BLOCKS
    # ==================
    seq.add_block(
        rf_exc,
        gz_exc,
        name="excitation",
        role="excitation",
        node="shot.kernel.excitation",
    )
    seq.add_block(
        gz_exc_reph,
        name="excitation_rephase",
        role="excitation_rephase",
        node="shot.kernel.excitation_rephase",
    )
    seq.add_block(
        first_refocus_fill,
        name="first_refocus_delay",
        role="echo_time_fill",
        node="shot.kernel.first_refocus_delay",
    )

    seq.add_block(
        crusher_x_pre,
        crusher_z_pre,
        name="crusher_pre",
        role="crusher_pre",
        node="shot.kernel.echo_train.crusher_pre",
    )
    seq.add_block(
        rf_ref,
        gz_ref,
        name="refocusing",
        role="refocusing",
        node="shot.kernel.echo_train.refocusing",
    )
    phase_block = seq.add_block(
        crusher_x_post,
        crusher_z_post,
        gy_phase,
        name="crusher_phase",
        role="crusher_post_phase_encode",
        node="shot.kernel.echo_train.crusher_phase",
    )
    seq.add_block(
        pre_readout_fill,
        name="pre_readout_delay",
        role="echo_time_fill",
        node="shot.kernel.echo_train.pre_readout_delay",
    )
    seq.add_block(
        gx_readout,
        adc_train,
        name="readout",
        role="readout",
        node="shot.kernel.echo_train.readout",
    )
    rewind_block = seq.add_block(
        gy_rewind,
        gx_spoiler,
        name="rewind_spoil",
        role="phase_rewind_spoiling",
        node="shot.kernel.echo_train.rewind_spoil",
    )

    echo_motif_occupancy = seq.duration(
        start_block_name="crusher_pre",
        end_block_name="rewind_spoil",
    )

    # Fail early with an actionable message rather than allowing the resolver
    # to encounter a negative delay. The echo motif includes the pre-crusher,
    # refocusing RF/slice gradient, post-crusher/phase encode, readout, and
    # rewind/spoiler blocks. echo_spacing must be at least this occupancy.
    default_echo_motif = float(echo_motif_occupancy.eval(seq))
    default_echo_spacing = float(protocol.parameters["echo_spacing"])
    if default_echo_spacing + 1e-12 < default_echo_motif:
        raise ValueError(
            "echo_spacing is shorter than the occupied TSE echo motif: "
            f"echo_spacing={default_echo_spacing * 1e3:.3f} ms, "
            f"minimum={default_echo_motif * 1e3:.3f} ms. "
            "Increase echo_spacing or shorten the RF, crusher, readout, "
            "guard, or spoiler durations."
        )

    # Do not add an explicit echo-period fill block inside the repeated node.
    # The node's repeat_every=p.echo_spacing is the single owner of inter-echo
    # padding. PulseqWriter materializes any remaining period delay after it
    # lowers the concrete motif (including ADC-train block padding). Keeping an
    # explicit fill here would double-count the same timing budget: once in the
    # motif and again in the nested-loop repeat-period padding.

    # ================================================================
    # NESTED LOOP BINDINGS
    # ================================================================
    # The same table drives phase encode and equal/opposite rewind. The writer
    # consumes generic nested-loop metadata; no TSE branch is required.
    phase_event = phase_block.get_event("gy_phase_encode")
    rewind_event = rewind_block.get_event("gy_phase_rewind")
    bind_gradient_area_to_loop_table(
        phase_event,
        protocol_table="phase_encode_steps",
        counter="echo_index",
        scale=float(1.0 / float(protocol.parameters["fov_phase"])),
        indices={"shot": "shot_index", "echo": "echo_index"},
    )
    bind_gradient_area_to_loop_table(
        rewind_event,
        protocol_table="phase_encode_steps",
        counter="echo_index",
        scale=float(-1.0 / float(protocol.parameters["fov_phase"])),
        indices={"shot": "shot_index", "echo": "echo_index"},
    )

    # gammaSTAR live editing uses a direct nested-loop expression rather than
    # indexing through a computed table. This avoids stale table caching when
    # a string-valued protocol control (linear/centric/reverse) changes.
    #
    # The writer contract is generic: the demo declares protocol inputs, loop
    # counters, and a Lua expression returning the event area. The writer only
    # lowers that declaration to backend-consumed ``grad.area`` and ``samples``.
    direct_area_script = (
        "local ny = math.floor((n_y or 1) + 0.5)\n"
        "local etl = math.floor((echo_train_length or 1) + 0.5)\n"
        "if ny < 1 or etl < 1 or ny % etl ~= 0 then return 0 end\n"
        "local nshots = ny / etl\n"
        "local s = math.floor((shot or 0) + 0.5)\n"
        "local e = math.floor((echo or 0) + 0.5)\n"
        "local target = math.floor((center_echo_index or 0) + 0.5)\n"
        "local key = string.lower(tostring(order or 'linear'))\n"
        "if key == 'centric' then target = 0 end\n"
        "local function acquisition_value(i)\n"
        "  if key == 'reverse' then return ny/2 - 1 - i end\n"
        "  if key == 'centric' then\n"
        "    if i == 0 then return 0 end\n"
        "    local radius = math.floor((i + 1) / 2)\n"
        "    if i % 2 == 1 then return -radius else return radius end\n"
        "  end\n"
        "  return -ny/2 + i\n"
        "end\n"
        "local zero_i = 0\n"
        "for i=0,ny-1 do if acquisition_value(i) == 0 then zero_i = i; break end end\n"
        "local zero_col = math.floor(zero_i / nshots)\n"
        "local shift = target - zero_col\n"
        "local source_col = (e - shift) % etl\n"
        "local source_i = s + source_col * nshots\n"
        "return acquisition_value(source_i) / fov_phase"
    )

    for event, sign in ((phase_event, 1.0), (rewind_event, -1.0)):
        event.metadata.pop("seqstar_nested_loop_binding", None)
        event.metadata["seqstar_nested_loop_expression_binding"] = {
            "property": "area",
            "protocol_inputs": {
                "n_y": "n_y",
                "echo_train_length": "echo_train_length",
                "order": "phase_encode_order",
                "center_echo_index": "center_echo_index",
                "fov_phase": "fov_phase",
            },
            "loop_inputs": {
                "shot": "outer",
                "echo": "inner",
            },
            "script": (
                direct_area_script
                if sign > 0
                else "return -(function()\n" + direct_area_script + "\nend)()"
            ),
        }
        event.metadata["gammastar_variant_policy"] = "vary"

    # Local plotting and Pulseq export currently lower loop-dependent events
    # through concrete repetition variants. Build these variants from the
    # same protocol table used by the symbolic gammaSTAR binding. This keeps
    # the phase-ordering policy in the demo while using the library's generic
    # row-major nested-loop variant mechanism.
    phase_table_default = protocol.parameters["phase_encode_steps"]
    fov_phase_default = float(protocol.parameters["fov_phase"])
    crusher_duration_default = float(protocol.parameters["crusher_duration"])
    phase_variants = []
    rewind_variants = []
    for shot_row in phase_table_default:
        for ky_index in shot_row:
            phase_area = float(ky_index) / fov_phase_default
            phase_variants.append(
                ppstar.make_trapezoid(
                    channel="y",
                    axis_role="phase",
                    area=phase_area,
                    duration=crusher_duration_default,
                    system=system,
                    name="gy_phase_encode",
                    role="phase_encode",
                )
            )
            rewind_variants.append(
                ppstar.make_trapezoid(
                    channel="y",
                    axis_role="phase",
                    area=-phase_area,
                    duration=crusher_duration_default,
                    system=system,
                    name="gy_phase_rewind",
                    role="phase_rewind",
                )
            )

    phase_event.metadata["seqstar_repetition_variants"] = phase_variants
    rewind_event.metadata["seqstar_repetition_variants"] = rewind_variants

    kernel_occupancy = (
        seq.duration(
            start_block_name="excitation",
            end_block_name="first_refocus_delay",
        )
        + p.echo_train_length * p.echo_spacing
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
        node="shot.kernel.repetition_delay",
    )

    return (
        seq,
        first_refocus_fill_duration,
        pre_readout_fill_duration,
        tr_fill_duration,
    )


def main(
    *,
    orientation: str = ORIENTATION,
    plot: bool = True,
    write_seq: bool = True,
    write_json: bool = True,
    debug: bool = False,
    dashboard: bool = False,
    protocol_overrides: dict[str, Any] | None = None,
    seq_filename: str | None = None,
    json_filename: str | None = None,
) -> None:
    # =============================
    # DEFINE SYSTEM AND PROTOCOL
    # =============================
    system = define_system_limits()
    protocol = define_protocol(
        orientation=orientation,
        overrides=protocol_overrides,
    )

    # Resolve the active orientation once and use it consistently. The
    # function argument remains the fallback for Protocol implementations that
    # normalize or omit the canonical key.
    active_orientation = str(
        protocol.get_parameter("slice_orientation", orientation)
    ).lower()
    protocol.parameters["orientation"] = active_orientation

    # ==================
    # BUILD THE SEQUENCE
    # ==================
    (
        seq,
        first_refocus_fill_duration,
        pre_readout_fill_duration,
        tr_fill_duration,
    ) = build_sequence(
        system,
        protocol,
        orientation=active_orientation,
        debug=debug,
    )

    print(f"Orientation: {active_orientation}")
    print(f"Read direction:  {seq.encoding_frame.read_dir}")
    print(f"Phase direction: {seq.encoding_frame.phase_dir}")
    print(f"Slice direction: {seq.encoding_frame.slice_dir}")
    print(
        "Requested first-refocus fill: "
        f"{first_refocus_fill_duration.eval(seq) * 1e3:.3f} ms"
    )
    print(
        "Requested pre-readout fill: "
        f"{pre_readout_fill_duration.eval(seq) * 1e3:.3f} ms"
    )
    print(
        f"Requested TR fill: {tr_fill_duration.eval(seq) * 1e3:.3f} ms"
    )

    # =========================
    # RESOLVE AND VALIDATE
    # =========================
    resolved = seq.resolve()
    ok, report = resolved.check_timing()
    print("Timing check passed successfully" if ok else "Timing check failed")
    for error in report:
        print(error)

    center_echo = int(round(resolved.protocol.center_echo_number))
    print(f"ETL: {resolved.protocol.echo_train_length}")
    print(f"Shots: {resolved.protocol.shot_count:g}")
    print(f"Echo spacing: {resolved.protocol.echo_spacing * 1e3:.3f} ms")
    print(f"Effective TE: {resolved.protocol.TE_effective * 1e3:.3f} ms")
    print(f"Center echo: {center_echo} (1-based)")
    print(f"Center echo time: {resolved.protocol.center_echo_time * 1e3:.3f} ms")
    print(f"Phase order: {resolved.protocol.phase_encode_order}")
    if str(resolved.protocol.phase_encode_order).strip().lower() == "centric":
        print(
            "Centric k-space center: first echo "
            f"({resolved.protocol.echo_spacing * 1e3:.3f} ms)."
        )
    else:
        print(
            "K-space center echo: "
            f"{int(round(resolved.protocol.center_echo_number))} "
            f"({resolved.protocol.center_echo_time * 1e3:.3f} ms)."
        )
    print(f"Refocusing flip: {resolved.protocol.refocusing_flip_angle:.1f} deg")
    print(
        "ADC train policy: one logical acquisition window per echo motif; "
        "dead time handled at train level."
    )

    # ==================
    # PLOT THE REALIZATION
    # ==================
    if plot:
        seq.plot(
            realization=resolved,
            repetitions=1,
            title=f"PyPulseq-Star TSE ({active_orientation})",
            gradient_scale="mt_per_m",
            debug=debug,
            one_tr=True,
        )

    # =================
    # WRITE DEMO OUTPUTS
    # =================
    output_dir = Path("out/tse")
    output_dir.mkdir(parents=True, exist_ok=True)
    orientation_name = active_orientation
    seq_filename = seq_filename or f"tse_{orientation_name}.seq"
    json_filename = json_filename or f"tse_{orientation_name}.seq.json"

    if write_seq:
        seq_path = PulseqWriter(
            seq,
            adc_train_policy="windows",
            allow_multiwindow_adc_with_other_events=True,
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
        graph_name="tse.relationship_graph.json",
        title=f"PyPulseq-Star TSE ({orientation_name}) relationship graph",
    )


def run_acceptance_cases() -> None:
    cases = [
        ("etl16", {"echo_train_length": 16}, "axial"),
        ("te96ms", {"TE_effective": 96e-3}, "axial"),
        ("centric", {"phase_encode_order": "centric"}, "axial"),
        ("reverse", {"phase_encode_order": "reverse"}, "axial"),
        ("refocus150", {"refocusing_flip_angle": 150.0}, "axial"),
        ("crusher32", {"crusher_area": 32.0}, "axial"),
        ("coronal", {}, "coronal"),
        ("sagittal", {}, "sagittal"),
    ]
    for label, overrides, orientation in cases:
        print(f"\n=== TSE ACCEPTANCE CASE: {label} ===")
        main(
            orientation=orientation,
            plot=False,
            protocol_overrides=overrides,
            seq_filename=f"tse_{label}.seq",
            json_filename=f"tse_{label}.seq.json",
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Protocol-centered loop-native TSE demo."
    )
    parser.add_argument("--n-y", type=int, help="Override phase-encode matrix size.")
    parser.add_argument("--etl", type=int, help="Override echo-train length.")
    parser.add_argument("--te-effective", type=float, help="Override effective TE in seconds.")
    parser.add_argument("--tr", type=float, help="Override repetition time in seconds.")
    parser.add_argument(
        "--fov-phase",
        type=float,
        help="Override phase-direction FOV in metres.",
    )
    parser.add_argument(
        "--phase-order",
        choices=("linear", "centric", "reverse"),
    )
    parser.add_argument("--refocusing-flip", type=float)
    parser.add_argument("--crusher-area", type=float)
    parser.add_argument("--crusher-duration", type=float)
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
    parser.add_argument("--acceptance-cases", action="store_true")
    args = parser.parse_args()

    if args.acceptance_cases:
        run_acceptance_cases()
    else:
        overrides: dict[str, Any] = {}
        if args.n_y is not None:
            overrides["n_y"] = args.n_y
        if args.etl is not None:
            overrides["echo_train_length"] = args.etl
        if args.te_effective is not None:
            overrides["TE_effective"] = args.te_effective
        if args.tr is not None:
            overrides["repetition_time"] = args.tr
        if args.fov_phase is not None:
            overrides["fov_phase"] = args.fov_phase
        if args.phase_order is not None:
            overrides["phase_encode_order"] = args.phase_order
        if args.refocusing_flip is not None:
            overrides["refocusing_flip_angle"] = args.refocusing_flip
        if args.crusher_area is not None:
            overrides["crusher_area"] = args.crusher_area
        if args.crusher_duration is not None:
            overrides["crusher_duration"] = args.crusher_duration

        main(
            orientation=args.orientation,
            plot=not args.no_plot,
            write_seq=not args.no_seq,
            write_json=not args.no_json,
            debug=args.debug,
            dashboard=args.dashboard,
            protocol_overrides=overrides,
        )
