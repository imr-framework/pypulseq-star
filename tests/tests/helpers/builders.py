"""Small sequence builders shared by unit and integration tests."""

from __future__ import annotations

import math
from typing import Any

import pypulseq_star as ppstar


def build_fid_sequence(
    system: ppstar.Opts,
    *,
    averages: int = 2,
    te: float = 5e-3,
    tr: float = 20e-3,
    num_samples: int = 32,
    dwell: float = 10e-6,
) -> Any:
    """Build a compact relationship-aware FID sequence."""
    seq = ppstar.Sequence(
        system=system,
        name="test_fid",
        parameters={
            "Name": "test_fid",
            "sequence_type": "FID",
            "TE": te,
            "TR": tr,
            "averages": averages,
            "num_samples": num_samples,
            "dwell": dwell,
        },
    )
    seq.set_definition("Name", "test_fid")
    seq.set_node(
        "kernel",
        role="kernel",
        repeat_count="averages",
        repeat_every="repetition_time",
        counter="average_index",
        repeat_mode="loop",
    )

    rf = ppstar.make_block_pulse(
        flip_angle=math.pi / 2.0,
        duration=300e-6,
        system=system,
        use="excitation",
        name="rf_excitation",
    )
    delay = ppstar.make_delay(0.0, system=system, name="te_fill", role="echo_time_fill")
    adc = ppstar.make_adc(
        num_samples=num_samples,
        dwell=dwell,
        system=system,
        name="fid_adc",
        role="acquisition",
    )

    excitation = seq.add_block(rf, role="excitation", node="kernel.excitation")
    echo_delay = seq.add_block(delay, role="echo_time_fill", node="kernel.echo_delay")
    readout = seq.add_block(adc, role="readout", node="kernel.readout")

    ppstar.relationships.set_anchor_after(
        seq=seq,
        target=readout.get_event("fid_adc"),
        target_anchor="start",
        reference=excitation.get_event("rf_excitation"),
        reference_anchor="center",
        offset="echo_time",
        solve_event=echo_delay.get_event("te_fill"),
        solve_property="duration",
        name="fid_adc_start_after_rf_center",
    )
    ppstar.relationships.resolve(seq)
    return seq


def build_gradient_sequence(system: ppstar.Opts, *, repetitions: int = 3) -> Any:
    """Build a generic repeated gradient sequence for graph/writer tests."""
    seq = ppstar.Sequence(
        system=system,
        name="gradient_module",
        parameters={
            "Name": "gradient_module",
            "sequence_type": "GENERIC",
            "TR": 8e-3,
            "n_rep": repetitions,
        },
    )
    seq.set_definition("Name", "gradient_module")
    seq.set_node(
        "module",
        role="module",
        repeat_every="TR",
        repeat_count="n_rep",
        counter="rep_index",
        repeat_mode="expanded",
    )

    gx = ppstar.make_trapezoid(
        channel="x",
        amplitude=2.0,
        flat_time=0.8e-3,
        system=system,
        name="gx",
        role="encoding",
    )
    gy = ppstar.make_trapezoid(
        channel="y",
        amplitude=1.5,
        flat_time=0.7e-3,
        system=system,
        name="gy",
        role="encoding",
    )

    for _ in range(repetitions):
        seq.add_block(gx, role="x_encoding", node="module.x", varies=["rep_index"])
        seq.add_block(gy, role="y_encoding", node="module.y", varies=["rep_index"])

    ppstar.relationships.resolve(seq)
    return seq
