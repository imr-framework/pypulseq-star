"""GRE reference-sequence contract tests."""
from __future__ import annotations

import pytest

from .helpers import assert_timing_passes, evaluate, load_demo

pytestmark = [pytest.mark.sequence, pytest.mark.gre]


def test_gre_phase_encode_grid_tracks_matrix_and_fov(repo_root) -> None:
    demo = load_demo(repo_root, "GRE")
    protocol = demo.define_protocol()
    start = evaluate(protocol.parameters["phase_encode_start"], protocol)
    step = evaluate(protocol.parameters["phase_encode_step"], protocol)
    assert start == pytest.approx(-0.5 * 64 / 0.256)
    assert step == pytest.approx(1.0 / 0.256)

    protocol.parameters["n_y"] = 128
    protocol.parameters["fov"] = 0.128
    start = evaluate(protocol.parameters["phase_encode_start"], protocol)
    step = evaluate(protocol.parameters["phase_encode_step"], protocol)
    assert start == pytest.approx(-0.5 * 128 / 0.128)
    assert step == pytest.approx(1.0 / 0.128)


@pytest.mark.parametrize("orientation", ["axial", "coronal", "sagittal"])
def test_gre_orientation_builds_without_changing_logical_roles(repo_root, orientation) -> None:
    demo = load_demo(repo_root, "GRE")
    protocol = demo.define_protocol(orientation=orientation)
    seq, te_fill, tr_fill = demo.build_sequence(
        demo.define_system_limits(), protocol, orientation=orientation
    )
    assert protocol.parameters["orientation"] == orientation
    assert evaluate(te_fill, seq) >= 0.0
    assert evaluate(tr_fill, seq) >= 0.0
    assert_timing_passes(seq)


def test_gre_ky_loop_is_protocol_driven(repo_root) -> None:
    demo = load_demo(repo_root, "GRE")
    protocol = demo.define_protocol()
    seq, _, _ = demo.build_sequence(demo.define_system_limits(), protocol)
    kernel = seq.timeline.nodes["kernel"]

    # Do not use Python ``or`` on symbolic expressions: their truth value is
    # deliberately undefined before resolution. Prefer repeat_count and fall
    # back to the legacy factor field only when the value is actually absent.
    repeat_count = kernel.get("repeat_count")
    if repeat_count is None:
        repeat_count = kernel.get("factor")
    assert repeat_count is not None

    dependencies = set(getattr(repeat_count, "dependencies", ()))
    assert "protocol.n_y" in dependencies
    assert evaluate(repeat_count, {"n_y": 96}) == pytest.approx(96)
