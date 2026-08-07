"""EPI reference-sequence contract tests."""
from __future__ import annotations

import pytest

from .helpers import assert_timing_passes, evaluate, load_demo

pytestmark = [pytest.mark.sequence, pytest.mark.epi]


def test_epi_protocol_derivations_are_self_consistent(repo_root) -> None:
    demo = load_demo(repo_root, "EPI")
    protocol = demo.define_protocol()
    p = protocol.parameters
    assert evaluate(p["segment_count"], protocol) == pytest.approx(
        p["n_y"] / p["echo_train_length"]
    )
    assert evaluate(p["readout_delta_k"], protocol) == pytest.approx(1.0 / p["fov_read"])
    assert evaluate(p["phase_blip_area"], protocol) == pytest.approx(1.0 / p["fov_phase"])


def test_epi_fov_edit_updates_phase_blip(repo_root) -> None:
    demo = load_demo(repo_root, "EPI")
    protocol = demo.define_protocol(overrides={"fov_phase": 0.11})
    assert evaluate(protocol.parameters["phase_blip_area"], protocol) == pytest.approx(1.0 / 0.11)


def test_epi_matrix_and_etl_edit_updates_segments(repo_root) -> None:
    demo = load_demo(repo_root, "EPI")
    protocol = demo.define_protocol(overrides={"n_y": 128, "echo_train_length": 16})
    assert evaluate(protocol.parameters["segment_count"], protocol) == pytest.approx(8)


@pytest.mark.parametrize("orientation", ["axial", "coronal", "sagittal"])
def test_epi_orientation_smoke(repo_root, orientation) -> None:
    demo = load_demo(repo_root, "EPI")
    protocol = demo.define_protocol(orientation=orientation)
    seq = demo.build_sequence(demo.define_system_limits(), protocol, orientation=orientation)
    # Orientation is a sequence-level encoding contract. Some demo revisions
    # also expose it as a protocol parameter, but that UI detail is not required
    # for the logical-to-physical mapping itself.
    frame = seq.encoding_frame
    assert frame.name == orientation

    expected_directions = {
        "axial": {
            "read": (1.0, 0.0, 0.0),
            "phase": (0.0, 1.0, 0.0),
            "slice": (0.0, 0.0, 1.0),
        },
        "coronal": {
            "read": (1.0, 0.0, 0.0),
            "phase": (0.0, 0.0, 1.0),
            "slice": (0.0, -1.0, 0.0),
        },
        "sagittal": {
            "read": (0.0, 1.0, 0.0),
            "phase": (0.0, 0.0, 1.0),
            "slice": (1.0, 0.0, 0.0),
        },
    }[orientation]
    assert frame.read_dir == pytest.approx(expected_directions["read"])
    assert frame.phase_dir == pytest.approx(expected_directions["phase"])
    assert frame.slice_dir == pytest.approx(expected_directions["slice"])
    assert_timing_passes(seq)
