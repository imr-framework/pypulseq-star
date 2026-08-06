"""Generic protocol-to-encoding-frame regression tests."""
from __future__ import annotations

import numpy as np
import pytest

import pypulseq_star as ppstar


@pytest.mark.unit
@pytest.mark.parametrize(
    ("orientation", "read_dir", "phase_dir", "slice_dir"),
    [
        ("axial", (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        # Preserve a right-handed encoding frame:
        # read x phase = slice, so +x x +z = -y.
        ("coronal", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        ("sagittal", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    ],
)
def test_sequence_initializes_encoding_frame_from_protocol(
    orientation, read_dir, phase_dir, slice_dir
) -> None:
    protocol = ppstar.Protocol(
        name="orientation-test",
        parameters={"orientation": orientation},
    )
    # The Protocol canonicalizes the public alias.
    assert protocol.get_parameter("slice_orientation") == orientation

    seq = ppstar.Sequence(protocol=protocol)
    frame = seq.encoding_frame

    assert tuple(frame.read_dir) == pytest.approx(read_dir)
    assert tuple(frame.phase_dir) == pytest.approx(phase_dir)
    assert tuple(frame.slice_dir) == pytest.approx(slice_dir)


@pytest.mark.unit
@pytest.mark.parametrize("orientation", ["axial", "coronal", "sagittal"])
def test_encoding_frame_is_orthonormal_and_right_handed(orientation: str) -> None:
    protocol = ppstar.Protocol(
        name="orientation-handedness-test",
        parameters={"orientation": orientation},
    )
    frame = ppstar.Sequence(protocol=protocol).encoding_frame

    read = np.asarray(frame.read_dir, dtype=float)
    phase = np.asarray(frame.phase_dir, dtype=float)
    slice_ = np.asarray(frame.slice_dir, dtype=float)

    assert np.linalg.norm(read) == pytest.approx(1.0)
    assert np.linalg.norm(phase) == pytest.approx(1.0)
    assert np.linalg.norm(slice_) == pytest.approx(1.0)

    assert float(np.dot(read, phase)) == pytest.approx(0.0)
    assert float(np.dot(read, slice_)) == pytest.approx(0.0)
    assert float(np.dot(phase, slice_)) == pytest.approx(0.0)

    assert tuple(np.cross(read, phase)) == pytest.approx(tuple(slice_))
