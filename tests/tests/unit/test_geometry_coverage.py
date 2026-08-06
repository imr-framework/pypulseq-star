from __future__ import annotations

import numpy as np
import pytest

from pypulseq_star.geometry.encoding import (
    EncodingFrame,
    normalize_logical_axis,
    normalize_orientation,
    orientation_choices,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("alias", "expected"),
    [("tra", "axial"), ("transverse", "axial"), ("cor", "coronal"), ("sag", "sagittal")],
)
def test_orientation_aliases(alias: str, expected: str) -> None:
    assert normalize_orientation(alias) == expected
    assert expected in orientation_choices()


@pytest.mark.unit
def test_orientation_and_axis_validation_errors() -> None:
    with pytest.raises(ValueError, match="Unknown orientation"):
        normalize_orientation("diagonal")
    with pytest.raises(ValueError, match="logical axis or channel"):
        normalize_logical_axis(None)
    with pytest.raises(ValueError, match="Logical axis"):
        normalize_logical_axis("invalid")


@pytest.mark.unit
def test_encoding_frame_factories_mapping_oblique_and_serialization() -> None:
    coronal = EncodingFrame.from_value("cor", position=(1, 2, 3))
    assert coronal.name == "coronal"
    assert coronal.position == pytest.approx((1, 2, 3))
    assert coronal.read_dir == pytest.approx((1, 0, 0))
    assert coronal.phase_dir == pytest.approx((0, 0, 1))
    assert coronal.slice_dir == pytest.approx((0, -1, 0))
    assert np.linalg.det(np.asarray(coronal.rotation)) == pytest.approx(1.0)

    copied = EncodingFrame.from_value(coronal, name="copy")
    assert copied.name == "copy"
    assert copied.rotation == coronal.rotation

    mapped = EncodingFrame.from_value(coronal.to_dict())
    assert mapped.rotation == coronal.rotation
    assert mapped.position == coronal.position

    oblique = EncodingFrame.from_value(np.eye(3), name="oblique-test")
    assert oblique.name == "oblique-test"
    payload = oblique.to_dict()
    assert payload["transform4x4"][3] == [0.0, 0.0, 0.0, 1.0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"rotation": ((1, 0), (0, 1))}, "3 x 3"),
        ({"position": (0, 0)}, "three values"),
        ({"rotation": ((1, 0, 0), (0, 2, 0), (0, 0, 1))}, "orthonormal"),
        ({"rotation": ((1, 0, 0), (0, 1, 0), (0, 0, -1))}, "right-handed"),
    ],
)
def test_encoding_frame_rejects_invalid_frames(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EncodingFrame(**kwargs)
