"""Unit and smoke tests for Pulseq and gammaSTAR writers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pypulseq_star.writers import GammaStarWriter, PulseqWriter
from tests.helpers.builders import build_fid_sequence, build_gradient_sequence

pytestmark = pytest.mark.unit


def test_gammastar_document_has_core_sections(system) -> None:
    seq = build_fid_sequence(system)
    document = GammaStarWriter(seq).to_dict()

    assert document["name"] == "test_fid"
    assert isinstance(document.get("parameters"), dict)
    assert isinstance(document.get("sequence_elements"), dict)


def test_gammastar_preserves_loop_metadata(system) -> None:
    seq = build_fid_sequence(system, averages=3)
    document = GammaStarWriter(seq).to_dict()
    parameters = document["parameters"]

    assert any(key.endswith(".length") for key in parameters)
    assert any("counter" in key or "tstart" in key for key in parameters)


def test_gammastar_exports_generic_block_dependencies(system) -> None:
    seq = build_gradient_sequence(system)
    document = GammaStarWriter(seq).to_dict()
    parameters = document["parameters"]

    block_keys = [key for key in parameters if ".seqstar_blocks." in key]
    assert block_keys
    assert any(key.endswith(".tstart") for key in block_keys)
    assert any(key.endswith(".duration") for key in block_keys)
    assert any(key.endswith(".tend") for key in block_keys)


def test_writers_create_nonempty_files(system, output_dir: Path) -> None:
    seq = build_fid_sequence(system)

    seq_path = PulseqWriter(seq).write(output_dir / "fid.seq")
    json_path = GammaStarWriter(seq).write(output_dir / "fid.seq.json")

    assert seq_path.exists() and seq_path.stat().st_size > 0
    assert json_path.exists() and json_path.stat().st_size > 0
    assert json.loads(json_path.read_text(encoding="utf-8"))["name"] == "test_fid"


def test_gammastar_adc_geometry_defaults_are_consistent_without_gradients(system) -> None:
    """Gradient-free acquisitions still export complete gammaSTAR geometry."""

    seq = build_fid_sequence(system, num_samples=48)
    document = GammaStarWriter(seq).to_dict()
    parameters = document["parameters"]

    assert parameters["root.info.seq_dim"]["inputs"] == {
        "seq_dim": "root.prot.seq_dim"
    }
    assert "return 0" not in parameters["root.info.seq_dim"]["script"]

    assert parameters["root.prot.n_x"]["script"] == "return 48"
    assert parameters["root.prot.n_y"]["script"] == "return 1"
    assert parameters["root.prot.n_z"]["script"] == "return 1"

    assert parameters["root.mat_size"]["inputs"] == {
        "n_x": "root.prot.n_x",
        "n_y": "root.prot.n_y",
        "n_z": "root.prot.n_z",
    }
    assert parameters["root.acq_size"]["inputs"] == {
        "mat_size": "root.mat_size"
    }
    assert parameters["root.fov"]["inputs"] == {
        "fov": "root.prot.fov",
        "slice_thickness": "root.prot.slice_thickness",
    }

    adc_paths = [
        key.removesuffix(".trajectory")
        for key in parameters
        if key.endswith(".adc.trajectory")
    ]
    assert adc_paths

    for adc_path in adc_paths:
        assert parameters[f"{adc_path}.trajectory"]["script"] == "return {}"
        assert parameters[f"{adc_path}.header.matrix_size"]["inputs"] == {
            "mat_size": "root.mat_size"
        }
        assert parameters[f"{adc_path}.header.field_of_view"]["inputs"] == {
            "fov": "root.fov"
        }
