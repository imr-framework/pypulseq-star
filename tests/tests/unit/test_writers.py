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
