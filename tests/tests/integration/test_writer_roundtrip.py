"""Cross-layer writer smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pypulseq_star.writers import GammaStarWriter, PulseqWriter
from tests.helpers.builders import build_fid_sequence

pytestmark = [pytest.mark.integration, pytest.mark.smoke]


def test_fid_exports_both_representations(system, tmp_path: Path) -> None:
    seq = build_fid_sequence(system, averages=2)

    seq_path = PulseqWriter(seq).write(tmp_path / "fid.seq")
    json_path = GammaStarWriter(seq).write(tmp_path / "fid.seq.json")

    assert seq_path.read_text(encoding="utf-8")
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["name"] == "test_fid"
    assert document["parameters"]
    assert document["sequence_elements"]
