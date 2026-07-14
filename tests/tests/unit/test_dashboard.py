"""Tests for relationship-dashboard data generation.

The dashboard UI is optional. These tests validate its serializable input
contract without launching Streamlit or opening a browser.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers.builders import build_fid_sequence

pytestmark = [pytest.mark.unit, pytest.mark.optional]


def test_dashboard_json_is_created(system, tmp_path: Path) -> None:
    dashboard = pytest.importorskip("apps.relationship_graph_spec")
    seq = build_fid_sequence(system)
    output = tmp_path / "relationship_graph.json"

    result = dashboard.write_relationship_dashboard_json(
        seq,
        output,
        title="Test Relationship Graph",
        include_protocol=True,
        include_system=True,
        include_relationships=True,
        include_derived=False,
        show_all_protocol_parameters=False,
    )

    result_path = Path(result)
    assert result_path.exists()
    document = json.loads(result_path.read_text(encoding="utf-8"))
    assert document


def test_dashboard_command_references_json(system, tmp_path: Path) -> None:
    dashboard = pytest.importorskip("apps.relationship_graph_spec")
    path = tmp_path / "graph.json"
    path.write_text("{}", encoding="utf-8")

    command = dashboard.relationship_dashboard_command(path)
    assert str(path) in command
