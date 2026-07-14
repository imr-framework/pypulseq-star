"""Unit tests for timing relationships and relationship graphs."""

from __future__ import annotations

import pytest

import pypulseq_star as ppstar
from tests.helpers.builders import build_fid_sequence, build_gradient_sequence

pytestmark = pytest.mark.unit


def test_fid_anchor_relationship_realizes_requested_te(system) -> None:
    requested_te = 5e-3
    seq = build_fid_sequence(system, te=requested_te)

    rf = seq.timeline.blocks[0].get_event("rf_excitation")
    adc = seq.timeline.blocks[2].get_event("fid_adc")
    rf_center = ppstar.relationships.get_anchor_time(rf, "center", seq=seq, frame="global")
    adc_start = ppstar.relationships.get_anchor_time(adc, "start", seq=seq, frame="global")

    assert adc_start - rf_center == pytest.approx(requested_te, abs=seq.grad_raster_time)


def test_relationship_summary_includes_named_relationship(system) -> None:
    seq = build_fid_sequence(system)
    summary = ppstar.relationships.summary(seq)

    assert "fid_adc_start_after_rf_center" in summary


def test_relationship_graph_contains_hierarchy_and_timeline(system) -> None:
    seq = build_gradient_sequence(system)
    graph = ppstar.relationships.relationship_graph(seq)

    assert "nodes" in graph
    assert "timeline" in graph
    assert "graph_nodes" in graph
    assert "graph_edges" in graph
    assert "module" in graph["nodes"]


def test_relationship_graph_is_json_serializable(system) -> None:
    import json

    seq = build_gradient_sequence(system)
    graph = ppstar.relationships.relationship_graph(seq)
    json.dumps(graph)
