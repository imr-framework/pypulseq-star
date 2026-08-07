"""Additional coverage for event, variation, and relationship primitives."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pypulseq_star.events.delay import SeqStarDelayEvent, _snap_time_to_raster
from pypulseq_star.relationships.expression import SeqStarExpression
from pypulseq_star.relationships.relationship import (
    SeqStarRelationship,
    attach_relationship,
    get_relationships,
)
from pypulseq_star.relationships.validation import SeqStarValidationResult
from pypulseq_star.sequence.vary import SeqStarVariation, vary

pytestmark = pytest.mark.unit


def test_delay_event_raster_modes_aliases_and_binding() -> None:
    assert _snap_time_to_raster(11e-6, 10e-6, mode="ceil") == pytest.approx(20e-6)
    assert _snap_time_to_raster(11e-6, 10e-6, mode="nearest") == pytest.approx(10e-6)
    assert _snap_time_to_raster(19e-6, 10e-6, mode="floor") == pytest.approx(10e-6)
    assert _snap_time_to_raster(11e-6, 0.0) == pytest.approx(11e-6)
    with pytest.raises(ValueError, match="non-negative"):
        _snap_time_to_raster(-1e-3, 10e-6)
    with pytest.raises(ValueError, match="Unsupported"):
        _snap_time_to_raster(1e-3, 10e-6, mode="bad")

    event = SeqStarDelayEvent(11e-6, raster=10e-6)
    assert event.duration == pytest.approx(20e-6)
    assert event.delay == event.duration
    event.delay = 21e-6
    assert event.duration == pytest.approx(30e-6)

    event = SeqStarDelayEvent(11e-6, parameters={"grad_raster_time": 5e-6})
    assert event.raster == pytest.approx(5e-6)
    event.bind_system(SimpleNamespace(grad_raster_time=2e-6, block_duration_raster=10e-6))
    assert event.raster == pytest.approx(2e-6)


def test_variation_factory_validation_and_record() -> None:
    event = SimpleNamespace(name="gy_pre", area=1.0)
    variation = vary(
        event,
        "area",
        strength=-2.0,
        step=0.5,
        mode="linear",
        wrap=4.0,
    )
    assert isinstance(variation, SeqStarVariation)
    assert variation.events == (event,)
    assert variation.attribute == "area"
    record = variation.to_record(node="kernel", counter="ky_index", factor=64)
    assert record["strength"] == -2.0
    assert record["step"] == 0.5
    assert record["event_names"] == ["gy_pre"]

    with pytest.raises(ValueError, match="at least one"):
        vary(None, "area", strength=0.0, step=1.0)
    with pytest.raises(ValueError, match="must not be empty"):
        vary(event, "", strength=0.0, step=1.0)
    with pytest.raises(ValueError, match="property name"):
        vary(event, "grad.area", strength=0.0, step=1.0)
    with pytest.raises(ValueError, match="Unsupported"):
        vary(event, "area", strength=0.0, step=1.0, mode="unknown")


def test_relationship_serialization_resolution_and_attachment() -> None:
    expression = SeqStarExpression(
        canonical="reference + offset",
        lua="return reference + offset",
        inputs={"reference": "root.rf.center", "offset": "root.prot.echo_time"},
    )
    relationship = SeqStarRelationship(
        name="echo_time",
        relation_type="center_to_center",
        description="RF-to-ADC echo time",
        target={"event": "adc", "anchor": "center"},
        reference={"event": "rf", "anchor": "center"},
        protocol_parameters={"echo_time": 5e-3},
        derived_parameters=["adc_delay"],
        expression=expression,
        metadata={"developer_defined": True, "object": SimpleNamespace(name="adc")},
    )
    validation = [
        SeqStarValidationResult(
            name="echo_time_feasible",
            passed=True,
            message="valid",
        )
    ]
    relationship.update_resolution(
        resolved={"target_value": 5e-3},
        validation=validation,
    )
    payload = relationship.to_dict()
    assert payload["relation_type"] == "center_to_center"
    assert payload["resolved"]["target_value"] == pytest.approx(5e-3)
    assert payload["metadata"]["object"]["name"] == "adc"
    assert expression.resolved_value == pytest.approx(5e-3)

    sequence = SimpleNamespace(relationships=[], metadata={})
    attach_relationship(sequence, relationship)
    assert get_relationships(sequence) == [relationship]

    metadata_only = SimpleNamespace(metadata={})
    attach_relationship(metadata_only, relationship)
    assert get_relationships(metadata_only) == [relationship]
