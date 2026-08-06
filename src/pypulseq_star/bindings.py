"""Declarative loop-dependent event-property bindings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def bind_event_property_to_loop_table(
    event: Any,
    *,
    property_name: str,
    protocol_table: str,
    counter: str,
    scale: float = 1.0,
    indices: Mapping[str, str] | None = None,
) -> Any:
    """Bind an event property to protocol data selected by loop counters."""

    property_name = str(property_name).strip()
    protocol_table = str(protocol_table).strip()
    counter = str(counter).strip()

    if not property_name:
        raise ValueError("property_name must not be empty.")
    if not protocol_table:
        raise ValueError("protocol_table must not be empty.")
    if not counter:
        raise ValueError("counter must not be empty.")

    binding = {
        "property": property_name,
        "protocol_table": protocol_table,
        "counter": counter,
        "scale": float(scale),
    }
    if indices:
        binding["indices"] = {
            str(name): str(counter_name)
            for name, counter_name in indices.items()
        }

    metadata = getattr(event, "metadata", None)
    if not isinstance(metadata, dict):
        try:
            event.metadata = {}
        except Exception:
            metadata = None
        else:
            metadata = event.metadata

    if isinstance(metadata, dict):
        metadata["seqstar_loop_binding"] = binding
        metadata.setdefault("seqstar_loop_bindings", {})[
            property_name
        ] = binding
    else:
        setattr(event, "_seqstar_loop_binding", binding)

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, dict):
        parameters.setdefault("_seqstar_loop_bindings", {})[
            property_name
        ] = binding

    return event


def bind_gradient_area_to_loop_table(
    gradient: Any,
    *,
    protocol_table: str,
    counter: str,
    scale: float = 1.0,
    indices: Mapping[str, str] | None = None,
) -> Any:
    """Bind a gradient area to a loop-indexed protocol table."""

    return bind_event_property_to_loop_table(
        gradient,
        property_name="area",
        protocol_table=protocol_table,
        counter=counter,
        scale=scale,
        indices=indices,
    )


__all__ = [
    "bind_event_property_to_loop_table",
    "bind_gradient_area_to_loop_table",
]
