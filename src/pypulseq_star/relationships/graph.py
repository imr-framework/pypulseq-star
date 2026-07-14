"""Relationship graph/debug export utilities.

This module intentionally stays independent of the Streamlit/relationship
dashboard app.  It provides a stable, JSON-serializable relationship graph
payload that can be consumed by:

- ``ppstar.relationships.summary(seq)``
- ``ppstar.relationships.write_debug_json(seq, path)``
- future dashboard/viewer layers
- writer/debug validation passes

Phase 3-4 additions
-------------------
``seq.add_block(..., node="kernel.readout", role="readout")`` now creates
implicit ``timing.block_after`` relationships.  This graph exporter surfaces
those relationships alongside the node hierarchy and executable timeline so
developers can debug the three layers separately:

1. hierarchy: ``sequence -> kernel -> kernel.readout``
2. timeline: ``kernel.prephase -> kernel.te_delay -> kernel.readout``
3. relationships: explicit anchor/protocol constraints and implicit block order
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .relationship import get_relationships


def relationship_graph(seq: Any) -> dict[str, Any]:
    """Return a JSON-serializable relationship graph/debug payload.

    The returned payload is deliberately richer than the original Phase-1
    ``{"relationships": [...]}`` structure, but remains backward compatible:
    consumers that only need ``relationships`` can continue to read that key.

    New keys:
        ``nodes``
            SeqStar node registry, usually populated by ``seq.set_node`` and
            ``seq.add_block``.

        ``timeline``
            Repr-safe timeline debug payload from ``seq.timeline.to_debug_dict``.

        ``graph_nodes`` / ``graph_edges``
            Flattened visualization-oriented nodes/edges for hierarchy,
            timeline, protocol, and relationship layers.

        ``counts``
            Quick diagnostics for implicit vs explicit relationships.
    """

    relationships = get_relationships(seq)
    relationship_dicts = [_relationship_to_dict(rel) for rel in relationships]

    nodes = _extract_node_registry(seq)
    timeline = _extract_timeline_debug(seq)

    graph_nodes: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []

    _add_sequence_graph_node(seq, graph_nodes)
    _add_protocol_graph_nodes(seq, graph_nodes)
    _add_hierarchy_graph(seq, nodes, graph_nodes, graph_edges)
    _add_timeline_graph(timeline, graph_nodes, graph_edges)
    _add_relationship_graph(relationship_dicts, graph_nodes, graph_edges)

    graph_nodes = _deduplicate_graph_nodes(graph_nodes)
    graph_edges = _deduplicate_graph_edges(_normalize_graph_edges(graph_edges))

    counts = _relationship_counts(relationship_dicts)

    return _json_safe(
        {
            "schema": "seqstar.relationship_graph",
            "schema_version": 3,
            "sequence": str(getattr(seq, "name", "sequence")),
            "nodes": nodes,
            "timeline": timeline,
            "counts": counts,
            "relationship_count": len(relationships),
            "compact_graph": True,
            "relationships": relationship_dicts,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
        }
    )


def write_debug_json(seq: Any, path: str | Path) -> Path:
    """Write relationship/debug metadata as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(relationship_graph(seq), f, indent=2)

    return output_path


def summary(seq: Any) -> str:
    """Return a compact text summary of nodes, timeline, and relationships."""

    relationships = get_relationships(seq)
    nodes = _extract_node_registry(seq)
    timeline = _extract_timeline_debug(seq)
    relationship_dicts = [_relationship_to_dict(rel) for rel in relationships]
    counts = _relationship_counts(relationship_dicts)

    lines: list[str] = ["[seqstar graph]"]
    lines.append(f"  sequence: {getattr(seq, 'name', 'sequence')}")
    lines.append(f"  nodes: {len(nodes)}")
    lines.append(f"  blocks: {_timeline_block_count(timeline)}")
    lines.append(
        "  relationships: "
        f"{counts['total']} total, "
        f"{counts['implicit']} implicit, "
        f"{counts['explicit']} explicit"
    )

    if nodes:
        lines.append("")
        lines.append("[nodes]")
        for node_name, node in sorted(nodes.items()):
            parent = node.get("parent")
            role = node.get("role")
            repeat_every = node.get("repeat_every")
            repeat_count = node.get("repeat_count")
            counter = node.get("counter")
            details = []
            if parent:
                details.append(f"parent={parent}")
            if role:
                details.append(f"role={role}")
            if repeat_every is not None:
                details.append(f"repeat_every={repeat_every}")
            if repeat_count is not None:
                details.append(f"repeat_count={repeat_count}")
            if counter is not None:
                details.append(f"counter={counter}")
            detail_text = ", ".join(details)
            lines.append(f"  {node_name}" + (f" ({detail_text})" if detail_text else ""))

    blocks = _timeline_blocks(timeline)
    if blocks:
        lines.append("")
        lines.append("[timeline]")
        for block in blocks:
            index = block.get("index")
            node = block.get("node") or block.get("path") or block.get("name")
            prev = block.get("previous_block_node")
            role = block.get("role")
            varies = block.get("varies") or []
            prefix = f"  [{index}] " if index is not None else "  "
            line = f"{prefix}{node}"
            details = []
            if role:
                details.append(f"role={role}")
            if varies:
                details.append(f"varies={varies}")
            if prev:
                details.append(f"after={prev}")
            if details:
                line += " (" + ", ".join(details) + ")"
            lines.append(line)

    if not relationships:
        lines.append("")
        lines.append("[relationships]")
        lines.append("  NONE")
        return "\n".join(lines)

    lines.append("")
    lines.append("[relationships]")

    for rel in relationships:
        rel_dict = _relationship_to_dict(rel)
        metadata = rel_dict.get("metadata") or {}
        implicit = bool(metadata.get("implicit", False))
        source = metadata.get("source")
        developer_defined = rel_dict.get("developer_defined")
        if developer_defined is None:
            developer_defined = not implicit

        header_flags = []
        if implicit:
            header_flags.append("implicit")
        else:
            header_flags.append("explicit")
        if source:
            header_flags.append(f"source={source}")

        lines.append(f"  {rel.name} [{' '.join(header_flags)}]")
        lines.append(f"    type: {rel.relation_type}")
        lines.append(f"    developer_defined: {bool(developer_defined)}")
        lines.append(f"    description: {rel.description}")

        if rel.resolved:
            lines.append("    resolved:")
            for key, value in rel.resolved.items():
                if isinstance(value, float):
                    lines.append(f"      {key}: {value:.9g}")
                else:
                    lines.append(f"      {key}: {value}")

        if rel.validation:
            lines.append("    validation:")
            for item in rel.validation:
                status = "PASS" if item.passed else "FAIL"
                message = f": {item.message}" if item.message else ""
                lines.append(f"      {status}: {item.name}{message}")

    return "\n".join(lines)


def _relationship_to_dict(rel: Any) -> dict[str, Any]:
    """Return a JSON-safe relationship dictionary with normalized flags."""

    if hasattr(rel, "to_dict"):
        try:
            rel_dict = dict(rel.to_dict())
        except Exception:
            rel_dict = {}
    else:
        rel_dict = {}

    metadata = rel_dict.get("metadata")
    if not isinstance(metadata, dict):
        metadata = dict(getattr(rel, "metadata", {}) or {})
        rel_dict["metadata"] = metadata

    implicit = bool(metadata.get("implicit", False))
    if "developer_defined" not in rel_dict or rel_dict.get("developer_defined") is None:
        rel_dict["developer_defined"] = not implicit
    elif implicit:
        # Older SeqStarRelationship.to_dict() always returned True.  For
        # implicit add_block relationships, override the visualization flag.
        rel_dict["developer_defined"] = False

    rel_dict.setdefault("name", str(getattr(rel, "name", "relationship")))
    rel_dict.setdefault("relation_type", str(getattr(rel, "relation_type", "relationship")))
    rel_dict.setdefault("description", str(getattr(rel, "description", "")))
    rel_dict.setdefault("target", _json_safe(getattr(rel, "target", {}) or {}))
    rel_dict.setdefault("reference", _json_safe(getattr(rel, "reference", {}) or {}))
    rel_dict.setdefault("protocol_parameters", _json_safe(getattr(rel, "protocol_parameters", {}) or {}))
    rel_dict.setdefault("derived_parameters", _json_safe(getattr(rel, "derived_parameters", []) or []))
    rel_dict.setdefault("resolved", _json_safe(getattr(rel, "resolved", {}) or {}))

    return _json_safe(rel_dict)


def _relationship_counts(relationship_dicts: list[dict[str, Any]]) -> dict[str, int]:
    implicit = 0
    explicit = 0
    validation_only = 0
    block_after = 0
    anchor = 0
    repeat = 0

    for rel in relationship_dicts:
        metadata = rel.get("metadata") or {}
        relation_type = str(rel.get("relation_type", ""))
        if bool(metadata.get("implicit", False)):
            implicit += 1
        else:
            explicit += 1
        if bool(metadata.get("validation_only", False)):
            validation_only += 1
        if relation_type == "timing.block_after":
            block_after += 1
        if "anchor" in relation_type or "center" in relation_type:
            anchor += 1
        if "repeat" in relation_type:
            repeat += 1

    return {
        "total": len(relationship_dicts),
        "implicit": implicit,
        "explicit": explicit,
        "validation_only": validation_only,
        "block_after": block_after,
        "anchor": anchor,
        "repeat": repeat,
    }


def _extract_node_registry(seq: Any) -> dict[str, dict[str, Any]]:
    """Extract the SeqStar node registry from sequence/timeline metadata."""

    # Preferred: timeline owns the node registry.
    timeline = getattr(seq, "timeline", None)
    for candidate in (
        getattr(timeline, "nodes", None),
        getattr(timeline, "node_registry", None),
    ):
        if isinstance(candidate, Mapping):
            return {
                str(name): _normalize_node_dict(name, node)
                for name, node in candidate.items()
            }

    # Metadata fallback from the sequence node-registry implementation.
    metadata = getattr(seq, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("seqstar_nodes", "nodes", "node_registry"):
            candidate = metadata.get(key)
            if isinstance(candidate, Mapping):
                return {
                    str(name): _normalize_node_dict(name, node)
                    for name, node in candidate.items()
                }

    # Timeline debug fallback.
    timeline_debug = _extract_timeline_debug(seq)
    candidate = timeline_debug.get("nodes")
    if isinstance(candidate, Mapping):
        return {
            str(name): _normalize_node_dict(name, node)
            for name, node in candidate.items()
        }

    return {}


def _normalize_node_dict(name: Any, node: Any) -> dict[str, Any]:
    if isinstance(node, Mapping):
        out = dict(node)
    else:
        out = {
            "name": getattr(node, "name", str(name)),
            "node": getattr(node, "node", str(name)),
            "parent": getattr(node, "parent", None),
            "local_name": getattr(node, "local_name", None),
            "role": getattr(node, "role", None),
            "repeat_every": getattr(node, "repeat_every", None),
            "repeat_count": getattr(node, "repeat_count", None),
            "counter": getattr(node, "counter", None),
            "metadata": getattr(node, "metadata", {}),
        }

    node_name = str(out.get("node") or out.get("name") or name)
    parent, local_name = _infer_parent_and_local_name(node_name)

    out.setdefault("name", node_name)
    out.setdefault("node", node_name)
    out.setdefault("parent", parent)
    out.setdefault("local_name", local_name)
    out.setdefault("role", local_name)
    out.setdefault("metadata", {})

    return _json_safe(out)


def _extract_timeline_debug(seq: Any) -> dict[str, Any]:
    timeline = getattr(seq, "timeline", None)
    if timeline is not None and hasattr(timeline, "to_debug_dict"):
        try:
            payload = timeline.to_debug_dict()
            if isinstance(payload, Mapping):
                return _json_safe(dict(payload))
        except Exception as exc:
            return {
                "error": f"timeline.to_debug_dict failed: {exc}",
                "num_blocks": len(getattr(timeline, "blocks", []) or []),
                "blocks": [],
            }

    blocks = getattr(timeline, "blocks", []) if timeline is not None else []
    return {
        "nodes": {},
        "num_blocks": len(blocks or []),
        "blocks": [_block_to_debug_dict(block, index) for index, block in enumerate(blocks or [])],
    }


def _timeline_blocks(timeline_debug: Mapping[str, Any]) -> list[dict[str, Any]]:
    blocks = timeline_debug.get("blocks", [])
    if isinstance(blocks, list):
        return [dict(block) for block in blocks if isinstance(block, Mapping)]
    return []


def _timeline_block_count(timeline_debug: Mapping[str, Any]) -> int:
    value = timeline_debug.get("num_blocks")
    try:
        return int(value)
    except Exception:
        return len(_timeline_blocks(timeline_debug))


def _block_to_debug_dict(block: Any, index: int) -> dict[str, Any]:
    metadata = getattr(block, "metadata", None)
    if not isinstance(metadata, Mapping):
        metadata = {}

    node = metadata.get("seqstar_node") or getattr(block, "node", None) or getattr(block, "path", None) or getattr(block, "name", None)
    parent = metadata.get("seqstar_parent") or _infer_parent_and_local_name(str(node))[0]
    local_name = metadata.get("seqstar_local_name") or _infer_parent_and_local_name(str(node))[1]

    events = []
    for event_index, event in enumerate(getattr(block, "events", []) or []):
        events.append(
            {
                "index": event_index,
                "name": str(getattr(event, "name", event.__class__.__name__)),
                "type": event.__class__.__name__,
                "path": getattr(event, "path", None),
                "role": getattr(event, "role", None),
            }
        )

    return _json_safe(
        {
            "index": index,
            "name": getattr(block, "name", None),
            "path": getattr(block, "path", None),
            "node": node,
            "parent": parent,
            "local_name": local_name,
            "role": metadata.get("seqstar_role") or getattr(block, "role", None),
            "varies": metadata.get("seqstar_varies", []),
            "repeat_parent": metadata.get("seqstar_repeat_parent"),
            "counter": metadata.get("seqstar_counter"),
            "previous_block_node": metadata.get("seqstar_previous_block_node"),
            "events": events,
        }
    )


def _add_sequence_graph_node(seq: Any, graph_nodes: list[dict[str, Any]]) -> None:
    graph_nodes.append(
        {
            "id": "sequence",
            "label": str(getattr(seq, "name", "sequence")),
            "kind": "sequence",
            "layer": "hierarchy",
        }
    )


def _add_protocol_graph_nodes(seq: Any, graph_nodes: list[dict[str, Any]]) -> None:
    parameters = getattr(seq, "parameters", None)
    if not isinstance(parameters, Mapping):
        return

    for key, value in parameters.items():
        graph_nodes.append(
            {
                "id": f"protocol.{key}",
                "label": str(key),
                "kind": "protocol_parameter",
                "layer": "protocol",
                "value": _json_safe(value),
            }
        )


def _add_hierarchy_graph(
    seq: Any,
    nodes: Mapping[str, Mapping[str, Any]],
    graph_nodes: list[dict[str, Any]],
    graph_edges: list[dict[str, Any]],
) -> None:
    for node_name, node in nodes.items():
        parent = node.get("parent") or "sequence"
        role = node.get("role")
        graph_nodes.append(
            {
                "id": str(node_name),
                "label": str(node.get("local_name") or node_name),
                "kind": "seqstar_node",
                "layer": "hierarchy",
                "role": role,
                "parent": parent,
                "repeat_every": node.get("repeat_every"),
                "repeat_count": node.get("repeat_count"),
                "counter": node.get("counter"),
            }
        )
        if parent and str(node_name) != str(parent):
            graph_edges.append(
                {
                    "id": f"hierarchy:{parent}->{node_name}",
                    "source": str(parent),
                    "target": str(node_name),
                    "kind": "contains",
                    "layer": "hierarchy",
                    "implicit": True,
                }
            )


def _add_timeline_graph(
    timeline: Mapping[str, Any],
    graph_nodes: list[dict[str, Any]],
    graph_edges: list[dict[str, Any]],
) -> None:
    """Add a compact motif-level timeline graph.

    Repeated block occurrences are collapsed by logical ``node`` path. Event
    occurrences are collapsed by a stable event signature with generated block/
    event indices removed. This preserves the developer-facing sequence
    hierarchy while avoiding hundreds of repeated nodes in looped sequences.
    """

    blocks = _timeline_blocks(timeline)
    if not blocks:
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []

    for block in blocks:
        node = str(block.get("node") or block.get("path") or block.get("name"))
        if node not in grouped:
            grouped[node] = []
            order.append(node)
        grouped[node].append(block)

    for node in order:
        occurrences = grouped[node]
        block = occurrences[0]
        block_id = f"block:{node}"

        graph_nodes.append(
            {
                "id": block_id,
                "label": str(block.get("local_name") or block.get("role") or node.rsplit(".", 1)[-1]),
                "kind": "block",
                "layer": "timeline",
                "node": node,
                "role": block.get("role"),
                "index": block.get("index"),
                "occurrence_count": len(occurrences),
                "varies": block.get("varies") or [],
                "compact": True,
            }
        )

        graph_edges.append(
            {
                "id": f"node-block:{node}",
                "source": node,
                "target": block_id,
                "kind": "materializes_as_block",
                "layer": "timeline",
                "implicit": True,
            }
        )

        seen_events: set[str] = set()
        for occurrence in occurrences:
            for event in occurrence.get("events", []) or []:
                if not isinstance(event, Mapping):
                    continue

                raw_path = str(
                    event.get("path")
                    or f"{node}.{event.get('name', event.get('index'))}"
                )
                event_name = str(event.get("name") or raw_path.rsplit(".", 1)[-1])
                signature = _compact_event_signature(
                    event_name=event_name,
                    event_type=event.get("type"),
                    role=event.get("role"),
                    path=raw_path,
                )
                if signature in seen_events:
                    continue
                seen_events.add(signature)

                event_id = f"event:{node}:{signature}"
                graph_nodes.append(
                    {
                        "id": event_id,
                        "label": _compact_event_label(event_name),
                        "kind": _compact_event_kind(event.get("type"), event.get("role"), event_name),
                        "layer": "timeline",
                        "event_type": event.get("type"),
                        "role": event.get("role"),
                        "node": node,
                        "compact": True,
                    }
                )
                graph_edges.append(
                    {
                        "id": f"block-event:{node}->{signature}",
                        "source": block_id,
                        "target": event_id,
                        "kind": "contains_event",
                        "layer": "timeline",
                        "implicit": True,
                    }
                )

    for previous_node, target_node in zip(order, order[1:]):
        graph_edges.append(
            {
                "id": f"timeline:{previous_node}->{target_node}",
                "source": f"block:{previous_node}",
                "target": f"block:{target_node}",
                "kind": "block_after",
                "layer": "timeline",
                "implicit": True,
            }
        )


def _compact_event_signature(
    *,
    event_name: str,
    event_type: Any,
    role: Any,
    path: str,
) -> str:
    """Return a stable signature with generated occurrence indices removed."""

    text = str(event_name or path)
    text = re.sub(r"_b\d+", "", text)
    text = re.sub(r"_e\d+", "", text)
    text = re.sub(r"(?:^|[._-])\d{3,}(?=$|[._-])", "_", text)
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").lower()
    type_token = re.sub(r"[^A-Za-z0-9_]+", "_", str(event_type or "")).strip("_").lower()
    role_token = re.sub(r"[^A-Za-z0-9_]+", "_", str(role or "")).strip("_").lower()
    return "__".join(token for token in (text, type_token, role_token) if token) or "event"


def _compact_event_label(name: str) -> str:
    text = re.sub(r"_b\d+", "", str(name))
    text = re.sub(r"_e\d+", "", text)
    return text


def _compact_event_kind(event_type: Any, role: Any, name: str) -> str:
    """Classify event kind without relying on user-defined names.

    Classification uses explicit type metadata first. ``role`` and ``name`` are
    intentionally not used to infer ADC/RF/gradient semantics because users may
    choose arbitrary labels for blocks and events.
    """

    type_text = str(event_type or "").lower()

    if any(token in type_text for token in ("rf", "radiofrequency")):
        return "rf"

    if any(token in type_text for token in ("adc", "acquisition")):
        return "adc"

    if any(token in type_text for token in ("grad", "gradient", "trap", "trapezoid")):
        return "gradient"

    return "event"


def _add_relationship_graph(
    relationships: list[dict[str, Any]],
    graph_nodes: list[dict[str, Any]],
    graph_edges: list[dict[str, Any]],
) -> None:
    for rel in relationships:
        name = str(rel.get("name", "relationship"))
        relation_type = str(rel.get("relation_type", "relationship"))
        metadata = rel.get("metadata") or {}
        implicit = bool(metadata.get("implicit", False))

        relationship_node_id = f"relationship:{name}"
        graph_nodes.append(
            {
                "id": relationship_node_id,
                "label": name,
                "kind": "relationship",
                "layer": "relationship",
                "relation_type": relation_type,
                "implicit": implicit,
                "developer_defined": rel.get("developer_defined", not implicit),
                "source": metadata.get("source"),
            }
        )

        # Protocol parameter dependencies.
        protocol_parameters = rel.get("protocol_parameters") or {}
        if isinstance(protocol_parameters, Mapping):
            for key in ("required", "independent"):
                for parameter in protocol_parameters.get(key, []) or []:
                    parameter_id = f"protocol.{parameter}"
                    graph_edges.append(
                        {
                            "id": f"protocol-rel:{parameter}->{name}:{key}",
                            "source": parameter_id,
                            "target": relationship_node_id,
                            "kind": f"protocol_{key}",
                            "layer": "protocol",
                            "implicit": False,
                        }
                    )

        # Relationship endpoint edges.
        reference_id = _relationship_endpoint_id(rel.get("reference") or {}, prefix="reference")
        target_id = _relationship_endpoint_id(rel.get("target") or {}, prefix="target")

        if reference_id:
            graph_edges.append(
                {
                    "id": f"rel-reference:{reference_id}->{name}",
                    "source": reference_id,
                    "target": relationship_node_id,
                    "kind": "relationship_reference",
                    "layer": "relationship",
                    "implicit": implicit,
                }
            )

        if target_id:
            graph_edges.append(
                {
                    "id": f"rel-target:{name}->{target_id}",
                    "source": relationship_node_id,
                    "target": target_id,
                    "kind": "relationship_target",
                    "layer": "relationship",
                    "implicit": implicit,
                }
            )

        # Special-case resolved block_after edges so the graph is useful even if
        # a viewer does not parse the endpoint dictionaries.
        resolved = rel.get("resolved") or {}
        if relation_type == "timing.block_after":
            ref_node = resolved.get("reference_block_node") or resolved.get("reference_block")
            target_node = resolved.get("target_block_node") or resolved.get("target_block")
            if ref_node and target_node:
                graph_edges.append(
                    {
                        "id": f"block-after:{ref_node}->{target_node}:{name}",
                        "source": f"block:{ref_node}",
                        "target": f"block:{target_node}",
                        "kind": "block_after",
                        "layer": "relationship",
                        "relationship": name,
                        "implicit": True,
                    }
                )


def _relationship_endpoint_id(endpoint: Mapping[str, Any], *, prefix: str) -> str | None:
    if not isinstance(endpoint, Mapping):
        return None

    if endpoint.get("block") is not None:
        return f"block:{endpoint.get('block')}"
    if endpoint.get("node") is not None:
        return str(endpoint.get("node"))
    if endpoint.get("event") is not None:
        # We usually do not know the full event path here. Use an event-name
        # handle and let app-level viewers optionally merge this with timeline
        # event paths by name.
        return f"event_name:{endpoint.get('event')}"
    if endpoint.get("parameter") is not None:
        return f"protocol.{endpoint.get('parameter')}"

    return None


def _infer_parent_and_local_name(node: str) -> tuple[str, str]:
    node = str(node)
    if "." not in node:
        return "sequence", node
    parent, local_name = node.rsplit(".", 1)
    return parent or "sequence", local_name


def _normalize_graph_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add stable edge aliases for graph viewers and tests.

    Earlier graph payloads used ``kind`` for the semantic edge label and
    ``layer`` for the visual layer. Some consumers reasonably ask for
    ``edge["type"]`` / ``edge["edge_type"]`` instead. Keep all three:

    - ``type``: broad layer/category, e.g. hierarchy, timeline, protocol, relationship
    - ``edge_type``: semantic edge kind, e.g. contains, block_after, relationship_target
    - ``kind``: backward-compatible semantic edge kind
    """

    out: list[dict[str, Any]] = []

    for edge in edges:
        normalized = dict(edge)
        layer = str(normalized.get("layer") or "relationship")
        kind = str(normalized.get("kind") or normalized.get("edge_type") or layer)

        normalized.setdefault("type", layer)
        normalized.setdefault("edge_type", kind)
        normalized.setdefault("kind", kind)

        out.append(normalized)

    return out


def _deduplicate_graph_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes:
        node_id = str(node.get("id"))
        if node_id in seen:
            continue
        seen.add(node_id)
        out.append(node)
    return out


def _deduplicate_graph_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for edge in edges:
        edge_id = str(edge.get("id"))
        if edge_id in seen:
            continue
        seen.add(edge_id)
        out.append(edge)
    return out


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe representation for debug/export payloads."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass

    # Avoid serializing live Python event/block objects in graph JSON.
    return str(value)
