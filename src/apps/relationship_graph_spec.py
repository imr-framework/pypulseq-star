"""Streamlit relationship graph adapter for PyPulseq-Star.

Builds a clean dashboard-ready graph specification from a SeqStar sequence.
The Streamlit app consumes this JSON-like spec and renders the graph.

Key behavior:
- sequence-general, not FID-specific
- compact protocol/system summaries
- deterministic lane layout
- cwd-independent path handling for terminal and VSCode Debug
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import shlex
import subprocess
import sys


def _project_root() -> Path:
    """Return the inner PyPulseq-Star project root.

    Expected file location:
        <project_root>/src/apps/relationship_graph_spec.py
    """

    return Path(__file__).resolve().parents[2]


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths against the inner project root, not process cwd."""

    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (_project_root() / path).resolve()


def build_relationship_dashboard_spec(
    seq: Any,
    *,
    title: str | None = None,
    include_protocol: bool = True,
    include_system: bool = True,
    include_relationships: bool = True,
    include_derived: bool = False,
    show_all_protocol_parameters: bool = False,
) -> dict[str, Any]:
    """Build a Streamlit-flow friendly dashboard spec from a SeqStar sequence."""

    from pypulseq_star.plotting.relationship_grapher import RelationshipGrapher

    grapher = RelationshipGrapher(
        seq,
        include_protocol=include_protocol,
        include_system=include_system,
        include_relationships=include_relationships,
        include_derived=include_derived,
        show_all_protocol_parameters=show_all_protocol_parameters,
    )
    graph = grapher.build()

    parameters = _get_parameters(seq)
    system = _get_system_dict(seq)
    system_summary_map = _extract_system_summary_map(system)

    sequence_name = str(getattr(seq, "name", "") or parameters.get("Name") or "Sequence")
    sequence_type = str(parameters.get("sequence_type") or "")

    node_records = _nodes_from_graph(graph)
    edge_records = _edges_from_graph(graph)

    return {
        "schema_version": "0.3",
        "title": title or _sequence_title(sequence_name, sequence_type),
        "sequence": {"name": sequence_name, "type": sequence_type},
        "protocol": {
            "summary": _summary_items(
                parameters,
                preferred=[
                    "TE",
                    "echo_time",
                    "TR",
                    "repetition_time",
                    "repetitions",
                    "average",
                    "averages",
                    "flip_angle",
                    "flip_angle_excitation",
                    "rf_duration",
                    "num_samples",
                    "dwell",
                ],
                max_items=7,
                aliases={
                    "TE": "Echo time (TE)",
                    "echo_time": "Echo time (TE)",
                    "TR": "Repetition time (TR)",
                    "repetition_time": "Repetition time (TR)",
                    "average": "Averages",
                    "averages": "Averages",
                    "repetitions": "Repetitions",
                    "flip_angle": "RF flip angle",
                    "flip_angle_excitation": "RF flip angle",
                    "rf_duration": "RF duration",
                    "num_samples": "Num samples",
                    "dwell": "Dwell time",
                },
            ),
            "parameters": _json_safe(parameters),
        },
        "system": {
            "summary": _summary_items(
                system_summary_map,
                preferred=[
                    "max_grad",
                    "max_slew",
                    "rf_ringdown_time",
                    "rf_dead_time",
                    "adc_dead_time",
                    "rf_raster_time",
                    "grad_raster_time",
                ],
                max_items=6,
                aliases={
                    "max_grad": "Max gradient",
                    "max_slew": "Max slew",
                    "rf_ringdown_time": "RF ringdown",
                    "rf_dead_time": "RF dead time",
                    "adc_dead_time": "ADC dead time",
                    "rf_raster_time": "RF raster",
                    "grad_raster_time": "Grad raster",
                },
            ),
            "parameters": _json_safe(system),
        },
        "nodes": node_records,
        "edges": edge_records,
        "layout": _lane_layout(node_records, edge_records),
        "legend": {
            "node_types": [
                {"kind": "root", "label": "Root / Entry"},
                {"kind": "loop", "label": "Aggregation / Summary"},
                {"kind": "block", "label": "Kernel"},
                {"kind": "module", "label": "Block / Module"},
                {"kind": "rf", "label": "RF event"},
                {"kind": "adc", "label": "ADC / readout"},
                {"kind": "gradient", "label": "Gradient"},
                {"kind": "relationship", "label": "Relationship"},
                {"kind": "system", "label": "System / External"},
            ],
            "edge_types": [
                {"kind": "contains", "label": "Contains"},
                {"kind": "protocol_dependency", "label": "Protocol dependency"},
                {"kind": "relationship_input", "label": "Timing / input"},
                {"kind": "relationship_output", "label": "Timing / output"},
                {"kind": "computes", "label": "Computes"},
            ],
        },
    }


def write_relationship_dashboard_json(
    seq: Any,
    out_path: str | Path,
    *,
    title: str | None = None,
    include_protocol: bool = True,
    include_system: bool = True,
    include_relationships: bool = True,
    include_derived: bool = False,
    show_all_protocol_parameters: bool = False,
) -> Path:
    """Write dashboard graph JSON and return the absolute path.

    Relative paths are resolved against the inner project root, not cwd.
    """

    out_path = _resolve_project_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spec = build_relationship_dashboard_spec(
        seq,
        title=title,
        include_protocol=include_protocol,
        include_system=include_system,
        include_relationships=include_relationships,
        include_derived=include_derived,
        show_all_protocol_parameters=show_all_protocol_parameters,
    )
    out_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return out_path


def relationship_dashboard_command(
    graph_json_path: str | Path,
    *,
    app_path: str | Path = "src/apps/relationship_graph_viewer.py",
) -> str:
    """Return a robust manual Streamlit command using absolute paths."""

    root = _project_root()
    app_path = _resolve_project_path(app_path)
    graph_json_path = _resolve_project_path(graph_json_path)
    return (
        f"cd {shlex.quote(str(root))} && "
        f"{shlex.quote(sys.executable)} -m streamlit run "
        f"{shlex.quote(str(app_path))} -- "
        f"--graph {shlex.quote(str(graph_json_path))}"
    )


def launch_relationship_dashboard(
    graph_json_path: str | Path,
    *,
    app_path: str | Path = "src/apps/relationship_graph_viewer.py",
    wait: bool = False,
) -> subprocess.Popen[Any] | subprocess.CompletedProcess[Any]:
    """Launch the Streamlit dashboard in a cwd-independent way."""

    root = _project_root()
    app_path = _resolve_project_path(app_path)
    graph_json_path = _resolve_project_path(graph_json_path)

    if not app_path.exists():
        raise FileNotFoundError(
            "Could not find relationship graph Streamlit app.\n"
            f"Expected app_path: {app_path}\n"
            f"Project root:       {root}\n"
            f"Current cwd:        {Path.cwd()}"
        )

    if not graph_json_path.exists():
        raise FileNotFoundError(
            "Could not find relationship graph JSON.\n"
            f"Expected graph:     {graph_json_path}\n"
            f"Project root:       {root}\n"
            f"Current cwd:        {Path.cwd()}"
        )

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--",
        "--graph",
        str(graph_json_path),
    ]

    print("[dashboard] launching:")
    print("    " + " ".join(shlex.quote(part) for part in command))
    print(f"[dashboard] cwd: {root}")

    if wait:
        return subprocess.run(command, cwd=root, check=False)
    return subprocess.Popen(command, cwd=root)


def _nodes_from_graph(graph: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for node in graph.nodes.values():
        kind = str(getattr(node, "kind", "generic"))
        node_id = str(getattr(node, "id", ""))
        label = str(getattr(node, "label", ""))

        if kind == "protocol_parameter" or node_id in {"prot", "sys"}:
            continue

        title, subtitle = _split_label(label, kind)
        metadata = getattr(node, "metadata", {}) or {}
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "title": title,
                "subtitle": subtitle,
                "kind": kind,
                "lane": _lane_for_kind(kind),
                "details": _details_from_metadata(metadata),
            }
        )
    return nodes


def _edges_from_graph(graph: Any) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    hidden_source_kinds = {"protocol_parameter"}
    hidden_node_ids = {"prot", "sys"}

    for index, edge in enumerate(graph.edges):
        source = str(getattr(edge, "source", ""))
        target = str(getattr(edge, "target", ""))
        kind = str(getattr(edge, "kind", "generic"))
        label = getattr(edge, "label", None)

        source_node = graph.nodes.get(source)
        target_node = graph.nodes.get(target)

        if source in hidden_node_ids or target in hidden_node_ids:
            continue

        if source_node is not None and getattr(source_node, "kind", "") in hidden_source_kinds:
            source = "protocol_summary"

        if target_node is not None and getattr(target_node, "kind", "") in hidden_source_kinds:
            continue

        normalized_kind = {
            "contains_event": "contains",
            "materializes_as_block": "contains",
            "block_after": "timing_sequence",
            "relationship_reference": "relationship_input",
            "relationship_target": "relationship_output",
            "protocol_required": "protocol_dependency",
            "protocol_independent": "protocol_dependency",
        }.get(kind, kind)

        edges.append(
            {
                "id": f"edge_{index}_{source}_{target}",
                "source": source,
                "target": target,
                "kind": normalized_kind,
                "raw_kind": kind,
                "label": str(label) if label else "",
            }
        )
    return edges


def _lane_layout(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a compact gammaSTAR-like hierarchy layout."""

    positions: dict[str, dict[str, float]] = {
        "protocol_summary": {"x": 80.0, "y": 30.0},
        "system_summary": {"x": 1060.0, "y": 30.0},
    }

    node_by_id = {str(node.get("id")): node for node in nodes}
    children: dict[str, list[str]] = {}
    parents: dict[str, str] = {}

    for edge in edges:
        if str(edge.get("kind")) != "contains":
            continue
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source not in node_by_id or target not in node_by_id:
            continue
        children.setdefault(source, []).append(target)
        parents.setdefault(target, source)

    roots = [
        node_id for node_id, node in node_by_id.items()
        if node.get("kind") == "root" and node_id not in parents
    ]

    depth: dict[str, int] = {}
    queue = [(root, 0) for root in roots]
    while queue:
        node_id, level = queue.pop(0)
        if node_id in depth and depth[node_id] <= level:
            continue
        depth[node_id] = level
        queue.extend((child, level + 1) for child in children.get(node_id, []))

    inferred_depth = {
        "root": 0,
        "loop": 1,
        "block": 2,
        "module": 3,
        "rf": 4,
        "adc": 4,
        "gradient": 4,
        "event": 4,
    }

    levels: dict[int, list[dict[str, Any]]] = {}
    for node in nodes:
        kind = str(node.get("kind", "generic"))
        if kind not in inferred_depth:
            continue
        level = depth.get(str(node.get("id")), inferred_depth[kind])
        levels.setdefault(level, []).append(node)

    y_by_level = {0: 90.0, 1: 200.0, 2: 310.0, 3: 440.0, 4: 590.0}
    for level, level_nodes in sorted(levels.items()):
        level_nodes.sort(key=_structure_sort_key)
        count = len(level_nodes)
        xs = [570.0] if count == 1 else _spread(100.0, 1040.0, count)
        for node, x in zip(level_nodes, xs):
            positions[str(node["id"])] = {
                "x": x,
                "y": y_by_level.get(level, 590.0 + 120.0 * (level - 4)),
            }

    rel_nodes = [
        node for node in nodes
        if node.get("kind") in {"relationship", "derived_parameter"}
    ]
    rel_nodes.sort(key=_relationship_sort_key)
    for node, x in zip(rel_nodes, _spread(120.0, 1020.0, len(rel_nodes))):
        positions[str(node["id"])] = {"x": x, "y": 760.0}

    for node in nodes:
        node_id = str(node.get("id"))
        positions.setdefault(node_id, {"x": 570.0, "y": 900.0})

    return {
        "positions": positions,
        "layout_mode": "compact_hierarchy",
        "lanes": [
            {"id": "structure", "label": "Sequence hierarchy", "y": 60, "height": 560},
            {"id": "relationships", "label": "Relationships / dependencies", "y": 690, "height": 180},
        ],
    }


def _get_parameters(seq: Any) -> dict[str, Any]:
    parameters = getattr(seq, "parameters", None)
    if isinstance(parameters, dict):
        return dict(parameters)
    if hasattr(seq, "protocol_dict"):
        try:
            value = seq.protocol_dict()
            if isinstance(value, dict):
                return dict(value)
        except Exception:
            pass
    return {}


def _get_system_dict(seq: Any) -> dict[str, Any]:
    system = getattr(seq, "system", None)
    if system is None:
        return {}
    if hasattr(system, "to_dict"):
        try:
            value = system.to_dict()
            if isinstance(value, dict):
                return dict(value)
        except Exception:
            pass
    try:
        return dict(vars(system))
    except Exception:
        return {"repr": repr(system)}


def _extract_system_summary_map(system: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in [
        "max_grad",
        "max_slew",
        "rf_ringdown_time",
        "rf_dead_time",
        "adc_dead_time",
        "rf_raster_time",
        "grad_raster_time",
    ]:
        value = _find_nested_value(system, key)
        if value is not None:
            summary[key] = value

    vendor = _find_nested_value(system, "vendor")
    model = _find_nested_value(system, "model")
    if vendor is not None:
        summary["vendor"] = vendor
    if model is not None:
        summary["model"] = model
    return summary


def _find_nested_value(obj: Any, key: str) -> Any | None:
    if isinstance(obj, Mapping):
        for k, value in obj.items():
            if str(k).lower() == key.lower():
                return value
        for value in obj.values():
            found = _find_nested_value(value, key)
            if found is not None:
                return found
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _find_nested_value(item, key)
            if found is not None:
                return found
    return None


def _summary_items(
    mapping: dict[str, Any],
    *,
    preferred: list[str],
    max_items: int,
    aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    aliases = aliases or {}
    keys: list[str] = []
    for key in preferred:
        if key in mapping and key not in keys:
            keys.append(key)
    if not keys:
        keys = list(mapping.keys())

    shown = keys[:max_items]
    items = [
        {"key": key, "label": aliases.get(key, key.replace("_", " ")), "value": _compact_value(mapping.get(key))}
        for key in shown
    ]

    remaining = max(0, len(mapping) - len(shown))
    if remaining:
        items.append({"key": "__remaining__", "label": "More", "value": f"+ {remaining} more"})
    return items


def _details_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in {"event", "relationship"}:
            details[key] = type(value).__name__
        else:
            details[key] = _json_safe(value)
    return details


def _lane_for_kind(kind: str) -> str:
    if kind in {"root", "loop", "block", "module"}:
        return "structure"
    if kind in {"rf", "adc", "gradient", "event"}:
        return "events"
    if kind == "relationship":
        return "relationships"
    if kind == "derived_parameter":
        return "derived"
    return "other"


def _structure_sort_key(node: dict[str, Any]) -> tuple[int, str]:
    kind_order = {"root": 0, "loop": 1, "block": 2, "module": 3}
    return (kind_order.get(node.get("kind", ""), 9), node.get("id", ""))


def _event_sort_key(node: dict[str, Any]) -> tuple[int, str]:
    """Sort by explicit event kind, never by user-defined labels."""

    kind_order = {
        "rf": 0,
        "gradient": 1,
        "adc": 2,
        "event": 3,
    }
    kind = str(node.get("kind", "event"))
    return (kind_order.get(kind, 9), str(node.get("id", "")))


def _relationship_sort_key(node: dict[str, Any]) -> tuple[int, str]:
    text = f"{node.get('id', '')} {node.get('label', '')}".lower()
    if "center" in text or "echo" in text or "te" in text:
        return (0, node.get("id", ""))
    if "repeat" in text or "tr" in text:
        return (1, node.get("id", ""))
    return (2, node.get("id", ""))


def _split_label(label: str, kind: str) -> tuple[str, str]:
    """Convert raw graph labels to cleaner title/subtitle pairs.

    Handles labels that may contain either real newlines or escaped "\\n".
    """

    text = str(label).replace("\\n", "\n").strip()

    if "\n" in text:
        first, rest = text.split("\n", 1)
        return first.strip(), rest.strip()

    if "Count:" in text:
        first, rest = text.split("Count:", 1)
        return first.strip(), f"Count: {rest.strip()}"

    if kind == "relationship":
        wrapped = _wrap_words(text, width=18)
        lines = wrapped.split("\n", 1)

        if len(lines) == 2:
            return lines[0].strip(), lines[1].strip()

        return wrapped.strip(), ""

    return text.strip(), ""

def _wrap_words(text: str, *, width: int = 18) -> str:
    words = text.split()
    if not words:
        return text
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "".join(lines)


def _sequence_title(name: str, sequence_type: str) -> str:
    if sequence_type:
        return f"{sequence_type}: {name}"
    return name


def _spread(start: float, stop: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [(start + stop) / 2.0]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def _compact_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-3:
            return f"{value:.3e}"
        return f"{value:.6g}"
    return str(value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        try:
            return _json_safe(asdict(value))
        except Exception:
            pass
    return repr(value)
