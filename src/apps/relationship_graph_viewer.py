"""PyPulseq-Star Relationship Graph Visualizer.

Run from the repository root:

    streamlit run src/apps/relationship_graph_viewer.py -- --graph out/fid/fid.relationship_graph.json
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from streamlit_flow import streamlit_flow
    from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
    from streamlit_flow.state import StreamlitFlowState
    try:
        from streamlit_flow.layouts import ManualLayout
    except Exception:
        ManualLayout = None
except ImportError as exc:
    streamlit_flow = None
    StreamlitFlowNode = None
    StreamlitFlowEdge = None
    StreamlitFlowState = None
    ManualLayout = None
    _STREAMLIT_FLOW_IMPORT_ERROR = exc
else:
    _STREAMLIT_FLOW_IMPORT_ERROR = None


PALETTE = {
    "root": {"bg": "#1fc9c7", "border": "#079b9d", "fg": "#ffffff"},
    "loop": {"bg": "#eef4ff", "border": "#2d6cdf", "fg": "#174ea6"},
    "block": {"bg": "#f4f4f5", "border": "#8b8b8f", "fg": "#262629"},
    "module": {"bg": "#eef4ff", "border": "#2d6cdf", "fg": "#174ea6"},
    "rf": {"bg": "#eef9ee", "border": "#58a850", "fg": "#236a27"},
    "adc": {"bg": "#f4efff", "border": "#8c6be8", "fg": "#5e3bc4"},
    "gradient": {"bg": "#eef8ff", "border": "#3d9ddd", "fg": "#0f5d90"},
    "event": {"bg": "#eef9ee", "border": "#58a850", "fg": "#236a27"},
    "relationship": {"bg": "#fff7de", "border": "#d09113", "fg": "#5b3b00"},
    "derived_parameter": {"bg": "#f7ecff", "border": "#b173d1", "fg": "#51256b"},
    "system": {"bg": "#f7f8fa", "border": "#98a2b3", "fg": "#344054"},
    "protocol": {"bg": "#f7ffff", "border": "#16a7a7", "fg": "#067c7c"},
    "generic": {"bg": "#eeeeee", "border": "#999999", "fg": "#222222"},
}

EDGE_STYLES = {
    "contains": {"color": "#616b79", "animated": False},
    "protocol_dependency": {"color": "#2d6cdf", "animated": False},
    "relationship_input": {"color": "#d48a3d", "animated": False},
    "relationship_output": {"color": "#995116", "animated": False},
    "computes": {"color": "#9a4d99", "animated": False},
    "timing_sequence": {"color": "#7a8798", "animated": False},
    "generic": {"color": "#666666", "animated": False},
}

def main() -> None:
    st.set_page_config(
        page_title="PyPulseq-Star Relationship Graph",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    graph_path = _parse_graph_arg()
    spec = _load_spec_from_source(graph_path)

    st.markdown(
        """
        <div class="pps-header">
          <div class="pps-logo-mark">⌁<span>★</span></div>
          <div class="pps-title">PyPulseq-Star</div>
          <div class="pps-divider"></div>
          <div class="pps-subtitle">Relationship Graph Visualizer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        _sidebar(spec)

    if spec is None:
        _empty_state()
        return

    _dashboard(spec)


def _parse_graph_arg() -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--graph", type=str, default=None)
    args, _unknown = parser.parse_known_args(sys.argv[1:])
    return args.graph


def _load_spec_from_source(graph_path: str | None) -> dict[str, Any] | None:
    if graph_path:
        path = Path(graph_path)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                st.error(f"Could not read graph JSON: {path}\n\n{exc}")
                return None
        st.warning(f"Graph JSON path does not exist yet: {path}")

    uploaded = st.sidebar.file_uploader("Load relationship graph JSON", type=["json"])
    if uploaded is not None:
        try:
            return json.loads(uploaded.getvalue().decode("utf-8"))
        except Exception as exc:
            st.sidebar.error(f"Could not parse uploaded JSON: {exc}")
    return None


def _dashboard(spec: dict[str, Any]) -> None:
    """Render the relationship graph dashboard.

    The graph JSON can contain many repeated event occurrences.
    The viewer therefore creates a display spec before rendering. The raw graph
    remains available in the Inspector, but the visible graph can be compacted
    into sequence-level concepts: protocol/system, hierarchy blocks, timing
    relationships, and grouped repeated events.
    """

    display_spec = _prepare_display_spec(spec)

    left, right = st.columns([0.72, 0.28], gap="large")

    with left:
        st.markdown(f"## {display_spec.get('title', spec.get('title', 'Relationship Graph'))}")
        _graph_metrics(display_spec, original_spec=spec)

        if streamlit_flow is None:
            st.error(
                "streamlit-flow-component is not installed.\n\n"
                "Install with:\n\npip install streamlit streamlit-flow-component"
            )
            st.exception(_STREAMLIT_FLOW_IMPORT_ERROR)
            return

        flow_state = _get_or_create_flow_state(display_spec)
        layout = ManualLayout() if ManualLayout is not None else None

        kwargs = {
            "fit_view": True,
            "height": int(st.session_state.get("pps_graph_height", 840)),
            "show_minimap": bool(st.session_state.get("pps_show_minimap", False)),
            "hide_watermark": True,

            # Keep menus disabled for stability.
            "enable_node_menu": False,
            "enable_edge_menu": False,
            "enable_pane_menu": False,

            # Click-based inspection.
            "get_node_on_click": True,
            "get_edge_on_click": True,

            "allow_new_edges": False,
            "min_zoom": 0.20,
        }

        if layout is not None:
            kwargs["layout"] = layout

        st.session_state["pps_flow_state"] = streamlit_flow(
            "pypulseq_star_relationship_flow",
            flow_state,
            **kwargs,
        )

    with right:
        _inspector(display_spec)
        with st.expander("Raw source graph", expanded=False):
            st.json(spec)


def _graph_metrics(display_spec: dict[str, Any], *, original_spec: dict[str, Any]) -> None:
    """Show small graph-size metrics above the canvas."""

    original_nodes = len(original_spec.get("nodes", []))
    original_edges = len(original_spec.get("edges", []))
    display_nodes = len(display_spec.get("nodes", []))
    display_edges = len(display_spec.get("edges", []))

    cols = st.columns(4)
    cols[0].metric("Visible nodes", display_nodes, delta=display_nodes - original_nodes)
    cols[1].metric("Visible edges", display_edges, delta=display_edges - original_edges)
    cols[2].metric("Raw nodes", original_nodes)
    cols[3].metric("Raw edges", original_edges)


def _prepare_display_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Return a viewer-specific graph spec.

    Modes
    -----
    Compact kernel
        Collapses repeated event occurrences into one
        grouped node and hides low-value edge clutter.

    Relationships only
        Shows hierarchy plus relationship/protocol edges, but hides ordinary
        containment edges to repeated low-level events.

    Full graph
        Keeps all nodes and edges, but still applies a cleaner default layout if
        requested.
    """

    mode = st.session_state.get("pps_view_mode", "Compact kernel")
    compact_repeated = bool(st.session_state.get("pps_compact_repeated", True))
    show_contains = bool(st.session_state.get("pps_show_contains_edges", True))
    show_protocol = bool(st.session_state.get("pps_show_protocol_edges", True))
    show_relationships = bool(st.session_state.get("pps_show_relationship_edges", True))
    show_events = bool(st.session_state.get("pps_show_event_nodes", False))
    force_layout = bool(st.session_state.get("pps_force_layout", True))

    display = copy.deepcopy(spec)
    display["title"] = spec.get("title", "Relationship Graph")
    display.setdefault("metadata", {})
    display["metadata"]["viewer_mode"] = mode

    if mode == "Full graph":
        if force_layout:
            display["layout"] = {"positions": _semantic_positions(display.get("nodes", []), display.get("edges", []))}
        return display

    nodes = list(display.get("nodes", []))
    edges = list(display.get("edges", []))

    if mode == "Relationships only":
        show_events = False
        compact_repeated = True


    node_map: dict[str, str] = {}
    grouped: dict[str, dict[str, Any]] = {}
    kept_nodes: list[dict[str, Any]] = []

    repeated_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for node in nodes:
        node_id = str(node.get("id", ""))
        kind = str(node.get("kind", "generic"))

        if not show_events and kind in {"rf", "adc", "gradient", "event"}:
            # Hide low-level event nodes in relationship-only mode. Edges to
            # these nodes are removed below.
            continue

        base_name, is_repeated = _repeated_event_base(node)
        if compact_repeated and is_repeated and kind in {"gradient", "adc", "rf", "event"}:
            repeated_groups[(kind, base_name)].append(node)
            continue

        node_map[node_id] = node_id
        kept_nodes.append(node)

    for (kind, base_name), group_nodes in repeated_groups.items():
        group_id = f"group::{kind}::{base_name}"
        for node in group_nodes:
            node_map[str(node.get("id", ""))] = group_id

        grouped[group_id] = _group_node(
            group_id=group_id,
            kind=kind,
            base_name=base_name,
            group_nodes=group_nodes,
        )

    kept_nodes.extend(grouped.values())

    kept_node_ids = {str(node.get("id")) for node in kept_nodes}
    edge_seen: set[tuple[str, str, str, str]] = set()
    kept_edges: list[dict[str, Any]] = []

    for edge in edges:
        kind = str(edge.get("kind", "generic"))

        kind = {
            "contains_event": "contains",
            "materializes_as_block": "contains",
            "block_after": "timing_sequence",
            "relationship_reference": "relationship_input",
            "relationship_target": "relationship_output",
            "protocol_required": "protocol_dependency",
            "protocol_independent": "protocol_dependency",
        }.get(kind, kind)

        if kind == "contains" and not show_contains:
            continue
        if kind == "protocol_dependency" and not show_protocol:
            continue
        if kind in {
            "relationship_input",
            "relationship_output",
            "computes",
            "timing_sequence",
        } and not show_relationships:
            continue

        source = node_map.get(str(edge.get("source", "")), str(edge.get("source", "")))
        target = node_map.get(str(edge.get("target", "")), str(edge.get("target", "")))

        if source == target:
            continue
        if source not in kept_node_ids or target not in kept_node_ids:
            continue

        label = str(edge.get("label") or "")
        key = (source, target, kind, label)
        if key in edge_seen:
            continue
        edge_seen.add(key)

        new_edge = dict(edge)
        new_edge["kind"] = kind
        new_edge["source"] = source
        new_edge["target"] = target
        new_edge["id"] = f"{kind}:{source}->{target}:{len(kept_edges)}"
        kept_edges.append(new_edge)

    display["nodes"] = kept_nodes
    display["edges"] = kept_edges

    if force_layout:
        display["layout"] = {"positions": _semantic_positions(kept_nodes, kept_edges)}

    return display


def _repeated_event_base(node: dict[str, Any]) -> tuple[str, bool]:
    """Return a compact grouping key without sequence-specific name rules.

    New graph payloads expose ``compact`` and ``occurrence_count`` metadata.
    Those fields are authoritative. A generic trailing-integer fallback is kept
    only for backward compatibility with older JSON exports.
    """

    metadata = node.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    raw = str(node.get("title") or node.get("label") or node.get("id") or "node").strip()

    occurrence_count = metadata.get("occurrence_count")
    compact = bool(metadata.get("compact", False))

    try:
        repeated = int(occurrence_count) > 1
    except (TypeError, ValueError):
        repeated = False

    if compact or repeated:
        node_path = str(metadata.get("node") or node.get("id") or raw)
        return node_path, repeated

    match = re.match(r"^(?P<base>.+?)[_\-](?P<index>\d{2,})$", raw)
    if match:
        return match.group("base"), True

    return raw, False


def _group_node(
    *,
    group_id: str,
    kind: str,
    base_name: str,
    group_nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create one visible node representing many repeated event nodes."""

    count = len(group_nodes)
    examples = [str(node.get("title") or node.get("id")) for node in group_nodes[:3]]
    subtitle_parts = [f"{kind} × {count}"]
    if examples:
        subtitle_parts.append("examples: " + ", ".join(examples))

    return {
        "id": group_id,
        "kind": kind,
        "title": f"{base_name}_*",
        "subtitle": "\n".join(subtitle_parts),
        "metadata": {
            "grouped": True,
            "count": count,
            "members": [node.get("id") for node in group_nodes],
            "examples": examples,
        },
    }


def _semantic_positions(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Generate a compact gammaSTAR-like hierarchy layout."""

    positions: dict[str, dict[str, float]] = {
        "protocol_summary": {"x": 70, "y": 20},
        "system_summary": {"x": 1080, "y": 20},
    }
    node_by_id = {str(node.get("id")): node for node in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    parents: dict[str, str] = {}

    for edge in edges:
        if str(edge.get("kind")) != "contains":
            continue
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source not in node_by_id or target not in node_by_id:
            continue
        children[source].append(target)
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

    inferred = {
        "root": 0,
        "loop": 1,
        "block": 2,
        "module": 3,
        "rf": 4,
        "adc": 4,
        "gradient": 4,
        "event": 4,
    }
    levels: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for node in nodes:
        kind = str(node.get("kind", "generic"))
        if kind not in inferred:
            continue
        levels[depth.get(str(node.get("id")), inferred[kind])].append(node)

    y_by_level = {0: 40, 1: 165, 2: 290, 3: 430, 4: 590}
    for level, level_nodes in sorted(levels.items()):
        level_nodes.sort(key=lambda n: str(n.get("title") or n.get("id")))
        count = len(level_nodes)
        xs = [575] if count == 1 else _spread(90, 1060, count)
        for node, x in zip(level_nodes, xs):
            positions[str(node.get("id"))] = {
                "x": x,
                "y": y_by_level.get(level, 590 + 130 * (level - 4)),
            }

    rel_nodes = [
        node for node in nodes
        if node.get("kind") in {"relationship", "derived_parameter"}
    ]
    rel_nodes.sort(key=lambda n: str(n.get("title") or n.get("id")))
    for node, x in zip(rel_nodes, _spread(120, 1030, len(rel_nodes))):
        positions[str(node.get("id"))] = {"x": x, "y": 760}

    for node in nodes:
        positions.setdefault(str(node.get("id")), {"x": 575, "y": 900})

    return positions



def _spread(start: float, stop: float, count: int) -> list[float]:
    """Return evenly spaced positions, including both endpoints."""

    if count <= 0:
        return []
    if count == 1:
        return [(float(start) + float(stop)) / 2.0]

    step = (float(stop) - float(start)) / float(count - 1)
    return [float(start) + index * step for index in range(count)]


def _get_or_create_flow_state(spec: dict[str, Any]) -> Any:
    """Create StreamlitFlowState once per graph spec and keep it in session_state.

    streamlit-flow 1.6.x synchronizes frontend/backend state. Recreating the
    state object on every Streamlit rerun can cause continuous rerendering.
    """

    graph_hash = _graph_spec_hash(spec)

    if st.session_state.get("pps_graph_hash") != graph_hash:
        st.session_state["pps_graph_hash"] = graph_hash
        st.session_state["pps_flow_state"] = _build_flow_state(spec)

    if "pps_flow_state" not in st.session_state:
        st.session_state["pps_flow_state"] = _build_flow_state(spec)

    return st.session_state["pps_flow_state"]


def _graph_spec_hash(spec: dict[str, Any]) -> str:
    """Stable hash for deciding when to reset the flow state."""

    import hashlib

    payload = json.dumps(
        {
            "schema_version": spec.get("schema_version"),
            "title": spec.get("title"),
            "sequence": spec.get("sequence"),
            "protocol": spec.get("protocol", {}).get("summary"),
            "system": spec.get("system", {}).get("summary"),
            "nodes": spec.get("nodes", []),
            "edges": spec.get("edges", []),
            "layout": spec.get("layout", {}),
        },
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_flow_state(spec: dict[str, Any]) -> Any:
    nodes = []
    edges = []
    positions = spec.get("layout", {}).get("positions", {})

    nodes.append(
        _flow_node(
            "protocol_summary",
            positions.get("protocol_summary", {"x": 70, "y": 30}),
            _summary_card_content("📋 Protocol Summary", spec.get("protocol", {}).get("summary", [])),
            "protocol",
            source_position="bottom",
            target_position="right",
        )
    )
    nodes.append(
        _flow_node(
            "system_summary",
            positions.get("system_summary", {"x": 900, "y": 30}),
            _summary_card_content("⚙ System", spec.get("system", {}).get("summary", [])),
            "system",
            source_position="left",
            target_position="left",
        )
    )

    for node in spec.get("nodes", []):
        node_id = node["id"]
        kind = node.get("kind", "generic")
        position = positions.get(node_id, {"x": 200, "y": 200})
        content = _node_content(node)
        nodes.append(_flow_node(node_id, position, content, kind))

    node_ids = {node.id for node in nodes}
    for edge in spec.get("edges", []):
        source = edge.get("source", "")
        target = edge.get("target", "")
        if source in node_ids and target in node_ids:
            edges.append(_flow_edge(edge))

    return StreamlitFlowState(nodes, edges)


def _flow_node(
    node_id: str,
    position: dict[str, float],
    content: str,
    kind: str,
    *,
    source_position: str = "bottom",
    target_position: str = "top",
) -> Any:
    style = _node_style(kind)
    try:
        return StreamlitFlowNode(
            node_id,
            (position.get("x", 0), position.get("y", 0)),
            {"content": content},
            "input" if kind == "root" else "default",
            source_position,
            target_position,
            style=style,
        )
    except TypeError:
        return StreamlitFlowNode(
            node_id,
            (position.get("x", 0), position.get("y", 0)),
            {"content": content},
            "input" if kind == "root" else "default",
            source_position,
            target_position,
        )


def _flow_edge(edge: dict[str, Any]) -> Any:
    kind = edge.get("kind", "generic")
    style = EDGE_STYLES.get(kind, EDGE_STYLES["generic"])
    label = edge.get("label") or ""
    marker_end = {"type": "arrowclosed"}
    try:
        return StreamlitFlowEdge(
            edge.get("id", f"{edge.get('source')}-{edge.get('target')}"),
            edge.get("source"),
            edge.get("target"),
            label=label,
            animated=style["animated"],
            marker_end=marker_end,
            style={"stroke": style["color"], "strokeWidth": 2.3},
        )
    except TypeError:
        return StreamlitFlowEdge(
            edge.get("id", f"{edge.get('source')}-{edge.get('target')}"),
            edge.get("source"),
            edge.get("target"),
            animated=style["animated"],
            marker_end=marker_end,
        )


def _summary_card_content(title: str, items: list[dict[str, Any]]) -> str:
    lines = [f"**{title}**"]
    for item in items:
        lines.append(f"• {item.get('label')}: {item.get('value')}")
    return "\n".join(lines)


def _node_content(node: dict[str, Any]) -> str:
    kind = node.get("kind", "generic")
    title = str(node.get("title") or node.get("label") or node.get("id") or "node").strip()
    subtitle = str(node.get("subtitle") or "").strip()
    metadata = node.get("metadata", {}) if isinstance(node.get("metadata", {}), dict) else {}
    icon = {
        "root": "★",
        "loop": "⟳",
        "block": "◇",
        "module": "▣",
        "rf": "∿",
        "adc": "▥",
        "gradient": "↗",
        "relationship": "◷",
        "derived_parameter": "⊕",
    }.get(kind, "•")
    if metadata.get("grouped"):
        count = metadata.get("count", "?")
        return f"**{icon} {title}**\n{kind} group × {count}"
    if subtitle:
        return f"**{icon} {title}**\n{subtitle}"
    return f"**{icon} {title}**"


def _node_style(kind: str) -> dict[str, Any]:
    palette = PALETTE.get(kind, PALETTE["generic"])
    width = {
        "protocol": 300,
        "system": 280,
        "root": 130,
        "loop": 190,
        "block": 150,
        "module": 165,
        "rf": 200,
        "adc": 210,
        "gradient": 210,
        "relationship": 210,
    }.get(kind, 180)
    min_height = {
        "protocol": 170,
        "system": 150,
        "root": 64,
        "loop": 72,
        "block": 64,
        "module": 68,
        "rf": 78,
        "adc": 78,
        "gradient": 78,
        "relationship": 88,
    }.get(kind, 72)
    return {
        "background": palette["bg"],
        "border": f"2px solid {palette['border']}",
        "color": palette["fg"],
        "borderRadius": "14px",
        "padding": "12px 14px",
        "width": f"{width}px",
        "minHeight": f"{min_height}px",
        "boxShadow": "0 3px 10px rgba(15, 23, 42, 0.08)",
        "fontSize": "14px",
        "lineHeight": "1.35",
        "whiteSpace": "pre-wrap",
        "textAlign": "left",
    }


def _inspector(spec: dict[str, Any]) -> None:
    st.markdown("## Inspector")
    state = st.session_state.get("pps_flow_state")
    selected_id = getattr(state, "selected_id", None) if state is not None else None

    if selected_id:
        st.success(f"Selected: `{selected_id}`")
        selected = _find_node_or_edge(spec, selected_id)
        st.json(selected)
    else:
        st.caption("Click a node or edge in the graph to inspect details.")

    with st.expander("Protocol parameters", expanded=False):
        st.json(spec.get("protocol", {}).get("parameters", {}))
    with st.expander("System parameters", expanded=False):
        st.json(spec.get("system", {}).get("parameters", {}))
    with st.expander("Raw graph spec", expanded=False):
        st.json(spec)


def _find_node_or_edge(spec: dict[str, Any], selected_id: str) -> dict[str, Any]:
    if selected_id == "protocol_summary":
        return {"id": "protocol_summary", **spec.get("protocol", {})}
    if selected_id == "system_summary":
        return {"id": "system_summary", **spec.get("system", {})}
    for node in spec.get("nodes", []):
        if node.get("id") == selected_id:
            return node
    for edge in spec.get("edges", []):
        if edge.get("id") == selected_id:
            return edge
    return {"id": selected_id, "note": "No metadata found."}


def _sidebar(spec: dict[str, Any] | None) -> None:
    st.markdown("## How to read")
    st.write("Nodes represent sequence components. Edges show timing relationships and dependencies.")

    st.markdown("---")
    st.markdown("## View")
    st.session_state["pps_view_mode"] = st.radio(
        "Graph detail",
        ["Compact kernel", "Relationships only", "Full graph"],
        index=["Compact kernel", "Relationships only", "Full graph"].index(
            st.session_state.get("pps_view_mode", "Compact kernel")
        ),
        help="Compact mode groups repeated event occurrences using graph metadata.",
    )
    st.session_state["pps_compact_repeated"] = st.checkbox(
        "Group repeated indexed events",
        value=bool(st.session_state.get("pps_compact_repeated", True)),
    )
    st.session_state["pps_show_event_nodes"] = st.checkbox(
        "Show low-level event nodes",
        value=bool(st.session_state.get("pps_show_event_nodes", False)),
    )
    st.session_state["pps_show_contains_edges"] = st.checkbox(
        "Show contains edges",
        value=bool(st.session_state.get("pps_show_contains_edges", True)),
    )
    st.session_state["pps_show_protocol_edges"] = st.checkbox(
        "Show protocol dependencies",
        value=bool(st.session_state.get("pps_show_protocol_edges", True)),
    )
    st.session_state["pps_show_relationship_edges"] = st.checkbox(
        "Show timing relationship edges",
        value=bool(st.session_state.get("pps_show_relationship_edges", True)),
    )
    st.session_state["pps_force_layout"] = st.checkbox(
        "Use semantic lane layout",
        value=bool(st.session_state.get("pps_force_layout", True)),
    )
    st.session_state["pps_show_minimap"] = st.checkbox(
        "Show minimap",
        value=bool(st.session_state.get("pps_show_minimap", False)),
    )
    st.session_state["pps_graph_height"] = st.slider(
        "Canvas height",
        min_value=520,
        max_value=1200,
        value=int(st.session_state.get("pps_graph_height", 840)),
        step=40,
    )

    st.markdown("---")
    st.markdown("## Node types")
    for kind, label in [
        ("root", "Root / Entry"),
        ("loop", "Aggregation / Summary"),
        ("block", "Kernel"),
        ("module", "Block / Module"),
        ("rf", "RF event"),
        ("adc", "ADC / readout"),
        ("gradient", "Gradient"),
        ("relationship", "Relationship"),
        ("system", "System / External"),
    ]:
        pal = PALETTE.get(kind, PALETTE["generic"])
        st.markdown(
            f'<div class="legend-row"><span class="legend-chip" style="background:{pal["bg"]}; border-color:{pal["border"]};"></span>{label}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("## Relationship types")
    for kind, label in [
        ("contains", "Contains"),
        ("protocol_dependency", "Protocol dependency"),
        ("relationship_input", "Timing / input"),
        ("relationship_output", "Timing / output"),
        ("timing_sequence", "Block order"),
    ]:
        color = EDGE_STYLES[kind]["color"]
        line_style = "dotted" if kind == "protocol_dependency" else "dashed" if kind == "relationship_input" else "solid"
        st.markdown(
            f'<div class="legend-row"><span class="edge-chip" style="border-top:3px {line_style} {color};"></span>{label}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if spec is not None:
        st.success("Graph loaded")
        st.caption(spec.get("title", "Relationship graph"))
    else:
        st.info("Provide a graph JSON path or upload a JSON file.")


def _empty_state() -> None:
    st.markdown("## No relationship graph loaded")
    st.write("Generate a graph JSON from a demo or upload one from the sidebar.")
    st.code(
        "streamlit run src/apps/relationship_graph_viewer.py -- --graph out/fid/fid.relationship_graph.json",
        language="bash",
    )


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 100%;
        }
        .pps-header {
            display: flex;
            align-items: center;
            gap: 14px;
            border: 1px solid #dbe3ef;
            border-radius: 14px;
            padding: 14px 18px;
            margin-bottom: 14px;
            background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
        }
        .pps-logo-mark {
            font-size: 30px;
            color: #10a6ad;
            font-weight: 800;
            line-height: 1;
        }
        .pps-logo-mark span {
            color: #2d6cdf;
            font-size: 19px;
            vertical-align: top;
            margin-left: -4px;
        }
        .pps-title {
            font-size: 26px;
            font-weight: 800;
            color: #13284b;
        }
        .pps-divider {
            width: 1px;
            height: 32px;
            background: #c7cfdd;
        }
        .pps-subtitle {
            font-size: 20px;
            color: #65748f;
            font-weight: 700;
        }
        .legend-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 8px 0;
            color: #2d3f63;
            font-size: 0.94rem;
        }
        .legend-chip {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 2px solid;
            border-radius: 4px;
        }
        .edge-chip {
            display: inline-block;
            width: 36px;
            height: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
