"""Relationship graph visualization utilities for PyPulseq-Star.

This module renders a relationship-aware sequence graph from an in-memory
SeqStarSequence.

Design goals
------------
- PNG-first static rendering
- sequence-general (no FID-specific hardcoding)
- panel-first dashboard layout
- consolidated protocol and system summary cards
- readable hierarchy / event / relationship lanes
- dependency-free Mermaid/DOT text export
- optional Graphviz rendering when the system `dot` executable is available
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import math
import re
import shutil
import subprocess
import textwrap


# =============================================================================
# Data model
# =============================================================================


@dataclass
class GraphNode:
    """A node in the relationship visualization graph."""

    id: str
    label: str
    kind: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """An edge in the relationship visualization graph."""

    source: str
    target: str
    kind: str = "generic"
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipGraph:
    """Simple graph container used before rendering."""

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)

    def add_node(
        self,
        node_id: str,
        label: str,
        *,
        kind: str = "generic",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        safe_node_id = _safe_id(node_id)

        if safe_node_id not in self.nodes:
            self.nodes[safe_node_id] = GraphNode(
                id=safe_node_id,
                label=label,
                kind=kind,
                metadata=metadata or {},
            )
        else:
            old = self.nodes[safe_node_id]
            if old.kind in {"generic", "event"} and kind not in {"generic", "event"}:
                old.kind = kind

        return safe_node_id

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        kind: str = "generic",
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        source_id = _safe_id(source)
        target_id = _safe_id(target)

        for edge in self.edges:
            if (
                edge.source == source_id
                and edge.target == target_id
                and edge.kind == kind
                and edge.label == label
            ):
                return

        self.edges.append(
            GraphEdge(
                source=source_id,
                target=target_id,
                kind=kind,
                label=label,
                metadata=metadata or {},
            )
        )


# =============================================================================
# Graph extraction
# =============================================================================


class RelationshipGrapher:
    """Build and render a relationship graph for a SeqStar sequence."""

    def __init__(
        self,
        seq: Any,
        *,
        include_protocol: bool = True,
        include_system: bool = True,
        include_relationships: bool = True,
        include_derived: bool = False,
        show_all_protocol_parameters: bool = False,
    ) -> None:
        self.seq = seq
        self.include_protocol = include_protocol
        self.include_system = include_system
        self.include_relationships = include_relationships
        self.include_derived = include_derived
        self.show_all_protocol_parameters = show_all_protocol_parameters

    def build(self) -> RelationshipGraph:
        graph = RelationshipGraph()

        root_id = graph.add_node("root", "root", kind="root")

        if self.include_protocol:
            self._add_protocol_nodes(graph, root_id)

        if self.include_system:
            self._add_system_node(graph, root_id)

        self._add_timeline_nodes(graph, root_id)

        if self.include_relationships:
            self._add_relationship_nodes(graph)

        return graph

    def write(
        self,
        out_path: str | Path,
        *,
        format: str | None = None,
        keep_dot: bool = True,
    ) -> Path:
        """Write a relationship graph.

        .mmd/.dot are dependency-free text outputs.
        .png/.svg/.pdf use Graphviz only if available; otherwise DOT is written.
        """

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        requested_format = format or out_path.suffix.lstrip(".") or "mmd"
        requested_format = requested_format.lower()

        if requested_format in {"mmd", "mermaid"} or out_path.suffix == ".mmd":
            return self.write_mermaid(out_path)

        if requested_format == "dot" or out_path.suffix == ".dot":
            return self.write_dot(out_path)

        dot_path = out_path.with_suffix(".dot")
        self.write_dot(dot_path)

        dot_executable = shutil.which("dot")
        if dot_executable is None:
            print(
                "[relationship_grapher] Graphviz executable 'dot' was not found. "
                f"Wrote DOT fallback instead: {dot_path}"
            )
            return dot_path

        command = [dot_executable, f"-T{requested_format}", str(dot_path), "-o", str(out_path)]
        subprocess.run(command, check=True)

        if not keep_dot:
            try:
                dot_path.unlink()
            except OSError:
                pass

        return out_path

    def write_mermaid(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        graph = self.build()
        out_path.write_text(self.to_mermaid(graph), encoding="utf-8")
        return out_path

    def write_dot(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        graph = self.build()
        out_path.write_text(self.to_dot(graph), encoding="utf-8")
        return out_path

    def to_mermaid(self, graph: RelationshipGraph) -> str:
        lines: list[str] = ["flowchart TD", ""]

        for node in graph.nodes.values():
            shape = _mermaid_node_shape(node)
            label = _mermaid_label(node.label)

            if shape == "stadium":
                lines.append(f'    {node.id}(["{label}"])')
            elif shape == "subroutine":
                lines.append(f'    {node.id}[["{label}"]]')
            elif shape == "database":
                lines.append(f'    {node.id}[("{label}")]')
            elif shape == "circle":
                lines.append(f'    {node.id}(("{label}"))')
            else:
                lines.append(f'    {node.id}["{label}"]')

        lines.append("")

        for edge in graph.edges:
            label = f"|{_mermaid_label(edge.label)}|" if edge.label else ""

            if edge.kind == "contains":
                arrow = "-->"
            elif edge.kind in {"protocol_dependency", "relationship_input", "computes"}:
                arrow = "-.->"
            elif edge.kind == "relationship_output":
                arrow = "==>"
            else:
                arrow = "-->"

            lines.append(f"    {edge.source} {arrow}{label} {edge.target}")

        lines.append("")
        lines.extend(_mermaid_style_lines(graph))
        lines.append("")
        return "\n".join(lines)

    def to_dot(self, graph: RelationshipGraph) -> str:
        lines: list[str] = []
        lines.append("digraph SeqStarRelationshipGraph {")
        lines.append(
            '  graph [rankdir=TB, bgcolor="white", pad="0.35", '
            'nodesep="0.55", ranksep="0.65"];'
        )
        lines.append(
            '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
            'fontsize=12, margin="0.12,0.08"];'
        )
        lines.append('  edge [fontname="Helvetica", fontsize=10, color="#444444", arrowsize=0.75];')
        lines.append("")

        for node in graph.nodes.values():
            attrs = self._node_attrs(node)
            lines.append(f'  "{node.id}" [{_format_attrs(attrs)}];')

        lines.append("")

        for edge in graph.edges:
            attrs = self._edge_attrs(edge)
            lines.append(f'  "{edge.source}" -> "{edge.target}" [{_format_attrs(attrs)}];')

        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Extraction internals
    # -------------------------------------------------------------------------

    def _add_protocol_nodes(self, graph: RelationshipGraph, root_id: str) -> None:
        parameters = _get_parameters(self.seq)

        prot_id = graph.add_node("prot", "protocol", kind="protocol")
        graph.add_edge(root_id, prot_id, kind="contains")

        if self.show_all_protocol_parameters:
            parameter_keys = sorted(parameters.keys())
        else:
            parameter_keys = _important_protocol_keys(parameters)

        for key in parameter_keys:
            value = parameters.get(key)
            label = f"{key}\n{_compact_value(value)}"
            node_id = graph.add_node(
                f"prot_{key}",
                label,
                kind="protocol_parameter",
                metadata={"key": key, "value": value},
            )
            graph.add_edge(prot_id, node_id, kind="contains")

    def _add_system_node(self, graph: RelationshipGraph, root_id: str) -> None:
        system = getattr(self.seq, "system", None)
        if system is None:
            return

        sys_id = graph.add_node("sys", "system", kind="system")
        graph.add_edge(root_id, sys_id, kind="contains")

    def _add_timeline_nodes(self, graph: RelationshipGraph, root_id: str) -> None:
        """Add a compact gammaSTAR-like hierarchy.

        Repeated timeline occurrences are collapsed into one logical kernel
        motif. Blocks are grouped by ``seqstar_node`` metadata and events are
        deduplicated by stable signatures with generated indices removed.
        """

        timeline = getattr(self.seq, "timeline", None)
        blocks = list(getattr(timeline, "blocks", []) or [])
        parameters = _get_parameters(self.seq)

        repetitions = _repeat_count_from_sequence(self.seq, parameters)
        try:
            repetitions_int = int(repetitions) if repetitions is not None else 1
        except Exception:
            repetitions_int = 1

        loop_label = f"seqstar_loop\nCount: 0 | Max: {max(repetitions_int - 1, 0)}"
        loop_id = graph.add_node("seqstar_loop", loop_label, kind="loop")
        graph.add_edge(root_id, loop_id, kind="contains")

        kernel_id = graph.add_node("kernel", "kernel", kind="block")
        graph.add_edge(loop_id, kernel_id, kind="contains")

        if not blocks:
            return

        grouped: dict[str, list[Any]] = {}
        order: list[str] = []
        for index, block in enumerate(blocks):
            metadata = getattr(block, "metadata", None)
            metadata = metadata if isinstance(metadata, dict) else {}
            node = str(
                metadata.get("seqstar_node")
                or getattr(block, "node", None)
                or getattr(block, "path", None)
                or getattr(block, "name", None)
                or f"kernel.block_{index}"
            )
            if node not in grouped:
                grouped[node] = []
                order.append(node)
            grouped[node].append(block)

        for node in order:
            occurrences = grouped[node]
            representative = occurrences[0]
            local_name = node.rsplit(".", 1)[-1]
            metadata = getattr(representative, "metadata", None)
            metadata = metadata if isinstance(metadata, dict) else {}
            role = metadata.get("seqstar_role") or getattr(representative, "role", None)

            block_id = graph.add_node(
                node,
                local_name,
                kind="module",
                metadata={
                    "node": node,
                    "role": role,
                    "occurrence_count": len(occurrences),
                    "compact": True,
                },
            )
            graph.add_edge(kernel_id, block_id, kind="contains")

            seen_events: set[str] = set()
            for occurrence in occurrences:
                for event_index, event in enumerate(_block_events(occurrence)):
                    event_name = _object_name(event, fallback=f"event_{event_index}")
                    signature = _compact_visual_event_signature(event, event_name)
                    if signature in seen_events:
                        continue
                    seen_events.add(signature)

                    event_kind = _event_kind(event)
                    event_id = graph.add_node(
                        f"{node}.{signature}",
                        _compact_visual_event_label(event_name),
                        kind=event_kind,
                        metadata={
                            "event": event,
                            "node": node,
                            "compact": True,
                        },
                    )
                    graph.add_edge(block_id, event_id, kind="contains")

    def _add_relationship_nodes(self, graph: RelationshipGraph) -> None:
        relationships = _collect_relationships(self.seq)

        for index, rel in enumerate(relationships):
            kind = _relationship_kind(rel)

            if kind in {"contains", "contains_block", "contains_event", "contains_child"}:
                continue

            if _is_redundant_structural_repeat(rel, relationships):
                continue

            if kind == "relationship":
                continue

            name = _relationship_name(rel, fallback=f"relationship_{index}")
            rel_label = _compact_relationship_label(name, kind)
            rel_id = graph.add_node(f"rel_{name}", rel_label, kind="relationship", metadata={"relationship": rel})

            if kind in {"set_center_after", "center_after"}:
                self._add_set_center_after_edges(graph, rel, rel_id)
            elif kind in {"repeat_every", "repeats_every", "set_repetition_time"}:
                self._add_repeat_every_edges(graph, rel, rel_id)
            else:
                self._add_generic_relationship_edges(graph, rel, rel_id)

    def _add_set_center_after_edges(self, graph: RelationshipGraph, rel: Any, rel_id: str) -> None:
        reference = _relationship_field(rel, "reference")
        target = _relationship_field(rel, "target")
        offset = _relationship_field(rel, "offset")

        if reference is None:
            reference = _relationship_field(rel, "source")

        reference_id = _resolve_object_node_id(reference)
        target_id = _resolve_object_node_id(target)

        if reference_id is not None:
            if reference_id not in graph.nodes:
                graph.add_node(reference_id, _prettify_node_id(reference_id), kind="event")
            graph.add_edge(reference_id, rel_id, kind="relationship_input", label="reference")

        if offset is not None:
            offset_node_id = self._protocol_node_for_reference(graph, offset)
            graph.add_edge(offset_node_id, rel_id, kind="protocol_dependency", label="offset")

        if target_id is not None:
            if target_id not in graph.nodes:
                graph.add_node(target_id, _prettify_node_id(target_id), kind="event")
            graph.add_edge(rel_id, target_id, kind="relationship_output", label="sets delay")

        if self.include_derived and target_id is not None:
            derived_id = graph.add_node(
                f"derived_{target_id}_delay",
                f"{_prettify_node_id(target_id)}.delay",
                kind="derived_parameter",
            )
            graph.add_edge(rel_id, derived_id, kind="computes", label="computed")

    def _add_repeat_every_edges(self, graph: RelationshipGraph, rel: Any, rel_id: str) -> None:
        period = _relationship_field(rel, "period")

        if period is None:
            metadata = _relationship_metadata(rel)
            period = metadata.get("period") or metadata.get("TR") or metadata.get("interval_s")

        events = _relationship_field(rel, "events") or []
        if isinstance(events, (str, bytes)):
            events = [events]

        if period is not None:
            period_node_id = self._protocol_node_for_reference(graph, period)
            graph.add_edge(period_node_id, rel_id, kind="protocol_dependency", label="period")

        kernel_id = _find_first_node_id_by_kind(graph, "block")
        if kernel_id is not None:
            graph.add_edge(kernel_id, rel_id, kind="relationship_input", label="kernel")
        else:
            for event in events:
                event_id = _resolve_object_node_id(event)
                if event_id is None:
                    continue
                if event_id not in graph.nodes:
                    graph.add_node(event_id, _prettify_node_id(event_id), kind="event")
                graph.add_edge(event_id, rel_id, kind="relationship_input", label="member")

        graph.add_edge(rel_id, "average", kind="relationship_output", label="schedule")

        if self.include_derived:
            derived_id = graph.add_node("derived_tr_fill", "TR fill", kind="derived_parameter")
            graph.add_edge(rel_id, derived_id, kind="computes", label="TR - kernel")

    def _add_generic_relationship_edges(self, graph: RelationshipGraph, rel: Any, rel_id: str) -> None:
        source = _relationship_field(rel, "source")
        target = _relationship_field(rel, "target")

        source_id = _resolve_object_node_id(source)
        target_id = _resolve_object_node_id(target)

        if source_id is not None:
            if source_id not in graph.nodes:
                graph.add_node(source_id, _prettify_node_id(source_id), kind="generic")
            graph.add_edge(source_id, rel_id, kind="relationship_input")

        if target_id is not None:
            if target_id not in graph.nodes:
                graph.add_node(target_id, _prettify_node_id(target_id), kind="generic")
            graph.add_edge(rel_id, target_id, kind="relationship_output")

    def _protocol_node_for_reference(self, graph: RelationshipGraph, reference: Any) -> str:
        parameters = _get_parameters(self.seq)
        reference_text = str(reference)

        if reference_text.startswith("root.prot."):
            reference_text = reference_text.removeprefix("root.prot.")

        actual_key = _resolve_protocol_alias(reference_text, parameters)
        value = parameters.get(actual_key) if actual_key is not None else None

        if actual_key is None:
            label = f"{reference_text}"
            node_id = f"prot_{reference_text}"
        elif actual_key == reference_text:
            label = f"{actual_key}\n{_compact_value(value)}"
            node_id = f"prot_{actual_key}"
        else:
            label = f"{reference_text}\n{actual_key} = {_compact_value(value)}"
            node_id = f"prot_{reference_text}"

        return graph.add_node(
            node_id,
            label,
            kind="protocol_parameter",
            metadata={"reference": reference_text, "actual_key": actual_key, "value": value},
        )

    def _node_attrs(self, node: GraphNode) -> dict[str, str]:
        style = _node_style(node.kind)
        return {
            "label": node.label,
            "fillcolor": style["facecolor"],
            "fontcolor": style["fontcolor"],
            "color": style["edgecolor"],
            "penwidth": "1.4",
        }

    def _edge_attrs(self, edge: GraphEdge) -> dict[str, str]:
        style = _edge_style(edge.kind)
        attrs: dict[str, str] = {}
        if edge.label:
            attrs["label"] = edge.label
        attrs.update(
            {
                "color": style["color"],
                "style": style["dot_style"],
                "penwidth": str(style["linewidth"]),
            }
        )
        return attrs


# =============================================================================
# Public helpers
# =============================================================================


def write_relationship_graph(
    seq: Any,
    out_path: str | Path,
    *,
    format: str | None = None,
    include_protocol: bool = True,
    include_system: bool = True,
    include_relationships: bool = True,
    include_derived: bool = False,
    show_all_protocol_parameters: bool = False,
    keep_dot: bool = True,
) -> Path:
    grapher = RelationshipGrapher(
        seq,
        include_protocol=include_protocol,
        include_system=include_system,
        include_relationships=include_relationships,
        include_derived=include_derived,
        show_all_protocol_parameters=show_all_protocol_parameters,
    )
    return grapher.write(out_path, format=format, keep_dot=keep_dot)


def write_relationship_graph_mermaid(
    seq: Any,
    out_path: str | Path,
    *,
    include_protocol: bool = True,
    include_system: bool = True,
    include_relationships: bool = True,
    include_derived: bool = False,
    show_all_protocol_parameters: bool = False,
) -> Path:
    grapher = RelationshipGrapher(
        seq,
        include_protocol=include_protocol,
        include_system=include_system,
        include_relationships=include_relationships,
        include_derived=include_derived,
        show_all_protocol_parameters=show_all_protocol_parameters,
    )
    return grapher.write_mermaid(out_path)


def write_relationship_graph_dot(
    seq: Any,
    out_path: str | Path,
    *,
    include_protocol: bool = True,
    include_system: bool = True,
    include_relationships: bool = True,
    include_derived: bool = False,
    show_all_protocol_parameters: bool = False,
) -> Path:
    grapher = RelationshipGrapher(
        seq,
        include_protocol=include_protocol,
        include_system=include_system,
        include_relationships=include_relationships,
        include_derived=include_derived,
        show_all_protocol_parameters=show_all_protocol_parameters,
    )
    return grapher.write_dot(out_path)


def plot_relationship_graph(
    seq: Any,
    *,
    title: str | None = None,
    show: bool = True,
    save: str | Path | None = None,
    include_protocol: bool = True,
    include_system: bool = True,
    include_relationships: bool = True,
    include_derived: bool = False,
    show_all_protocol_parameters: bool = False,
    layout: str = "dashboard",
    enable_hover: bool = False,
    figsize: tuple[float, float] = (16.0, 9.2),
    dpi: int = 160,
):
    """Plot a relationship graph.

    Parameters
    ----------
    layout
        "dashboard" is the default and recommended static PNG layout.

        "dot" uses the optional PyPI stack:
            pip install networkx netgraph grandalf

    enable_hover
        If True and mplcursors is installed, hover tooltips will be enabled.
        The static PNG renderer does not depend on it.
    """

    grapher = RelationshipGrapher(
        seq,
        include_protocol=include_protocol,
        include_system=include_system,
        include_relationships=include_relationships,
        include_derived=include_derived,
        show_all_protocol_parameters=show_all_protocol_parameters,
    )
    graph = grapher.build()

    if layout == "dot":
        return _plot_relationship_graph_netgraph(
            graph,
            title=title,
            show=show,
            save=save,
            figsize=figsize,
            dpi=dpi,
        )

    if layout not in {"dashboard", "workflow"}:
        raise ValueError(f"Unsupported relationship graph layout {layout!r}. Use 'dashboard', 'workflow', or 'dot'.")

    return _plot_relationship_graph_dashboard(
        graph,
        seq=seq,
        title=title,
        show=show,
        save=save,
        enable_hover=enable_hover,
        figsize=figsize,
        dpi=dpi,
    )


def relationship_graph_text(seq: Any) -> str:
    lines: list[str] = ["Sequence relationship graph", "===========================", "", "Hierarchy", "---------", "root"]
    timeline = getattr(seq, "timeline", None)
    blocks = list(getattr(timeline, "blocks", []) or [])

    if blocks:
        lines.append("  average")
        for block_index, block in enumerate(blocks):
            block_name = _object_name(block, fallback=f"block_{block_index}")
            lines.append(f"    {block_name}")
            for event_index, event in enumerate(_block_events(block)):
                event_name = _object_name(event, fallback=f"event_{block_index}_{event_index}")
                lines.append(f"      {event_name}")
    else:
        lines.append("  <no timeline blocks>")

    lines.append("")
    lines.append("Relationships")
    lines.append("-------------")
    relationships = _collect_relationships(seq)

    if not relationships:
        lines.append("<no relationships>")
    else:
        for index, rel in enumerate(relationships):
            kind = _relationship_kind(rel)
            name = _relationship_name(rel, fallback=f"relationship_{index}")
            lines.append(f"- {name}: {kind}")

    return "\n".join(lines)


# =============================================================================
# Matplotlib dashboard renderer
# =============================================================================


def _plot_relationship_graph_dashboard(
    graph: RelationshipGraph,
    *,
    seq: Any,
    title: str | None,
    show: bool,
    save: str | Path | None,
    enable_hover: bool,
    figsize: tuple[float, float],
    dpi: int,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    artists: list[Any] = []
    tooltip_text: dict[int, str] = {}

    def remember(artist: Any, text: str) -> None:
        artists.append(artist)
        tooltip_text[id(artist)] = text

    # ------------------------------------------------------------------
    # Layout constants
    # ------------------------------------------------------------------
    outer = (0.8, 0.8, 98.4, 98.0)
    header = (2.0, 92.0, 96.0, 6.0)
    sidebar = (2.0, 4.0, 14.0, 86.5)
    main = (17.0, 4.0, 80.8, 86.5)

    # Main internal areas
    top_row = (18.0, 75.0, 78.8, 14.0)
    top_left = (21.0, 77.0, 23.0, 10.0)
    top_right = (76.0, 77.0, 16.5, 10.0)

    lane_label_w = 10.0
    lane_x = 18.0
    lane_w = 78.8

    lane_protocol = (18.0, 69.0, 78.8, 6.5)
    lane_structure = (18.0, 44.0, 78.8, 24.0)
    lane_events = (18.0, 27.0, 78.8, 15.5)
    lane_relationships = (18.0, 9.0, 78.8, 16.0)

    content_left = 18.0 + lane_label_w + 2.0
    content_right = 95.0
    content_w = content_right - content_left

    _rounded_panel(ax, *outer, radius=1.1, face="#ffffff", edge="#d5ddeb", lw=1.0, z=0)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    _draw_logo(ax, 3.0, 95.2)
    ax.text(5.4, 95.2, "PyPulseq-Star", fontsize=19, fontweight="bold", va="center", color="#13284b")
    ax.text(17.0, 95.2, "|", fontsize=24, va="center", color="#c7cfdd")
    header_title = title or f"{_sequence_title(seq)} Relationship Graph"
    ax.text(18.5, 95.2, header_title, fontsize=19, fontweight="bold", va="center", color="#65748f")

    _draw_header_button(ax, 79.5, 93.5, 5.5, 4.3, "Fit")
    _draw_header_button(ax, 86.2, 93.5, 7.2, 4.3, "Legend")
    _draw_header_button(ax, 94.2, 93.5, 5.0, 4.3, "Export")

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    _rounded_panel(ax, *sidebar, radius=0.9, face="#fbfdff", edge="#dbe3ef", lw=1.0, z=0)

    ax.text(4.0, 87.0, "How to read", fontsize=11.5, fontweight="bold", color="#13284b", va="top")
    ax.text(
        4.0,
        82.8,
        "Nodes represent sequence\ncomponents. Edges show timing\nrelationships and dependencies.",
        fontsize=8.9,
        color="#2d3f63",
        va="top",
        linespacing=1.55,
    )
    ax.plot([4.0, 14.0], [74.4, 74.4], color="#dfe6f0", lw=1.0)

    ax.text(4.0, 71.8, "Node types", fontsize=11.0, fontweight="bold", color="#13284b", va="top")
    _sidebar_node_type(ax, 4.4, 67.9, "root", "Root / Entry")
    _sidebar_node_type(ax, 4.4, 63.8, "loop", "Aggregation / Summary")
    _sidebar_node_type(ax, 4.4, 59.7, "block", "Kernel / Module")
    _sidebar_node_type(ax, 4.4, 55.6, "rf", "Event")
    _sidebar_node_type(ax, 4.4, 51.5, "system", "System / External")

    ax.plot([4.0, 14.0], [45.3, 45.3], color="#dfe6f0", lw=1.0)

    ax.text(4.0, 42.9, "Relationship types", fontsize=11.0, fontweight="bold", color="#13284b", va="top")
    _sidebar_edge_type(ax, 4.4, 37.8, "contains", "Contains")
    _sidebar_edge_type(ax, 4.4, 33.7, "protocol_dependency", "Protocol dependency")
    _sidebar_edge_type(ax, 4.4, 29.6, "relationship_input", "Timing / Input")
    _sidebar_edge_type(ax, 4.4, 25.5, "relationship_output", "Timing / Output")

    _rounded_panel(ax, 2.0, 4.0, 14.0, 7.8, radius=0.9, face="#fbfdff", edge="#dbe3ef", lw=1.0, z=0)
    ax.text(4.6, 10.0, "Tip", fontsize=10.0, fontweight="bold", color="#5f6f88", va="center")
    ax.text(
        4.6,
        6.3,
        "This visualizer is optimized\nfor static PNG export.\nHover is optional.",
        fontsize=8.2,
        color="#5f6f88",
        va="center",
        linespacing=1.45,
    )

    # ------------------------------------------------------------------
    # Main panel & top cards
    # ------------------------------------------------------------------
    _rounded_panel(ax, *main, radius=1.0, face="#ffffff", edge="#dbe3ef", lw=1.0, z=0)

    _draw_wave_icon(ax, 21.5, 87.3, color="#8a96aa")
    ax.text(24.0, 87.3, _sequence_title(seq), fontsize=16.0, fontweight="bold", color="#13284b", va="center")

    _rounded_panel(ax, *top_row, radius=0.8, face="#fbfdff", edge="#edf1f7", lw=0.9, z=0)

    protocol_lines = _protocol_summary_lines(seq, max_items=7)
    system_lines = _system_summary_lines(seq, max_items=5)

    prot_card = _draw_summary_card(
        ax,
        *top_left,
        title="Protocol Summary",
        lines=protocol_lines,
        accent="#16a7a7",
        icon="protocol",
    )
    remember(prot_card, "Protocol summary\n" + "\n".join(protocol_lines))

    sys_card = _draw_summary_card(
        ax,
        *top_right,
        title="System",
        lines=system_lines,
        accent="#7d8799",
        icon="system",
    )
    remember(sys_card, "System summary\n" + "\n".join(system_lines))

    # ------------------------------------------------------------------
    # Lanes
    # ------------------------------------------------------------------
    lane_specs = [
        (lane_protocol, "Protocol"),
        (lane_structure, "Sequence\nStructure"),
        (lane_events, "Events"),
        (lane_relationships, "Relationships /\nDependencies"),
    ]
    for box, label in lane_specs:
        _rounded_panel(ax, *box, radius=0.7, face="#fbfdff", edge="#edf1f7", lw=0.9, z=0)
        ax.text(
            box[0] + 1.8,
            box[1] + box[3] / 2,
            label,
            fontsize=10.5,
            fontweight="bold",
            color="#18345f",
            va="center",
            ha="left",
        )

    # ------------------------------------------------------------------
    # Determine node positions
    # ------------------------------------------------------------------
    positions = _dashboard_positions_general(
        graph,
        content_left=content_left,
        content_right=content_right,
        y_structure_top=62.0,
        y_structure_mid=55.2,
        y_events=34.4,
        y_relationships=17.0,
    )

    # Protocol/system card anchor points
    protocol_anchor = (top_left[0] + top_left[2], top_left[1] + top_left[3] / 2)
    system_anchor = (top_right[0], top_right[1] + top_right[3] / 2)

    # ------------------------------------------------------------------
    # Draw arrows (structure first)
    # ------------------------------------------------------------------
    for edge in graph.edges:
        if edge.kind != "contains":
            continue
        if edge.source not in positions or edge.target not in positions:
            continue
        _draw_arrow(ax, positions[edge.source], positions[edge.target], kind="contains")

    # Non-contains edges, except protocol dependency (special routed from card)
    for edge in graph.edges:
        if edge.kind in {"contains", "protocol_dependency"}:
            continue
        if edge.source not in positions or edge.target not in positions:
            continue
        _draw_arrow(
            ax,
            positions[edge.source],
            positions[edge.target],
            kind=edge.kind,
            label=edge.label,
        )

    # Protocol dependency edges from consolidated card
    for edge in graph.edges:
        if edge.kind != "protocol_dependency":
            continue
        if edge.target not in positions:
            continue
        _draw_arrow(
            ax,
            protocol_anchor,
            positions[edge.target],
            kind="protocol_dependency",
            label=edge.label,
            force_rad=0.15,
        )

    # Optional system influence visual
    kernel_nodes = [n.id for n in graph.nodes.values() if n.kind == "block"]
    if kernel_nodes:
        _draw_arrow(
            ax,
            system_anchor,
            positions[kernel_nodes[0]],
            kind="relationship_input",
            label=None,
            force_rad=-0.16,
            alpha_scale=0.65,
        )

    # ------------------------------------------------------------------
    # Draw nodes
    # ------------------------------------------------------------------
    hidden_ids = {"prot", "sys"} | {n.id for n in graph.nodes.values() if n.kind == "protocol_parameter"}

    draw_order = sorted(
        [n for n in graph.nodes.values() if n.id not in hidden_ids],
        key=lambda node: _draw_priority(node.kind),
    )

    for node in draw_order:
        if node.id not in positions:
            continue
        patch = _draw_node(ax, positions[node.id][0], positions[node.id][1], node)
        remember(patch, _tooltip_for_node(node))

    # Footer
    ax.text(
        57.2,
        5.0,
        "Static PNG-first relationship visualization for sequence inspection and documentation.",
        fontsize=9.0,
        color="#6d7b91",
        ha="center",
        va="center",
    )

    if enable_hover:
        _try_enable_mplcursors(artists, tooltip_text)

    fig.tight_layout(pad=0.2)

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def _dashboard_positions_general(
    graph: RelationshipGraph,
    *,
    content_left: float,
    content_right: float,
    y_structure_top: float,
    y_structure_mid: float,
    y_events: float,
    y_relationships: float,
) -> dict[str, tuple[float, float]]:
    """Sequence-general lane-based placement."""

    positions: dict[str, tuple[float, float]] = {}
    x_center = 0.5 * (content_left + content_right)

    root_nodes = [n for n in graph.nodes.values() if n.kind == "root"]
    loop_nodes = [n for n in graph.nodes.values() if n.kind == "loop"]
    block_nodes = [n for n in graph.nodes.values() if n.kind in {"block", "module"}]
    event_nodes = [n for n in graph.nodes.values() if n.kind in {"rf", "adc", "gradient", "event"}]
    rel_nodes = [n for n in graph.nodes.values() if n.kind == "relationship"]
    derived_nodes = [n for n in graph.nodes.values() if n.kind == "derived_parameter"]

    if root_nodes:
        positions[root_nodes[0].id] = (x_center - 9.0, y_structure_top)

    if loop_nodes:
        loop_xs = _spread(x_center - 3.0, x_center + 3.0, len(loop_nodes))
        for node, x in zip(loop_nodes, loop_xs):
            positions[node.id] = (x, y_structure_top)

    if block_nodes:
        block_xs = _spread(x_center + 8.0, x_center + 8.0 if len(block_nodes) == 1 else x_center + 18.0, len(block_nodes))
        for node, x in zip(block_nodes, block_xs):
            positions[node.id] = (x, y_structure_top)

    if event_nodes:
        sorted_events = sorted(event_nodes, key=_event_sort_key)
        xs = _spread(content_left + 6.0, content_right - 6.0, len(sorted_events))
        for node, x in zip(sorted_events, xs):
            positions[node.id] = (x, y_events)

    if rel_nodes:
        sorted_rels = sorted(rel_nodes, key=_relationship_sort_key)
        xs = _spread(content_left + 7.0, content_right - 7.0, len(sorted_rels))
        for node, x in zip(sorted_rels, xs):
            positions[node.id] = (x, y_relationships)

    if derived_nodes:
        xs = _spread(content_left + 8.0, content_right - 8.0, len(derived_nodes))
        for node, x in zip(derived_nodes, xs):
            positions[node.id] = (x, y_relationships - 6.0)

    for node in graph.nodes.values():
        if node.kind == "protocol_parameter":
            positions[node.id] = (content_left, y_structure_mid)

    if "prot" in graph.nodes:
        positions["prot"] = (content_left, y_structure_mid)
    if "sys" in graph.nodes:
        positions["sys"] = (content_right, y_structure_mid)

    for node in graph.nodes.values():
        if node.id not in positions:
            positions[node.id] = (content_left + 2.0, y_structure_mid)

    return positions


def _draw_priority(kind: str) -> int:
    priority = {
        "root": 1,
        "loop": 2,
        "block": 3,
        "module": 3,
        "rf": 4,
        "adc": 4,
        "gradient": 4,
        "event": 4,
        "relationship": 5,
        "derived_parameter": 6,
        "generic": 9,
    }
    return priority.get(kind, 9)


# =============================================================================
# Drawing helpers
# =============================================================================


def _rounded_panel(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    radius: float,
    face: str,
    edge: str,
    lw: float,
    z: int = 0,
) -> Any:
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def _draw_logo(ax: Any, x: float, y: float) -> None:
    """Compact PyPulseq-Star header mark.

    This remains dependency-light and does not require SVG rendering.
    """
    teal = "#10a6ad"
    navy = "#13284b"
    blue = "#2d6cdf"

    ax.plot(
        [x, x + 0.35, x + 0.60, x + 0.88, x + 1.15, x + 1.42, x + 1.75],
        [y, y + 0.72, y - 0.75, y + 1.02, y - 0.62, y + 0.40, y],
        color=teal,
        lw=2.5,
        solid_capstyle="round",
        zorder=5,
    )
    ax.scatter([x + 2.35], [y + 1.10], marker="*", s=120, color=blue, edgecolor=navy, linewidth=0.8, zorder=6)


def _draw_header_button(ax: Any, x: float, y: float, w: float, h: float, text: str) -> None:
    _rounded_panel(ax, x, y, w, h, radius=0.55, face="#ffffff", edge="#dbe3ef", lw=1.0, z=2)
    ax.text(x + w / 2, y + h / 2, text, fontsize=9.8, color="#263957", ha="center", va="center")


def _draw_wave_icon(ax: Any, x: float, y: float, *, color: str) -> None:
    ax.plot(
        [x - 1.0, x - 0.5, x, x + 0.5, x + 1.0],
        [y, y + 0.8, y - 1.0, y + 0.8, y],
        color=color,
        lw=1.8,
        zorder=3,
    )


def _draw_summary_card(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    lines: list[str],
    accent: str,
    icon: str,
) -> Any:
    face = "#f8ffff" if icon == "protocol" else "#fbfcff"
    patch = _rounded_panel(ax, x, y, w, h, radius=0.8, face=face, edge=accent, lw=1.2, z=2)

    icon_text = "📋" if icon == "protocol" else "⚙"
    ax.text(x + 1.3, y + h - 1.7, icon_text, fontsize=13, color=accent, va="center", ha="left", zorder=3)
    ax.text(
        x + 3.3,
        y + h - 1.7,
        title,
        fontsize=11.0,
        fontweight="bold",
        color=accent if icon == "protocol" else "#3e4b5e",
        va="center",
        ha="left",
        zorder=3,
    )

    text_y = y + h - 3.3
    for i, line in enumerate(lines[:7]):
        ax.text(
            x + 1.6,
            text_y - i * 1.4,
            line,
            fontsize=8.7,
            color="#1f2f4a",
            va="top",
            ha="left",
            zorder=3,
        )

    return patch


def _sidebar_node_type(ax: Any, x: float, y: float, kind: str, label: str) -> None:
    style = _node_style(kind)
    _rounded_panel(ax, x, y - 1.1, 1.4, 2.2, radius=0.25, face=style["facecolor"], edge=style["edgecolor"], lw=1.0, z=2)
    ax.text(x + 2.1, y, label, fontsize=8.9, color="#2a3b5f", va="center", ha="left")


def _sidebar_edge_type(ax: Any, x: float, y: float, kind: str, label: str) -> None:
    style = _edge_style(kind)
    ax.plot([x, x + 3.0], [y, y], color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"])
    ax.text(x + 4.0, y, label, fontsize=8.8, color="#2a3b5f", va="center", ha="left")


def _draw_node(ax: Any, x: float, y: float, node: GraphNode) -> Any:
    from matplotlib.patches import FancyBboxPatch

    style = _node_style(node.kind)
    width, height = _dashboard_node_box_size(node)
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.05,rounding_size=0.7",
        facecolor=style["facecolor"],
        edgecolor=style["edgecolor"],
        linewidth=style["linewidth"],
        alpha=style["alpha"],
        zorder=4,
    )
    ax.add_patch(patch)

    icon = _icon_for_kind(node.kind)
    lines = node.label.splitlines()

    if icon:
        ax.text(
            x - width / 2 + 1.2,
            y + 0.2,
            icon,
            fontsize=13,
            color=style["fontcolor"],
            ha="left",
            va="center",
            zorder=5,
        )
        tx = x + 0.4
    else:
        tx = x

    ax.text(
        tx,
        y,
        "\n".join(lines),
        fontsize=style["fontsize"],
        color=style["fontcolor"],
        ha="center",
        va="center",
        linespacing=1.05,
        zorder=5,
    )

    return patch


def _dashboard_node_box_size(node: GraphNode) -> tuple[float, float]:
    lines = node.label.splitlines()
    longest = max((len(line) for line in lines), default=8)

    if node.kind == "root":
        return (5.5, 4.3)

    if node.kind == "loop":
        return (8.8, 5.0)

    if node.kind in {"block", "module"}:
        return (7.8, 4.2)

    if node.kind in {"rf", "adc", "gradient", "event"}:
        width = max(8.0, min(12.5, 3.8 + 0.22 * longest))
        return (width, 4.8)

    if node.kind == "relationship":
        width = max(10.5, min(14.5, 5.4 + 0.22 * longest))
        height = max(5.2, 3.3 + 1.25 * len(lines))
        return (width, height)

    if node.kind == "derived_parameter":
        return (8.5, 4.2)

    return (7.5, 4.2)


def _draw_arrow(
    ax: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    kind: str,
    label: str | None = None,
    force_rad: float | None = None,
    alpha_scale: float = 1.0,
) -> Any:
    from matplotlib.patches import FancyArrowPatch

    style = _edge_style(kind)
    rad = force_rad if force_rad is not None else _dashboard_edge_rad(start, end, kind)

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11.0,
        linewidth=style["linewidth"],
        linestyle=style["linestyle"],
        alpha=style["alpha"] * alpha_scale,
        color=style["color"],
        shrinkA=18,
        shrinkB=18,
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(arrow)

    if label and kind != "contains":
        xm = 0.5 * (start[0] + end[0])
        ym = 0.5 * (start[1] + end[1])
        ax.text(
            xm,
            ym + 1.0,
            str(label),
            fontsize=7.8,
            color=style["color"],
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.95},
            zorder=3,
        )

    return arrow


def _dashboard_edge_rad(start: tuple[float, float], end: tuple[float, float], kind: str) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    if kind == "contains":
        return 0.0

    if kind == "protocol_dependency":
        return 0.15 if dx >= 0 else -0.15

    if kind == "relationship_input":
        return 0.10 if dy <= 0 else -0.10

    if kind == "relationship_output":
        return -0.08 if dx >= 0 else 0.08

    return 0.06


def _try_enable_mplcursors(artists: list[Any], tooltip_text: dict[int, str]) -> None:
    try:
        import mplcursors
    except ImportError:
        return

    cursor = mplcursors.cursor(artists, hover=True)

    @cursor.connect("add")
    def _on_add(sel: Any) -> None:
        text = tooltip_text.get(id(sel.artist), "")
        sel.annotation.set_text(text)
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.95)


def _plot_relationship_graph_netgraph(
    graph: RelationshipGraph,
    *,
    title: str | None,
    show: bool,
    save: str | Path | None,
    figsize: tuple[float, float],
    dpi: int,
):
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        from netgraph import Graph
    except ImportError as exc:
        raise ImportError(
            "layout='dot' requires optional visualization packages:\n"
            "    pip install networkx netgraph grandalf\n"
        ) from exc

    nx_graph = nx.DiGraph()
    for node in graph.nodes.values():
        nx_graph.add_node(node.id)
    for edge in graph.edges:
        nx_graph.add_edge(edge.source, edge.target)

    node_labels = {node.id: node.label for node in graph.nodes.values()}
    edge_list = list(nx_graph.edges())
    edge_labels = {(edge.source, edge.target): edge.label for edge in graph.edges if edge.label and (edge.source, edge.target) in edge_list}
    node_color = {node.id: _node_style(node.kind)["facecolor"] for node in graph.nodes.values()}
    node_edge_color = {node.id: _node_style(node.kind)["edgecolor"] for node in graph.nodes.values()}
    node_size = {node.id: _netgraph_node_size(node) for node in graph.nodes.values()}
    edge_color = {(edge.source, edge.target): _edge_style(edge.kind)["color"] for edge in graph.edges if (edge.source, edge.target) in edge_list}
    edge_width = {(edge.source, edge.target): _edge_style(edge.kind)["linewidth"] for edge in graph.edges if (edge.source, edge.target) in edge_list}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_title(title or "SeqStar Relationship Graph", fontsize=17, pad=16)
    ax.axis("off")

    Graph(
        nx_graph,
        node_layout="dot",
        edge_layout="curved",
        arrows=True,
        node_labels=node_labels,
        edge_labels=edge_labels,
        node_color=node_color,
        node_edge_color=node_edge_color,
        node_size=node_size,
        edge_color=edge_color,
        edge_width=edge_width,
        node_label_fontdict={"size": 8, "color": "black"},
        edge_label_fontdict={"size": 7, "color": "#444444"},
        edge_label_rotate=False,
        ax=ax,
    )

    fig.tight_layout()
    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


# =============================================================================
# Data extraction and normalization helpers
# =============================================================================


def _collect_relationships(seq: Any) -> list[Any]:
    relationships: list[Any] = []
    candidate_attrs = [
        "relationship_definitions",
        "relationships",
        "_relationships",
        "_seqstar_relationships",
        "_pypulseq_star_relationships",
    ]

    for attr in candidate_attrs:
        value = getattr(seq, attr, None)
        if value is None:
            continue
        if isinstance(value, dict):
            relationships.extend(value.values())
        elif isinstance(value, (list, tuple)):
            relationships.extend(value)
        else:
            try:
                relationships.extend(list(value))
            except TypeError:
                relationships.append(value)

    unique: list[Any] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for index, rel in enumerate(relationships):
        kind = _relationship_kind(rel)
        name = _relationship_name(rel, fallback=f"relationship_{index}")
        source = str(_relationship_field(rel, "source"))
        target = str(_relationship_field(rel, "target"))
        key = (kind, name, source + "->" + target)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(rel)
    return unique


def _relationship_kind(rel: Any) -> str:
    return str(
        _relationship_field(rel, "kind")
        or _relationship_field(rel, "type")
        or _relationship_field(rel, "relationship_type")
        or "relationship"
    )


def _relationship_name(rel: Any, *, fallback: str) -> str:
    return str(_relationship_field(rel, "name") or _relationship_field(rel, "id") or fallback)


def _relationship_metadata(rel: Any) -> dict[str, Any]:
    if isinstance(rel, dict):
        metadata = rel.get("metadata", {})
        return metadata if isinstance(metadata, dict) else {}
    metadata = getattr(rel, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _relationship_field(rel: Any, field_name: str) -> Any:
    if isinstance(rel, dict):
        if field_name in rel:
            return rel[field_name]
        metadata = rel.get("metadata", {})
        if isinstance(metadata, dict) and field_name in metadata:
            return metadata[field_name]
        return None
    if hasattr(rel, field_name):
        return getattr(rel, field_name)
    metadata = getattr(rel, "metadata", None)
    if isinstance(metadata, dict) and field_name in metadata:
        return metadata[field_name]
    return None


def _is_redundant_structural_repeat(rel: Any, all_relationships: list[Any]) -> bool:
    kind = _relationship_kind(rel)
    if kind != "repeats_every":
        return False
    name = _relationship_name(rel, fallback="")
    metadata = _relationship_metadata(rel)
    if "period" in metadata or "TR" in metadata:
        return False
    if "fid_kernel_repeats" in name:
        return False
    explicit_repeat_exists = any(
        _relationship_kind(candidate) in {"repeat_every", "set_repetition_time"}
        for candidate in all_relationships
    )
    return explicit_repeat_exists


def _get_parameters(seq: Any) -> dict[str, Any]:
    parameters = getattr(seq, "parameters", None)
    if isinstance(parameters, dict):
        return parameters
    if hasattr(seq, "protocol_dict"):
        try:
            protocol_dict = seq.protocol_dict()
            if isinstance(protocol_dict, dict):
                return protocol_dict
        except Exception:
            pass
    return {}


def _important_protocol_keys(parameters: dict[str, Any]) -> list[str]:
    preferred = [
        "TE",
        "echo_time",
        "TR",
        "repetition_time",
        "average",
        "averages",
        "repetitions",
        "flip_angle",
        "flip_angle_excitation",
        "rf_duration",
        "num_samples",
        "dwell",
    ]
    keys: list[str] = []
    for key in preferred:
        if key in parameters and key not in keys:
            keys.append(key)
    return keys


def _protocol_summary_lines(seq: Any, *, max_items: int) -> list[str]:
    parameters = _get_parameters(seq)
    keys = _important_protocol_keys(parameters)
    if not keys:
        keys = list(parameters.keys())[:max_items]
    shown = keys[:max_items]
    lines = [f"• {_display_parameter_name(key)}: {_compact_value(parameters.get(key))}" for key in shown]
    remaining = max(0, len(parameters) - len(shown))
    if remaining > 0:
        lines.append(f"• … {remaining} more")
    return lines


def _system_summary_lines(seq: Any, *, max_items: int) -> list[str]:
    system = getattr(seq, "system", None)
    if system is None:
        return ["• no system"]

    if hasattr(system, "to_dict"):
        try:
            values = system.to_dict()
        except Exception:
            values = {}
    else:
        try:
            values = dict(vars(system))
        except Exception:
            values = {}

    preferred = [
        "max_grad",
        "max_slew",
        "rf_ringdown_time",
        "rf_dead_time",
        "adc_dead_time",
    ]
    lines: list[str] = []
    for key in preferred:
        if key in values:
            lines.append(f"• {_display_parameter_name(key)}: {_compact_value(values[key])}")
        if len(lines) >= max_items:
            break

    if not lines:
        lines = ["• dict", "• hardware"]

    return lines


def _display_parameter_name(key: str) -> str:
    aliases = {
        "echo_time": "Echo time (TE)",
        "repetition_time": "Repetition time (TR)",
        "flip_angle_excitation": "RF flip angle",
        "num_samples": "Num samples",
        "rf_duration": "RF duration",
        "dwell": "Dwell time",
        "max_grad": "Max gradient",
        "max_slew": "Max slew",
        "rf_ringdown_time": "RF ringdown",
        "rf_dead_time": "RF dead time",
        "adc_dead_time": "ADC dead time",
    }
    return aliases.get(key, key.replace("_", " "))


def _resolve_protocol_alias(reference: str, parameters: dict[str, Any]) -> str | None:
    aliases: dict[str, tuple[str, ...]] = {
        "TE": ("TE", "te", "echo_time", "EchoTime", "echoTime", "echo_time_s"),
        "TR": ("TR", "tr", "repetition_time", "RepetitionTime", "repetitionTime", "repetition_time_s"),
        "FA": ("FA", "fa", "flip_angle", "flip_angle_excitation", "excitation_flip_angle"),
    }
    candidate_keys = aliases.get(reference, (reference,))

    for canonical_aliases in aliases.values():
        if any(reference.lower() == alias.lower() for alias in canonical_aliases):
            candidate_keys = canonical_aliases
            break

    for candidate in candidate_keys:
        if candidate in parameters:
            return candidate
    for candidate in candidate_keys:
        for existing_key in parameters:
            if str(existing_key).lower() == str(candidate).lower():
                return existing_key
    return None


def _first_present(parameters: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in parameters:
            return parameters[key]
    for key in keys:
        for existing_key, value in parameters.items():
            if str(existing_key).lower() == key.lower():
                return value
    return None


def _block_events(block: Any) -> list[Any]:
    events = getattr(block, "events", None)
    if isinstance(events, list):
        return events
    if isinstance(events, tuple):
        return list(events)
    children = getattr(block, "children", None)
    if isinstance(children, list):
        return children
    if isinstance(children, tuple):
        return list(children)
    return []


def _repeat_count_from_sequence(seq: Any, parameters: dict[str, Any]) -> Any:
    """Resolve repeat count from node metadata before generic parameter fallbacks."""

    timeline = getattr(seq, "timeline", None)
    registries = (
        getattr(timeline, "nodes", None),
        getattr(timeline, "node_registry", None),
        getattr(seq, "metadata", {}).get("seqstar_nodes")
        if isinstance(getattr(seq, "metadata", None), dict)
        else None,
    )

    for registry in registries:
        if not isinstance(registry, dict):
            continue
        for node in registry.values():
            if isinstance(node, dict):
                repeat_count = node.get("repeat_count")
            else:
                repeat_count = getattr(node, "repeat_count", None)

            if repeat_count is None:
                continue

            if isinstance(repeat_count, str) and repeat_count in parameters:
                return parameters[repeat_count]

            return repeat_count

    return _first_present(
        parameters,
        "repetitions",
        "average",
        "averages",
        "n_avg",
        "num_averages",
    )


def _compact_visual_event_signature(event: Any, name: str) -> str:
    text = str(name)
    text = re.sub(r"_b\d+", "", text)
    text = re.sub(r"_e\d+", "", text)
    role = str(getattr(event, "role", "") or "")
    kind = str(getattr(event, "kind", "") or type(event).__name__)
    token = "__".join(part for part in (text, role, kind) if part)
    return re.sub(r"[^A-Za-z0-9_]+", "_", token).strip("_").lower() or "event"


def _compact_visual_event_label(name: str) -> str:
    text = re.sub(r"_b\d+", "", str(name))
    return re.sub(r"_e\d+", "", text)


def _object_name(obj: Any, *, fallback: str) -> str:
    if isinstance(obj, str):
        return obj
    name = getattr(obj, "name", None)
    if name:
        return str(name)
    role = getattr(obj, "role", None)
    if role:
        return str(role)
    return fallback


def _resolve_object_node_id(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, str):
        return _safe_id(obj)
    name = getattr(obj, "name", None)
    if name:
        return _safe_id(str(name))
    role = getattr(obj, "role", None)
    if role:
        return _safe_id(str(role))
    return _safe_id(type(obj).__name__)


def _event_kind(event: Any) -> str:
    role = str(getattr(event, "role", "") or "").lower()
    kind = str(getattr(event, "kind", "") or "").lower()
    event_type = type(event).__name__.lower()
    text = " ".join([role, kind, event_type])

    if "rf" in text or "pulse" in text:
        return "rf"
    if "adc" in text or "readout" in text or "acquisition" in text or "fid" in text:
        return "adc"
    if "grad" in text or "trap" in text:
        return "gradient"
    return "event"


def _event_label(name: str, kind: str) -> str:
    if kind == "rf":
        return f"{name}\nRF pulse"
    if kind == "adc":
        return f"{name}\nacquire signal"
    if kind == "gradient":
        return f"{name}\ngradient"
    return _prettify_node_id(name)


def _sequence_title(seq: Any) -> str:
    name = getattr(seq, "name", None)
    name_text = str(name) if name else "Sequence"
    seq_type = _get_parameters(seq).get("sequence_type")
    if seq_type:
        return f"{seq_type}: {name_text}"
    return name_text


def _event_sort_key(node: GraphNode) -> tuple[int, str]:
    text = (node.id + " " + node.label + " " + node.kind).lower()
    if "rf" in text:
        return (0, node.id)
    if "grad" in text:
        return (1, node.id)
    if "adc" in text or "fid" in text or "readout" in text:
        return (2, node.id)
    return (3, node.id)


def _relationship_sort_key(node: GraphNode) -> tuple[int, str]:
    text = (node.id + " " + node.label).lower()
    if "center" in text or "echo" in text or "te" in text:
        return (0, node.id)
    if "repeat" in text or "tr" in text:
        return (1, node.id)
    return (2, node.id)


def _find_first_node_id_by_kind(graph: RelationshipGraph, kind: str) -> str | None:
    for node in graph.nodes.values():
        if node.kind == kind:
            return node.id
    return None


def _compact_relationship_label(name: str, kind: str) -> str:
    clean = name
    clean = clean.replace("_after_", "\nafter\n")
    clean = clean.replace("_repeats_at_", "\nrepeats at\n")
    clean = clean.replace("_repeats_every_", "\nrepeats every\n")
    clean = clean.replace("_", " ")
    if len(clean) > 34 and "\n" not in clean:
        clean = _wrap_text(clean, width=20)

    if kind in {"set_center_after", "center_after"}:
        kind_text = "relationship"
    elif kind in {"repeat_every", "repeats_every", "set_repetition_time"}:
        kind_text = "relationship"
    else:
        kind_text = kind.replace("_", " ")

    return f"{clean}\n[{kind_text}]"


def _wrap_text(text: str, *, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def _spread(start: float, stop: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [(start + stop) / 2]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def _safe_id(value: Any) -> str:
    text = str(value).strip()
    if not text:
        text = "node"
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "node"
    if text[0].isdigit():
        text = f"n_{text}"
    return text


def _prettify_node_id(node_id: str) -> str:
    return str(node_id).replace("_", " ")


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


def _tooltip_for_node(node: GraphNode) -> str:
    lines = [node.label, f"kind: {node.kind}"]
    if node.metadata:
        for key, value in node.metadata.items():
            if key in {"event", "relationship"}:
                lines.append(f"{key}: {type(value).__name__}")
            else:
                lines.append(f"{key}: {value}")
    return "\n".join(lines)


# =============================================================================
# Styling and export helpers
# =============================================================================


def _node_style(kind: str) -> dict[str, Any]:
    styles: dict[str, dict[str, Any]] = {
        "root": {
            "facecolor": "#1fc9c7",
            "fontcolor": "white",
            "edgecolor": "#079b9d",
            "linewidth": 1.7,
            "fontsize": 11.0,
            "alpha": 1.0,
        },
        "loop": {
            "facecolor": "#eef4ff",
            "fontcolor": "#174ea6",
            "edgecolor": "#2d6cdf",
            "linewidth": 1.5,
            "fontsize": 9.8,
            "alpha": 1.0,
        },
        "block": {
            "facecolor": "#fff2df",
            "fontcolor": "#b15f00",
            "edgecolor": "#f28c1a",
            "linewidth": 1.5,
            "fontsize": 10.2,
            "alpha": 1.0,
        },
        "rf": {
            "facecolor": "#eff9ed",
            "fontcolor": "#236a27",
            "edgecolor": "#55a84f",
            "linewidth": 1.4,
            "fontsize": 9.6,
            "alpha": 1.0,
        },
        "adc": {
            "facecolor": "#f3efff",
            "fontcolor": "#5e3bc4",
            "edgecolor": "#8c6be8",
            "linewidth": 1.4,
            "fontsize": 9.6,
            "alpha": 1.0,
        },
        "gradient": {
            "facecolor": "#edf7ff",
            "fontcolor": "#0c5c8a",
            "edgecolor": "#49a0d8",
            "linewidth": 1.4,
            "fontsize": 9.6,
            "alpha": 1.0,
        },
        "event": {
            "facecolor": "#eff9ed",
            "fontcolor": "#236a27",
            "edgecolor": "#55a84f",
            "linewidth": 1.4,
            "fontsize": 9.6,
            "alpha": 1.0,
        },
        "relationship": {
            "facecolor": "#fff7de",
            "fontcolor": "#5b3b00",
            "edgecolor": "#d09113",
            "linewidth": 1.4,
            "fontsize": 9.3,
            "alpha": 1.0,
        },
        "protocol": {
            "facecolor": "#e8f1ff",
            "fontcolor": "#174ea6",
            "edgecolor": "#2d6cdf",
            "linewidth": 1.3,
            "fontsize": 9.2,
            "alpha": 1.0,
        },
        "protocol_parameter": {
            "facecolor": "#e8f1ff",
            "fontcolor": "#174ea6",
            "edgecolor": "#2d6cdf",
            "linewidth": 1.1,
            "fontsize": 8.2,
            "alpha": 1.0,
        },
        "system": {
            "facecolor": "#f3f4f6",
            "fontcolor": "#344054",
            "edgecolor": "#9aa3b2",
            "linewidth": 1.3,
            "fontsize": 9.2,
            "alpha": 1.0,
        },
        "derived_parameter": {
            "facecolor": "#f5d0ff",
            "fontcolor": "#000000",
            "edgecolor": "#aa72bc",
            "linewidth": 1.2,
            "fontsize": 9.0,
            "alpha": 1.0,
        },
        "generic": {
            "facecolor": "#eeeeee",
            "fontcolor": "#222222",
            "edgecolor": "#999999",
            "linewidth": 1.0,
            "fontsize": 9.0,
            "alpha": 1.0,
        },
    }
    return styles.get(kind, styles["generic"])


def _edge_style(kind: str) -> dict[str, Any]:
    if kind == "contains":
        return {"color": "#666d7a", "linestyle": "solid", "dot_style": "solid", "linewidth": 1.6, "alpha": 0.82}
    if kind == "protocol_dependency":
        return {"color": "#2d6cdf", "linestyle": "dotted", "dot_style": "dotted", "linewidth": 1.9, "alpha": 0.95}
    if kind == "relationship_input":
        return {"color": "#cb7f39", "linestyle": "dashed", "dot_style": "dashed", "linewidth": 1.75, "alpha": 0.92}
    if kind == "relationship_output":
        return {"color": "#8b4716", "linestyle": "solid", "dot_style": "bold", "linewidth": 2.1, "alpha": 0.96}
    if kind == "computes":
        return {"color": "#9a4d99", "linestyle": "dashed", "dot_style": "dashed", "linewidth": 1.55, "alpha": 0.88}
    return {"color": "#666666", "linestyle": "solid", "dot_style": "solid", "linewidth": 1.2, "alpha": 0.8}


def _icon_for_kind(kind: str) -> str:
    icons = {
        "rf": "∿",
        "adc": "▥",
        "gradient": "↗",
        "relationship": "◷",
        "block": "◇",
        "loop": "",
        "root": "",
        "derived_parameter": "⊕",
    }
    return icons.get(kind, "")


def _netgraph_node_size(node: GraphNode) -> float:
    if node.kind in {"root", "loop"}:
        return 6.0
    if node.kind == "relationship":
        return 7.0
    if node.kind in {"protocol_parameter", "derived_parameter"}:
        return 4.8
    return 5.4


def _format_attrs(attrs: dict[str, str]) -> str:
    return ", ".join(f'{key}="{_escape_dot(value)}"' for key, value in attrs.items())


def _escape_dot(value: Any) -> str:
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    return text


def _mermaid_label(value: Any) -> str:
    text = str(value) if value is not None else ""
    text = text.replace("\\", "\\\\")
    text = text.replace('"', "'")
    text = text.replace("\n", "<br/>")
    return text


def _mermaid_node_shape(node: GraphNode) -> str:
    if node.kind == "root":
        return "stadium"
    if node.kind == "relationship":
        return "subroutine"
    if node.kind in {"protocol", "system"}:
        return "database"
    if node.kind == "derived_parameter":
        return "circle"
    return "box"


def _mermaid_style_lines(graph: RelationshipGraph) -> list[str]:
    lines: list[str] = []
    class_defs = {
        "root": "fill:#1fc9c7,color:#ffffff,stroke:#079b9d",
        "loop": "fill:#eef4ff,color:#174ea6,stroke:#2d6cdf",
        "block": "fill:#fff2df,color:#b15f00,stroke:#f28c1a",
        "rf": "fill:#eff9ed,color:#236a27,stroke:#55a84f",
        "adc": "fill:#f3efff,color:#5e3bc4,stroke:#8c6be8",
        "gradient": "fill:#edf7ff,color:#0c5c8a,stroke:#49a0d8",
        "event": "fill:#eff9ed,color:#236a27,stroke:#55a84f",
        "protocol": "fill:#e8f1ff,color:#174ea6,stroke:#2d6cdf",
        "protocol_parameter": "fill:#e8f1ff,color:#174ea6,stroke:#2d6cdf",
        "system": "fill:#f3f4f6,color:#344054,stroke:#9aa3b2",
        "relationship": "fill:#fff7de,color:#5b3b00,stroke:#d09113",
        "derived_parameter": "fill:#f5d0ff,color:#000000,stroke:#aa72bc",
        "generic": "fill:#eeeeee,color:#000000,stroke:#999999",
    }
    for class_name, style in class_defs.items():
        lines.append(f"    classDef {class_name} {style};")
    for node in graph.nodes.values():
        class_name = node.kind if node.kind in class_defs else "generic"
        lines.append(f"    class {node.id} {class_name};")
    return lines