"""Timeline utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pypulseq_star.blocks import SeqStarBlock


def _node_debug_enabled(value: bool | None = None) -> bool:
    """Return whether timeline/node debug output should be printed."""

    if value is not None:
        return bool(value)

    # Keep imports local so this small utility has no import-time side effects.
    import os

    env_value = os.environ.get("PYPULSEQ_STAR_SEQUENCE_DEBUG", "")
    return str(env_value).strip().lower() in {"1", "true", "yes", "on", "debug"}


def _safe_setattr(obj: Any, name: str, value: Any) -> None:
    try:
        setattr(obj, name, value)
    except Exception:
        pass


def _ensure_dict_attribute(obj: Any, attr: str) -> dict[str, Any] | None:
    value = getattr(obj, attr, None)
    if isinstance(value, dict):
        return value

    try:
        setattr(obj, attr, {})
        value = getattr(obj, attr, None)
    except Exception:
        value = None

    if isinstance(value, dict):
        return value
    return None


def split_node_path(node: str | None) -> tuple[str | None, str | None, str | None]:
    """Return ``(full_node, parent_node, local_name)`` for a dotted node path."""

    if node is None:
        return None, None, None

    full_node = str(node).strip()
    if not full_node:
        return None, None, None

    parts = [part for part in full_node.split(".") if part]
    if not parts:
        return None, None, None

    local_name = parts[-1]
    parent_node = ".".join(parts[:-1]) if len(parts) > 1 else "sequence"
    return full_node, parent_node, local_name


@dataclass(slots=True)
class SeqStarTimeline:
    """Ordered block timeline plus lightweight SeqStar hierarchy metadata.

    The timeline remains the executable list of blocks.  The added node registry
    is deliberately small: it records named hierarchy nodes such as ``kernel``
    or ``kernel.echo_train`` and annotates blocks with their node path, role,
    variation tags, inferred parent, and automatic block index.
    """

    blocks: list[SeqStarBlock] = field(default_factory=list)
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    block_count_by_node: dict[str, int] = field(default_factory=dict)
    block_count_by_parent: dict[str, int] = field(default_factory=dict)
    last_block_by_parent: dict[str, SeqStarBlock] = field(default_factory=dict)
    debug: bool = False

    def set_debug(self, enabled: bool = True) -> None:
        """Enable or disable timeline debug printing."""

        self.debug = bool(enabled)

    def _debug(self, message: str) -> None:
        if bool(self.debug) or _node_debug_enabled(None):
            print(f"[seqstar.timeline] {message}")

    def set_node(
        self,
        name: str,
        *,
        parent: str | None = None,
        role: str | None = None,
        repeat_every: str | float | None = None,
        repeat_count: str | int | None = None,
        counter: str | None = None,
        repeat_mode: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register or update a hierarchy node.

        ``parent`` is normally inferred from dotted names.  For example,
        ``kernel.readout`` implies parent ``kernel`` and local name ``readout``.
        """

        full_node, inferred_parent, local_name = split_node_path(name)
        if full_node is None or local_name is None:
            raise ValueError("Node name must be a non-empty string.")

        parent_node = str(parent).strip() if parent is not None else inferred_parent
        if not parent_node:
            parent_node = "sequence"

        # Ensure parent nodes exist for nested paths.  The top-level sequence
        # pseudo-node is intentionally not inserted as a normal child node.
        if parent_node != "sequence" and parent_node not in self.nodes:
            self.set_node(parent_node)

        if repeat_mode is not None:
            repeat_mode = str(repeat_mode).strip().lower()
            if repeat_mode not in {"loop", "expanded"}:
                raise ValueError(
                    "repeat_mode must be 'loop' or 'expanded'. "
                    f"Passed: {repeat_mode!r}."
                )
        if repeat_count is not None and repeat_mode is None:
            existing_mode = self.nodes.get(full_node, {}).get("repeat_mode")
            if existing_mode is None:
                raise ValueError(
                    "repeat_count requires repeat_mode='loop' or "
                    "repeat_mode='expanded'."
                )

        record = dict(self.nodes.get(full_node, {}))
        record.update(
            {
                "name": full_node,
                "node": full_node,
                "parent": parent_node,
                "local_name": local_name,
                "role": role or record.get("role") or local_name,
                "repeat_every": repeat_every if repeat_every is not None else record.get("repeat_every"),
                "repeat_count": repeat_count if repeat_count is not None else record.get("repeat_count"),
                "counter": counter if counter is not None else record.get("counter"),
                "repeat_mode": repeat_mode if repeat_mode is not None else record.get("repeat_mode"),
            }
        )
        if metadata:
            existing_metadata = dict(record.get("metadata", {}) or {})
            existing_metadata.update(metadata)
            record["metadata"] = existing_metadata
        else:
            record.setdefault("metadata", {})

        self.nodes[full_node] = record
        self._debug(
            f"set_node name={full_node!r} parent={parent_node!r} role={record.get('role')!r} "
            f"repeat_every={record.get('repeat_every')!r} repeat_count={record.get('repeat_count')!r} "
            f"counter={record.get('counter')!r} repeat_mode={record.get('repeat_mode')!r}"
        )
        return record

    def get_node(self, name: str) -> dict[str, Any] | None:
        """Return a registered node record, if present."""

        return self.nodes.get(str(name))

    def ensure_node(self, name: str | None, *, role: str | None = None) -> dict[str, Any] | None:
        """Ensure that a node path exists, returning its record."""

        if name is None:
            return None
        full_node, _, local_name = split_node_path(name)
        if full_node is None:
            return None
        if full_node not in self.nodes:
            return self.set_node(full_node, role=role or local_name)
        if role is not None and not self.nodes[full_node].get("role"):
            self.nodes[full_node]["role"] = role
        return self.nodes[full_node]

    def nearest_repeat_node(self, node: str | None) -> dict[str, Any] | None:
        """Return the nearest ancestor node carrying repeat metadata."""

        full_node, parent_node, _ = split_node_path(node)
        candidate = full_node
        while candidate:
            record = self.nodes.get(candidate)
            if record and (
                record.get("repeat_every") is not None
                or record.get("repeat_count") is not None
                or record.get("counter") is not None
            ):
                return record
            if candidate == "sequence":
                break
            _, candidate, _ = split_node_path(candidate)
            if candidate == "sequence":
                record = self.nodes.get(candidate)
                if record and (
                    record.get("repeat_every") is not None
                    or record.get("repeat_count") is not None
                    or record.get("counter") is not None
                ):
                    return record
                break
        if parent_node and parent_node != full_node:
            return self.nearest_repeat_node(parent_node)
        return None

    def append(
        self,
        block: SeqStarBlock,
        *,
        node: str | None = None,
        role: str | None = None,
        varies: list[str] | tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Append a block to the timeline and annotate hierarchy metadata."""

        block_index = len(self.blocks)
        node_record = self.ensure_node(node, role=role)
        full_node, parent_node, local_name = split_node_path(node)

        if full_node is None:
            local_name = role or getattr(block, "name", None) or f"block_{block_index}"
            parent_node = "sequence"
            full_node = str(local_name)

        parent_node = parent_node or "sequence"
        node_role = role or (node_record or {}).get("role") or local_name
        varies_list = list(varies or [])

        node_block_index = self.block_count_by_node.get(full_node, 0)
        parent_block_index = self.block_count_by_parent.get(parent_node, 0)
        previous_block = self.last_block_by_parent.get(parent_node)
        repeat_record = self.nearest_repeat_node(full_node)

        block_metadata: dict[str, Any] = {
            "seqstar_node": full_node,
            "path": full_node,
            "seqstar_path": full_node,
            "seqstar_parent": parent_node,
            "seqstar_local_name": local_name,
            "seqstar_role": node_role,
            "seqstar_varies": varies_list,
            "seqstar_block_index": block_index,
            "seqstar_global_block_index": block_index,
            "seqstar_node_block_index": node_block_index,
            "seqstar_parent_block_index": parent_block_index,
            "seqstar_relationship_source": "seq.add_block",
            "seqstar_implicit_timeline_relationship": previous_block is not None,
        }

        if previous_block is not None:
            block_metadata.update(
                {
                    "seqstar_previous_block_name": getattr(previous_block, "name", None),
                    "seqstar_previous_block_node": _block_metadata_value(previous_block, "seqstar_node"),
                    "seqstar_previous_block_object_id": id(previous_block),
                }
            )

        if repeat_record is not None:
            block_metadata.update(
                {
                    "seqstar_repeat_parent": repeat_record.get("name"),
                    "seqstar_counter": repeat_record.get("counter"),
                    "seqstar_repeat_every": repeat_record.get("repeat_every"),
                    "seqstar_repeat_count": repeat_record.get("repeat_count"),
                    "seqstar_repeat_mode": repeat_record.get("repeat_mode"),
                }
            )

        if metadata:
            block_metadata.update(metadata)

        _attach_block_metadata(block, block_metadata)

        self.blocks.append(block)
        self.block_count_by_node[full_node] = node_block_index + 1
        self.block_count_by_parent[parent_node] = parent_block_index + 1
        self.last_block_by_parent[parent_node] = block

        self._debug(
            f"append block[{block_index}] name={getattr(block, 'name', None)!r} "
            f"node={full_node!r} parent={parent_node!r} role={node_role!r} "
            f"varies={varies_list!r} repeat_parent={block_metadata.get('seqstar_repeat_parent')!r}"
        )
        return block_index



    def to_debug_dict(self) -> dict[str, Any]:
        """Return a repr-safe summary of nodes and blocks.

        This is intended for smoke tests and relationship/writer debugging.  It
        avoids printing the raw dataclass block/event repr, which can be fragile
        while event classes are still evolving.
        """

        block_summaries: list[dict[str, Any]] = []
        for index, block in enumerate(self.blocks):
            events: list[dict[str, Any]] = []
            for event_index, event in enumerate(_iter_block_events_for_debug(block)):
                events.append(
                    {
                        "index": event_index,
                        "name": getattr(event, "name", None),
                        "type": event.__class__.__name__,
                        "path": _block_metadata_value(event, "path"),
                        "role": _block_metadata_value(event, "seqstar_role"),
                        "source_event_object_id": _block_metadata_value(event, "source_event_object_id"),
                        "sequence_occurrence_index": _block_metadata_value(event, "sequence_occurrence_index"),
                    }
                )

            block_summaries.append(
                {
                    "index": index,
                    "name": getattr(block, "name", None),
                    "path": _block_metadata_value(block, "path"),
                    "node": _block_metadata_value(block, "seqstar_node"),
                    "parent": _block_metadata_value(block, "seqstar_parent"),
                    "local_name": _block_metadata_value(block, "seqstar_local_name"),
                    "role": _block_metadata_value(block, "seqstar_role"),
                    "varies": _block_metadata_value(block, "seqstar_varies") or [],
                    "repeat_parent": _block_metadata_value(block, "seqstar_repeat_parent"),
                    "counter": _block_metadata_value(block, "seqstar_counter"),
                    "previous_block_node": _block_metadata_value(block, "seqstar_previous_block_node"),
                    "events": events,
                }
            )

        return {
            "nodes": dict(self.nodes),
            "num_blocks": len(self.blocks),
            "blocks": block_summaries,
        }



def _iter_block_events_for_debug(block: Any) -> list[Any]:
    """Return block events without relying on block repr."""

    events = getattr(block, "events", None)
    if events is None:
        return []
    if isinstance(events, dict):
        return list(events.values())
    if isinstance(events, (list, tuple)):
        return list(events)
    try:
        return list(events)
    except Exception:
        return []


def _block_metadata_value(block: Any, key: str) -> Any:
    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    parameters = getattr(block, "parameters", None)
    if isinstance(parameters, dict) and key in parameters:
        return parameters[key]
    return getattr(block, key, None)


def _attach_block_metadata(block: Any, values: dict[str, Any]) -> None:
    metadata = _ensure_dict_attribute(block, "metadata")
    if metadata is not None:
        metadata.update(values)

    parameters = _ensure_dict_attribute(block, "parameters")
    if parameters is not None:
        parameters.update(values)

    # Common direct attributes for writer/viewer convenience.  These are best
    # effort because several SeqStar objects may use slots.
    direct_names = {
        "path": values.get("seqstar_node"),
        "node": values.get("seqstar_node"),
        "parent_node": values.get("seqstar_parent"),
        "local_name": values.get("seqstar_local_name"),
        "role": values.get("seqstar_role"),
        "varies": values.get("seqstar_varies"),
        "counter": values.get("seqstar_counter"),
        "repeat_parent": values.get("seqstar_repeat_parent"),
        "block_index": values.get("seqstar_block_index"),
    }
    for name, value in direct_names.items():
        if value is not None:
            _safe_setattr(block, name, value)
