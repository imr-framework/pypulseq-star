"""Top-level SeqStar sequence."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import os
from typing import Any

from pypulseq_star.blocks import SeqStarBlock
from pypulseq_star.core import SeqStarNode, SeqStarRelationship
from pypulseq_star.opts import Opts
from pypulseq_star.check_timing import check_timing
from pypulseq_star.calc_duration import calc_duration as _shared_calc_duration
from pypulseq_star.plotting.plotter import plot
from .timeline import SeqStarTimeline, split_node_path


def _safe_setattr(obj: Any, name: str, value: Any) -> None:
    """Assign an attribute when it is present in the slotted object model."""

    try:
        setattr(obj, name, value)
    except AttributeError:
        pass


def _metadata_value(obj: Any, key: str, default: Any = None) -> Any:
    """Return a value from metadata/parameters/attributes in that order."""

    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]

    parameters = getattr(obj, "parameters", None)
    if isinstance(parameters, dict) and key in parameters:
        return parameters[key]

    return getattr(obj, key, default)


@dataclass(slots=True)
class SeqStarSequence(SeqStarNode):
    """A sequence that owns the hierarchy and the executable timeline.

    The sequence is the top-level object that binds together:

    - system limits and timing options from ``ppstar.Opts``
    - protocol-level parameters from ``ppstar.Protocol``
    - executable blocks in the timeline
    - hierarchy and relationships for template/JSON export

    Example
    -------
    system = ppstar.Opts(...)

    protocol = ppstar.Protocol(
        name="Block RF train",
        description="A simple repeated RF block-pulse train",
        parameters={
            "average": 20,
            "TR": 2.0,
        },
    )

    seq = ppstar.Sequence(
        name="Block RF train",
        system=system,
        parameters=protocol.parameters,
    )
    """

    system: Opts = field(default_factory=Opts)
    timeline: SeqStarTimeline = field(default_factory=SeqStarTimeline)
    debug: bool = False

    def __init__(
        self,
        system: Opts | None = None,
        name: str | None = None,
        parameters: dict[str, object] | None = None,
        *,
        debug: bool | None = None,
    ) -> None:
        """Create an enriched PyPulseq-Star sequence."""

        system = system or Opts()
        sequence_name = name or "seqstar_sequence"
        sequence_parameters = dict(parameters or {})
        debug_enabled = self._debug_env_enabled() if debug is None else bool(debug)

        self.system = system
        self.timeline = SeqStarTimeline(debug=debug_enabled)
        self.debug = debug_enabled

        # These fields are inherited from SeqStarNode in the current object model.
        # Assign only fields that exist so this class remains compatible if the
        # node model is tightened later.
        _safe_setattr(self, "name", sequence_name)
        _safe_setattr(self, "role", "sequence")
        _safe_setattr(self, "parameters", sequence_parameters)
        _safe_setattr(self, "relationships", [])
        _safe_setattr(self, "children", [])
        _safe_setattr(self, "metadata", {})
        _safe_setattr(self, "parent", None)
        _safe_setattr(self, "enabled", True)
        _safe_setattr(self, "tstart", None)
        _safe_setattr(self, "duration", None)
        _safe_setattr(self, "path", sequence_name)

        # Convenience raster/system mirrors used by PyPulseq-style examples.
        _safe_setattr(self, "grad_raster_time", float(getattr(system, "grad_raster_time", 10e-6)))
        _safe_setattr(self, "rf_raster_time", float(getattr(system, "rf_raster_time", 1e-6)))
        _safe_setattr(
            self,
            "block_duration_raster",
            float(getattr(system, "block_duration_raster", getattr(self, "grad_raster_time", 10e-6))),
        )
        _safe_setattr(self, "adc_raster_time", float(getattr(system, "adc_raster_time", 1e-7)))

        # Sequence/timeline bookkeeping. These are mirrored in metadata so the
        # implementation remains compatible with the current slotted base node.
        if hasattr(self, "metadata") and isinstance(self.metadata, dict):
            self.metadata.setdefault("event_object_occurrences", {})
            self.metadata.setdefault("event_object_first_names", {})
            self.metadata.setdefault("seqstar_nodes", self.timeline.nodes)
            self.metadata.setdefault("seqstar_debug", debug_enabled)
            self.metadata.setdefault("seqstar_timeline_version", "node-metadata-phase1-block-after")

        self._debug(
            f"created sequence name={sequence_name!r} debug={debug_enabled} "
            f"parameters={len(sequence_parameters)}"
        )

    @staticmethod
    def _debug_env_enabled() -> bool:
        value = os.environ.get("PYPULSEQ_STAR_SEQUENCE_DEBUG", "")
        return str(value).strip().lower() in {"1", "true", "yes", "on", "debug"}

    def set_debug(self, enabled: bool = True) -> None:
        """Enable or disable sequence/timeline debug printing."""

        self.debug = bool(enabled)
        if hasattr(self, "timeline") and hasattr(self.timeline, "set_debug"):
            self.timeline.set_debug(bool(enabled))
        if hasattr(self, "metadata") and isinstance(self.metadata, dict):
            self.metadata["seqstar_debug"] = bool(enabled)

    def _debug(self, message: str) -> None:
        if bool(getattr(self, "debug", False)) or self._debug_env_enabled():
            print(f"[seqstar.sequence] {message}")

    def set_node(
        self,
        name: str,
        *,
        role: str | None = None,
        repeat_every: str | float | None = None,
        repeat_count: str | int | None = None,
        counter: str | None = None,
        repeat_mode: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register or update a SeqStar hierarchy node.

        This is the first implementation step toward using ``seq.add_block`` as
        the source of truth for ordered timelines and node hierarchy.  Dotted
        node paths infer their parent automatically:

            seq.set_node("kernel", repeat_every="TR", repeat_count="n_y")
            seq.set_node("kernel.echo_train", repeat_every="echo_spacing")

        The public API intentionally avoids a separate ``parent`` argument for
        normal use; parentage is derived from the dotted node path.
        """

        record = self.timeline.set_node(
            name,
            role=role,
            repeat_every=repeat_every,
            repeat_count=repeat_count,
            counter=counter,
            repeat_mode=repeat_mode,
            metadata=metadata,
        )

        if hasattr(self, "metadata") and isinstance(self.metadata, dict):
            self.metadata["seqstar_nodes"] = self.timeline.nodes

        self._debug(
            f"set_node name={record.get('name')!r} parent={record.get('parent')!r} "
            f"role={record.get('role')!r} repeat_every={record.get('repeat_every')!r} "
            f"repeat_count={record.get('repeat_count')!r} counter={record.get('counter')!r} "
            f"repeat_mode={record.get('repeat_mode')!r}"
        )
        return record

    def get_node(self, name: str) -> dict[str, Any] | None:
        """Return a registered hierarchy node record, if present."""

        return self.timeline.get_node(name)

    @property
    def nodes(self) -> dict[str, dict[str, Any]]:
        """Return the sequence hierarchy node registry."""

        return self.timeline.nodes


    def _attach_implicit_block_after_relationship(
        self,
        *,
        target_block: SeqStarBlock,
        reference_block: SeqStarBlock | None,
    ) -> None:
        """Attach an implicit block-after relationship created by add_block order.

        This is the first bridge from the concrete SeqStar timeline into the
        relationship layer.  Developers should not have to restate block order:
        the sequential ``seq.add_block(...)`` calls are the source of truth.
        This helper records that ordering as an implicit relationship so it can
        be resolved, validated, exported to debug JSON, and visualized.
        """

        if reference_block is None:
            return

        try:
            from pypulseq_star.relationships.timing import block_after
        except Exception as exc:
            self._debug(
                "could not attach implicit block_after relationship for "
                f"{_metadata_value(target_block, 'seqstar_node')!r}: {exc}"
            )
            return

        target_node = _metadata_value(target_block, "seqstar_node") or getattr(target_block, "name", "block")
        reference_node = _metadata_value(reference_block, "seqstar_node") or getattr(reference_block, "name", "block")
        parent_node = _metadata_value(target_block, "seqstar_parent")

        relationship_name = (
            "implicit_block_after_"
            f"{str(target_node).replace('.', '_')}_after_{str(reference_node).replace('.', '_')}"
        )

        block_after(
            seq=self,
            target_block=target_block,
            reference_block=reference_block,
            name=relationship_name,
            implicit=True,
            source="seq.add_block",
            parent_node=parent_node,
        )

        self._debug(
            f"implicit block_after target={target_node!r} reference={reference_node!r} "
            f"parent={parent_node!r}"
        )

    def add_block(
        self,
        *events: Any,
        role: str | None = None,
        name: str | None = None,
        node: str | None = None,
        varies: list[str] | tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SeqStarBlock:
        """Create and append a block from one or more events.

        ``seq.add_block`` is the source of truth for concrete timeline order.
        The optional ``node``/``role``/``varies`` arguments add SeqStar hierarchy
        metadata without asking developers to repeat block order in a separate
        relationship call.

        Examples
        --------
        PyPulseq-compatible call:

            seq.add_block(rf, gz, role="excitation")

        SeqStar hierarchy-aware call:

            seq.add_block(
                gx,
                adc,
                role="readout",
                node="kernel.readout",
                varies=["rf_phase"],
            )

        Dotted node paths infer parent and local name.  For example,
        ``kernel.readout`` implies parent ``kernel`` and local name ``readout``.

        Event occurrence ownership
        --------------------------
        PyPulseq-style examples often reuse an event object across multiple
        ``add_block`` calls, and some examples mutate an event between block
        insertions.  A sequence timeline must represent concrete event
        occurrences, not a set of live Python object aliases.  Therefore the
        first occurrence is kept by identity for relationship compatibility and
        later occurrences are copied before insertion.
        """

        block_index = len(self.timeline.blocks)
        full_node, _parent_node, local_name = split_node_path(node)
        block_role = role or local_name
        block_name = name or local_name or block_role or f"block_{block_index}"

        block = SeqStarBlock(
            name=block_name,
            role=block_role,
        )

        self._attach_sequence_context(block)

        # Capture the previous block in the same parent node before appending.
        # timeline.append() will update the cursor to the current block.
        _full_node_for_order, _parent_for_order, _local_for_order = split_node_path(full_node)
        if _full_node_for_order is None:
            _parent_for_order = "sequence"
        _parent_for_order = _parent_for_order or "sequence"
        previous_block_for_parent = self.timeline.last_block_by_parent.get(_parent_for_order)

        # Append early enough to annotate the block with hierarchy metadata
        # before event provenance is marked.  The block object is still empty,
        # but its node/role/repeat metadata is now available to event metadata.
        actual_block_index = self.timeline.append(
            block,
            node=full_node,
            role=block_role,
            varies=varies,
            metadata=metadata,
        )

        self._attach_implicit_block_after_relationship(
            target_block=block,
            reference_block=previous_block_for_parent,
        )

        for event_index, event in enumerate(events):
            event_to_add = self._event_occurrence_for_block(
                event,
                block=block,
                block_index=actual_block_index,
                event_index=event_index,
                block_role=block_role,
            )
            self._attach_sequence_context(event_to_add)
            block.add_event(event_to_add)

        self.add_child(block, relationship="contains_block")

        if hasattr(self, "metadata") and isinstance(self.metadata, dict):
            self.metadata["seqstar_nodes"] = self.timeline.nodes
            self.metadata["seqstar_num_blocks"] = len(self.timeline.blocks)

        self._debug(
            f"add_block index={actual_block_index} name={getattr(block, 'name', None)!r} "
            f"node={_metadata_value(block, 'seqstar_node')!r} "
            f"parent={_metadata_value(block, 'seqstar_parent')!r} "
            f"role={block_role!r} varies={list(varies or [])!r} events={len(events)}"
        )

        return block

    def _event_path_for_block(
        self,
        *,
        occurrence: Any,
        block: SeqStarBlock,
        block_index: int,
        event_index: int,
    ) -> str:
        """Return a stable debug path for an event occurrence.

        Some event classes inherit a slotted/dataclass ``path`` field from the
        core node model, but older constructors may not initialize it.  Python's
        generated dataclass ``repr`` then fails when a timeline/block is printed.
        Setting a path when the event enters the sequence fixes that class of
        debug error without requiring every event constructor to know its final
        block/node location.
        """

        block_node = _metadata_value(block, "seqstar_node")
        block_name = getattr(block, "name", None) or f"block_{block_index}"
        event_name = getattr(occurrence, "name", None) or getattr(occurrence, "id", None)
        if event_name is None:
            event_name = f"event_{event_index:03d}"
        event_name = str(event_name)

        if block_node:
            return f"{block_node}.{event_name}"
        return f"{block_name}.{event_name}"

    def _ensure_node_debug_fields(
        self,
        obj: Any,
        *,
        path: str | None = None,
        node: str | None = None,
        parent_node: str | None = None,
        local_name: str | None = None,
    ) -> None:
        """Best-effort initialization of common core-node debug fields.

        This is deliberately conservative: it only fills missing fields that are
        commonly used by dataclass ``repr``/debug tooling.  It does not change
        timing semantics or writer behavior.
        """

        if path is not None:
            _safe_setattr(obj, "path", path)
        if node is not None:
            _safe_setattr(obj, "node", node)
        if parent_node is not None:
            _safe_setattr(obj, "parent_node", parent_node)
        if local_name is not None:
            _safe_setattr(obj, "local_name", local_name)

    def _event_occurrence_for_block(
        self,
        event: Any,
        *,
        block: SeqStarBlock,
        block_index: int,
        event_index: int,
        block_role: str | None,
    ) -> Any:
        """Return the event object to store for one block occurrence.

        The first occurrence is kept by identity for relationship compatibility.
        Repeated occurrences are copied so each concrete block has an
        independently mutable event occurrence.
        """

        if event is None or isinstance(event, (float, int, str, bool)):
            return event

        source_id = id(event)
        occurrences = self._event_occurrence_counts()
        occurrence_index = int(occurrences.get(source_id, 0))

        if occurrence_index == 0:
            occurrence = event
            self._remember_first_event_name(event)
        else:
            occurrence = self._clone_event_occurrence(event)

        occurrences[source_id] = occurrence_index + 1

        self._mark_event_occurrence(
            occurrence,
            source_event=event,
            source_id=source_id,
            occurrence_index=occurrence_index,
            block=block,
            block_index=block_index,
            event_index=event_index,
            block_role=block_role,
        )

        return occurrence

    def _event_occurrence_counts(self) -> dict[int, int]:
        """Return mutable original-object occurrence counts."""

        if hasattr(self, "metadata") and isinstance(self.metadata, dict):
            counts = self.metadata.setdefault("event_object_occurrences", {})
            if isinstance(counts, dict):
                return counts

        # Fallback should rarely be used, but keeps the class robust if the base
        # node model changes and no metadata dict is available.
        return {}

    def _remember_first_event_name(self, event: Any) -> None:
        """Store a human-readable name for debug reports."""

        if not (hasattr(self, "metadata") and isinstance(self.metadata, dict)):
            return

        table = self.metadata.setdefault("event_object_first_names", {})
        if not isinstance(table, dict):
            return

        try:
            name = getattr(event, "name", None) or getattr(event, "id", None)
        except Exception:
            name = None

        table.setdefault(id(event), str(name or event.__class__.__name__))

    def _clone_event_occurrence(self, event: Any) -> Any:
        """Best-effort event copy for a concrete timeline occurrence."""

        if hasattr(event, "copy") and callable(getattr(event, "copy")):
            try:
                return event.copy()
            except TypeError:
                pass
            except Exception:
                pass

        try:
            return copy.deepcopy(event)
        except Exception:
            try:
                return copy.copy(event)
            except Exception:
                # Last resort: keep the original.  The resolver-side guard will
                # still catch impossible shared-object sensitivities.
                return event

    def _mark_event_occurrence(
        self,
        occurrence: Any,
        *,
        source_event: Any,
        source_id: int,
        occurrence_index: int,
        block: SeqStarBlock,
        block_index: int,
        event_index: int,
        block_role: str | None,
    ) -> None:
        """Attach provenance metadata to an event occurrence when possible."""

        source_name = getattr(source_event, "name", None)
        occurrence_name = getattr(occurrence, "name", None)

        block_node = _metadata_value(block, "seqstar_node")
        block_parent = _metadata_value(block, "seqstar_parent")
        block_local_name = _metadata_value(block, "seqstar_local_name")
        block_varies = _metadata_value(block, "seqstar_varies") or []
        block_counter = _metadata_value(block, "seqstar_counter")
        block_repeat_parent = _metadata_value(block, "seqstar_repeat_parent")
        event_path = self._event_path_for_block(
            occurrence=occurrence,
            block=block,
            block_index=block_index,
            event_index=event_index,
        )

        self._ensure_node_debug_fields(
            occurrence,
            path=event_path,
            node=block_node,
            parent_node=block_parent,
            local_name=str(getattr(occurrence, "name", None) or f"event_{event_index:03d}"),
        )

        # Keep copied names readable and unique for debug/writer output while
        # avoiding name changes for the first occurrence used by relationships.
        if occurrence_index > 0 and occurrence_name:
            try:
                setattr(occurrence, "name", f"{occurrence_name}_occ{occurrence_index:03d}")
            except Exception:
                pass
            # Rebuild the debug path after assigning a copied-occurrence name.
            event_path = self._event_path_for_block(
                occurrence=occurrence,
                block=block,
                block_index=block_index,
                event_index=event_index,
            )
            self._ensure_node_debug_fields(
                occurrence,
                path=event_path,
                node=block_node,
                parent_node=block_parent,
                local_name=str(getattr(occurrence, "name", None) or f"event_{event_index:03d}"),
            )

        metadata = getattr(occurrence, "metadata", None)
        if isinstance(metadata, dict):
            metadata.setdefault("sequence_name", self.name)
            metadata["source_event_object_id"] = source_id
            metadata["source_event_name"] = str(source_name or occurrence_name or "event")
            metadata["sequence_occurrence_index"] = occurrence_index
            metadata["sequence_block_index"] = block_index
            metadata["sequence_event_index"] = event_index
            metadata["sequence_block_name"] = getattr(block, "name", None)
            metadata["sequence_block_role"] = block_role
            metadata["sequence_event_path"] = event_path
            metadata["path"] = event_path
            metadata["sequence_block_node"] = block_node
            metadata["sequence_block_parent"] = block_parent
            metadata["sequence_block_local_name"] = block_local_name
            metadata["sequence_block_varies"] = list(block_varies)
            metadata["sequence_counter"] = block_counter
            metadata["sequence_repeat_parent"] = block_repeat_parent
            metadata["seqstar_node"] = block_node
            metadata["seqstar_parent"] = block_parent
            metadata["seqstar_role"] = block_role
            metadata["seqstar_varies"] = list(block_varies)
            metadata["is_copied_occurrence"] = occurrence is not source_event

        parameters = getattr(occurrence, "parameters", None)
        if isinstance(parameters, dict):
            parameters.setdefault("sequence_name", self.name)
            parameters["source_event_object_id"] = source_id
            parameters["source_event_name"] = str(source_name or occurrence_name or "event")
            parameters["sequence_occurrence_index"] = occurrence_index
            parameters["sequence_block_index"] = block_index
            parameters["sequence_event_index"] = event_index
            parameters["sequence_event_path"] = event_path
            parameters["path"] = event_path
            parameters["sequence_block_role"] = block_role
            parameters["sequence_block_node"] = block_node
            parameters["sequence_block_parent"] = block_parent
            parameters["sequence_block_local_name"] = block_local_name
            parameters["sequence_block_varies"] = list(block_varies)
            parameters["sequence_counter"] = block_counter
            parameters["sequence_repeat_parent"] = block_repeat_parent
            parameters["seqstar_node"] = block_node
            parameters["seqstar_parent"] = block_parent
            parameters["seqstar_role"] = block_role
            parameters["seqstar_varies"] = list(block_varies)
            parameters["is_copied_occurrence"] = occurrence is not source_event

    def add_repeating_block(
        self,
        *events: Any,
        repetitions: int | None = None,
        tr: float | None = None,
        role: str | None = None,
        name: str | None = None,
        expression: str | None = None,
    ) -> SeqStarBlock:
        """Append a block and mark it as a repeated kernel.

        If ``repetitions`` or ``tr`` are not explicitly supplied, they are
        resolved from the sequence protocol parameters.

        Accepted protocol keys:
            repetitions:
                "repetitions", "average", "averages", "n_avg", "num_averages"

            TR:
                "repetition_time", "TR", "tr"

        The expression is optional and developer-controlled. If not supplied,
        no timing expression is injected into the relationship metadata.
        """

        if repetitions is None:
            repetitions = self._get_protocol_int(
                "repetitions",
                "average",
                "averages",
                "n_avg",
                "num_averages",
                default=1,
            )

        if tr is None:
            tr = self._get_protocol_float(
                "repetition_time",
                "TR",
                "tr",
                default=None,
            )

        if repetitions < 1:
            raise ValueError("repetitions must be at least 1")

        if tr is None:
            raise ValueError(
                "TR/repetition_time must be supplied either as the 'tr' argument "
                "or in sequence parameters, e.g. parameters={'TR': 2.0} or "
                "parameters={'repetition_time': 2.0}."
            )

        if tr <= 0:
            raise ValueError("TR/repetition_time must be positive")

        block = self.add_block(
            *events,
            role=role or "kernel",
            name=name or "kernel",
        )

        # Store canonical protocol-derived values on the sequence.
        self.parameters["repetitions"] = repetitions
        self.parameters["repetition_time"] = tr

        # Also store them on the block, because the block is the repeated object.
        block.parameters["repetitions"] = repetitions
        block.parameters["repetition_time"] = tr

        relationship_metadata: dict[str, Any] = {
            "counter": "repetition_counter",
            "count": repetitions,
            "interval_s": tr,
        }

        if expression is not None:
            relationship_metadata["expression"] = expression

        self.relationships.append(
            SeqStarRelationship(
                source=self.name,
                target=block.name,
                kind="repeats_every",
                metadata=relationship_metadata,
            )
        )

        return block

    def get_protocol_parameter(self, *keys: str, default: Any = None) -> Any:
        """Return the first matching protocol/sequence parameter.

        This allows callers to use either canonical protocol keys or older
        PyPulseq-style shorthand keys.
        """

        for key in keys:
            if key in self.parameters:
                return self.parameters[key]

        return default

    def set_protocol_parameter(self, key: str, value: Any) -> None:
        """Set or update a sequence-level protocol parameter."""

        self.parameters[key] = value

    def system_dict(self) -> dict[str, Any]:
        """Return the system dictionary, if supported by Opts."""

        if hasattr(self.system, "to_dict"):
            return self.system.to_dict()

        return dict(vars(self.system))

    def protocol_dict(self) -> dict[str, Any]:
        """Return sequence-level protocol parameters."""

        return dict(self.parameters)

    def context_dict(self) -> dict[str, Any]:
        """Return combined sequence context for writers/exporters."""

        return {
            "name": self.name,
            "system": self.system_dict(),
            "protocol": self.protocol_dict(),
        }

    def _get_protocol_int(
        self,
        *keys: str,
        default: int | None = None,
    ) -> int:
        value = self.get_protocol_parameter(*keys, default=default)

        if value is None:
            raise ValueError(f"Missing required integer protocol parameter from keys: {keys}")

        return int(value)

    def _get_protocol_float(
        self,
        *keys: str,
        default: float | None = None,
    ) -> float | None:
        value = self.get_protocol_parameter(*keys, default=default)

        if value is None:
            return None

        return float(value)

    def _attach_sequence_context(self, obj: Any) -> None:
        """Attach system/protocol context to blocks or events when supported.

        This is intentionally non-invasive. It supports several possible future
        class styles:

        - obj.bind_system(system)
        - obj.system = system
        - obj.parameters[...] update
        - obj.metadata[...] update
        """

        if hasattr(obj, "bind_system"):
            obj.bind_system(self.system)
        elif hasattr(obj, "system"):
            try:
                obj.system = self.system
            except AttributeError:
                pass

        if hasattr(obj, "parameters") and isinstance(obj.parameters, dict):
            obj.parameters.setdefault("sequence_name", self.name)

            for key, value in self.parameters.items():
                obj.parameters.setdefault(key, value)

        if hasattr(obj, "metadata") and isinstance(obj.metadata, dict):
            obj.metadata.setdefault("sequence_name", self.name)
            
    def set_definition(self, key: str, value: Any) -> None:
        """Set a Pulseq-style sequence definition.

        PyPulseq stores sequence definitions such as FOV and Name separately.
        In PyPulseq-Star, we store them in the existing sequence-level
        parameter dictionary so they remain visible to Pulseq and gammaSTAR
        writers.
        """

        self.parameters[key] = value

    def get_definition(self, key: str, default: Any = None) -> Any:
        """Return a Pulseq-style sequence definition."""

        return self.parameters.get(key, default)

    def calc_duration(self, *events: Any) -> float:
        """Return the occupied duration of one or more simultaneous events.

        This delegates to the package-level canonical duration implementation.
        Keeping one timing engine prevents the sequence API, relationship
        resolver, timing checker, plotter, and writers from assigning different
        durations to the same event.

        For multiple events in one block, the result is the maximum occupied
        extent, matching PyPulseq block-duration semantics.
        """

        return _shared_calc_duration(*events)

    def calc_timeline_duration(self, blocks: Any) -> float:
        """Return elapsed duration of sequential blocks.

        ``calc_duration(a, b)`` follows PyPulseq semantics and treats ``a`` and
        ``b`` as simultaneous events, returning their maximum occupied extent.
        This complementary operation sums the canonical occupied duration of an
        ordered block group.

        Parameters
        ----------
        blocks
            Iterable of blocks in executable order.

        Returns
        -------
        float
            Total elapsed duration in seconds.
        """

        return sum(
            float(_shared_calc_duration(block))
            for block in blocks
            if block is not None
        )


    def calc_rf_center(self, rf_event: Any) -> tuple[float, int]:
        """Return the RF center time and sample index.

        The first return value mirrors ``pypulseq.calc_rf_center(rf)[0]``:
        RF center time measured from the start of the block, including RF delay.

        The second return value is a best-effort center sample index.
        """

        delay = self._event_delay(rf_event)
        active_duration = self._event_active_duration(rf_event)
        asymmetry = float(getattr(rf_event, "asymmetry", 0.5) or 0.5)

        center_time = delay + asymmetry * active_duration

        samples = getattr(rf_event, "samples", None)
        if samples is None:
            shape = getattr(rf_event, "shape", None)
            samples = getattr(shape, "samples", None)

        try:
            center_index = int(round(asymmetry * (len(samples) - 1)))
        except Exception:
            center_index = 0

        return center_time, center_index

    def _event_time_extent(self, event: Any) -> float:
        """Return event delay plus active duration."""

        delay = self._event_delay(event)
        active_duration = self._event_active_duration(event)

        # Delay-only events often encode their duration as ``delay``.
        if active_duration == 0.0:
            event_type = str(getattr(event, "type", getattr(event, "kind", ""))).lower()
            if event_type == "delay":
                return delay

        return delay + active_duration

    def _event_delay(self, event: Any) -> float:
        """Return event delay in seconds."""

        return float(getattr(event, "delay", 0.0) or 0.0)
    
    def calc_delay(self, event: Any) -> float:
        """Return event delay in seconds.

        Public PyPulseq-Star timing accessor for examples and sequence
        calculations. Internally, this delegates to the existing private
        helper so demos do not call ``_event_delay`` directly.
        """

        return self._event_delay(event)

    def _event_active_duration(self, event: Any) -> float:
        """Return event active duration in seconds, excluding event delay."""

        for attr in ("duration", "shape_dur", "active_duration"):
            value = getattr(event, attr, None)
            if value not in (None, 0):
                return float(value)

        rise = float(getattr(event, "rise_time", 0.0) or 0.0)
        flat = float(getattr(event, "flat_time", 0.0) or 0.0)
        fall = float(getattr(event, "fall_time", 0.0) or 0.0)

        if rise or flat or fall:
            return rise + flat + fall

        timing = getattr(event, "timing", None)
        if timing is not None:
            value = getattr(timing, "duration", None)
            if value not in (None, 0):
                return float(value)

        shape = getattr(event, "shape", None)
        if shape is not None:
            for attr in ("duration", "shape_dur", "active_duration"):
                value = getattr(shape, attr, None)
                if value not in (None, 0):
                    return float(value)

        return 0.0

    

    def check_timing(self) -> tuple[bool, list]:
        """Check timing of the sequence.

        Mirrors PyPulseq usage:

            ok, error_report = seq.check_timing()
        """

        return check_timing(self)
    
    @property
    def grad_raster_time(self) -> float:
        """Return gradient raster time in seconds.

        PyPulseq-style compatibility accessor used by examples.
        """

        return float(getattr(self.system, "grad_raster_time", 10e-6))

    @property
    def rf_raster_time(self) -> float:
        """Return RF raster time in seconds."""

        return float(getattr(self.system, "rf_raster_time", 1e-6))

    @property
    def adc_raster_time(self) -> float:
        """Return ADC raster time in seconds."""

        return float(getattr(self.system, "adc_raster_time", 1e-7))

    @property
    def block_duration_raster(self) -> float:
        """Return block duration raster time in seconds."""

        return float(getattr(self.system, "block_duration_raster", self.grad_raster_time))
    
    def plot(
        self,
        time_range: tuple[float, float] | None = None,
        *,
        title: str | None = None,
        show: bool = True,
        save: str | None = None,
        one_tr: bool = False,
        rf_scale: str = "normalized",
        gradient_scale: str = "auto",
        figsize: tuple[float, float] = (15.0, 8.5),
        dpi: int = 140,
        **_: Any,
    ):
        """Plot the sequence using a gammaSTAR-like stacked timeline.

        Mirrors PyPulseq usage:

            seq.plot()
            seq.plot(time_range=(0, 2.0))

        Additional pypulseq_star options:

            seq.plot(one_tr=True)
            seq.plot(save="sequence.png")
            seq.plot(rf_scale="tesla", gradient_scale="mt_per_m")
        """

        

        return plot(
            self,
            time_range=time_range,
            title=title,
            show=show,
            save=save,
            one_tr=one_tr,
            rf_scale=rf_scale,
            gradient_scale=gradient_scale,
            figsize=figsize,
            dpi=dpi,
        )
