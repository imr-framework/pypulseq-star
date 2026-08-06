"""Generic timing relationship APIs.

These are developer-facing convenience functions backed by generic relationship
records. They are intentionally not FID-specific.

Key timing convention
---------------------
Relationships are resolved in global sequence time, then written back into the
local coordinate frame of the target event's parent block. This matters for
PyPulseq-style sequences built with sequential ``seq.add_block(...)`` calls:

    RF/Gz block -> prephase block -> TE delay block -> readout block

For such sequences, ``target.delay`` is a local delay inside the target block,
not a global time from sequence start.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from .context import (
    event_duration,
    event_name,
    protocol_float,
    set_event_property,
    store_derived_parameter,
)
from .expression import expr
from .relationship import SeqStarRelationship, attach_relationship, get_relationships
from .validation import SeqStarValidationResult, validation_result

# Object-id based lookup is intentionally simple for Phase 1. The relationship
# record keeps object ids for debug/serialization safety, but the live
# relationship object also needs to resolve against in-memory event objects.
_OBJECT_REGISTRY: dict[int, Any] = {}


def _relationship_debug_enabled() -> bool:
    """Return True when verbose relationship debug printing is enabled."""

    value = os.environ.get("PYPULSEQ_STAR_RELATIONSHIP_DEBUG", "")
    return str(value).strip().lower() in {"1", "true", "yes", "on", "debug"}


def _relationship_debug(message: str) -> None:
    if _relationship_debug_enabled():
        print(f"[seqstar.relationships.timing] {message}")


def set_center_after(
    *,
    seq: Any,
    target: Any,
    reference: Any,
    offset: str | float,
    target_anchor: str = "center",
    reference_anchor: str = "center",
    set_property: str = "delay",
    solve_event: Any | None = None,
    solve_property: str | None = None,
    name: str = "set_center_after",
) -> SeqStarRelationship:
    """Set a target event anchor after a reference event anchor by an offset.

    The generic relationship is:

        target.anchor = reference.anchor + offset

    The resolver adjusts one chosen solve variable to satisfy that relationship.

    Backward-compatible default:
        solve_event = target
        solve_property = set_property

    Examples
    --------
    FID-style relationship, where ADC delay is the timing degree of freedom:

        set_center_after(
            seq=seq,
            target=adc,
            reference=rf,
            offset="TE",
            solve_event=adc,
            solve_property="delay",
        )

    GRE-style relationship, where an explicit TE delay block is the timing
    degree of freedom:

        set_center_after(
            seq=seq,
            target=adc,
            reference=rf,
            offset="TE",
            solve_event=te_delay_event,
            solve_property="duration",
        )

    Timing convention
    -----------------
    Event ``delay`` remains a local event/block offset. It should not be
    overloaded to mean TE or global wait time. Delay/wait events should use
    ``duration`` as the solved wait-time property.
    """

    if solve_event is None:
        solve_event = target

    if solve_property is None:
        solve_property = set_property

    _register_object(target)
    _register_object(reference)
    _register_object(solve_event)

    target_name = event_name(target)
    reference_name = event_name(reference)
    solve_event_name = event_name(solve_event)

    rel = SeqStarRelationship(
        name=name,
        relation_type="timing.set_center_after",
        description=(
            f"Set {target_name} {target_anchor} after "
            f"{reference_name} {reference_anchor} by adjusting "
            f"{solve_event_name}.{solve_property}."
        ),
        target={
            "event": target_name,
            "anchor": target_anchor,
            "property": set_property,
        },
        reference={
            "event": reference_name,
            "anchor": reference_anchor,
        },
        protocol_parameters={
            "required": [offset] if isinstance(offset, str) else [],
            "independent": [offset] if isinstance(offset, str) else [],
        },
        derived_parameters=[
            f"{reference_name}.{reference_anchor}",
            f"{target_name}.duration",
            f"{target_name}.{target_anchor}",
            f"{solve_event_name}.{solve_property}",
        ],
        expression=expr(
            f"{target_name}.{target_anchor} = "
            f"{reference_name}.{reference_anchor} + {offset}",
            lua="return reference_anchor + offset",
            inputs={
                "reference_anchor": f"{reference_name}.{reference_anchor}",
                "offset": str(offset),
                "target_duration": f"{target_name}.duration",
                "solve_value": f"{solve_event_name}.{solve_property}",
            },
            unit="s",
        ),
        metadata={
            "resolver": "set_center_after",
            "target_object_id": id(target),
            "reference_object_id": id(reference),
            "solve_object_id": id(solve_event),
            "target_object": target,
            "reference_object": reference,
            "solve_object": solve_event,
            "offset": offset,
            "target_anchor": target_anchor,
            "reference_anchor": reference_anchor,
            # Backward-compatible name. Older code may still inspect this.
            "set_property": set_property,
            # New generic solve API.
            "solve_event": solve_event_name,
            "solve_property": solve_property,
            "coordinate_model": "global_anchor_with_explicit_solve_variable",
        },
    )

    attach_relationship(seq, rel)
    return rel


def set_anchor_after(
    *,
    seq: Any,
    target: Any,
    reference: Any,
    offset: str | float = 0.0,
    target_anchor: str = "center",
    reference_anchor: str = "center",
    set_property: str = "delay",
    solve_event: Any | None = None,
    solve_property: str | None = None,
    name: str = "set_anchor_after",
) -> SeqStarRelationship:
    """Set any target anchor after any reference anchor by an offset.

    This is the generic form behind ``set_center_after``. It is intended for
    coupling simultaneously meaningful events such as a readout gradient and an
    ADC window:

        set_anchor_after(
            seq=seq,
            target=gx_readout,
            target_anchor="center",
            reference=adc_readout,
            reference_anchor="center",
            offset=0.0,
            solve_event=gx_readout,
            solve_property="delay",
            name="gx_center_follows_adc_center",
        )

    The relationship itself is always expressed in global time:

        target.anchor_global = reference.anchor_global + offset

    The requested ``solve_event.solve_property`` is then adjusted in the local
    coordinate system of the solve event's parent block.
    """

    if solve_event is None:
        solve_event = target

    if solve_property is None:
        solve_property = set_property

    _register_object(target)
    _register_object(reference)
    _register_object(solve_event)

    target_name = event_name(target)
    reference_name = event_name(reference)
    solve_event_name = event_name(solve_event)

    rel = SeqStarRelationship(
        name=name,
        relation_type="timing.set_anchor_after",
        description=(
            f"Set {target_name} {target_anchor} after "
            f"{reference_name} {reference_anchor} by adjusting "
            f"{solve_event_name}.{solve_property}."
        ),
        target={
            "event": target_name,
            "anchor": target_anchor,
            "property": set_property,
        },
        reference={
            "event": reference_name,
            "anchor": reference_anchor,
        },
        protocol_parameters={
            "required": [offset] if isinstance(offset, str) else [],
            "independent": [offset] if isinstance(offset, str) else [],
        },
        derived_parameters=[
            f"{reference_name}.{reference_anchor}",
            f"{target_name}.duration",
            f"{target_name}.{target_anchor}",
            f"{solve_event_name}.{solve_property}",
        ],
        expression=expr(
            f"{target_name}.{target_anchor} = "
            f"{reference_name}.{reference_anchor} + {offset}",
            lua="return reference_anchor + offset",
            inputs={
                "reference_anchor": f"{reference_name}.{reference_anchor}",
                "offset": str(offset),
                "target_duration": f"{target_name}.duration",
                "solve_value": f"{solve_event_name}.{solve_property}",
            },
            unit="s",
        ),
        metadata={
            "resolver": "set_anchor_after",
            "target_object_id": id(target),
            "reference_object_id": id(reference),
            "solve_object_id": id(solve_event),
            "target_object": target,
            "reference_object": reference,
            "solve_object": solve_event,
            "offset": offset,
            "target_anchor": target_anchor,
            "reference_anchor": reference_anchor,
            "set_property": set_property,
            "solve_event": solve_event_name,
            "solve_property": solve_property,
            "coordinate_model": "global_anchor_with_explicit_solve_variable",
        },
    )

    attach_relationship(seq, rel)
    return rel


def set_same_anchor(
    *,
    seq: Any,
    target: Any,
    reference: Any,
    anchor: str = "center",
    target_anchor: str | None = None,
    reference_anchor: str | None = None,
    solve_event: Any | None = None,
    solve_property: str = "delay",
    name: str = "set_same_anchor",
) -> SeqStarRelationship:
    """Constrain a target anchor to equal a reference anchor.

    This is a readability wrapper for ``set_anchor_after(..., offset=0)`` and
    is useful for readout/ADC coupling. It is generic; it does not know GRE,
    EPI, or TSE.
    """

    return set_anchor_after(
        seq=seq,
        target=target,
        reference=reference,
        offset=0.0,
        target_anchor=target_anchor or anchor,
        reference_anchor=reference_anchor or anchor,
        solve_event=solve_event if solve_event is not None else target,
        solve_property=solve_property,
        name=name,
    )


def set_center_equal(
    *,
    seq: Any,
    target: Any,
    reference: Any,
    solve_event: Any | None = None,
    solve_property: str = "delay",
    name: str = "set_center_equal",
) -> SeqStarRelationship:
    """Constrain ``target.center == reference.center``.

    Convenience API for tying an ADC and its readout gradient together.
    """

    return set_same_anchor(
        seq=seq,
        target=target,
        reference=reference,
        anchor="center",
        solve_event=solve_event if solve_event is not None else target,
        solve_property=solve_property,
        name=name,
    )


def require_same_anchor(
    *,
    seq: Any,
    target: Any,
    reference: Any,
    anchor: str = "center",
    target_anchor: str | None = None,
    reference_anchor: str | None = None,
    offset: str | float = 0.0,
    tolerance: float | None = None,
    name: str = "require_same_anchor",
) -> SeqStarRelationship:
    """Require two event anchors to match, without solving or mutating events.

    This is a validation-only timing relationship. It is intentionally different
    from ``set_center_after`` / ``set_anchor_after``: no solve variable is
    changed. Use it when two events should already be aligned by construction,
    such as a readout gradient and its ADC window:

        require_same_anchor(
            seq=seq,
            target=gx_readout,
            reference=adc_readout,
            anchor="center",
            name="gre_gx_center_requires_adc_center",
        )

    The relationship validated is:

        target.target_anchor_global = reference.reference_anchor_global + offset

    ``offset`` can be a literal number of seconds or a protocol parameter name.
    ``tolerance`` defaults to a conservative timing raster derived from the
    sequence system.
    """

    target_anchor = target_anchor or anchor
    reference_anchor = reference_anchor or anchor

    _register_object(target)
    _register_object(reference)

    target_name = event_name(target)
    reference_name = event_name(reference)

    rel = SeqStarRelationship(
        name=name,
        relation_type="timing.require_same_anchor",
        description=(
            f"Require {target_name} {target_anchor} to equal "
            f"{reference_name} {reference_anchor} plus {offset}, without solving."
        ),
        target={
            "event": target_name,
            "anchor": target_anchor,
            "property": None,
        },
        reference={
            "event": reference_name,
            "anchor": reference_anchor,
        },
        protocol_parameters={
            "required": [offset] if isinstance(offset, str) else [],
            "independent": [offset] if isinstance(offset, str) else [],
        },
        derived_parameters=[
            f"{reference_name}.{reference_anchor}",
            f"{target_name}.{target_anchor}",
            f"{target_name}.{target_anchor}_minus_{reference_name}.{reference_anchor}",
        ],
        expression=expr(
            f"{target_name}.{target_anchor} = {reference_name}.{reference_anchor} + {offset}",
            lua="return target_anchor - reference_anchor - offset",
            inputs={
                "target_anchor": f"{target_name}.{target_anchor}",
                "reference_anchor": f"{reference_name}.{reference_anchor}",
                "offset": str(offset),
            },
            unit="s",
        ),
        metadata={
            "resolver": "require_same_anchor",
            "target_object_id": id(target),
            "reference_object_id": id(reference),
            "target_object": target,
            "reference_object": reference,
            "offset": offset,
            "target_anchor": target_anchor,
            "reference_anchor": reference_anchor,
            "tolerance": tolerance,
            "coordinate_model": "global_anchor_validation_only",
            "validation_only": True,
        },
    )

    attach_relationship(seq, rel)
    return rel


def use_same_anchor(
    *,
    seq: Any,
    target: Any,
    reference: Any,
    anchor: str = "center",
    target_anchor: str | None = None,
    reference_anchor: str | None = None,
    offset: str | float = 0.0,
    tolerance: float | None = None,
    name: str = "use_same_anchor",
) -> SeqStarRelationship:
    """Alias for ``require_same_anchor`` with softer wording.

    This is validation-only and does not mutate either event.
    """

    return require_same_anchor(
        seq=seq,
        target=target,
        reference=reference,
        anchor=anchor,
        target_anchor=target_anchor,
        reference_anchor=reference_anchor,
        offset=offset,
        tolerance=tolerance,
        name=name,
    )


def set_readout_after(
    *,
    seq: Any,
    adc: Any,
    readout_gradient: Any,
    reference: Any,
    offset: str | float,
    adc_anchor: str = "center",
    readout_anchor: str = "center",
    reference_anchor: str = "center",
    solve_event: Any | None = None,
    solve_property: str | None = None,
    adc_name: str | None = None,
    readout_name: str | None = None,
    name: str = "set_readout_after",
) -> list[SeqStarRelationship]:
    """Composite relationship for protocol-timed readout placement.

    This convenience API keeps simple Cartesian examples readable while still
    preserving the explicit relationship graph. It creates two relationships:

    1. A solved relationship that places the ADC anchor after a reference anchor
       by a protocol or numeric offset, for example ``ADC.center = RF.center + TE``.
    2. A validation-only relationship that requires the readout gradient anchor
       to match the ADC anchor, for example ``Gx.center == ADC.center``.

    No sequence-specific assumptions are made. The function does not know GRE,
    FLASH, EPI, or TSE. It only encodes the common acquisition pattern that an
    ADC event is the protocol-timed readout anchor and the physical readout
    gradient must remain aligned with that ADC event.

    For EPI or other multi-window readouts, this may eventually be superseded by
    a richer composite that understands ADC trains, alternating polarity, and
    per-window gradient lobes. For a first GRE-style readout, this keeps the demo
    compact while still exposing the two underlying relationships for debugging.

    Example
    -------
    set_readout_after(
        seq=seq,
        adc=adc,
        readout_gradient=gx,
        reference=rf,
        offset="TE",
        solve_event=te_delay_event,
        solve_property="duration",
        name="gre_readout_after_rf_center",
    )

    Returns
    -------
    list[SeqStarRelationship]
        ``[adc_timing_relationship, readout_alignment_relationship]``.
    """

    base_name = str(name or "set_readout_after")

    adc_relationship_name = adc_name or f"{base_name}_adc_{adc_anchor}_after_reference_{reference_anchor}"
    readout_relationship_name = readout_name or f"{base_name}_readout_{readout_anchor}_requires_adc_{adc_anchor}"

    adc_rel = set_anchor_after(
        seq=seq,
        target=adc,
        target_anchor=adc_anchor,
        reference=reference,
        reference_anchor=reference_anchor,
        offset=offset,
        solve_event=solve_event if solve_event is not None else adc,
        solve_property=solve_property,
        name=adc_relationship_name,
    )

    readout_rel = require_same_anchor(
        seq=seq,
        target=readout_gradient,
        target_anchor=readout_anchor,
        reference=adc,
        reference_anchor=adc_anchor,
        offset=0.0,
        name=readout_relationship_name,
    )

    # Mark the two child relationships as belonging to one composite helper.
    # This is debug metadata only; the resolver still handles each child
    # relationship with its native semantics.
    for index, child in enumerate((adc_rel, readout_rel)):
        child.metadata.setdefault("composite_relationship", base_name)
        child.metadata.setdefault("composite_api", "set_readout_after")
        child.metadata.setdefault("composite_child_index", index)
        child.metadata.setdefault("composite_child_count", 2)

    return [adc_rel, readout_rel]

def fill_to_period(
    *,
    seq: Any,
    blocks: Iterable[Any],
    period: str | float,
    fill_event: Any,
    fill_property: str = "duration",
    name: str = "fill_to_period",
) -> SeqStarRelationship:
    """Solve one structural fill event so a block group occupies ``period``.

    The blocks must be supplied in executable order. The fill event may share
    its final block with RF, gradients, ADC, or other events; the resolver
    verifies that the solved fill remains at least as long as every simultaneous
    non-fill event in that block.
    """
    block_list = list(blocks)
    if not block_list:
        raise ValueError("fill_to_period requires at least one block.")
    if fill_property not in {"duration", "wait", "wait_time", "delay_duration"}:
        raise ValueError("fill_to_period currently requires a duration-like fill_property.")
    _register_object(fill_event)
    for block in block_list:
        _register_object(block)
    rel = SeqStarRelationship(
        name=name,
        relation_type="timing.fill_to_period",
        description=(
            f"Set {event_name(fill_event)}.{fill_property} so the selected "
            f"block group occupies {period}."
        ),
        target={"event": event_name(fill_event), "property": fill_property},
        protocol_parameters={
            "required": [period] if isinstance(period, str) else [],
            "independent": [period] if isinstance(period, str) else [],
        },
        derived_parameters=["group.duration", f"{event_name(fill_event)}.{fill_property}"],
        expression=expr(
            "fill = period - duration_before_fill_block",
            lua="return period - duration_before_fill_block",
            inputs={"period": str(period), "duration_before_fill_block": "group.duration_before_fill"},
            unit="s",
        ),
        metadata={
            "resolver": "fill_to_period",
            "block_object_ids": [id(block) for block in block_list],
            "block_objects": block_list,
            "fill_object_id": id(fill_event),
            "fill_object": fill_event,
            "fill_property": fill_property,
            "period": period,
            "coordinate_model": "ordered_block_group_period",
        },
    )
    attach_relationship(seq, rel)
    return rel


def repeat_every(
    *,
    seq: Any,
    events: list[Any] | tuple[Any, ...],
    period: str | float,
    name: str = "repeat_every",
) -> SeqStarRelationship:
    """Declare that a group of events repeats every period.

    This validates feasibility and stores the symbolic relationship:

        fill_delay = period - kernel_duration
    """

    for event in events:
        _register_object(event)

    event_names = [event_name(event) for event in events]

    rel = SeqStarRelationship(
        name=name,
        relation_type="timing.repeat_every",
        description=f"Repeat events {event_names} every {period}.",
        target={
            "node": "kernel",
            "property": "repetition_time",
            "events": event_names,
        },
        protocol_parameters={
            "required": [period] if isinstance(period, str) else [],
            "independent": [period] if isinstance(period, str) else [],
        },
        derived_parameters=[
            "kernel.duration",
            "tr_fill_delay",
        ],
        expression=expr(
            "fill_delay = period - kernel_duration",
            lua="return period - kernel_duration",
            inputs={
                "period": str(period),
                "kernel_duration": "kernel.duration",
            },
            unit="s",
        ),
        metadata={
            "resolver": "repeat_every",
            "event_object_ids": [id(event) for event in events],
            "event_objects": list(events),
            "period": period,
        },
    )

    attach_relationship(seq, rel)
    return rel


def block_after(
    *,
    seq: Any,
    target_block: Any,
    reference_block: Any,
    name: str | None = None,
    implicit: bool = False,
    source: str = "developer",
    parent_node: str | None = None,
) -> SeqStarRelationship:
    """Declare that one block starts when another block ends.

    This is the generic block-order primitive created implicitly by
    ``SeqStarSequence.add_block``.  It is intentionally not sequence-specific:
    it only records the executable timeline fact

        target_block.start = reference_block.end

    The resolver validates the already-built timeline and stores the resolved
    block starts/durations.  It does not mutate events or blocks in this phase.
    """

    _register_object(target_block)
    _register_object(reference_block)

    target_name = _block_display_name(target_block)
    reference_name = _block_display_name(reference_block)
    relationship_name = name or f"block_after_{target_name}_after_{reference_name}"

    rel = SeqStarRelationship(
        name=str(relationship_name),
        relation_type="timing.block_after",
        description=(
            f"Require block {target_name} to start after block "
            f"{reference_name} ends."
        ),
        target={
            "block": target_name,
            "node": _block_node(target_block),
            "anchor": "start",
            "property": "tstart",
        },
        reference={
            "block": reference_name,
            "node": _block_node(reference_block),
            "anchor": "end",
        },
        derived_parameters=[
            f"{reference_name}.end",
            f"{target_name}.start",
            f"{target_name}.start_minus_{reference_name}.end",
        ],
        expression=expr(
            f"{target_name}.start = {reference_name}.end",
            lua="return reference_end",
            inputs={"reference_end": f"{reference_name}.end"},
            unit="s",
        ),
        metadata={
            "resolver": "block_after",
            "target_block_object_id": id(target_block),
            "reference_block_object_id": id(reference_block),
            "target_block_object": target_block,
            "reference_block_object": reference_block,
            "target_block": target_name,
            "reference_block": reference_name,
            "target_block_node": _block_node(target_block),
            "reference_block_node": _block_node(reference_block),
            "parent_node": parent_node,
            "implicit": bool(implicit),
            "developer_defined": not bool(implicit),
            "source": source,
            "relationship_source": source,
            "coordinate_model": "block_order_global_timeline",
            "validation_only": True,
        },
    )

    attach_relationship(seq, rel)
    return rel


def resolve(seq: Any) -> list[SeqStarRelationship]:
    """Resolve all relationships attached to a sequence.

    Relationships are resolved generically in sequence/global coordinates and
    then written back to local event properties. This avoids double-applying TE
    when a sequence already realizes TE structurally with blocks and delay
    events.
    """

    _register_relationship_objects(seq)
    relationships = get_relationships(seq)

    for rel in relationships:
        resolver = rel.metadata.get("resolver")

        if resolver in {"set_center_after", "set_anchor_after"}:
            _resolve_set_center_after(seq, rel)
        elif resolver == "require_same_anchor":
            _resolve_require_same_anchor(seq, rel)
        elif resolver == "block_after":
            _resolve_block_after(seq, rel)
        elif resolver == "repeat_every":
            _resolve_repeat_every(seq, rel)
        elif resolver == "fill_to_period":
            _resolve_fill_to_period(seq, rel)

    return relationships


def _resolve_set_center_after(seq: Any, rel: SeqStarRelationship) -> None:
    """Resolve target.anchor = reference.anchor + offset.

    Generic solve model
    -------------------
    The relationship itself is always:

        target.anchor_global = reference.anchor_global + offset

    The resolver adjusts the requested solve variable:

        solve_object.solve_property

    Backward-compatible default:
        solve_object = target
        solve_property = set_property

    Examples
    --------
    FID:
        solve_object = adc
        solve_property = "delay"

    GRE:
        solve_object = te_delay_event
        solve_property = "duration"
    """

    target = _live_object_from_relationship(rel, "target")
    reference = _live_object_from_relationship(rel, "reference")
    solve_object = _live_solve_object_from_relationship(rel)

    if solve_object is None:
        solve_object = target

    if target is None or reference is None or solve_object is None:
        raise RuntimeError(
            f"Relationship {rel.name!r} lost access to target/reference/solve objects."
        )

    offset_ref = rel.metadata["offset"]
    offset_value = protocol_float(seq, offset_ref, name=str(offset_ref))

    target_anchor = str(rel.metadata.get("target_anchor", "center"))
    reference_anchor = str(rel.metadata.get("reference_anchor", "center"))

    set_property = str(rel.metadata.get("set_property", "delay"))
    solve_property = str(
        rel.metadata.get(
            "solve_property",
            set_property,
        )
    )

    # Friendly aliases. Internally delay/wait events use duration.
    if solve_property in {"wait", "wait_time", "delay_duration"}:
        solve_property = "duration"

    layout_before = _sequence_timing_layout(seq)

    # Relationships may be declared against the original Python event objects,
    # while SeqStarSequence.add_block() may insert frozen/copied occurrences into
    # the executable timeline. Resolve each endpoint to the concrete timeline
    # occurrence before computing anchors or probing sensitivities. If the
    # original object is already present, this is a no-op.
    target = layout_before.resolve_event(target)
    reference = layout_before.resolve_event(reference)
    solve_object = layout_before.resolve_event(solve_object)

    reference_block = layout_before.event_to_block.get(id(reference))
    target_block = layout_before.event_to_block.get(id(target))
    solve_block = layout_before.event_to_block.get(id(solve_object))

    reference_block_start = layout_before.block_starts.get(id(reference_block), 0.0)
    target_block_start = layout_before.block_starts.get(id(target_block), 0.0)
    solve_block_start = layout_before.block_starts.get(id(solve_block), 0.0)

    reference_anchor_local = _event_anchor_time_local(
        reference,
        reference_anchor,
        include_delay=True,
    )

    target_anchor_local_before = _event_anchor_time_local(
        target,
        target_anchor,
        include_delay=True,
    )

    reference_anchor_global = reference_block_start + reference_anchor_local
    target_anchor_global_before = target_block_start + target_anchor_local_before

    desired_target_anchor_global = reference_anchor_global + offset_value

    timing_error = desired_target_anchor_global - target_anchor_global_before

    old_solve_value = _get_solve_property_value(solve_object, solve_property)

    _relationship_debug(
        f"resolve {rel.name}: target={event_name(target)}#{id(target)} "
        f"ref={event_name(reference)}#{id(reference)} solve={event_name(solve_object)}#{id(solve_object)} "
        f"target_block_start={target_block_start:.9g} ref_block_start={reference_block_start:.9g} "
        f"solve_block_start={solve_block_start:.9g} target_anchor_before={target_anchor_global_before:.9g} "
        f"desired={desired_target_anchor_global:.9g} old_solve={old_solve_value:.9g}"
    )

    sensitivity = _solve_property_sensitivity(
        seq=seq,
        target=target,
        target_anchor=target_anchor,
        solve_object=solve_object,
        solve_property=solve_property,
        layout_before=layout_before,
    )

    solve_occurrences = layout_before.occurrence_count(solve_object)
    target_occurrences = layout_before.occurrence_count(target)
    reference_occurrences = layout_before.occurrence_count(reference)

    _relationship_debug(
        f"sensitivity {rel.name}: sensitivity={sensitivity:.9g} "
        f"target_occurrences={target_occurrences} ref_occurrences={reference_occurrences} "
        f"solve_occurrences={solve_occurrences}"
    )

    _guard_ambiguous_shared_solve_object(
        seq=seq,
        rel=rel,
        solve_object=solve_object,
        solve_property=solve_property,
        sensitivity=sensitivity,
        solve_occurrences=solve_occurrences,
        target_occurrences=target_occurrences,
        reference_occurrences=reference_occurrences,
    )

    if abs(sensitivity) < 1e-12:
        _relationship_debug(
            f"ZERO SENSITIVITY {rel.name}: target_block_start={target_block_start:.9g}, "
            f"reference_block_start={reference_block_start:.9g}, solve_block_start={solve_block_start:.9g}, "
            f"target_anchor_before={target_anchor_global_before:.9g}, desired={desired_target_anchor_global:.9g}, "
            f"old_solve={old_solve_value:.9g}"
        )
        raise ValueError(
            f"Relationship {rel.name!r} cannot be resolved by changing "
            f"{event_name(solve_object)}.{solve_property}: target anchor does not "
            "move with this solve variable.\n"
            f"target={event_name(target)!r}, reference={event_name(reference)!r}, "
            f"solve_object={event_name(solve_object)!r}, "
            f"solve_property={solve_property!r}, "
            f"target_occurrences={target_occurrences}, "
            f"reference_occurrences={reference_occurrences}, "
            f"solve_occurrences={solve_occurrences}."
        )

    raw_solved_value = old_solve_value + timing_error / sensitivity
    solved_value = _snap_solve_value_to_raster(
        seq=seq,
        solve_object=solve_object,
        solve_property=solve_property,
        value=raw_solved_value,
        mode="ceil",
    )

    system = getattr(seq, "system", None)
    adc_dead_time = float(getattr(system, "adc_dead_time", 0.0) or 0.0)

    validation: list[SeqStarValidationResult] = [
        validation_result(
            name=f"{offset_ref}_positive",
            expression=f"{offset_ref} > 0",
            passed=offset_value > 0,
            value=offset_value,
        ),
        validation_result(
            name="solve_value_nonnegative",
            expression=f"{event_name(solve_object)}.{solve_property} >= 0",
            passed=solved_value >= -1e-12,
            value=solved_value,
            minimum=0.0,
        ),
    ]

    solve_raster = _solve_property_raster(
        seq=seq,
        solve_object=solve_object,
        solve_property=solve_property,
    )
    if solve_raster is not None:
        validation.append(
            validation_result(
                name="solve_value_on_raster",
                expression=f"{event_name(solve_object)}.{solve_property} is on raster",
                passed=_is_on_raster(solved_value, solve_raster),
                value=solved_value,
                minimum=0.0,
            )
        )

    if solved_value < -1e-12:
        raise ValueError(
            f"Relationship {rel.name!r} gives a negative solve value. "
            f"Computed {event_name(solve_object)}.{solve_property}="
            f"{solved_value:.9g} s after raster snapping "
            f"(raw={raw_solved_value:.9g} s).\n"
            f"old_value={old_solve_value:.9g}, "
            f"timing_error={timing_error:.9g}, "
            f"sensitivity={sensitivity:.9g}, "
            f"reference_anchor_global={reference_anchor_global:.9g}, "
            f"target_anchor_global_before={target_anchor_global_before:.9g}, "
            f"desired_target_anchor_global={desired_target_anchor_global:.9g}."
        )

    solved_value = max(float(solved_value), 0.0)

    # Apply the solved value to the representative occurrence and to any
    # copied/frozen occurrences that share the same original source object. This
    # makes repeated-kernel relationships idempotent and keeps .seq export from
    # re-solving a stale copy on a second pass.
    equivalent_solve_events = layout_before.equivalent_events(solve_object)
    for equivalent_event in equivalent_solve_events:
        _set_solve_property_value(equivalent_event, solve_property, solved_value)
    _relationship_debug(
        f"applied {rel.name}: solved={solved_value:.9g} to "
        f"{len(equivalent_solve_events)} equivalent solve event(s)"
    )

    # Recompute the whole layout after changing the solve variable. This is
    # important when solve_property is a delay-event duration because all later
    # blocks shift in time.
    layout_after = _sequence_timing_layout(seq)

    reference_block_after = layout_after.event_to_block.get(id(reference))
    target_block_after = layout_after.event_to_block.get(id(target))
    solve_block_after = layout_after.event_to_block.get(id(solve_object))

    reference_block_start_after = layout_after.block_starts.get(id(reference_block_after), 0.0)
    target_block_start_after = layout_after.block_starts.get(id(target_block_after), 0.0)
    solve_block_start_after = layout_after.block_starts.get(id(solve_block_after), 0.0)

    reference_anchor_local_after = _event_anchor_time_local(
        reference,
        reference_anchor,
        include_delay=True,
    )

    target_anchor_local_after = _event_anchor_time_local(
        target,
        target_anchor,
        include_delay=True,
    )

    reference_anchor_global_after = (
        reference_block_start_after + reference_anchor_local_after
    )
    target_anchor_global_after = target_block_start_after + target_anchor_local_after

    final_error = (
        target_anchor_global_after
        - reference_anchor_global_after
        - offset_value
    )

    _relationship_debug(
        f"final {rel.name}: target_anchor_after={target_anchor_global_after:.9g} "
        f"reference_anchor_after={reference_anchor_global_after:.9g} offset={offset_value:.9g} "
        f"final_error={final_error:.9g} solved={solved_value:.9g}"
    )

    anchor_tolerance = _relationship_final_error_tolerance(
        seq=seq,
        solve_object=solve_object,
        solve_property=solve_property,
    )
    validation.append(
        validation_result(
            name="anchor_relationship_satisfied",
            expression=(
                f"{event_name(target)}.{target_anchor} = "
                f"{event_name(reference)}.{reference_anchor} + {offset_ref}"
            ),
            passed=abs(final_error) <= anchor_tolerance,
            value=final_error,
            minimum=-anchor_tolerance,
        )
    )

    # Only enforce ADC dead time when the solved property is actually the ADC
    # local delay. In GRE, the solved variable may be a TE delay-event duration,
    # while the ADC local delay is independently set by make_adc(..., delay=...).
    if _is_adc_like(target):
        target_delay_after = _event_delay(target)
        validation.append(
            validation_result(
                name="adc_delay_respects_adc_dead_time",
                expression=f"{event_name(target)}.delay >= system.adc_dead_time",
                passed=target_delay_after >= adc_dead_time - 1e-12,
                value=target_delay_after,
                minimum=adc_dead_time,
            )
        )

        if target_delay_after < adc_dead_time - 1e-12:
            raise ValueError(
                f"Relationship {rel.name!r} leaves ADC delay shorter than "
                f"adc_dead_time. ADC delay={target_delay_after:.9g} s, "
                f"adc_dead_time={adc_dead_time:.9g} s. "
                "This should be fixed by the ADC constructor or by solving "
                "the ADC local delay explicitly."
            )

    target_duration = _event_active_duration(target)
    target_center_pos = _event_center_pos(target, default=0.5)

    resolved = {
        "offset": offset_value,
        "reference_block_start": reference_block_start_after,
        "target_block_start": target_block_start_after,
        "solve_block_start": solve_block_start_after,
        "reference_anchor_local": reference_anchor_local_after,
        "reference_anchor_global": reference_anchor_global_after,
        "target_anchor_local": target_anchor_local_after,
        "target_anchor_global": target_anchor_global_after,
        "target_duration": target_duration,
        "target_center_pos": target_center_pos,
        "solve_object": event_name(solve_object),
        "solve_property": solve_property,
        "old_solve_value": old_solve_value,
        "raw_solved_value": raw_solved_value,
        "solved_value": solved_value,
        "solve_raster": solve_raster,
        "raster_snap_delta": solved_value - raw_solved_value,
        "timing_error_before": timing_error,
        "sensitivity": sensitivity,
        "solve_occurrences": solve_occurrences,
        "target_occurrences": target_occurrences,
        "reference_occurrences": reference_occurrences,
        "anchor_tolerance": anchor_tolerance,
        "final_error": final_error,
        "target_value": solved_value,
        "coordinate_model": "global_anchor_with_explicit_solve_variable",
    }

    if str(offset_ref).upper() == "TE":
        resolved.update(
            {
                "TE": offset_value,
                "rf_center": reference_anchor_global_after,
                "adc_duration": target_duration,
                "adc_delay": _event_delay(target),
                "adc_center": target_anchor_global_after,
                "te_solve_object": event_name(solve_object),
                "te_solve_property": solve_property,
                "te_solve_value": solved_value,
            }
        )

    rel.update_resolution(resolved=resolved, validation=validation)

    store_derived_parameter(
        seq,
        f"{event_name(solve_object)}.{solve_property}",
        solved_value,
    )
    store_derived_parameter(
        seq,
        f"{event_name(target)}.{target_anchor}",
        target_anchor_global_after,
    )



def _resolve_require_same_anchor(seq: Any, rel: SeqStarRelationship) -> None:
    """Validate target.anchor == reference.anchor + offset without mutation."""

    target = _live_object_from_relationship(rel, "target")
    reference = _live_object_from_relationship(rel, "reference")

    if target is None or reference is None:
        raise RuntimeError(
            f"Relationship {rel.name!r} lost access to target/reference objects."
        )

    offset_ref = rel.metadata.get("offset", 0.0)
    offset_value = protocol_float(seq, offset_ref, name=str(offset_ref))

    target_anchor = str(rel.metadata.get("target_anchor", "center"))
    reference_anchor = str(rel.metadata.get("reference_anchor", "center"))

    layout = _sequence_timing_layout(seq)

    # Resolve relationship endpoints to concrete timeline occurrences. This is
    # important when add_block() has copied/frozen an event after the
    # relationship was declared against the original object.
    target = layout.resolve_event(target)
    reference = layout.resolve_event(reference)

    reference_block = layout.event_to_block.get(id(reference))
    target_block = layout.event_to_block.get(id(target))

    reference_block_start = layout.block_starts.get(id(reference_block), 0.0)
    target_block_start = layout.block_starts.get(id(target_block), 0.0)

    reference_anchor_local = _event_anchor_time_local(
        reference,
        reference_anchor,
        include_delay=True,
    )
    target_anchor_local = _event_anchor_time_local(
        target,
        target_anchor,
        include_delay=True,
    )

    reference_anchor_global = reference_block_start + reference_anchor_local
    target_anchor_global = target_block_start + target_anchor_local
    delta = target_anchor_global - reference_anchor_global - offset_value

    tolerance = rel.metadata.get("tolerance")
    if tolerance is None:
        tolerance = _validation_only_anchor_tolerance(seq=seq, target=target, reference=reference)
    else:
        tolerance = float(tolerance)

    target_occurrences = layout.occurrence_count(target)
    reference_occurrences = layout.occurrence_count(reference)

    validation: list[SeqStarValidationResult] = [
        validation_result(
            name="anchor_relationship_satisfied",
            expression=(
                f"{event_name(target)}.{target_anchor} = "
                f"{event_name(reference)}.{reference_anchor} + {offset_ref}"
            ),
            passed=abs(delta) <= tolerance,
            value=delta,
            minimum=-tolerance,
        )
    ]

    if isinstance(offset_ref, str):
        validation.insert(
            0,
            validation_result(
                name=f"{offset_ref}_finite",
                expression=f"{offset_ref} is finite",
                passed=math.isfinite(offset_value),
                value=offset_value,
            ),
        )

    resolved = {
        "offset": offset_value,
        "reference_block_start": reference_block_start,
        "target_block_start": target_block_start,
        "reference_anchor_local": reference_anchor_local,
        "reference_anchor_global": reference_anchor_global,
        "target_anchor_local": target_anchor_local,
        "target_anchor_global": target_anchor_global,
        "target_duration": _event_active_duration(target),
        "reference_duration": _event_active_duration(reference),
        "target_center_pos": _event_center_pos(target, default=0.5),
        "reference_center_pos": _event_center_pos(reference, default=0.5),
        "target_occurrences": target_occurrences,
        "reference_occurrences": reference_occurrences,
        "anchor_delta": delta,
        "anchor_tolerance": tolerance,
        "target_value": target_anchor_global,
        "coordinate_model": "global_anchor_validation_only",
        "validation_only": True,
    }

    rel.update_resolution(resolved=resolved, validation=validation)

    store_derived_parameter(
        seq,
        f"{event_name(target)}.{target_anchor}",
        target_anchor_global,
    )
    store_derived_parameter(
        seq,
        f"{event_name(reference)}.{reference_anchor}",
        reference_anchor_global,
    )
    store_derived_parameter(
        seq,
        f"{event_name(target)}.{target_anchor}_minus_{event_name(reference)}.{reference_anchor}",
        delta,
    )


def _validation_only_anchor_tolerance(*, seq: Any, target: Any, reference: Any) -> float:
    """Tolerance for non-mutating anchor validation relationships."""

    system = getattr(seq, "system", None)
    raster = _first_positive_float(
        getattr(system, "grad_raster_time", None),
        getattr(system, "block_duration_raster", None),
        getattr(system, "adc_raster_time", None),
        10e-6,
    )
    return max(1e-9, float(raster or 10e-6) + 1e-12)

def _guard_ambiguous_shared_solve_object(
    *,
    seq: Any,
    rel: SeqStarRelationship,
    solve_object: Any,
    solve_property: str,
    sensitivity: float,
    solve_occurrences: int,
    target_occurrences: int,
    reference_occurrences: int,
) -> None:
    """Fail loudly when a solve object is reused in a way that corrupts solving.

    A structural solve variable such as the duration of a delay event should
    usually affect the selected target occurrence with sensitivity ~1. If the
    same mutable event object appears in many repeated blocks, finite-difference
    probing changes every occurrence at once. The result is a bogus sensitivity
    such as 64 or 128 and a bad solve value. This guard catches that library
    state before it silently writes nonsense back to the event.
    """

    if solve_occurrences <= 1:
        return

    # Some repeated relationships intentionally use the same object value
    # everywhere. That is safe only if the selected target anchor still moves
    # locally with near-unit sensitivity. Large sensitivity means global
    # repeated-object coupling, not a valid local degree of freedom.
    if abs(float(sensitivity)) <= 2.0:
        return

    if not _is_structural_solve_property(solve_object, solve_property):
        return

    raise ValueError(
        f"Relationship {rel.name!r} is trying to solve "
        f"{event_name(solve_object)}.{solve_property}, but that same mutable "
        f"solve object appears {solve_occurrences} times in the sequence and "
        f"the measured sensitivity is {sensitivity:.9g} instead of ~1.\n"
        "This usually means a repeated PyPulseq-style sequence reused one "
        "delay/wait event object across many blocks. The resolver cannot know "
        "which occurrence is intended, so it refuses to write a corrupted solve "
        "value. Library-level fixes are: freeze/copy events on seq.add_block(), "
        "or create occurrence-specific solve events before resolving.\n"
        f"target_occurrences={target_occurrences}, "
        f"reference_occurrences={reference_occurrences}."
    )


def _is_structural_solve_property(obj: Any, property_name: str) -> bool:
    """Return True when a solve value can alter sequence/block duration."""

    property_name = str(property_name)
    if property_name in {"duration", "wait", "wait_time", "delay_duration"}:
        return True
    if property_name in {"delay", "tstart", "start", "start_s"}:
        # ADC/RF local delay may be a legitimate local solve variable. Delay
        # objects, however, often use delay/duration as the structural wait.
        return _is_delay_like(obj)
    return False


def _solve_property_raster(
    *,
    seq: Any,
    solve_object: Any,
    solve_property: str,
) -> float | None:
    """Return the timing raster that applies to a solve property."""

    system = getattr(seq, "system", None)
    solve_property = str(solve_property)

    if solve_property in {"duration", "wait", "wait_time", "delay_duration"}:
        if _is_adc_like(solve_object):
            return _first_positive_float(
                getattr(system, "adc_raster_time", None),
                getattr(system, "adc_sample_raster_time", None),
                getattr(system, "grad_raster_time", None),
                1e-6,
            )
        return _first_positive_float(
            getattr(system, "grad_raster_time", None),
            getattr(system, "block_duration_raster", None),
            10e-6,
        )

    if solve_property in {"delay", "tstart", "start", "start_s"}:
        if _is_adc_like(solve_object):
            return _first_positive_float(
                getattr(system, "adc_raster_time", None),
                getattr(system, "adc_sample_raster_time", None),
                getattr(system, "grad_raster_time", None),
                1e-6,
            )
        return _first_positive_float(
            getattr(system, "grad_raster_time", None),
            getattr(system, "block_duration_raster", None),
            10e-6,
        )

    return None


def _snap_solve_value_to_raster(
    *,
    seq: Any,
    solve_object: Any,
    solve_property: str,
    value: float,
    mode: str = "ceil",
) -> float:
    """Snap a relationship solve value to the appropriate event raster."""

    value = float(value)
    raster = _solve_property_raster(
        seq=seq,
        solve_object=solve_object,
        solve_property=solve_property,
    )
    if raster is None or raster <= 0:
        return value

    eps = min(1e-12, raster * 1e-6)
    if mode == "floor":
        return math.floor((value + eps) / raster) * raster
    if mode == "round":
        return round(value / raster) * raster
    return math.ceil((value - eps) / raster) * raster


def _relationship_final_error_tolerance(
    *,
    seq: Any,
    solve_object: Any,
    solve_property: str,
) -> float:
    """Tolerance for validating a snapped timing relationship."""

    raster = _solve_property_raster(
        seq=seq,
        solve_object=solve_object,
        solve_property=solve_property,
    )
    if raster is None or raster <= 0:
        return 1e-9
    return max(1e-9, float(raster) + 1e-12)


def _is_on_raster(value: float, raster: float) -> bool:
    if raster <= 0:
        return True
    nearest = round(float(value) / float(raster)) * float(raster)
    return abs(float(value) - nearest) <= max(1e-12, float(raster) * 1e-9)


def _first_positive_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            return numeric
    return None


def _resolve_block_after(seq: Any, rel: SeqStarRelationship) -> None:
    """Validate target_block.start == reference_block.end."""

    target_block = _live_block_from_relationship(rel, "target")
    reference_block = _live_block_from_relationship(rel, "reference")

    if target_block is None or reference_block is None:
        raise RuntimeError(
            f"Relationship {rel.name!r} lost access to target/reference blocks."
        )

    layout = _sequence_timing_layout(seq)

    reference_events = list(_iter_events(reference_block))
    target_events = list(_iter_events(target_block))

    reference_block_start = layout.block_starts.get(id(reference_block), 0.0)
    target_block_start = layout.block_starts.get(id(target_block), 0.0)
    reference_block_duration = _block_duration(seq, reference_block, reference_events)
    target_block_duration = _block_duration(seq, target_block, target_events)
    reference_block_end = reference_block_start + reference_block_duration
    target_block_end = target_block_start + target_block_duration

    delta = target_block_start - reference_block_end
    tolerance = _block_order_tolerance(seq)

    target_name = _block_display_name(target_block)
    reference_name = _block_display_name(reference_block)
    target_node = _block_node(target_block)
    reference_node = _block_node(reference_block)

    validation: list[SeqStarValidationResult] = [
        validation_result(
            name="block_order_satisfied",
            expression=f"{target_name}.start = {reference_name}.end",
            passed=abs(delta) <= tolerance,
            value=delta,
            minimum=-tolerance,
            maximum=tolerance,
            metadata={
                "target_block_node": target_node,
                "reference_block_node": reference_node,
            },
        ),
        validation_result(
            name="reference_block_duration_nonnegative",
            expression=f"{reference_name}.duration >= 0",
            passed=reference_block_duration >= -1e-12,
            value=reference_block_duration,
            minimum=0.0,
        ),
        validation_result(
            name="target_block_duration_nonnegative",
            expression=f"{target_name}.duration >= 0",
            passed=target_block_duration >= -1e-12,
            value=target_block_duration,
            minimum=0.0,
        ),
    ]

    resolved = {
        "reference_block": reference_name,
        "target_block": target_name,
        "reference_block_node": reference_node,
        "target_block_node": target_node,
        "parent_node": rel.metadata.get("parent_node"),
        "reference_block_start": reference_block_start,
        "reference_block_duration": reference_block_duration,
        "reference_block_end": reference_block_end,
        "target_block_start": target_block_start,
        "target_block_duration": target_block_duration,
        "target_block_end": target_block_end,
        "block_start_delta": delta,
        "anchor_delta": delta,
        "block_order_tolerance": tolerance,
        "implicit": bool(rel.metadata.get("implicit", False)),
        "source": rel.metadata.get("source", rel.metadata.get("relationship_source")),
        "coordinate_model": "block_order_global_timeline",
        "validation_only": True,
        "target_value": target_block_start,
    }

    rel.update_resolution(resolved=resolved, validation=validation)

    store_derived_parameter(seq, f"{reference_name}.end", reference_block_end)
    store_derived_parameter(seq, f"{target_name}.start", target_block_start)
    store_derived_parameter(
        seq,
        f"{target_name}.start_minus_{reference_name}.end",
        delta,
    )

    _relationship_debug(
        f"block_after {rel.name}: reference={reference_node!r} end={reference_block_end:.9g} "
        f"target={target_node!r} start={target_block_start:.9g} delta={delta:.9g}"
    )


def _resolve_fill_to_period(seq: Any, rel: SeqStarRelationship) -> None:
    """Resolve a generic ordered block group to a requested period."""
    blocks = [obj for obj in rel.metadata.get("block_objects", []) if obj is not None]
    fill_event = rel.metadata.get("fill_object") or _object_from_id(rel.metadata.get("fill_object_id"))
    if not blocks or fill_event is None:
        raise RuntimeError(f"Relationship {rel.name!r} lost its blocks or fill event.")

    period_ref = rel.metadata["period"]
    period_value = protocol_float(seq, period_ref, name=str(period_ref))
    fill_property = str(rel.metadata.get("fill_property", "duration"))
    if fill_property in {"wait", "wait_time", "delay_duration"}:
        fill_property = "duration"

    # Resolve copied/frozen fill occurrence when needed.
    layout = _sequence_timing_layout(seq)
    fill_event = layout.resolve_event(fill_event)
    fill_block = layout.event_to_block.get(id(fill_event))
    if fill_block is None:
        raise ValueError(f"Fill event {event_name(fill_event)!r} is not in the timeline.")
    try:
        fill_index = blocks.index(fill_block)
    except ValueError as exc:
        raise ValueError("fill_event must belong to one of the supplied blocks.") from exc

    duration_before = 0.0
    for block in blocks[:fill_index]:
        events = list(_iter_events(block))
        duration_before += _block_duration(seq, block, events)

    fill_block_events = list(_iter_events(fill_block))
    simultaneous_extent = max(
        (_event_delay(event) + _event_active_duration(event)
         for event in fill_block_events if id(event) != id(fill_event)),
        default=0.0,
    )
    raw_fill = period_value - duration_before
    solved_fill = _snap_solve_value_to_raster(
        seq=seq,
        solve_object=fill_event,
        solve_property=fill_property,
        value=raw_fill,
        mode="ceil",
    )
    tolerance = _relationship_final_error_tolerance(
        seq=seq, solve_object=fill_event, solve_property=fill_property
    )
    if solved_fill < simultaneous_extent - tolerance:
        raise ValueError(
            f"Relationship {rel.name!r} cannot fit the simultaneous events in the fill block. "
            f"required_fill={solved_fill:.9g} s, simultaneous_extent={simultaneous_extent:.9g} s."
        )
    if solved_fill < -tolerance:
        raise ValueError(
            f"Relationship {rel.name!r} gives a negative fill: period={period_value:.9g} s, "
            f"duration_before_fill={duration_before:.9g} s."
        )
    solved_fill=max(0.0,float(solved_fill))
    for equivalent in layout.equivalent_events(fill_event):
        _set_solve_property_value(equivalent, fill_property, solved_fill)

    realized = duration_before + max(solved_fill, simultaneous_extent)
    error = realized - period_value
    validation = [
        validation_result(name="period_positive", expression="period > 0", passed=period_value > 0, value=period_value),
        validation_result(name="fill_nonnegative", expression="fill >= 0", passed=solved_fill >= 0, value=solved_fill, minimum=0.0),
        validation_result(name="fill_contains_simultaneous_events", expression="fill >= simultaneous_extent", passed=solved_fill + tolerance >= simultaneous_extent, value=solved_fill, minimum=simultaneous_extent),
        validation_result(name="period_satisfied", expression="group.duration = period", passed=abs(error) <= tolerance, value=error, minimum=-tolerance, maximum=tolerance),
    ]
    rel.update_resolution(
        resolved={
            "period": period_value,
            "duration_before_fill": duration_before,
            "simultaneous_extent": simultaneous_extent,
            "fill_property": fill_property,
            "fill_value": solved_fill,
            "realized_duration": realized,
            "final_error": error,
            "target_value": solved_fill,
        },
        validation=validation,
    )
    store_derived_parameter(seq, f"{event_name(fill_event)}.{fill_property}", solved_fill)
    store_derived_parameter(seq, "group.duration", realized)


def _resolve_repeat_every(seq: Any, rel: SeqStarRelationship) -> None:
    events = [_object_from_id(obj_id) for obj_id in rel.metadata.get("event_object_ids", [])]
    events = [event for event in events if event is not None]

    if not events:
        live_events = rel.metadata.get("event_objects", []) or []
        events = [event for event in live_events if event is not None]

    if not events:
        raise RuntimeError(f"Relationship {rel.name!r} has no events to resolve.")

    period_ref = rel.metadata["period"]
    period_value = protocol_float(seq, period_ref, name=str(period_ref))

    # Keep repeat_every intentionally kernel-local for Phase 1. If events are
    # reused across many blocks, a global-span computation would measure the
    # entire sequence, not one kernel. The local max end mirrors the previous
    # behavior and remains useful for feasibility/debug metadata.
    ends = [_event_delay(event) + _event_active_duration(event) for event in events]
    kernel_duration = max(ends) if ends else 0.0
    fill_delay = period_value - kernel_duration

    validation = [
        validation_result(
            name=f"{period_ref}_positive",
            expression=f"{period_ref} > 0",
            passed=period_value > 0,
            value=period_value,
        ),
        validation_result(
            name="period_exceeds_kernel_duration",
            expression="period >= kernel.duration",
            passed=period_value >= kernel_duration,
            value=period_value,
            minimum=kernel_duration,
        ),
    ]

    if fill_delay < 0:
        raise ValueError(
            f"Relationship {rel.name!r} gives a negative fill delay. "
            f"period={period_value:.9g} s, kernel_duration={kernel_duration:.9g} s."
        )

    resolved = {
        "period": period_value,
        "kernel_duration": kernel_duration,
        "fill_delay": fill_delay,
        "tr_fill_delay": fill_delay,
        "target_value": period_value,
    }

    if str(period_ref).upper() == "TR":
        resolved["TR"] = period_value

    rel.update_resolution(resolved=resolved, validation=validation)
    store_derived_parameter(seq, "kernel.duration", kernel_duration)
    store_derived_parameter(seq, "tr_fill_delay", fill_delay)


class _SequenceTimingLayout:
    def __init__(
        self,
        *,
        block_starts: dict[int, float],
        event_to_block: dict[int, Any],
        event_occurrences: dict[int, list[Any]] | None = None,
        event_aliases: dict[int, Any] | None = None,
        event_alias_groups: dict[int, list[Any]] | None = None,
    ) -> None:
        self.block_starts = block_starts
        self.event_to_block = event_to_block
        self.event_occurrences = event_occurrences or {}
        self.event_aliases = event_aliases or {}
        self.event_alias_groups = event_alias_groups or {}

    def resolve_event(self, event: Any) -> Any:
        """Return the concrete timeline occurrence for a relationship object.

        Relationship APIs often store an original/prototype event object, while
        the sequence timeline may contain copied/frozen occurrences that carry
        the prototype object id in provenance metadata. If an object id is both
        an alias key and an event_to_block key, prefer the concrete aliased
        timeline event. This avoids solving a stale prototype object that does
        not affect the executable timeline.
        """

        if event is None:
            return event
        event_id = id(event)
        if event_id in self.event_aliases:
            return self.event_aliases[event_id]
        if event_id in self.event_to_block:
            return event
        return event

    def occurrence_count(self, event: Any) -> int:
        if event is None:
            return 0
        event_id = id(event)
        if event_id in self.event_alias_groups:
            return len(self.event_alias_groups[event_id])
        if event_id in self.event_occurrences:
            return len(self.event_occurrences[event_id])
        concrete = self.event_aliases.get(event_id)
        if concrete is not None:
            return len(self.event_occurrences.get(id(concrete), []))
        return 0

    def equivalent_events(self, event: Any) -> list[Any]:
        """Return all concrete events sharing the same prototype/source id.

        This is used after a solve to propagate the solved value across repeated
        copied/frozen kernel occurrences. The previous one-to-one alias mapping
        only updated the first concrete occurrence; repeated GRE needs all 64
        equivalent TE-delay occurrences to receive the solved wait length.
        """

        if event is None:
            return []

        equivalent: list[Any] = []
        seen: set[int] = set()

        def add(candidate: Any) -> None:
            if candidate is None:
                return
            candidate_id = id(candidate)
            if candidate_id not in seen:
                equivalent.append(candidate)
                seen.add(candidate_id)

        event_id = id(event)

        # Original/prototype object id: include every concrete timeline event
        # carrying this alias, not only the first one.
        for candidate in self.event_alias_groups.get(event_id, []):
            add(candidate)

        # Concrete copied event: include other events sharing any source alias.
        aliases = set(_event_alias_ids(event))
        for alias_id in aliases:
            for candidate in self.event_alias_groups.get(alias_id, []):
                add(candidate)

        # Direct timeline object fallback.
        if event_id in self.event_to_block:
            add(event)

        # Alias-first concrete fallback.
        add(self.event_aliases.get(event_id))

        return equivalent or [event]


def _sequence_timing_layout(seq: Any) -> _SequenceTimingLayout:
    """Return sequential block starts and event-parent block mapping."""

    block_starts: dict[int, float] = {}
    event_to_block: dict[int, Any] = {}
    event_occurrences: dict[int, list[Any]] = {}
    event_aliases: dict[int, Any] = {}
    event_alias_groups: dict[int, list[Any]] = {}
    current_time = 0.0

    for block in _iter_blocks(seq):
        explicit_start = _get_explicit_block_tstart(block)
        block_start = current_time if explicit_start is None else explicit_start
        block_starts[id(block)] = block_start

        events = list(_iter_events(block))
        for event in events:
            # If an event object is reused across repeated kernels, the exact
            # occurrence is ambiguous. Storing the latest occurrence is still
            # valid for a steady-state kernel because local timing is identical
            # in each occurrence, and it avoids stale pre-construction mappings.
            # Use the first occurrence as the representative timing occurrence.
            # Repeated PyPulseq-style kernels often reuse the same event object or
            # copied occurrences carrying a source-object alias. Relationship APIs
            # are declared against the prototype event, so solving should be based
            # on the first kernel occurrence, not whichever repeated block happened
            # to be visited last. All occurrences are still counted for diagnostics.
            event_to_block.setdefault(id(event), block)
            event_occurrences.setdefault(id(event), []).append(block)

            # Also map original/source object ids to this concrete occurrence.
            # This lets relationships declared against pre-freeze objects work
            # after sequence.add_block() has copied/frozen events.
            for alias_id in _event_alias_ids(event):
                event_to_block.setdefault(alias_id, block)
                event_occurrences.setdefault(alias_id, []).append(block)
                event_aliases.setdefault(alias_id, event)
                event_alias_groups.setdefault(alias_id, []).append(event)

        block_duration = _block_duration(seq, block, events)
        current_time = max(current_time, block_start + block_duration)

    return _SequenceTimingLayout(
        block_starts=block_starts,
        event_to_block=event_to_block,
        event_occurrences=event_occurrences,
        event_aliases=event_aliases,
        event_alias_groups=event_alias_groups,
    )


def _event_alias_ids(event: Any) -> list[int]:
    """Return original/source object ids associated with a concrete event.

    SeqStarSequence may copy/freeze event objects on insertion and preserve
    provenance in parameters/metadata. Relationship records usually hold the
    original object id. Exposing these aliases lets the timing resolver map the
    relationship endpoint to the concrete timeline event.
    """

    out: list[int] = []
    for container in (getattr(event, "parameters", None), getattr(event, "metadata", None)):
        if not isinstance(container, Mapping):
            continue
        for key in (
            "source_event_object_id",
            "source_object_id",
            "original_event_object_id",
            "original_object_id",
            "seqstar_source_object_id",
        ):
            value = container.get(key)
            if isinstance(value, int):
                out.append(value)
    # Preserve order while removing duplicates and the event's own id.
    seen: set[int] = {id(event)}
    unique: list[int] = []
    for value in out:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _block_duration(seq: Any, block: Any, events: list[Any]) -> float:
    """Return block duration for relationship solving.

    Do not delegate to ``seq.calc_duration`` here. Relationship solving must use
    a local, self-consistent timing model. In particular, delay/wait events are
    structural wait blocks whose extent is their wait length, not
    ``delay + duration``. Some sequence-level compatibility helpers and event
    property setters deliberately mirror PyPulseq fields and can double-count
    delay-like events during finite-difference probes.
    """

    if events:
        return max(_event_delay(event) + _event_active_duration(event) for event in events)

    return 0.0

def _event_anchor_time_local(
    event: Any,
    anchor: str,
    *,
    include_delay: bool,
) -> float:
    """Return an event anchor in the event's parent-block local frame."""

    delay = _event_delay(event) if include_delay else 0.0
    duration = _event_active_duration(event)
    anchor_normalized = str(anchor).lower()

    if anchor_normalized in {"start", "begin", "left", "tstart"}:
        return delay

    if anchor_normalized in {"center", "centre", "middle", "tcenter", "adc_center", "rf_center"}:
        return delay + _event_center_pos(event, default=0.5) * duration

    if anchor_normalized in {"end", "right", "stop", "tend"}:
        return delay + duration

    raise ValueError(f"Unsupported timing anchor: {anchor!r}")


def get_anchor_time(
    event: Any,
    anchor: str = "center",
    *,
    seq: Any | None = None,
    frame: str = "local",
    include_delay: bool = True,
) -> float:
    """Return an event anchor time in local or global sequence coordinates.

    Parameters
    ----------
    event
        SeqStar event or an original/prototype event used to construct a
        timeline occurrence.

    anchor
        Anchor name. Supported aliases are the same as
        ``_event_anchor_time_local``:

        - start: ``start``, ``begin``, ``left``, ``tstart``
        - center: ``center``, ``centre``, ``middle``, ``tcenter``,
          ``adc_center``, ``rf_center``
        - end: ``end``, ``right``, ``stop``, ``tend``

    seq
        Sequence containing the event. Required when ``frame="global"``.
        When supplied, prototype events are resolved to their concrete
        timeline occurrence using the same alias machinery as the
        relationship resolver.

    frame
        ``"local"`` returns time relative to the event's parent block.
        ``"global"`` returns time relative to sequence start.

    include_delay
        Include the event-local delay/tstart. This should normally remain
        True. Set False only when the anchor within the active waveform is
        needed independently of its local event delay.

    Returns
    -------
    float
        Anchor time in seconds.

    Notes
    -----
    For delay/wait events, ``_event_delay`` returns zero because their wait
    length is represented by ``duration``. This avoids counting a structural
    delay twice.

    Examples
    --------
    Before adding an RF event to a sequence, obtain its local center:

        rf_center_local = get_anchor_time(rf, "center")

    After timeline construction, obtain the global ADC start:

        adc_start_global = get_anchor_time(
            adc,
            "start",
            seq=seq,
            frame="global",
        )
    """

    normalized_frame = str(frame).strip().lower()

    if normalized_frame not in {"local", "global"}:
        raise ValueError(
            f"Unsupported anchor frame {frame!r}; use 'local' or 'global'."
        )

    resolved_event = event
    layout = None

    if seq is not None:
        layout = _sequence_timing_layout(seq)
        resolved_event = layout.resolve_event(event)

    local_time = _event_anchor_time_local(
        resolved_event,
        anchor,
        include_delay=include_delay,
    )

    if normalized_frame == "local":
        return float(local_time)

    if seq is None or layout is None:
        raise ValueError(
            "get_anchor_time(..., frame='global') requires seq=<sequence>."
        )

    parent_block = layout.event_to_block.get(id(resolved_event))

    if parent_block is None:
        # The caller may have supplied a prototype id that maps directly to
        # a block but was not replaced by resolve_event for an unusual event
        # container. Try the original object id before failing.
        parent_block = layout.event_to_block.get(id(event))

    if parent_block is None:
        raise ValueError(
            f"Event {event_name(event)!r} is not present in the sequence "
            "timeline, so its global anchor cannot be determined."
        )

    block_start = layout.block_starts.get(id(parent_block))

    if block_start is None:
        raise ValueError(
            f"Parent block for event {event_name(event)!r} has no resolved "
            "global start time."
        )

    return float(block_start + local_time)


def _event_center_pos(event: Any, *, default: float = 0.5) -> float:
    """Return fractional event center position."""

    try:
        from .context import event_center_pos

        return float(event_center_pos(event, default=default))
    except Exception:
        pass

    for attr in ("center_pos", "asymmetry"):
        value = getattr(event, attr, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in ("center_pos", "asymmetry"):
            value = parameters.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

    return float(default)


def _event_delay(event: Any) -> float:
    """Return event local delay/tstart in seconds.

    Delay/wait events are structural blocks: their wait length belongs to
    duration, while their local offset within the block is zero. This avoids
    double counting in the relationship layout even when the concrete delay
    event object mirrors ``delay`` and ``duration`` for PyPulseq compatibility.
    """

    if _is_delay_like(event):
        return 0.0

    try:
        from .context import event_delay

        return float(event_delay(event))
    except Exception:
        pass

    for attr in ("delay", "tstart", "start", "start_s"):
        value = getattr(event, attr, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    timing = getattr(event, "timing", None)
    if timing is not None:
        value = getattr(timing, "tstart", None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in ("delay", "tstart", "start", "start_s"):
            value = parameters.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

    return 0.0

def _event_active_duration(event: Any) -> float:
    """Return active duration using explicit event-family classification.

    Shared protocol dictionaries may contain parameters for several event
    families. Family-specific values such as ``num_samples`` and ``dwell`` must
    therefore be consulted only after the concrete event has been classified
    as ADC-like. Likewise, gradient ramp fields must be used only for gradients.

    This function is sequence-family agnostic. It knows RF, ADC, gradient,
    delay/wait, and generic event timing, but it does not know FID, GRE, EPI,
    RARE, spectroscopy, or any user-defined node name.
    """

    if _is_delay_like(event):
        # A structural delay/wait has one occupied interval. Some compatibility
        # objects mirror that interval through both delay and duration; do not
        # add the aliases together.
        for source in (
            event,
            getattr(event, "timing", None),
            getattr(event, "parameters", None),
        ):
            if source is None:
                continue

            for key in ("duration", "wait_time", "delay"):
                if isinstance(source, Mapping):
                    value = source.get(key)
                else:
                    value = getattr(source, key, None)

                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue

        return 0.0

    if _is_adc_like(event):
        num_samples = getattr(event, "num_samples", None)
        dwell = getattr(event, "dwell", None)

        parameters = getattr(event, "parameters", None)
        if isinstance(parameters, Mapping):
            if num_samples is None:
                num_samples = parameters.get("num_samples")
            if dwell is None:
                dwell = parameters.get("dwell")

        timing = getattr(event, "timing", None)
        if timing is not None:
            if num_samples is None:
                num_samples = getattr(timing, "num_samples", None)
            if dwell is None:
                dwell = getattr(timing, "dwell", None)

        if num_samples is not None and dwell is not None:
            return float(num_samples) * float(dwell)

        for source in (event, timing, parameters):
            if source is None:
                continue
            for key in ("duration", "adc_duration", "readout_duration"):
                value = (
                    source.get(key)
                    if isinstance(source, Mapping)
                    else getattr(source, key, None)
                )
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue

        return 0.0

    if _is_gradient_like_for_timing(event):
        rise_time = getattr(event, "rise_time", None)
        flat_time = getattr(event, "flat_time", None)
        fall_time = getattr(event, "fall_time", None)

        parameters = getattr(event, "parameters", None)
        if isinstance(parameters, Mapping):
            if rise_time is None:
                rise_time = parameters.get("rise_time")
            if flat_time is None:
                flat_time = parameters.get("flat_time")
            if fall_time is None:
                fall_time = parameters.get("fall_time")

        if (
            rise_time is not None
            and flat_time is not None
            and fall_time is not None
        ):
            return (
                float(rise_time)
                + float(flat_time)
                + float(fall_time)
            )

    # RF and generic events use their own explicit duration/shape. Crucially,
    # this path does not inspect ADC-specific fields from a shared protocol.
    for source in (
        event,
        getattr(event, "timing", None),
    ):
        if source is None:
            continue
        for key in ("duration", "shape_duration", "active_duration"):
            value = getattr(source, key, None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

    shape = getattr(event, "shape", None)
    if shape is not None:
        for key in ("duration", "shape_duration", "active_duration"):
            value = getattr(shape, key, None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

    # Parameter fallback is restricted to generic duration aliases. It must not
    # infer event family from unrelated protocol fields.
    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in ("duration", "shape_duration", "active_duration"):
            value = parameters.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

    # Last resort for custom event classes that provide a safe duration helper.
    try:
        return float(event_duration(event))
    except Exception:
        return 0.0


def _is_gradient_like_for_timing(event: Any) -> bool:
    """Return True when an event is explicitly gradient-like."""

    event_type = str(
        getattr(event, "event_type", getattr(event, "type", "")) or ""
    ).lower()
    kind = str(getattr(event, "kind", "") or "").lower()
    class_name = event.__class__.__name__.lower()

    if event_type in {"grad", "gradient", "trap", "trapezoid"}:
        return True

    if kind in {
        "grad",
        "gradient",
        "trap",
        "trapezoid",
        "arbitrary",
        "arbitrary_grad",
        "arbitrary_gradient",
        "split",
        "split_gradient",
    }:
        return True

    if (
        "gradient" in class_name
        or "trapezoid" in class_name
        or "splitgrad" in class_name
    ):
        return True

    return bool(
        hasattr(event, "channel")
        and (
            hasattr(event, "area")
            or hasattr(event, "amplitude")
            or hasattr(event, "flat_area")
        )
    )


def _set_event_delay_consistently(event: Any, delay: float) -> None:
    """Best-effort synchronization of event delay/tstart fields."""

    delay = float(delay)

    try:
        setattr(event, "delay", delay)
    except Exception:
        pass

    timing = getattr(event, "timing", None)
    if timing is not None:
        try:
            setattr(timing, "tstart", delay)
        except Exception:
            pass

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, dict):
        parameters["delay"] = delay
        parameters["tstart"] = delay


def _get_solve_property_value(obj: Any, property_name: str) -> float:
    """Read the current numeric value of a solve property."""

    property_name = str(property_name)

    if property_name in {"delay", "tstart", "start", "start_s"}:
        return _event_delay(obj)

    if property_name in {"duration", "wait_time", "wait"}:
        return _event_active_duration(obj)

    value = getattr(obj, property_name, None)
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    parameters = getattr(obj, "parameters", None)
    if isinstance(parameters, Mapping):
        value = parameters.get(property_name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, Mapping):
        value = metadata.get(property_name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return 0.0


def _set_solve_property_value(obj: Any, property_name: str, value: float) -> None:
    """Set a solve property consistently on an event-like object."""

    property_name = str(property_name)
    value = float(value)

    if property_name in {"delay", "tstart", "start", "start_s"}:
        set_event_property(obj, "delay", value)
        _set_event_delay_consistently(obj, value)
        return

    if property_name in {"duration", "wait_time", "wait"}:
        _set_event_duration_consistently(obj, value)
        return

    try:
        setattr(obj, property_name, value)
    except Exception:
        pass

    parameters = getattr(obj, "parameters", None)
    if isinstance(parameters, dict):
        parameters[property_name] = value


def _set_event_duration_consistently(event: Any, duration: float) -> None:
    """Best-effort synchronization of event duration fields.

    For delay/wait events, set the wait length but do not forcibly set
    ``delay``/``tstart`` to zero on the event object. Some delay-event classes
    intentionally mirror ``delay`` and ``duration`` through property setters;
    setting ``delay = 0`` after setting ``duration`` can erase the wait length.

    The relationship layout itself treats delay-like events as zero-offset
    structural wait blocks via ``_event_delay() == 0`` and
    ``_event_active_duration() == wait_length``. That gives correct solve
    sensitivity without mutating the event into an inconsistent state.
    """

    duration = float(duration)

    try:
        setattr(event, "duration", duration)
    except Exception:
        pass

    timing = getattr(event, "timing", None)
    if timing is not None:
        try:
            setattr(timing, "duration", duration)
        except Exception:
            pass

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, dict):
        parameters["duration"] = duration
        parameters["wait_time"] = duration

    if _is_delay_like(event):
        # Keep mirrored delay fields consistent with constructors/writers that
        # still expect delay-like events to carry the wait length in ``delay``.
        # Do NOT set tstart to zero here: several delay-event implementations
        # mirror tstart/delay/duration through property setters, and setting
        # tstart after duration can erase the wait length. The relationship
        # layout already treats delay-like events as zero-offset structural
        # wait blocks via _event_delay() == 0.
        try:
            setattr(event, "delay", duration)
        except Exception:
            pass

        if isinstance(parameters, dict):
            parameters["delay"] = duration
            # Preserve an existing tstart if present. Relationship layout ignores
            # local offset for delay-like events, so no tstart mutation is needed.

def _solve_property_sensitivity(
    *,
    seq: Any,
    target: Any,
    target_anchor: str,
    solve_object: Any,
    solve_property: str,
    layout_before: _SequenceTimingLayout,
) -> float:
    """Return d(target_anchor_global) / d(solve_property).

    Uses a small finite-difference probe so the resolver remains generic. This
    supports both local properties like ADC delay and structural properties like
    a delay-event duration that shifts later blocks.
    """

    old_value = _get_solve_property_value(solve_object, solve_property)

    target_block_before = layout_before.event_to_block.get(id(target))
    target_block_start_before = layout_before.block_starts.get(id(target_block_before), 0.0)
    target_anchor_before = target_block_start_before + _event_anchor_time_local(
        target,
        target_anchor,
        include_delay=True,
    )

    requested_probe = max(abs(old_value) * 1e-6, 1e-9)

    _set_solve_property_value(solve_object, solve_property, old_value + requested_probe)

    try:
        # The setter may snap to a timing raster. Use the *actual* applied
        # property perturbation as the finite-difference denominator. Using the
        # requested 1 ns probe after a 10 us snap makes sensitivity appear
        # artificially huge.
        applied_value = _get_solve_property_value(solve_object, solve_property)
        applied_delta = applied_value - old_value

        layout_after = _sequence_timing_layout(seq)

        target_block_after = layout_after.event_to_block.get(id(target))
        target_block_start_after = layout_after.block_starts.get(id(target_block_after), 0.0)
        target_anchor_after = target_block_start_after + _event_anchor_time_local(
            target,
            target_anchor,
            include_delay=True,
        )

    finally:
        _set_solve_property_value(solve_object, solve_property, old_value)

    if abs(applied_delta) < 1e-15:
        return 0.0

    return (target_anchor_after - target_anchor_before) / applied_delta


def _is_delay_like(event: Any) -> bool:
    """Return True for delay/wait events."""

    event_type = str(getattr(event, "event_type", getattr(event, "type", "")) or "").lower()
    kind = str(getattr(event, "kind", "") or "").lower()
    class_name = event.__class__.__name__.lower()
    name = event_name(event).lower()

    return (
        event_type == "delay"
        or kind == "delay"
        or "delayevent" in class_name
        or name == "delay"
        or name.endswith("_delay")
        or "wait" in name
    )

def _is_rf_like_for_timing(event: Any) -> bool:
    """Return True when an event has explicit RF identity.

    RF classification takes precedence over ADC fallback fields. Enriched event
    containers may expose generic attributes such as ``num_samples`` even when
    those fields are irrelevant to the concrete event.
    """

    event_type = str(
        getattr(event, "event_type", getattr(event, "type", "")) or ""
    ).lower()
    kind = str(getattr(event, "kind", "") or "").lower()
    class_name = event.__class__.__name__.lower()

    if event_type in {"rf", "radiofrequency"}:
        return True

    if kind in {
        "rf",
        "radiofrequency",
        "rf_block",
        "rf_sinc",
        "rf_gauss",
        "rf_gaussian",
        "rf_arbitrary",
    }:
        return True

    if (
        "rfevent" in class_name
        or "rfpulse" in class_name
        or "rfblock" in class_name
        or class_name.startswith("seqstarrf")
    ):
        return True

    return hasattr(event, "flip_angle")


def _is_adc_like(event: Any) -> bool:
    """Return True only for explicitly ADC-like events.

    Merely exposing ``num_samples`` is not sufficient because generic enriched
    event containers or shared protocol-backed objects may expose that field for
    RF, gradient, trigger, or delay events.
    """

    if _is_rf_like_for_timing(event):
        return False

    event_type = str(
        getattr(event, "event_type", getattr(event, "type", "")) or ""
    ).lower()
    kind = str(getattr(event, "kind", "") or "").lower()
    class_name = event.__class__.__name__.lower()
    name = event_name(event).lower()

    if event_type == "adc":
        return True

    if kind.startswith("adc"):
        return True

    if "adcevent" in class_name or "adctrain" in class_name:
        return True

    if hasattr(event, "to_pulseq_windows"):
        return True

    if hasattr(event, "windows") and "adc" in name:
        return True

    # Structural fallback for custom ADC classes: both sampling fields must be
    # concrete object attributes. Do not consult a shared parameters mapping for
    # family classification.
    return (
        getattr(event, "num_samples", None) is not None
        and getattr(event, "dwell", None) is not None
        and not hasattr(event, "flip_angle")
    )


def _iter_blocks(seq: Any) -> Iterable[Any]:
    if hasattr(seq, "timeline") and hasattr(seq.timeline, "blocks"):
        blocks = seq.timeline.blocks
        if isinstance(blocks, Mapping):
            yield from blocks.values()
        else:
            yield from blocks
        return

    if hasattr(seq, "blocks"):
        blocks = seq.blocks
        if isinstance(blocks, Mapping):
            yield from blocks.values()
        else:
            yield from blocks
        return

    raise TypeError("Sequence does not expose timeline.blocks or blocks.")


def _iter_events(block: Any) -> Iterable[Any]:
    visited: set[int] = set()
    yield from _iter_node_events(block, visited=visited, is_root=True)


def _iter_node_events(node: Any, *, visited: set[int], is_root: bool = False) -> Iterable[Any]:
    if node is None or isinstance(node, (float, int, str, bool)):
        return

    node_id = id(node)
    if node_id in visited:
        return
    visited.add(node_id)

    if not is_root and _looks_like_event(node):
        yield node
        return

    if isinstance(node, Mapping):
        for value in node.values():
            yield from _iter_node_events(value, visited=visited)
        return

    if isinstance(node, (list, tuple, set)):
        for value in node:
            yield from _iter_node_events(value, visited=visited)
        return

    if hasattr(node, "events"):
        yield from _iter_node_events(getattr(node, "events"), visited=visited)

    if hasattr(node, "children"):
        yield from _iter_node_events(getattr(node, "children"), visited=visited)

    if is_dataclass(node):
        for field_info in fields(node):
            if field_info.name in {
                "parent",
                "relationships",
                "metadata",
                "parameters",
                "shape",
                "system",
            }:
                continue
            try:
                value = getattr(node, field_info.name)
            except AttributeError:
                continue
            yield from _iter_node_events(value, visited=visited)
    elif hasattr(node, "__dict__"):
        for key, value in vars(node).items():
            if key in {
                "parent",
                "relationships",
                "metadata",
                "parameters",
                "shape",
                "system",
            }:
                continue
            yield from _iter_node_events(value, visited=visited)


def _looks_like_event(obj: Any) -> bool:
    if obj is None or isinstance(obj, (float, int, str, bool)):
        return False

    class_name = obj.__class__.__name__.lower()
    if "shape" in class_name and not any(
        hasattr(obj, marker)
        for marker in (
            "flip_angle",
            "num_samples",
            "dwell",
            "channel",
            "axis",
            "rise_time",
            "flat_time",
            "fall_time",
            "area",
            "amplitude",
            "delay",
        )
    ):
        return False

    event_type = str(getattr(obj, "event_type", getattr(obj, "type", "")) or "").lower()
    kind = str(getattr(obj, "kind", "") or "").lower()

    return (
        event_type in {"rf", "adc", "grad", "gradient", "trap", "trapezoid", "delay"}
        or kind in {"rf", "adc", "grad", "gradient", "trap", "trapezoid", "delay"}
        or _is_delay_like(obj)
        or hasattr(obj, "flip_angle")
        or hasattr(obj, "num_samples")
        or hasattr(obj, "channel")
        or hasattr(obj, "axis")
        or all(hasattr(obj, attr) for attr in ("rise_time", "flat_time", "fall_time"))
    )


def _get_explicit_block_tstart(block: Any) -> float | None:
    """Return explicit absolute block start time, if present."""

    for attr in ("tstart", "start", "start_s", "absolute_tstart", "absolute_start"):
        value = getattr(block, attr, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    parameters = getattr(block, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in ("tstart", "start", "start_s", "absolute_tstart", "absolute_start"):
            value = parameters.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("tstart", "start", "start_s", "absolute_tstart", "absolute_start"):
            value = metadata.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

    return None

def _block_order_tolerance(seq: Any) -> float:
    system = getattr(seq, "system", None)
    raster = _first_positive_float(
        getattr(system, "block_duration_raster", None),
        getattr(system, "grad_raster_time", None),
        10e-6,
    )
    return max(1e-9, float(raster or 10e-6) + 1e-12)


def _block_display_name(block: Any) -> str:
    return str(
        _block_metadata_value(block, "seqstar_node")
        or _block_metadata_value(block, "path")
        or getattr(block, "name", block.__class__.__name__)
    )


def _block_node(block: Any) -> str | None:
    value = _block_metadata_value(block, "seqstar_node") or _block_metadata_value(block, "path")
    return str(value) if value is not None else None


def _block_metadata_value(block: Any, key: str, default: Any = None) -> Any:
    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, Mapping) and key in metadata:
        return metadata[key]
    parameters = getattr(block, "parameters", None)
    if isinstance(parameters, Mapping) and key in parameters:
        return parameters[key]
    return getattr(block, key, default)


def _live_block_from_relationship(rel: SeqStarRelationship, role: str) -> Any | None:
    object_key = f"{role}_block_object"
    id_key = f"{role}_block_object_id"

    obj = rel.metadata.get(object_key)
    if obj is not None:
        _register_object(obj)
        return obj

    return _object_from_id(rel.metadata.get(id_key))


def _live_solve_object_from_relationship(rel: SeqStarRelationship) -> Any | None:
    obj = rel.metadata.get("solve_object")
    if obj is not None:
        _register_object(obj)
        return obj

    return _object_from_id(rel.metadata.get("solve_object_id"))

def _live_object_from_relationship(rel: SeqStarRelationship, role: str) -> Any | None:
    object_key = f"{role}_object"
    id_key = f"{role}_object_id"

    obj = rel.metadata.get(object_key)
    if obj is not None:
        _register_object(obj)
        return obj

    return _object_from_id(rel.metadata.get(id_key))




def _object_from_id(obj_id: Any) -> Any | None:
    if not isinstance(obj_id, int):
        return None
    return _OBJECT_REGISTRY.get(obj_id)


def _register_relationship_objects(seq: Any) -> None:
    """Populate registry from attached relationships."""

    for rel in get_relationships(seq):
        for key in ("target_object", "reference_object", "solve_object", "target_block_object", "reference_block_object"):
            obj = rel.metadata.get(key)
            if obj is not None:
                _register_object(obj)

        for event in rel.metadata.get("event_objects", []) or []:
            if event is not None:
                _register_object(event)


def _register_object(obj: Any) -> None:
    _OBJECT_REGISTRY[id(obj)] = obj
