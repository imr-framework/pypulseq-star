"""PyPulseq .seq writer for pypulseq_star.

This writer converts an enriched SeqStarSequence into a PyPulseq Sequence.

Design principle:
    - pypulseq_star owns the enriched object model.
    - This writer is only an adapter to the PyPulseq .seq backend.
    - No RF-train-specific or ADC-train-specific assumptions should live in
      the public sequence object.

Important ADC policy:
    SeqStar/gammaSTAR may represent ADC as one enriched ADC-train object.

    Pulseq .seq does not have a first-class ADC-train object. Therefore, this
    writer lowers each enriched ADC train into one or more standard PyPulseq ADC
    events/blocks at write time.

    Timing policy for lowered ADC events and trains:
        - The enriched ADC event/train is the timing source of truth.
        - Concrete ADC start is event.delay plus window-local delay.
        - ADC frontend dead time is treated as a train-level concept by default.
        - During Pulseq lowering, PyPulseq's per-ADC dead-time enforcement is
          suppressed for lowered train windows so the dead time is not added
          repeatedly.
        - The first lowered ADC window is protected by applying the train-level
          ADC dead time once.
        - Each emitted Pulseq block is padded to block_duration_raster before
          calling Sequence.write(...), because Pulseq block durations must be
          representable on the block-duration raster.
"""

from __future__ import annotations

import copy
import inspect
import math
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pypulseq as pp

from pypulseq_star.writers._realization import sequence_view


class PulseqWriter:
    """Write a SeqStarSequence to a Pulseq `.seq` file.

    Parameters
    ----------
    sequence
        Enriched SeqStarSequence-like object.

    adc_train_policy
        Strategy for lowering enriched ADC trains to Pulseq-compatible ADC
        events.

        "windows"
            Default. Lower every ADC train window to a standard one-window
            Pulseq ADC block when representable. If tightly packed windows
            cannot be represented after block-raster padding but can be merged
            safely, the writer falls back to one continuous ADC event and warns.

        "single_if_possible"
            If an ADC train has one window, export one standard ADC event. If it
            has multiple windows, lower window-by-window when representable.

        "continuous_if_possible"
            If ADC windows are contiguous and share sample/frequency/phase
            settings, merge them into one long ADC event. Otherwise fall back to
            window-by-window lowering.

        "error_if_multiwindow"
            Strict compatibility mode. Raise if a SeqStar ADC object contains
            more than one ADC window.

    allow_multiwindow_adc_with_other_events
        Conservative safety flag. If a multi-window ADC train appears in the
        same SeqStar block as RF/gradient/delay events, naive lowering could
        move or duplicate those non-ADC events. By default this writer raises a
        clear error. Set this to True only if the block is known to be safe for
        simple sequential lowering.
    """

    def __init__(
        self,
        sequence: Any,
        *,
        adc_train_policy: str = "windows",
        allow_multiwindow_adc_with_other_events: bool = False,
        auto_resolve_relationships: bool = True,
        profile: bool = False,
    ) -> None:
        self.sequence = sequence
        self.adc_train_policy = adc_train_policy
        self.allow_multiwindow_adc_with_other_events = (
            allow_multiwindow_adc_with_other_events
        )
        self.auto_resolve_relationships = auto_resolve_relationships
        self.profile = bool(profile)

    def write(
        self,
        path: str | Path,
        *,
        realization: Any | None = None,
    ) -> Path:
        """Write a Pulseq file, optionally from a numeric realization."""

        view = sequence_view(
            self.sequence,
            realization=realization,
        )

        if not view.is_resolved:
            return self._write_current_sequence(path)

        resolved_writer = type(self)(
            view.export_sequence,
            adc_train_policy=self.adc_train_policy,
            allow_multiwindow_adc_with_other_events=(
                self.allow_multiwindow_adc_with_other_events
            ),
            auto_resolve_relationships=False,
            profile=self.profile,
        )
        return resolved_writer._write_current_sequence(path)

    def _write_current_sequence(self, path: str | Path) -> Path:
        """Write the enriched SeqStar sequence as a `.seq` file.

        Relationship policy
        -------------------
        Relationships are resolved before export by default. This is sequence-
        general: the writer does not know about FID, GRE, EPI, ASL, or pCASL. It
        simply ensures that any relationship-derived timing fields have been
        materialized onto the enriched sequence/events before lowering to Pulseq.

        TR policy
        ---------
        TR fill is computed from the actual emitted Pulseq duration after event
        conversion, channel-conflict splitting, ADC-train lowering, and block-raster
        padding. This avoids drift when export-time raster snapping changes the
        emitted duration relative to the enriched block duration.
        """

        total_start = perf_counter()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        resolve_start = perf_counter()
        if self.auto_resolve_relationships:
            _resolve_sequence_relationships_for_export(self.sequence)
        if self.profile:
            print(f"[profile] Pulseq relationship resolve      {perf_counter() - resolve_start:9.3f} s")

        pulseq_seq = pp.Sequence(system=_to_pypulseq_opts(self.sequence))

        if _sequence_has_nested_repeated_nodes(self.sequence):
            lowering_start = perf_counter()
            _emit_nested_node_timeline(
                sequence=self.sequence,
                pulseq_seq=pulseq_seq,
                adc_train_policy=self.adc_train_policy,
                allow_multiwindow_adc_with_other_events=(
                    self.allow_multiwindow_adc_with_other_events
                ),
            )
            if self.profile:
                print(
                    f"[profile] Pulseq lower nested timeline    "
                    f"{perf_counter() - lowering_start:9.3f} s"
                )
            write_start = perf_counter()
            pulseq_seq.write(str(output))
            if self.profile:
                print(
                    f"[profile] Pulseq serialize .seq           "
                    f"{perf_counter() - write_start:9.3f} s"
                )
                print(
                    f"[profile] Pulseq writer total             "
                    f"{perf_counter() - total_start:9.3f} s"
                )
            return output

        # Build executable export units from the timeline and logical-node
        # registry. A repeated node is emitted as one contiguous motif:
        #
        #     all blocks under node -> optional period fill -> repeat
        #
        # This is intentionally different from inheriting sequence-level
        # ``averages`` or ``TR`` independently into every block.
        units_start = perf_counter()
        export_units = _build_node_aware_export_units(self.sequence)
        if self.profile:
            print(f"[profile] Pulseq build export units       {perf_counter() - units_start:9.3f} s")

        lowering_start = perf_counter()
        for unit in export_units:
            motif_blocks = unit["blocks"]
            motif_repeat_count = int(unit["repeat_count"])
            motif_repeat_period = unit["repeat_period"]
            motif_name = str(unit["name"])

            if motif_repeat_count < 1:
                raise ValueError(
                    f"Repeated node/motif {motif_name!r} resolved to "
                    f"repeat_count={motif_repeat_count}; repeat_count must be >= 1."
                )

            for _motif_index in range(motif_repeat_count):
                motif_emitted_duration = 0.0

                for block in motif_blocks:
                    # Block-local repetition remains supported, but only when it
                    # is explicitly attached to the block. Global protocol
                    # ``averages`` is never inherited here.
                    # Only explicit block-local repetition metadata may
                    # repeat a single block. Sequence protocol parameters and
                    # inherited node metadata (averages/TR) belong to the
                    # complete logical motif and are handled by
                    # _build_node_aware_export_units().
                    block_repetitions = _get_explicit_block_repeat_count(block)
                    block_repetition_time = _get_repetition_time_for_block(
                        block,
                        sequence=self.sequence,
                    )

                    events = list(_iter_events(block))
                    if not events:
                        continue

                    for _block_repeat_index in range(block_repetitions):
                        pulseq_event_groups = _events_to_pulseq_event_groups(
                            events,
                            system=pulseq_seq.system,
                            adc_train_policy=self.adc_train_policy,
                            allow_multiwindow_adc_with_other_events=(
                                self.allow_multiwindow_adc_with_other_events
                            ),
                        )

                        block_emitted_duration = 0.0

                        for pulseq_events in pulseq_event_groups:
                            pulseq_events = [
                                event for event in pulseq_events if event is not None
                            ]

                            for pulseq_subgroup in _split_pulseq_event_group_for_channel_conflicts(
                                pulseq_events
                            ):
                                if not pulseq_subgroup:
                                    continue

                                pulseq_subgroup = _pad_pulseq_events_to_block_raster(
                                    pulseq_subgroup,
                                    pulseq_seq.system,
                                )

                                subgroup_duration = _pulseq_event_group_duration(
                                    pulseq_subgroup
                                )
                                block_emitted_duration += subgroup_duration
                                motif_emitted_duration += subgroup_duration

                                pulseq_seq.add_block(*pulseq_subgroup)

                        # Explicit block-local periods are still legal. They are
                        # applied only to this block repetition and never inferred
                        # from sequence-level TR.
                        if block_repetition_time is not None:
                            block_fill = (
                                float(block_repetition_time)
                                - block_emitted_duration
                            )

                            if block_fill < -1e-12:
                                raise ValueError(
                                    "Pulseq export produced a block repetition "
                                    "longer than its explicitly requested local "
                                    "period.\n"
                                    f"motif={motif_name!r}\n"
                                    f"requested_block_period="
                                    f"{float(block_repetition_time):.12g} s\n"
                                    f"emitted_block_duration="
                                    f"{block_emitted_duration:.12g} s\n"
                                    f"excess="
                                    f"{block_emitted_duration - float(block_repetition_time):.12g} s"
                                )

                            if block_fill > 0:
                                block_fill = _snap_time_to_raster(
                                    block_fill,
                                    _block_duration_raster(pulseq_seq.system),
                                    name="block-local repetition delay",
                                    mode="nearest",
                                )

                                if block_fill > 0:
                                    pulseq_seq.add_block(
                                        pp.make_delay(block_fill)
                                    )
                                    motif_emitted_duration += block_fill

                # Node-level repeat_every/TR is applied once after the entire
                # motif, not after each constituent block.
                if motif_repeat_period is not None:
                    motif_fill = (
                        float(motif_repeat_period)
                        - motif_emitted_duration
                    )

                    tolerance = max(
                        1e-12,
                        0.5 * _block_duration_raster(pulseq_seq.system),
                    )

                    if motif_fill < -tolerance:
                        raise ValueError(
                            "Pulseq export produced a repeated logical-node "
                            "motif longer than its requested repeat period.\n"
                            f"motif={motif_name!r}\n"
                            f"requested_repeat_period="
                            f"{float(motif_repeat_period):.12g} s\n"
                            f"emitted_motif_duration="
                            f"{motif_emitted_duration:.12g} s\n"
                            f"excess="
                            f"{motif_emitted_duration - float(motif_repeat_period):.12g} s"
                        )

                    if motif_fill > tolerance:
                        motif_fill = _snap_time_to_raster(
                            motif_fill,
                            _block_duration_raster(pulseq_seq.system),
                            name=f"{motif_name} repeat-period delay",
                            mode="nearest",
                        )

                        if motif_fill > 0:
                            pulseq_seq.add_block(pp.make_delay(motif_fill))

        if self.profile:
            print(f"[profile] Pulseq lower timeline           {perf_counter() - lowering_start:9.3f} s")
        write_start = perf_counter()
        pulseq_seq.write(str(output))
        if self.profile:
            print(f"[profile] Pulseq serialize .seq           {perf_counter() - write_start:9.3f} s")
            print(f"[profile] Pulseq writer total             {perf_counter() - total_start:9.3f} s")
        return output

def _split_pulseq_event_group_for_channel_conflicts(events: list[Any]) -> list[list[Any]]:
    """Split a Pulseq event group if it contains repeated gradient channels.

    Pulseq blocks may contain at most one gradient event per physical channel
    (x, y, z). SeqStar shape-focused demos may deliberately place a split
    gradient's ramp-up, flat-top, and ramp-down parts in one logical block.
    Those parts must be emitted as sequential Pulseq blocks, not as one block
    with multiple Z/X/Y gradient events.

    This keeps simultaneous orthogonal gradients together, e.g. one x + one y
    + one z gradient, while starting a new Pulseq block whenever a second
    gradient on an already-used channel appears. Non-gradient events stay in
    the first subgroup in which they appear; they are not duplicated.
    """

    if not events:
        return []

    groups: list[list[Any]] = []
    current_group: list[Any] = []
    used_gradient_channels: set[str] = set()

    for event in events:
        channel = _pulseq_gradient_channel(event)

        if channel is not None and channel in used_gradient_channels:
            if current_group:
                groups.append(current_group)
            current_group = []
            used_gradient_channels = set()

        current_group.append(event)

        if channel is not None:
            used_gradient_channels.add(channel)

    if current_group:
        groups.append(current_group)

    return groups


def _pulseq_gradient_channel(event: Any) -> str | None:
    """Return x/y/z for a PyPulseq gradient event, otherwise None."""

    if event is None:
        return None

    event_type = str(getattr(event, "type", "")).lower()
    channel = getattr(event, "channel", None)

    if channel is None:
        return None

    if event_type not in {"grad", "trap"}:
        # PyPulseq gradient events normally expose type='grad' or type='trap'.
        # A channel attribute on another object should not be treated as a
        # gradient-channel conflict unless the type also looks gradient-like.
        class_name = event.__class__.__name__.lower()
        if "grad" not in class_name and "trap" not in class_name:
            return None

    channel = str(channel).lower()
    if channel in {"x", "gx", "read", "readout", "ro"}:
        return "x"
    if channel in {"y", "gy", "phase", "phase_encode", "pe"}:
        return "y"
    if channel in {"z", "gz", "slice", "slice_select", "ss"}:
        return "z"

    return channel

def _to_pypulseq_opts(sequence: Any) -> pp.Opts:
    """Convert SeqStar Opts to PyPulseq Opts."""

    system = sequence.system

    if hasattr(system, "to_pypulseq_kwargs"):
        return pp.Opts(**system.to_pypulseq_kwargs())

    return pp.Opts(
        max_grad=float(system.max_grad),
        grad_unit=getattr(system, "grad_unit", "Hz/m"),
        max_slew=float(system.max_slew),
        slew_unit=getattr(system, "slew_unit", "Hz/m/s"),
        rf_ringdown_time=float(system.rf_ringdown_time),
        rf_dead_time=float(system.rf_dead_time),
        adc_dead_time=float(system.adc_dead_time),
        rf_raster_time=float(system.rf_raster_time),
        grad_raster_time=float(system.grad_raster_time),
        block_duration_raster=float(system.block_duration_raster),
        adc_raster_time=float(getattr(system, "adc_raster_time", 100e-9)),
        gamma=float(system.gamma),
        B0=float(getattr(system, "B0", 1.5)),
    )

def _resolve_sequence_relationships_for_export(sequence: Any) -> None:
    """Resolve relationship-derived timing before Pulseq export.

    This helper is intentionally defensive and sequence-general. It tries the
    public relationship resolver if available, then falls back to common sequence
    methods. If no resolver exists, it does nothing.
    """

    try:
        from pypulseq_star import relationships as _relationships
    except Exception:
        _relationships = None

    if _relationships is not None and hasattr(_relationships, "resolve"):
        _relationships.resolve(sequence)
        return

    for method_name in (
        "resolve_relationships",
        "resolve_timing_relationships",
        "resolve",
    ):
        method = getattr(sequence, method_name, None)
        if callable(method):
            method()
            return


def _get_repetition_time_for_block(
    block: Any,
    *,
    sequence: Any,
) -> float | None:
    """Return only an explicitly block-local repetition period.

    ``SeqStarSequence._attach_sequence_context`` may copy the complete protocol
    dictionary onto each block. Consequently, reading ``TR`` from
    ``block.parameters`` is not evidence that the developer requested a local
    one-block period. Node-level ``repeat_every`` is handled once for the whole
    motif by ``_build_node_aware_export_units``.

    Supported explicit metadata keys are intentionally namespaced to avoid
    colliding with inherited protocol values.
    """

    del sequence

    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in (
            "seqstar_block_repeat_every",
            "seqstar_block_repetition_time",
            "block_repeat_every",
            "block_repetition_time",
        ):
            value = metadata.get(key)
            if value is not None:
                return float(value)

    return None


def _get_explicit_block_repeat_count(block: Any) -> int:
    """Return only an explicitly block-local repeat count.

    Inherited ``averages``/``repeat_count`` values in ``block.parameters`` and
    ``seqstar_repeat_count`` in block metadata describe an ancestor logical
    node. They must not duplicate each constituent block independently.
    """

    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in (
            "seqstar_block_repeat_count",
            "block_repeat_count",
            "block_repetitions",
        ):
            value = metadata.get(key)
            if value is not None:
                count = int(value)
                if count < 1:
                    raise ValueError(
                        f"Explicit block repeat count must be >= 1; got {count}."
                    )
                return count

    return 1

def _get_resolved_relationship_float(
    sequence: Any,
    *,
    keys: tuple[str, ...],
    relationship_kinds: tuple[str, ...],
) -> float | None:
    """Read a resolved scalar from relationship metadata.

    This does not depend on any one relationship class. It supports dictionaries,
    dataclasses/objects with fields, and relationship metadata dictionaries.
    """

    for relationship in _iter_sequence_relationships(sequence):
        kind = _relationship_kind_for_export(relationship)

        if kind not in relationship_kinds:
            continue

        resolved = _relationship_resolved_mapping_for_export(relationship)

        for key in keys:
            if key in resolved and resolved[key] is not None:
                try:
                    return float(resolved[key])
                except (TypeError, ValueError):
                    continue

    return None


def _iter_sequence_relationships(sequence: Any) -> Iterable[Any]:
    """Yield relationship-like objects attached to a sequence."""

    candidate_attrs = (
        "relationship_definitions",
        "relationships",
        "_relationships",
        "_seqstar_relationships",
        "_pypulseq_star_relationships",
    )

    for attr in candidate_attrs:
        value = getattr(sequence, attr, None)

        if value is None:
            continue

        if isinstance(value, Mapping):
            yield from value.values()
            continue

        if isinstance(value, (list, tuple, set)):
            yield from value
            continue

        try:
            yield from list(value)
        except TypeError:
            yield value


def _relationship_kind_for_export(relationship: Any) -> str:
    """Return a normalized relationship kind/type string."""

    if isinstance(relationship, Mapping):
        for key in ("kind", "type", "relationship_type"):
            value = relationship.get(key)
            if value is not None:
                return str(value)

        metadata = relationship.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("kind", "type", "relationship_type"):
                value = metadata.get(key)
                if value is not None:
                    return str(value)

        return "relationship"

    for attr in ("kind", "type", "relationship_type"):
        value = getattr(relationship, attr, None)
        if value is not None:
            return str(value)

    metadata = getattr(relationship, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("kind", "type", "relationship_type"):
            value = metadata.get(key)
            if value is not None:
                return str(value)

    return "relationship"


def _relationship_resolved_mapping_for_export(
    relationship: Any,
) -> dict[str, Any]:
    """Return resolved relationship values as a plain dictionary."""

    if isinstance(relationship, Mapping):
        resolved = relationship.get("resolved")
        if isinstance(resolved, Mapping):
            return dict(resolved)

        metadata = relationship.get("metadata")
        if isinstance(metadata, Mapping):
            resolved = metadata.get("resolved")
            if isinstance(resolved, Mapping):
                return dict(resolved)

        return {}

    resolved = getattr(relationship, "resolved", None)
    if isinstance(resolved, Mapping):
        return dict(resolved)

    metadata = getattr(relationship, "metadata", None)
    if isinstance(metadata, Mapping):
        resolved = metadata.get("resolved")
        if isinstance(resolved, Mapping):
            return dict(resolved)

    return {}

def _events_to_pulseq_event_groups(
    events: list[Any],
    *,
    system: pp.Opts,
    adc_train_policy: str,
    allow_multiwindow_adc_with_other_events: bool,
) -> list[list[Any]]:
    """Convert SeqStar events to one or more PyPulseq event groups.

    A normal SeqStar block becomes one Pulseq block.

    A multi-window ADC train becomes multiple Pulseq blocks, one per lowered
    ADC window, when representable. If the train is too tightly packed for
    block-by-block lowering but can be merged into one continuous ADC, the
    writer can fall back to that representation.
    """

    adc_train_events: list[Any] = []
    ordinary_events: list[Any] = []

    for event in events:
        if _is_adc_event(event) and _has_adc_train_windows(event):
            adc_train_events.append(event)
        else:
            ordinary_events.append(event)

    if not adc_train_events:
        pulseq_events = _ordinary_events_to_pulseq_with_rotation(
            ordinary_events, system
        )
        return [[event for event in pulseq_events if event is not None]]

    total_adc_windows = sum(
        len(_get_adc_pulseq_windows(event))
        for event in adc_train_events
    )

    has_multiwindow_adc = total_adc_windows > len(adc_train_events)

    if ordinary_events and has_multiwindow_adc:
        synchronized_adc_events = [
            event
            for event in adc_train_events
            if _adc_train_requests_synchronized_lowering(event)
        ]

        if synchronized_adc_events:
            if len(adc_train_events) != 1:
                raise ValueError(
                    "Synchronized train lowering currently requires exactly "
                    "one multi-window ADC train in the SeqStar block."
                )

            if not all(_is_gradient_event(event) for event in ordinary_events):
                raise ValueError(
                    "Synchronized train lowering supports one ADC train with "
                    "gradient events in the same block. RF and delay events "
                    "must remain in separate blocks."
                )

            return _synchronized_gradient_adc_train_to_pulseq_groups(
                adc_event=adc_train_events[0],
                gradient_events=ordinary_events,
                system=system,
            )

        if not allow_multiwindow_adc_with_other_events:
            raise ValueError(
                "A multi-window SeqStar ADC train was found in the same block "
                "as non-ADC events. For synchronized gradient/ADC trains, set "
                "parameters['synchronized_with_block_gradients']=True and "
                "provide parameters['segment_duration']. Otherwise split the "
                "SeqStar block before writing, or set "
                "allow_multiwindow_adc_with_other_events=True only when simple "
                "sequential lowering is known to be safe."
            )

    if total_adc_windows == len(adc_train_events):
        pulseq_events: list[Any] = []

        pulseq_events.extend(
            _ordinary_events_to_pulseq_with_rotation(ordinary_events, system)
        )

        for adc_event in adc_train_events:
            converted = _adc_to_pypulseq(adc_event, system)
            if converted is not None:
                pulseq_events.append(converted)

        return [pulseq_events]

    pulseq_event_groups: list[list[Any]] = []

    if ordinary_events:
        ordinary_group = _ordinary_events_to_pulseq_with_rotation(
            ordinary_events, system
        )

        if ordinary_group:
            pulseq_event_groups.append(ordinary_group)

    for adc_event in adc_train_events:
        pulseq_event_groups.extend(
            _adc_train_to_pulseq_event_groups(
                adc_event,
                system=system,
                adc_train_policy=adc_train_policy,
            )
        )

    return pulseq_event_groups



def _adc_train_requests_synchronized_lowering(event: Any) -> bool:
    """Return whether an ADC train opts into synchronized gradient lowering."""

    for container_name in ("parameters", "metadata"):
        container = getattr(event, container_name, None)
        if not isinstance(container, Mapping):
            continue

        value = container.get("synchronized_with_block_gradients")
        if value is not None:
            return bool(value)

        nested = container.get("adc_train")
        if isinstance(nested, Mapping):
            value = nested.get("synchronized_with_block_gradients")
            if value is not None:
                return bool(value)

    return False


def _adc_train_segment_duration(event: Any) -> float:
    """Return the regular segment duration for synchronized train lowering."""

    for container_name in ("parameters", "metadata"):
        container = getattr(event, container_name, None)
        if not isinstance(container, Mapping):
            continue

        for key in ("segment_duration", "echo_spacing", "window_spacing"):
            value = container.get(key)
            if value is not None:
                value = float(value)
                if value <= 0:
                    raise ValueError(
                        f"Synchronized train {key} must be > 0; got {value}."
                    )
                return value

        nested = container.get("adc_train")
        if isinstance(nested, Mapping):
            for key in ("segment_duration", "echo_spacing", "window_spacing"):
                value = nested.get(key)
                if value is not None:
                    value = float(value)
                    if value <= 0:
                        raise ValueError(
                            f"Synchronized train {key} must be > 0; got {value}."
                        )
                    return value

    windows = sorted(
        _get_adc_pulseq_windows(event),
        key=lambda item: float(item.get("delay", item.get("tstart", 0.0))),
    )

    if len(windows) < 2:
        raise ValueError(
            "Synchronized train lowering requires segment_duration metadata "
            "or at least two regularly spaced ADC windows."
        )

    starts = [
        float(window.get("delay", window.get("tstart", 0.0)))
        for window in windows
    ]
    spacings = np.diff(starts)

    if not np.allclose(spacings, spacings[0], rtol=0.0, atol=1e-12):
        raise ValueError(
            "Synchronized train ADC windows are not regularly spaced. "
            "Provide an explicit segment_duration or use a future "
            "variable-segment lowering policy."
        )

    return float(spacings[0])


def _synchronized_gradient_adc_train_to_pulseq_groups(
    *,
    adc_event: Any,
    gradient_events: list[Any],
    system: pp.Opts,
) -> list[list[Any]]:
    """Lower one regular gradient/ADC train block without losing overlap.

    One enriched block may contain long arbitrary gradient waveforms and one
    multi-window ADC train. Standard Pulseq has no multi-window ADC primitive,
    so this function divides the train into regular segments. Each emitted
    Pulseq block contains:

        gradient waveform slice(s) for that segment
        the corresponding ADC window with segment-local delay

    This is sequence-family agnostic. The event explicitly opts in and provides
    the regular segment duration; no EPI, GRE, RARE, or GRASE names are used.
    """

    windows = sorted(
        _get_adc_pulseq_windows(adc_event),
        key=lambda item: float(item.get("delay", item.get("tstart", 0.0))),
    )

    if not windows:
        raise ValueError("Synchronized ADC train has no windows.")

    segment_duration = _adc_train_segment_duration(adc_event)
    grad_raster = float(getattr(system, "grad_raster_time", 10e-6))
    segment_samples_float = segment_duration / grad_raster
    segment_samples = int(round(segment_samples_float))

    if abs(segment_samples_float - segment_samples) > 1e-9:
        raise ValueError(
            "Synchronized train segment_duration must be representable on the "
            "gradient raster. "
            f"segment_duration={segment_duration}, grad_raster={grad_raster}."
        )

    if segment_samples < 1:
        raise ValueError("Synchronized train segment contains no gradient samples.")

    waveforms: list[tuple[str, np.ndarray]] = []

    for gradient_event in gradient_events:
        channel = _gradient_channel(gradient_event)
        waveform = _gradient_waveform_for_pulseq(gradient_event)

        if channel is None or waveform is None:
            raise ValueError(
                "Synchronized train gradients must expose channel and waveform."
            )

        waveforms.append(
            (channel, np.asarray(waveform, dtype=float))
        )

    expected_samples = segment_samples * len(windows)

    for channel, waveform in waveforms:
        if len(waveform) != expected_samples:
            raise ValueError(
                "Synchronized gradient waveform length does not match the ADC "
                "train segmentation. "
                f"channel={channel!r}, waveform_samples={len(waveform)}, "
                f"expected_samples={expected_samples}, "
                f"segments={len(windows)}, samples_per_segment={segment_samples}."
            )

    train_event_delay = _get_event_delay(adc_event)
    if train_event_delay < -1e-12:
        raise ValueError(
            f"ADC train event delay must be non-negative; got {train_event_delay}."
        )
    train_event_delay = max(train_event_delay, 0.0)
    train_dead_time = _get_adc_train_dead_time(adc_event, system)
    groups: list[list[Any]] = []

    for index, window in enumerate(windows):
        segment_start = index * segment_duration
        group: list[Any] = []

        start = index * segment_samples
        stop = start + segment_samples

        for channel, waveform in waveforms:
            segment = waveform[start:stop]

            group.append(
                pp.make_arbitrary_grad(
                    channel=channel,
                    waveform=segment,
                    first=float(segment[0]),
                    last=float(segment[-1]),
                    system=system,
                )
            )

        absolute_window_start = (
            train_event_delay
            + float(window.get("delay", window.get("tstart", 0.0)))
        )
        local_delay = absolute_window_start - segment_start

        if index == 0:
            local_delay = max(local_delay, train_dead_time)

        if local_delay < -1e-12:
            raise ValueError(
                "ADC window begins before its synchronized gradient segment. "
                f"window={index}, local_delay={local_delay}."
            )

        lowered_window = dict(window)
        lowered_window["delay"] = max(local_delay, 0.0)

        group.append(
            _adc_window_dict_to_pypulseq(
                lowered_window,
                system,
                suppress_adc_dead_time=True,
            )
        )

        groups.append(group)

    return groups


def _adc_train_to_pulseq_event_groups(
    event: Any,
    *,
    system: pp.Opts,
    adc_train_policy: str,
) -> list[list[Any]]:
    """Lower one enriched ADC train into standard Pulseq ADC blocks.

    Important:
        We track exported block time, not only ADC-window end time.

    Why:
        If an ADC window duration is legal on adc_raster_time but not on
        block_duration_raster, the writer pads the emitted Pulseq block.
        The next lowered window must then be delayed relative to the padded
        block end, otherwise cumulative timing drift or Pulseq write assertions
        occur.
    """

    windows = [
        _adc_window_with_parent_delay(event, window)
        for window in _get_adc_pulseq_windows(event)
    ]

    if not windows:
        raise ValueError(f"ADC event {event!r} has no Pulseq-lowerable windows.")

    if adc_train_policy == "error_if_multiwindow" and len(windows) > 1:
        raise ValueError(
            "PulseqWriter was called with adc_train_policy='error_if_multiwindow', "
            f"but ADC event {event!r} has {len(windows)} windows."
        )

    if adc_train_policy == "continuous_if_possible":
        if _can_merge_adc_windows_for_continuous_export(event, windows):
            return _adc_train_to_continuous_pulseq_group(
                event,
                windows=windows,
                system=system,
                reason="requested continuous_if_possible",
            )

    if adc_train_policy not in {
        "windows",
        "single_if_possible",
        "continuous_if_possible",
        "error_if_multiwindow",
    }:
        raise ValueError(
            f"Unsupported adc_train_policy={adc_train_policy!r}. "
            "Use 'windows', 'single_if_possible', 'continuous_if_possible', "
            "or 'error_if_multiwindow'."
        )

    sorted_windows = sorted(
        windows,
        key=lambda window: float(window.get("delay", window.get("tstart", 0.0))),
    )

    if len(sorted_windows) == 1:
        return [
            [
                _adc_window_dict_to_pypulseq(
                    sorted_windows[0],
                    system,
                    suppress_adc_dead_time=False,
                )
            ]
        ]

    if _sequential_adc_lowering_will_violate_timing(
        event,
        windows=sorted_windows,
        system=system,
    ):
        if _can_merge_adc_windows_for_continuous_export(event, sorted_windows):
            return _adc_train_to_continuous_pulseq_group(
                event,
                windows=sorted_windows,
                system=system,
                reason=(
                    "window-by-window lowering would violate timing after "
                    "block-duration-raster padding"
                ),
            )

        raise ValueError(
            "This ADC train cannot be represented as separate one-window Pulseq "
            "ADC blocks without changing timing, and it also cannot be safely "
            "merged into one continuous ADC event. Use a larger echo spacing, "
            "choose adc_train_policy='continuous_if_possible' for compatible "
            "trains, or split the sequence into hardware-valid blocks."
        )

    warnings.warn(
        f"Lowering SeqStar ADC train {getattr(event, 'name', 'adc')!r} with "
        f"{len(sorted_windows)} windows to {len(sorted_windows)} standard "
        "Pulseq ADC blocks. ADC dead time is applied once at the train level; "
        "per-window PyPulseq adc_dead_time enforcement is suppressed during "
        "lowering. Each emitted ADC block is padded to block_duration_raster "
        "during .seq export.",
        stacklevel=2,
    )

    groups: list[list[Any]] = []
    exported_time = 0.0
    train_dead_time = _get_adc_train_dead_time(event, system)
    block_raster = _block_duration_raster(system)

    for window_index, window in enumerate(sorted_windows):
        absolute_delay = float(window.get("delay", window.get("tstart", 0.0)))

        if window_index == 0:
            target_absolute_start = max(absolute_delay, train_dead_time)
        else:
            target_absolute_start = absolute_delay

        relative_delay = target_absolute_start - exported_time

        if relative_delay < -1e-12:
            raise ValueError(
                "ADC train timing cannot be represented by sequential Pulseq "
                "ADC blocks after block-raster padding.\n"
                f"window_index={window_index}, "
                f"target_absolute_start={target_absolute_start}, "
                f"already_exported_time={exported_time}, "
                f"deficit={exported_time - target_absolute_start}"
            )

        relative_delay = max(relative_delay, 0.0)
        relative_delay = _snap_time_to_raster(
            relative_delay,
            block_raster,
            name=f"lowered ADC window {window_index} relative delay",
            mode="nearest",
        )

        lowered_window = dict(window)
        lowered_window["delay"] = relative_delay
        lowered_window.setdefault("metadata", {})
        lowered_window["metadata"]["seqstar_original_absolute_delay"] = absolute_delay
        lowered_window["metadata"]["seqstar_target_absolute_start"] = target_absolute_start
        lowered_window["metadata"]["seqstar_exported_time_before_window"] = exported_time
        lowered_window["metadata"]["seqstar_lowered_relative_delay"] = relative_delay
        lowered_window["metadata"]["seqstar_adc_dead_time_applied_once"] = (
            train_dead_time if window_index == 0 else 0.0
        )

        adc_pulseq_event = _adc_window_dict_to_pypulseq(
            lowered_window,
            system,
            suppress_adc_dead_time=True,
        )

        group = [adc_pulseq_event]
        raw_group_duration = _pulseq_event_group_duration(group)
        padded_group_duration = _ceil_time_to_raster(raw_group_duration, block_raster)

        groups.append(group)
        exported_time += padded_group_duration

    return groups


def _adc_train_to_continuous_pulseq_group(
    event: Any,
    *,
    windows: list[dict[str, Any]],
    system: pp.Opts,
    reason: str,
) -> list[list[Any]]:
    """Merge a compatible ADC train into one continuous Pulseq ADC event."""

    merged = _merge_adc_windows_for_continuous_export(windows)
    train_dead_time = _get_adc_train_dead_time(event, system)

    merged["delay"] = max(float(merged.get("delay", 0.0)), train_dead_time)
    merged["delay"] = _snap_time_to_raster(
        float(merged["delay"]),
        _block_duration_raster(system),
        name="merged ADC train delay",
        mode="nearest",
    )

    merged.setdefault("metadata", {})
    merged["metadata"]["seqstar_adc_train_lowering"] = "continuous_merge"
    merged["metadata"]["seqstar_adc_dead_time_applied_once"] = train_dead_time
    merged["metadata"]["seqstar_continuous_merge_reason"] = reason

    warnings.warn(
        f"Lowered SeqStar ADC train {getattr(event, 'name', 'adc')!r} to one "
        f"continuous Pulseq ADC event because {reason}. ADC dead time was "
        "applied once at the train level.",
        stacklevel=2,
    )

    return [
        [
            _adc_window_dict_to_pypulseq(
                merged,
                system,
                suppress_adc_dead_time=True,
            )
        ]
    ]


def _sequential_adc_lowering_will_violate_timing(
    event: Any,
    *,
    windows: list[dict[str, Any]],
    system: pp.Opts,
) -> bool:
    """Return True if separate Pulseq blocks cannot preserve ADC timing.

    This simulates the lower-and-pad process. If padding one lowered block makes
    the exported timeline pass the start time of the next ADC window, then
    window-by-window .seq export cannot preserve the enriched timing.
    """

    exported_time = 0.0
    train_dead_time = _get_adc_train_dead_time(event, system)
    block_raster = _block_duration_raster(system)

    for window_index, window in enumerate(windows):
        absolute_delay = float(window.get("delay", window.get("tstart", 0.0)))

        if window_index == 0:
            target_absolute_start = max(absolute_delay, train_dead_time)
        else:
            target_absolute_start = absolute_delay

        relative_delay = target_absolute_start - exported_time

        if relative_delay < -1e-12:
            return True

        relative_delay = max(relative_delay, 0.0)
        relative_delay = _snap_time_to_raster(
            relative_delay,
            block_raster,
            name=f"simulated lowered ADC window {window_index} relative delay",
            mode="nearest",
            warn=False,
        )

        simulated_window = dict(window)
        simulated_window["delay"] = relative_delay

        duration = _adc_window_export_duration(simulated_window, system)
        raw_group_duration = relative_delay + duration
        padded_group_duration = _ceil_time_to_raster_no_warn(
            raw_group_duration,
            block_raster,
        )

        exported_time += padded_group_duration

    return False


def _ordinary_events_to_pulseq_with_rotation(
    events: list[Any],
    system: pp.Opts,
) -> list[Any]:
    """Lower ordinary events, rotating logical gradients into physical axes.

    Axis-aligned frames retain trapezoids. Oblique frames are sampled on the
    gradient raster, summed per physical channel, and emitted as arbitrary
    gradients. This ensures simultaneous read/phase/slice gradients are
    combined before physical-axis hardware checks.
    """
    gradients=[event for event in events if _is_gradient_event(event)]
    nongradients=[event for event in events if not _is_gradient_event(event)]
    result=[_to_pypulseq_event(event,system) for event in nongradients]
    result=[event for event in result if event is not None]
    if not gradients: 
        return result

    directions=[_gradient_physical_direction(event) for event in gradients]
    axis_aligned=all(
        sum(abs(v)>1e-9 for v in direction)==1
        and any(abs(abs(v)-1.0)<1e-9 for v in direction)
        for direction in directions
    )
    physical_axes=[]
    for direction in directions:
        physical_axes.append(max(range(3), key=lambda i: abs(direction[i])))
    no_collision=len(set(physical_axes))==len(physical_axes)

    if axis_aligned and no_collision:
        for event,direction,axis_index in zip(gradients,directions,physical_axes,strict=True):
            coefficient=direction[axis_index]
            result.append(_gradient_to_pypulseq(
                event, system,
                channel=("x","y","z")[axis_index],
                scale=coefficient,
            ))
        return result

    raster=float(getattr(system,"grad_raster_time",10e-6))
    block_duration=max(_get_event_delay(event)+_get_event_active_duration(event) for event in gradients)
    count=max(1,int(math.ceil(block_duration/raster-1e-12)))
    sample_times=(np.arange(count,dtype=float)+0.5)*raster
    physical=np.zeros((3,count),dtype=float)
    for event,direction in zip(gradients,directions,strict=True):
        delay=_get_event_delay(event)
        tt=np.asarray(getattr(event,"tt"),dtype=float)+delay
        waveform=np.asarray(getattr(event,"waveform"),dtype=float)
        if tt.size != waveform.size:
            raise ValueError("Gradient time and waveform arrays must have equal length for rotation.")
        logical=np.interp(sample_times,tt,waveform,left=0.0,right=0.0)
        for axis in range(3): 
            physical[axis]+=direction[axis]*logical

    max_grad=float(getattr(system,"max_grad",float("inf")))
    max_slew=float(getattr(system,"max_slew",float("inf")))
    for axis,channel in enumerate(("x","y","z")):
        waveform=physical[axis]
        if not np.any(np.abs(waveform)>1e-12): 
            continue
        if np.max(np.abs(waveform))>max_grad*(1+1e-12):
            raise ValueError(f"Rotated {channel}-gradient exceeds max_grad.")
        if waveform.size>1 and np.max(np.abs(np.diff(waveform))/raster)>max_slew*(1+1e-12):
            raise ValueError(f"Rotated {channel}-gradient exceeds max_slew.")
        result.append(pp.make_arbitrary_grad(channel=channel,waveform=waveform,system=system))
    return result


def _gradient_physical_direction(event: Any) -> tuple[float,float,float]:
    value=getattr(event,"physical_direction",None)
    if isinstance(value,(list,tuple)) and len(value)==3:
        return tuple(float(v) for v in value)
    metadata=getattr(event,"metadata",None)
    if isinstance(metadata,Mapping):
        value=metadata.get("physical_direction")
        if isinstance(value,(list,tuple)) and len(value)==3:
            return tuple(float(v) for v in value)
    channel=_gradient_channel(event) or "x"
    return {"x":(1.0,0.0,0.0),"y":(0.0,1.0,0.0),"z":(0.0,0.0,1.0)}[channel]


def _to_pypulseq_event(event: Any, system: pp.Opts) -> Any | None:
    """Convert one enriched SeqStar event to a PyPulseq event.

    Dispatch policy
    ---------------
    Use explicit event-family checks before generic conversion hooks. This avoids
    routing gradients with roles like ``excitation``/``rephase`` into the RF
    converter, which then tries to read RF-only fields such as ``flip_angle``.
    """

    if event is None:
        return None

    if _is_delay_event(event):
        duration = _get_event_duration(event)
        duration = _snap_time_to_raster(
            duration,
            _block_duration_raster(system),
            name="delay event duration",
            mode="nearest",
        )
        return pp.make_delay(duration)

    if _is_adc_event(event):
        return _adc_to_pypulseq(event, system)

    if _is_gradient_event(event):
        return _gradient_to_pypulseq(event, system)

    if _is_rf_event(event):
        return _rf_to_pypulseq(event, system)

    # Last-resort object-provided conversion hooks. Keep these after explicit
    # dispatch so writer-side family-specific fixes are honored.
    if hasattr(event, "to_pypulseq"):
        return event.to_pypulseq(system=system)

    if hasattr(event, "to_pulseq"):
        return event.to_pulseq(system=system)

    return None

def _rf_to_pypulseq(event: Any, system: pp.Opts) -> Any:
    """Convert a SeqStar RF event to PyPulseq RF.

    Current conservative behavior:
        RF exports as a block pulse.

    Export-time raster policy:
        RF delay and RF duration are snapped to the Pulseq block raster so the
        emitted .seq block duration is representable.
    """

    flip_angle = float(getattr(event, "flip_angle"))

    duration = _get_event_active_duration(event)
    duration = _snap_time_to_raster(
        duration,
        _block_duration_raster(system),
        name="RF duration",
        mode="nearest",
    )

    delay = _get_event_delay(event)
    delay = _snap_time_to_raster(
        delay,
        _block_duration_raster(system),
        name="RF delay",
        mode="nearest",
    )

    phase_offset = _get_timing_or_attr(
        event,
        keys=("phase_offset", "phase", "rf_phase", "excitation_phase"),
        default=0.0,
    )

    freq_offset = _get_timing_or_attr(
        event,
        keys=("freq_offset", "frequency", "frequency_offset", "rf_frequency_offset"),
        default=0.0,
    )

    use = getattr(event, "use", None) or getattr(event, "role", None) or ""

    return pp.make_block_pulse(
        flip_angle=flip_angle,
        delay=delay,
        duration=duration,
        phase_offset=float(phase_offset),
        freq_offset=float(freq_offset),
        system=system,
        use=use,
    )


def _adc_to_pypulseq(event: Any, system: pp.Opts) -> Any:
    """Convert a SeqStar ADC event to one standard PyPulseq ADC event."""

    if _has_adc_train_windows(event):
        windows = _get_adc_pulseq_windows(event)

        if len(windows) != 1:
            raise ValueError(
                "Internal writer error: multi-window ADC train reached "
                "_adc_to_pypulseq(). It should have been lowered by "
                "_adc_train_to_pulseq_event_groups()."
            )

        window = _adc_window_with_parent_delay(event, windows[0])
        return _adc_window_dict_to_pypulseq(
            window,
            system,
            suppress_adc_dead_time=False,
        )

    delay = _get_event_delay(event)

    num_samples = _get_timing_or_attr(
        event,
        keys=("num_samples", "samples", "n_samples"),
        default=None,
    )

    duration = _get_timing_or_attr(
        event,
        keys=("duration", "adc_duration", "readout_duration"),
        default=None,
    )

    dwell = _get_timing_or_attr(
        event,
        keys=("dwell", "dwell_time", "sample_time"),
        default=None,
    )

    phase_offset = _get_timing_or_attr(
        event,
        keys=("phase_offset", "phase"),
        default=0.0,
    )

    freq_offset = _get_timing_or_attr(
        event,
        keys=("freq_offset", "frequency", "frequency_offset"),
        default=0.0,
    )

    freq_ppm = _get_timing_or_attr(
        event,
        keys=("freq_ppm", "adc_freq_ppm"),
        default=0.0,
    )

    phase_ppm = _get_timing_or_attr(
        event,
        keys=("phase_ppm", "adc_phase_ppm"),
        default=0.0,
    )

    phase_modulation = _get_timing_or_attr(
        event,
        keys=("phase_modulation", "adc_phase_modulation"),
        default=None,
    )

    if num_samples is None:
        raise ValueError(f"ADC event {event!r} is missing num_samples.")

    window = {
        "num_samples": int(num_samples),
        "duration": duration,
        "dwell": dwell,
        "delay": delay,
        "phase_offset": phase_offset,
        "freq_offset": freq_offset,
        "freq_ppm": freq_ppm,
        "phase_ppm": phase_ppm,
        "phase_modulation": phase_modulation,
    }

    return _adc_window_dict_to_pypulseq(
        window,
        system,
        suppress_adc_dead_time=False,
    )


def _adc_window_dict_to_pypulseq(
    window: Mapping[str, Any],
    system: pp.Opts,
    *,
    suppress_adc_dead_time: bool,
) -> Any:
    """Convert one lowered ADC-window dictionary to PyPulseq ADC.

    Timing policy:
        - dwell is snapped to adc_raster_time.
        - delay is snapped to block_duration_raster.
        - duration is represented through dwell whenever possible.
        - when suppress_adc_dead_time=True, the temporary PyPulseq system has
          adc_dead_time=0 so dead time is not repeatedly added for every lowered
          ADC-train window.
    """

    num_samples = window.get("num_samples", window.get("number_of_samples", None))

    if num_samples is None:
        raise ValueError(f"ADC window {window!r} is missing num_samples.")

    num_samples = int(num_samples)

    delay = float(window.get("delay", window.get("tstart", 0.0)))
    delay = _snap_time_to_raster(
        delay,
        _block_duration_raster(system),
        name="ADC delay",
        mode="nearest",
    )

    dwell = window.get("dwell", window.get("sample_time", None))
    duration = window.get("duration", window.get("adc_duration", None))

    if dwell is None and duration is None:
        raise ValueError(f"ADC window {window!r} must define dwell or duration.")

    adc_raster = _adc_raster_time(system)

    if dwell is not None:
        dwell = float(dwell)
    else:
        dwell = float(duration) / num_samples

    dwell = _snap_time_to_raster(
        dwell,
        adc_raster,
        name="ADC dwell",
        mode="nearest",
    )

    active_duration = num_samples * dwell

    if duration is not None:
        original_duration = float(duration)

        if abs(original_duration - active_duration) > max(1e-12, 0.5 * adc_raster):
            warnings.warn(
                "ADC duration changed during Pulseq export because dwell must "
                "be adc-raster aligned. "
                f"original_duration={original_duration}, "
                f"exported_duration={active_duration}, "
                f"num_samples={num_samples}, dwell={dwell}",
                stacklevel=2,
            )

    phase_offset = float(window.get("phase_offset", window.get("phase", 0.0)))
    freq_offset = float(
        window.get("freq_offset", window.get("frequency_offset", 0.0))
    )
    freq_ppm = float(window.get("freq_ppm", 0.0))
    phase_ppm = float(window.get("phase_ppm", 0.0))
    phase_modulation = window.get("phase_modulation", None)

    adc_system = (
        _copy_pypulseq_system_with_adc_dead_time(system, 0.0)
        if suppress_adc_dead_time
        else system
    )

    kwargs: dict[str, Any] = {
        "num_samples": num_samples,
        "delay": delay,
        "dwell": dwell,
        "phase_offset": phase_offset,
        "freq_offset": freq_offset,
        "system": adc_system,
    }

    if freq_ppm != 0.0:
        kwargs["freq_ppm"] = freq_ppm

    if phase_ppm != 0.0:
        kwargs["phase_ppm"] = phase_ppm

    if phase_modulation is not None:
        kwargs["phase_modulation"] = list(phase_modulation)

    return _call_make_adc_with_supported_kwargs(kwargs)


def _call_make_adc_with_supported_kwargs(kwargs: dict[str, Any]) -> Any:
    """Call pp.make_adc while tolerating PyPulseq-version differences.

    Some PyPulseq versions may not support newer keyword arguments such as
    freq_ppm, phase_ppm, or phase_modulation. This helper filters unsupported
    keyword arguments based on the installed make_adc signature.
    """

    signature = inspect.signature(pp.make_adc)
    parameters = signature.parameters

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if accepts_var_kwargs:
        return pp.make_adc(**kwargs)

    supported_kwargs = {
        key: value for key, value in kwargs.items() if key in parameters
    }

    dropped = sorted(set(kwargs) - set(supported_kwargs))

    if dropped:
        warnings.warn(
            "Installed PyPulseq pp.make_adc does not support these keyword "
            f"arguments and they were omitted during .seq export: {dropped}",
            stacklevel=2,
        )

    return pp.make_adc(**supported_kwargs)


def _copy_pypulseq_system_with_adc_dead_time(
    system: pp.Opts,
    adc_dead_time: float,
) -> pp.Opts:
    """Return a PyPulseq system copy with a modified ADC dead time."""

    try:
        copied_system = copy.copy(system)
        copied_system.adc_dead_time = float(adc_dead_time)
        return copied_system
    except Exception:
        pass

    return pp.Opts(
        max_grad=float(getattr(system, "max_grad")),
        grad_unit=getattr(system, "grad_unit", "Hz/m"),
        max_slew=float(getattr(system, "max_slew")),
        slew_unit=getattr(system, "slew_unit", "Hz/m/s"),
        rf_ringdown_time=float(getattr(system, "rf_ringdown_time", 0.0)),
        rf_dead_time=float(getattr(system, "rf_dead_time", 0.0)),
        adc_dead_time=float(adc_dead_time),
        rf_raster_time=float(getattr(system, "rf_raster_time", 1e-6)),
        grad_raster_time=float(getattr(system, "grad_raster_time", 10e-6)),
        block_duration_raster=float(getattr(system, "block_duration_raster", 10e-6)),
        adc_raster_time=float(getattr(system, "adc_raster_time", 100e-9)),
        gamma=float(getattr(system, "gamma", 42.575575e6)),
        B0=float(getattr(system, "B0", 1.5)),
    )


def _get_adc_train_dead_time(event: Any, system: pp.Opts) -> float:
    """Return train-level ADC dead time."""

    value = _get_timing_or_attr(
        event,
        keys=("dead_time", "adc_dead_time", "frontend_dead_time"),
        default=None,
    )

    if value is not None:
        return float(value)

    return float(getattr(system, "adc_dead_time", 0.0))


def _has_adc_train_windows(event: Any) -> bool:
    """Return True if an ADC event exposes train/window information."""

    try:
        windows = _get_adc_pulseq_windows(event)
    except Exception:
        return False

    return bool(windows)


def _get_adc_pulseq_windows(event: Any) -> list[dict[str, Any]]:
    """Return Pulseq-lowerable ADC window dictionaries."""

    if hasattr(event, "to_pulseq_windows"):
        windows = event.to_pulseq_windows()
        return [dict(window) for window in windows]

    shape = getattr(event, "shape", None)

    if shape is not None and hasattr(shape, "to_pulseq_windows"):
        windows = shape.to_pulseq_windows()
        return [dict(window) for window in windows]

    windows_attr = getattr(event, "windows", None)

    if windows_attr is not None:
        windows: list[dict[str, Any]] = []

        for window in windows_attr:
            if isinstance(window, Mapping):
                windows.append(dict(window))
            elif hasattr(window, "to_pulseq_dict"):
                windows.append(dict(window.to_pulseq_dict()))
            elif hasattr(window, "to_dict"):
                windows.append(dict(window.to_dict()))
            else:
                windows.append(
                    {
                        "num_samples": getattr(window, "num_samples"),
                        "dwell": getattr(window, "dwell"),
                        "duration": getattr(window, "duration"),
                        "delay": getattr(window, "delay", 0.0),
                        "phase_offset": getattr(window, "phase_offset", 0.0),
                        "freq_offset": getattr(window, "freq_offset", 0.0),
                        "freq_ppm": getattr(window, "freq_ppm", 0.0),
                        "phase_ppm": getattr(window, "phase_ppm", 0.0),
                    }
                )

        return windows

    return []



def _adc_window_with_parent_delay(
    event: Any,
    window: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one ADC window with event-level timing applied exactly once.

    SeqStar ADC timing is hierarchical:

        event.delay + window-local delay

    ``make_adc`` and ``make_adc_train`` may keep the parent/event delay separate
    from the window table. PyPulseq accepts only one delay on the concrete ADC
    event, so the writer must combine those two timing levels during lowering.

    The synchronized gradient/ADC path performs the same addition while
    calculating segment-local delay and therefore does not call this helper.
    """

    out = dict(window)
    parent_delay = _get_event_delay(event)
    local_delay = float(out.get("delay", out.get("tstart", 0.0)) or 0.0)

    if parent_delay < -1e-12:
        raise ValueError(
            f"ADC event delay must be non-negative; got {parent_delay}."
        )
    if local_delay < -1e-12:
        raise ValueError(
            f"ADC window delay must be non-negative; got {local_delay}."
        )

    resolved_delay = max(parent_delay, 0.0) + max(local_delay, 0.0)
    out["delay"] = resolved_delay
    out["tstart"] = resolved_delay

    metadata = out.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    metadata["seqstar_parent_event_delay"] = parent_delay
    metadata["seqstar_window_local_delay"] = local_delay
    metadata["seqstar_resolved_adc_delay"] = resolved_delay
    out["metadata"] = metadata

    return out

def _adc_window_duration(window: Mapping[str, Any]) -> float:
    """Return active duration of a lowered ADC window."""

    duration = window.get("duration", window.get("adc_duration", None))

    if duration is not None:
        return float(duration)

    num_samples = window.get("num_samples", window.get("number_of_samples", None))
    dwell = window.get("dwell", window.get("sample_time", None))

    if num_samples is None or dwell is None:
        raise ValueError(f"Cannot determine ADC window duration for {window!r}.")

    return int(num_samples) * float(dwell)


def _adc_window_export_duration(window: Mapping[str, Any], system: pp.Opts) -> float:
    """Return ADC active duration after export-time dwell raster snapping."""

    num_samples = window.get("num_samples", window.get("number_of_samples", None))

    if num_samples is None:
        raise ValueError(f"ADC window {window!r} is missing num_samples.")

    num_samples = int(num_samples)

    dwell = window.get("dwell", window.get("sample_time", None))
    duration = window.get("duration", window.get("adc_duration", None))

    if dwell is None and duration is None:
        raise ValueError(f"ADC window {window!r} must define dwell or duration.")

    if dwell is None:
        dwell = float(duration) / num_samples

    dwell = _snap_time_to_raster(
        float(dwell),
        _adc_raster_time(system),
        name="simulated ADC dwell",
        mode="nearest",
        warn=False,
    )

    return num_samples * dwell


def _can_merge_adc_windows_for_continuous_export(
    event: Any,
    windows: list[dict[str, Any]],
) -> bool:
    """Return True if ADC windows can be merged into one long ADC event."""

    shape = getattr(event, "shape", None)

    if shape is not None and hasattr(shape, "can_merge_for_pulseq_continuous"):
        try:
            return bool(shape.can_merge_for_pulseq_continuous())
        except Exception:
            return False

    if len(windows) <= 1:
        return True

    sorted_windows = sorted(
        windows,
        key=lambda window: float(window.get("delay", window.get("tstart", 0.0))),
    )

    first = sorted_windows[0]
    first_dwell = float(first.get("dwell", first.get("sample_time")))
    first_freq_offset = float(first.get("freq_offset", 0.0))
    first_phase_offset = float(first.get("phase_offset", 0.0))
    first_freq_ppm = float(first.get("freq_ppm", 0.0))
    first_phase_ppm = float(first.get("phase_ppm", 0.0))

    previous_end = float(first.get("delay", first.get("tstart", 0.0))) + (
        _adc_window_duration(first)
    )

    for window in sorted_windows[1:]:
        delay = float(window.get("delay", window.get("tstart", 0.0)))

        if abs(delay - previous_end) > 1e-12:
            return False

        dwell = float(window.get("dwell", window.get("sample_time")))

        if abs(dwell - first_dwell) > 1e-12:
            return False

        if abs(float(window.get("freq_offset", 0.0)) - first_freq_offset) > 1e-12:
            return False

        if abs(float(window.get("phase_offset", 0.0)) - first_phase_offset) > 1e-12:
            return False

        if abs(float(window.get("freq_ppm", 0.0)) - first_freq_ppm) > 1e-12:
            return False

        if abs(float(window.get("phase_ppm", 0.0)) - first_phase_ppm) > 1e-12:
            return False

        previous_end = delay + _adc_window_duration(window)

    return True


def _merge_adc_windows_for_continuous_export(
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge contiguous ADC windows into one long Pulseq ADC event."""

    sorted_windows = sorted(
        windows,
        key=lambda window: float(window.get("delay", window.get("tstart", 0.0))),
    )

    first = dict(sorted_windows[0])
    dwell = float(first.get("dwell", first.get("sample_time")))
    total_num_samples = sum(
        int(window.get("num_samples", window.get("number_of_samples")))
        for window in sorted_windows
    )

    first["num_samples"] = total_num_samples
    first["dwell"] = dwell
    first.pop("duration", None)
    first.setdefault("metadata", {})
    first["metadata"]["seqstar_merged_adc_windows"] = len(sorted_windows)

    return first



def _gradient_to_pypulseq(
    event: Any,
    system: pp.Opts,
    *,
    channel: str | None = None,
    scale: float = 1.0,
) -> Any:
    """Convert a SeqStar gradient event/shape to a PyPulseq gradient.

    This function is intentionally writer-side and conservative. The SeqStar
    object model may store gradient-defining quantities either on the event,
    on ``event.shape``, in ``parameters``, in ``metadata``, or in dictionaries
    returned by ``to_dict()``. The Pulseq writer should therefore normalize the
    object at export time and then call the standard PyPulseq constructors.

    Supported cases
    ---------------
    - Trapezoids: exported through ``pp.make_trapezoid`` using exactly one of
      ``amplitude``, ``flat_area``, or ``area`` plus available timing fields.
    - Arbitrary gradients: exported through ``pp.make_arbitrary_grad`` using
      center-sampled waveform values.
    - Split gradients: a split container cannot be one Pulseq primitive. Add
      its parts as separate events/blocks, or allow the block iterator to find
      those parts before they reach this function.
    """

    if _is_split_gradient_event(event):
        raise ValueError(
            "A split-gradient container cannot be lowered to one Pulseq event. "
            "Use ppstar.split_gradient(...) / ppstar.make_split_gradient(...) and "
            "add the returned ramp_up, flat_top, and ramp_down events separately."
        )

    channel = channel or _gradient_channel(event)

    if channel is None:
        raise ValueError(f"Gradient event {event!r} is missing channel/axis.")

    if _is_arbitrary_gradient_event(event):
        return _arbitrary_gradient_to_pypulseq(
            event, system, channel=channel, scale=scale
        )

    return _trapezoid_gradient_to_pypulseq(
        event, system, channel=channel, scale=scale
    )


def _trapezoid_gradient_to_pypulseq(
    event: Any, system: pp.Opts, *, channel: str, scale: float = 1.0
) -> Any:
    """Lower a SeqStar trapezoid-like gradient to ``pp.make_trapezoid``."""

    delay = _gradient_float(
        event,
        keys=("delay", "tstart", "start", "start_s"),
        default=0.0,
    )
    delay = _snap_time_to_raster(
        delay,
        _block_duration_raster(system),
        name="gradient delay",
        mode="nearest",
    )

    amplitude = _gradient_value(event, keys=("amplitude", "amp"), default=None)
    flat_area = _gradient_value(event, keys=("flat_area",), default=None)
    area = _gradient_value(event, keys=("area",), default=None)
    if amplitude is not None: 
        amplitude = float(amplitude) * scale
    if flat_area is not None: 
        flat_area = float(flat_area) * scale
    if area is not None: 
        area = float(area) * scale

    rise_time = _gradient_value(event, keys=("rise_time", "rut"), default=None)
    flat_time = _gradient_value(event, keys=("flat_time", "ft"), default=None)
    fall_time = _gradient_value(event, keys=("fall_time", "rdt"), default=None)
    duration = _gradient_value(event, keys=("duration", "shape_duration"), default=None)

    grad_raster = float(getattr(system, "grad_raster_time", 10e-6))

    kwargs: dict[str, Any] = {
        "channel": channel,
        "delay": delay,
        "system": system,
    }

    # PyPulseq's make_trapezoid requires exactly one defining quantity from
    # amplitude / flat_area / area. Prefer amplitude when the SeqStar shape has
    # already solved the trapezoid geometry, because that preserves the exact
    # waveform generated by pypulseq_star.make_trapezoid.
    if amplitude is not None:
        kwargs["amplitude"] = float(amplitude)
    elif flat_area is not None:
        kwargs["flat_area"] = float(flat_area)
    elif area is not None:
        kwargs["area"] = float(area)
    else:
        # Last-resort fallback: integrate a waveform if the object exposes one.
        waveform = _gradient_waveform_for_pulseq(event)
        if waveform is not None:
            return _arbitrary_gradient_to_pypulseq(event, system, channel=channel)
        raise ValueError(
            "Trapezoid gradient export requires at least one of amplitude, "
            "flat_area, or area. The event/shape did not expose any of them. "
            f"event={event!r}"
        )

    if rise_time is not None:
        kwargs["rise_time"] = _snap_time_to_raster(
            float(rise_time),
            grad_raster,
            name="gradient rise_time",
            mode="nearest",
        )
    if flat_time is not None:
        kwargs["flat_time"] = _snap_time_to_raster(
            float(flat_time),
            grad_raster,
            name="gradient flat_time",
            mode="nearest",
        )
    if fall_time is not None:
        kwargs["fall_time"] = _snap_time_to_raster(
            float(fall_time),
            grad_raster,
            name="gradient fall_time",
            mode="nearest",
        )

    # If an amplitude-defined trapezoid lacks flat_time but has a duration,
    # pass duration through to PyPulseq. This mirrors PyPulseq's public API.
    if "amplitude" in kwargs and "flat_time" not in kwargs and duration is not None:
        active_duration = float(duration) - delay
        if active_duration <= 0:
            active_duration = float(duration)
        kwargs["duration"] = _snap_time_to_raster(
            active_duration,
            grad_raster,
            name="gradient duration",
            mode="nearest",
        )

    return _call_make_trapezoid_with_supported_kwargs(kwargs)


def _arbitrary_gradient_to_pypulseq(
    event: Any, system: pp.Opts, *, channel: str, scale: float = 1.0
) -> Any:
    """Lower a SeqStar arbitrary-gradient-like event to ``pp.make_arbitrary_grad``."""

    waveform = _gradient_waveform_for_pulseq(event)

    if waveform is None:
        raise ValueError(f"Arbitrary gradient event {event!r} is missing waveform/samples.")

    delay = _gradient_float(
        event,
        keys=("delay", "tstart", "start", "start_s"),
        default=0.0,
    )
    delay = _snap_time_to_raster(
        delay,
        _block_duration_raster(system),
        name="gradient delay",
        mode="nearest",
    )

    first = _gradient_value(event, keys=("first", "first_value"), default=None)
    last = _gradient_value(event, keys=("last", "last_value"), default=None)
    oversampling = bool(_gradient_value(event, keys=("oversampling",), default=False))

    kwargs: dict[str, Any] = {
        "channel": channel,
        "waveform": np.asarray(waveform, dtype=float),
        "delay": delay,
        "system": system,
    }

    if first is not None:
        kwargs["first"] = float(first)
    if last is not None:
        kwargs["last"] = float(last)
    if oversampling:
        kwargs["oversampling"] = True

    return _call_make_arbitrary_grad_with_supported_kwargs(kwargs)


def _call_make_trapezoid_with_supported_kwargs(kwargs: dict[str, Any]) -> Any:
    """Call ``pp.make_trapezoid`` while tolerating PyPulseq-version differences."""

    signature = inspect.signature(pp.make_trapezoid)
    parameters = signature.parameters

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if accepts_var_kwargs:
        return pp.make_trapezoid(**kwargs)

    supported_kwargs = {
        key: value for key, value in kwargs.items() if key in parameters
    }

    dropped = sorted(set(kwargs) - set(supported_kwargs))

    if dropped:
        warnings.warn(
            "Installed PyPulseq pp.make_trapezoid does not support these "
            f"keyword arguments and they were omitted during .seq export: {dropped}",
            stacklevel=2,
        )

    return pp.make_trapezoid(**supported_kwargs)


def _call_make_arbitrary_grad_with_supported_kwargs(kwargs: dict[str, Any]) -> Any:
    """Call ``pp.make_arbitrary_grad`` while tolerating PyPulseq-version differences."""

    signature = inspect.signature(pp.make_arbitrary_grad)
    parameters = signature.parameters

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if accepts_var_kwargs:
        return pp.make_arbitrary_grad(**kwargs)

    supported_kwargs = {
        key: value for key, value in kwargs.items() if key in parameters
    }

    dropped = sorted(set(kwargs) - set(supported_kwargs))

    if dropped:
        warnings.warn(
            "Installed PyPulseq pp.make_arbitrary_grad does not support these "
            f"keyword arguments and they were omitted during .seq export: {dropped}",
            stacklevel=2,
        )

    return pp.make_arbitrary_grad(**supported_kwargs)


def _is_split_gradient_event(event: Any) -> bool:
    """Return True if the object is a split-gradient container."""

    shape = getattr(event, "shape", None)
    class_names = " ".join(
        str(obj.__class__.__name__).lower()
        for obj in (event, shape)
        if obj is not None
    )
    kind = str(_gradient_value(event, keys=("kind", "type"), default="")).lower()

    return "split" in class_names or kind == "split"


def _is_arbitrary_gradient_event(event: Any) -> bool:
    """Return True if the object should be exported as an arbitrary gradient."""

    if _gradient_waveform_for_pulseq(event) is None:
        return False

    shape = getattr(event, "shape", None)
    class_names = " ".join(
        str(obj.__class__.__name__).lower()
        for obj in (event, shape)
        if obj is not None
    )
    kind = str(_gradient_value(event, keys=("kind", "type"), default="")).lower()

    # Trapezoids also expose a waveform for plotting. Keep them as trapezoids
    # when they expose trapezoid-defining fields, unless the class/kind clearly
    # marks the object as arbitrary/grad.
    if "trapezoid" in class_names or kind in {"trap", "trapezoid"}:
        return False

    if "arbitrary" in class_names or kind in {
        "arbitrary",
        "arbitrary_grad",
        "arbitrary_gradient",
    }:
        return True

    # Do not treat generic kind/type='grad' as arbitrary. Many SeqStar
    # gradient events use the generic physical-event type 'grad' while the
    # shape carries the actual subtype. Only fall back to arbitrary when no
    # trapezoid-defining quantity is available.
    return _gradient_value(event, keys=("amplitude", "area", "flat_area"), default=None) is None


def _gradient_waveform_for_pulseq(event: Any) -> list[float] | None:
    """Return waveform samples for arbitrary-gradient export, if available."""

    candidates = [event, getattr(event, "shape", None)]

    for candidate in candidates:
        if candidate is None:
            continue

        for attr in ("waveform", "samples", "signal"):
            if not hasattr(candidate, attr):
                continue

            value = getattr(candidate, attr)

            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue

            if value is None:
                continue

            try:
                samples = [float(x) for x in value]
            except TypeError:
                continue

            if samples:
                return samples

    return None


def _gradient_value(event: Any, *, keys: tuple[str, ...], default: Any = None) -> Any:
    """Read a gradient value from event, shape, parameters, metadata, or dicts."""

    candidates: list[Any] = [event, getattr(event, "shape", None)]

    for candidate in list(candidates):
        if candidate is None:
            continue

        # Direct attributes/properties.
        for key in keys:
            if hasattr(candidate, key):
                try:
                    value = getattr(candidate, key)
                except Exception:
                    continue
                if value is not None:
                    return value

        # Mapping-like candidate.
        if isinstance(candidate, Mapping):
            for key in keys:
                if key in candidate and candidate[key] is not None:
                    return candidate[key]

        # Common containers.
        for container_name in ("parameters", "metadata"):
            container = getattr(candidate, container_name, None)
            if isinstance(container, Mapping):
                for key in keys:
                    if key in container and container[key] is not None:
                        return container[key]
                timing = container.get("timing")
                if isinstance(timing, Mapping):
                    for key in keys:
                        if key in timing and timing[key] is not None:
                            return timing[key]

        # Serialized dictionaries from SeqStar objects often include the fields
        # even when the event wrapper itself does not.
        if hasattr(candidate, "to_dict"):
            try:
                data = candidate.to_dict()
            except Exception:
                data = None
            if isinstance(data, Mapping):
                for key in keys:
                    if key in data and data[key] is not None:
                        return data[key]
                nested_shape = data.get("shape")
                if isinstance(nested_shape, Mapping):
                    for key in keys:
                        if key in nested_shape and nested_shape[key] is not None:
                            return nested_shape[key]

    return default


def _gradient_float(event: Any, *, keys: tuple[str, ...], default: float = 0.0) -> float:
    """Read a gradient value and convert it to float."""

    value = _gradient_value(event, keys=keys, default=default)
    if value is None:
        return float(default)
    return float(value)



def _sequence_has_nested_repeated_nodes(sequence: Any) -> bool:
    registry = _sequence_node_registry(sequence)
    repeated = [
        name
        for name, record in registry.items()
        if (
            str(record.get("repeat_mode") or "").lower() == "loop"
            and (
                record.get("repeat_count") is not None
                or record.get("repeat_every") is not None
            )
        )
    ]
    return any(
        a != b and _node_is_within(b, a)
        for a in repeated
        for b in repeated
    )


def _event_variants_for_context(
    event: Any,
    *,
    sequence: Any,
    context: dict[str, int],
) -> Any:
    """Return a loop-indexed event variant, or the original event."""

    metadata = getattr(event, "metadata", None)
    variants = (
        metadata.get("seqstar_repetition_variants")
        if isinstance(metadata, Mapping)
        else None
    )
    binding = (
        metadata.get("seqstar_nested_loop_binding")
        if isinstance(metadata, Mapping)
        else None
    )

    # Backward-compatible fallback for any non-slotted custom event objects.
    if variants is None:
        variants = getattr(
            event,
            "_seqstar_repetition_variants",
            None,
        )
    if binding is None:
        binding = getattr(
            event,
            "_seqstar_nested_loop_binding",
            None,
        )

    if not isinstance(variants, list) or not variants:
        return event
    if not isinstance(binding, Mapping):
        return event

    registry = _sequence_node_registry(sequence)
    counter_names = []
    for node_name, record in sorted(
        registry.items(),
        key=lambda item: _node_depth(item[0]),
    ):
        counter = record.get("counter")
        if counter:
            counter_names.append(str(counter))

    indices = [int(context.get(name, 0)) for name in counter_names]
    if not indices:
        return variants[0]

    # Row-major flattening for arbitrary nested loop depth.
    dimensions = []
    for node_name, record in sorted(
        registry.items(),
        key=lambda item: _node_depth(item[0]),
    ):
        if not record.get("counter"):
            continue
        dimensions.append(
            _resolve_node_repeat_count(
                sequence,
                record.get("repeat_count"),
                node_name=node_name,
            )
        )

    flat_index = 0
    stride = 1
    for index, dimension in zip(reversed(indices), reversed(dimensions)):
        flat_index += index * stride
        stride *= dimension

    if flat_index < 0 or flat_index >= len(variants):
        raise IndexError(
            f"Loop variant index {flat_index} is outside "
            f"{len(variants)} attached variants."
        )
    return copy.deepcopy(variants[flat_index])


def _emit_one_block_with_context(
    *,
    sequence: Any,
    block: Any,
    context: dict[str, int],
    pulseq_seq: pp.Sequence,
    adc_train_policy: str,
    allow_multiwindow_adc_with_other_events: bool,
) -> float:
    events = [
        _event_variants_for_context(
            event,
            sequence=sequence,
            context=context,
        )
        for event in _iter_events(block)
    ]
    groups = _events_to_pulseq_event_groups(
        events,
        system=pulseq_seq.system,
        adc_train_policy=adc_train_policy,
        allow_multiwindow_adc_with_other_events=(
            allow_multiwindow_adc_with_other_events
        ),
    )

    emitted = 0.0
    for group in groups:
        group = [event for event in group if event is not None]
        for subgroup in _split_pulseq_event_group_for_channel_conflicts(group):
            if not subgroup:
                continue
            subgroup = _pad_pulseq_events_to_block_raster(
                subgroup,
                pulseq_seq.system,
            )
            emitted += _pulseq_event_group_duration(subgroup)
            pulseq_seq.add_block(*subgroup)
    return emitted


def _emit_nested_node_timeline(
    *,
    sequence: Any,
    pulseq_seq: pp.Sequence,
    adc_train_policy: str,
    allow_multiwindow_adc_with_other_events: bool,
) -> None:
    """Recursively lower nested logical Loop nodes to Pulseq blocks."""

    blocks = list(_iter_blocks(sequence))
    registry = _sequence_node_registry(sequence)
    repeated = {
        name: record
        for name, record in registry.items()
        if str(record.get("repeat_mode") or "").lower() == "loop"
    }

    def immediate_children(parent: str | None) -> list[str]:
        out = []
        for name in repeated:
            ancestors = [
                other
                for other in repeated
                if other != name and _node_is_within(name, other)
            ]
            nearest = (
                max(ancestors, key=_node_depth)
                if ancestors
                else None
            )
            if nearest == parent:
                out.append(name)
        return sorted(out, key=_node_depth)

    def belongs(block: Any, node: str) -> bool:
        return _node_is_within(_block_node_path(block), node)

    def emit_range(
        range_blocks: list[Any],
        parent: str | None,
        context: dict[str, int],
    ) -> float:
        children = immediate_children(parent)
        emitted_total = 0.0
        i = 0
        while i < len(range_blocks):
            child = next(
                (
                    name
                    for name in children
                    if belongs(range_blocks[i], name)
                ),
                None,
            )
            if child is None:
                emitted_total += _emit_one_block_with_context(
                    sequence=sequence,
                    block=range_blocks[i],
                    context=context,
                    pulseq_seq=pulseq_seq,
                    adc_train_policy=adc_train_policy,
                    allow_multiwindow_adc_with_other_events=(
                        allow_multiwindow_adc_with_other_events
                    ),
                )
                i += 1
                continue

            j = i
            while j < len(range_blocks) and belongs(range_blocks[j], child):
                j += 1

            record = repeated[child]
            count = _resolve_node_repeat_count(
                sequence,
                record.get("repeat_count"),
                node_name=child,
            )
            period = _resolve_node_repeat_period(
                sequence,
                record.get("repeat_every"),
                node_name=child,
            )
            counter = str(record.get("counter") or f"{child}_counter")

            for iteration in range(count):
                child_context = dict(context)
                child_context[counter] = iteration
                child_duration = emit_range(
                    range_blocks[i:j],
                    child,
                    child_context,
                )
                emitted_total += child_duration

                if period is not None:
                    fill = float(period) - child_duration
                    tolerance = max(
                        1e-12,
                        0.5 * _block_duration_raster(pulseq_seq.system),
                    )
                    if fill < -tolerance:
                        raise ValueError(
                            f"Nested node {child!r} emitted "
                            f"{child_duration:.12g} s, longer than its "
                            f"period {float(period):.12g} s."
                        )
                    if fill > tolerance:
                        fill = _snap_time_to_raster(
                            fill,
                            _block_duration_raster(pulseq_seq.system),
                            name=f"{child} repeat-period delay",
                            mode="nearest",
                        )
                        if fill > 0:
                            pulseq_seq.add_block(pp.make_delay(fill))
                            emitted_total += fill
            i = j
        return emitted_total

    emit_range(blocks, None, {})



def _build_node_aware_export_units(sequence: Any) -> list[dict[str, Any]]:
    """Return ordered Pulseq export units using logical-node repetition.

    Each returned unit has:

        name
            Logical repeated-node name or a synthetic block name.

        blocks
            Contiguous executable timeline blocks belonging to the motif.

        repeat_count
            Resolved integer repeat count.

        repeat_period
            Resolved period in seconds, or None.

    Repetition metadata is read from ``seq.set_node(...)`` records. Only
    outermost repeated nodes are selected as export motifs. A repeated child
    nested under an already repeated parent is preserved as metadata but is not
    independently expanded by this pass; nested train lowering requires a
    dedicated recursive/interleaved representation and must not silently
    duplicate blocks.

    The selected motif's blocks must be contiguous in timeline order. This
    protects executable semantics and prevents a logical label from gathering
    separated blocks into a reordered Pulseq motif.
    """

    blocks = list(_iter_blocks(sequence))
    if not blocks:
        return []

    node_registry = _sequence_node_registry(sequence)
    repeated_nodes = {
        name: record
        for name, record in node_registry.items()
        if (
            record.get("repeat_count") is not None
            or record.get("repeat_every") is not None
        )
    }

    if not repeated_nodes:
        return [
            {
                "name": _block_export_name(block, index),
                "blocks": [block],
                "repeat_count": 1,
                "repeat_period": None,
            }
            for index, block in enumerate(blocks)
        ]

    # Select only outermost repeated nodes. A nested repeated node is already
    # contained in its repeated parent's executable motif.
    selected_nodes: dict[str, dict[str, Any]] = {}
    for name, record in sorted(
        repeated_nodes.items(),
        key=lambda item: (_node_depth(item[0]), item[0]),
    ):
        if any(_node_is_within(name, parent) for parent in selected_nodes):
            continue
        selected_nodes[name] = record

    block_owner: list[str | None] = []
    for block in blocks:
        block_node = _block_node_path(block)
        owner = None

        matching_nodes = [
            node_name
            for node_name in selected_nodes
            if _node_is_within(block_node, node_name)
        ]

        if matching_nodes:
            # Outermost selection means there should normally be one match.
            # Use the deepest defensively.
            owner = max(matching_nodes, key=_node_depth)

        block_owner.append(owner)

    # Validate that each repeated motif is represented by one contiguous range.
    for node_name in selected_nodes:
        indices = [
            index
            for index, owner in enumerate(block_owner)
            if owner == node_name
        ]

        if not indices:
            continue

        expected = list(range(indices[0], indices[-1] + 1))
        if indices != expected:
            raise ValueError(
                f"Repeated logical node {node_name!r} does not occupy one "
                "contiguous timeline range. Pulseq export will not reorder "
                "blocks to manufacture a motif. Keep all blocks belonging to "
                "a repeated node contiguous, or expand the repetitions "
                "explicitly in seq.add_block(...)."
            )

    units: list[dict[str, Any]] = []
    index = 0

    while index < len(blocks):
        owner = block_owner[index]

        if owner is None:
            units.append(
                {
                    "name": _block_export_name(blocks[index], index),
                    "blocks": [blocks[index]],
                    "repeat_count": 1,
                    "repeat_period": None,
                }
            )
            index += 1
            continue

        end = index + 1
        while end < len(blocks) and block_owner[end] == owner:
            end += 1

        record = selected_nodes[owner]
        repeat_mode = str(record.get("repeat_mode") or "").strip().lower()
        if repeat_mode not in {"loop", "expanded"}:
            raise ValueError(
                f"Repeated node {owner!r} requires repeat_mode='loop' or 'expanded'."
            )
        if repeat_mode == "expanded":
            repeat_count = 1
            repeat_period = None
        else:
            repeat_count = _resolve_node_repeat_count(
                sequence, record.get("repeat_count"), node_name=owner
            )
            repeat_period = _resolve_node_repeat_period(
                sequence, record.get("repeat_every"), node_name=owner
            )

        units.append(
            {
                "name": owner,
                "blocks": blocks[index:end],
                "repeat_count": repeat_count,
                "repeat_period": repeat_period,
                "repeat_mode": repeat_mode,
            }
        )
        index = end

    return units


def _sequence_node_registry(sequence: Any) -> dict[str, dict[str, Any]]:
    """Return a normalized logical-node registry."""

    candidates = [
        getattr(sequence, "nodes", None),
        getattr(getattr(sequence, "timeline", None), "nodes", None),
        getattr(getattr(sequence, "timeline", None), "node_registry", None),
    ]

    metadata = getattr(sequence, "metadata", None)
    if isinstance(metadata, Mapping):
        candidates.extend(
            [
                metadata.get("seqstar_nodes"),
                metadata.get("nodes"),
                metadata.get("node_registry"),
            ]
        )

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue

        normalized: dict[str, dict[str, Any]] = {}
        for name, record in candidate.items():
            if isinstance(record, Mapping):
                normalized[str(name)] = dict(record)
                continue

            normalized[str(name)] = {
                "name": getattr(record, "name", str(name)),
                "parent": getattr(record, "parent", None),
                "repeat_count": getattr(record, "repeat_count", None),
                "repeat_every": getattr(record, "repeat_every", None),
                "counter": getattr(record, "counter", None),
                "role": getattr(record, "role", None),
            }

        return normalized

    return {}


def _resolve_node_repeat_count(
    sequence: Any,
    value: Any,
    *,
    node_name: str,
) -> int:
    """Resolve a literal or protocol-referenced node repeat count."""

    if value is None:
        return 1

    resolved = _resolve_sequence_parameter_reference(
        sequence,
        value,
        field_name="repeat_count",
        node_name=node_name,
    )

    try:
        numeric = float(resolved)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Node {node_name!r} repeat_count={value!r} did not resolve "
            "to a numeric value."
        ) from exc

    integer = int(round(numeric))
    if abs(numeric - integer) > 1e-9:
        raise ValueError(
            f"Node {node_name!r} repeat_count={value!r} resolved to "
            f"non-integer value {numeric!r}."
        )

    if integer < 1:
        raise ValueError(
            f"Node {node_name!r} repeat_count must be >= 1; got {integer}."
        )

    return integer


def _resolve_node_repeat_period(
    sequence: Any,
    value: Any,
    *,
    node_name: str,
) -> float | None:
    """Resolve a literal or protocol-referenced node repeat period."""

    if value is None:
        return None

    resolved = _resolve_sequence_parameter_reference(
        sequence,
        value,
        field_name="repeat_every",
        node_name=node_name,
    )

    try:
        period = float(resolved)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Node {node_name!r} repeat_every={value!r} did not resolve "
            "to a numeric period in seconds."
        ) from exc

    if period <= 0:
        raise ValueError(
            f"Node {node_name!r} repeat_every must be > 0; got {period}."
        )

    return period


def _resolve_sequence_parameter_reference(
    sequence: Any,
    value: Any,
    *,
    field_name: str,
    node_name: str,
) -> Any:
    """Resolve a literal or sequence protocol-parameter reference.

    Lookup is explicit and sequence-family agnostic. Node metadata may refer to
    any user-defined protocol parameter by name, for example:

        repeat_count="averages"
        repeat_every="TR"
        repeat_count="number_of_lines"

    The writer does not assign meaning to those names; it only resolves the
    exact reference supplied by the developer.
    """

    if not isinstance(value, str):
        return value

    parameter_name = value
    aliases = {
        "TR": ("TR", "repetition_time"),
        "repetition_time": ("repetition_time", "TR"),
        "n_slices": ("n_slices", "num_slices"),
        "num_slices": ("num_slices", "n_slices"),
    }
    parameter_candidates = aliases.get(
        parameter_name,
        (parameter_name,),
    )

    # 1. Canonical sequence parameters.
    parameters = getattr(sequence, "parameters", None)

    if isinstance(parameters, Mapping):
        for candidate in parameter_candidates:
            if candidate in parameters:
                return parameters[candidate]

    # 2. Optional protocol object attached to the sequence.
    protocol = getattr(sequence, "protocol", None)

    if protocol is not None:
        protocol_parameters = getattr(protocol, "parameters", None)

        if isinstance(protocol_parameters, Mapping):
            for candidate in parameter_candidates:
                if candidate in protocol_parameters:
                    return protocol_parameters[candidate]

        get_parameter = getattr(protocol, "get_parameter", None)

        if callable(get_parameter):
            for candidate in parameter_candidates:
                try:
                    resolved = get_parameter(candidate)
                except (KeyError, TypeError, ValueError):
                    resolved = None
                if resolved is not None:
                    return resolved

    # 3. Stable metadata mirrors.
    metadata = getattr(sequence, "metadata", None)

    if isinstance(metadata, Mapping):
        for container_key in (
            "parameters",
            "protocol_parameters",
            "protocol",
            "definitions",
        ):
            container = metadata.get(container_key)

            if isinstance(container, Mapping):
                for candidate in parameter_candidates:
                    if candidate in container:
                        return container[candidate]

    # 4. Timeline metadata, if the sequence wrapper stores protocol context
    # there.
    timeline = getattr(sequence, "timeline", None)
    timeline_metadata = getattr(timeline, "metadata", None)

    if isinstance(timeline_metadata, Mapping):
        for container_key in (
            "parameters",
            "protocol_parameters",
            "protocol",
            "definitions",
        ):
            container = timeline_metadata.get(container_key)

            if isinstance(container, Mapping):
                for candidate in parameter_candidates:
                    if candidate in container:
                        return container[candidate]

    # 5. Numeric strings remain valid literals.
    try:
        return float(parameter_name)
    except ValueError:
        pass

    available_keys: set[str] = set()

    if isinstance(parameters, Mapping):
        available_keys.update(str(key) for key in parameters)

    if isinstance(metadata, Mapping):
        for container_key in (
            "parameters",
            "protocol_parameters",
            "protocol",
            "definitions",
        ):
            container = metadata.get(container_key)

            if isinstance(container, Mapping):
                available_keys.update(str(key) for key in container)

    available_text = (
        ", ".join(sorted(available_keys))
        if available_keys
        else "<none>"
    )

    raise KeyError(
        f"Node {node_name!r} {field_name} references protocol parameter "
        f"{parameter_name!r}, but that parameter was not found on the "
        f"sequence. Available parameters: {available_text}."
    )

def _block_node_path(block: Any) -> str:
    """Return the logical node path attached by seq.add_block(...)."""

    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in (
            "seqstar_node",
            "seqstar_path",
            "path",
            "node",
        ):
            value = metadata.get(key)
            if value is not None:
                return str(value)

    parameters = getattr(block, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in (
            "seqstar_node",
            "seqstar_path",
            "path",
            "node",
        ):
            value = parameters.get(key)
            if value is not None:
                return str(value)

    for attr in ("node", "path"):
        value = getattr(block, attr, None)
        if value is not None:
            return str(value)

    return ""


def _node_is_within(node_path: str, ancestor_path: str) -> bool:
    """Return True when node_path is ancestor_path or one of its descendants."""

    node_path = str(node_path or "").strip(".")
    ancestor_path = str(ancestor_path or "").strip(".")

    if not node_path or not ancestor_path:
        return False

    return (
        node_path == ancestor_path
        or node_path.startswith(f"{ancestor_path}.")
    )


def _node_depth(node_path: str) -> int:
    """Return dotted logical-node depth."""

    node_path = str(node_path or "").strip(".")
    if not node_path:
        return 0
    return len(node_path.split("."))


def _block_export_name(block: Any, index: int) -> str:
    """Return a stable debug name for an ungrouped export block."""

    node = _block_node_path(block)
    if node:
        return node

    name = getattr(block, "name", None)
    if name:
        return str(name)

    return f"block_{index}"

def _iter_blocks(sequence: Any) -> Iterable[Any]:
    """Iterate over blocks in a SeqStarSequence-like object."""

    if hasattr(sequence, "timeline") and hasattr(sequence.timeline, "blocks"):
        blocks = sequence.timeline.blocks
        if isinstance(blocks, Mapping):
            yield from blocks.values()
        else:
            yield from blocks
        return

    if hasattr(sequence, "blocks"):
        blocks = sequence.blocks
        if isinstance(blocks, Mapping):
            yield from blocks.values()
        else:
            yield from blocks
        return

    raise TypeError("Sequence does not expose timeline.blocks or blocks.")


def _iter_events(block: Any) -> Iterable[Any]:
    """Iterate over renderable/writable events in a SeqStar block."""

    visited: set[int] = set()
    yield from _iter_node_events(block, visited=visited, is_root=True)


def _iter_node_events(
    node: Any,
    *,
    visited: set[int],
    is_root: bool = False,
) -> Iterable[Any]:
    """Recursively yield writable events from a node."""

    if node is None or isinstance(node, (float, int, str, bool)):
        return

    node_id = id(node)
    if node_id in visited:
        return
    visited.add(node_id)

    if not is_root and _looks_writable_event(node):
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


def _looks_writable_event(obj: Any) -> bool:
    """Return True if object looks like an RF/ADC/gradient/delay event."""

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
        )
    ):
        return False

    return (
        _is_delay_event(obj)
        or _is_adc_event(obj)
        or _is_gradient_event(obj)
        or _is_rf_event(obj)
    )

def _is_rf_event(event: Any) -> bool:
    """Return True if event is an RF event.

    This intentionally avoids loose matching such as ``"rf" in name`` or
    treating ``role="excitation"`` alone as RF. Slice-select gradients may share
    excitation/rephasing roles, so RF classification must be based on RF-specific
    identity or RF-specific fields.
    """

    event_type = str(getattr(event, "event_type", "")).lower()
    type_value = str(getattr(event, "type", "")).lower()
    kind = str(getattr(event, "kind", "")).lower()
    use = str(getattr(event, "use", "") or "").lower()
    class_name = event.__class__.__name__.lower()

    if event_type in {"rf", "radiofrequency"}:
        return True

    if type_value in {"rf", "radiofrequency"}:
        return True

    if kind in {"rf", "radiofrequency", "rf_block", "rf_sinc", "rf_gauss"}:
        return True

    if (
        "rfevent" in class_name
        or "rfblock" in class_name
        or "rfpulse" in class_name
        or class_name.startswith("seqstarrf")
    ):
        return True

    if hasattr(event, "flip_angle"):
        return True

    # ``use`` alone is not enough, but RF events usually also expose RF phase.
    if use in {"excitation", "excite", "refocusing", "refocus", "inversion", "invert"}:
        return hasattr(event, "duration") and hasattr(event, "phase_offset")

    return False

def _is_adc_event(event: Any) -> bool:
    """Return True only for explicitly ADC/readout-like events.

    RF identity takes precedence. A generic ``num_samples`` attribute alone is
    insufficient because enriched event containers may expose shared protocol
    fields that do not belong to the concrete event.
    """

    if _is_rf_event(event):
        return False

    event_type = str(getattr(event, "event_type", "") or "").lower()
    type_value = str(getattr(event, "type", "") or "").lower()
    kind = str(getattr(event, "kind", "") or "").lower()
    class_name = event.__class__.__name__.lower()
    name = str(getattr(event, "name", "") or "").lower()

    if event_type == "adc":
        return True

    if type_value == "adc":
        return True

    if kind.startswith("adc"):
        return True

    if "adcevent" in class_name or "adctrain" in class_name:
        return True

    if hasattr(event, "to_pulseq_windows"):
        return True

    if hasattr(event, "windows") and "adc" in name:
        return True

    # Structural fallback for custom ADC event classes. Both sampling fields
    # must be concrete attributes on the event itself.
    return (
        getattr(event, "num_samples", None) is not None
        and getattr(event, "dwell", None) is not None
        and not hasattr(event, "flip_angle")
    )

def _is_gradient_event(event: Any) -> bool:
    """Return True if event is a gradient event."""

    event_type = str(getattr(event, "event_type", "")).lower()
    type_value = str(getattr(event, "type", "")).lower()
    kind = str(getattr(event, "kind", "")).lower()
    class_name = event.__class__.__name__.lower()

    if event_type in {"grad", "gradient", "trap", "trapezoid"}:
        return True

    if type_value in {"grad", "gradient", "trap", "trapezoid"}:
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
        "gradientevent" in class_name
        or "gradientshape" in class_name
        or "trapezoid" in class_name
        or "splitgradient" in class_name
    ):
        return True

    if hasattr(event, "channel") and (
        hasattr(event, "area")
        or hasattr(event, "amplitude")
        or hasattr(event, "flat_area")
        or all(hasattr(event, attr) for attr in ("rise_time", "flat_time", "fall_time"))
    ):
        return True

    if hasattr(event, "axis") and (
        hasattr(event, "area")
        or hasattr(event, "amplitude")
        or hasattr(event, "flat_area")
    ):
        return True

    return False

def _is_delay_event(event: Any) -> bool:
    """Return True if event is a delay-only event."""

    event_type = str(getattr(event, "event_type", "")).lower()
    type_value = str(getattr(event, "type", "")).lower()
    kind = str(getattr(event, "kind", "")).lower()
    class_name = event.__class__.__name__.lower()
    name = str(getattr(event, "name", "") or "").lower()

    return (
        event_type == "delay"
        or type_value == "delay"
        or kind == "delay"
        or "delayevent" in class_name
        or name == "delay"
        or name.endswith("_delay")
    )

def _get_event_delay(event: Any) -> float:
    """Return event delay/start in seconds."""

    return float(
        _get_timing_or_attr(
            event,
            keys=("delay", "tstart", "start", "start_s"),
            default=0.0,
        )
    )


def _get_event_duration(event: Any) -> float:
    """Return generic event duration in seconds."""

    duration = _get_timing_or_attr(
        event,
        keys=("duration", "shape_duration"),
        default=None,
    )

    if duration is not None:
        return float(duration)

    shape = getattr(event, "shape", None)
    if shape is not None and hasattr(shape, "duration"):
        return float(shape.duration)

    raise ValueError(f"Could not determine duration for event {event!r}")


def _get_event_active_duration(event: Any) -> float:
    """Return active duration after explicit event-family classification.

    Event objects may carry a shared protocol dictionary. ADC-only fields such
    as ``num_samples`` and ``dwell`` must never determine an RF, gradient,
    trigger, or delay duration merely because those fields exist in the shared
    protocol.

    The dispatch is sequence-family agnostic and relies only on the concrete
    event family.
    """

    if _is_delay_event(event):
        return _get_event_duration(event)

    if _is_adc_event(event):
        num_samples = _get_timing_or_attr(
            event,
            keys=("num_samples", "samples", "n_samples"),
            default=None,
        )
        dwell = _get_timing_or_attr(
            event,
            keys=("dwell", "dwell_time", "sample_time"),
            default=None,
        )

        if num_samples is not None and dwell is not None:
            return float(num_samples) * float(dwell)

        duration = _get_timing_or_attr(
            event,
            keys=("duration", "adc_duration", "readout_duration"),
            default=None,
        )
        return float(duration or 0.0)

    if _is_gradient_event(event):
        rise_time = _get_timing_or_attr(
            event,
            keys=("rise_time",),
            default=None,
        )
        flat_time = _get_timing_or_attr(
            event,
            keys=("flat_time",),
            default=None,
        )
        fall_time = _get_timing_or_attr(
            event,
            keys=("fall_time",),
            default=None,
        )

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

    # RF and other event families use explicit event/shape duration. Do not
    # inspect ADC-specific protocol keys in this path.
    duration = getattr(event, "duration", None)
    if duration is not None:
        return float(duration)

    shape = getattr(event, "shape", None)
    if shape is not None:
        shape_duration = getattr(shape, "duration", None)
        if shape_duration is not None:
            return float(shape_duration)

    timing = getattr(event, "timing", None)
    if timing is not None:
        for key in ("duration", "shape_duration", "active_duration"):
            value = getattr(timing, key, None)
            if value is not None:
                return float(value)

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in ("duration", "shape_duration", "active_duration"):
            value = parameters.get(key)
            if value is not None:
                return float(value)

    return _get_event_duration(event)


def _get_timing_or_attr(
    obj: Any,
    *,
    keys: tuple[str, ...],
    default: Any = None,
) -> Any:
    """Read value from attr, parameters, or metadata.timing."""

    for key in keys:
        if hasattr(obj, key) and getattr(obj, key) is not None:
            return getattr(obj, key)

    parameters = getattr(obj, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in keys:
            if key in parameters and parameters[key] is not None:
                return parameters[key]

    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, Mapping):
        timing = metadata.get("timing")
        if isinstance(timing, Mapping):
            for key in keys:
                if key in timing and timing[key] is not None:
                    return timing[key]

        for key in keys:
            if key in metadata and metadata[key] is not None:
                return metadata[key]

    return default


def _get_int_from_parameters(
    obj: Any,
    *,
    keys: tuple[str, ...],
    default: int,
) -> int:
    value = _get_from_parameters(obj, keys=keys, default=default)
    return int(value)


def _get_float_from_parameters(
    obj: Any,
    *,
    keys: tuple[str, ...],
    default: float | None,
) -> float | None:
    value = _get_from_parameters(obj, keys=keys, default=default)
    if value is None:
        return None
    return float(value)


def _get_from_parameters(obj: Any, *, keys: tuple[str, ...], default: Any) -> Any:
    if hasattr(obj, "parameters") and isinstance(obj.parameters, Mapping):
        for key in keys:
            if key in obj.parameters:
                return obj.parameters[key]
    return default


def _block_duration_raster(system: pp.Opts) -> float:
    """Return Pulseq block-duration raster."""

    return float(getattr(system, "block_duration_raster", 10e-6))


def _adc_raster_time(system: pp.Opts) -> float:
    """Return Pulseq ADC raster time."""

    return float(getattr(system, "adc_raster_time", 100e-9))


def _snap_time_to_raster(
    value: float,
    raster: float,
    *,
    name: str,
    mode: str = "nearest",
    warn: bool = True,
) -> float:
    """Snap a non-negative time value to a raster.

    Parameters
    ----------
    value
        Time in seconds.

    raster
        Raster in seconds.

    name
        Human-readable name for warning/error messages.

    mode
        ``"nearest"``, ``"ceil"``, or ``"floor"``.

    warn
        If True, warn when snapping changes the value beyond numerical noise.
    """

    value = float(value)
    raster = float(raster)

    if raster <= 0:
        raise ValueError(f"{name}: raster must be positive. Passed: {raster}")

    if value < 0:
        raise ValueError(f"{name}: time must be non-negative. Passed: {value}")

    scaled = value / raster

    if mode == "nearest":
        snapped = round(scaled) * raster
    elif mode == "ceil":
        snapped = math.ceil(scaled - 1e-12) * raster
    elif mode == "floor":
        snapped = math.floor(scaled + 1e-12) * raster
    else:
        raise ValueError(f"Unsupported raster snap mode: {mode!r}")

    snapped = max(snapped, 0.0)

    if warn and abs(snapped - value) > max(1e-12, 1e-6 * raster):
        warnings.warn(
            f"Snapped {name} from {value:.12g} s to {snapped:.12g} s "
            f"to satisfy raster {raster:.12g} s.",
            stacklevel=2,
        )

    return snapped


def _ceil_time_to_raster(value: float, raster: float) -> float:
    """Ceil a non-negative time to a raster."""

    return _snap_time_to_raster(
        value,
        raster,
        name="block duration",
        mode="ceil",
    )


def _ceil_time_to_raster_no_warn(value: float, raster: float) -> float:
    """Ceil a non-negative time to a raster without warning."""

    return _snap_time_to_raster(
        value,
        raster,
        name="block duration",
        mode="ceil",
        warn=False,
    )


def _pulseq_event_group_duration(events: list[Any]) -> float:
    """Return duration of a PyPulseq event group."""

    if not events:
        return 0.0

    try:
        return float(pp.calc_duration(*events))
    except Exception:
        pass

    durations = [_pulseq_event_duration_fallback(event) for event in events]
    return max(durations, default=0.0)


def _pulseq_event_duration_fallback(event: Any) -> float:
    """Fallback duration extraction for PyPulseq events."""

    if event is None:
        return 0.0

    if hasattr(event, "delay") and hasattr(event, "num_samples") and hasattr(event, "dwell"):
        return float(event.delay) + int(event.num_samples) * float(event.dwell)

    if hasattr(event, "delay") and hasattr(event, "duration"):
        return float(event.delay) + float(event.duration)

    if hasattr(event, "duration"):
        return float(event.duration)

    if all(hasattr(event, attr) for attr in ("delay", "rise_time", "flat_time", "fall_time")):
        return (
            float(event.delay)
            + float(event.rise_time)
            + float(event.flat_time)
            + float(event.fall_time)
        )

    return 0.0


def _pad_pulseq_events_to_block_raster(
    events: list[Any],
    system: pp.Opts,
) -> list[Any]:
    """Pad a PyPulseq event group so its block duration is raster-compliant.

    Pulseq blocks must have durations that land on block_duration_raster. ADC
    active durations can be legal on adc_raster_time but still not legal as a
    block duration. The safe export-time fix is to add a simultaneous delay
    event with duration equal to the padded block duration.
    """

    if not events:
        return events

    block_raster = _block_duration_raster(system)
    duration = _pulseq_event_group_duration(events)
    padded_duration = _ceil_time_to_raster(duration, block_raster)

    if padded_duration <= duration + 1e-12:
        return events

    padded_events = list(events)
    padded_events.append(pp.make_delay(padded_duration))

    return padded_events


def _gradient_channel(event: Any) -> str | None:
    """Return PyPulseq gradient channel x/y/z."""

    value = _get_timing_or_attr(
        event,
        keys=("channel", "axis"),
        default=None,
    )

    if value is None:
        name = str(getattr(event, "name", "")).lower()
        for axis in ("x", "y", "z"):
            if name.endswith(axis) or name in {
                f"g{axis}",
                f"grad_{axis}",
                f"gradient_{axis}",
                f"gradient{axis}",
            }:
                return axis
        return None

    value = str(value).lower()

    if value in {"x", "gx", "read", "readout", "ro"}:
        return "x"
    if value in {"y", "gy", "phase", "phase_encode", "pe"}:
        return "y"
    if value in {"z", "gz", "slice", "slice_select", "ss"}:
        return "z"

    return None