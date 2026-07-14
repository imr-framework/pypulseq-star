"""Duration calculation utilities for pypulseq_star.

This module intentionally keeps the PyPulseq-compatible filename:

    pypulseq_star.calc_duration

so users can transition from:

    from pypulseq.calc_duration import calc_duration

to:

    from pypulseq_star.calc_duration import calc_duration

The implementation is SeqStar-aware and understands enriched event objects,
shape objects, event.parameters dictionaries, and PyPulseq-like namespaces.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def calc_duration(*objects: Any) -> float:
    """Return the duration in seconds of one or more objects.

    If multiple objects are supplied, they are interpreted as simultaneous
    events in one block, so the returned duration is the maximum occupied
    duration.

    Supported object styles
    -----------------------
    - SeqStarBlock with ``events``
    - SeqStarNode with ``children``
    - SeqStarEvent with ``duration`` and/or ``shape.duration``
    - PyPulseq-like objects with ``delay``, ``t``, ``dwell``, ``num_samples``
    - Gradient trapezoids with ``rise_time``, ``flat_time``, ``fall_time``
    - Objects using ``parameters`` for delay/dead-time/post-time metadata
    """

    if not objects:
        return 0.0

    durations = [_duration_of_object(obj) for obj in objects if obj is not None]

    if not durations:
        return 0.0

    return max(durations)


def _duration_of_object(obj: Any) -> float:
    """Return occupied duration of one object in seconds.

    Explicit event identity takes precedence over generic container markers.
    Enriched SeqStar events may expose ``children=[]`` for graph compatibility;
    that must not cause an RF, ADC, gradient, trigger, or delay event to be
    interpreted as a block.
    """

    if obj is None:
        return 0.0

    if isinstance(obj, (float, int)):
        return float(obj)

    # An explicit event container is authoritative block identity. SeqStarBlock
    # inherits generic node fields such as role, duration, and children, so a
    # broad event-marker test must not run before this check.
    if _has_explicit_event_container(obj):
        return _duration_of_block(obj)

    if _is_event_like(obj):
        return _duration_of_event(obj)

    if _is_block_like(obj):
        return _duration_of_block(obj)

    return _duration_of_event(obj)


def _duration_of_block(block: Any) -> float:
    """Return duration of a block from its contained events."""

    explicit_duration = _get_optional_float(block, "duration")

    event_durations = [
        _duration_of_event(event)
        for event in _iter_events(block)
        if event is not None
    ]

    content_duration = max(event_durations, default=0.0)

    if explicit_duration is not None:
        return max(explicit_duration, content_duration)

    return content_duration


def _duration_of_event(event: Any) -> float:
    """Return occupied duration of an event.

    Occupied duration is:

        local_start + active_duration + post_dead_time

    where:
        local_start may be delay/tstart/start_s
        active_duration is event/shape/waveform/ADC/trapezoid duration
        post_dead_time is RF ringdown, ADC dead time, or generic post time

    This makes RF, ADC, and gradients follow the same timing model.
    """

    # SeqStar delay events represent a structural wait. Their constructor may
    # expose both ``delay`` and ``duration`` as compatibility aliases for the
    # same wait length. Summing them would double every explicit TE/TR fill.
    if _is_delay_like(event):
        duration = _get_optional_float(event, "duration")
        if duration is None:
            duration = _get_parameter_float(event, "duration")
        if duration is None:
            duration = _get_metadata_timing_float(event, "duration")
        if duration is None:
            duration = _get_optional_float(event, "delay")
        if duration is None:
            duration = _get_parameter_float(event, "delay")
        if duration is None:
            duration = _get_metadata_timing_float(event, "delay")
        return float(duration or 0.0)

    # Prefer an explicitly materialized occupied extent when the constructor
    # provides one. This is authoritative and avoids reconstructing protected
    # timing from attributes that may be stored only in parameters or metadata.
    #
    # Examples include:
    #   RF: local delay + active waveform + ringdown
    #   ADC: local delay + acquisition extent + protected post-time
    #   custom events: any constructor-defined occupied interval
    #
    # This rule is sequence-family agnostic and event-family agnostic.
    occupied_duration = _get_optional_float(event, "occupied_duration")
    if occupied_duration is None:
        occupied_duration = _get_parameter_float(event, "occupied_duration")
    if occupied_duration is None:
        occupied_duration = _get_metadata_timing_float(
            event,
            "occupied_duration",
        )

    if occupied_duration is not None:
        return occupied_duration

    delay = _get_event_delay(event)
    active_duration = _get_event_active_duration(event)
    post_time = _get_event_post_time(event)

    return delay + active_duration + post_time


def _get_event_delay(event: Any) -> float:
    """Return local event start/delay in seconds."""

    # Attribute wins because it is closest to PyPulseq behavior.
    for attr in ("delay", "tstart", "start", "start_s"):
        value = _get_optional_float(event, attr)
        if value is not None:
            return value

    # Then parameters.
    for key in ("delay", "tstart", "start", "start_s"):
        value = _get_parameter_float(event, key)
        if value is not None:
            return value

    # Then metadata.timing.
    for key in ("delay", "tstart", "start", "start_s"):
        value = _get_metadata_timing_float(event, key)
        if value is not None:
            return value

    return 0.0


def _get_event_active_duration(event: Any) -> float:
    """Return active duration without cross-family protocol leakage.

    Family-specific fields from ``event.parameters`` are read only after the
    object has been classified as that family. This is essential when events
    carry a shared protocol dictionary containing RF, ADC, and gradient
    parameters.
    """

    if _is_adc_like(event):
        num_samples = _get_optional_float(event, "num_samples")
        dwell = _get_optional_float(event, "dwell")

        if num_samples is None:
            num_samples = _get_parameter_float(event, "num_samples")
        if dwell is None:
            dwell = _get_parameter_float(event, "dwell")

        if num_samples is not None and dwell is not None:
            return num_samples * dwell

        for key in ("duration", "adc_duration", "readout_duration"):
            duration = _get_optional_float(event, key)
            if duration is None:
                duration = _get_parameter_float(event, key)
            if duration is not None:
                return duration

        return 0.0

    if _is_gradient_like(event):
        rise_time = _get_optional_float(event, "rise_time")
        flat_time = _get_optional_float(event, "flat_time")
        fall_time = _get_optional_float(event, "fall_time")

        if rise_time is None:
            rise_time = _get_parameter_float(event, "rise_time")
        if flat_time is None:
            flat_time = _get_parameter_float(event, "flat_time")
        if fall_time is None:
            fall_time = _get_parameter_float(event, "fall_time")

        if (
            rise_time is not None
            and flat_time is not None
            and fall_time is not None
        ):
            return rise_time + flat_time + fall_time

    # PyPulseq RF-like t array.
    if _is_rf_like(event):
        t = getattr(event, "t", None)
        if t is not None:
            try:
                return float(t[-1])
            except (TypeError, IndexError, ValueError):
                pass

    # Explicit event duration. This is the primary path for RF, delay, trigger,
    # and future event families.
    duration = _get_optional_float(event, "duration")
    if duration is not None:
        return duration

    # Family-appropriate parameter fallback.
    if _is_rf_like(event):
        for key in ("duration", "rf_duration", "shape_duration"):
            duration = _get_parameter_float(event, key)
            if duration is not None:
                return duration
    elif _is_gradient_like(event):
        duration = _get_parameter_float(event, "duration")
        if duration is not None:
            return duration
    else:
        duration = _get_parameter_float(event, "duration")
        if duration is not None:
            return duration

    shape = getattr(event, "shape", None)
    if shape is not None:
        shape_duration = _get_optional_float(shape, "duration")
        if shape_duration is not None:
            return shape_duration

        shape_duration = _get_parameter_float(shape, "duration")
        if shape_duration is not None:
            return shape_duration

    duration = _get_metadata_timing_float(event, "duration")
    if duration is not None:
        return duration

    shape_duration = _get_metadata_timing_float(event, "shape_duration")
    if shape_duration is not None:
        return shape_duration

    return 0.0


def _get_event_post_time(event: Any) -> float:
    """Return family-appropriate post-event protected time.

    Shared protocol dictionaries may contain both RF ringdown and ADC dead
    time. Select the post time only after classifying the event family.
    """

    # Generic explicit post-event fields are safe for every family.
    for key in (
        "post_time",
        "post_delay",
        "post_dead_time",
        "dead_time_after",
    ):
        value = _get_optional_float(event, key)
        if value is not None:
            return value
        value = _get_parameter_float(event, key)
        if value is not None:
            return value
        value = _get_metadata_timing_float(event, key)
        if value is not None:
            return value

    if _is_rf_like(event):
        for key in ("ringdown_time", "rf_ringdown_time"):
            value = _get_optional_float(event, key)
            if value is not None:
                return value
            value = _get_parameter_float(event, key)
            if value is not None:
                return value
            value = _get_metadata_timing_float(event, key)
            if value is not None:
                return value
        return 0.0

    if _is_adc_like(event):
        for key in ("dead_time", "adc_dead_time"):
            value = _get_optional_float(event, key)
            if value is not None:
                return value
            value = _get_parameter_float(event, key)
            if value is not None:
                return value
            value = _get_metadata_timing_float(event, key)
            if value is not None:
                return value
        return 0.0

    return 0.0


def _iter_events(block: Any) -> Iterable[Any]:
    """Yield events from a block-like object."""

    if hasattr(block, "events"):
        events = getattr(block, "events")

        if isinstance(events, Mapping):
            yield from events.values()
            return

        try:
            yield from events
            return
        except TypeError:
            pass

    if hasattr(block, "children"):
        children = getattr(block, "children")

        if isinstance(children, Mapping):
            yield from children.values()
            return

        try:
            yield from children
            return
        except TypeError:
            pass

    # PyPulseq block-like namespace.
    if hasattr(block, "__dict__"):
        for key, value in vars(block).items():
            if key.startswith("_"):
                continue
            if value is None or isinstance(value, (float, int, str, bool)):
                continue

            if isinstance(value, Mapping):
                yield from value.values()
            elif isinstance(value, (list, tuple)):
                yield from value
            else:
                yield value


def _has_explicit_event_container(obj: Any) -> bool:
    """Return True when an object explicitly owns a block event collection."""

    if not hasattr(obj, "events"):
        return False

    events = getattr(obj, "events", None)
    return isinstance(events, (list, tuple, dict)) or hasattr(events, "__iter__")


def _is_block_like(obj: Any) -> bool:
    """Return True if object should be interpreted as a block/container.

    Explicit ``events`` ownership is authoritative. Generic graph fields such
    as ``children``, ``role``, and ``duration`` are not sufficient because they
    are also present on enriched physical events.
    """

    if _has_explicit_event_container(obj):
        return True

    if _is_event_like(obj):
        return False

    children = getattr(obj, "children", None)
    if isinstance(children, (list, tuple, dict)) and len(children) > 0:
        return True

    # PyPulseq-style block namespace with event fields.
    if hasattr(obj, "__dict__"):
        values = [
            value
            for value in vars(obj).values()
            if value is not None and not isinstance(value, (float, int, str, bool))
        ]
        return bool(values)

    return False


def _is_event_like(obj: Any) -> bool:
    """Return True when an object has concrete physical-event identity.

    Do not classify from generic node fields such as ``role``, ``duration``,
    ``tstart``, or ``children``; SeqStarBlock inherits those fields too.
    """

    if _has_explicit_event_container(obj):
        return False

    event_type = str(
        getattr(obj, "event_type", getattr(obj, "type", "")) or ""
    ).lower()
    kind = str(getattr(obj, "kind", "") or "").lower()
    class_name = obj.__class__.__name__.lower()

    if event_type in {
        "rf", "radiofrequency", "adc", "grad", "gradient", "trap",
        "trapezoid", "delay", "wait", "trigger", "output", "digital",
    }:
        return True

    if kind.startswith("adc") or kind.startswith("rf"):
        return True

    if kind in {
        "grad", "gradient", "trap", "trapezoid", "arbitrary_grad",
        "arbitrary_gradient", "split_gradient", "delay", "wait",
    }:
        return True

    if any(token in class_name for token in (
        "rfevent", "adcevent", "gradientevent", "delayevent", "triggerevent",
    )):
        return True

    return any(
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
        )
    )



def _is_delay_like(event: Any) -> bool:
    """Return True for structural delay/wait events."""

    event_type = str(
        getattr(event, "event_type", getattr(event, "type", "")) or ""
    ).lower()
    kind = str(getattr(event, "kind", "") or "").lower()
    class_name = event.__class__.__name__.lower()
    role = str(getattr(event, "role", "") or "").lower()

    return (
        event_type in {"delay", "wait"}
        or kind in {"delay", "wait"}
        or "delayevent" in class_name
        or class_name.endswith("delay")
        or role in {"delay", "wait", "echo_time_fill", "repetition_fill"}
    )


def _is_gradient_like(event: Any) -> bool:
    """Return True for gradient/trapezoid/arbitrary-gradient events."""

    event_type = str(
        getattr(event, "event_type", getattr(event, "type", "")) or ""
    ).lower()
    kind = str(getattr(event, "kind", "") or "").lower()
    class_name = event.__class__.__name__.lower()

    return (
        event_type in {"grad", "gradient", "trap", "trapezoid"}
        or kind in {
            "grad",
            "gradient",
            "trap",
            "trapezoid",
            "arbitrary",
            "arbitrary_grad",
            "arbitrary_gradient",
            "split",
            "split_gradient",
        }
        or "gradient" in class_name
        or "trapezoid" in class_name
        or (
            hasattr(event, "channel")
            and (
                hasattr(event, "area")
                or hasattr(event, "amplitude")
                or hasattr(event, "flat_area")
                or all(
                    hasattr(event, attr)
                    for attr in ("rise_time", "flat_time", "fall_time")
                )
            )
        )
    )


def _is_rf_like(event: Any) -> bool:
    """Return True if event is RF-like."""

    event_type = str(getattr(event, "type", "")).lower()
    kind = str(getattr(event, "kind", "")).lower()
    class_name = event.__class__.__name__.lower()
    name = str(getattr(event, "name", "")).lower()

    return (
        event_type == "rf"
        or kind.startswith("rf")
        or "rf" in class_name
        or "rf" in name
        or hasattr(event, "flip_angle")
    )


def _is_adc_like(event: Any) -> bool:
    """Return True only for explicitly ADC-like events.

    RF identity takes precedence. ``num_samples`` alone is not an ADC type
    discriminator because enriched events may expose shared protocol fields.
    """

    if _is_rf_like(event):
        return False

    event_type = str(
        getattr(event, "event_type", getattr(event, "type", "")) or ""
    ).lower()
    kind = str(getattr(event, "kind", "") or "").lower()
    class_name = event.__class__.__name__.lower()
    name = str(getattr(event, "name", "") or "").lower()

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

    return (
        getattr(event, "num_samples", None) is not None
        and getattr(event, "dwell", None) is not None
        and not hasattr(event, "flip_angle")
    )


def _get_optional_float(obj: Any, attr: str) -> float | None:
    """Return object attribute as float if present and not None."""

    if not hasattr(obj, attr):
        return None

    value = getattr(obj, attr)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_parameter_float(event: Any, key: str) -> float | None:
    """Return event.parameters[key] as float if present."""

    parameters = getattr(event, "parameters", None)

    if not isinstance(parameters, Mapping):
        return None

    if key not in parameters:
        return None

    value = parameters[key]

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_metadata_timing_float(event: Any, key: str) -> float | None:
    """Return event.metadata['timing'][key] as float if present."""

    metadata = getattr(event, "metadata", None)

    if not isinstance(metadata, Mapping):
        return None

    timing = metadata.get("timing")

    if not isinstance(timing, Mapping):
        return None

    if key not in timing:
        return None

    value = timing[key]

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None