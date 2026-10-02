"""Timing checks for pypulseq_star.

This module keeps the PyPulseq-compatible filename:

    pypulseq_star.check_timing

and supports the familiar use:

    ok, error_report = seq.check_timing()

The implementation checks timing generically for RF, ADC, gradients, delays,
and future SeqStar events by reading timing from attributes, parameters, and
metadata.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any

from pypulseq_star.calc_duration import calc_duration

EPS = 1e-12

error_messages = {
    "RASTER": (
        "{value*multiplier:.2f} {unit} does not align to {raster} "
        "(Nearest valid value: {value_rounded*multiplier:.0f} {unit}, "
        "error: {error*multiplier:.2f} {unit})"
    ),
    "DELAY_DEAD_TIME": (
        "Delay is smaller than required event dead time "
        "({value*multiplier:.2f} {unit} < {dead_time*multiplier:.0f} {unit})"
    ),
    "POST_DEAD_TIME": (
        "Post-event protected time exceeds block duration "
        "({value*multiplier:.2f} {unit} + {dead_time*multiplier:.0f} {unit} "
        "> {duration*multiplier:.2f} {unit})"
    ),
    "ADC_DEAD_TIME": (
        "ADC delay is smaller than ADC dead time "
        "({value*multiplier:.2f} {unit} < {dead_time*multiplier:.0f} {unit})"
    ),
    "POST_ADC_DEAD_TIME": (
        "Post-ADC dead time exceeds block duration "
        "({value*multiplier:.2f} {unit} + {dead_time*multiplier:.0f} {unit} "
        "> {duration*multiplier:.2f} {unit})"
    ),
    "RF_DEAD_TIME": (
        "Delay of {value*multiplier:.2f} {unit} is smaller than the RF dead time "
        "{dead_time*multiplier:.0f} {unit}"
    ),
    "RF_RINGDOWN_TIME": (
        "Time between the end of the RF pulse at {value*multiplier:.2f} {unit} "
        "and the end of the block at {duration*multiplier:.2f} {unit} is shorter "
        "than rf_ringdown_time ({ringdown_time*multiplier:.0f} {unit})"
    ),
    "BLOCK_DURATION_MISMATCH": (
        "Inconsistency between the stored block duration "
        "({duration*multiplier:.2f} {unit}) and the content of the block "
        "({value*multiplier:.2f} {unit})"
    ),
    "NEGATIVE_DELAY": "Delay is negative {value*multiplier:.2f} {unit}",
    "NEGATIVE_DURATION": "Duration is negative {value*multiplier:.2f} {unit}",
    "RF_MAX": (
        "RF amplitude exceeds system.max_rf "
        "({value:.6g} T > {max_rf:.6g} T)"
    ),
}


def check_timing(seq: Any) -> tuple[bool, list[SimpleNamespace]]:
    """Check timing of a SeqStarSequence-like object.

    Returns
    -------
    ok
        True if no timing errors were found.

    error_report
        List of SimpleNamespace errors, intentionally similar to PyPulseq.
    """

    error_report: list[SimpleNamespace] = []
    system = seq.system

    for block_counter, block in _iter_blocks(seq):
        block_duration = calc_duration(block)

        _div_check(
            value=block_duration,
            raster=system.block_duration_raster,
            block=block_counter,
            event="block",
            field="duration",
            raster_name="block_duration_raster",
            error_report=error_report,
        )

        stored_duration = _get_stored_block_duration(seq, block, block_counter)

        if stored_duration is not None and abs(block_duration - stored_duration) > EPS:
            error_report.append(
                SimpleNamespace(
                    block=block_counter,
                    event="block",
                    field="duration",
                    error_type="BLOCK_DURATION_MISMATCH",
                    value=block_duration,
                    duration=stored_duration,
                )
            )
            block_duration = stored_duration

        for event_name, event in _iter_block_events(block):
            if event is None or isinstance(event, (float, int, str, bool)):
                continue

            _check_event_rasters(
                event=event,
                event_name=event_name,
                block=block_counter,
                system=system,
                error_report=error_report,
            )

            _check_event_dead_times(
                event=event,
                event_name=event_name,
                block=block_counter,
                block_duration=block_duration,
                system=system,
                error_report=error_report,
            )

            if _is_rf(event):
                _check_rf_amplitude(
                    event=event,
                    event_name=event_name,
                    block=block_counter,
                    system=system,
                    error_report=error_report,
                )

    return len(error_report) == 0, error_report


def _check_event_rasters(
    *,
    event: Any,
    event_name: str,
    block: int,
    system: Any,
    error_report: list[SimpleNamespace],
) -> None:
    """Check delay, duration, dwell, and trapezoid timing against rasters."""

    raster, raster_name = _event_raster(event, system)

    delay = _get_event_delay(event)
    if delay < -EPS:
        error_report.append(
            SimpleNamespace(
                block=block,
                event=event_name,
                field="delay",
                error_type="NEGATIVE_DELAY",
                value=delay,
            )
        )

    _div_check(
        value=delay,
        raster=raster,
        block=block,
        event=event_name,
        field="delay",
        raster_name=raster_name,
        error_report=error_report,
    )

    active_duration = _get_event_active_duration(event)
    if active_duration < -EPS:
        error_report.append(
            SimpleNamespace(
                block=block,
                event=event_name,
                field="duration",
                error_type="NEGATIVE_DURATION",
                value=active_duration,
            )
        )

    if active_duration > 0:
        _div_check(
            value=active_duration,
            raster=raster,
            block=block,
            event=event_name,
            field="duration",
            raster_name=raster_name,
            error_report=error_report,
        )

    dwell = _get_event_dwell(event)
    if dwell is not None:
        _div_check(
            value=dwell,
            raster=system.adc_raster_time,
            block=block,
            event=event_name,
            field="dwell",
            raster_name="adc_raster_time",
            error_report=error_report,
        )

    # Trapezoid gradient subfields always use gradient raster.
    for field in ("rise_time", "flat_time", "fall_time"):
        value = _get_timing_value(event, field)
        if value is not None:
            _div_check(
                value=value,
                raster=system.grad_raster_time,
                block=block,
                event=event_name,
                field=field,
                raster_name="grad_raster_time",
                error_report=error_report,
            )


def _check_event_dead_times(
    *,
    event: Any,
    event_name: str,
    block: int,
    block_duration: float,
    system: Any,
    error_report: list[SimpleNamespace],
) -> None:
    """Check pre-event dead time and post-event protected time.

    Generic model:
        delay >= required_pre_dead_time

        delay + active_duration + required_post_time <= block_duration

    RF and ADC still use PyPulseq-compatible error type names for familiar
    reports.

    Notes
    -----
    A zero post-dead-time must never generate a POST_DEAD_TIME error. This is
    important for ordinary gradient events such as slice-select gradients, where
    the event may have a nonzero delay but no protected post-event interval.
    """

    delay = _get_event_delay(event)
    active_duration = _get_event_active_duration(event)

    pre_dead_time = _get_required_pre_dead_time(event, system)
    post_dead_time = _get_required_post_dead_time(event, system)

    # -------------------
    # Pre-event dead time
    # -------------------

    if delay - pre_dead_time < -EPS:
        if _is_rf(event):
            error_type = "RF_DEAD_TIME"
        elif _is_adc(event):
            error_type = "ADC_DEAD_TIME"
        else:
            error_type = "DELAY_DEAD_TIME"

        error_report.append(
            SimpleNamespace(
                block=block,
                event=event_name,
                field="delay",
                error_type=error_type,
                value=delay,
                dead_time=pre_dead_time,
            )
        )

    # ------------------------------
    # Post-event protected dead time
    # ------------------------------

    # No protected post time is required. Do not emit POST_DEAD_TIME errors
    # for ordinary gradients or delay-free objects.
    if post_dead_time <= EPS:
        return

    event_end = delay + active_duration
    occupied_end = event_end + post_dead_time

    if occupied_end - block_duration <= EPS:
        return

    if _is_rf(event):
        error_report.append(
            SimpleNamespace(
                block=block,
                event=event_name,
                field="duration",
                error_type="RF_RINGDOWN_TIME",
                value=occupied_end,
                duration=block_duration,
                ringdown_time=post_dead_time,
            )
        )
    elif _is_adc(event):
        error_report.append(
            SimpleNamespace(
                block=block,
                event=event_name,
                field="duration",
                error_type="POST_ADC_DEAD_TIME",
                value=occupied_end,
                duration=block_duration,
                dead_time=post_dead_time,
            )
        )
    else:
        error_report.append(
            SimpleNamespace(
                block=block,
                event=event_name,
                field="duration",
                error_type="POST_DEAD_TIME",
                value=occupied_end,
                duration=block_duration,
                dead_time=post_dead_time,
            )
        )

def _check_rf_amplitude(
    *,
    event: Any,
    event_name: str,
    block: int,
    system: Any,
    error_report: list[SimpleNamespace],
) -> None:
    """Check RF amplitude against system.max_rf if inferable."""

    if not hasattr(system, "max_rf"):
        return

    amplitude_t = _get_rf_amplitude_t(event, system)

    if amplitude_t is None:
        return

    max_rf = float(system.max_rf)

    if amplitude_t - max_rf > EPS:
        error_report.append(
            SimpleNamespace(
                block=block,
                event=event_name,
                field="amplitude",
                error_type="RF_MAX",
                value=amplitude_t,
                max_rf=max_rf,
            )
        )


def _iter_blocks(seq: Any) -> Iterable[tuple[int, Any]]:
    """Yield ``(block_index, block)`` from SeqStar or PyPulseq-like sequences."""

    if hasattr(seq, "timeline") and hasattr(seq.timeline, "blocks"):
        blocks = seq.timeline.blocks
        if isinstance(blocks, Mapping):
            for idx, block in enumerate(blocks.values()):
                yield idx, block
        else:
            for idx, block in enumerate(blocks):
                yield idx, block
        return

    if hasattr(seq, "block_events") and hasattr(seq, "get_block"):
        for block_counter in seq.block_events:
            yield block_counter, seq.get_block(block_counter)
        return

    if hasattr(seq, "blocks"):
        blocks = seq.blocks
        if isinstance(blocks, Mapping):
            for idx, block in enumerate(blocks.values()):
                yield idx, block
        else:
            for idx, block in enumerate(blocks):
                yield idx, block
        return

    raise TypeError("Sequence object does not expose timeline.blocks, block_events, or blocks.")


def _iter_block_events(block: Any) -> Iterable[tuple[str, Any]]:
    """Yield ``(event_name, event)`` from a block."""

    if hasattr(block, "events"):
        events = getattr(block, "events")

        if isinstance(events, Mapping):
            yield from ((str(key), value) for key, value in events.items())
            return

        for idx, event in enumerate(events):
            event_name = str(getattr(event, "name", f"event_{idx}"))
            yield event_name, event
        return

    if hasattr(block, "children"):
        children = getattr(block, "children")

        if isinstance(children, Mapping):
            yield from ((str(key), value) for key, value in children.items())
            return

        for idx, event in enumerate(children):
            event_name = str(getattr(event, "name", f"event_{idx}"))
            yield event_name, event
        return

    if hasattr(block, "__dict__"):
        for event_name, event in vars(block).items():
            if event_name.startswith("_"):
                continue
            yield event_name, event
        return

    raise TypeError("Block object does not expose events, children, or __dict__.")


def _get_stored_block_duration(seq: Any, block: Any, block_counter: int) -> float | None:
    """Return stored block duration if available."""

    if hasattr(seq, "block_durations"):
        block_durations = getattr(seq, "block_durations")
        try:
            return float(block_durations[block_counter])
        except (KeyError, IndexError, TypeError):
            pass

    duration = _get_optional_float(block, "duration")
    if duration is not None:
        return duration

    return _get_timing_value(block, "duration")


def _event_raster(event: Any, system: Any) -> tuple[float, str]:
    """Return raster time and name for an event."""

    if _is_adc(event):
        return float(system.adc_raster_time), "adc_raster_time"

    if _is_rf(event):
        return float(system.rf_raster_time), "rf_raster_time"

    return float(system.grad_raster_time), "grad_raster_time"


def _get_event_delay(event: Any) -> float:
    """Return local event delay/start in seconds."""

    for attr in ("delay", "tstart", "start", "start_s"):
        value = _get_optional_float(event, attr)
        if value is not None:
            return value

    for key in ("delay", "tstart", "start", "start_s"):
        value = _get_timing_value(event, key)
        if value is not None:
            return value

    return 0.0


def _get_event_active_duration(event: Any) -> float:
    """Return active event duration after explicit family classification.

    Shared protocol dictionaries may contain fields for several event families.
    ADC-specific fields such as ``num_samples`` and ``dwell`` are therefore
    consulted only after the concrete event has been classified as ADC-like.

    This function is sequence-family agnostic. It knows event families, not
    sequence names or user-defined node labels.
    """

    if _is_delay(event):
        # Structural delay/wait events represent one occupied interval. Some
        # compatibility objects mirror that value through both delay and
        # duration; use the structural duration once.
        duration = _get_optional_float(event, "duration")
        if duration is not None:
            return duration

        duration = _get_timing_value(event, "duration")
        if duration is not None:
            return duration

        wait_time = _get_optional_float(event, "wait_time")
        if wait_time is not None:
            return wait_time

        return max(_get_event_delay(event), 0.0)

    if _is_adc(event):
        num_samples = _get_optional_float(event, "num_samples")
        dwell = _get_optional_float(event, "dwell")

        if num_samples is None:
            num_samples = _get_timing_value(event, "num_samples")
        if dwell is None:
            dwell = _get_timing_value(event, "dwell")

        if num_samples is not None and dwell is not None:
            return num_samples * dwell

        for key in ("duration", "adc_duration", "readout_duration"):
            value = _get_optional_float(event, key)
            if value is not None:
                return value

            value = _get_timing_value(event, key)
            if value is not None:
                return value

        return 0.0

    if _is_gradient(event):
        rise_time = _get_optional_float(event, "rise_time")
        flat_time = _get_optional_float(event, "flat_time")
        fall_time = _get_optional_float(event, "fall_time")

        if rise_time is None:
            rise_time = _get_timing_value(event, "rise_time")
        if flat_time is None:
            flat_time = _get_timing_value(event, "flat_time")
        if fall_time is None:
            fall_time = _get_timing_value(event, "fall_time")

        if (
            rise_time is not None
            and flat_time is not None
            and fall_time is not None
        ):
            return rise_time + flat_time + fall_time

    # RF, trigger, and generic events use their own explicit waveform duration.
    # Do not inspect ADC-specific protocol fields on this path.
    t = getattr(event, "t", None)
    if t is not None:
        try:
            return float(t[-1])
        except (TypeError, IndexError, ValueError):
            pass

    for key in ("duration", "shape_duration", "active_duration"):
        value = _get_optional_float(event, key)
        if value is not None:
            return value

    shape = getattr(event, "shape", None)
    if shape is not None:
        for key in ("duration", "shape_duration", "active_duration"):
            value = _get_optional_float(shape, key)
            if value is not None:
                return value

            value = _get_timing_value(shape, key)
            if value is not None:
                return value

    # Parameter fallback is limited to generic duration aliases.
    for key in ("duration", "shape_duration", "active_duration"):
        value = _get_timing_value(event, key)
        if value is not None:
            return value

    return 0.0


def _get_event_dwell(event: Any) -> float | None:
    """Return ADC dwell if available."""

    value = _get_optional_float(event, "dwell")
    if value is not None:
        return value

    return _get_timing_value(event, "dwell")


def _get_required_pre_dead_time(event: Any, system: Any) -> float:
    """Return required pre-event dead time."""

    if _is_gradient(event) or _is_delay(event):
        return 0.0

    # Generic keys first.
    for key in ("pre_dead_time", "dead_time_before", "required_delay"):
        value = _get_timing_value(event, key)
        if value is not None:
            return value

    # PyPulseq-compatible attributes.
    value = _get_optional_float(event, "dead_time")
    if value is not None:
        return value

    # System defaults by event type.
    if _is_rf(event):
        return float(getattr(system, "rf_dead_time", 0.0))

    if _is_adc(event):
        return float(getattr(system, "adc_dead_time", 0.0))

    return 0.0


def _get_required_post_dead_time(event: Any, system: Any) -> float:
    """Return required post-event protected time."""

    if _is_gradient(event) or _is_delay(event):
        return 0.0

    for key in (
        "post_dead_time",
        "post_time",
        "post_delay",
        "dead_time_after",
        "ringdown_time",
        "rf_ringdown_time",
        "adc_dead_time",
    ):
        value = _get_timing_value(event, key)
        if value is not None:
            return value

    # PyPulseq-compatible attributes.
    if _is_rf(event):
        value = _get_optional_float(event, "ringdown_time")
        if value is not None:
            return value

        return float(getattr(system, "rf_ringdown_time", 0.0))

    if _is_adc(event):
        value = _get_optional_float(event, "dead_time")
        if value is not None:
            return value

        return float(getattr(system, "adc_dead_time", 0.0))

    return 0.0


def _get_rf_amplitude_t(event: Any, system: Any) -> float | None:
    """Return RF B1 amplitude in tesla if inferable."""

    value = _get_optional_float(event, "amplitude_t")
    if value is not None:
        return value

    value = _get_timing_value(event, "amplitude_t")
    if value is not None:
        return value

    shape = getattr(event, "shape", None)
    if shape is not None:
        value = _get_optional_float(shape, "amplitude_t")
        if value is not None:
            return value

        if hasattr(shape, "to_gammastar_samples") and hasattr(event, "flip_angle"):
            try:
                samples = shape.to_gammastar_samples(
                    flip_angle=float(event.flip_angle),
                    gamma_hz_per_t=float(system.gamma),
                    max_rf=None,
                )
                v = samples.get("v", [])
                if v and isinstance(v[0], Mapping):
                    am = v[0].get("am", [])
                    if am:
                        return max(abs(float(x)) for x in am)
            except Exception:
                return None

    return None


def _div_check(
    *,
    value: float,
    raster: float,
    block: int,
    event: str,
    field: str,
    raster_name: str,
    error_report: list[SimpleNamespace],
) -> None:
    """Check whether value aligns to a raster."""

    if raster <= 0:
        raise ValueError(f"{raster_name} must be positive. Passed: {raster}")

    c = value / raster
    c_rounded = round(c)
    is_ok = abs(c - c_rounded) < 1e-6

    if not is_ok:
        error_report.append(
            SimpleNamespace(
                block=block,
                event=event,
                field=field,
                value=value,
                value_rounded=c_rounded * raster,
                error=value - c_rounded * raster,
                raster=raster_name,
                error_type="RASTER",
            )
        )


def _is_rf(event: Any) -> bool:
    """Return True only for explicitly RF-like events."""

    # Explicit non-RF event families take precedence.
    if _is_gradient(event) or _is_delay(event):
        return False

    event_type = str(
        getattr(event, "event_type", getattr(event, "type", "")) or ""
    ).lower()
    kind = str(getattr(event, "kind", "") or "").lower()
    role = str(getattr(event, "role", "") or "").lower()
    use = str(getattr(event, "use", "") or "").lower()
    class_name = event.__class__.__name__.lower()

    if event_type == "rf":
        return True

    if kind.startswith("rf"):
        return True

    if "rfevent" in class_name or "rfblockevent" in class_name:
        return True

    if role in {"rf", "excitation", "refocusing", "inversion", "saturation"}:
        return True

    if use in {"rf", "excitation", "refocusing", "inversion", "saturation"}:
        return True

    return getattr(event, "flip_angle", None) is not None


def _is_adc(event: Any) -> bool:
    """Return True only for explicitly ADC-like events.

    RF identity takes precedence. Merely exposing ``num_samples`` is not an ADC
    discriminator because enriched objects may expose shared protocol fields.
    """

    if _is_rf(event):
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

    # Structural fallback for custom ADC classes. Sampling fields must be
    # concrete event attributes, not merely values in a shared parameters map.
    return (
        getattr(event, "num_samples", None) is not None
        and getattr(event, "dwell", None) is not None
        and not hasattr(event, "flip_angle")
    )


def _is_delay(event: Any) -> bool:
    """Return True when an event is explicitly a structural delay/wait."""

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
        or class_name in {"delay", "wait"}
        or role in {"delay", "wait", "timing_fill", "repetition_fill"}
    )


def _is_gradient(event: Any) -> bool:
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


def _get_timing_value(obj: Any, key: str) -> float | None:
    """Read timing value from parameters or metadata.timing."""

    parameters = getattr(obj, "parameters", None)

    if isinstance(parameters, Mapping) and key in parameters:
        try:
            return float(parameters[key])
        except (TypeError, ValueError):
            return None

    metadata = getattr(obj, "metadata", None)

    if isinstance(metadata, Mapping):
        timing = metadata.get("timing")
        if isinstance(timing, Mapping) and key in timing:
            try:
                return float(timing[key])
            except (TypeError, ValueError):
                return None

    return None


def _get_optional_float(obj: Any, attr: str) -> float | None:
    """Return object attribute as float if available."""

    if not hasattr(obj, attr):
        return None

    value = getattr(obj, attr)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_string(template: str, **kwargs: Any) -> str:
    """Evaluate a formatted string using f-string syntax."""

    return eval(f'f"""{template}"""', kwargs)


def indent_string(x: str, n: int = 2) -> str:
    """Add indentation to every line in a string."""

    return "\n".join(" " * n + y for y in x.splitlines())


def print_error_report(
    seq: Any,
    error_report: list[SimpleNamespace],
    full_report: bool = False,
    max_errors: int = 10,
    colored: bool = True,
) -> None:
    """Print a readable timing error report."""

    current_block = None

    if full_report:
        max_errors = len(error_report)

    for error in error_report[:max_errors]:
        if error.block != current_block:
            print(f"Block {error.block}:")
            current_block = error.block

        unit = "us"
        multiplier = 1e6

        if error.field == "dwell":
            unit = "ns"
            multiplier = 1e9

        error_message = format_string(
            error_messages[error.error_type],
            **error.__dict__,
            unit=unit,
            multiplier=multiplier,
        )

        prefix = "\x1b[38;5;9m" if colored else ""
        suffix = "\x1b[0m" if colored else ""

        print(f"- {error.event}.{error.field}: " + prefix + error_message + suffix)

    if len(error_report) > max_errors:
        blocks = [error.block for error in error_report[max_errors:]]
        print(
            f"--- {len(error_report) - max_errors} more errors in blocks "
            f"{min(blocks)} to {max(blocks)} hidden ---"
        )