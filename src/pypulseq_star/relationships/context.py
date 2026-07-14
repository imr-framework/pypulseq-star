"""Relationship evaluation context and generic object access helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


_MISSING = object()


def protocol_value(seq: Any, key_or_value: Any) -> Any:
    """Resolve a protocol parameter reference or literal value.

    Strings such as "TE" and "TR" are treated as protocol parameter references.
    This resolver supports common aliases so relationship calls can remain
    MRI/PyPulseq-like even if the protocol uses more descriptive names.

    Examples
    --------
    protocol_value(seq, "TE")
        -> seq.parameters["TE"] or seq.parameters["echo_time"]

    protocol_value(seq, "TR")
        -> seq.parameters["TR"] or seq.parameters["repetition_time"]

    protocol_value(seq, 20e-3)
        -> 20e-3
    """

    if not isinstance(key_or_value, str):
        return key_or_value

    key = key_or_value

    # Support gammaSTAR-ish paths or future internal path notation.
    # "root.prot.TE" -> "TE"
    if key.startswith("root.prot."):
        key = key.removeprefix("root.prot.")

    aliases: dict[str, tuple[str, ...]] = {
        "TE": (
            "TE",
            "te",
            "echo_time",
            "EchoTime",
            "echoTime",
            "echo_time_s",
        ),
        "TR": (
            "TR",
            "tr",
            "repetition_time",
            "RepetitionTime",
            "repetitionTime",
            "repetition_time_s",
        ),
        "FA": (
            "FA",
            "fa",
            "flip_angle",
            "flip_angle_excitation",
            "excitation_flip_angle",
        ),
    }

    candidate_keys = aliases.get(key, (key,))

    # If a user passes lower-case aliases directly, also support them.
    for canonical_key, canonical_aliases in aliases.items():
        if key.lower() == canonical_key.lower():
            candidate_keys = canonical_aliases
            break

        if any(key.lower() == alias.lower() for alias in canonical_aliases):
            candidate_keys = canonical_aliases
            break

    # 1. Preferred SeqStarSequence API.
    if hasattr(seq, "get_protocol_parameter"):
        for candidate in candidate_keys:
            try:
                value = seq.get_protocol_parameter(candidate, default=_MISSING)
                if value is not _MISSING:
                    return value
            except TypeError:
                # Some future/alternate versions may not accept default=.
                try:
                    value = seq.get_protocol_parameter(candidate)
                    if value is not None:
                        return value
                except Exception:
                    pass
            except Exception:
                pass

    # 2. Direct sequence parameters.
    parameters = getattr(seq, "parameters", None)

    if isinstance(parameters, Mapping):
        for candidate in candidate_keys:
            if candidate in parameters:
                return parameters[candidate]

        # Case-insensitive fallback.
        for candidate in candidate_keys:
            for existing_key, value in parameters.items():
                if str(existing_key).lower() == str(candidate).lower():
                    return value

    # 3. Sequence context_dict() fallback.
    if hasattr(seq, "context_dict"):
        try:
            context = seq.context_dict()
            protocol = context.get("protocol", None)

            if isinstance(protocol, Mapping):
                for candidate in candidate_keys:
                    if candidate in protocol:
                        return protocol[candidate]

                for candidate in candidate_keys:
                    for existing_key, value in protocol.items():
                        if str(existing_key).lower() == str(candidate).lower():
                            return value
        except Exception:
            pass

    # 4. Attached protocol object fallback, if one exists later.
    protocol = getattr(seq, "protocol", None)

    if protocol is not None and hasattr(protocol, "get_parameter"):
        for candidate in candidate_keys:
            try:
                return protocol.get_parameter(candidate)
            except Exception:
                pass

    protocol_parameters = getattr(protocol, "parameters", None)

    if isinstance(protocol_parameters, Mapping):
        for candidate in candidate_keys:
            if candidate in protocol_parameters:
                return protocol_parameters[candidate]

        for candidate in candidate_keys:
            for existing_key, value in protocol_parameters.items():
                if str(existing_key).lower() == str(candidate).lower():
                    return value

    # 5. Metadata fallback.
    metadata = getattr(seq, "metadata", None)

    if isinstance(metadata, Mapping):
        metadata_parameters = metadata.get("parameters")

        if isinstance(metadata_parameters, Mapping):
            for candidate in candidate_keys:
                if candidate in metadata_parameters:
                    return metadata_parameters[candidate]

            for candidate in candidate_keys:
                for existing_key, value in metadata_parameters.items():
                    if str(existing_key).lower() == str(candidate).lower():
                        return value

        protocol_metadata = metadata.get("protocol")

        if isinstance(protocol_metadata, Mapping):
            protocol_metadata_parameters = protocol_metadata.get("parameters")

            if isinstance(protocol_metadata_parameters, Mapping):
                for candidate in candidate_keys:
                    if candidate in protocol_metadata_parameters:
                        return protocol_metadata_parameters[candidate]

                for candidate in candidate_keys:
                    for existing_key, value in protocol_metadata_parameters.items():
                        if str(existing_key).lower() == str(candidate).lower():
                            return value

    return key_or_value

def protocol_float(
    seq: Any,
    key_or_value: Any,
    *,
    name: str | None = None,
) -> float:
    """Resolve a protocol parameter reference or literal value as float.

    Parameters
    ----------
    seq
        Sequence object carrying protocol/sequence parameters.

    key_or_value
        Either a literal numeric value or a protocol parameter name such as
        "TE", "TR", or "root.prot.TE".

    name
        Optional human-readable name used only for error messages.
    """

    value = protocol_value(seq, key_or_value)

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        available_keys = []

        parameters = getattr(seq, "parameters", None)
        if isinstance(parameters, Mapping):
            available_keys = [str(key) for key in parameters.keys()]

        display_name = name if name is not None else str(key_or_value)

        raise ValueError(
            f"Protocol value {display_name!r} must resolve to a float. "
            f"Resolved value: {value!r}. "
            f"Original input: {key_or_value!r}. "
            f"Available sequence parameter keys: {available_keys}"
        ) from exc
    
def protocol_int(seq: Any, key_or_value: Any, *, name: str | None = None) -> int:
    """Resolve a protocol parameter reference or literal as int."""

    value = protocol_value(seq, key_or_value)

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        label = name or str(key_or_value)
        raise ValueError(f"Protocol value {label!r} must resolve to an int. Got {value!r}.") from exc


def event_name(event: Any) -> str:
    """Return a stable event name for debug records."""

    return str(getattr(event, "name", event.__class__.__name__))


def event_delay(event: Any) -> float:
    """Return event delay/start time from common locations."""

    for attr in ("delay", "tstart", "start", "start_s"):
        value = _float_attr(event, attr)
        if value is not None:
            return value

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in ("delay", "tstart", "start", "start_s"):
            value = _float_from_mapping(parameters, key)
            if value is not None:
                return value

    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping):
        timing = metadata.get("timing")
        if isinstance(timing, Mapping):
            for key in ("delay", "tstart", "start", "start_s"):
                value = _float_from_mapping(timing, key)
                if value is not None:
                    return value

        for key in ("delay", "tstart", "start", "start_s"):
            value = _float_from_mapping(metadata, key)
            if value is not None:
                return value

    return 0.0


def set_event_delay(event: Any, delay: float) -> None:
    """Set event delay in writer-friendly places."""

    delay = float(delay)

    for attr in ("delay", "tstart"):
        if hasattr(event, attr):
            try:
                setattr(event, attr, delay)
            except AttributeError:
                pass

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, dict):
        parameters["delay"] = delay
        parameters.setdefault("tstart", delay)

    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, dict):
        timing = metadata.setdefault("timing", {})
        if isinstance(timing, dict):
            timing["delay"] = delay
            timing.setdefault("tstart", delay)


def event_duration(event: Any) -> float:
    """Return event active duration from common event/shape fields."""

    # ADC shortcut.
    adc_duration = try_adc_duration(event)
    if adc_duration is not None:
        return adc_duration

    for attr in ("duration", "shape_duration", "flat_time"):
        value = _float_attr(event, attr)
        if value is not None:
            return value

    # Trapezoid duration.
    rise = timing_value(event, "rise_time")
    flat = timing_value(event, "flat_time")
    fall = timing_value(event, "fall_time")
    if rise is not None and flat is not None and fall is not None:
        return rise + flat + fall

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in ("duration", "shape_duration", "rf_duration", "adc_duration"):
            value = _float_from_mapping(parameters, key)
            if value is not None:
                return value

    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping):
        timing = metadata.get("timing")
        if isinstance(timing, Mapping):
            for key in ("duration", "shape_duration", "rf_duration", "adc_duration"):
                value = _float_from_mapping(timing, key)
                if value is not None:
                    return value

    shape = getattr(event, "shape", None)
    if shape is not None:
        for attr in ("duration", "shape_duration", "flat_time"):
            value = _float_attr(shape, attr)
            if value is not None:
                return value

    raise ValueError(f"Could not determine duration for event {event_name(event)!r}.")


def try_adc_duration(event: Any) -> float | None:
    """Return ADC duration if event looks like an ADC, else None."""

    num_samples = None
    dwell = None

    for attr in ("num_samples", "number_of_samples"):
        if hasattr(event, attr) and getattr(event, attr) is not None:
            try:
                num_samples = int(getattr(event, attr))
                break
            except (TypeError, ValueError):
                pass

    for attr in ("dwell", "sample_time"):
        if hasattr(event, attr) and getattr(event, attr) is not None:
            try:
                dwell = float(getattr(event, attr))
                break
            except (TypeError, ValueError):
                pass

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, Mapping):
        if num_samples is None:
            for key in ("num_samples", "number_of_samples"):
                if key in parameters and parameters[key] is not None:
                    try:
                        num_samples = int(parameters[key])
                        break
                    except (TypeError, ValueError):
                        pass

        if dwell is None:
            for key in ("dwell", "sample_time"):
                if key in parameters and parameters[key] is not None:
                    try:
                        dwell = float(parameters[key])
                        break
                    except (TypeError, ValueError):
                        pass

    if num_samples is not None and dwell is not None:
        return float(num_samples) * float(dwell)

    return None


def event_center_pos(event: Any, *, default: float = 0.5) -> float:
    """Return normalized center position within event duration."""

    for attr in ("center_pos", "center_position"):
        value = _float_attr(event, attr)
        if value is not None:
            return value

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in ("center_pos", "rf_center_pos", "center_position"):
            value = _float_from_mapping(parameters, key)
            if value is not None:
                return value

    shape = getattr(event, "shape", None)
    if shape is not None:
        for attr in ("center_pos", "center_position"):
            value = _float_attr(shape, attr)
            if value is not None:
                return value

    return float(default)


def event_anchor_time(event: Any, anchor: str) -> float:
    """Return event start/center/end time using event delay + duration."""

    anchor = anchor.lower()
    start = event_delay(event)
    duration = event_duration(event)

    if anchor in {"start", "tstart", "delay"}:
        return start

    if anchor in {"center", "centre", "tcenter", "rf_center", "adc_center"}:
        return start + event_center_pos(event) * duration

    if anchor in {"end", "tend", "stop"}:
        return start + duration

    raise ValueError(f"Unsupported event anchor {anchor!r}.")


def set_event_property(event: Any, property_name: str, value: float) -> None:
    """Set a derived event property."""

    property_name = property_name.lower()

    if property_name in {"delay", "tstart", "start"}:
        set_event_delay(event, value)
        return

    if hasattr(event, property_name):
        try:
            setattr(event, property_name, value)
        except AttributeError:
            pass

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, dict):
        parameters[property_name] = value

    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, dict):
        derived = metadata.setdefault("derived", {})
        if isinstance(derived, dict):
            derived[property_name] = value


def timing_value(obj: Any, key: str) -> float | None:
    """Read a timing value from attr, parameters, or metadata.timing."""

    value = _float_attr(obj, key)
    if value is not None:
        return value

    parameters = getattr(obj, "parameters", None)
    if isinstance(parameters, Mapping):
        value = _float_from_mapping(parameters, key)
        if value is not None:
            return value

    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, Mapping):
        timing = metadata.get("timing")
        if isinstance(timing, Mapping):
            value = _float_from_mapping(timing, key)
            if value is not None:
                return value

        value = _float_from_mapping(metadata, key)
        if value is not None:
            return value

    return None


def store_derived_parameter(seq: Any, key: str, value: Any) -> None:
    """Store a derived parameter without making it look user-defined."""

    if hasattr(seq, "derived_parameters"):
        derived = getattr(seq, "derived_parameters")
        if isinstance(derived, dict):
            derived[key] = value
            return

    try:
        setattr(seq, "derived_parameters", {key: value})
        return
    except AttributeError:
        pass

    metadata = getattr(seq, "metadata", None)
    if isinstance(metadata, dict):
        metadata.setdefault("derived_parameters", {})[key] = value
        return

    try:
        setattr(seq, "metadata", {"derived_parameters": {key: value}})
    except AttributeError:
        pass


def _float_attr(obj: Any, attr: str) -> float | None:
    if not hasattr(obj, attr):
        return None
    value = getattr(obj, attr)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _float_from_mapping(mapping: Mapping[str, Any], key: str) -> float | None:
    if key not in mapping or mapping[key] is None:
        return None
    try:
        value = float(mapping[key])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value
