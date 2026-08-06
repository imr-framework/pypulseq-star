"""Base classes and introspection helpers for sequence-agnostic constraints."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable

from pypulseq_star.feasibility.diagnostics import FeasibilityDiagnostic

EPS = 1e-12


@dataclass(frozen=True, slots=True)
class Constraint:
    """A reusable validity rule.

    A constraint is a callable over an ``EvaluationState`` and returns zero or
    more structured diagnostics.  It should know event families and system
    concepts, not sequence names.
    """

    name: str
    description: str
    evaluator: Callable[["EvaluationState"], Iterable[FeasibilityDiagnostic]]
    severity: str = "error"

    def evaluate(self, state: "EvaluationState") -> list[FeasibilityDiagnostic]:
        return list(self.evaluator(state))


@dataclass(slots=True)
class EvaluationState:
    """Normalized view used by constraints."""

    sequence: Any
    system: Any
    protocol_values: dict[str, Any]
    derived_values: dict[str, Any]


def iter_blocks(sequence: Any) -> Iterable[tuple[int, Any]]:
    if hasattr(sequence, "timeline") and hasattr(sequence.timeline, "blocks"):
        blocks = sequence.timeline.blocks
        if isinstance(blocks, Mapping):
            for index, block in enumerate(blocks.values()):
                yield index, block
        else:
            for index, block in enumerate(blocks):
                yield index, block
        return
    if hasattr(sequence, "blocks"):
        blocks = sequence.blocks
        if isinstance(blocks, Mapping):
            for index, block in enumerate(blocks.values()):
                yield index, block
        else:
            for index, block in enumerate(blocks):
                yield index, block
        return
    if hasattr(sequence, "block_events") and hasattr(sequence, "get_block"):
        for index in sequence.block_events:
            yield int(index), sequence.get_block(index)
        return


def iter_block_events(block: Any) -> Iterable[tuple[int, str, Any]]:
    events = getattr(block, "events", None)
    if isinstance(events, Mapping):
        for index, (name, event) in enumerate(events.items()):
            yield index, str(name), event
        return
    if events is not None:
        for index, event in enumerate(events):
            name = getattr(event, "name", None) or f"event_{index}"
            yield index, str(name), event
        return
    children = getattr(block, "children", None)
    if isinstance(children, Mapping):
        for index, (name, event) in enumerate(children.items()):
            yield index, str(name), event
        return
    if children is not None:
        for index, event in enumerate(children):
            name = getattr(event, "name", None) or f"event_{index}"
            yield index, str(name), event


def event_path(block_index: int, event_name: str, event: Any | None = None) -> str:
    meta = getattr(event, "metadata", None) if event is not None else None
    if isinstance(meta, Mapping):
        for key in ("seqstar_event_path", "event_path", "source_event_path"):
            value = meta.get(key)
            if value:
                return str(value)
    return f"block[{block_index}].{event_name}"


def block_path(block_index: int, block: Any) -> str:
    meta = getattr(block, "metadata", None)
    if isinstance(meta, Mapping):
        value = meta.get("seqstar_node") or meta.get("node")
        if value:
            return str(value)
    return str(getattr(block, "name", None) or f"block[{block_index}]")


def get_mapping_value(obj: Any, *keys: str) -> Any:
    for container_name in ("parameters", "metadata"):
        container = getattr(obj, container_name, None)
        if isinstance(container, Mapping):
            for key in keys:
                if key in container and container[key] is not None:
                    return container[key]
    return None


def get_float(obj: Any, *names: str, default: float | None = None) -> float | None:
    for name in names:
        try:
            value = getattr(obj, name)
        except Exception:
            value = None
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    value = get_mapping_value(obj, *names)
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return default


def get_int(obj: Any, *names: str, default: int | None = None) -> int | None:
    value = get_float(obj, *names, default=None)
    if value is None:
        return default
    return int(round(value))


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def approx_raster_aligned(value: float, raster: float, *, tol: float = 1e-6) -> bool:
    if raster <= 0:
        return True
    scaled = float(value) / float(raster)
    return abs(scaled - round(scaled)) <= tol


def is_adc(event: Any) -> bool:
    cls = event.__class__.__name__.lower()
    typ = str(getattr(event, "type", getattr(event, "kind", ""))).lower()
    return typ.startswith("adc") or "adc" in cls


def is_rf(event: Any) -> bool:
    cls = event.__class__.__name__.lower()
    typ = str(getattr(event, "type", getattr(event, "kind", ""))).lower()
    return typ == "rf" or "rf" in cls


def is_gradient(event: Any) -> bool:
    cls = event.__class__.__name__.lower()
    typ = str(getattr(event, "type", getattr(event, "kind", ""))).lower()
    if is_adc(event) or is_rf(event):
        return False
    return (
        "grad" in cls
        or "gradient" in cls
        or typ in {"grad", "trap", "trapezoid", "arbitrary_grad", "grad_train"}
    )


def is_delay(event: Any) -> bool:
    cls = event.__class__.__name__.lower()
    typ = str(getattr(event, "type", getattr(event, "kind", ""))).lower()
    return typ == "delay" or "delay" in cls


def active_duration(event: Any) -> float | None:
    if is_delay(event):
        return get_float(event, "duration", "delay", default=0.0)
    if is_adc(event):
        samples = get_float(event, "num_samples", "number_of_samples", default=None)
        dwell = get_float(event, "dwell", "sample_time", default=None)
        if samples is not None and dwell is not None:
            return float(samples) * float(dwell)
    if is_gradient(event):
        rise = get_float(event, "rise_time", default=None)
        flat = get_float(event, "flat_time", default=None)
        fall = get_float(event, "fall_time", default=None)
        if rise is not None and flat is not None and fall is not None:
            return rise + flat + fall
    duration = get_float(event, "duration", "active_duration", "shape_duration", default=None)
    if duration is not None:
        return duration
    shape = getattr(event, "shape", None)
    if shape is not None:
        return get_float(shape, "duration", "active_duration", "shape_duration", default=None)
    return None


def occupied_duration(event: Any) -> float:
    if is_delay(event):
        return float(active_duration(event) or 0.0)
    delay = get_float(event, "delay", "tstart", "start", default=0.0) or 0.0
    dur = active_duration(event) or 0.0
    return float(delay + dur)


def protocol_values(sequence: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    protocol = getattr(sequence, "protocol", None)
    pvals = getattr(protocol, "parameters", None)
    if isinstance(pvals, Mapping):
        values.update({str(k): v for k, v in pvals.items()})
    svals = getattr(sequence, "parameters", None)
    if isinstance(svals, Mapping):
        for key, value in svals.items():
            if not str(key).startswith("_"):
                values.setdefault(str(key), value)
    return values
