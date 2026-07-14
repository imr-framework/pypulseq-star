"""PyPulseq-like gradient constructors for pypulseq_star.

The public API mirrors PyPulseq where practical:

- ``make_trapezoid``
- ``make_arbitrary_grad``
- ``make_arbitrary_gradient`` alias
- ``split_gradient``
- ``make_split_gradient`` alias

The constructors return enriched ``SeqStarGradientEvent`` objects while keeping
PyPulseq-compatible fields available through event properties and
``event.to_pulseq()``.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal

import numpy as np

from pypulseq_star.events.grad import SeqStarGradientEvent
from pypulseq_star.shapes.grad import (
    SeqStarArbitraryGradientShape,
    SeqStarSplitGradientShape,
    SeqStarTrapezoidGradientShape,
)

_EPS = 1e-12


def _default_system() -> Any:
    try:
        from pypulseq.opts import Opts

        return Opts.default
    except Exception:  # pragma: no cover - only used without pypulseq
        class _FallbackSystem:
            max_grad = 28e6
            max_slew = 120e6
            grad_raster_time = 10e-6

        return _FallbackSystem()


def _system_value(system: Any, name: str, default: float) -> float:
    return float(getattr(system, name, default))


def _ceil_to_raster(value: float, raster: float) -> float:
    return math.ceil(float(value) / raster - 1e-12) * raster


def _round_to_raster(value: float, raster: float) -> float:
    return round(float(value) / raster) * raster


def calculate_shortest_params_for_area(
    area: float,
    max_slew: float,
    max_grad: float,
    grad_raster_time: float,
) -> tuple[float, float, float, float]:
    """Calculate a short rastered trapezoid for the requested area."""

    if area == 0:
        return 0.0, grad_raster_time, 0.0, grad_raster_time

    rise_time = _ceil_to_raster(math.sqrt(abs(area) / max_slew), grad_raster_time)
    rise_time = max(rise_time, grad_raster_time)
    amplitude = area / rise_time
    effective_time = rise_time

    if abs(amplitude) > max_grad + _EPS:
        effective_time = _ceil_to_raster(abs(area) / max_grad, grad_raster_time)
        amplitude = area / effective_time
        rise_time = _ceil_to_raster(abs(amplitude) / max_slew, grad_raster_time)
        rise_time = max(rise_time, grad_raster_time)

    flat_time = max(0.0, effective_time - rise_time)
    fall_time = rise_time
    return amplitude, rise_time, flat_time, fall_time


def calculate_shortest_rise_time(amplitude: float, max_slew: float, grad_raster_time: float) -> float:
    """Calculate the shortest rastered rise/fall time for an amplitude."""

    return _ceil_to_raster(max(abs(amplitude) / max_slew, grad_raster_time), grad_raster_time)


def make_trapezoid(
    channel: str,
    amplitude: float | None = None,
    area: float | None = None,
    delay: float = 0.0,
    duration: float | None = None,
    fall_time: float | None = None,
    flat_area: float | None = None,
    flat_time: float | None = None,
    max_grad: float | None = None,
    max_slew: float | None = None,
    rise_time: float | None = None,
    system: Any | None = None,
    *,
    name: str | None = None,
    role: str | None = None,
    axis_role: str | None = None,
    encoding_role: str | None = None,
    polarity: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> SeqStarGradientEvent:
    """Create an enriched trapezoidal gradient event.

    The accepted input combinations intentionally follow PyPulseq's
    ``make_trapezoid`` behavior for the common cases: area-based, amplitude-
    based, and flat-area-based construction.
    """

    if system is None:
        system = _default_system()
    grad_raster_time = _system_value(system, "grad_raster_time", 10e-6)
    if max_grad is None:
        max_grad = _system_value(system, "max_grad", 28e6)
    if max_slew is None:
        max_slew = _system_value(system, "max_slew", 120e6)

    if channel not in {"x", "y", "z"}:
        raise ValueError(f"Invalid channel. Must be one of 'x', 'y', or 'z'. Passed: {channel!r}")

    if rise_time is None and fall_time is not None:
        rise_time = fall_time
    if fall_time is None and rise_time is not None:
        fall_time = rise_time

    calc_path: Literal["area", "flat_area", "amplitude"]
    if area is not None and flat_area is None and amplitude is None:
        calc_path = "area"
    elif area is None and flat_area is not None and amplitude is None:
        calc_path = "flat_area"
    elif area is None and flat_area is None and amplitude is not None:
        calc_path = "amplitude"
    elif area is None and flat_area is not None and amplitude is not None:
        raise NotImplementedError("Flat area + amplitude input pair is not implemented yet.")
    elif area is not None and flat_area is None and amplitude is not None:
        raise NotImplementedError("Amplitude + area input pair is not implemented yet.")
    else:
        raise ValueError("Must supply exactly one of 'area', 'flat_area', or 'amplitude'.")

    if flat_time is not None and flat_area is None and amplitude is None and (rise_time is None or area is None):
        raise ValueError(
            "When flat_time is provided, supply flat_area, amplitude, or both rise_time and area."
        )

    if calc_path == "area":
        assert area is not None
        if duration is not None and flat_time is None:
            if rise_time is None:
                _, rise_time, flat_time, fall_time = calculate_shortest_params_for_area(
                    area, max_slew, max_grad, grad_raster_time
                )
            assert rise_time is not None
            if fall_time is None:
                fall_time = rise_time
            min_duration = rise_time + fall_time
            if duration < min_duration - _EPS:
                raise ValueError(
                    f"Requested area is too large for this gradient. Minimum duration is {min_duration:.9g} s."
                )
            flat_time = duration - rise_time - fall_time
            amplitude2 = area / (0.5 * rise_time + flat_time + 0.5 * fall_time)
        elif flat_time is not None:
            if rise_time is None:
                raise ValueError("Must supply rise_time when area and flat_time are provided.")
            if fall_time is None:
                fall_time = rise_time
            amplitude2 = area / (0.5 * rise_time + flat_time + 0.5 * fall_time)
        else:
            if rise_time is not None or fall_time is not None:
                warnings.warn(
                    "rise_time and fall_time are ignored when calculating shortest duration from area.",
                    stacklevel=2,
                )
            amplitude2, rise_time, flat_time, fall_time = calculate_shortest_params_for_area(
                area, max_slew, max_grad, grad_raster_time
            )

    elif calc_path == "flat_area":
        assert flat_area is not None
        if duration is not None:
            raise NotImplementedError("Flat area + duration input pair is not implemented yet.")
        if flat_time is None:
            raise ValueError("flat_area requires flat_time.")
        amplitude2 = flat_area / flat_time
        if rise_time is None:
            rise_time = calculate_shortest_rise_time(amplitude2, max_slew, grad_raster_time)
        if fall_time is None:
            fall_time = rise_time

    else:  # amplitude
        assert amplitude is not None
        amplitude2 = float(amplitude)
        if rise_time is None:
            rise_time = calculate_shortest_rise_time(amplitude2, max_slew, grad_raster_time)
        if fall_time is None:
            fall_time = rise_time
        if duration is not None and flat_time is None:
            flat_time = duration - rise_time - fall_time
        elif flat_time is not None and duration is None:
            pass
        else:
            raise ValueError("Amplitude construction requires duration or flat_time.")

    assert rise_time is not None
    assert flat_time is not None
    assert fall_time is not None

    rise_time = _round_to_raster(rise_time, grad_raster_time)
    flat_time = _round_to_raster(flat_time, grad_raster_time)
    fall_time = _round_to_raster(fall_time, grad_raster_time)
    delay = _round_to_raster(delay, grad_raster_time)

    if flat_time < -_EPS:
        raise ValueError("Computed flat_time is negative; requested gradient is too short.")
    flat_time = max(0.0, flat_time)

    if abs(amplitude2) > max_grad + _EPS:
        raise ValueError(f"Refined amplitude ({abs(amplitude2):.6g}) is larger than max_grad ({max_grad:.6g}).")
    if rise_time > 0 and abs(amplitude2) / rise_time > max_slew * (1 + _EPS):
        raise ValueError("Slew rate violation during ramp up.")
    if fall_time > 0 and abs(amplitude2) / fall_time > max_slew * (1 + _EPS):
        raise ValueError("Slew rate violation during ramp down.")

    shape = SeqStarTrapezoidGradientShape(
        channel=channel,  # type: ignore[arg-type]
        delay=delay,
        grad_raster_time=grad_raster_time,
        amplitude=amplitude2,
        rise_time=rise_time,
        flat_time=flat_time,
        fall_time=fall_time,
        role=role,
        metadata=dict(metadata or {}),
    )
    return SeqStarGradientEvent(
        shape=shape,
        name=name,
        role=role,
        axis_role=axis_role,
        encoding_role=encoding_role,
        polarity=polarity,
        metadata=dict(metadata or {}),
        system=system,
    )


def make_arbitrary_grad(
    channel: str,
    waveform: np.ndarray | list[float],
    first: float | None = None,
    last: float | None = None,
    delay: float = 0.0,
    max_grad: float | None = None,
    max_slew: float | None = None,
    system: Any | None = None,
    oversampling: bool = False,
    *,
    name: str | None = None,
    role: str | None = None,
    axis_role: str | None = None,
    encoding_role: str | None = None,
    polarity: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> SeqStarGradientEvent:
    """Create an enriched arbitrary gradient event."""

    if system is None:
        system = _default_system()
    grad_raster_time = _system_value(system, "grad_raster_time", 10e-6)
    if max_grad is None or max_grad == 0:
        max_grad = _system_value(system, "max_grad", 28e6)
    if max_slew is None or max_slew == 0:
        max_slew = _system_value(system, "max_slew", 120e6)

    if channel not in {"x", "y", "z"}:
        raise ValueError(f"Invalid channel. Must be one of 'x', 'y', or 'z'. Passed: {channel!r}")

    delay = _round_to_raster(delay, grad_raster_time)
    shape = SeqStarArbitraryGradientShape(
        channel=channel,  # type: ignore[arg-type]
        delay=delay,
        grad_raster_time=grad_raster_time,
        samples=np.asarray(waveform, dtype=float),
        first_value=first,
        last_value=last,
        oversampling=oversampling,
        role=role,
        metadata=dict(metadata or {}),
    )
    shape.validate(max_grad=max_grad, max_slew=max_slew)
    return SeqStarGradientEvent(
        shape=shape,
        name=name,
        role=role,
        axis_role=axis_role,
        encoding_role=encoding_role,
        polarity=polarity,
        metadata=dict(metadata or {}),
        system=system,
    )


def make_arbitrary_gradient(*args: Any, **kwargs: Any) -> SeqStarGradientEvent:
    """Readable alias for ``make_arbitrary_grad``."""

    return make_arbitrary_grad(*args, **kwargs)


def split_gradient(
    grad: SeqStarGradientEvent,
    system: Any | None = None,
    *,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[SeqStarGradientEvent, SeqStarGradientEvent, SeqStarGradientEvent]:
    """Split a trapezoidal gradient into ramp-up, flat-top, and ramp-down events.

    The returned parts are arbitrary gradient events with delays chosen so they
    can be grouped or visualized independently while preserving the timing of
    the original trapezoid.
    """

    if system is None:
        system = grad.system if getattr(grad, "system", None) is not None else _default_system()
    grad_raster_time = _system_value(system, "grad_raster_time", 10e-6)

    if grad.type != "trap":
        raise ValueError("Splitting of arbitrary or unsupported gradients is not implemented yet.")

    parent = grad.shape
    if not isinstance(parent, SeqStarTrapezoidGradientShape):
        raise ValueError("split_gradient expects a trapezoid SeqStarGradientEvent.")

    delay = _round_to_raster(parent.delay, grad_raster_time)
    rise_time = _round_to_raster(parent.rise_time, grad_raster_time)
    flat_time = _round_to_raster(parent.flat_time, grad_raster_time)
    fall_time = _round_to_raster(parent.fall_time, grad_raster_time)
    amp = parent.amplitude

    def _two_sample_part(samples: list[float], part_delay: float, part_name: str) -> SeqStarGradientEvent:
        return make_arbitrary_grad(
            channel=parent.channel,
            waveform=np.asarray(samples, dtype=float),
            first=samples[0],
            last=samples[-1],
            delay=part_delay,
            system=system,
            name=part_name,
            role=grad.role,
            axis_role=grad.axis_role,
            encoding_role=grad.encoding_role,
            polarity=grad.polarity,
            metadata={**grad.metadata, **dict(metadata or {}), "split_parent": grad.name},
        )

    ramp_up = _two_sample_part([0.0, amp], delay, f"{name or grad.name}_ramp_up")
    flat_top = _two_sample_part([amp, amp], delay + rise_time, f"{name or grad.name}_flat_top")
    ramp_down = _two_sample_part([amp, 0.0], delay + rise_time + flat_time, f"{name or grad.name}_ramp_down")

    split_shape = SeqStarSplitGradientShape(
        parent=parent,
        ramp_up=ramp_up.shape,  # type: ignore[arg-type]
        flat_top=flat_top.shape,  # type: ignore[arg-type]
        ramp_down=ramp_down.shape,  # type: ignore[arg-type]
        metadata={"source_event": grad.name, **dict(metadata or {})},
    )
    grad.metadata["split_shape"] = split_shape.to_dict()
    return ramp_up, flat_top, ramp_down


def make_split_gradient(*args: Any, **kwargs: Any) -> tuple[SeqStarGradientEvent, SeqStarGradientEvent, SeqStarGradientEvent]:
    """Alias for ``split_gradient``."""

    return split_gradient(*args, **kwargs)
