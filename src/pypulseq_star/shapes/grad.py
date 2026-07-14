"""Gradient shapes for pypulseq_star.

This module contains waveform geometry only. It does not own sequence placement,
block composition, or file writing. Public constructors live in
``pypulseq_star.make.grad`` and return ``SeqStarGradientEvent`` objects that
wrap these shapes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

GradientChannel = Literal["x", "y", "z"]
GradientKind = Literal["trap", "grad", "split"]

_VALID_CHANNELS = {"x", "y", "z"}


def _as_float(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:  # pragma: no cover - defensive path
        raise TypeError(f"{name} must be convertible to float. Got {value!r}.") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite. Got {out!r}.")
    return out


def _validate_channel(channel: str) -> GradientChannel:
    if channel not in _VALID_CHANNELS:
        raise ValueError(f"Invalid gradient channel. Must be one of 'x', 'y', or 'z'. Passed: {channel!r}")
    return channel  # type: ignore[return-value]


@dataclass(slots=True)
class SeqStarGradientShape:
    """Base class for gradient waveform geometry.

    Parameters
    ----------
    channel
        Gradient channel, one of ``"x"``, ``"y"`` or ``"z"``.
    delay
        Delay before the active gradient shape, in seconds.
    grad_raster_time
        Gradient raster time in seconds.
    kind
        Shape kind. ``"trap"`` for trapezoids, ``"grad"`` for arbitrary
        waveforms, and ``"split"`` for split-gradient containers.
    role
        Optional semantic role, for example ``"readout"`` or ``"spoiler"``.
    metadata
        Free-form metadata that should survive object JSON export.
    """

    channel: GradientChannel
    delay: float = 0.0
    grad_raster_time: float = 10e-6
    kind: GradientKind = "grad"
    role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.channel = _validate_channel(self.channel)
        self.delay = _as_float(self.delay, name="delay")
        self.grad_raster_time = _as_float(self.grad_raster_time, name="grad_raster_time")
        if self.delay < 0:
            raise ValueError("Gradient delay must be non-negative.")
        if self.grad_raster_time <= 0:
            raise ValueError("Gradient raster time must be positive.")

    @property
    def active_duration(self) -> float:
        """Duration of the active shape, excluding delay."""

        raise NotImplementedError

    @property
    def duration(self) -> float:
        """Total event duration, including delay."""

        return self.delay + self.active_duration

    @property
    def area(self) -> float:
        """Gradient area in Pulseq units, excluding delay."""

        raise NotImplementedError

    @property
    def first(self) -> float:
        """Gradient value at the beginning of the event."""

        raise NotImplementedError

    @property
    def last(self) -> float:
        """Gradient value at the end of the event."""

        raise NotImplementedError

    def waveform(self) -> np.ndarray:
        """Return center-sampled waveform values on the gradient raster."""

        raise NotImplementedError

    def time_axis(self) -> np.ndarray:
        """Return center-sample times for ``waveform()``, excluding delay."""

        wf = self.waveform()
        return (np.arange(len(wf), dtype=float) + 0.5) * self.grad_raster_time

    def validate(self, *, max_grad: float | None = None, max_slew: float | None = None) -> None:
        """Validate timing, amplitude, and slew limits if limits are supplied."""

        wf = np.asarray(self.waveform(), dtype=float)
        if wf.ndim != 1:
            raise ValueError("Gradient waveform must be one-dimensional.")
        if len(wf) == 0:
            raise ValueError("Gradient waveform cannot be empty.")
        if max_grad is not None and np.max(np.abs(wf)) > float(max_grad) * (1 + 1e-12):
            raise ValueError(
                f"Gradient amplitude violation: {np.max(np.abs(wf)):.6g} > {float(max_grad):.6g}."
            )
        if max_slew is not None:
            edge_values = np.concatenate(([self.first], wf, [self.last]))
            slew = np.diff(edge_values) / self.grad_raster_time
            if np.max(np.abs(slew)) > float(max_slew) * (1 + 1e-12):
                raise ValueError(
                    f"Gradient slew violation: {np.max(np.abs(slew)):.6g} > {float(max_slew):.6g}."
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize shape metadata and waveform summary."""

        wf = np.asarray(self.waveform(), dtype=float)
        return {
            "class": self.__class__.__name__,
            "kind": self.kind,
            "channel": self.channel,
            "delay": self.delay,
            "grad_raster_time": self.grad_raster_time,
            "active_duration": self.active_duration,
            "duration": self.duration,
            "area": self.area,
            "first": self.first,
            "last": self.last,
            "num_samples": int(len(wf)),
            "waveform": wf.tolist(),
            "role": self.role,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class SeqStarTrapezoidGradientShape(SeqStarGradientShape):
    """Trapezoidal gradient geometry."""

    amplitude: float = 0.0
    rise_time: float = 0.0
    flat_time: float = 0.0
    fall_time: float = 0.0
    kind: GradientKind = "trap"

    def __post_init__(self) -> None:
        SeqStarGradientShape.__post_init__(self)
        self.amplitude = _as_float(self.amplitude, name="amplitude")
        self.rise_time = _as_float(self.rise_time, name="rise_time")
        self.flat_time = _as_float(self.flat_time, name="flat_time")
        self.fall_time = _as_float(self.fall_time, name="fall_time")
        if self.rise_time < 0 or self.flat_time < 0 or self.fall_time < 0:
            raise ValueError("Trapezoid rise_time, flat_time, and fall_time must be non-negative.")
        if self.active_duration <= 0:
            raise ValueError("Trapezoid active duration must be positive.")

    @property
    def active_duration(self) -> float:
        return self.rise_time + self.flat_time + self.fall_time

    @property
    def flat_area(self) -> float:
        return self.amplitude * self.flat_time

    @property
    def area(self) -> float:
        return self.amplitude * (self.flat_time + 0.5 * self.rise_time + 0.5 * self.fall_time)

    @property
    def first(self) -> float:
        return 0.0

    @property
    def last(self) -> float:
        return 0.0

    def waveform(self) -> np.ndarray:
        """Return a center-sampled trapezoid waveform on the gradient raster."""

        n = max(1, int(round(self.active_duration / self.grad_raster_time)))
        t = (np.arange(n, dtype=float) + 0.5) * self.grad_raster_time
        wf = np.zeros(n, dtype=float)
        rise = self.rise_time
        flat_end = self.rise_time + self.flat_time
        end = self.active_duration

        if rise > 0:
            rise_mask = t < rise
            wf[rise_mask] = self.amplitude * t[rise_mask] / rise
        else:
            rise_mask = np.zeros_like(t, dtype=bool)

        flat_mask = (t >= rise) & (t < flat_end)
        wf[flat_mask] = self.amplitude

        fall_mask = (t >= flat_end) & (t <= end)
        if self.fall_time > 0:
            wf[fall_mask] = self.amplitude * (1.0 - (t[fall_mask] - flat_end) / self.fall_time)
        else:
            wf[fall_mask] = self.amplitude

        return wf

    def to_dict(self) -> dict[str, Any]:
        out = SeqStarGradientShape.to_dict(self)
        out.update(
            {
                "amplitude": self.amplitude,
                "rise_time": self.rise_time,
                "flat_time": self.flat_time,
                "fall_time": self.fall_time,
                "flat_area": self.flat_area,
            }
        )
        return out


@dataclass(slots=True)
class SeqStarArbitraryGradientShape(SeqStarGradientShape):
    """Arbitrary center-sampled gradient waveform."""

    samples: np.ndarray | list[float] = field(default_factory=list)
    first_value: float | None = None
    last_value: float | None = None
    oversampling: bool = False
    kind: GradientKind = "grad"

    def __post_init__(self) -> None:
        SeqStarGradientShape.__post_init__(self)
        wf = np.asarray(self.samples, dtype=float)
        if wf.ndim != 1:
            raise ValueError("Arbitrary gradient waveform must be one-dimensional.")
        if len(wf) < 2:
            raise ValueError("Arbitrary gradient waveform must contain at least two samples.")
        if not np.all(np.isfinite(wf)):
            raise ValueError("Arbitrary gradient waveform contains non-finite values.")
        if self.oversampling and len(wf) % 2 == 0:
            raise ValueError("When oversampling is active, waveform must have an odd number of samples.")
        self.samples = wf
        if self.first_value is None:
            self.first_value = self._extrapolate(wf[0], wf[1])
        else:
            self.first_value = _as_float(self.first_value, name="first")
        if self.last_value is None:
            self.last_value = self._extrapolate(wf[-1], wf[-2])
        else:
            self.last_value = _as_float(self.last_value, name="last")

    def _extrapolate(self, a: float, b: float) -> float:
        if self.oversampling:
            return float(2 * a - b)
        return float(0.5 * (3 * a - b))

    @property
    def active_duration(self) -> float:
        if self.oversampling:
            return 0.5 * (len(self.samples) + 1) * self.grad_raster_time
        return len(self.samples) * self.grad_raster_time

    @property
    def area(self) -> float:
        wf = np.asarray(self.samples, dtype=float)
        if self.oversampling:
            return float(np.sum(wf[::2]) * self.grad_raster_time)
        return float(np.sum(wf) * self.grad_raster_time)

    @property
    def first(self) -> float:
        assert self.first_value is not None
        return self.first_value

    @property
    def last(self) -> float:
        assert self.last_value is not None
        return self.last_value

    def waveform(self) -> np.ndarray:
        return np.asarray(self.samples, dtype=float)

    def time_axis(self) -> np.ndarray:
        if self.oversampling:
            return np.arange(1, len(self.samples) + 1, dtype=float) * 0.5 * self.grad_raster_time
        return SeqStarGradientShape.time_axis(self)

    def to_dict(self) -> dict[str, Any]:
        out = SeqStarGradientShape.to_dict(self)
        out.update({"oversampling": self.oversampling, "tt": self.time_axis().tolist()})
        return out


@dataclass(slots=True)
class SeqStarSplitGradientShape:
    """Container for split-gradient parts.

    This is not a primitive Pulseq shape. It preserves the semantic fact that a
    parent trapezoid has been split into ramp-up, flat-top, and ramp-down parts.
    """

    parent: SeqStarTrapezoidGradientShape
    ramp_up: SeqStarArbitraryGradientShape
    flat_top: SeqStarArbitraryGradientShape
    ramp_down: SeqStarArbitraryGradientShape
    kind: GradientKind = "split"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def parts(self) -> tuple[SeqStarArbitraryGradientShape, SeqStarArbitraryGradientShape, SeqStarArbitraryGradientShape]:
        return self.ramp_up, self.flat_top, self.ramp_down

    @property
    def duration(self) -> float:
        return self.parent.duration

    @property
    def channel(self) -> GradientChannel:
        return self.parent.channel

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "kind": self.kind,
            "channel": self.channel,
            "duration": self.duration,
            "parent": self.parent.to_dict(),
            "parts": [part.to_dict() for part in self.parts],
            "metadata": dict(self.metadata),
        }
