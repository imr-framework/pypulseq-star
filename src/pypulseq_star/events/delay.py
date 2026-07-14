"""Delay event for pypulseq_star with central raster enforcement.

Drop this in as: src/pypulseq_star/events/delay.py

The important behavior is that both ``delay`` and ``duration`` are represented
as the same physical wait time and are always snapped to the active system
raster. This keeps make_delay(), relationship solving, check_timing(), Pulseq
writing, and gammaSTAR writing from each applying separate delay fixes.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Mapping

try:
    from pypulseq_star.core import SeqStarNode
except Exception:  # pragma: no cover - defensive during partial imports
    SeqStarNode = object  # type: ignore[misc,assignment]


def _get_mapping_value(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _snap_time_to_raster(
    value: float,
    raster: float,
    *,
    mode: str = "ceil",
    eps: float = 1e-12,
) -> float:
    """Snap a non-negative time to a raster."""

    value = float(value)
    raster = float(raster)

    if value < -eps:
        raise ValueError(f"Delay duration must be non-negative. Passed: {value}")

    value = max(value, 0.0)

    if raster <= 0:
        return value

    scaled = value / raster

    if mode == "ceil":
        snapped = math.ceil(scaled - eps) * raster
    elif mode == "nearest":
        snapped = round(scaled) * raster
    elif mode == "floor":
        snapped = math.floor(scaled + eps) * raster
    else:
        raise ValueError(f"Unsupported delay raster snap mode: {mode!r}")

    return max(float(snapped), 0.0)


class SeqStarDelayEvent(SeqStarNode):
    """A PyPulseq-like delay event with raster-safe duration semantics.

    A delay event has one physical time quantity. For compatibility with both
    PyPulseq-style code and PyPulseq-Star relationship code, it is exposed as
    both ``delay`` and ``duration``. Setting either one updates the same snapped
    value.

    The active raster is resolved in this order:
        1. explicit ``raster`` passed to the constructor
        2. bound system ``grad_raster_time``
        3. bound system ``block_duration_raster``
        4. protocol/parameters ``grad_raster_time`` or ``block_duration_raster``
        5. default 10 us
    """

    event_type = "delay"
    kind = "delay"
    type = "delay"

    def __init__(
        self,
        duration: float,
        *,
        name: str | None = None,
        role: str | None = None,
        system: Any | None = None,
        raster: float | None = None,
        snap_mode: str = "ceil",
        warn_on_snap: bool = False,
        parameters: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            super().__init__()  # type: ignore[misc]
        except Exception:
            pass

        self.name = name or "delay"
        self.role = role or "delay"
        self.system = system
        self.parameters = dict(parameters or {})
        self.metadata = dict(metadata or {})
        self.enabled = True
        self.shape = None

        self._explicit_raster = float(raster) if raster is not None else None
        self._snap_mode = str(snap_mode or "ceil")
        self._warn_on_snap = bool(warn_on_snap)
        self._duration = 0.0

        self.duration = float(duration)

    def bind_system(self, system: Any) -> None:
        """Bind scanner system context and re-snap the current duration."""

        self.system = system
        self.duration = self._duration

    @property
    def raster(self) -> float:
        """Return the raster used for delay duration snapping."""

        if self._explicit_raster is not None:
            return float(self._explicit_raster)

        system = getattr(self, "system", None)
        if system is not None:
            value = getattr(system, "grad_raster_time", None)
            if value is not None:
                return float(value)
            value = getattr(system, "block_duration_raster", None)
            if value is not None:
                return float(value)

        value = _get_mapping_value(
            getattr(self, "parameters", None),
            "grad_raster_time",
            "block_duration_raster",
            "raster",
            "delay_raster",
        )
        if value is not None:
            return float(value)

        return 10e-6

    def _snap(self, value: float) -> float:
        snapped = _snap_time_to_raster(
            float(value),
            self.raster,
            mode=self._snap_mode,
        )

        if self._warn_on_snap and abs(snapped - float(value)) > max(1e-12, 1e-6 * self.raster):
            warnings.warn(
                f"Snapped delay duration from {float(value):.12g} s to {snapped:.12g} s "
                f"to satisfy raster {self.raster:.12g} s.",
                stacklevel=3,
            )

        return snapped

    @property
    def duration(self) -> float:
        return float(self._duration)

    @duration.setter
    def duration(self, value: float) -> None:
        snapped = self._snap(float(value))
        self._duration = snapped
        self._sync_parameters(snapped)

    @property
    def delay(self) -> float:
        return float(self._duration)

    @delay.setter
    def delay(self, value: float) -> None:
        snapped = self._snap(float(value))
        self._duration = snapped
        self._sync_parameters(snapped)

    @property
    def tstart(self) -> float:
        return float(self._duration)

    @tstart.setter
    def tstart(self, value: float) -> None:
        self.delay = float(value)

    def _sync_parameters(self, value: float) -> None:
        if not hasattr(self, "parameters") or not isinstance(self.parameters, dict):
            self.parameters = {}
        self.parameters["delay"] = float(value)
        self.parameters["duration"] = float(value)
        self.parameters["tstart"] = float(value)
        self.parameters["raster"] = float(self.raster)
        self.parameters["snap_mode"] = self._snap_mode

    def to_simple_namespace(self) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            type="delay",
            delay=float(self.delay),
            duration=float(self.duration),
            name=self.name,
        )

    def to_pulseq(self) -> Any:
        try:
            import pypulseq as pp
            return pp.make_delay(float(self.duration))
        except Exception:
            return self.to_simple_namespace()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "delay",
            "name": self.name,
            "role": self.role,
            "delay": float(self.delay),
            "duration": float(self.duration),
            "raster": float(self.raster),
            "parameters": dict(getattr(self, "parameters", {}) or {}),
            "metadata": dict(getattr(self, "metadata", {}) or {}),
        }


__all__ = ["SeqStarDelayEvent"]
