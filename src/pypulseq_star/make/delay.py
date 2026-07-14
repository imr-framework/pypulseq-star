"""PyPulseq-compatible delay constructor with central raster enforcement.

Drop this in as: src/pypulseq_star/make/delay.py

This keeps examples short: callers can continue to write

    ppstar.make_delay(te_delay)

and the delay event itself will enforce raster alignment at construction time
and whenever a relationship resolver later updates ``duration`` or ``delay``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pypulseq_star.events.delay import SeqStarDelayEvent


def make_delay(
    delay: float,
    *,
    name: str | None = None,
    role: str | None = None,
    system: Any | None = None,
    raster: float | None = None,
    snap_mode: str = "ceil",
    warn_on_snap: bool = False,
    parameters: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SeqStarDelayEvent:
    """Create a raster-safe delay event.

    Parameters
    ----------
    delay
        Requested delay duration in seconds. The returned event snaps it to the
        appropriate raster immediately.
    system
        Optional scanner/system object. If provided, ``system.grad_raster_time``
        is used as the delay raster. If omitted, the event uses a 10 us default
        until it is attached to a Sequence, at which point ``bind_system``
        re-snaps to the actual sequence raster.
    raster
        Optional explicit raster override in seconds.
    snap_mode
        ``"ceil"`` by default, matching the safe sequence-design convention that
        delays should not be shortened accidentally. ``"nearest"`` and
        ``"floor"`` are available for advanced/debug use.
    warn_on_snap
        False by default to avoid verbose logs during relationship solving.
    """

    return SeqStarDelayEvent(
        float(delay),
        name=name,
        role=role,
        system=system,
        raster=raster,
        snap_mode=snap_mode,
        warn_on_snap=warn_on_snap,
        parameters=parameters,
        metadata=metadata,
    )


__all__ = ["make_delay"]
