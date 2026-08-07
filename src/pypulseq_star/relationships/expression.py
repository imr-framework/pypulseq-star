"""Lightweight relationship expressions.

Phase-1 design
--------------
This module intentionally uses a small string-based expression wrapper. It is
simple enough for the first FID relationship milestone, but it gives us the API
surface we can later back with SymPy without changing user-facing scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class SeqStarExpression:
    """A symbolic/scriptable expression with optional resolved value.

    Parameters
    ----------
    canonical:
        Human-readable canonical expression, e.g.
        ``adc.start = rf.center + TE - 0.5 * adc.duration``.
    lua:
        gammaSTAR/Lua-compatible expression body or full return statement.
    inputs:
        Mapping from local variable names to semantic or gammaSTAR-like paths.
    resolved_value:
        Concrete value resolved for Pulseq ``.seq`` export.
    unit:
        Optional unit label for debug/reporting.
    metadata:
        Extra information for writers/debuggers.
    """

    canonical: str
    lua: str | None = None
    inputs: dict[str, str] = field(default_factory=dict)
    resolved_value: Any | None = None
    unit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_lua_script(self) -> str:
        """Return a gammaSTAR/Lua-style script string."""

        script = self.lua if self.lua is not None else self.canonical
        script = script.strip()

        if script.startswith("return "):
            return script

        return f"return {script}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "canonical": self.canonical,
            "lua": self.to_lua_script(),
            "inputs": dict(self.inputs),
            "resolved_value": self.resolved_value,
            "unit": self.unit,
            "metadata": dict(self.metadata),
        }


def expr(
    canonical: str,
    *,
    lua: str | None = None,
    inputs: Mapping[str, str] | None = None,
    resolved_value: Any | None = None,
    unit: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SeqStarExpression:
    """Create a lightweight relationship expression."""

    return SeqStarExpression(
        canonical=canonical,
        lua=lua,
        inputs=dict(inputs or {}),
        resolved_value=resolved_value,
        unit=unit,
        metadata=dict(metadata or {}),
    )
