"""Symbolic relationships for gammaSTAR-facing timing and parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import sympy as sp
except ImportError:  # pragma: no cover - only used when SymPy is unavailable.
    sp = None  # type: ignore[assignment]


@dataclass(slots=True)
class SeqStarExpression:
    """A symbolic expression with gammaSTAR JSON bindings.

    SeqStar keeps relationships readable in Python while exporting to the
    Lua-like ``return ...`` scripts expected by gammaSTAR sequence JSON.
    """

    expression: Any
    inputs: dict[str, str] = field(default_factory=dict)
    script_expression: str | None = None

    @classmethod
    def literal(cls, value: Any) -> "SeqStarExpression":
        """Wrap a fixed value as a gammaSTAR parameter script."""

        return cls(value)

    @classmethod
    def relation(cls, expression: str, **inputs: str) -> "SeqStarExpression":
        """Create a symbolic relationship from a compact expression string."""

        parsed = sp.sympify(expression) if sp is not None else expression
        return cls(parsed, dict(inputs), expression)

    def to_script(self) -> str:
        """Return a gammaSTAR-compatible script."""

        value = self.expression
        if self.script_expression is not None:
            return f"return {self.script_expression}"
        if isinstance(value, bool):
            return f"return {str(value).lower()}"
        if isinstance(value, str):
            if self.inputs:
                return f"return {value}"
            return f"return {value!r}"
        if isinstance(value, (int, float)):
            return f"return {value:.17g}"
        if sp is not None and isinstance(value, sp.Basic):
            return f"return {sp.ccode(value)}"
        if isinstance(value, (list, dict)):
            return "return " + _lua_table(value)
        return f"return {value!r}"

    def to_gammastar_parameter(self) -> dict[str, Any]:
        """Return the ``inputs``/``script`` object used in gammaSTAR JSON."""

        return {"inputs": self.inputs, "script": self.to_script()}


def _lua_table(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{value:.17g}"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, list):
        return "{" + ", ".join(_lua_table(item) for item in value) + "}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}={_lua_table(item)}" for key, item in value.items()) + "}"
    return repr(value)
