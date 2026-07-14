"""Protocol-parameter helpers for relationships."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class MissingProtocolParameterError(KeyError):
    """Raised when a required protocol parameter is missing."""


def require_protocol_parameters(seq: Any, names: list[str] | tuple[str, ...]) -> None:
    """Validate that sequence parameters contain required names."""

    parameters = getattr(seq, "parameters", None)
    if not isinstance(parameters, Mapping):
        raise MissingProtocolParameterError(
            "Sequence does not expose a parameters mapping for relationship validation."
        )

    missing = [name for name in names if name not in parameters]
    if missing:
        raise MissingProtocolParameterError(
            f"Missing required protocol parameter(s): {', '.join(missing)}"
        )


def is_protocol_reference(value: Any) -> bool:
    """Return True for string values intended to reference protocol parameters."""

    return isinstance(value, str)
