"""Writer-boundary helpers for symbolic sequences and numeric realizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class WriterSequenceView:
    symbolic_sequence: Any
    export_sequence: Any
    realization: Any | None
    is_resolved: bool


def sequence_view(
    sequence: Any,
    *,
    realization: Any | None = None,
    defaults: Any | None = None,
) -> WriterSequenceView:
    """Normalize symbolic sequence and optional numeric realization inputs."""

    if realization is not None and defaults is not None and realization is not defaults:
        raise ValueError(
            "Pass either realization= or defaults=, not two different objects."
        )

    selected = realization if realization is not None else defaults

    if selected is None:
        return WriterSequenceView(
            symbolic_sequence=sequence,
            export_sequence=sequence,
            realization=None,
            is_resolved=False,
        )

    export_sequence = getattr(selected, "sequence", selected)

    if export_sequence is None:
        raise TypeError(
            "The supplied realization/defaults object does not contain a numeric sequence."
        )

    if not hasattr(export_sequence, "timeline") or not hasattr(export_sequence, "system"):
        raise TypeError(
            "The supplied realization/defaults object is not sequence-like."
        )

    return WriterSequenceView(
        symbolic_sequence=sequence,
        export_sequence=export_sequence,
        realization=selected,
        is_resolved=True,
    )
