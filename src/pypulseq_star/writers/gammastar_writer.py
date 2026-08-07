"""Generic gammaSTAR JSON writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pypulseq_star.sequence import SeqStarSequence
from pypulseq_star.writers._realization import sequence_view
from pypulseq_star.writers.gammastar_generic_document import (
    GenericGammaStarDocumentBuilder,
)


class GammaStarWriter:
    """Write gammaSTAR JSON from a symbolic sequence and optional realization."""

    def __init__(self, sequence: SeqStarSequence) -> None:
        self.sequence = sequence

    def to_dict(
        self,
        *,
        realization: Any | None = None,
        defaults: Any | None = None,
    ) -> dict[str, Any]:
        """Build a gammaSTAR JSON document."""

        view = sequence_view(
            self.sequence,
            realization=realization,
            defaults=defaults,
        )

        return GenericGammaStarDocumentBuilder(
            view.export_sequence,
            symbolic_sequence=view.symbolic_sequence,
            realization=view.realization,
            auto_resolve_relationships=not view.is_resolved,
        ).to_dict()

    def write(
        self,
        path: str | Path,
        *,
        realization: Any | None = None,
        defaults: Any | None = None,
    ) -> Path:
        """Write the gammaSTAR JSON document."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        document = self.to_dict(
            realization=realization,
            defaults=defaults,
        )
        output.write_text(json.dumps(document, indent=2))
        return output
