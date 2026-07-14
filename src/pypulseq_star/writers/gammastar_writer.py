"""Generic gammaSTAR JSON writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pypulseq_star.sequence import SeqStarSequence
from pypulseq_star.writers.gammastar_generic_document import (
    GenericGammaStarDocumentBuilder,
)


class GammaStarWriter:
    """Write gammaSTAR JSON from an enriched SeqStarSequence."""

    def __init__(self, sequence: SeqStarSequence) -> None:
        self.sequence = sequence

    def to_dict(self) -> dict[str, Any]:
        """Build a gammaSTAR JSON document from the enriched sequence graph."""

        return GenericGammaStarDocumentBuilder(self.sequence).to_dict()

    def write(self, path: str | Path) -> Path:
        """Write the gammaSTAR JSON document."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2))
        return output