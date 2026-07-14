"""Document-level gammaSTAR converter selection."""

from __future__ import annotations

from typing import Any

from pypulseq_star.sequence import SeqStarSequence

from .rf import GammaStarRFConverter


class GammaStarDocumentConverter:
    """Convert a SeqStar sequence into gammaSTAR sequence JSON."""

    def __init__(self, sequence: SeqStarSequence) -> None:
        self.sequence = sequence

    def to_dict(self) -> dict[str, Any]:
        """Return the gammaSTAR JSON document for the supported graph."""

        rf_converter = GammaStarRFConverter(self.sequence)
        if rf_converter.can_convert():
            return rf_converter.to_document()
        raise NotImplementedError(
            "No gammaSTAR converter is registered for this SeqStar graph. "
            "Start with supported building blocks or add a converter."
        )

