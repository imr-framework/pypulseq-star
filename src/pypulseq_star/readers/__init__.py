"""File readers for pypulseq_star.

The reader package keeps backend import/parsing responsibilities separate from
backend writers.  Reader APIs are intentionally conservative: they preserve or
return the native backend representation first, while richer conversion into
SeqStar semantics can be added incrementally without changing basic read paths.
"""

from .gammastar_reader import GammaStarDocument, GammaStarReader
from .pulseq_reader import PulseqReader

__all__ = [
    "GammaStarDocument",
    "GammaStarReader",
    "PulseqReader",
]
