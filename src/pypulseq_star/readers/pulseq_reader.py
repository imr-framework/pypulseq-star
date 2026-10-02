"""Pulseq ``.seq`` reader for pypulseq_star.

This module intentionally delegates parsing to PyPulseq.  It is the standalone
reader equivalent of the temporary ``PulseqWriter.read(...)`` compatibility
method and is kept separate now so writer and reader responsibilities can be
cleanly de-duplicated after the reader APIs stabilize.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pypulseq as pp


class PulseqReader:
    """Read a conventional Pulseq ``.seq`` file with PyPulseq."""

    @classmethod
    def read(
        cls,
        path: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> pp.Sequence:
        """Read a Pulseq ``.seq`` file and return a native PyPulseq sequence.

        Positional and keyword arguments are forwarded directly to the installed
        ``pypulseq.Sequence.read`` method.  This keeps pypulseq-star from
        duplicating version-specific parser behavior and provides an independent
        backend parser for compliance and regression testing.

        Notes
        -----
        The returned object is a native ``pypulseq.Sequence``.  This reader does
        not yet import a Pulseq file into the relationship-aware SeqStar object
        model.
        """

        del cls
        pulseq_seq = pp.Sequence()
        pulseq_seq.read(str(Path(path)), *args, **kwargs)
        return pulseq_seq
