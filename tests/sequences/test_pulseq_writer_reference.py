"""PulseqWriter regression tests against a trusted PyPulseq GRE reference.

The reference file ``write_gre.seq`` is copied from PyPulseq's expected-output
suite.  These tests deliberately parse both files with PyPulseq, so the writer
is checked by an independent backend implementation rather than by inspecting
our own internal objects.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pypulseq_star.writers import PulseqWriter

from .helpers import load_demo

pytestmark = [pytest.mark.sequence, pytest.mark.gre]


def _first_rf(sequence):
    """Return the first RF event from a native PyPulseq Sequence."""

    for block_id in sorted(sequence.block_events):
        block = sequence.get_block(block_id)
        rf = getattr(block, "rf", None)
        if rf is not None:
            return rf
    raise AssertionError("No RF event found in parsed Pulseq sequence.")


def _normalized_rf_magnitude(rf, *, n: int = 1001) -> np.ndarray:
    """Return peak-normalized RF magnitude on a normalized time grid.

    The PyPulseq reference currently uses a 1 us RF raster while the STAR GRE
    demo may use a different RF raster.  Interpolating on normalized pulse time
    compares the pulse *shape* rather than serialization/sample count.
    """

    magnitude = np.abs(np.asarray(rf.signal, dtype=complex)).astype(float)
    assert magnitude.ndim == 1
    assert magnitude.size >= 2
    peak = float(np.max(magnitude))
    assert peak > 0.0

    source_t = np.linspace(0.0, 1.0, magnitude.size)
    target_t = np.linspace(0.0, 1.0, n)
    return np.interp(target_t, source_t, magnitude / peak)


def test_pulseq_writer_read_uses_pypulseq_parser(repo_root: Path) -> None:
    """PulseqWriter.read should expose the native PyPulseq read path."""

    reference_path = repo_root / "tests" / "sequences" / "write_gre.seq"
    assert reference_path.exists(), reference_path

    demo = load_demo(repo_root, "GRE")
    protocol = demo.define_protocol()
    seq, _, _ = demo.build_sequence(demo.define_system_limits(), protocol)

    parsed = PulseqWriter(seq).read(reference_path)

    assert parsed.block_events
    assert _first_rf(parsed) is not None

    ok, report = parsed.check_timing()
    assert ok, "\n".join(str(item) for item in report)


def test_pulseq_writer_gre_rf_matches_pypulseq_reference_shape(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    """Regression: PulseqWriter must not turn the GRE sinc RF into a block pulse.

    This catches the writer bug in which a valid STAR sinc excitation was
    exported through ``make_block_pulse``.  The demo itself is not the oracle:
    both the trusted reference and the STAR-written file are independently
    decoded by PyPulseq and their RF magnitude envelopes are compared.
    """

    reference_path = repo_root / "tests" / "sequences" / "write_gre.seq"
    assert reference_path.exists(), reference_path

    demo = load_demo(repo_root, "GRE")
    protocol = demo.define_protocol()
    seq, _, _ = demo.build_sequence(demo.define_system_limits(), protocol)
    resolved = seq.resolve()

    writer = PulseqWriter(seq)
    candidate_path = writer.write(
        tmp_path / "gre_star.seq",
        realization=resolved,
    )

    reference = writer.read(reference_path)
    candidate = writer.read(candidate_path)

    # The generated file must first be a valid Pulseq sequence according to
    # PyPulseq's own timing checker.
    ok, report = candidate.check_timing()
    assert ok, "\n".join(str(item) for item in report)

    reference_rf = _first_rf(reference)
    candidate_rf = _first_rf(candidate)

    reference_mag = _normalized_rf_magnitude(reference_rf)
    candidate_mag = _normalized_rf_magnitude(candidate_rf)

    # A rectangular/block RF has essentially zero variation and should fail
    # before the more specific reference-shape comparison below.
    assert np.ptp(candidate_mag) > 0.25, (
        "PulseqWriter collapsed the GRE excitation to an approximately "
        "rectangular RF envelope."
    )

    # The STAR GRE uses the same sinc-family parameters as the PyPulseq GRE
    # reference.  Raster/sample-count differences are intentionally removed by
    # interpolation above, so this checks waveform semantics rather than text.
    correlation = float(np.corrcoef(reference_mag, candidate_mag)[0, 1])
    assert correlation > 0.995, (
        "PulseqWriter did not preserve the expected GRE sinc RF shape: "
        f"normalized-shape correlation={correlation:.6f}."
    )
