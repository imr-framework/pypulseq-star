"""Regression tests for the standalone PulseqReader.

These tests use the trusted PyPulseq GRE reference sequence stored under
``tests/sequences/write_gre.seq``.  The goal is to verify that the public
pypulseq-star reader behaves like PyPulseq's native ``Sequence.read`` and
returns a fully usable native PyPulseq Sequence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pypulseq as pp

from pypulseq_star.readers import PulseqReader

SEQUENCE_DIR = Path(__file__).resolve().parents[2] / "sequences"
REFERENCE_SEQ = SEQUENCE_DIR / "write_gre.seq"


def _first_rf(sequence: pp.Sequence):
    """Return the first RF event in a parsed PyPulseq sequence."""

    for block_id in sequence.block_events:
        block = sequence.get_block(block_id)
        rf = getattr(block, "rf", None)
        if rf is not None:
            return rf

    raise AssertionError("Reference Pulseq sequence contains no RF event.")


def test_pulseq_reader_matches_native_pypulseq_read() -> None:
    """PulseqReader should be a faithful wrapper around PyPulseq parsing."""

    assert REFERENCE_SEQ.exists(), (
        f"Missing Pulseq reference sequence: {REFERENCE_SEQ}"
    )

    # Read through pypulseq-star.
    parsed = PulseqReader.read(REFERENCE_SEQ)

    # Read independently through PyPulseq itself.
    native = pp.Sequence()
    native.read(str(REFERENCE_SEQ))

    assert isinstance(parsed, pp.Sequence)

    # The same file should produce the same basic parsed sequence structure.
    assert len(parsed.block_events) == len(native.block_events)

    def _assert_definitions_equal(
        actual: dict,
        expected: dict,
    ) -> None:
        assert actual.keys() == expected.keys()

        for key in actual:
            a = actual[key]
            b = expected[key]

            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                np.testing.assert_array_equal(
                    np.asarray(a),
                    np.asarray(b),
                    err_msg=f"Definition mismatch for {key!r}",
                )
            else:
                assert a == b, f"Definition mismatch for {key!r}: {a!r} != {b!r}"

    _assert_definitions_equal(parsed.definitions, native.definitions)

    # Compare a shape-bearing event rather than only file-level metadata.
    parsed_rf = _first_rf(parsed)
    native_rf = _first_rf(native)

    np.testing.assert_allclose(
        np.asarray(parsed_rf.signal),
        np.asarray(native_rf.signal),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(parsed_rf.t),
        np.asarray(native_rf.t),
        rtol=0.0,
        atol=0.0,
    )

    assert parsed_rf.freq_offset == native_rf.freq_offset
    assert parsed_rf.phase_offset == native_rf.phase_offset


def test_pulseq_reader_returns_timing_checkable_sequence() -> None:
    """A sequence returned by PulseqReader must remain usable by PyPulseq."""

    parsed = PulseqReader.read(REFERENCE_SEQ)

    ok, errors = parsed.check_timing()

    assert ok, "\n".join(str(error) for error in errors)

    # This also protects against a pathological parse that loses RF shape data.
    rf = _first_rf(parsed)
    magnitude = np.abs(np.asarray(rf.signal))

    assert magnitude.size > 2
    assert np.ptp(magnitude) > 0
