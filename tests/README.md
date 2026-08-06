# PyPulseq-Star test suite

The suite is organized by responsibility:

- `tests/tests/unit/`: focused library/API tests.
- `tests/tests/integration/`: installation, writer, and end-to-end demo tests.
- `tests/sequences/`: FID, GRE, EPI, and TSE semantic acceptance tests.
- `tests/sequences/test_no_sequence_intrusions.py`: protects generic library layers from hidden reference-sequence branches.

## Recommended commands

```bash
# Complete suite
python tests/run_tests.py

# Sequence-specific acceptance tests
python tests/run_tests.py --sequences

# Generic semantic-intrusion guards
python tests/run_tests.py --semantic

# Coverage with the v0.2.0 gate
python tests/run_tests.py --coverage --fail-under 75

# Forward additional pytest options
python tests/run_tests.py --coverage -- -x -vv
```

The default coverage target is 75% with branch coverage enabled. This is a
release gate, not a reason to write shallow tests. New tests should protect a
public contract, a scientific/sequence invariant, or a previously observed
regression.

## Sequence-test intent

### FID

Aliases, TE/TR fills, averages loop ownership, and timing realization.

### GRE

Protocol-driven ky coverage, FOV scaling, loop length, orientation, and timing.

### EPI

Segment/ETL relationships, phase-blip scaling, orientation, and timing.
Interactive echo-spacing mutation remains deferred to v0.3.0 when so labeled.

### TSE

Complete ky coverage, strict centric first-echo behavior, linear/reverse
ordering, equal-and-opposite phase rewind, direct live gammaSTAR dependencies,
and timing.

## CI

`.github/workflows/ci.yml` runs:

1. Ruff and mypy checks.
2. Full tests on Python 3.10-3.13.
3. A dedicated 75% branch-coverage gate with XML and HTML artifacts.
4. Wheel build, clean installation, and smoke tests on Python 3.10 and 3.13.

The former duplicate `tests.yml` and `lint.yml` workflows are retained with a
`.disabled` suffix so CI has one authoritative workflow.
