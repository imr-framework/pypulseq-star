# Testing and Acceptance Strategy

PyPulseq-Star uses layered tests because file existence alone does not demonstrate that a relationship-aware sequence is correct.

## Test layers

### 1. Unit tests

Unit tests cover:

- protocol references and expressions;
- event constructors and anchors;
- hierarchy and node registration;
- relationship resolution;
- orientation mapping;
- ADC trains;
- plotting metadata; and
- writer helpers.

### 2. Integration tests

Integration tests cover:

- clean installation and import;
- execution of the four beginner demos;
- generation of `.seq` and `.seq.json` outputs;
- Pulseq timing checks;
- gammaSTAR hierarchy and dependency fields; and
- round-trip or summary-level consistency where applicable.

### 3. Sequence-specific contract tests

Sequence-specific tests assert the release behaviors defined in [`release_scope_v0.2.0.md`](release_scope_v0.2.0.md).

Recommended modules:

```text
tests/integration/test_demo_exports.py
tests/integration/test_cross_representation.py
tests/contracts/test_fid_contract.py
tests/contracts/test_gre_contract.py
tests/contracts/test_epi_contract.py
tests/contracts/test_tse_contract.py
tests/contracts/test_gammastar_dependencies.py
tests/contracts/test_loop_contracts.py
tests/contracts/test_orientation_contract.py
```

## Strict, flexible, and deferred assertions

### Strict assertions

A strict test should fail the release when:

- a required event family disappears;
- loop length no longer depends on the intended protocol value;
- a timing fill becomes negative;
- timing differs beyond raster tolerance;
- ADC windows merge into an unintended continuous gate;
- gradient polarity or orientation is incorrect;
- a live gammaSTAR control becomes a stale literal; or
- Pulseq and gammaSTAR no longer represent the same logical sequence.

### Flexible assertions

A flexible test should assert invariants rather than exact serialized details. Prefer:

- expected event family and ordering rather than exact block IDs;
- moment, sign, and dependency checks rather than pixel-identical plots;
- tolerance-based timing rather than exact floating-point strings;
- logical loop count rather than backend-specific expanded block count; and
- presence of direct protocol dependencies rather than exact generated variable names.

GRE-01 through GRE-08 use this flexible style for v0.2.0.

### Deferred assertions

Deferred tests should be marked clearly and excluded from the v0.2.0 release gate. They may be retained as `xfail`, skipped tests with an issue link, or roadmap specifications.

For v0.2.0:

- EPI-06 is deferred.
- TSE-02 is deferred.

Do not silently weaken a deferred test into a passing shallow assertion. Preserve the intended future contract in the test name, docstring, and linked issue.

## Coverage policy

Coverage is a diagnostic and release-quality signal, not a substitute for sequence correctness.

### v0.2.0 priorities

Prioritize coverage for:

1. protocol and expression evaluation;
2. relationship resolution and validation;
3. hierarchy and loop handling;
4. Pulseq writer lowering;
5. gammaSTAR dependency export;
6. orientation mapping;
7. ADC-train handling; and
8. the four demo execution paths.

### Recommended gates

- All release-critical changed lines: at least 90% line coverage.
- Core protocol/resolution/writer modules: at least 75% meaningful branch coverage.
- Repository total: publish a truthful baseline and prevent regression.
- Four demos: 100% smoke execution in CI.

## Reference outputs

Keep reference fixtures small and semantic.

Good fixtures include:

- compact JSON fragments containing dependency paths;
- event-count summaries;
- timing summaries;
- gradient moment/polarity arrays; and
- checksums for deliberately frozen small outputs.

Avoid committing large generated JSON files, plots, DICOM data, or scanner outputs to the main repository.

## Manual release evidence

For v0.2.0, retain a release record containing:

- one default local plot for each sequence;
- one gammaSTAR plot for each sequence;
- representative edited-protocol screenshots for committed live controls;
- timing-check output;
- CI and coverage reports; and
- a list of known relaxed/deferred tests.
