# Changelog

All notable changes to PyPulseq-Star are documented here.

The project is pre-1.0 research software. Public API changes may occur, but release notes and migration guidance should accompany intentional breaking changes.

## [0.2.0] - 2026-08-06

### Added

- Protocol-centered symbolic parameter references and derived expressions.
- Immutable numeric sequence realization through `Sequence.resolve()`.
- Compact hierarchy and loop-native sequence definitions.
- Logical read/phase/slice axes and encoding-frame orientation mapping.
- Generic loop variation and expression-binding support.
- ADC-train abstractions for multiple acquisition windows.
- Publication-oriented FID, GRE, EPI, and TSE reference demos.
- Sequence-specific release scope and acceptance documentation.
- Expanded Pulseq and gammaSTAR export behavior.

### Changed

- Updated the four demos to a common protocol/expression workflow.
- Clarified writer boundaries: Pulseq resolves and materializes; gammaSTAR preserves compact live dependencies where supported.
- Improved plotting of logical orientation and repeated motifs.
- Reframed project status from an unconstrained pre-alpha scaffold to a bounded v0.2.0 alpha release candidate.

### Deferred

- EPI-06: fully interactive echo-spacing/readout-duration retiming across backends.
- TSE-02: strict cross-backend ETL mutation contract.

### Known limitations

- Multi-vendor and multi-site scanner validation is not complete.
- gammaSTAR live editing depends on target-runtime expression support.
- Some backend-specific block splitting, raster rounding, and graphical interpolation differ while remaining logically equivalent.
- The core v0.2.0 workflow is supported and publication-ready. Selected advanced extension interfaces remain pre-1.0 and may evolve with documented migration guidance.

## [0.1.0-alpha] - 2026-07-14

- Initial public-facing development scaffold.
- PyPulseq-like constructors, hierarchy, relationships, basic loops, dual writers, plotting, and four early demos.
