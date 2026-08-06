# Roadmap

## v0.2.0: publication-oriented alpha

Primary objective: establish the protocol/expression contract and publish a bounded, reproducible software release around FID, GRE, EPI, and TSE.

Release themes:

- protocol-centered sequence construction;
- compact hierarchy and loops;
- immutable numeric realization;
- Pulseq and gammaSTAR export;
- logical orientation mapping;
- ADC trains;
- sequence-specific acceptance tests; and
- SoftwareX publication package.

The exact release gate is defined in [`release_scope_v0.2.0.md`](release_scope_v0.2.0.md).

## v0.3.0: contract completion and hardening

Planned priorities:

### Deferred v0.2.0 tests

- **EPI-06:** robust interactive echo-spacing/readout-duration retiming for arbitrary gradient/ADC trains across local, Pulseq, and gammaSTAR representations.
- **TSE-02:** strict cross-backend ETL mutation, including refocusing/readout counts and shot recalculation.

### TSE contract expansion

- release-gated effective-TE behavior;
- formal linear, centric, and reverse phase-order definitions;
- refocusing-flip trains and variable-flip schedules;
- crusher/spoiler moment checks;
- broader orientation tests; and
- sequence-specific k-space coverage summaries.

### Writer and expression hardening

- additional generic loop-indexed expression bindings;
- clearer writer capability negotiation;
- validation of unsupported live-edit dependencies;
- improved cross-representation summaries; and
- removal of temporary metadata contracts where a stable public API is appropriate.

### Testing and documentation

- higher branch coverage in resolver and writers;
- deterministic semantic reference fixtures;
- generated API documentation;
- expanded tutorials; and
- compatibility matrix across supported PyPulseq and gammaSTAR versions.

## Post-v0.3.0

- multi-kernel and preparation/calibration branches;
- stronger feasibility and minimum TE/TR tools;
- PNS, gradient duty-cycle, and RF-related analysis integrations;
- external sequence library and templates;
- multi-site and multi-vendor phantom validation; and
- a follow-up MR methods paper centered on experimental validation rather than software architecture.
