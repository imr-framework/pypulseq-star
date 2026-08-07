# PyPulseq-Star v0.2.0 Release Scope

This document defines what v0.2.0 promises, what is accepted with relaxed constraints, and what is explicitly deferred to v0.3.0.

## Scope legend

- **Committed:** strict v0.2.0 release gate. A regression blocks the release.
- **Flexible:** included in v0.2.0, but exact numerical fixtures, plot appearance, block counts, or raster-level details may vary when structural and timing invariants remain correct.
- **Deferred:** excluded from the v0.2.0 release gate and planned for v0.3.0.

## FID: committed v0.2.0 contract

| ID | Test | v0.2.0 expectation |
|---|---|---|
| FID-01 | Smoke export | Default demo runs, timing passes, and `.seq` plus `.seq.json` are generated with discrete RF and ADC events. |
| FID-02 | Averages loop | Changing averages changes the logical repetition count without double counting; repetition duration remains TR. |
| FID-03 | TE timing | ADC placement changes with TE and agrees with the requested TE within raster tolerance. |
| FID-04 | TR timing | Repetition spacing changes with TR; the TR fill remains nonnegative and events do not overlap. |
| FID-05 | ADC protocol | `num_samples` and dwell update ADC metadata and active gate duration. |
| FID-06 | Cross representation | SeqStar, Pulseq, and gammaSTAR agree on RF/ADC event families, counts, and timing. |

## GRE: flexible v0.2.0 contract

GRE is a supported v0.2.0 reference sequence. The tests below are required structurally, but exact plot appearance, block counts, raster rounding, and backend-specific lowering may differ when documented.

| ID | Test | Flexible v0.2.0 expectation |
|---|---|---|
| GRE-01 | Smoke export | RF, prephase, readout, spoiling, `.seq`, and `.seq.json` are present and timing passes. |
| GRE-02 | ky loop length | The phase-encoding loop follows `n_y`; no stale default count remains. |
| GRE-03 | Phase-encode start | The Gy sweep responds to `n_y` and FOV and covers the intended Cartesian range. |
| GRE-04 | FOV scaling | Phase-encode step scales as approximately `1/fov`. |
| GRE-05 | RF/ADC spoiling | RF and ADC phase progressions respond to the spoiling increment and remain aligned. |
| GRE-06 | TE/TR timing | Readout placement and repetition spacing follow TE/TR without negative fills. |
| GRE-07 | Orientation | Axial, coronal, and sagittal remap logical read/phase/slice axes consistently. |
| GRE-08 | Cross representation | SeqStar, Pulseq, and gammaSTAR preserve the same structural order and core loop behavior. |

### Relaxed GRE details

The following differences do not block v0.2.0 when the core invariant is satisfied and the behavior is documented:

- one-raster differences in solved delays;
- backend-specific block splitting or padding;
- different but equivalent graphical interpolation at gradient corners;
- minor differences in summary block counts after loop lowering; and
- gammaSTAR workplace display differences that do not change the underlying exported dependency.

## EPI: committed and deferred contracts

| ID | Status | Test | v0.2.0 expectation |
|---|---|---|---|
| EPI-01 | Committed | Smoke export | RF, prephaser, alternating readout train, blips, and discrete ADC windows are present. |
| EPI-02 | Committed | Readout polarity | Successive readout lobes alternate polarity correctly. |
| EPI-03 | Committed | ADC windowing | One acquisition window is associated with each readout lobe; no giant continuous ADC gate is produced. |
| EPI-04 | Committed | `n_y` train length | Changing `n_y` updates readout-window/blip count without expanded-timeline double counting. |
| EPI-05 | Committed | FOV/blip scaling | Phase-blip area responds to phase FOV with `Delta ky = 1/fov` within tolerance. |
| EPI-06 | **Deferred to v0.3.0** | Echo-spacing/readout retiming | Fully interactive retiming of arbitrary readout trains across local, Pulseq, and live gammaSTAR backends is not a v0.2.0 guarantee. |
| EPI-07 | Committed | Segmentation | Outer segment/shot and inner readout loop counts remain consistent and cover the requested phase lines. |
| EPI-08 | Committed | Orientation | Logical read/phase/slice axes remap without changing train logic. |

### v0.2.0 EPI timing statement

The default EPI demo and Python-regenerated parameter combinations must pass timing checks. Website-side arbitrary waveform resizing after export is explicitly outside the v0.2.0 contract.

## TSE: narrow v0.2.0 contract

| ID | Status | Test | v0.2.0 expectation |
|---|---|---|---|
| TSE-01 | Committed | Smoke export | The default demo contains excitation, a refocusing train, crushers, readouts, spoilers, and valid `.seq`/`.seq.json` outputs. |
| TSE-02 | **Deferred to v0.3.0** | Strict ETL mutation | A single release-gated test requiring local plots, Pulseq lowering, and live gammaSTAR refocusing/readout counts to update for arbitrary ETL edits is deferred. |

### Demonstrated but not release-gated TSE features

The v0.2.0 demo may expose and illustrate:

- effective-TE metadata;
- linear, centric, and reverse phase ordering;
- refocusing-flip edits;
- crusher and spoiler controls;
- orientation mapping; and
- compact echo-train hierarchy.

These are valuable demonstrations and should be tested opportunistically, but only TSE-01 blocks the v0.2.0 release. The strict cross-backend ETL contract and a broader sequence-specific TSE test matrix move to v0.3.0.

## Cross-cutting v0.2.0 gates

The release is blocked if any of the following fail:

1. Clean installation on a supported Python version.
2. Import of the public package API.
3. Default FID, GRE, EPI, and TSE demo execution.
4. Generation of both Pulseq and gammaSTAR outputs for all four demos.
5. Timing validation for default generated sequences.
6. No sequence-specific logic introduced into generic writers.
7. No credentials, proprietary scanner material, PHI, or private URLs in the release tree/history.
8. Documentation accurately distinguishes strict, flexible, and deferred behavior.

## Non-goals for v0.2.0

- Clinical validation or regulatory claims.
- Complete scanner/interpreter compatibility across vendors.
- Full symbolic feasibility solving or automatic minimum TE/TR optimization.
- Guaranteed live editing of every arbitrary waveform parameter in gammaSTAR.
- Full sequence-family coverage beyond the four reference demos.
- Stable v1.0 API compatibility.
