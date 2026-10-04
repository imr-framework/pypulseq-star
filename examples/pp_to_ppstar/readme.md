# Migrating PyPulseq Sequences to PyPulseq-Star

This directory provides worked examples showing how an existing PyPulseq
sequence can be migrated to PyPulseq-Star while preserving the original
sequence logic, making protocol relationships explicit, and enabling export
to multiple backends, including gammaSTAR.

The first example starts directly from the canonical PyPulseq GRE sequence,
`write_gre.py`, and incrementally adapts it into
`demo_GRE_pp_to_ppstar.py`.

The goal is not to rewrite the sequence from scratch or hide sequence-specific
logic behind convenience functions. Instead, the migration shows the changes
required to move from a conventional PyPulseq script to a relationship-aware
PyPulseq-Star implementation while keeping the resulting code compact and
recognizable to existing PyPulseq users.

This example is organized as six migration steps. Migration validation can be
invoked optionally from the demo, while deterministic registration remains
implemented and enforced separately by the migration test suite.

## 1. Replace the PyPulseq import

The migration begins with the package import. The original PyPulseq GRE uses:

```python
import numpy as np
import pypulseq as pp
```

The corresponding PyPulseq-Star imports are:

```python
from pathlib import Path

import numpy as np

import pypulseq_star as ppstar
from pypulseq_star.sequence import vary
from pypulseq_star.writers import GammaStarWriter, PulseqWriter
```

The main package namespace changes from `pp` to `ppstar`. The `vary` helper is
used for retained repetition-dependent behavior, while the backend writers
lower the same sequence definition to Pulseq `.seq` and gammaSTAR `.seq.json`
representations.

`Path` is used only for the example's output paths so generated files can be
kept in a dedicated directory rather than written into the source directory.

## 2. Add a protocol alongside the existing system definition

The scanner hardware limits do not need to be reorganized for the migration.
The original PyPulseq `Opts` block remains recognizable, with the package
namespace changed from `pp` to `ppstar`:

```python
system = ppstar.Opts(
    max_grad=28,
    grad_unit="mT/m",
    max_slew=150,
    slew_unit="T/m/s",
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
)
```

The original handling of scalar or two-dimensional FOV is retained:

```python
fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov
```

The main structural addition is a `Protocol` object containing the
user-adjustable acquisition parameters already exposed by the original
PyPulseq function:

```python
protocol = ppstar.Protocol(
    name="gre",
    parameters={
        "fov_x": fov_x,
        "fov_y": fov_y,
        "n_x": n_x,
        "n_y": n_y,
        "flip_angle_deg": flip_angle_deg,
        "slice_thickness": slice_thickness,
        "tr": tr,
        "te": te,
    },
)

p = protocol.symbols
```

The sequence is then created using both the existing system limits and the
protocol:

```python
seq = ppstar.Sequence(system=system, protocol=protocol, name="gre")
```

This is the direct PyPulseq-Star counterpart to:

```python
seq = pp.Sequence(system)
```

Associating the protocol with the sequence allows subsequent event properties
and timing expressions to retain their dependencies on acquisition parameters
rather than consuming only their initial numerical values.

## 3. Migrate event construction

Most PyPulseq event constructors have direct PyPulseq-Star counterparts. The
GRE RF pulse, gradients, ADC, prephasers, and spoilers therefore require two
principal changes:

1. replace the `pp` namespace with `ppstar`; and
2. use retained protocol symbols such as `p.n_x`, `p.fov_x`, and
   `p.slice_thickness` where the corresponding values were already exposed as
   acquisition parameters.

For example:

```python
gx = pp.make_trapezoid(
    channel="x",
    flat_area=n_x * delta_kx,
    flat_time=3.2e-3,
    system=system,
)
```

becomes:

```python
gx = ppstar.make_trapezoid(
    channel="x",
    flat_area=p.n_x * delta_kx,
    flat_time=3.2e-3,
    system=system,
)
```

The sequence-specific constants and event structure are otherwise retained.
Additional metadata is intentionally avoided at this stage so the migrated
script remains close to the original PyPulseq implementation.

Loop-dependent phase encoding and RF/ADC phase spoiling are handled separately
in Step 5 because PyPulseq-Star retains those changes as explicit relationships
rather than materializing them through a Python loop during sequence
construction.

## 4. Retain TE and TR as timing relationships

Timing is the first part of the migration where PyPulseq-Star differs
substantially from the numerical PyPulseq workflow.

The original PyPulseq GRE computes TE and TR delays numerically and immediately
rounds them to the scanner raster. In PyPulseq-Star, the corresponding timing
quantities can remain explicit relationships to protocol parameters and
sequence events:

```python
excitation_tail = seq.duration([rf, gz]) - rf.anchor("center")
prephase_duration = seq.duration([gx_pre])
readout_to_echo = adc.anchor("center")

te_delay = p.te - excitation_tail - prephase_duration - readout_to_echo
te_fill = ppstar.make_delay(te_delay, system=system)
```

TR is represented in the same way:

```python
tr_delay = (
    p.tr
    - seq.duration([rf, gz])
    - prephase_duration
    - seq.duration([gx])
    - te_delay
)
tr_fill = ppstar.make_delay(tr_delay, system=system)
```

The important difference is that `te_delay` and `tr_delay` are not immediately
reduced to fixed numerical values. Their relationships to TE, TR, and the
dependent sequence events are retained until the sequence is resolved.

Consequently, numerical assertions and raster-dependent checks from the
original PyPulseq script are deferred to the PyPulseq-Star
resolution/validation stage rather than reproduced as construction-time Python
assertions.

## 5. Replace the explicit phase-encoding loop with retained variations

The largest structural change in the GRE migration is the phase-encoding loop.

In the original PyPulseq implementation, every phase-encoding repetition is
constructed explicitly in Python. PyPulseq-Star instead keeps one
representative GRE kernel and records how selected event properties vary across
the logical loop.

The repeated kernel is declared as:

```python
kernel = seq.set_node(
    "kernel",
    factor=p.n_y,
    repeat_every=p.tr,
    counter="ky_index",
    repeat_mode="loop",
)
```

Here, `factor=p.n_y` retains the number of phase-encoding repetitions,
`repeat_every=p.tr` retains their temporal spacing, and `ky_index` identifies
the loop counter used when repetition-dependent values are realized.

The phase-encoding trajectory is represented by its starting value and step:

```python
phase_encode_start = -p.n_y / (2 * p.fov_y)
phase_encode_step = 1 / p.fov_y
```

Representative phase-encoding and rewinding gradients are created once:

```python
gy_pre = ppstar.make_trapezoid(
    channel="y", area=1.0, duration=seq.duration([gx_pre]), system=system
)
gy_reph = ppstar.make_trapezoid(
    channel="y", area=-1.0, duration=seq.duration([gx_pre]), system=system
)
```

The repetition-dependent changes are then attached to the logical kernel:

```python
kernel.vary(
    vary(gy_pre, "area", strength=phase_encode_start, step=phase_encode_step),
    vary(gy_reph, "area", strength=-phase_encode_start, step=-phase_encode_step),
    vary(
        [rf, adc],
        "phase_offset",
        strength=0.0,
        step=rf_spoiling_inc * np.pi / 180,
        mode="accumulated",
        wrap=2 * np.pi,
    ),
)
```

The executable blocks are added only once, but each is explicitly associated
with the logical `kernel` node:

```python
seq.add_block(rf, gz, node="kernel.excitation")
seq.add_block(gx_pre, gy_pre, gz_reph, node="kernel.prephase")
seq.add_block(te_fill, node="kernel.echo_delay")
seq.add_block(gx, adc, node="kernel.readout")
seq.add_block(
    tr_fill,
    gx_spoil,
    gy_reph,
    gz_spoil,
    node="kernel.spoiling",
)
```

This association makes the complete GRE motif the repeated unit. The backend
writer can then materialize each kernel repetition while applying the
appropriate `ky_index`-dependent phase encoding and RF/ADC phase variation.

The migrated source therefore no longer needs the original precomputed
`phase_areas`, `rf_phase`, or `rf_inc` variables.

## 6. Resolve, validate, plot, and export

PyPulseq builds a concrete numerical sequence directly. In PyPulseq-Star,
protocol parameters and their dependent relationships remain symbolic during
sequence construction until a numerical realization is required.

The sequence is first resolved:

```python
resolved = seq.resolve()
```

Timing validation is then performed on the resolved sequence:

```python
ok, error_report = resolved.check_timing()
if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed. Error listing follows:")
    for error in error_report:
        print(error)

if test_report:
    print(resolved.test_report())
```

A short portion of the resolved realization can be plotted for visual
inspection:

```python
if plot:
    seq.plot(
        realization=resolved,
        time_range=(0, 4 * resolved.protocol.repetition_time),
    )
```

The same source sequence is then lowered to the two supported backend
representations:

```python
if write_seq:
    PulseqWriter(seq).write(seq_filename, realization=resolved)

if write_json:
    GammaStarWriter(seq).write(json_filename, defaults=resolved)
```

The example keeps generated artifacts under a dedicated output directory:

```python
output_dir = Path("out/gre_pp_to_ppstar")
seq_filename = output_dir / "gre_pp_to_ppstar.seq"
json_filename = output_dir / "gre_pp_to_ppstar.seq.json"

main(
    plot=True,
    write_seq=True,
    write_json=True,
    seq_filename=seq_filename,
    json_filename=json_filename,
)
```

This keeps generated `.seq` and `.seq.json` files separate from the migration
source while preserving explicit backend-specific filenames.

Migration validation can also be invoked optionally from the same demo:

```python
main(
    plot=True,
    write_seq=True,
    write_json=True,
    validate_migration=True,
    seq_filename=seq_filename,
    json_filename=json_filename,
)
```

When enabled, the demo uses the same `MigrationValidator` interface exercised
by the migration test suite:

```python
source_path = Path(__file__).resolve().parents[2] / "src/pypulseq_star/migration/reference_scripts/write_gre.py"
report = MigrationValidator(
    source_path=source_path,
    migrated_sequence=seq,
    migrated_realization=resolved,
).validate()
print(report.to_text())
```

The validator is optional so sequence construction and export remain the
primary focus of the example.

The FOV and sequence-name definitions from the PyPulseq reference are also
retained before resolution and export:

```python
seq.set_definition("FOV", [p.fov_x, p.fov_y, p.slice_thickness])
seq.set_definition("Name", "gre")
```

## What the migration adds

The six-step migration preserves the recognizable PyPulseq sequence structure
while adding capabilities that are explicit in PyPulseq-Star:

- **Retained protocol relationships.** FOV, matrix size, flip angle, TE, and TR
  remain connected to dependent sequence quantities until realization.
- **Explicit repetition semantics.** Phase encoding and RF/ADC phase variation
  are described once on a logical GRE kernel rather than materialized by a
  source-level Python loop.
- **Relationship-aware timing.** TE and TR remain expressed through dependent
  event timing until the sequence is numerically resolved.
- **Deferred realization and validation.** The symbolic protocol is converted
  to a concrete realization before timing and hardware consistency are checked.
- **Multi-backend export.** The same source definition can be lowered to both
  conventional Pulseq `.seq` and gammaSTAR `.seq.json`.
- **Incremental migration.** RF, gradient, ADC, system-limit, and overall
  sequence construction remain close to the corresponding PyPulseq code; the
  larger changes are concentrated where PyPulseq-Star adds retained
  relationships, variation, and backend-independent realization.


## Registration and validation

Migration validation is kept separate from sequence construction but can be
invoked from the demo with `validate_migration=True`. The same
`MigrationValidator` API is intended for unit tests and future automated or
LLM-assisted migration workflows.

The validator is organized around acquisition parameters, sequence
structure/events, and timing. Its purpose is not only to compare concrete
Pulseq realizations, but also to determine whether relevant source constructs
are accounted for in the PyPulseq-Star representation.

Deterministic registration criteria will be expanded through the migration test
suite as the validator matures, including realization-level checks such as
sequence timing, phase-encoding trajectory, RF/ADC phase progression,
repetition count, duration, and backend readability.
