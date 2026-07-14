# PyPulseq-Star

PyPulseq-Star is an experimental Python framework for MRI pulse-sequence development that combines a familiar PyPulseq-style scripting workflow with a richer representation of sequence hierarchy, protocol parameters, timing relationships, and loop structure.

The same sequence description can be used to produce:

- a concrete Pulseq `.seq` file with numerically resolved events and timing; and
- a gammaSTAR `.seq.json` document that preserves hierarchy, loop structure, protocol dependencies, and relationship-aware behavior where supported.

PyPulseq-Star is a bridge to leverage the complementary strengths of Pypulseq and GammaStar:

```text
PyPulseq-like Python script
        |
        v
SeqStar hierarchy + protocol + relationships
        |
        +--> Pulseq writer --> concrete .seq
        |
        +--> gammaSTAR writer --> hierarchical .seq.json
        |
        +--> plotting, timing checks, and relationship inspection
```

> **Project status:** pre-alpha and under active development. APIs may change during the protocol/relationship contract sprint. Generated sequences must be independently reviewed and validated before scanner use.

## Why PyPulseq-Star?

PyPulseq provides a concise and widely understood Python interface for constructing RF, gradient, ADC, delay, block, and sequence objects. Pulseq provides a portable concrete sequence representation.

gammaSTAR adds a hierarchical and dynamic view of sequence execution. Its representation can preserve loops, protocol-controlled values, dependencies, and scanner-side sequence organization.

PyPulseq-Star explores a developer workflow in which:

1. sequence construction remains close to PyPulseq;
2. semantic hierarchy and repetition are declared explicitly;
3. generic relationships describe why events occur when they do;
4. the Pulseq writer resolves the sequence to a concrete numerical timeline; and
5. the gammaSTAR writer preserves the compact hierarchy and dynamic dependencies when possible.

## Core concepts

### PyPulseq-like event construction

The public constructors follow familiar sequence-building patterns:

```python
rf = ppstar.make_block_pulse(...)
gz = ppstar.make_trapezoid(...)
adc = ppstar.make_adc(...)
seq.add_block(rf, gz, role="excitation", node="kernel.excitation")
```

### Explicit hierarchy and loops

Logical structure is declared independently of the materialized source timeline:

```python
seq.set_node(
    "shot.kernel.echo_train",
    role="echo_train",
    repeat_count="echo_train_length",
    repeat_every="echo_spacing",
    counter="echo_index",
    repeat_mode="loop",
)
```

This allows a compact source motif to represent a larger execution hierarchy. The Pulseq writer can lower loops into concrete events, while the gammaSTAR writer can preserve the loop structure.

### Relationship-aware timing

Relationships express timing intent rather than requiring developers to manually calculate every delay:

```python
ppstar.relationships.set_readout_after(
    seq=seq,
    adc=adc_occurrence,
    readout_gradient=gx_occurrence,
    reference=rf_occurrence,
    offset="echo_time",
    solve_event=te_fill_occurrence,
    solve_property="duration",
)
```

The relationship layer resolves the requested timing for concrete export and retains dependency information for inspection and richer export paths.

### Two export targets

The writers have different responsibilities:

```text
Pulseq .seq
    concrete numerical events
    resolved timing
    scanner/interpreter-oriented output

gammaSTAR .seq.json
    hierarchy and loops
    protocol-facing values
    relationship-aware expressions where supported
    workplace-oriented sequence organization
```

## Included beginner examples

The four primary examples use a common structure:

```text
1. define system limits and protocol parameters
2. create the sequence and declare logical nodes
3. create events
4. construct the executable timeline
5. define relationships
6. resolve and check timing
7. plot
8. export
```

| Example | Main idea | Representation |
|---|---|---|
| `demo_FID.py` | RF-to-ADC timing with TE and repeated averages | One loop-native kernel |
| `demo_GRE.py` | Phase encoding, RF spoiling, TE placement, and TR filling | Explicitly expanded phase-encode lines |
| `demo_EPI.py` | Alternating readout train, phase blips, and multiple ADC windows | Compact synchronized Gx/Gy/ADC train |
| `demo_TSE.py` | Nested shots and echo trains with loop-indexed phase encoding | Compact nested `shot × echo_train` hierarchy |

The examples are intended to teach the public workflow, not to serve as clinically validated protocols.

## Installation

Create a clean environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install development tools:

```bash
python -m pip install -e ".[dev]"
```

Install dashboard dependencies when working on the relationship viewer:

```bash
python -m pip install -e ".[dashboard]"
```

## Run the examples

From the repository root:

```bash
python examples/demo_FID.py
python examples/demo_GRE.py
python examples/demo_EPI.py
python examples/demo_TSE.py
```

By default, each example may plot and write outputs under its `out/<sequence>/` directory. The `main()` arguments can disable plotting or either writer during development:

```python
main(plot=False, write_seq=True, write_json=True)
```

Typical outputs are:

```text
out/fid/fid.seq
out/fid/fid.seq.json
out/gre/gre.seq
out/gre/gre.seq.json
out/epi/epi.seq
out/epi/epi.seq.json
out/tse/tse.seq
out/tse/tse.seq.json
```

## Minimal example

```python
import math
import pypulseq_star as ppstar
from pypulseq_star.writers import GammaStarWriter, PulseqWriter

system = ppstar.Opts(max_grad=28, grad_unit="mT/m", max_slew=100, slew_unit="T/m/s")
protocol = ppstar.Protocol(
    name="fid",
    parameters={
        "Name": "fid",
        "flip_angle": 90.0,
        "rf_duration": 300e-6,
        "TE": 20e-3,
        "TR": 1.0,
        "averages": 1,
        "num_samples": 2048,
        "dwell": 20e-6,
    },
)

seq = ppstar.Sequence(system=system, name="fid", parameters=protocol.parameters)
rf = ppstar.make_block_pulse(
    flip_angle=math.radians(90.0),
    duration=300e-6,
    system=system,
    name="rf_excitation",
)
adc = ppstar.make_adc(num_samples=2048, dwell=20e-6, system=system, name="fid_adc")

seq.add_block(rf, role="excitation", node="kernel.excitation")
seq.add_block(adc, role="readout", node="kernel.readout")

PulseqWriter(seq).write("fid.seq")
GammaStarWriter(seq).write("fid.seq.json")
```

The full FID demo shows the preferred relationship-aware form.

## Development checks

```bash
python -m pytest
ruff check .
mypy src/pypulseq_star
python -m build
python -m twine check dist/*
```

## Repository organization

```text
src/pypulseq_star/
  blocks/          block containers
  core/            shared objects and parameter support
  events/          RF, gradient, ADC, and delay events
  make/            PyPulseq-like constructors
  plotting/        sequence and relationship visualization
  relationships/   timing relationships and validation
  resources/       package data used by exporters/viewers
  sequence/        sequence and timeline containers
  writers/         Pulseq and gammaSTAR export
examples/           beginner-facing sequence scripts
tests/              automated tests
```

## Relationship-contract work

The next architecture sprint will formalize the contract among:

```text
Protocol parameters
        -> relationships and derived values
        -> compact resolved timeline
        -> plotter
        -> Pulseq and gammaSTAR writers
```

This work will clarify parameter naming, loop-counter bindings, timing-fill ownership, dynamic versus resolved values, and consistency across plotting and export.

## Contributing

Contributions from PyPulseq users, Pulseq developers, gammaSTAR users, MRI physicists, and research-software developers are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

Useful early contributions include:

- testing installation on additional operating systems and Python versions;
- comparing generated `.seq` files with canonical PyPulseq examples;
- testing gammaSTAR JSON import and workplace behavior;
- adding focused timing and relationship tests;
- reporting scanner/interpreter compatibility results; and
- improving beginner documentation.

## Safety and validation

MRI pulse sequences can cause hardware, peripheral nerve stimulation, SAR, timing, and image-quality risks. PyPulseq-Star is research software. A generated file should not be executed on a scanner solely because software timing checks pass. Users are responsible for independent sequence review, vendor/interpreter validation, institutional approvals, and scanner safety procedures.

## License

PyPulseq-Star is distributed under the license in [`LICENSE`](LICENSE).

## Citation

A formal software citation will be added with the first archival release and software-paper submission. Until then, cite the repository release and commit used for your work.
