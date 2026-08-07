# PyPulseq-Star

PyPulseq-Star is an evolution of the PyPulseq-style sequence-development workflow for applications that require retained protocol intent, explicit relationships, and more than one output representation. It preserves familiar Python-first construction while enriching the sequence model with protocol parameters, symbolic dependencies, hierarchy, timing relationships, loops, orientation, validation, and backend-specific export semantics.

The project does not replace PyPulseq or redefine the Pulseq standard. Instead, it extends the authoring model around them: a PyPulseq-like script is used to construct a relationship-aware internal representation that can be resolved into a conventional Pulseq sequence or lowered into a dynamic gammaSTAR document. This separation supports consistent, extensible, and multi-backend workflows without requiring users to maintain independent implementations for each target.

A single sequence definition can produce:

- a concrete Pulseq `.seq` file with resolved numerical events and timing; and
- a gammaSTAR `.seq.json` document that preserves compact hierarchy, editable protocol dependencies, loops, and relationship-aware behavior where supported.

```text
PyPulseq-like Python script
        |
        v
Protocol + expressions + hierarchy + relationships
        |
        +--> immutable numeric realization
        |       +--> local plotting and validation
        |       +--> Pulseq writer --> concrete .seq
        |
        +--> gammaSTAR writer --> hierarchical, protocol-aware .seq.json
```

> **Release status:** v0.2.0a1 is the first publication-oriented alpha release. It establishes the protocol/expression contract, relationship-aware internal representation, four reference demonstrations, and Pulseq/gammaSTAR export paths intended for software evaluation. PyPulseq-Star remains research software, and generated sequences require independent review and validation before scanner use.

## What is in v0.2.0

The v0.2.0 scope is intentionally bounded. It prioritizes a dependable protocol-oriented workflow across four reference sequences rather than claiming complete sequence-family coverage.

### Release-gated capabilities

- PyPulseq-like RF, gradient, ADC, delay, block, and sequence construction.
- Protocol parameters represented as editable symbolic references.
- Explicit sequence hierarchy and compact loop-native motifs.
- Timing relationships, anchors, symbolic fills, immutable resolution, and timing validation.
- Pulseq `.seq` export from the resolved numeric realization.
- gammaSTAR `.seq.json` export with compact hierarchy and live protocol dependencies where supported.
- Logical read/phase/slice axes with axial, coronal, and sagittal orientation mapping.
- Single ADC events and multi-window ADC-train abstractions.
- Four beginner-facing demonstrations: FID, spoiled GRE, EPI, and TSE.
- Sequence-specific acceptance tests defined in [`docs/release_scope_v0.2.0.md`](docs/release_scope_v0.2.0.md).

### Flexible or relaxed v0.2.0 acceptance areas

The following behaviors must remain structurally correct, while small implementation-dependent differences in exact block counts, raster rounding, or graphical presentation are acceptable when documented:

- phase-encoding loop length;
- FOV-dependent phase encoding;
- RF/ADC spoiling progression;
- TE/TR placement;
- orientation mapping; and
- cross-representation agreement.

### Explicitly deferred to v0.3.0

The following tests are not v0.2.0 release gates:

- **EPI-06:** fully interactive echo-spacing/readout-duration retiming across all backends.
- **TSE-02:** strict ETL-change contract requiring all local, Pulseq, and live gammaSTAR event counts to update under one release-gated test.

The current demos may expose parts of these behaviors, but v0.2.0 does not guarantee them as stable public contracts. See [`docs/roadmap.md`](docs/roadmap.md).

## Why PyPulseq-Star?

PyPulseq made vendor-neutral MR sequence development substantially more accessible by providing a concise Python interface for constructing Pulseq-compatible events and sequences. PyPulseq-Star builds on that foundation for workflows in which a sequence must remain editable at the protocol level, preserve the reasoning that connects parameters to events, and target both concrete and dynamic representations.

The motivation is not to introduce a separate sequence-programming language. It is to evolve the familiar PyPulseq workflow by retaining information that is normally lost when a script is reduced immediately to numerical events. In PyPulseq-Star, protocol values, symbolic expressions, hierarchy, loop structure, event relationships, and timing intent remain part of the sequence model until an explicit resolution or export step.

This enables three central properties:

- **Consistent:** one source definition drives local visualization, validation, concrete Pulseq export, and relationship-aware gammaSTAR export.
- **Extensible:** the internal representation separates sequence semantics from backend-specific lowering, allowing new sequence families, relationships, validators, and writers to be added without rewriting the overall workflow.
- **Multi-backend:** the same sequence model can produce a resolved Pulseq `.seq` file and a compact gammaSTAR `.seq.json` document while preserving the capabilities appropriate to each target.

PyPulseq-Star connects these strengths through one development model:

1. sequence construction remains recognizably PyPulseq-like;
2. protocol-derived values retain their source and dependencies;
3. hierarchy and repetition are declared explicitly;
4. timing intent is represented through expressions and relationships;
5. resolution creates an immutable numerical realization for validation, plotting, and Pulseq export; and
6. gammaSTAR export preserves compact structure and editable dependencies where the target supports them.

The claim is therefore deliberately bounded: PyPulseq-Star is an evolution of the PyPulseq authoring workflow for relationship-aware, consistent, extensible, and multi-backend sequence development—not a replacement for PyPulseq, Pulseq, or scanner-specific safety validation.

## Core concepts

### Protocol-centered construction

```python
protocol = ppstar.Protocol(
    name="fid",
    parameters={
        "sequence_name": "fid",
        "flip_angle": 90.0,
        "echo_time": 20e-3,
        "repetition_time": 1.0,
        "num_averages": 4,
    },
)
p = protocol.symbols

seq = ppstar.Sequence(system=system, protocol=protocol, name=p.sequence_name)
```

Protocol references behave like scalars during sequence construction while preserving their canonical source for resolution and gammaSTAR export.

### Explicit hierarchy and loops

```python
seq.set_node(
    "kernel",
    role="kernel",
    repeat_count=p.num_averages,
    repeat_every=p.repetition_time,
    counter="average_index",
    repeat_mode="loop",
)
```

A compact source motif can therefore represent a larger execution hierarchy. The Pulseq writer lowers loops to concrete events; the gammaSTAR writer preserves loop structure where supported.

### Relationship-aware timing

```python
te_fixed = seq.duration(
    start=rf.anchor("center"),
    end=adc.anchor("start"),
)
te_fill = ppstar.make_delay(
    p.echo_time - te_fixed,
    system=system,
    name="te_fill",
)
```

The relationship and expression layers describe timing intent, while `seq.resolve()` evaluates the current protocol and creates an immutable numeric realization.

### Two export targets

```text
Pulseq .seq
    concrete numerical events
    resolved timing and repetitions
    interpreter-oriented output

gammaSTAR .seq.json
    compact hierarchy and loops
    protocol-facing controls
    live expressions and dependencies where supported
    workplace-oriented organization
```

## Included reference demos

| Example | Main idea | v0.2.0 release status |
|---|---|---|
| `demo_FID.py` | RF-to-ADC timing, TE/TR, samples, dwell, and repeated averages | Strict release-gated reference demo |
| `demo_GRE.py` | Phase encoding, RF/ADC spoiling, TE/TR, and orientation | Included with flexible acceptance constraints |
| `demo_EPI.py` | Alternating readouts, phase blips, segmentation, ADC windows, and orientation | Release-gated except dynamic echo-spacing/readout retiming |
| `demo_TSE.py` | Excitation, refocusing train, crushers, readouts, spoilers, and phase-order demonstrations | Smoke export release-gated; strict ETL mutation deferred |

The demos teach the public API and provide reproducible software examples. They are not clinically validated protocols.

## Installation

### Published alpha release

The publication-facing notebooks and examples for this release should install the exact PyPI artifact:

```bash
python -m pip install "pypulseq-star==0.2.0a1"
```

For notebooks, use the IPython `%pip` magic so installation occurs in the active kernel environment:

```python
%pip install "pypulseq-star==0.2.0a1"
```

Then verify the imported version:

```python
import pypulseq_star as ppstar

assert ppstar.__version__ == "0.2.0a1"
print(ppstar.__version__)
```

Pinning the alpha release makes the notebook reproducible and prevents it from silently changing when a later package version is published.

### Development installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Development tools:

```bash
python -m pip install -e ".[dev]"
```

Optional plotting and dashboard dependencies:

```bash
python -m pip install -e ".[plotting,dashboard]"
```

## Run the examples

```bash
python examples/demo_FID.py
python examples/demo_GRE.py
python examples/demo_EPI.py
python examples/demo_TSE.py
```

Outputs are written under `out/<sequence>/`. Use each demo's command-line options to disable plotting or adjust supported protocol controls.

## Testing

Run the complete test suite:

```bash
python -m pytest
```

Run with coverage:

```bash
python -m pytest --cov=pypulseq_star --cov-report=term-missing
```

The v0.2.0 acceptance scope, strict/flexible/deferred tests, and sequence-level pass criteria are documented in:

- [`docs/release_scope_v0.2.0.md`](docs/release_scope_v0.2.0.md)
- [`docs/testing_and_acceptance.md`](docs/testing_and_acceptance.md)
- [`docs/releases/v0.2.0.md`](docs/releases/v0.2.0.md)

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
  expressions/     protocol references and expression evaluation
  make/            PyPulseq-like constructors
  plotting/        sequence and relationship visualization
  relationships/   timing relationships and validation
  resources/       package data used by exporters/viewers
  sequence/        hierarchy, loops, and executable timeline
  writers/         Pulseq and gammaSTAR export
examples/           beginner-facing sequence scripts
tests/              unit, integration, and release-contract tests
docs/               architecture, release scope, roadmap, and release notes
```

## Compatibility and limitations

- Python 3.10-3.13.
- PyPulseq is required for Pulseq `.seq` export.
- gammaSTAR behavior depends on the target workplace/runtime and supported expression semantics.
- Public APIs may still change before v1.0, but v0.2.0a1 establishes the first publication-oriented protocol/expression contract.
- Multi-site and multi-vendor scanner validation is outside the v0.2.0 software-release scope and is planned as a separate experimental study.

## Contributing

Contributions from PyPulseq users, Pulseq developers, gammaSTAR users, MRI physicists, and research-software developers are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) and the active release scope before proposing changes.

Release-blocking changes must include a regression test. Generic writers must not contain sequence-family conditionals when metadata or a general contract can express the behavior.

## Safety and validation

MRI pulse sequences can create hardware, peripheral nerve stimulation, SAR, timing, and image-quality risks. PyPulseq-Star is research software. A generated file must not be executed on a scanner solely because software timing checks pass. Users are responsible for independent sequence review, interpreter/vendor validation, institutional approvals, and scanner safety procedures.

## License

PyPulseq-Star is distributed under the MIT License. See [`LICENSE`](LICENSE).

## Citation

Use [`CITATION.cff`](CITATION.cff) and cite both the exact package release and repository commit used. For the alpha notebooks, the intended software identifier is `pypulseq-star==0.2.0a1`; the preferred article citation will be added after publication of the SoftwareX paper.
