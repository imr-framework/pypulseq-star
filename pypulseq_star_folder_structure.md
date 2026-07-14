# PyPulseq-Star Repository Structure and Architecture

## Purpose

`pypulseq-star` should be a structured sequence-design library that preserves PyPulseq ergonomics while enriching sequence objects with hierarchy, timing, relationships, metadata, validation, plotting, and template-based JSON writing.

The package should remain independent from GammaSTAR blueprint files. The only external GammaSTAR artifact assumed by this design is a **GammaSTAR template JSON** used as a writer target. The source of truth is the internal `pypulseq_star` object graph.

In one sentence:

> PyPulseq-Star is a unified, typed, PyPulseq-compatible object model for MR pulse sequences that can export both Pulseq `.seq` files and meaningful template-populated JSON.

---

## Critique of the Previous Draft

The previous structure was directionally strong, but it risked building several “half bridges” — folders and abstractions that sound useful but are not yet tied to a concrete first milestone.

### Issues to fix

1. **Too many early folders**
   - `blocks/rf_block.py`, `blocks/readout_block.py`, `blocks/excitation_block.py`, and `blocks/composite_block.py` are useful later, but premature for the first commit.
   - These may become thin wrappers around `SeqStarBlock` unless there is a clear reason for subclassing.

2. **`io/` can be confusing**
   - `io` is a standard-library module name. `pypulseq_star.io` is technically legal, but `writers/` and `readers/` are more explicit and avoid ambiguity.

3. **`utils/` is a catch-all risk**
   - A broad `utils/` folder often becomes a dumping ground. Keep it small, or prefer purpose-specific modules such as `serialization.py`, `naming.py`, and `units.py` near the code that uses them.

4. **`adapters/` vs `writers/` boundary needs to be sharper**
   - Adapters should convert internal objects into external intermediate forms.
   - Writers should write files.
   - Validators should check correctness.
   - Keeping these roles distinct will prevent circular dependencies.

5. **Template support should not become a hidden GammaSTAR package**
   - The package should not have `gammastar/blueprints/` or hard-coded external blueprint dependencies.
   - Template population should be handled by `templates/`, `adapters/template.py`, and `writers/template_json_writer.py`.

6. **No file should exist without a testable responsibility**
   - Every folder in the MVP should support the first deliverable: build FID → export `.seq` → export JSON from template → validate → plot.

---

## Design Principles

1. **Use PyPulseq names where users interact with the library**
   - `make_block_pulse`, `make_adc`, `make_trapezoid`, `Sequence`, `add_block`.

2. **Use SeqStar names where the object model becomes richer**
   - `SeqStarNode`, `SeqStarEvent`, `SeqStarShape`, `SeqStarBlock`, `SeqStarSequence`, `SeqStarRelationship`.

3. **Keep the internal model unified**
   - Avoid separate conceptual worlds for PyPulseq and GammaSTAR.
   - The unified object graph should support both output targets.

4. **Separate construction, adaptation, validation, and writing**
   - `make/`: user-facing constructors.
   - `adapters/`: conversion to external representations.
   - `writers/`: file output.
   - `validation/`: correctness checks.

5. **Avoid premature subclassing**
   - Start with flexible base classes and factory functions.
   - Add specialized subclasses only when repeated code or validation rules justify them.

6. **Keep the package PEP-friendly**
   - Distribution name: `pypulseq-star`.
   - Import package name: `pypulseq_star`.
   - Modules and functions: `snake_case`.
   - Classes: `CapWords`, e.g., `SeqStarRFEvent`.
   - Constants: `UPPER_SNAKE_CASE`.
   - Use type hints from the start.

---

## Recommended Repository Structure

```text
pypulseq-star/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   └── feature_request.yml
│   ├── workflows/
│   │   ├── tests.yml
│   │   └── lint.yml
│   ├── pull_request_template.md
│   └── dependabot.yml
│
├── docs/
│   ├── architecture.md
│   ├── object_model.md
│   ├── sequence_hierarchy.md
│   ├── template_json_export.md
│   ├── pypulseq_compatibility.md
│   └── developer_guide.md
│
├── examples/
│   ├── fid.py
│   ├── fid_export_json.py
│   ├── gre_basic.py
│   └── inspect_sequence_tree.py
│
├── src/
│   └── pypulseq_star/
│       ├── __init__.py
│       ├── py.typed
│       ├── opts.py
│       ├── units.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── node.py
│       │   ├── parameter.py
│       │   ├── relationship.py
│       │   ├── timing.py
│       │   ├── metadata.py
│       │   └── ids.py
│       │
│       ├── shapes/
│       │   ├── __init__.py
│       │   ├── shape.py
│       │   ├── block.py
│       │   ├── arbitrary.py
│       │   ├── sinc.py
│       │   ├── gaussian.py
│       │   ├── trapezoid.py
│       │   └── spiral.py
│       │
│       ├── events/
│       │   ├── __init__.py
│       │   ├── event.py
│       │   ├── rf.py
│       │   ├── gradient.py
│       │   ├── adc.py
│       │   ├── delay.py
│       │   ├── trigger.py
│       │   └── label.py
│       │
│       ├── blocks/
│       │   ├── __init__.py
│       │   └── block.py
│       │
│       ├── sequence/
│       │   ├── __init__.py
│       │   ├── sequence.py
│       │   ├── kernel.py
│       │   ├── section.py
│       │   └── timeline.py
│       │
│       ├── make/
│       │   ├── __init__.py
│       │   ├── rf.py
│       │   ├── gradients.py
│       │   ├── adc.py
│       │   ├── delays.py
│       │   └── blocks.py
│       │
│       ├── adapters/
│       │   ├── __init__.py
│       │   ├── pypulseq.py
│       │   ├── simple_namespace.py
│       │   └── template_json.py
│       │
│       ├── templates/
│       │   ├── __init__.py
│       │   ├── reader.py
│       │   ├── mapper.py
│       │   └── resolver.py
│       │
│       ├── writers/
│       │   ├── __init__.py
│       │   ├── seq_writer.py
│       │   ├── object_json_writer.py
│       │   ├── template_json_writer.py
│       │   └── file_writer.py
│       │
│       ├── validation/
│       │   ├── __init__.py
│       │   ├── validator.py
│       │   ├── timing_validator.py
│       │   ├── relationship_validator.py
│       │   ├── hardware_validator.py
│       │   ├── template_validator.py
│       │   └── pypulseq_validator.py
│       │
│       ├── plotting/
│       │   ├── __init__.py
│       │   ├── plotter.py
│       │   ├── waveform_plotter.py
│       │   ├── timeline_plotter.py
│       │   └── tree_plotter.py
│       │
│       └── serialization/
│           ├── __init__.py
│           ├── to_dict.py
│           ├── from_dict.py
│           └── json.py
│
├── tests/
│   ├── conftest.py
│   ├── test_imports.py
│   ├── test_core_node.py
│   ├── test_relationships.py
│   ├── test_shapes.py
│   ├── test_rf.py
│   ├── test_gradients.py
│   ├── test_adc.py
│   ├── test_blocks.py
│   ├── test_sequence.py
│   ├── test_pypulseq_export.py
│   ├── test_template_json_export.py
│   └── test_validation.py
│
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── SECURITY.md
├── pyproject.toml
└── tox.ini
```

---

## Minimal First Commit Structure

The first commit should not create every future file. It should create only the files needed for the first end-to-end demonstration.

Target first demo:

```text
FID sequence → enriched object graph → PyPulseq .seq export → template JSON export → validation → plot
```

Recommended first commit:

```text
pypulseq-star/
├── .github/
│   ├── workflows/
│   │   ├── tests.yml
│   │   └── lint.yml
│   └── pull_request_template.md
│
├── docs/
│   ├── architecture.md
│   └── object_model.md
│
├── examples/
│   ├── fid.py
│   └── fid_export_json.py
│
├── src/
│   └── pypulseq_star/
│       ├── __init__.py
│       ├── py.typed
│       ├── opts.py
│       ├── units.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── node.py
│       │   ├── relationship.py
│       │   ├── parameter.py
│       │   └── timing.py
│       │
│       ├── shapes/
│       │   ├── __init__.py
│       │   ├── shape.py
│       │   └── block.py
│       │
│       ├── events/
│       │   ├── __init__.py
│       │   ├── event.py
│       │   ├── rf.py
│       │   └── adc.py
│       │
│       ├── blocks/
│       │   ├── __init__.py
│       │   └── block.py
│       │
│       ├── sequence/
│       │   ├── __init__.py
│       │   ├── sequence.py
│       │   └── timeline.py
│       │
│       ├── make/
│       │   ├── __init__.py
│       │   ├── rf.py
│       │   └── adc.py
│       │
│       ├── adapters/
│       │   ├── __init__.py
│       │   ├── pypulseq.py
│       │   ├── simple_namespace.py
│       │   └── template_json.py
│       │
│       ├── templates/
│       │   ├── __init__.py
│       │   ├── reader.py
│       │   └── mapper.py
│       │
│       ├── writers/
│       │   ├── __init__.py
│       │   ├── seq_writer.py
│       │   └── template_json_writer.py
│       │
│       ├── validation/
│       │   ├── __init__.py
│       │   └── validator.py
│       │
│       └── plotting/
│           ├── __init__.py
│           └── plotter.py
│
├── tests/
│   ├── conftest.py
│   ├── test_imports.py
│   ├── test_fid.py
│   ├── test_rf.py
│   ├── test_adc.py
│   ├── test_sequence.py
│   ├── test_pypulseq_export.py
│   └── test_template_json_export.py
│
├── .gitignore
├── .pre-commit-config.yaml
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── pyproject.toml
└── SECURITY.md
```

Files such as `CODE_OF_CONDUCT.md`, `CITATION.cff`, `CHANGELOG.md`, issue templates, and more specialized plotters/validators should be added before broader public release, but they do not need to block the first working prototype.

---

## Module Responsibilities

### `core/`

Defines the common object grammar.

Core classes should include:

```text
SeqStarNode
SeqStarParameter
SeqStarRelationship
SeqStarTiming
SeqStarMetadata
```

Every major object should be a node or contain a node-like interface:

```python
class SeqStarNode:
    id: str
    name: str
    path: str | None
    parent: "SeqStarNode | None"
    children: list["SeqStarNode"]
    parameters: dict[str, object]
    relationships: list["SeqStarRelationship"]
    metadata: dict[str, object]
    enabled: bool
    tstart: float | None
    duration: float | None
```

### `shapes/`

Contains waveform geometry only. Shapes should not know about sequence placement or export files.

Examples:

```text
SeqStarShape
SeqStarBlockShape
SeqStarSincShape
SeqStarGaussianShape
SeqStarTrapezoidShape
SeqStarArbitraryShape
```

### `events/`

Contains physical sequence events.

Examples:

```text
SeqStarRFEvent
SeqStarGradientEvent
SeqStarADCEvent
SeqStarDelayEvent
SeqStarTriggerEvent
SeqStarLabelEvent
```

Events may contain shapes:

```text
SeqStarRFEvent
└── SeqStarBlockShape
```

Events should not write files directly. They can expose structured data and PyPulseq-compatible forms through adapters.

### `blocks/`

Contains the PyPulseq-style time container.

A block contains one or more events that begin together or are intentionally grouped:

```text
SeqStarBlock
├── SeqStarRFEvent
├── SeqStarGradientEvent
└── SeqStarADCEvent
```

Do not add many block subclasses until they are needed. Prefer a generic `SeqStarBlock` with `role` metadata:

```python
block.role = "excitation"
block.role = "readout"
block.role = "spoiler"
```

### `sequence/`

Contains the sequence graph and timeline.

Recommended hierarchy:

```text
SeqStarSequence
└── SeqStarSection
    └── SeqStarKernel
        └── SeqStarBlock
            └── SeqStarEvent
                └── SeqStarShape
```

For FID:

```text
SeqStarSequence: fid
└── SeqStarSection: main
    └── SeqStarKernel: fid_kernel
        ├── SeqStarBlock: excitation
        │   └── SeqStarRFEvent: rf
        │       └── SeqStarBlockShape: block_shape
        └── SeqStarBlock: acquisition
            └── SeqStarADCEvent: adc
```

### `make/`

Public PyPulseq-like constructor layer.

Keep these names familiar:

```python
make_block_pulse(...)
make_sinc_pulse(...)
make_gauss_pulse(...)
make_trapezoid(...)
make_adc(...)
make_delay(...)
```

These functions should return enriched SeqStar objects, not raw `SimpleNamespace` objects.

### `adapters/`

Converts internal objects into external representations.

- `pypulseq.py`: convert `SeqStarSequence`, blocks, and events into PyPulseq-compatible objects.
- `simple_namespace.py`: build PyPulseq-style `SimpleNamespace` events where needed.
- `template_json.py`: convert object graph nodes into template-addressable dictionaries.

### `templates/`

Handles GammaSTAR template JSON reading and mapping.

This module should not contain separate blueprint definitions. It should only:

- Read a template.
- Resolve target paths.
- Map internal object fields into the template.
- Report unmapped required fields.

### `writers/`

Writes files.

- `seq_writer.py`: writes Pulseq `.seq`.
- `object_json_writer.py`: writes a generic object-tree JSON for debugging.
- `template_json_writer.py`: writes populated template JSON.
- `file_writer.py`: shared file-writing helpers.

### `validation/`

Checks correctness.

Validators should be composable:

```text
TimingValidator
RelationshipValidator
HardwareValidator
TemplateValidator
PyPulseqCompatibilityValidator
```

Validation categories:

- RF dead time and ringdown.
- ADC dead time.
- Gradient amplitude and slew.
- Missing `tstart` or `duration` after compilation.
- Invalid parent-child relationships.
- Template fields that could not be populated.
- PyPulseq export compatibility.

### `plotting/`

Keep plotting optional and import-light. Avoid importing Matplotlib at package import time.

Initial API:

```python
seq.plot()
seq.plot_tree()
seq.plot_timeline()
```

Internally, these can call:

```text
plotter.py
waveform_plotter.py
timeline_plotter.py
tree_plotter.py
```

---

## Public API Target

The package should feel like PyPulseq to the user:

```python
import numpy as np
import pypulseq_star as ppstar

system = ppstar.Opts(
    max_grad=28,
    grad_unit="mT/m",
    max_slew=120,
    slew_unit="mT/m/ms",
)

seq = ppstar.Sequence(system=system, name="fid")

rf = ppstar.make_block_pulse(
    flip_angle=np.pi / 2,
    duration=300e-6,
    use="excitation",
)

adc = ppstar.make_adc(
    num_samples=2048,
    duration=40e-3,
    delay=200e-6,
)

seq.add_block(rf, role="excitation")
seq.add_block(adc, role="acquisition")

seq.validate()
seq.plot()
seq.write_seq("fid.seq")
seq.write_template_json("fid.json", template="gammastar_template.json")
```

Each object should also expose the richer model:

```python
seq.tree()
seq.relationships
seq.to_dict()
seq.to_pypulseq()
```

---

## Dependency Direction Rules

To prevent circular imports, use this dependency direction:

```text
core
  ↓
shapes
  ↓
events
  ↓
blocks
  ↓
sequence
  ↓
make
  ↓
adapters / templates / validation / plotting / writers
```

Rules:

1. `core/` imports no package modules except standard library.
2. `shapes/` may import `core/`.
3. `events/` may import `core/` and `shapes/`.
4. `blocks/` may import `core/` and `events/`.
5. `sequence/` may import `core/` and `blocks/`.
6. `make/` may import `shapes/`, `events/`, and `blocks/`.
7. `adapters/`, `templates/`, `writers/`, `validation/`, and `plotting/` may import from the model, but the model should not import them.
8. `__init__.py` should re-export only stable public APIs.

---

## Packaging and PEP Compliance

### Distribution and import names

Use:

```text
Repository name:       pypulseq-star
Distribution name:    pypulseq-star
Python package name:  pypulseq_star
Import alias:         ppstar
```

### `src/` layout

Use the `src/` layout to avoid accidentally importing local source files during tests.

### `pyproject.toml`

Use `pyproject.toml` as the single source for packaging and tool configuration.

Recommended baseline:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pypulseq-star"
dynamic = ["version"]
description = "Structured PyPulseq-compatible sequence design with enriched hierarchy and template JSON export."
readme = "README.md"
requires-python = ">=3.10"
license = { file = "LICENSE" }
authors = [
  { name = "Sairam Geethanath" }
]
keywords = ["MRI", "Pulseq", "PyPulseq", "pulse sequence", "low-field MRI", "GammaSTAR"]
classifiers = [
  "Development Status :: 2 - Pre-Alpha",
  "Intended Audience :: Science/Research",
  "License :: OSI Approved :: MIT License",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Topic :: Scientific/Engineering :: Medical Science Apps.",
]
dependencies = [
  "numpy",
  "pypulseq",
]

[project.optional-dependencies]
dev = [
  "pytest",
  "pytest-cov",
  "ruff",
  "mypy",
  "pre-commit",
]
plotting = [
  "matplotlib",
]

[project.urls]
Homepage = "https://github.com/<ORG-OR-USER>/pypulseq-star"
Repository = "https://github.com/<ORG-OR-USER>/pypulseq-star"
Issues = "https://github.com/<ORG-OR-USER>/pypulseq-star/issues"

[tool.hatch.version]
path = "src/pypulseq_star/__init__.py"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N", "D"]
ignore = ["D100", "D104"]

[tool.mypy]
python_version = "3.10"
strict = true
packages = ["pypulseq_star"]
```

Adjust the license classifier to match the final license.

### PEP 8 naming rules

Use:

```text
Modules:      snake_case.py
Functions:    snake_case
Variables:    snake_case
Classes:      CapWords
Constants:    UPPER_SNAKE_CASE
Private APIs: _leading_underscore
```

Examples:

```text
Good: rf.py
Good: template_json_writer.py
Good: SeqStarRFEvent
Good: make_block_pulse
Avoid: RFEvent.py
Avoid: gammastarWriter.py
Avoid: seqStarNode
```

### Type hints

Use type hints from the beginning. Add `py.typed` so downstream users and type checkers know the package is typed.

### Docstrings

Use docstrings for public classes and public functions. Prefer concise NumPy-style docstrings because the MRI/scientific Python community is familiar with them.

---

## GitHub Repository Health Checklist

For a public scientific open-source repository, include:

```text
README.md
LICENSE
CONTRIBUTING.md
CODE_OF_CONDUCT.md
SECURITY.md
CITATION.cff
.github/workflows/tests.yml
.github/workflows/lint.yml
.github/ISSUE_TEMPLATE/bug_report.yml
.github/ISSUE_TEMPLATE/feature_request.yml
.github/pull_request_template.md
.gitignore
pyproject.toml
tests/
examples/
docs/
```

Minimum first commit:

```text
README.md
LICENSE
CONTRIBUTING.md
SECURITY.md
.github/workflows/tests.yml
.github/workflows/lint.yml
pyproject.toml
src/pypulseq_star/
tests/
examples/
```

---

## Recommended CI Checks

The GitHub Actions workflow should run at least:

```text
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src/pypulseq_star
pytest
```

Test Python versions:

```text
3.10
3.11
3.12
```

Add 3.13 later after dependencies support it.

---

## Do Not Build These Yet

To avoid half bridges, do not build these until there is a concrete use case:

```text
blocks/rf_block.py
blocks/readout_block.py
blocks/excitation_block.py
blocks/composite_block.py
sequence/average.py
sequence/repetition.py
sequence/loop.py
plotting/tree_plotter.py
plotting/timeline_plotter.py
validation/hardware_validator.py
validation/template_validator.py
serialization/from_dict.py
```

They are valid future modules, but they should be added only when the first FID/GRE path needs them.

---

## Milestone Plan

### Milestone 1: FID end-to-end

Deliver:

```text
make_block_pulse
make_adc
SeqStarSequence.add_block
SeqStarBlock
SeqStarRFEvent
SeqStarADCEvent
SeqStarBlockShape
Pulseq .seq export
template JSON export
basic validation
basic waveform plot
```

### Milestone 2: GRE minimum viable sequence

Add:

```text
make_trapezoid
SeqStarGradientEvent
SeqStarTrapezoidShape
RF + slice-select relationship
readout gradient + ADC relationship
spoiler gradients
basic sequence tree inspection
```

### Milestone 3: richer template writing

Add:

```text
template path resolver
unmapped-field report
required-field validation
relationship serialization
object-tree JSON debug export
```

### Milestone 4: developer-quality release

Add:

```text
full GitHub community files
CITATION.cff
CHANGELOG.md
API docs
coverage reporting
pre-commit hooks
examples gallery
```

---

## Final Recommendation

Build the repository in two layers:

1. **A small, working MVP** with FID, RF, ADC, blocks, sequence, `.seq` export, JSON template export, validation, and plotting.
2. **A planned full architecture** that adds gradients, relationships, tree visualization, hardware validators, and richer template mapping only when they become necessary.

The final architecture should remain unified around:

```text
core → shapes → events → blocks → sequence
```

Everything else should support that model:

```text
make      = construct enriched objects
adapters  = convert enriched objects
writers   = write files
templates = read and resolve template JSON
validation = check correctness
plotting  = visualize waveforms and hierarchy
```

This avoids half-built bridges while keeping the project scalable, PyPulseq-friendly, and compliant with modern Python packaging and GitHub repository practices.

---

## Reference Standards to Keep in Mind

- GitHub community profile files: README, license, contributing guidelines, code of conduct, security policy, issue templates, and pull request templates.
- Python packaging: `pyproject.toml`, `src/` layout, and standardized project metadata.
- PEP 8: naming, layout, readability, imports, and line-length discipline.
- PEP 621: standardized project metadata in `pyproject.toml`.
- PEP 561: include `py.typed` if the package ships inline type hints.
