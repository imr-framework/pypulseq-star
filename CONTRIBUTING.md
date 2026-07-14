# Contributing to PyPulseq-Star

Thank you for helping improve PyPulseq-Star. The project is intended to serve PyPulseq/Pulseq developers, gammaSTAR users, MRI physicists, and researchers building reproducible sequence-development workflows.

PyPulseq-Star is currently pre-alpha. Focused issues, reproducible tests, and small pull requests are especially valuable while the protocol/relationship contract is being finalized.

## Project principles

Contributions should preserve these boundaries:

1. **Keep the public scripting experience close to PyPulseq.** Prefer familiar event constructors and `seq.add_block(...)` patterns unless a richer API is necessary.
2. **Store sequence intent explicitly.** Hierarchy, roles, loops, protocol controls, and relationships should not be inferred from a sequence name.
3. **Use generic relationship primitives.** Avoid FID-, GRE-, EPI-, TSE-, or vendor-specific logic in the relationship engine or writers.
4. **Keep writers target-specific.** The Pulseq writer produces a concrete numerical timeline; the gammaSTAR writer preserves hierarchy and dynamic dependencies where supported.
5. **Do not hide sequence logic in opaque templates.** Templates and low-level mappings belong inside writer/resource boundaries, not in developer-facing event construction.
6. **Add tests for public behavior.** Every constructor, relationship primitive, loop behavior, or writer change should have a focused regression test.
7. **Treat scanner execution as an experimental result.** Do not claim vendor or scanner compatibility without reporting the interpreter, system, software version, and validation procedure.

## Ways to contribute

### PyPulseq and Pulseq users

Helpful contributions include:

- comparing a PyPulseq-Star example with the corresponding canonical PyPulseq implementation;
- checking event timing, gradient moments, ADC windows, and `.seq` duration;
- testing generated `.seq` files with Pulseq tooling or research interpreters;
- reporting compatibility with a specific PyPulseq or Pulseq specification version; and
- proposing API changes that reduce friction for existing PyPulseq developers.

### gammaSTAR users

Helpful contributions include:

- importing generated `.seq.json` files into the gammaSTAR workplace;
- checking hierarchy, loops, event grouping, protocol controls, and plots;
- testing whether protocol edits propagate through exported dependencies;
- reporting differences between generated JSON and native gammaSTAR sequence organization; and
- proposing general mappings that apply to multiple sequence families.

### MRI sequence developers

Helpful contributions include:

- adding focused timing or gradient-moment validation;
- testing FID, GRE, EPI, and TSE behavior on additional systems;
- contributing phantoms, expected outputs, or site-validation procedures; and
- identifying sequence concepts that require new generic relationship primitives.

## Before opening an issue

Search existing issues first. For a bug report, include:

- PyPulseq-Star commit or release;
- Python version and operating system;
- PyPulseq version;
- Pulseq specification/interpreter version when relevant;
- gammaSTAR version or workplace environment when relevant;
- the smallest script that reproduces the problem;
- the complete traceback or validation report;
- expected and observed behavior; and
- generated files only when they contain no private, proprietary, or sensitive information.

For scanner-related reports, also include vendor, scanner model, software version, field strength, gradient system, coil, and any local sequence modifications. Do not upload patient data or protected health information.

## Development setup

```bash
git clone <repository-url>
cd pypulseq-star
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install dashboard dependencies only when needed:

```bash
python -m pip install -e ".[dashboard]"
```

Run the core checks:

```bash
python -m pytest
ruff check .
mypy src/pypulseq_star
python -m build
python -m twine check dist/*
```

Run one or more beginner examples:

```bash
python examples/demo_FID.py
python examples/demo_GRE.py
python examples/demo_EPI.py
python examples/demo_TSE.py
```

## Pull-request scope

Prefer pull requests that address one coherent concern. Good examples are:

- one bug plus its regression test;
- one generic relationship primitive plus tests and one example use;
- one writer correction plus reference-output tests;
- one documentation improvement; or
- one compatibility fix for a supported dependency version.

Large changes should begin with an issue describing the proposed contract and affected layers.

## Coding style

- Target Python 3.10 and later.
- Use type hints for public APIs and nontrivial internal functions.
- Keep lines within 100 characters unless an example call is clearer on one line; example scripts have limited `E501` exceptions.
- Use descriptive names for nodes, roles, counters, and relationships.
- Do not add sequence-family conditionals to generic writers when metadata or a general API can express the behavior.
- Do not add literals to gammaSTAR output when the value is an editable protocol parameter or should be represented by a dependency.
- Keep comments focused on sequence intent and non-obvious architectural decisions.

## Tests

Tests should be deterministic and should not require scanner access. Depending on the change, verify some combination of:

- resolved event timing and anchors;
- raster compliance and system limits;
- loop counts and event counts;
- gradient polarity or moment;
- ADC-window timing;
- Pulseq writer output and duration;
- gammaSTAR hierarchy, protocol references, and expressions;
- plotter logical expansion; and
- failure behavior for invalid protocols.

Use small fixtures. Do not commit large generated `.seq.json`, image, or scanner-data files to the main repository unless they are intentionally maintained test resources.

## Example-script expectations

The four beginner demos follow a common structure:

```text
SETUP
CREATE EVENTS
CONSTRUCT SEQUENCE
DEFINE RELATIONSHIPS
RESOLVE AND VALIDATE
PLOT
EXPORT
```

New examples should follow this structure where practical. They should teach the public API rather than expose writer internals or extensive diagnostic machinery.

## Commit and pull-request guidance

- Write a concise imperative commit title.
- Explain why the change is needed, not only what changed.
- Link the relevant issue.
- State which examples and tests were run.
- Note any changes to generated `.seq` or `.seq.json` behavior.
- Include screenshots only when they clarify plotting or workplace hierarchy.
- Keep generated outputs out of Git unless they are approved reference fixtures.

## Compatibility and deprecation

During pre-alpha development, APIs may change. Even so, contributors should:

- document intentional breaking changes;
- update all affected examples and tests in the same pull request;
- avoid silent changes to timing conventions; and
- provide a migration note when renaming a public parameter or relationship.

## Safety, data, and proprietary material

Do not contribute:

- credentials or private URLs;
- patient data or protected health information;
- proprietary vendor sequence code or documentation without permission;
- scanner files that cannot legally be redistributed; or
- unreviewed claims of clinical safety or efficacy.

All scanner-facing contributions must clearly identify their validation status.

## Attribution

Code contributors are recognized through Git history, release notes, and repository contributor records. Manuscript authorship is determined separately according to substantive scholarly contribution and responsibility for the submitted work. Please identify third-party code, algorithms, or data and preserve their licenses and citations.

## Questions

Use GitHub Discussions for design questions and community support when enabled. Use GitHub Issues for reproducible bugs, proposed features, and scoped development tasks.
