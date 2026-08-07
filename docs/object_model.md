# Object Model

The PyPulseq-Star object model separates protocol intent, symbolic specification, executable hierarchy, and numeric realization.

## Main objects

```text
Protocol
  parameters / aliases / metadata
  symbols -> ParameterRef / Expression

Sequence
  system limits
  encoding frame
  definitions
  Node hierarchy
  ordered Blocks
  Relationships

Block
  shared local time origin
  one event per physical channel/type where allowed

Event
  RF / gradient / ADC / ADC train / delay
  symbolic specification
  role and logical axis
  anchors
  metadata and variation bindings

ResolvedSequence
  immutable numeric protocol
  resolved events, blocks, nodes, and durations
  timing-validation interface
```

## Protocol and expressions

A `Protocol` is the canonical source of editable sequence parameters. `protocol.symbols` exposes references that can be used in ordinary arithmetic while preserving their source.

Expressions may represent:

- timing fills;
- event amplitudes and areas;
- loop counts and periods;
- phase progressions;
- derived protocol values; and
- orientation- or counter-dependent values.

## Hierarchy and loops

Nodes describe semantic organization independently of the concrete source block list.

```python
seq.set_node(
    "shot.kernel.echo_train",
    role="echo_train",
    repeat_count=p.echo_train_length,
    repeat_every=p.echo_spacing,
    counter="echo_index",
    repeat_mode="loop",
)
```

Writers and plotters may materialize the compact node structure differently, but they must preserve the same logical count, period, counter semantics, and event variations.

## Relationships and anchors

Events expose anchors such as start, center, and end. Relationships and duration expressions define why events are placed at particular times rather than storing only the final delay.

Examples:

- RF center to ADC center defines TE.
- Kernel occupancy plus fill defines TR.
- A readout gradient flat contains an ADC window.
- A phase-encode gradient is rewound after acquisition.
- A nested motif repeats with a protocol-controlled period.

## Logical and physical axes

Gradient events can carry:

- a construction channel (`x`, `y`, or `z`); and
- a logical axis role (`read`, `phase`, or `slice`).

The active `EncodingFrame` maps logical axes to physical scanner channels for plotting and export. Sequence logic remains unchanged across axial, coronal, and sagittal orientations.

## Numeric realization

`Sequence.resolve()` evaluates all expressions against the active protocol, validates ranges and timing, and returns an immutable numeric realization. Pulseq export should consume this realization. gammaSTAR export consumes the symbolic sequence plus validated defaults so it can preserve supported live dependencies.

## Metadata policy

Metadata may declare generic contracts that are not yet represented by a dedicated public object. Such metadata must:

- be backend-neutral at the sequence level;
- identify the property being varied;
- declare protocol and loop inputs explicitly;
- avoid sequence-name conditionals; and
- have tests before becoming release-critical.

Temporary metadata contracts should be promoted to stable APIs when they are used by multiple sequence families.
