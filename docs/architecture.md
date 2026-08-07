# Architecture

PyPulseq-Star keeps the scripting surface close to PyPulseq while carrying a richer protocol-, hierarchy-, and relationship-aware model underneath.

## Core pipeline

```text
Protocol parameters
        |
        v
Symbolic expressions and event specifications
        |
        v
Sequence hierarchy, nodes, loops, blocks, and relationships
        |
        +--> immutable numeric realization
        |       +--> timing and safety-oriented software validation
        |       +--> local plotting
        |       +--> Pulseq writer --> concrete .seq
        |
        +--> gammaSTAR writer --> compact hierarchy + live dependencies
```

## Architectural boundaries

### Public construction layer

Developers use PyPulseq-like constructors and `seq.add_block(...)`. The public API should expose sequence intent without requiring knowledge of writer templates or backend field names.

### Protocol and expression layer

Protocol references retain:

- canonical parameter names;
- current/default values;
- units and constraints where available;
- dependencies; and
- evaluation behavior.

Derived values should remain expressions until resolution unless a target explicitly requires a concrete value.

### Sequence and hierarchy layer

The sequence owns:

- executable block order;
- semantic node hierarchy;
- loop counts and periods;
- counters;
- variation declarations;
- encoding frame; and
- relationships.

A compact motif should not be expanded merely to make a writer easier to implement.

### Resolution layer

Resolution evaluates protocol expressions, applies raster and constraint rules, validates event properties, and creates an immutable numeric realization. Writers must not silently reinterpret unresolved symbolic timing.

### Writer layer

The writers have different responsibilities:

```text
Pulseq writer
    lowers loops and variations
    emits concrete events and delays
    preserves numerical timing

gammaSTAR writer
    preserves compact hierarchy
    exports protocol controls and dependencies
    lowers declared generic expression bindings
```

Generic writers must not contain FID-, GRE-, EPI-, TSE-, or vendor-specific branches. Sequence-specific behavior belongs in the demo or a reusable public abstraction declared through generic metadata/API contracts.

### Plotting layer

The plotter consumes the same sequence/realization model as the writers. Logical read/phase/slice gradients must be mapped through the active `EncodingFrame`, not grouped solely by their construction channel.

## Dependency direction

A practical dependency direction is:

```text
core/protocol/expressions
        -> shapes/events
        -> blocks/sequence/relationships
        -> resolution
        -> plotting/writers/adapters
```

The model must not depend on output writers or gammaSTAR implementation details.

## v0.2.0 architecture contract

v0.2.0 establishes:

- protocol-centered symbolic construction;
- immutable numeric realization;
- compact nodes and loops;
- generic variation/expression bindings;
- logical orientation mapping;
- ADC trains; and
- dual Pulseq/gammaSTAR export.

The exact release boundary is documented in [`release_scope_v0.2.0.md`](release_scope_v0.2.0.md).
