# Expression layer — first pass

This package implements the first Phase 2 expression contract:

- protocol parameters become lightweight `ParameterRef` objects;
- ordinary arithmetic builds an inspectable expression tree;
- expressions retain canonical dependency names;
- `Expression.eval(...)` accepts a mapping, Protocol, Sequence, resolved
  realization, or explicit `EvaluationContext`;
- deferred sequence-owned values are represented by generic reference nodes;
- no raw SymPy objects are exposed to developers.

## Intended initial use

```python
p = protocol.symbols

te_fixed_occupancy = seq.duration(
    start=rf.anchor("center"),
    end=adc.anchor("start"),
)

te_fill_duration = p.echo_time - te_fixed_occupancy

print(te_fill_duration.dependencies)
print(te_fill_duration.eval(seq))
```

## Deliberately deferred

This first pass does not yet:

- add `Protocol.symbols`;
- add symbolic bindings to event properties;
- implement `event.anchor(...)`;
- implement `Sequence.duration(...)`;
- implement `Sequence.resolve()`;
- compile expressions to gammaSTAR;
- change Pulseq export.

Those integrations should be added one layer at a time after reviewing this
expression API.
