# Object Model

The object model is organized around five layers:

```text
SeqStarSequence
  SeqStarBlock
    SeqStarEvent
      SeqStarShape
```

Shapes describe waveform geometry. Events describe physical actions. Blocks group events
with a shared local time origin. Sequences own the executable timeline and the semantic
tree.

Relationships are explicit records:

```text
source -> target, kind, metadata
```

Examples:

- `excitation_block contains_event rf`
- `rf has_shape rf_shape`
- `readout_adc samples readout_gradient`
- `slice_refocus balances slice_select`

The point is to let developers code in familiar PyPulseq-like syntax while giving the
writer enough information to produce meaningful JSON without rediscovering intent from
raw waveforms.
