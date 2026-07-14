# Architecture

PyPulseq-Star should feel like PyPulseq at the scripting surface while carrying a richer
SeqStar object model underneath.

The key architectural choice is to make relationships first-class. Pulseq blocks answer
"what happens at this time"; gammaSTAR-style JSON also needs "why this event exists",
"where it sits in the hierarchy", and "which object it balances, samples, or contains."

Dependency direction:

```text
core -> shapes -> events -> blocks -> sequence -> make
```

Adapters, templates, writers, validators, and plotting depend on that model. The model
must not depend on file writers or gammaSTAR-specific implementation details.

For this first scaffold, the package includes only the RF/ADC-ready minimum. Gradient,
relationship-heavy, and tree plotting modules should be added when they are needed by
the next concrete sequence milestone.
