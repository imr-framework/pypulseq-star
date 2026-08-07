# Folder Structure Critique

The proposed structure is strong because it separates construction, adaptation,
validation, plotting, and writing. The main risk is building too much architecture before
one sequence works end to end.

Changes applied in the scaffold:

- Kept `src/pypulseq_star` with PEP-friendly package naming.
- Used `writers/` and `readers/templates` style boundaries instead of a broad `io/`.
- Kept `utils/` out of the first scaffold.
- Created only generic `SeqStarBlock`; no RF/readout/excitation block subclasses yet.
- Added relationship primitives early, because they are the main gap between PyPulseq
  waveform construction and gammaSTAR JSON semantics.
- Added placeholders only where they define package boundaries for the RF exercise.

The next RF milestone should deepen `events/rf.py`, `shapes/block.py`, and `make/rf.py`
before adding more folders.
