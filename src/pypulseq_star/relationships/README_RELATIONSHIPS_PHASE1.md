    # relationships phase 1

    This folder implements the first minimal relationship layer needed by `demo_FID.py`.

    Implemented public APIs:

    ```python
    ppstar.relationships.set_center_after(...)
    ppstar.relationships.repeat_every(...)
    ppstar.relationships.resolve(seq)
    ppstar.relationships.summary(seq)
    ppstar.relationships.write_debug_json(seq, path)
    ```

    Required top-level package export:

    ```python
    # src/pypulseq_star/__init__.py
    from pypulseq_star import relationships as relationships
    ```

    This phase stores both resolved numerical timings and lightweight string/Lua-style relationship expressions. The SymPy backend is intentionally a placeholder for a later phase.
