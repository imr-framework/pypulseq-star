# PyPulseq-Star relationship graph assets

Place the icon folder at:

```text
src/pypulseq_star/resources/icons/relationship_graph/
```

Suggested packaging note for `pyproject.toml`:

```toml
[tool.setuptools.package-data]
pypulseq_star = ["resources/icons/relationship_graph/*.svg"]
```

The plotting code currently draws most symbols with Matplotlib primitives so the figure works even if the icon assets are absent. These SVGs are intended for documentation, future HTML/Plotly views, and eventual richer icon rendering.

Logo note: the PyPulseq-Star logo uses the existing pulse-like PyPulseq motif plus one blue star in the top-right. It avoids a yellow star.
