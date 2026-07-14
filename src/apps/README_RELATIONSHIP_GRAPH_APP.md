# PyPulseq-Star relationship graph app

This app is an optional, adoption-facing relationship graph visualizer.

## Install

```bash
pip install streamlit streamlit-flow-component
```

or, once `pyproject.toml` has a `viz` extra:

```bash
pip install -e ".[viz]"
```

## Run from the repository root

```bash
streamlit run src/apps/relationship_graph_viewer.py -- --graph out/fid/fid.relationship_graph.json
```

## Design split

The core sequence library builds and writes a graph JSON spec. The Streamlit app consumes that spec and displays an interactive React Flow dashboard.

This keeps the app sequence-general and avoids hardcoding FID, GRE, EPI, ASL, or pCASL logic into the UI.
