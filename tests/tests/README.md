# PyPulseq-Star tests

The suite is organized by subsystem rather than by development sprint:

```text
tests/
  helpers/         reusable sequence builders
  unit/            shapes, events, sequence, relationships, plotter,
                   dashboard-data, and writer contracts
  integration/     four demos, installation, and cross-writer smoke tests
```

Run all tests from the repository root:

```bash
python run_tests.py
```

Common alternatives:

```bash
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --smoke
python run_tests.py --coverage
python run_tests.py --coverage -- -x -vv
```

The dashboard tests validate the generated JSON and command contract without
launching Streamlit. Scanner execution is not part of the automated suite.
