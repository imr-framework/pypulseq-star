#!/usr/bin/env python3
"""Print a concise summary from pytest-cov's coverage.json output."""
from __future__ import annotations

import json
from pathlib import Path

path = Path("coverage.json")
if not path.exists():
    raise SystemExit("coverage.json not found. Run ./tests/run_tests.sh html first.")

data = json.loads(path.read_text())
totals = data["totals"]
print(
    f"Total coverage: {totals['percent_covered_display']}% | "
    f"statements {totals['covered_lines']}/{totals['num_statements']} | "
    f"branches {totals.get('covered_branches', 0)}/{totals.get('num_branches', 0)}"
)

files = sorted(
    data["files"].items(),
    key=lambda item: item[1]["summary"]["percent_covered"],
)
print("\nLowest-covered files:")
for filename, payload in files[:15]:
    summary = payload["summary"]
    print(f"{summary['percent_covered']:6.1f}%  {filename}")
