"""Helpers shared by the four reference-sequence test modules."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_demo(repo_root: Path, stem: str) -> ModuleType:
    path = repo_root / "examples" / f"demo_{stem}.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"pypulseq_star_test_demo_{stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(value, owner) -> float:
    """Evaluate a symbolic value or coerce an already numeric value."""
    evaluator = getattr(value, "eval", None)
    return float(evaluator(owner) if callable(evaluator) else value)


def assert_timing_passes(sequence) -> None:
    resolved = sequence.resolve()
    ok, report = resolved.check_timing()
    assert ok, "\n".join(str(item) for item in report)
