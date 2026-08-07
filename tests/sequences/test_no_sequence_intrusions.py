"""Guard generic library layers against sequence-specific special cases."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.sequence, pytest.mark.semantic]

GENERIC_LAYERS = (
    "writers",
    "resolution",
    "relationships",
    "sequence",
)
SEQUENCE_TOKENS = {"fid", "gre", "epi", "tse"}
# This is a trajectory classification exported as root.info.is_epi, not a
# sequence-specific lowering branch. Keep exceptions narrow and reviewed.
ALLOWED_LITERAL_OCCURRENCES = {
    ("writers/gammastar_generic_document.py", "epi"),
}


def _suspicious_literals(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value.strip().lower()
        # Exact literals are the high-risk form used by sequence-specific
        # branches, dispatch tables, and hidden writer fallbacks. Prose and
        # documentation strings are deliberately ignored.
        relative = path.as_posix().split("/pypulseq_star/", 1)[-1]
        if value in SEQUENCE_TOKENS and (relative, value) not in ALLOWED_LITERAL_OCCURRENCES:
            hits.append((getattr(node, "lineno", 0), node.value))
    return hits


def test_generic_layers_do_not_dispatch_on_reference_sequence_names(repo_root) -> None:
    package = repo_root / "src" / "pypulseq_star"
    failures: list[str] = []
    for layer in GENERIC_LAYERS:
        for path in sorted((package / layer).rglob("*.py")):
            for line, value in _suspicious_literals(path):
                failures.append(f"{path.relative_to(repo_root)}:{line}: {value!r}")
    assert not failures, (
        "Generic layers must use event semantics, metadata contracts, and "
        "protocol relationships—not FID/GRE/EPI/TSE-specific dispatch:\n"
        + "\n".join(failures)
    )
