"""Lightweight symbolic dependency backend for SeqStar relationships.

The public relationship API stays free of SymPy. This module provides a small
backend that can run with or without SymPy installed. It is intentionally useful
for writer/export validation: it tracks parameter dependencies, detects whether
one exported timing quantity depends on another, and can flag literal values in
live downstream paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def is_available() -> bool:
    """Return True if SymPy is importable in the current environment."""

    try:
        import sympy  # noqa: F401
    except Exception:
        return False
    return True


@dataclass(slots=True)
class DependencyExpression:
    """One symbolic dependency assignment.

    Parameters
    ----------
    target
        Name/path of the derived quantity, for example
        ``root.seqstar_loop.kernel.seqstar_blocks.kernel_readout.tstart``.
    inputs
        Mapping from local script variable names to source parameter paths.
    script
        Lua/gammaSTAR expression body used by the writer.
    metadata
        Optional debug metadata.
    """

    target: str
    inputs: dict[str, str] = field(default_factory=dict)
    script: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "inputs": dict(self.inputs),
            "script": self.script,
            "metadata": dict(self.metadata),
        }


class SymbolicDependencyGraph:
    """Small dependency graph for writer/debug validation.

    The graph is deliberately string/path based. It does not require SymPy, but
    exposes a stable place to add SymPy simplification later.
    """

    def __init__(self) -> None:
        self.expressions: dict[str, DependencyExpression] = {}

    def add_expression(
        self,
        target: str,
        *,
        inputs: Mapping[str, str] | None = None,
        script: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.expressions[str(target)] = DependencyExpression(
            target=str(target),
            inputs={str(k): str(v) for k, v in dict(inputs or {}).items()},
            script=script,
            metadata=dict(metadata or {}),
        )

    def dependencies_of(self, target: str, *, transitive: bool = True) -> set[str]:
        target = str(target)
        direct = set(self.expressions.get(target, DependencyExpression(target)).inputs.values())
        if not transitive:
            return direct
        out = set(direct)
        stack = list(direct)
        while stack:
            item = stack.pop()
            for dep in self.dependencies_of(item, transitive=False):
                if dep not in out:
                    out.add(dep)
                    stack.append(dep)
        return out

    def depends_on(self, target: str, source: str) -> bool:
        return str(source) in self.dependencies_of(str(target), transitive=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available_sympy": is_available(),
            "expressions": {key: expr.to_dict() for key, expr in self.expressions.items()},
        }


def parameter_is_literal(parameter: Any) -> bool:
    """Return True for a gammaSTAR-style parameter without input dependencies."""

    if not isinstance(parameter, Mapping):
        return True
    inputs = parameter.get("inputs")
    return not isinstance(inputs, Mapping) or len(inputs) == 0


def graph_from_gammastar_parameters(parameters: Mapping[str, Any]) -> SymbolicDependencyGraph:
    """Build a dependency graph from gammaSTAR JSON parameter entries."""

    graph = SymbolicDependencyGraph()
    for key, parameter in parameters.items():
        if not isinstance(parameter, Mapping):
            continue
        inputs = parameter.get("inputs")
        if isinstance(inputs, Mapping) and inputs:
            graph.add_expression(
                str(key),
                inputs={str(k): str(v) for k, v in inputs.items()},
                script=str(parameter.get("script", "")),
            )
    return graph
