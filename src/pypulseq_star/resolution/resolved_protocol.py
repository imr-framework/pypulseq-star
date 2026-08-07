"""Numeric protocol view produced by sequence resolution."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ResolvedProtocol(Mapping[str, Any]):
    """Read-only numeric protocol values with canonical and alias access.

    ``values`` stores canonical parameter names only. ``aliases`` maps public
    names such as ``num_averages`` and ``averages`` to the canonical key.
    """

    values: Mapping[str, Any]
    aliases: Mapping[str, str] = field(default_factory=dict)

    def canonical_name(self, key: str) -> str:
        return str(self.aliases.get(key, key))

    def __getitem__(self, key: str) -> Any:
        return self.values[self.canonical_name(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            available = ", ".join(sorted(str(key) for key in self.values))
            raise AttributeError(
                f"{name!r} is not present in the resolved protocol. "
                f"Available canonical parameters: {available}"
            ) from exc

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def has_parameter(self, key: str) -> bool:
        return self.canonical_name(key) in self.values

    def to_dict(self, *, include_aliases: bool = False) -> dict[str, Any]:
        result = dict(self.values)
        if include_aliases:
            for alias, canonical in self.aliases.items():
                if canonical in self.values:
                    result.setdefault(alias, self.values[canonical])
        return result
