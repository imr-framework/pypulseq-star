"""Phase-1 gammaSTAR/Lua relationship serialization helpers."""

from __future__ import annotations

from typing import Any

from .relationship import SeqStarRelationship


def relationship_to_lua_payload(relationship: SeqStarRelationship) -> dict[str, Any]:
    """Return a gammaSTAR-style script payload for one relationship."""

    expression = relationship.expression
    if expression is None:
        return {}

    return {
        "name": relationship.name,
        "relation_type": relationship.relation_type,
        "inputs": dict(expression.inputs),
        "script": expression.to_lua_script(),
        "resolved": dict(relationship.resolved),
    }
