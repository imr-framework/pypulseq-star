"""Public relationship APIs for pypulseq-star.

The package-level namespace re-exports the generic relationship primitives so
developers can use:

    ppstar.relationships.set_anchor_after(...)
    ppstar.relationships.set_same_anchor(...)
    ppstar.relationships.require_same_anchor(...)
    ppstar.relationships.repeat_every(...)
    ppstar.relationships.resolve(seq)

No sequence-specific behavior is introduced here.
"""

from __future__ import annotations

from .expression import SeqStarExpression, expr
from .graph import relationship_graph, summary, write_debug_json
from .relationship import (
    SeqStarRelationship,
    attach_relationship,
    get_relationships,
)
from .timing import (
    block_after,
    fill_to_period,
    get_anchor_time,
    repeat_every,
    require_same_anchor,
    resolve,
    set_anchor_after,
    set_center_after,
    set_center_equal,
    set_readout_after,
    set_same_anchor,
    use_same_anchor,
)
from .validation import SeqStarValidationResult

__all__ = [
    "SeqStarExpression",
    "SeqStarRelationship",
    "SeqStarValidationResult",
    "attach_relationship",
    "block_after",
    "fill_to_period",
    "expr",
    "get_anchor_time",
    "get_relationships",
    "relationship_graph",
    "repeat_every",
    "require_same_anchor",
    "resolve",
    "set_anchor_after",
    "set_center_after",
    "set_center_equal",
    "set_readout_after",
    "set_same_anchor",
    "summary",
    "use_same_anchor",
    "write_debug_json",
]
