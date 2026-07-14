"""PyPulseq-Star public API."""

from __future__ import annotations

from pypulseq_star import relationships as relationships
from pypulseq_star.make import (
    make_adc,
    make_adc_train,
    make_arbitrary_grad,
    make_arbitrary_rf,
    make_block_pulse,
    make_gauss_pulse,
    make_sinc_pulse,
    make_trapezoid,
    split_gradient,
    make_delay,
)
from pypulseq_star.opts import Opts
from pypulseq_star.plotting.plotter import SeqStarPlotter, plot
from pypulseq_star.plotting.relationship_grapher import (
    RelationshipGrapher,
    plot_relationship_graph,
    relationship_graph_text,
    write_relationship_graph,
    write_relationship_graph_dot,
    write_relationship_graph_mermaid,
)
from pypulseq_star.protocol import Protocol
from pypulseq_star.sequence import SeqStarSequence as Sequence

__version__ = "0.1.0a1"
# PyPulseq-compatible numerical tolerance.
eps = 1e-12

__all__ = [
    "Opts",
    "Protocol",
    "RelationshipGrapher",
    "Sequence",
    "SeqStarPlotter",
    "make_adc",
    "make_adc_train",
    "make_arbitrary_grad",
    "make_arbitrary_rf",
    "make_block_pulse",
    "make_gauss_pulse",
    "make_sinc_pulse",
    "make_trapezoid",
    "make_delay",
    "plot",
    "relationship_graph_text",
    "relationships",
    "split_gradient",
    "write_relationship_graph",
    "write_relationship_graph_dot",
    "write_relationship_graph_mermaid",
    "plot_relationship_graph",
  
]