"""
graph.py — LangGraph construction and compilation.

Builds:  START → ocr → should_continue? → solver → critic → END
                              ↓ (error)
                             END
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from nodes.ocr import node_ocr
from nodes.solver import node_solver
from nodes.critic import node_critic
from state import HWGraderState


def _should_continue(state: HWGraderState) -> str:
    """Conditional edge: abort to END if any previous node set an error."""
    if state.get("error"):
        return "abort"
    return "ok"


@lru_cache
def build_graph():
    """
    Build and compile the LangGraph.

    Returns:
        CompiledGraph — exposes .invoke() (blocking) and .stream() (node-by-node).
    """
    builder = StateGraph(HWGraderState)

    # Register nodes
    builder.add_node("ocr", node_ocr)
    builder.add_node("solver", node_solver)
    builder.add_node("critic", node_critic)

    # Edges
    builder.add_edge(START, "ocr")
    builder.add_conditional_edges(
        "ocr",
        _should_continue,
        {"ok": "solver", "abort": END},
    )
    builder.add_edge("solver", "critic")
    builder.add_edge("critic", END)

    return builder.compile()
