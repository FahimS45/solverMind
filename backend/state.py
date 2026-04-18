"""
state.py — LangGraph state definition.

Every node receives the full HWGraderState and returns only
a partial dict update. LangGraph merges automatically.
"""

from __future__ import annotations
from typing import TypedDict, Optional, Union


class HWGraderState(TypedDict):
    """Shared state flowing through the LangGraph pipeline."""

    image_input: Union[str, list[str], list]       # file path(s) or PIL Images
    ocr_result: Optional[dict]                     # OCRResult.model_dump()
    parsed_problem: Optional[dict]                 # ParsedProblem.model_dump()
    units: Optional[dict]                          # ProblemUnits.model_dump()
    solver_result: Optional[dict]                  # SolverResult.model_dump()
    critic_result: Optional[dict]                  # CriticOutput fields + raw
    error: Optional[str]


def make_initial_state(image_input) -> HWGraderState:
    """Build a clean initial state from image input(s)."""
    return HWGraderState(
        image_input=image_input,
        ocr_result=None,
        parsed_problem=None,
        units=None,
        solver_result=None,
        critic_result=None,
        error=None,
    )
