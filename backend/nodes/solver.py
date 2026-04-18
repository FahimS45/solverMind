"""
nodes/solver.py — LangGraph Node 2: Parser + Solver dispatch.

Reads  : state['ocr_result']['markdown']
Writes : state['parsed_problem'], state['units'], state['solver_result']
         OR state['error']
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from schemas import ParsedProblem, ProblemUnits, SolverResult
from state import HWGraderState
from nodes.parser import parse_problem_from_ocr
from solvers.beam_reactions import solve_beam_reactions
from solvers.moment_of_inertia import solve_moment_of_inertia
from solvers.e2b_unified import solve_with_e2b

logger = logging.getLogger(__name__)


def dispatch_solver(
    parsed: ParsedProblem,
    units: ProblemUnits,
    ocr_text: str = "",
) -> SolverResult:
    """
    Route to the correct solver based on problem_type.
    Returns a validated SolverResult in all cases.
    """
    ptype = parsed.problem_type if isinstance(parsed, BaseModel) else parsed.get("problem_type", "")

    if ptype == "beam_reactions":
        return solve_beam_reactions(parsed, units=units)
    elif ptype == "moment_of_inertia":
        return solve_moment_of_inertia(parsed)
    elif ptype in ("graph_theory", "other"):
        return solve_with_e2b(parsed, ocr_text)
    else:
        raise ValueError(f"Unknown problem_type: '{ptype}'")


def node_solver(state: HWGraderState) -> dict:
    """
    LangGraph Node 2: Parser + Solver.

    Internally:
        2a — Universal parser → (ParsedProblem, ProblemUnits)
        2b — dispatch_solver  → SolverResult
    """
    logger.info("[Node 2/3] Solver — Parsing problem & computing solution")

    try:
        ocr_markdown = state["ocr_result"]["markdown"]

        # 2a — Universal parser
        logger.info("   2a: Universal parser → extracting problem parameters")
        parsed, units = parse_problem_from_ocr(ocr_markdown)

        # 2b — Dispatch to solver
        ptype = parsed.problem_type
        logger.info("   2b: Solver → computing correct solution (%s)", ptype)
        solved = dispatch_solver(parsed, units=units, ocr_text=ocr_markdown)

        logger.info("Solver complete")
        return {
            "parsed_problem": parsed.model_dump(),
            "units": units.model_dump(),
            "solver_result": solved.model_dump(),
        }

    except Exception as e:
        logger.exception("Solver node failed")
        return {"error": f"[Node 2 Solver] {type(e).__name__}: {e}"}
