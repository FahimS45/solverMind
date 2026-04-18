"""
solvers/e2b_unified.py — Unified E2B sandbox solver.

Handles BOTH graph_theory and other (general math) problem types
through a single LLM code-gen → E2B execution pipeline.

For graph theory:  instructs LLM to use NetworkX.
For general math:  LLM picks sympy / numpy / scipy as appropriate.
"""

from __future__ import annotations

import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import get_settings
from schemas import SolverResult

logger = logging.getLogger(__name__)

# ── Module-level chain cache ─────────────────────────────────────────────────
_e2b_code_chain = None


def _get_e2b_code_chain():
    """Single LangChain chain for all E2B-solved problem types."""
    global _e2b_code_chain
    if _e2b_code_chain is not None:
        return _e2b_code_chain

    settings = get_settings()
    llm = ChatOpenAI(model=settings.parser_model, max_tokens=2000)

    SYSTEM = (
        "You are an expert mathematician, computer scientist, and Python programmer.\n"
        "Given a problem extracted from handwritten student work (OCR),\n"
        "write Python code that:\n"
        "  1. Solves the problem step by step, printing EVERY intermediate result.\n"
        "  2. Uses the most appropriate library:\n"
        "     - networkx : for graph theory (BFS, DFS, shortest path, MST, etc.)\n"
        "     - sympy    : for symbolic math, calculus, algebra, ODEs, trigonometry\n"
        "     - numpy / scipy : for numerical linear algebra, statistics, root finding\n"
        "     - plain print() : for proofs, logic, or problems needing only reasoning\n"
        "  3. Prints the final answer with a CLEAR label at the end.\n"
        "  4. Mirrors the solution style a textbook would use (show working, not just answer).\n\n"
        "{extra_instructions}\n\n"
        "Return ONLY raw Python code — no markdown fences, no explanation.\n"
        "networkx, sympy, numpy, scipy are pre-installed; do NOT add pip/install lines."
    )

    USER = (
        "Problem type     : {problem_type}\n"
        "Subject area     : {subject}\n"
        "Question to solve: {question}\n\n"
        "OCR text (full student submission including their attempt):\n---\n{ocr_text}\n---\n\n"
        "Write the Python solution code now:"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("user", USER),
    ])
    _e2b_code_chain = prompt | llm
    return _e2b_code_chain


def _strip_fences(raw: str) -> str:
    """Remove ```python / ``` markdown fences the LLM might add."""
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()


def solve_with_e2b(params, ocr_text: str) -> SolverResult:
    """
    Unified E2B solver for graph_theory and other problem types.
    LLM generates Python code → E2B executes → stdout becomes solution_steps.

    Returns a validated SolverResult.
    """
    settings = get_settings()

    if isinstance(params, BaseModel):
        ptype = params.problem_type
        p = params.model_dump()
    else:
        ptype = params.get("problem_type", "other")
        p = params

    # ── Build type-specific context ───────────────────────────────────────
    if ptype == "graph_theory":
        subject = "graph theory / discrete mathematics"
        question = p.get("question", "Solve the graph theory problem")
        extra = (
            "GRAPH THEORY SPECIFIC:\n"
            "  - Import networkx as nx.\n"
            "  - Build the EXACT graph described (vertices, edges, weights).\n"
            "  - Print every intermediate result (adjacency, traversal order, etc.).\n"
            f"  - Graph description: {p.get('graph_description', 'See OCR text')}"
        )
    else:
        subject = p.get("subject", "general mathematics")
        question = p.get("question", "Solve the problem")
        extra = ""

    logger.info("[E2B/%s] Generating Python code via LLM", subject)
    chain = _get_e2b_code_chain()
    response = chain.invoke({
        "problem_type": ptype,
        "subject": subject,
        "question": question,
        "extra_instructions": extra,
        "ocr_text": ocr_text,
    })
    code = _strip_fences(response.content)
    logger.info("[E2B/%s] Code generated (%d lines). Running in sandbox", subject, len(code.splitlines()))

    # ── Execute in E2B sandbox ────────────────────────────────────────────
    if not settings.e2b_api_key:
        result_text = (
            "[E2B not configured]\n"
            "Set E2B_API_KEY in .env to enable sandbox execution.\n"
        )
    else:
        from e2b_code_interpreter import Sandbox

        sbx = None
        try:
            sbx = Sandbox.create()
            execution = sbx.run_code(code)

            stdout = "\n".join(execution.logs.stdout) if execution.logs.stdout else ""
            stderr = "\n".join(execution.logs.stderr) if execution.logs.stderr else ""

            if stdout:
                result_text = stdout
            elif stderr:
                result_text = f"[stderr output]\n{stderr}"
            else:
                result_text = "(sandbox ran with no output)"

            if stderr and stdout:
                result_text += f"\n\n[stderr warnings]\n{stderr}"

        except Exception as exc:
            result_text = f"[E2B error]  {type(exc).__name__}: {exc}"
        finally:
            if sbx is not None:
                try:
                    sbx.kill()
                except Exception:
                    pass

    logger.info("[E2B/%s] Solver complete", subject)

    lines = (
        ["=" * 60,
         f"  {subject.upper()} — CORRECT SOLUTION (E2B sandbox)",
         "=" * 60, "",
         "GENERATED PYTHON CODE", "-" * 40]
        + code.split("\n")
        + ["", "EXECUTION OUTPUT", "-" * 40]
        + result_text.split("\n")
        + ["", "=" * 60]
    )

    return SolverResult(
        solution_steps="\n".join(lines),
        generated_code=code,
        execution_result=result_text,
    )
