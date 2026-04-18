"""
nodes/critic.py — LangGraph Node 3: Critic.

Compares student OCR work against the solver's correct solution.
Uses CriticOutput Pydantic model bound via .with_structured_output().

Reads  : state['ocr_result'], state['solver_result'], state['parsed_problem']
Writes : state['critic_result']  OR  state['error']
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import get_settings
from schemas import CriticOutput
from state import HWGraderState

logger = logging.getLogger(__name__)

# ── Module-level chain cache ─────────────────────────────────────────────────
_critic_chain = None

SYSTEM_PROMPT = (
    "You are a fair and experienced {role} grading handwritten homework that was "
    "digitised by an OCR model.\n\n"

    "═══ GROUND RULES ═══\n\n"

    "RULE 1 — VERIFY NUMERICALLY, NOT VISUALLY.\n"
    "Before calling anything an error, evaluate BOTH the student's expression "
    "and the reference expression to a number. If they produce the same result, "
    "they are correct — full stop. Do NOT penalise:\n"
    "  - Different factor ordering  (24×4×½  is identical to  ½×24×4)\n"
    "  - Different grouping style   (a×b×c  vs  (a×b)×c)\n"
    "  - Different but equivalent forms  (2L/3 from zero end  =  L/3 from max end)\n"
    "  - Rearranged equilibrium equations that balance correctly\n\n"

    "RULE 2 — CROSS-CHECK THE FINAL ANSWER FIRST.\n"
    "Before scrutinising any individual step, check whether the student's final "
    "numerical answer matches the correct answer.\n"
    "  - If the final answer matches → look hard for an innocent explanation "
    "before declaring any step wrong.\n"
    "  - If the final answer does NOT match → work backwards from where the "
    "numbers diverge.\n\n"

    "RULE 3 — OCR DOUBT BEFORE STUDENT DOUBT.\n"
    "When a value or symbol looks wrong, ask:\n"
    "  (a) Could OCR have misread it? (0↔6, 1↔7, +↔-, ×↔x, missing exponents)\n"
    "  (b) Does the rest of the working follow logically?\n"
    "  (c) Only if BOTH fail → genuine error. Otherwise flag as NOTE, not error.\n\n"

    "RULE 4 — HOLISTIC METHOD BEFORE LINE-BY-LINE DETAIL.\n"
    "Read the entire solution once before judging. A student who applies the right "
    "method with a single arithmetic slip ≠ conceptual misunderstanding.\n\n"

    "═══ GRADING STEPS ═══\n\n"

    "1. Number each line/step.\n"
    "2. Apply Rule 2: check final answer.\n"
    "3. Apply Rule 1: evaluate each expression numerically.\n"
    "4. Apply Rule 3: OCR doubt test.\n"
    "5. Identify the FIRST genuine error.\n"
    "6. Fill error_location_detail with step number AND exact expression.\n"
    "7. Classify: Arithmetic / Conceptual / Formula / Sign Error / Units.\n\n"

    "═══ PASS CONDITION ═══\n"
    "Grade as PASS if the final answer is correct OR all discrepancies are "
    "explainable by OCR noise / equivalent forms."
)

USER_TEMPLATE = (
    "Topic: {topic}\n\n"
    "STUDENT HANDWRITTEN WORK (from OCR):\n"
    "{divider}\n"
    "{student_work}\n"
    "{divider}\n\n"
    "CORRECT SOLUTION (computed by solver):\n"
    "{divider}\n"
    "{correct_solution}\n"
    "{divider}\n\n"
    "Grade the student's work now."
)

# Role map per problem type
ROLE_MAP = {
    "beam_reactions": "Civil Engineering professor specialising in structural analysis",
    "moment_of_inertia": "Civil Engineering professor specialising in structural mechanics",
    "graph_theory": "Computer Science professor specialising in graph theory and algorithms",
    "other": "mathematics professor experienced in grading undergraduate work",
}


def _get_critic_chain():
    """Build and cache the critic chain with structured output."""
    global _critic_chain
    if _critic_chain is not None:
        return _critic_chain

    settings = get_settings()
    llm = ChatOpenAI(model=settings.critic_model)
    structured_llm = llm.with_structured_output(CriticOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_TEMPLATE),
    ])

    _critic_chain = prompt | structured_llm
    return _critic_chain


def node_critic(state: HWGraderState) -> dict:
    """
    LangGraph Node 3 — Critic.

    Returns CriticOutput fields in state['critic_result'], including
    the serialised Pydantic object for FastAPI.
    """
    logger.info("[Node 3/3] Critic — Grading student work")

    try:
        chain = _get_critic_chain()
        divider = "-" * 50
        ptype = state["parsed_problem"]["problem_type"]
        topic = ptype.replace("_", " ").title()
        role = ROLE_MAP.get(ptype, "professor")

        output: CriticOutput = chain.invoke({
            "topic": topic,
            "role": role,
            "divider": divider,
            "student_work": state["ocr_result"]["markdown"],
            "correct_solution": state["solver_result"]["solution_steps"],
        })

        result = {
            "grade": output.grade,
            "error_line": output.error_line,
            "error_location_detail": output.error_location_detail,
            "error_type": output.error_type,
            "explanation": output.explanation,
            "correct_step": output.correct_step,
            "additional_notes": output.additional_notes,
            "output": output.model_dump(),
        }

        logger.info("Critic complete: grade=%s", output.grade)
        return {"critic_result": result}

    except Exception as e:
        logger.exception("Critic node failed")
        return {"error": f"[Node 3 Critic] {type(e).__name__}: {e}"}


def format_critic_report(critic: dict) -> str:
    """Format critic result as a human-readable string."""
    if critic is None:
        return "No critic result — check state['error'] for details."

    icon = "✅  PASS" if critic["grade"] == "Pass" else "❌  FAIL"
    lines = [
        "=" * 60,
        "  GRADING REPORT",
        "=" * 60,
        f"  {icon}  —  GRADE : {critic['grade']}",
        f"  ERROR LINE     : {critic['error_line']}",
        f"  ERROR TYPE     : {critic['error_type']}",
    ]

    loc = critic.get("error_location_detail", "None")
    if loc not in ("", "None", None):
        lines.append("\n  ERROR LOCATION:")
        for line in loc.split("\n"):
            lines.append(f"       {line}")

    lines.append("\n  EXPLANATION :")
    for line in critic["explanation"].split("\n"):
        lines.append(f"       {line}")

    lines.append(f"\n  CORRECT STEP:\n       {critic['correct_step']}")

    notes = critic.get("additional_notes")
    if notes not in ("", "None", None):
        lines.append("\n  NOTES:")
        for line in notes.split("\n"):
            lines.append(f"       {line}")

    lines.append("=" * 60)
    return "\n".join(lines)
