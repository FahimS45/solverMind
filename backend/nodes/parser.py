"""
nodes/parser.py — Universal problem parser (LangChain).

Domain-agnostic: detects beam_reactions, moment_of_inertia, graph_theory,
or ANY other math subject. Extracts units dynamically from OCR text.

Returns:
    tuple[ParsedProblem, ProblemUnits] — both Pydantic-validated.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import get_settings
from schemas import (
    PARSED_VALIDATORS,
    ParsedProblem,
    ProblemUnits,
)

logger = logging.getLogger(__name__)

# ── Module-level chain cache (built once, reused) ────────────────────────────
_parser_chain = None


SYSTEM_PROMPT = """\
You are a UNIVERSAL math / science problem parser.
Your job: read OCR text from handwritten student homework and extract structured
parameters so a downstream solver can compute the correct answer.

Return ONLY valid JSON — no explanation, no markdown fences.

─── STEP 1: IDENTIFY THE PROBLEM TYPE ───

Read the OCR text and classify it into ONE of these categories:

  "beam_reactions"     — beam with supports, spans, point/distributed loads, moments
  "moment_of_inertia"  — cross-section geometry (rectangle, circle, I-section, etc.)
  "graph_theory"       — vertices, edges, paths, cycles, trees, BFS, DFS, shortest path, MST
  "other"              — ANY other math/science subject (calculus, algebra, statistics,
                         linear algebra, differential equations, probability, physics, etc.)

─── STEP 2: IDENTIFY UNITS ───

Look for unit indicators in the OCR text. Return a "units" object:
{{
  "units": {{
    "force" : "<force unit found, e.g. kN, N, lb, kip — default kN>",
    "moment": "<moment unit found, e.g. kN.m, N.m, lb.ft — default kN.m>",
    "length": "<length unit found, e.g. m, mm, ft, in — default m>"
  }}
}}

Unit detection rules:
- If the problem says "10 kN" → force="kN"
- If the problem says "5 N/m" → force="N", length="m"
- If the problem says "20 lb" or "20 lbs" → force="lb"
- If the problem says "span = 8 ft" → length="ft"
- Derive moment unit from force × length (e.g. force="kN" + length="m" → moment="kN.m")
- If no units are visible, use defaults: force="kN", moment="kN.m", length="m"

─── STEP 3: RETURN TYPE-SPECIFIC JSON ───

For BEAM REACTIONS return:
{{
  "problem_type": "beam_reactions",
  "support_type": "simply_supported" | "cantilever",
  "span": <total span, number>,
  "loads": [
    {{"type": "point", "magnitude": <force-units>, "position": <length-units from left>}},
    {{"type": "udl", "magnitude": <force/length>, "position": <start>, "length": <length>}},
    {{"type": "uvl", "w_start": <intensity at left>, "w_end": <intensity at right>, "position": <start>, "length": <length>}},
    {{"type": "moment", "magnitude": <moment-units>, "position": <from left>, "direction": "clockwise" | "anticlockwise"}}
  ],
  "units": {{ ... }}
}}

Load type rules:
- "point"  : concentrated force, always downward
- "udl"    : uniform load over a length, "magnitude" is intensity (force/length)
- "uvl"    : linearly varying — w_start=0 for zero-to-max triangle, w_end=0 for max-to-zero
- "moment" : applied couple — direction must be "clockwise" or "anticlockwise"

Support type rules:
- "simply_supported" : hinge (pin) at A, roller at B
- "cantilever"       : fixed wall at A, free end at B

For MOMENT OF INERTIA return:
{{
  "problem_type": "moment_of_inertia",
  "section_type": "rectangle" | "circle" | "hollow_circle" | "I_section",
  "dimensions": {{
    "b": <mm>, "h": <mm>,
    "d": <mm>,
    "D": <mm>,
    "bf": <mm>, "tf": <mm>, "hw": <mm>, "tw": <mm>
  }},
  "units": {{ "force": "N", "moment": "N.mm", "length": "mm" }}
}}

For GRAPH THEORY (CSE / discrete-math) return:
{{
  "problem_type"     : "graph_theory",
  "graph_description": "<directed/undirected/weighted, vertices, edges, weights>",
  "question"         : "<what must be solved or verified>",
  "student_claim"    : "<student answer from OCR, empty string if not visible>"
}}

For ANY OTHER MATH SUBJECT return:
{{
  "problem_type"  : "other",
  "subject"       : "<calculus | algebra | linear_algebra | statistics | probability | trigonometry | differential_equations | physics | ...>",
  "question"      : "<precise problem statement extracted from OCR>",
  "student_claim" : "<student's final answer from OCR, empty string if not visible>"
}}

If values are missing, make a reasonable assumption based on context."""


def _get_parser_chain():
    """Build and cache the parser chain (module-level singleton)."""
    global _parser_chain
    if _parser_chain is not None:
        return _parser_chain

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.parser_model,
        max_tokens=1000,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "Parse the following handwritten problem from OCR:\n\n{ocr_text}\n\nReturn JSON only."),
    ])

    _parser_chain = prompt | llm
    return _parser_chain


def parse_problem_from_ocr(ocr_text: str) -> tuple[ParsedProblem, ProblemUnits]:
    """
    Run the universal parser chain.

    Returns:
        (ParsedProblem, ProblemUnits) — both Pydantic-validated.

    Raises:
        ValueError: If the LLM returns invalid JSON or an unknown problem_type.
    """
    chain = _get_parser_chain()
    response = chain.invoke({"ocr_text": ocr_text})
    raw = response.content.strip()
    raw = re.sub(r"^```[a-z]*\n?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        params = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Parser returned invalid JSON:\n{raw}\n\nError: {e}")

    # ── Extract and validate units ────────────────────────────────────────
    raw_units = params.pop("units", {})
    units = ProblemUnits(**raw_units) if raw_units else ProblemUnits()

    # ── Validate parsed problem through Pydantic ──────────────────────────
    ptype = params.get("problem_type", "other")
    validator_cls = PARSED_VALIDATORS.get(ptype)
    if validator_cls is None:
        raise ValueError(
            f"Unknown problem_type '{ptype}' — expected one of {list(PARSED_VALIDATORS)}"
        )

    # Inject units for beam_reactions (it has a units field on the model)
    if ptype == "beam_reactions":
        params["units"] = units.model_dump()

    validated = validator_cls(**params)

    logger.info(
        "Parsed: type=%s, units=(force=%s, moment=%s, length=%s)",
        ptype, units.force, units.moment, units.length,
    )
    return validated, units
