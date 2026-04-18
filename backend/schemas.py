"""
schemas.py — Pydantic models for every node boundary in the pipeline.

Production contract:
  OCR   → OCRResult
  Parse → (ParsedProblem, ProblemUnits)
  Solve → SolverResult
  Critic→ CriticOutput

Every node validates its output through these models before returning,
ensuring deterministic shape for FastAPI serialisation and downstream nodes.
"""

from __future__ import annotations
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# OCR Result
# ═══════════════════════════════════════════════════════════════════════════════

class OCRResult(BaseModel):
    """Output of Node 1 (OCR). Deterministic shape for downstream nodes."""

    markdown: str = Field(description="Combined OCR markdown from all pages")
    raw_ocr: str = Field(description="Unprocessed OCR output")
    page_count: int = Field(ge=1, description="Number of pages processed")
    image: Optional[str] = Field(default=None, description="Optional image reference")


# ═══════════════════════════════════════════════════════════════════════════════
# Units (dynamic — parsed from OCR)
# ═══════════════════════════════════════════════════════════════════════════════

class ProblemUnits(BaseModel):
    """
    Units extracted by the parser from the OCR text.
    The solver uses these instead of hardcoded 'kN' / 'kN.m' / 'm'.
    """

    force: str = Field(default="kN", description="Force unit, e.g. kN, N, lb, kip")
    moment: str = Field(default="kN.m", description="Moment unit, e.g. kN.m, N.m, lb.ft")
    length: str = Field(default="m", description="Length unit, e.g. m, mm, ft, in")


# ═══════════════════════════════════════════════════════════════════════════════
# Load models (typed components of ParsedBeamReactions)
# ═══════════════════════════════════════════════════════════════════════════════

class PointLoad(BaseModel):
    type: Literal["point"] = "point"
    magnitude: float = Field(description="Force magnitude in force-units")
    position: float = Field(description="Distance from left support in length-units")


class UDLoad(BaseModel):
    type: Literal["udl"] = "udl"
    magnitude: float = Field(description="Load intensity in force/length units")
    position: float = Field(description="Start position from left support")
    length: float = Field(description="Length of distributed load")


class UVLoad(BaseModel):
    type: Literal["uvl"] = "uvl"
    w_start: float = Field(description="Intensity at left end of load region")
    w_end: float = Field(description="Intensity at right end of load region")
    position: float = Field(description="Start position from left support")
    length: float = Field(description="Length of load region")


class MomentLoad(BaseModel):
    type: Literal["moment"] = "moment"
    magnitude: float = Field(description="Moment magnitude in moment-units")
    position: float = Field(description="Position from left support")
    direction: Literal["clockwise", "anticlockwise"] = "clockwise"


LoadUnion = Union[PointLoad, UDLoad, UVLoad, MomentLoad]


# ═══════════════════════════════════════════════════════════════════════════════
# Parsed Problem — discriminated by problem_type
# ═══════════════════════════════════════════════════════════════════════════════

class ParsedBeamReactions(BaseModel):
    problem_type: Literal["beam_reactions"] = "beam_reactions"
    support_type: Literal["simply_supported", "cantilever"]
    span: float = Field(gt=0, description="Total span in length-units")
    loads: list[LoadUnion]
    units: ProblemUnits = Field(default_factory=ProblemUnits)


class ParsedMomentOfInertia(BaseModel):
    problem_type: Literal["moment_of_inertia"] = "moment_of_inertia"
    section_type: Literal["rectangle", "circle", "hollow_circle", "I_section"]
    dimensions: dict = Field(description="Section dimensions in mm")
    units: ProblemUnits = Field(
        default_factory=lambda: ProblemUnits(force="N", moment="N.mm", length="mm")
    )


class ParsedGraphTheory(BaseModel):
    problem_type: Literal["graph_theory"] = "graph_theory"
    graph_description: str = Field(description="Graph structure description")
    question: str = Field(description="What must be solved or verified")
    student_claim: str = Field(default="", description="Student's answer from OCR")


class ParsedOther(BaseModel):
    problem_type: Literal["other"] = "other"
    subject: str = Field(description="Math subject area")
    question: str = Field(description="The problem statement to solve")
    student_claim: str = Field(default="", description="Student's answer from OCR")


ParsedProblem = Union[ParsedBeamReactions, ParsedMomentOfInertia, ParsedGraphTheory, ParsedOther]

# Map for validation dispatch
PARSED_VALIDATORS: dict[str, type[BaseModel]] = {
    "beam_reactions": ParsedBeamReactions,
    "moment_of_inertia": ParsedMomentOfInertia,
    "graph_theory": ParsedGraphTheory,
    "other": ParsedOther,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Solver Result
# ═══════════════════════════════════════════════════════════════════════════════

class SolverResult(BaseModel):
    """Deterministic output contract for every solver path."""

    solution_steps: str = Field(description="Human-readable step-by-step solution")
    generated_code: Optional[str] = Field(default=None, description="Python code (E2B only)")
    execution_result: Optional[str] = Field(default=None, description="E2B stdout (E2B only)")
    numeric_results: Optional[dict] = Field(default=None, description="Key answers, e.g. {RA, RB}")
    sfd: Optional[dict] = Field(default=None, description="Shear force diagram data")
    bmd: Optional[dict] = Field(default=None, description="Bending moment diagram data")


# ═══════════════════════════════════════════════════════════════════════════════
# Critic Output
# ═══════════════════════════════════════════════════════════════════════════════

class CriticOutput(BaseModel):
    """Structured grading output from the Math Tutor critic."""

    grade: Literal["Pass", "Fail"] = Field(
        description="Pass if fully correct, Fail if any error is found."
    )
    error_line: str = Field(
        description="Step/line of the FIRST error, e.g. 'Step 3'. 'None' if no error."
    )
    error_location_detail: str = Field(
        default="None",
        description=(
            "Precise, quotable description of WHERE the error appears "
            "so a teacher can locate it on the physical paper."
        ),
    )
    error_type: Literal[
        "Arithmetic", "Conceptual", "Formula", "Sign Error", "Units", "None"
    ] = Field(description="Error category. 'None' if grade is Pass.")
    explanation: str = Field(
        description="2-4 sentences explaining what went wrong."
    )
    correct_step: str = Field(
        description="Correct version of the erroneous line. 'N/A' if Pass."
    )
    additional_notes: str = Field(
        default="None", description="Other observations or 'None'."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# API response envelopes (FastAPI endpoint schemas)
# ═══════════════════════════════════════════════════════════════════════════════

class SSENodeEvent(BaseModel):
    """Shape of each SSE event pushed to the frontend."""

    node: str = Field(description="Which node just completed: ocr | solver | critic")
    status: Literal["success", "error"] = "success"
    data: dict = Field(default_factory=dict, description="Node output payload")
    elapsed_seconds: float = Field(description="Seconds since pipeline start")


class GradeResponse(BaseModel):
    """Final synchronous response (non-SSE fallback)."""

    ocr_result: Optional[OCRResult] = None
    parsed_problem: Optional[dict] = None
    units: Optional[ProblemUnits] = None
    solver_result: Optional[SolverResult] = None
    critic_result: Optional[CriticOutput] = None
    error: Optional[str] = None
    total_seconds: float = 0.0
