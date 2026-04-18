"""
solvers/moment_of_inertia.py — SymPy moment of inertia solver.
"""

from __future__ import annotations

import math

from pydantic import BaseModel

from schemas import SolverResult


def solve_moment_of_inertia(params) -> SolverResult:
    """Compute Ixx (and Iyy where applicable) for standard cross-sections."""

    if isinstance(params, BaseModel):
        section = params.section_type
        dims = params.dimensions
    else:
        section = params["section_type"]
        dims = params["dimensions"]

    steps: list[str] = ["=" * 56, "  MOMENT OF INERTIA — CORRECT SOLUTION", "=" * 56]
    res: dict = {}

    if section == "rectangle":
        b, h = float(dims["b"]), float(dims["h"])
        Ixx = (b * h**3) / 12
        Iyy = (h * b**3) / 12
        steps += [
            f"  Section: Rectangle  b={b}mm  h={h}mm", "",
            "STEP 1 — Ixx = b·h³ / 12", "-" * 40,
            f"  Ixx = {b} × {h}³ / 12 = {Ixx:,.2f} mm⁴", "",
            "STEP 2 — Iyy = h·b³ / 12", "-" * 40,
            f"  Iyy = {Iyy:,.2f} mm⁴",
            "", "RESULT", "=" * 56,
            f"  Ixx = {Ixx:,.2f} mm⁴",
            f"  Iyy = {Iyy:,.2f} mm⁴",
        ]
        res = {"Ixx": Ixx, "Iyy": Iyy}

    elif section == "circle":
        d = float(dims["d"])
        I = math.pi * d**4 / 64
        steps += [
            f"  Section: Solid Circle  d={d}mm", "",
            f"  I = π·d⁴/64 = {I:,.4f} mm⁴",
            "", "RESULT", "=" * 56,
            f"  I = {I:,.4f} mm⁴",
        ]
        res = {"I": I}

    elif section == "hollow_circle":
        D, d = float(dims["D"]), float(dims["d"])
        if d >= D:
            raise ValueError("Inner diameter d must be less than outer D")
        I = math.pi * (D**4 - d**4) / 64
        steps += [
            f"  Section: Hollow Circle  D={D}mm  d={d}mm", "",
            f"  I = π(D⁴ - d⁴)/64 = {I:,.4f} mm⁴",
            "", "RESULT", "=" * 56,
            f"  I = {I:,.4f} mm⁴",
        ]
        res = {"I": I}

    elif section == "I_section":
        bf, tf = float(dims["bf"]), float(dims["tf"])
        hw, tw = float(dims["hw"]), float(dims["tw"])
        H = hw + 2 * tf
        vw = (bf - tw) / 2
        I_gross = (bf * H**3) / 12
        I_voids = 2 * (vw * hw**3) / 12
        Ixx = I_gross - I_voids
        steps += [
            f"  Section: I-Section  bf={bf} tf={tf} hw={hw} tw={tw} (mm)",
            f"  H = {H} mm", "",
            "STEP 1 — Subtraction method", "-" * 40,
            f"  I_gross = {I_gross:,.2f} mm⁴",
            f"  I_voids = {I_voids:,.2f} mm⁴",
            f"  Ixx = {I_gross:,.2f} - {I_voids:,.2f} = {Ixx:,.2f} mm⁴",
            "", "RESULT", "=" * 56,
            f"  Ixx = {Ixx:,.2f} mm⁴",
        ]
        res = {"Ixx": Ixx}

    else:
        raise ValueError(f"Unknown section_type: {section}")

    return SolverResult(solution_steps="\n".join(steps), numeric_results=res)
