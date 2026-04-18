"""
solvers/beam_reactions.py — SymPy beam reactions solver with SFD + BMD.

All output text uses dynamic units from the parser (fu, mu, lu)
instead of hardcoded kN / kN.m / m.
"""

from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel
from sympy import Eq, symbols, solve as sym_solve

from schemas import ProblemUnits, SolverResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers (shear / moment at a point)
# ═══════════════════════════════════════════════════════════════════════════════

def _to_dict(load) -> dict:
    return load if isinstance(load, dict) else load.model_dump()


def _shear_at(x: float, ra: float, loads: list) -> float:
    """Shear force at position x, accumulating from left support."""
    V = ra
    for load in loads:
        ld = _to_dict(load)
        lt = ld["type"]
        if lt == "point":
            if x > ld["position"]:
                V -= ld["magnitude"]
        elif lt == "udl":
            x0, L, w = ld["position"], ld["length"], ld["magnitude"]
            if x > x0:
                V -= w * min(x - x0, L)
        elif lt == "uvl":
            x0, L = ld["position"], ld["length"]
            ws, we = ld.get("w_start", 0.0), ld.get("w_end", 0.0)
            if x > x0:
                s = min(x - x0, L)
                V -= ws * s + (we - ws) * s ** 2 / (2 * L)
        # applied moments do not affect shear
    return V


def _moment_at(x: float, ra: float, loads: list) -> float:
    """Bending moment at position x, accumulating from left support."""
    M = ra * x
    for load in loads:
        ld = _to_dict(load)
        lt = ld["type"]
        if lt == "point":
            a = ld["position"]
            if x > a:
                M -= ld["magnitude"] * (x - a)
        elif lt == "udl":
            x0, L, w = ld["position"], ld["length"], ld["magnitude"]
            if x > x0:
                s = min(x - x0, L)
                M -= w * s * (x - x0 - s / 2)
        elif lt == "uvl":
            x0, L = ld["position"], ld["length"]
            ws, we = ld.get("w_start", 0.0), ld.get("w_end", 0.0)
            if x > x0:
                s = min(x - x0, L)
                xrel = x - x0
                M -= ws * s * (xrel - s / 2) + (we - ws) / L * (s**2 / 2 * xrel - s**3 / 3)
        elif lt == "moment":
            a = ld["position"]
            direction = ld.get("direction", "clockwise").lower()
            mag = ld["magnitude"]
            if x > a:
                M -= mag if direction == "clockwise" else -mag
    return M


def _find_zero_shear(ra: float, loads: list, span: float) -> list[float]:
    """Binary-search sign changes in V(x) → locations of max BM."""
    xs = np.linspace(0, span, 5000)
    Vs = np.array([_shear_at(float(xi), ra, loads) for xi in xs])
    crossings = []
    for k in range(len(Vs) - 1):
        if Vs[k] * Vs[k + 1] < 0:
            lo, hi = float(xs[k]), float(xs[k + 1])
            for _ in range(50):
                mid = (lo + hi) / 2
                if _shear_at(mid, ra, loads) * _shear_at(lo, ra, loads) < 0:
                    hi = mid
                else:
                    lo = mid
            crossings.append((lo + hi) / 2)
    return crossings


# ═══════════════════════════════════════════════════════════════════════════════
# Main solver
# ═══════════════════════════════════════════════════════════════════════════════

def solve_beam_reactions(params, units: ProblemUnits | None = None) -> SolverResult:
    """
    Solve beam reactions analytically with dynamic units.

    Args:
        params: ParsedBeamReactions (Pydantic) or raw dict.
        units:  ProblemUnits from the parser. Falls back to kN/m defaults.

    Returns:
        SolverResult with solution_steps, numeric_results, sfd, bmd.
    """
    if units is None:
        units = ProblemUnits()

    fu = units.force
    mu = units.moment
    lu = units.length
    flu = f"{fu}/{lu}"

    RA_sym, RB_sym = symbols("RA RB", real=True)

    if isinstance(params, BaseModel):
        span = float(params.span)
        loads = [l.model_dump() for l in params.loads]
        stype = params.support_type
    else:
        span = float(params["span"])
        loads = params["loads"]
        stype = params.get("support_type", "simply_supported")

    steps: list[str] = [
        "=" * 60,
        "  BEAM REACTIONS + SFD + BMD  —  CORRECT SOLUTION",
        "=" * 60,
        f"  Span         : {span} {lu}",
        f"  Support type : {stype.replace('_', ' ').title()}",
        f"  Units        : force={fu}, moment={mu}, length={lu}",
        "",
        "  SIGN CONVENTION:",
        "    Vertical forces  — Upward = positive",
        "    Moments          — Anti-clockwise (ACW) = positive",
        "    Shear force      — Upward forces to LEFT = positive",
        "    Bending moment   — Sagging = positive",
        "",
    ]

    # ── STEP 1: Resolve loads ────────────────────────────────────────────
    steps += ["=" * 60, "STEP 1 — IDENTIFY & RESOLVE ALL LOADS", "=" * 60, ""]

    total_vert = 0.0
    mom_A = 0.0

    for i, load in enumerate(loads):
        lt = load["type"]
        steps.append(f"  --- Load {i + 1} ---")

        if lt == "point":
            F, x = float(load["magnitude"]), float(load["position"])
            total_vert += F
            mom_A += F * x
            steps += [
                f"  Type: Point Load | F = {F} {fu} at x = {x} {lu}",
                f"  Moment about A: {F} × {x} = {F * x} {mu} (ACW)",
                "",
            ]
        elif lt == "udl":
            w, x0, L = float(load["magnitude"]), float(load["position"]), float(load["length"])
            Feq = w * L
            xeq = x0 + L / 2
            total_vert += Feq
            mom_A += Feq * xeq
            steps += [
                f"  Type: UDL | w = {w} {flu} from x = {x0} to {x0 + L} {lu}",
                f"  Feq = {w} × {L} = {Feq} {fu} at centroid x = {xeq} {lu}",
                f"  Moment about A: {Feq} × {xeq} = {round(Feq * xeq, 4)} {mu}",
                "",
            ]
        elif lt == "uvl":
            ws = float(load.get("w_start", 0.0))
            we = float(load.get("w_end", 0.0))
            x0, L = float(load["position"]), float(load["length"])
            Feq = 0.5 * (ws + we) * L
            xc = L * (2 * we + ws) / (3 * (ws + we)) if (ws + we) > 0 else L / 2
            xeq = x0 + xc
            total_vert += Feq
            mom_A += Feq * xeq
            steps += [
                f"  Type: UVL | {ws} → {we} {flu} from x = {x0} to {x0 + L} {lu}",
                f"  Feq = 0.5 × ({ws}+{we}) × {L} = {Feq} {fu}",
                f"  Centroid at x = {round(xeq, 4)} {lu}",
                f"  Moment about A: {round(Feq * xeq, 4)} {mu}",
                "",
            ]
        elif lt == "moment":
            M_val, x = float(load["magnitude"]), float(load["position"])
            d = load.get("direction", "clockwise").lower()
            sign = -1 if d == "clockwise" else 1
            mom_A += sign * M_val
            steps += [
                f"  Type: Applied Moment | M = {M_val} {mu} ({d.upper()}) at x = {x} {lu}",
                f"  Contribution to ΣMA: {'+' if sign > 0 else '-'}{M_val} {mu}",
                "",
            ]

    steps += [
        "-" * 60,
        f"  ΣFy = {total_vert} {fu}",
        f"  ΣMA = {round(mom_A, 4)} {mu}",
        "",
    ]

    # ── STEP 2+3: Equilibrium ────────────────────────────────────────────
    if stype == "simply_supported":
        steps += [
            "=" * 60, "STEP 2 — EQUILIBRIUM EQUATIONS", "=" * 60,
            f"  ΣFy = 0:  RA + RB = {total_vert}     … (1)",
            f"  ΣMA = 0:  RB × {span} = {round(mom_A, 4)}  … (2)",
            "",
        ]
        sol = sym_solve(
            [Eq(RA_sym + RB_sym, total_vert), Eq(RB_sym * span, mom_A)],
            [RA_sym, RB_sym],
        )
        ra, rb = float(sol[RA_sym]), float(sol[RB_sym])
        steps += [
            "=" * 60, "STEP 3 — SOLVE", "=" * 60,
            f"  RB = {round(mom_A, 4)} / {span} = {rb:.4f} {fu}",
            f"  RA = {total_vert} - {rb:.4f} = {ra:.4f} {fu}",
            "",
            "=" * 60, "STEP 4 — VERIFICATION", "=" * 60,
            f"  RA + RB = {round(ra + rb, 4)} {fu}  →  {'OK' if abs(ra + rb - total_vert) < 0.01 else 'ERROR'}",
            f"  RB × L = {round(rb * span, 4)} {mu}  →  {'OK' if abs(rb * span - mom_A) < 0.01 else 'ERROR'}",
            "",
        ]
    elif stype == "cantilever":
        ra, rb = total_vert, 0.0
        steps += [
            "=" * 60, "STEP 2 — EQUILIBRIUM (CANTILEVER)", "=" * 60,
            f"  RA = {ra:.4f} {fu}",
            f"  MA_fix = {round(mom_A, 4)} {mu}",
            "",
        ]
    else:
        raise ValueError(f"Unknown support_type: '{stype}'")

    # ── STEP 5: SFD ──────────────────────────────────────────────────────
    steps += ["=" * 60, "STEP 5 — SHEAR FORCE DIAGRAM (SFD)", "=" * 60, ""]

    key_xs = sorted(set(
        [0.0, span]
        + [float(l["position"]) for l in loads]
        + [float(l["position"]) + float(l.get("length", 0)) for l in loads if "length" in l]
    ))

    steps.append(f"  {'x (' + lu + ')':<12} {'V(x) ' + fu:<16} Notes")
    steps.append("  " + "-" * 55)

    for idx, x in enumerate(key_xs):
        V_after = _shear_at(x + 1e-9, ra, loads)
        if x == 0:
            steps.append(f"  {x:<12.3f} {V_after:<16.4f} START: V = RA = {ra:.4f} {fu}")
        elif x == span:
            V_before = _shear_at(x - 1e-9, ra, loads)
            steps.append(f"  {x:<12.3f} {V_before:<16.4f} END")
            steps.append(
                f"  {'':12} {'After RB:':<16} {V_before + rb:.4f} {fu}  "
                f"{'(OK)' if abs(V_before + rb) < 0.01 else '(check)'}"
            )
        else:
            V_before = _shear_at(x - 1e-9, ra, loads)
            if abs(V_after - V_before) > 0.01:
                steps.append(f"  {x:<12.3f} {V_before:<16.4f} Just LEFT")
                steps.append(f"  {x:<12.3f} {V_after:<16.4f} Just RIGHT (Δ = {V_after - V_before:.4f} {fu})")
            else:
                steps.append(f"  {x:<12.3f} {V_after:<16.4f}")

    zero_x = _find_zero_shear(ra, loads, span)
    if zero_x:
        steps.append("")
        steps.append("  Zero shear crossing(s):")
        for zx in zero_x:
            steps.append(f"    x = {zx:.4f} {lu}")

    xs_plot = np.linspace(0, span, 300)
    Vs_plot = [_shear_at(float(xi), ra, loads) for xi in xs_plot]
    max_sf = max(abs(v) for v in Vs_plot)
    steps += ["", f"  Max shear force = {max_sf:.4f} {fu}", ""]

    # ── STEP 6: BMD ──────────────────────────────────────────────────────
    steps += ["=" * 60, "STEP 6 — BENDING MOMENT DIAGRAM (BMD)", "=" * 60, ""]

    steps.append(f"  {'x (' + lu + ')':<12} {'M(x) ' + mu:<16} Notes")
    steps.append("  " + "-" * 55)

    for x in key_xs:
        M_val = _moment_at(x, ra, loads)
        M_before = _moment_at(x - 1e-9, ra, loads)
        M_after = _moment_at(x + 1e-9, ra, loads)

        note = ""
        if x == 0:
            note = "M = 0 (hinge)"
        elif x == span:
            note = f"{'OK' if abs(M_val) < 0.05 else 'CHECK'}"

        if abs(M_after - M_before) > 0.01 and x not in (0.0, span):
            steps.append(f"  {x:<12.3f} {M_before:<16.4f} Just LEFT")
            steps.append(f"  {x:<12.3f} {M_after:<16.4f} Just RIGHT (step = {M_after - M_before:.4f} {mu})")
        else:
            steps.append(f"  {x:<12.3f} {M_val:<16.4f} {note}")

    xs_bm = np.linspace(0, span, 2000)
    Ms_bm = [_moment_at(float(xi), ra, loads) for xi in xs_bm]
    max_bm = max(Ms_bm, key=abs)
    max_x = float(xs_bm[Ms_bm.index(max_bm)])

    steps += [
        "",
        f"  Max BM = {max_bm:.4f} {mu}  at x = {max_x:.4f} {lu}",
        "",
        "=" * 60, "  FINAL RESULTS", "=" * 60,
        f"  RA     = {ra:.3f} {fu}",
    ]
    numeric: dict = {"RA": ra}
    if stype == "simply_supported":
        steps.append(f"  RB     = {rb:.3f} {fu}")
        numeric["RB"] = rb
    else:
        steps.append(f"  MA_fix = {mom_A:.3f} {mu}")
        numeric["MA"] = mom_A
    steps += [
        f"  Vmax   = {max_sf:.3f} {fu}",
        f"  Mmax   = {abs(max_bm):.3f} {mu}  at x = {max_x:.3f} {lu}",
        "=" * 60,
    ]

    return SolverResult(
        solution_steps="\n".join(steps),
        numeric_results=numeric,
        sfd={"x": [float(v) for v in xs_plot], "V": Vs_plot},
        bmd={"x": [float(v) for v in xs_bm], "M": Ms_bm},
    )
