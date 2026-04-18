"""
main.py — FastAPI application with SSE streaming endpoint.

Endpoints:
    POST /grade          → SSE stream (real-time node-by-node progress)
    POST /grade/sync     → JSON response (blocking, returns final result)
    GET  /health         → Health check

SSE event format (each event is a JSON line):
    event: node_complete
    data: {"node": "ocr", "status": "success", "data": {...}, "elapsed_seconds": 2.3}

    event: pipeline_complete
    data: {"status": "success", "total_seconds": 12.5}

    event: pipeline_error
    data: {"status": "error", "error": "...", "total_seconds": 5.1}
"""


import json
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import get_settings
from graph import build_graph
from schemas import GradeResponse, SSENodeEvent
from state import make_initial_state
from nodes.critic import format_critic_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-5s  %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Lifespan — warm up graph at startup
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build and cache the LangGraph once at startup."""
    settings = get_settings()
    logger.info("Building LangGraph pipeline...")
    graph = build_graph()
    logger.info("Pipeline ready. Listening on %s:%s", settings.host, settings.port)
    yield


# ═══════════════════════════════════════════════════════════════════════════════
# App factory
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="The Math Tutor",
    description="AI-powered handwritten homework grader with real-time SSE streaming.",
    version="2.0.0",
    lifespan=lifespan,
    openapi_version="3.0.2",  # 3.1 uses contentMediaType which Swagger UI can't render as file pickers
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

NODE_LABELS = {
    "ocr": "Chandra OCR",
    "solver": "Parser + Solver",
    "critic": "Critic",
}


def _sse_event(event_type: str, data: dict) -> str:
    """Format a single SSE event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _save_upload(upload: UploadFile, tmpdir: str) -> str:
    """Save an uploaded file to a temp directory, return path."""
    dest = os.path.join(tmpdir, upload.filename or "upload.jpg")
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


# ═══════════════════════════════════════════════════════════════════════════════
# POST /grade — SSE streaming endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/grade", response_class=StreamingResponse)
async def grade_sse(files: List[UploadFile] = File(description="Homework image files")):
    """
    Grade handwritten homework with real-time SSE progress.

    Upload one or more image files. The pipeline streams events as each
    node completes:

        event: node_complete       — after ocr / solver / critic
        event: pipeline_complete   — final summary with critic report
        event: pipeline_error      — if any node fails

    Connect from the frontend with EventSource or fetch().
    """

    async def generate() -> AsyncGenerator[str, None]:
        tmpdir = tempfile.mkdtemp(prefix="math_tutor_")

        try:
            # Save uploaded files
            paths = [_save_upload(f, tmpdir) for f in files]
            image_input = paths[0] if len(paths) == 1 else paths

            graph = build_graph()
            initial_state = make_initial_state(image_input)
            final_state = initial_state.copy()
            t_start = time.time()

            # ── Stream through graph nodes ────────────────────────────────
            for step in graph.stream(initial_state):
                node_name = list(step.keys())[0]
                node_output = step[node_name]
                final_state.update(node_output)

                elapsed = time.time() - t_start
                label = NODE_LABELS.get(node_name, node_name)

                # Check for error
                if final_state.get("error"):
                    event = SSENodeEvent(
                        node=node_name,
                        status="error",
                        data={"error": final_state["error"], "label": label},
                        elapsed_seconds=round(elapsed, 2),
                    )
                    yield _sse_event("node_complete", event.model_dump())
                    yield _sse_event("pipeline_error", {
                        "status": "error",
                        "error": final_state["error"],
                        "total_seconds": round(elapsed, 2),
                    })
                    return

                # Build node-specific summary
                summary: dict = {"label": label}
                if node_name == "ocr":
                    ocr = final_state.get("ocr_result", {})
                    summary["page_count"] = ocr.get("page_count", 1)
                    summary["char_count"] = len(ocr.get("markdown", ""))
                    summary["preview"] = ocr.get("markdown", "")[:400]
                elif node_name == "solver":
                    parsed = final_state.get("parsed_problem", {})
                    summary["problem_type"] = parsed.get("problem_type", "unknown")
                    summary["units"] = final_state.get("units", {})
                    sr = final_state.get("solver_result", {})
                    if sr.get("execution_result"):
                        summary["e2b_output_preview"] = sr["execution_result"][:600]
                elif node_name == "critic":
                    cr = final_state.get("critic_result", {})
                    summary["grade"] = cr.get("grade")
                    summary["error_type"] = cr.get("error_type")
                    summary["error_line"] = cr.get("error_line")

                event = SSENodeEvent(
                    node=node_name,
                    status="success",
                    data=summary,
                    elapsed_seconds=round(elapsed, 2),
                )
                yield _sse_event("node_complete", event.model_dump())

            # ── Pipeline complete ─────────────────────────────────────────
            total = time.time() - t_start
            yield _sse_event("pipeline_complete", {
                "status": "success",
                "total_seconds": round(total, 2),
                "critic_result": final_state.get("critic_result"),
                "parsed_problem": final_state.get("parsed_problem"),
                "units": final_state.get("units"),
                "solver_steps": final_state.get("solver_result", {}).get("solution_steps", ""),
            })

        except Exception as e:
            logger.exception("Pipeline crashed")
            yield _sse_event("pipeline_error", {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "total_seconds": round(time.time() - t_start, 2),
            })
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# POST /grade/sync — Synchronous JSON endpoint (non-SSE fallback)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/grade/sync", response_model=GradeResponse)
async def grade_sync(files: List[UploadFile] = File(description="Homework image files")):
    """
    Grade homework synchronously. Returns a single JSON response
    after the full pipeline completes. Use /grade for SSE streaming.
    """
    tmpdir = tempfile.mkdtemp(prefix="math_tutor_")
    try:
        paths = [_save_upload(f, tmpdir) for f in files]
        image_input = paths[0] if len(paths) == 1 else paths

        graph = build_graph()
        initial_state = make_initial_state(image_input)
        t_start = time.time()

        final_state = graph.invoke(initial_state)
        total = time.time() - t_start

        return GradeResponse(
            ocr_result=final_state.get("ocr_result"),
            parsed_problem=final_state.get("parsed_problem"),
            units=final_state.get("units"),
            solver_result=final_state.get("solver_result"),
            critic_result=final_state.get("critic_result"),
            error=final_state.get("error"),
            total_seconds=round(total, 2),
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# GET /health
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "models": {
            "parser": settings.parser_model,
            "critic": settings.critic_model,
        },
        "services": {
            "e2b": bool(settings.e2b_api_key),
            "chandra": bool(settings.chandra_api_key),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )