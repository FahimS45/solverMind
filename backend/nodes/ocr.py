"""
nodes/ocr.py — LangGraph Node 1: Chandra OCR.

Reads  : state['image_input']
Writes : state['ocr_result'] (validated OCRResult)  OR  state['error']
"""

from __future__ import annotations

import logging
from PIL import Image

from schemas import OCRResult
from state import HWGraderState
from services.ocr_api import ocr_via_api_sync

logger = logging.getLogger(__name__)


def node_ocr(state: HWGraderState) -> dict:
    """
    LangGraph Node 1 — OCR via Chandra hosted API.

    Accepts single or multiple images. All pages are concatenated into one
    markdown string with --- PAGE N --- separators for multi-page submissions.
    """
    logger.info("[Node 1/3] OCR — Chandra hosted API")

    try:
        img_input = state["image_input"]
        inputs = (
            [img_input]
            if isinstance(img_input, (str, Image.Image))
            else img_input
        )

        all_markdown: list[str] = []

        for i, item in enumerate(inputs):
            logger.info("   Page %d/%d", i + 1, len(inputs))

            if isinstance(item, Image.Image):
                tmp_path = f"_tmp_ocr_page_{i}.jpg"
                item.save(tmp_path, "JPEG")
                path = tmp_path
            else:
                path = item

            page_md = ocr_via_api_sync(path)
            all_markdown.append(
                f"--- PAGE {i + 1} ---\n{page_md}" if len(inputs) > 1 else page_md
            )

        combined = "\n\n".join(all_markdown)
        logger.info("OCR complete — %d page(s), %d chars", len(inputs), len(combined))

        ocr_out = OCRResult(
            markdown=combined,
            raw_ocr=combined,
            page_count=len(inputs),
            image=None,
        )
        return {"ocr_result": ocr_out.model_dump()}

    except Exception as e:
        logger.exception("OCR node failed")
        return {"error": f"[Node 1 OCR] {type(e).__name__}: {e}"}
