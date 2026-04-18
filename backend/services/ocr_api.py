"""
services/ocr_api.py — Chandra hosted OCR API client.

Handles submission, polling, and result extraction.
Async-ready with httpx for FastAPI compatibility.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import httpx

from config import get_settings

logger = logging.getLogger(__name__)


async def ocr_via_api(image_path: str) -> str:
    """
    Submit one image to Chandra hosted API, poll until done, return markdown.

    Raises:
        RuntimeError:  API returned non-200 or processing error.
        TimeoutError:  Polling exceeded max attempts.
    """
    settings = get_settings()
    headers = {"X-API-Key": settings.chandra_api_key}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # ── Submit ────────────────────────────────────────────────────────
        with open(image_path, "rb") as f:
            resp = await client.post(
                settings.chandra_api_url,
                headers=headers,
                files={"file": (Path(image_path).name, f, "image/jpeg")},
                data={"output_format": "markdown", "mode": "balanced"},
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Chandra API returned {resp.status_code}: {resp.text[:200]}"
            )

        job = resp.json()
        if not job.get("success"):
            raise RuntimeError(f"Submission failed: {job.get('error')}")

        check_url = job["request_check_url"]
        logger.info("Chandra job submitted. Polling %s", check_url)

        # ── Poll ──────────────────────────────────────────────────────────
        for attempt in range(settings.chandra_max_polls):
            await asyncio.sleep(settings.chandra_poll_interval)
            result = (await client.get(check_url, headers=headers)).json()
            status = result.get("status", "")
            logger.debug("Poll attempt %d: %s", attempt + 1, status)

            if status == "complete":
                return result.get("markdown", "")
            if status == "error":
                raise RuntimeError(f"API processing error: {result.get('error')}")

    raise TimeoutError("Chandra API did not complete within the timeout window.")


def ocr_via_api_sync(image_path: str) -> str:
    """
    Synchronous wrapper for environments without an event loop (Jupyter, CLI).
    """
    import requests
    import time

    settings = get_settings()
    headers = {"X-API-Key": settings.chandra_api_key}

    with open(image_path, "rb") as f:
        resp = requests.post(
            settings.chandra_api_url,
            headers=headers,
            files={"file": (Path(image_path).name, f, "image/jpeg")},
            data={"output_format": "markdown", "mode": "balanced"},
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Chandra API returned {resp.status_code}: {resp.text[:200]}")

    job = resp.json()
    if not job.get("success"):
        raise RuntimeError(f"Submission failed: {job.get('error')}")

    check_url = job["request_check_url"]
    logger.info("Chandra job submitted. Polling %s", check_url)

    for attempt in range(settings.chandra_max_polls):
        time.sleep(settings.chandra_poll_interval)
        result = requests.get(check_url, headers=headers).json()
        status = result.get("status", "")

        if status == "complete":
            return result.get("markdown", "")
        if status == "error":
            raise RuntimeError(f"API processing error: {result.get('error')}")

    raise TimeoutError("Chandra API did not complete within the timeout window.")
