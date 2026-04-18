"""
config.py — Application settings loaded from environment / .env file.

Single source of truth for every configurable value.
Uses pydantic-settings so validation happens at startup, not at runtime.
"""

from functools import lru_cache
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# Load .env into os.environ FIRST — LangChain's ChatOpenAI reads
# OPENAI_API_KEY directly from os.environ, not from our Settings object.
load_dotenv()


class Settings(BaseSettings):
    """All configuration is read from environment variables or .env file."""

    # ── API Keys ──────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    e2b_api_key: str = Field(default="", description="E2B sandbox API key")
    chandra_api_key: str = Field(default="", description="Chandra OCR API key")

    # ── Model IDs ─────────────────────────────────────────────────────────
    parser_model: str = Field(default="gpt-5.4-mini-2026-03-17")
    critic_model: str = Field(default="gpt-5.4-2026-03-05")
    max_output_tokens: int = Field(default=12384)

    # ── Chandra OCR ───────────────────────────────────────────────────────
    chandra_api_url: str = Field(default="https://www.datalab.to/api/v1/convert")
    chandra_poll_interval: int = Field(default=2, description="Seconds between poll attempts")
    chandra_max_polls: int = Field(default=60, description="Max polling attempts before timeout")

    # ── Server ────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],
        description="Allowed CORS origins for the frontend",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — parsed once at startup."""
    return Settings()