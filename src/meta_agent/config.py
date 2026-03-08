"""
Application-wide configuration using pydantic-settings.

Loads from .env file or environment variables. Every layer of the system
imports settings from here — there is exactly one source of truth.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised configuration for the Meta-Agent platform."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI / Groq / compatible API key")
    openai_api_base: str = Field(
        default="",
        description="Override base URL for OpenAI-compatible providers (e.g. Groq)",
    )
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    # ── Meta-Agent ──────────────────────────────────────────────────
    meta_agent_model: str = Field(default="gpt-4o", description="Model for the Meta-Agent")
    meta_agent_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    meta_agent_max_tokens: int = Field(default=4096, ge=256, le=128_000)
    max_repair_iterations: int = Field(default=2, ge=1, le=10)
    evaluation_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum score to accept a result without repair",
    )

    # ── Redis (Short-Term Memory) ───────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ── ChromaDB (Long-Term Memory) ─────────────────────────────────
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8000)
    chroma_collection: str = Field(default="meta_agent_memory")

    # ── API Server ──────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080)
    api_workers: int = Field(default=4, ge=1)
    api_key: str = Field(default="")

    # ── Database (for database_query tool) ──────────────────────────
    database_path: str = Field(
        default="",
        description="Path to SQLite database file for the database_query tool",
    )

    # ── Observability ───────────────────────────────────────────────
    otel_exporter_otlp_endpoint: str = Field(default="http://localhost:4317")
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="meta-agent")

    # ── Security ────────────────────────────────────────────────────
    sandbox_timeout_seconds: int = Field(default=30, ge=5, le=300)
    max_tool_retries: int = Field(default=3, ge=0, le=10)
    enable_code_execution: bool = Field(default=False)

    # ── Logging ─────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "text"] = "json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
