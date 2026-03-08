"""
FastAPI Application — Meta-Agent API
=====================================

Main entry point for the Meta-Agent platform's HTTP API.
Provides REST endpoints for task submission, status tracking,
blueprint inspection, and real-time updates via WebSocket.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from meta_agent.api.routes import tasks, blueprints, agents, health
from meta_agent.observability.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    configure_logging()
    logger.info("Meta-Agent API starting up")

    # Register default tools
    from meta_agent.tools.registry import ToolRegistry
    registry = ToolRegistry()
    registry.register_defaults()

    yield

    logger.info("Meta-Agent API shutting down")


app = FastAPI(
    title="Meta-Agent API",
    description=(
        "An LLM System that Designs, Builds, and Orchestrates Other LLM Agents. "
        "Submit a task, and the Meta-Agent will design a multi-agent pipeline, "
        "execute it, evaluate the result, and self-improve."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, tags=["Health"])
app.include_router(tasks.router, prefix="/api/v1", tags=["Tasks"])
app.include_router(blueprints.router, prefix="/api/v1", tags=["Blueprints"])
app.include_router(agents.router, prefix="/api/v1", tags=["Agents"])
