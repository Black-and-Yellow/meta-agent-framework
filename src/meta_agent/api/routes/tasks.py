"""Task routes — submit, track, and retrieve task results."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from meta_agent.core.meta_agent import MetaAgent

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory task store (use Redis/DB in production)
_task_store: dict[str, dict[str, Any]] = {}


class TaskSubmission(BaseModel):
    task: str = Field(..., min_length=10, max_length=32_768, description="The task description")
    model: str = Field(
        default_factory=lambda: __import__('meta_agent.config', fromlist=['get_settings']).get_settings().meta_agent_model,
        description="LLM model to use",
    )
    max_repair_iterations: int = Field(default=3, ge=1, le=10)


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskResult(BaseModel):
    task_id: str
    status: str
    result: Any = None
    blueprint: Any = None
    evaluations: list[Any] = []
    elapsed_seconds: float = 0.0
    error: str | None = None


async def _run_task(task_id: str, submission: TaskSubmission) -> None:
    """Background task runner."""
    try:
        _task_store[task_id]["status"] = "running"
        meta = MetaAgent()
        result = await meta.solve(submission.task)
        _task_store[task_id].update({
            "status": "completed",
            "result": result.get("result"),
            "blueprint": result.get("blueprint"),
            "evaluations": result.get("evaluations", []),
            "elapsed_seconds": result.get("elapsed_seconds", 0),
        })
    except Exception as e:
        logger.error("Task %s failed: %s", task_id, e)
        _task_store[task_id].update({
            "status": "failed",
            "error": str(e),
        })


@router.post("/tasks", response_model=TaskResponse, status_code=202)
async def submit_task(
    submission: TaskSubmission,
    background_tasks: BackgroundTasks,
) -> TaskResponse:
    """
    Submit a new task for the Meta-Agent to solve.

    The task runs asynchronously in the background. Use GET /tasks/{task_id}
    to poll for status and results.
    """
    task_id = f"task_{uuid.uuid4().hex[:12]}"
    _task_store[task_id] = {
        "status": "queued",
        "task": submission.task,
    }
    background_tasks.add_task(_run_task, task_id, submission)
    logger.info("Task %s submitted: %s", task_id, submission.task[:80])

    return TaskResponse(
        task_id=task_id,
        status="queued",
        message="Task submitted. Poll GET /api/v1/tasks/{task_id} for results.",
    )


@router.get("/tasks/{task_id}", response_model=TaskResult)
async def get_task(task_id: str) -> TaskResult:
    """Get the status and result of a submitted task."""
    if task_id not in _task_store:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    data = _task_store[task_id]
    return TaskResult(
        task_id=task_id,
        status=data.get("status", "unknown"),
        result=data.get("result"),
        blueprint=data.get("blueprint"),
        evaluations=data.get("evaluations", []),
        elapsed_seconds=data.get("elapsed_seconds", 0),
        error=data.get("error"),
    )


@router.get("/tasks")
async def list_tasks() -> list[dict[str, Any]]:
    """List all submitted tasks."""
    return [
        {"task_id": tid, "status": data["status"], "task": data.get("task", "")[:100]}
        for tid, data in _task_store.items()
    ]
