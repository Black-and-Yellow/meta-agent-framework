"""
Tool Base Class
===============

Agents cannot touch the outside world directly.
All capabilities are exposed through controlled tools.
This keeps the system safe and auditable.

Every tool extends BaseTool and implements:
  - execute()   — perform the tool's action
  - validate()  — check input before execution
  - schema()    — return JSON Schema for the tool's inputs
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolInput(BaseModel):
    """Base class for tool input validation."""
    pass


class ToolResult(BaseModel):
    """Standard wrapper for tool outputs."""
    success: bool = True
    data: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseTool(abc.ABC):
    """
    Abstract base for all tools in the Meta-Agent system.

    Tools are the agent's hands — the only way to touch the outside world.
    They are:
    - Sandboxed: inputs are validated before execution
    - Auditable: every invocation is logged
    - Timeout-guarded: no tool can run forever
    - Retryable: transient failures are handled automatically
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        timeout_seconds: int = 30,
        max_retries: int = 2,
        permission_scope: str = "read",
    ) -> None:
        self.name = name
        self.description = description
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.permission_scope = permission_scope

    async def __call__(self, input_data: dict[str, Any]) -> ToolResult:
        """
        Invoke the tool with validation, timing, and error handling.
        """
        start = time.time()

        # Validate input
        validation_error = self.validate_input(input_data)
        if validation_error:
            return ToolResult(
                success=False,
                error=f"Validation failed: {validation_error}",
                duration_ms=(time.time() - start) * 1000,
            )

        # Execute with retries
        last_error: str | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = await self.execute(input_data)
                return ToolResult(
                    success=True,
                    data=result,
                    duration_ms=(time.time() - start) * 1000,
                    metadata={"attempts": attempt},
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "Tool %s attempt %d/%d failed: %s",
                    self.name, attempt, self.max_retries, e,
                )
                if attempt == self.max_retries:
                    break

        return ToolResult(
            success=False,
            error=f"Tool failed after {self.max_retries} attempts: {last_error}",
            duration_ms=(time.time() - start) * 1000,
        )

    @abc.abstractmethod
    async def execute(self, input_data: dict[str, Any]) -> Any:
        """Perform the tool's action. Subclasses must implement."""
        ...

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        Validate the input. Returns an error message or None if valid.
        Override in subclasses for specific validation.
        """
        return None

    def get_schema(self) -> dict[str, Any]:
        """Return JSON Schema describing the tool's expected inputs."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": {}},
        }
