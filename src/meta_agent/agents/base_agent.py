"""
Base Agent
==========

Agents are not hardcoded.
They are instantiated dynamically based on the blueprint.

This module defines the abstract base class that all agents extend.
Every agent has a lifecycle:

  1. receive_context()  — get the shared execution context
  2. execute()          — perform the agent's work
  3. emit_result()      — publish results to the shared state

Agents may invoke tools during execution through the tool registry.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import random
import time
from typing import Any

from meta_agent.schemas.blueprint import AgentConfig
from meta_agent.schemas.state import (
    AgentExecutionRecord,
    AgentMessage,
    MessageRole,
    ToolInvocation,
)

logger = logging.getLogger(__name__)

# Lightweight request queue: limits concurrent LLM calls to prevent bursts
_LLM_REQUEST_SEMAPHORE = asyncio.Semaphore(3)

class TokenBucket:
    """A simple token bucket rate limiter for LLM calls."""
    def __init__(self, capacity: int = 60, refill_rate: float = 1.0):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self.lock:
                now = time.time()
                elapsed = now - self.last_refill
                if elapsed > 0:
                    self.tokens = min(float(self.capacity), self.tokens + elapsed * self.refill_rate)
                    self.last_refill = now
                
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
            # If not enough tokens, wait before checking again
            await asyncio.sleep(0.1)

_LLM_TOKEN_BUCKET = TokenBucket(capacity=60, refill_rate=1.0)


class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents in the Meta-Agent system.

    Provides:
    - Structured lifecycle (receive_context → execute → emit_result)
    - Tool invocation via the tool registry
    - Message emission for inter-agent communication
    - Automatic execution recording for observability
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.agent_id = config.agent_id
        self.name = config.name
        self._tool_registry: Any | None = None
        self._llm_client: Any | None = None
        self._execution_record: AgentExecutionRecord | None = None

    # ── LLM Client ──────────────────────────────────────────────────

    @property
    def llm_client(self) -> Any:
        """Lazily initialise the LLM client for this agent."""
        if self._llm_client is None:
            from langchain_openai import ChatOpenAI
            from meta_agent.config import get_settings
            settings = get_settings()
            kwargs: dict = dict(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=settings.openai_api_key,
            )
            if settings.openai_api_base:
                kwargs["base_url"] = settings.openai_api_base
            self._llm_client = ChatOpenAI(**kwargs)
        return self._llm_client

    @llm_client.setter
    def llm_client(self, client: Any) -> None:
        self._llm_client = client

    # ── Rate-Limited LLM Invocation ─────────────────────────────────

    async def invoke_llm_with_retry(
        self,
        messages: list[dict[str, str]],
        max_retries: int = 3,
        base_delay: float = 2.0,
    ) -> Any:
        """
        Invoke the LLM client with exponential backoff + jitter on rate-limit errors.

        Catches HTTP 429 (rate_limit_exceeded) from the LLM provider and
        retries with exponential backoff plus random jitter, capped at 15s.
        Uses a module-level semaphore to prevent burst requests.

        Args:
            messages: Chat messages to send to the LLM.
            max_retries: Maximum retry attempts (default 3).
            base_delay: Initial delay in seconds (used as base for exp backoff).

        Returns:
            The LLM response object.

        Raises:
            RuntimeError: When all retries are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                await _LLM_TOKEN_BUCKET.acquire()
                async with _LLM_REQUEST_SEMAPHORE:
                    response = await self.llm_client.ainvoke(messages)
                return response
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "429" in error_str
                    or "rate_limit" in error_str
                    or "rate limit" in error_str
                    or "too many requests" in error_str
                )

                if is_rate_limit and attempt < max_retries:
                    delay = min(
                        base_delay * (2 ** attempt) + random.uniform(0, 1),
                        15.0,
                    )
                    logger.warning(
                        "Rate limit hit (attempt %d/%d). "
                        "Retrying in %.1fs: %s",
                        attempt, max_retries, delay, e,
                    )
                    await asyncio.sleep(delay)
                    last_error = e
                    continue
                elif is_rate_limit:
                    # All retries exhausted for rate limit
                    last_error = e
                    logger.error(
                        "Rate limit: all %d retries exhausted. "
                        "Aborting this LLM call.",
                        max_retries,
                    )
                    break
                else:
                    # Non-rate-limit error — re-raise immediately
                    raise

        raise RuntimeError(
            f"LLM call failed after {max_retries} retries due to rate limiting: "
            f"{last_error}"
        )

    # ── Tool Registry ───────────────────────────────────────────────

    def set_tool_registry(self, registry: Any) -> None:
        """Inject the tool registry from the orchestrator."""
        self._tool_registry = registry

    async def invoke_tool(self, tool_name: str, input_data: dict[str, Any]) -> Any:
        """
        Invoke a tool through the registry.

        All tool access goes through this method so it can be:
        - Logged and traced
        - Sandboxed and validated
        - Rate-limited and audited
        """
        if self._tool_registry is None:
            raise RuntimeError(f"Agent {self.agent_id} has no tool registry")

        start = time.time()
        invocation = ToolInvocation(
            agent_id=self.agent_id,
            tool_id=tool_name,
            tool_name=tool_name,
            input_data=input_data,
        )

        try:
            result = await self._tool_registry.invoke(tool_name, input_data)
            invocation.output_data = result
            invocation.duration_ms = (time.time() - start) * 1000
            logger.info(
                "Tool %s invoked by %s (%.0fms)",
                tool_name, self.agent_id, invocation.duration_ms,
            )
        except Exception as e:
            invocation.error = str(e)
            invocation.duration_ms = (time.time() - start) * 1000
            logger.error("Tool %s failed: %s", tool_name, e)
            raise
        finally:
            if self._execution_record:
                self._execution_record.tool_invocations.append(invocation)

        return result

    # ── Message Passing ─────────────────────────────────────────────

    def create_message(
        self,
        content: str,
        recipient_id: str = "broadcast",
        content_type: str = "text",
    ) -> AgentMessage:
        """Create a message from this agent."""
        msg = AgentMessage(
            role=MessageRole.AGENT,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            content=content,
            content_type=content_type,
        )
        if self._execution_record:
            self._execution_record.output_messages.append(msg)
        return msg

    # ── Lifecycle ───────────────────────────────────────────────────

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Full agent lifecycle wrapper.

        Handles: initialisation, execution, error handling, and recording.
        Subclasses implement execute() — not this method.
        """
        self._execution_record = AgentExecutionRecord(
            agent_id=self.agent_id,
            agent_name=self.name,
        )

        logger.info("Agent [%s] %s starting", self.agent_id, self.name)
        start = time.time()

        try:
            result = await self.execute(input_data)
            self._execution_record.status = "completed"
        except Exception as e:
            self._execution_record.status = "failed"
            self._execution_record.error = str(e)
            logger.error("Agent [%s] %s failed: %s", self.agent_id, self.name, e)
            raise
        finally:
            from datetime import datetime, timezone
            self._execution_record.completed_at = datetime.now(timezone.utc)
            elapsed = time.time() - start
            logger.info(
                "Agent [%s] %s finished in %.1fs (status=%s)",
                self.agent_id,
                self.name,
                elapsed,
                self._execution_record.status,
            )

        return result

    @abc.abstractmethod
    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Perform the agent's core work.

        Args:
            input_data: Dict with at least 'task' key and optional
                       'context', 'previous_results', etc.

        Returns:
            Dict with at least 'output' key containing the result.
        """
        ...

    def get_system_prompt(self, task_description: str = "") -> str:
        """Build the system prompt from template and config."""
        from jinja2 import Template
        template_str = self.config.system_prompt_template or self._default_prompt()
        template = Template(template_str)
        return template.render(
            role_description=self.config.role_description,
            task_description=task_description,
            agent_name=self.name,
        )

    def _default_prompt(self) -> str:
        """Fallback system prompt."""
        return (
            f"You are {self.name}.\n\n"
            f"Role: {self.config.role_description}\n\n"
            "Complete the given task to the best of your ability."
        )

    @property
    def execution_record(self) -> AgentExecutionRecord | None:
        return self._execution_record
