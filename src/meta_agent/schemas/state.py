"""
Runtime State Models
====================

Models for execution state, inter-agent messaging, memory entries,
and evaluation results. These are the transient objects that flow
through the system while a blueprint is being executed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    BUILDING = "building"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    REPAIRING = "repairing"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    AGENT = "agent"
    TOOL = "tool"
    ORCHESTRATOR = "orchestrator"


class RepairAction(str, Enum):
    """Decision output from the repair loop."""
    ACCEPT = "accept"
    REFINE = "refine"
    REBUILD = "rebuild"


# ═══════════════════════════════════════════════════════════════════════════════
# Inter-Agent Messages
# ═══════════════════════════════════════════════════════════════════════════════

class AgentMessage(BaseModel):
    """
    A single message exchanged between agents, tools, or the orchestrator.

    Messages are the blood of the system — they carry data between nodes,
    record tool outputs, and accumulate the shared understanding.
    """

    message_id: str = Field(
        default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    role: MessageRole = Field(...)
    sender_id: str = Field(
        ...,
        description="agent_id, tool_id, or 'orchestrator'",
    )
    recipient_id: str = Field(
        default="broadcast",
        description="Target agent_id, or 'broadcast' for all",
    )
    content: str = Field(..., max_length=131_072)
    content_type: str = Field(
        default="text",
        description="MIME-like type: text, json, code, error",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_message_id: str | None = Field(
        default=None,
        description="For threaded conversations: the message this replies to",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Invocation Records
# ═══════════════════════════════════════════════════════════════════════════════

class ToolInvocation(BaseModel):
    """Record of a single tool call made by an agent."""

    invocation_id: str = Field(
        default_factory=lambda: f"inv_{uuid.uuid4().hex[:8]}",
    )
    agent_id: str = Field(...)
    tool_id: str = Field(...)
    tool_name: str = Field(...)
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: Any = Field(default=None)
    error: str | None = Field(default=None)
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    completed_at: datetime | None = Field(default=None)
    duration_ms: float | None = Field(default=None, ge=0)
    token_usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token counts: prompt_tokens, completion_tokens, total_tokens",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Execution Record
# ═══════════════════════════════════════════════════════════════════════════════

class AgentExecutionRecord(BaseModel):
    """Complete record of a single agent's execution within the pipeline."""

    agent_id: str
    agent_name: str
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    completed_at: datetime | None = None
    status: str = Field(default="running")
    input_messages: list[AgentMessage] = Field(default_factory=list)
    output_messages: list[AgentMessage] = Field(default_factory=list)
    tool_invocations: list[ToolInvocation] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(
        default_factory=list,
        description="Chain-of-thought reasoning captured during execution",
    )
    token_usage: dict[str, int] = Field(default_factory=dict)
    error: str | None = None
    retry_count: int = Field(default=0, ge=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Result
# ═══════════════════════════════════════════════════════════════════════════════

class DimensionScore(BaseModel):
    """Score on a single evaluation dimension."""
    name: str
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(default="")
    passed: bool = Field(default=False)


class EvaluationResult(BaseModel):
    """
    The verdict of the evaluation system.

    Solving a task is not enough. The system must judge its own work.
    This model captures the multi-dimensional assessment and the
    recommended repair action.
    """

    evaluation_id: str = Field(
        default_factory=lambda: f"eval_{uuid.uuid4().hex[:8]}",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool = Field(default=False)
    recommended_action: RepairAction = Field(default=RepairAction.ACCEPT)
    reasoning: str = Field(
        default="",
        description="The evaluator's explanation of the scores",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Specific improvement suggestions if repair is needed",
    )
    iteration: int = Field(default=1, ge=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Entry (for long-term vector store)
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryEntry(BaseModel):
    """
    A single entry in the long-term vector memory.

    Long-term memory remembers past successes and failures.
    Each entry is a semantic chunk that can be retrieved by similarity.
    """

    entry_id: str = Field(
        default_factory=lambda: f"mem_{uuid.uuid4().hex[:8]}",
    )
    content: str = Field(..., max_length=32_768)
    entry_type: str = Field(
        default="observation",
        description="Type: observation, result, blueprint, error_log, lesson",
    )
    source_task_id: str = Field(default="")
    source_agent_id: str = Field(default="")
    embedding: list[float] | None = Field(
        default=None,
        description="Pre-computed embedding vector (optional)",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Populated at retrieval time",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Execution Context — the shared state for an entire pipeline run
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutionContext(BaseModel):
    """
    The full runtime state of a single pipeline execution.

    Short-term memory holds the current conversation — this object IS
    that short-term memory. Every agent reads from and writes to it.
    """

    execution_id: str = Field(
        default_factory=lambda: f"exec_{uuid.uuid4().hex[:12]}",
    )
    task_id: str = Field(...)
    blueprint_id: str = Field(...)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    # ── Data Flow ───────────────────────────────────────────────────
    original_input: str = Field(...)
    intermediate_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyed by agent_id: results from each completed node",
    )
    final_output: Any = Field(default=None)

    # ── Agent Records ───────────────────────────────────────────────
    agent_records: list[AgentExecutionRecord] = Field(default_factory=list)
    messages: list[AgentMessage] = Field(default_factory=list)

    # ── Evaluation ──────────────────────────────────────────────────
    evaluations: list[EvaluationResult] = Field(default_factory=list)
    current_iteration: int = Field(default=1, ge=1)

    # ── Resource Tracking ───────────────────────────────────────────
    total_tokens_used: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    wall_clock_seconds: float = Field(default=0.0, ge=0.0)

    # ── Error Log ───────────────────────────────────────────────────
    errors: list[str] = Field(default_factory=list)

    # ── Configuration Flags ─────────────────────────────────────────
    enable_summarizer: bool = Field(default=True)

    def add_agent_result(self, agent_id: str, result: Any) -> None:
        """Store an intermediate result from a completed agent."""
        self.intermediate_results[agent_id] = result
        self.updated_at = datetime.now(timezone.utc)

    def append_message(self, message: AgentMessage) -> None:
        """Append a message to the shared conversation log."""
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)

    def record_error(self, error: str) -> None:
        """Log an error."""
        self.errors.append(f"[{datetime.now(timezone.utc).isoformat()}] {error}")
        self.updated_at = datetime.now(timezone.utc)
