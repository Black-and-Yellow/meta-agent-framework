"""
Blueprint Schema Definitions
=============================

Blueprints define how intelligence is assembled.
If this schema is wrong, the entire AI pipeline collapses.
Therefore we validate aggressively and fail loudly.

This module contains the strict Pydantic v2 models that describe every aspect
of a dynamically-generated multi-agent system:

- What agents exist and what they do
- What tools each agent can use
- How agents are connected (edges)
- How the result is evaluated
- The overall execution topology

The Meta-Agent produces a Blueprint; the Orchestrator consumes it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class TopologyType(str, Enum):
    """
    How agents are wired together.

    SEQUENTIAL  — A → B → C (pipeline)
    PARALLEL    — A, B, C run concurrently, then fan-in
    HIERARCHICAL — Manager delegates to workers, aggregates results
    DAG         — Arbitrary directed acyclic graph with conditional edges
    """
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    DAG = "dag"


class AgentType(str, Enum):
    """Registry of all known specialised agent archetypes."""
    RESEARCH = "research"
    PLANNING = "planning"
    CODING = "coding"
    DATA_ANALYSIS = "data_analysis"
    VERIFICATION = "verification"
    SUMMARIZATION = "summarization"
    CRITIC = "critic"
    CUSTOM = "custom"


class ToolType(str, Enum):
    """Registry of all tool categories available to agents."""
    WEB_SEARCH = "web_search"
    CODE_EXECUTOR = "code_executor"
    DATABASE_QUERY = "database_query"
    VECTOR_RETRIEVAL = "vector_retrieval"
    API_CALLER = "api_caller"
    FILE_READER = "file_reader"
    CUSTOM = "custom"


class EdgeConditionType(str, Enum):
    """
    When is this edge traversed?

    ALWAYS      — unconditional
    ON_SUCCESS  — only if the source node succeeds
    ON_FAILURE  — only if the source node fails
    CONDITIONAL — evaluated by a Python expression or LLM decision
    SCORE_ABOVE — only if evaluation score exceeds a threshold
    SCORE_BELOW — only if evaluation score is below a threshold
    """
    ALWAYS = "always"
    ON_SUCCESS = "on_success"
    ON_FAILURE = "on_failure"
    CONDITIONAL = "conditional"
    SCORE_ABOVE = "score_above"
    SCORE_BELOW = "score_below"


class EvaluationStrategy(str, Enum):
    """How to evaluate the output of a node or the overall pipeline."""
    LLM_JUDGE = "llm_judge"
    RULE_BASED = "rule_based"
    SIMILARITY = "similarity"
    COMPOSITE = "composite"
    HUMAN_IN_LOOP = "human_in_loop"


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class ToolConfig(BaseModel):
    """
    Declares a single tool an agent may invoke.

    Tools are the agent's hands — the only way to touch the outside world.
    Each tool has a type, a permission scope, and optional configuration.
    """

    tool_id: str = Field(
        default_factory=lambda: f"tool_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this tool instance",
    )
    tool_type: ToolType = Field(
        ...,
        description="Category of tool (web_search, code_executor, etc.)",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Human-readable tool name",
    )
    description: str = Field(
        default="",
        max_length=1024,
        description="What this tool does, shown to the agent in its prompt",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific configuration (API keys, endpoints, limits)",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=600,
        description="Hard timeout for a single invocation",
    )
    max_retries: int = Field(default=2, ge=0, le=5)
    requires_approval: bool = Field(
        default=False,
        description="If True, tool invocations require human-in-the-loop approval",
    )
    permission_scope: Literal["read", "write", "execute"] = Field(
        default="read",
        description="Security scope for this tool",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class AgentConfig(BaseModel):
    """
    Declares a single agent node in the execution graph.

    Agents are not hardcoded. They are instantiated dynamically based on
    this configuration. The factory reads the AgentConfig and produces
    a living agent with the right model, prompt, tools, and behaviour.
    """

    agent_id: str = Field(
        default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}",
        description="Unique node identifier within the graph",
    )
    agent_type: AgentType = Field(
        ...,
        description="Archetype that determines base behaviour and prompt",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Human-readable name (e.g. 'Senior Research Analyst')",
    )
    role_description: str = Field(
        ...,
        min_length=10,
        max_length=4096,
        description="Detailed role description injected into the agent's system prompt",
    )
    model: str = Field(
        default_factory=lambda: __import__('meta_agent.config', fromlist=['get_settings']).get_settings().meta_agent_model,
        description="LLM model identifier",
    )
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=64, le=128_000)
    tools: list[ToolConfig] = Field(
        default_factory=list,
        description="Tools available to this agent",
    )
    system_prompt_template: str = Field(
        default="",
        max_length=16384,
        description="Jinja2 template for the system prompt; "
                    "if empty, a default template is used based on agent_type",
    )
    retry_policy: RetryPolicy | None = Field(
        default=None,
        description="How to handle failures for this agent",
    )
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing expected input shape",
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing expected output shape",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs for tagging and filtering",
    )

    @field_validator("tools")
    @classmethod
    def unique_tool_ids(cls, tools: list[ToolConfig]) -> list[ToolConfig]:
        ids = [t.tool_id for t in tools]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate tool_id values in agent tool list")
        return tools


class RetryPolicy(BaseModel):
    """How many times and how to retry a failed agent invocation."""
    max_retries: int = Field(default=2, ge=0, le=10)
    backoff_factor: float = Field(default=1.0, ge=0.1, le=60.0)
    retry_on_exceptions: list[str] = Field(
        default_factory=lambda: ["TimeoutError", "RateLimitError"],
        description="Exception class names that trigger retries",
    )


# Fix forward reference — RetryPolicy is defined after AgentConfig uses it
AgentConfig.model_rebuild()


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Configuration (wiring between agents)
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeConfig(BaseModel):
    """
    A directed edge in the execution graph.

    Edges connect agents and carry conditions. The orchestrator evaluates
    these conditions at runtime to decide which path to follow.
    """

    edge_id: str = Field(
        default_factory=lambda: f"edge_{uuid.uuid4().hex[:8]}",
    )
    source_agent_id: str = Field(
        ...,
        description="ID of the agent that produces output",
    )
    target_agent_id: str = Field(
        ...,
        description="ID of the agent that consumes the output",
    )
    condition_type: EdgeConditionType = Field(
        default=EdgeConditionType.ALWAYS,
    )
    condition_value: str | float | None = Field(
        default=None,
        description="Threshold for score-based conditions, "
                    "or expression string for CONDITIONAL type",
    )
    transform_template: str = Field(
        default="",
        max_length=4096,
        description="Optional Jinja2 template to transform the source output "
                    "before passing to target",
    )
    priority: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Higher priority edges are evaluated first",
    )

    @model_validator(mode="after")
    def validate_condition_value(self) -> EdgeConfig:
        """Ensure score-based conditions have a numeric threshold."""
        if self.condition_type in (
            EdgeConditionType.SCORE_ABOVE,
            EdgeConditionType.SCORE_BELOW,
        ):
            if not isinstance(self.condition_value, (int, float)):
                raise ValueError(
                    f"condition_type={self.condition_type.value} requires a "
                    f"numeric condition_value, got {type(self.condition_value)}"
                )
        if self.condition_type == EdgeConditionType.CONDITIONAL:
            if not self.condition_value or not isinstance(self.condition_value, str):
                raise ValueError(
                    "CONDITIONAL edges require a non-empty string condition_value"
                )
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationDimension(BaseModel):
    """A single axis along which output quality is measured."""
    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(default="", max_length=512)
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable score on this dimension",
    )


class EvaluationConfig(BaseModel):
    """
    How to judge the quality of a pipeline's output.

    The system must judge its own work. This configuration tells the
    evaluator what to measure and how to aggregate scores.
    """

    strategy: EvaluationStrategy = Field(
        default=EvaluationStrategy.LLM_JUDGE,
    )
    judge_model: str = Field(
        default_factory=lambda: __import__('meta_agent.config', fromlist=['get_settings']).get_settings().meta_agent_model,
        description="Model used for LLM-as-judge evaluation",
    )
    dimensions: list[EvaluationDimension] = Field(
        default_factory=lambda: [
            EvaluationDimension(
                name="correctness",
                description="Is the answer factually correct?",
                weight=3.0,
                threshold=0.6,
            ),
            EvaluationDimension(
                name="completeness",
                description="Does the answer fully address the task?",
                weight=2.0,
                threshold=0.5,
            ),
            EvaluationDimension(
                name="coherence",
                description="Is the answer well-structured and clear?",
                weight=1.0,
                threshold=0.4,
            ),
            EvaluationDimension(
                name="task_completion",
                description="Did the output fully solve the original task?",
                weight=2.0,
                threshold=0.5,
            ),
        ],
    )
    overall_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Weighted-average score below which a repair loop is triggered",
    )
    max_repair_iterations: int = Field(default=3, ge=0, le=10)
    evaluation_prompt_template: str = Field(
        default="",
        max_length=8192,
        description="Custom Jinja2 prompt for the LLM judge",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Execution Graph (the entire wiring)
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutionGraph(BaseModel):
    """
    The complete execution graph: agents as nodes, edges as wiring.

    This is the compiled form of the Meta-Agent's design. The orchestrator
    reads this and brings it to life.
    """

    topology: TopologyType = Field(
        ...,
        description="Overall graph topology strategy",
    )
    entry_point: str = Field(
        ...,
        description="agent_id of the first node to execute",
    )
    agents: list[AgentConfig] = Field(
        ...,
        min_length=1,
        description="All agent nodes in the graph",
    )
    edges: list[EdgeConfig] = Field(
        default_factory=list,
        description="All directed edges between agents",
    )
    parallel_groups: list[list[str]] = Field(
        default_factory=list,
        description="Groups of agent_ids that may run concurrently",
    )

    @field_validator("agents")
    @classmethod
    def unique_agent_ids(cls, agents: list[AgentConfig]) -> list[AgentConfig]:
        ids = [a.agent_id for a in agents]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate agent_id values in execution graph")
        return agents

    @model_validator(mode="after")
    def validate_edge_references(self) -> ExecutionGraph:
        """Every edge must reference agents that exist in the graph."""
        agent_ids = {a.agent_id for a in self.agents}
        if self.entry_point not in agent_ids:
            raise ValueError(
                f"entry_point '{self.entry_point}' is not among declared agent_ids"
            )
        for edge in self.edges:
            if edge.source_agent_id not in agent_ids:
                raise ValueError(
                    f"Edge {edge.edge_id}: source '{edge.source_agent_id}' not found"
                )
            if edge.target_agent_id not in agent_ids:
                raise ValueError(
                    f"Edge {edge.edge_id}: target '{edge.target_agent_id}' not found"
                )
        for group in self.parallel_groups:
            for aid in group:
                if aid not in agent_ids:
                    raise ValueError(
                        f"Parallel group references unknown agent '{aid}'"
                    )
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# Blueprint — the top-level document the Meta-Agent produces
# ═══════════════════════════════════════════════════════════════════════════════

class Blueprint(BaseModel):
    """
    The complete blueprint for a dynamically-generated multi-agent system.

    The Meta-Agent imagines an architecture.
    This document IS that architecture, serialised to JSON.
    The Orchestration layer reads it and makes it executable.

    A Blueprint is:
      1. Self-describing  — contains enough information to reproduce itself
      2. Validated         — if it parses, it is structurally sound
      3. Versioned         — every revision gets a new revision number
      4. Immutable at rest — once produced, a blueprint is a snapshot
    """

    blueprint_id: str = Field(
        default_factory=lambda: f"bp_{uuid.uuid4().hex[:12]}",
    )
    version: str = Field(default="1.0.0")
    revision: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Task Context ────────────────────────────────────────────────
    task_description: str = Field(
        ...,
        min_length=10,
        max_length=32_768,
        description="The original user task that spawned this blueprint",
    )
    task_decomposition: list[str] = Field(
        default_factory=list,
        description="Ordered list of sub-tasks derived from the main task",
    )
    reasoning_trace: str = Field(
        default="",
        max_length=65_536,
        description="The Meta-Agent's chain-of-thought reasoning that led "
                    "to this design",
    )

    # ── Architecture ────────────────────────────────────────────────
    execution_graph: ExecutionGraph = Field(
        ...,
        description="The full agent graph",
    )

    # ── Evaluation ──────────────────────────────────────────────────
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
    )

    # ── Metadata ────────────────────────────────────────────────────
    estimated_token_budget: int = Field(
        default=0,
        ge=0,
        description="Meta-Agent's estimate of total tokens this pipeline will use",
    )
    estimated_latency_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Meta-Agent's estimate of wall-clock time",
    )
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def next_revision(self) -> Blueprint:
        """Create a copy with an incremented revision number."""
        data = self.model_dump()
        data["revision"] = self.revision + 1
        data["created_at"] = datetime.now(timezone.utc)
        return Blueprint.model_validate(data)
