"""
Task Planner
============

The planner is the Meta-Agent's first stage of reasoning. It takes
a raw user task and produces:

1. A decomposition into ordered sub-tasks
2. A topology selection (sequential / parallel / hierarchical / DAG)
3. Agent-to-subtask assignments
4. Tool requirements per agent
5. Evaluation criteria

The planner does NOT build agents or graphs. It produces a *plan*
that the BlueprintGenerator converts into a validated Blueprint.

Think of it as:
  User says "Analyse competitor pricing and write a report."
  The planner says:
    Sub-task 1: Search for competitor pricing data  → Research Agent (web_search)
    Sub-task 2: Analyse pricing trends              → Data Analysis Agent (code_executor)
    Sub-task 3: Draft the report                    → Summarization Agent
    Sub-task 4: Review for accuracy                 → Critic Agent
    Topology: SEQUENTIAL (each step depends on the previous)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from meta_agent.config import get_settings
from meta_agent.schemas.blueprint import AgentType, TopologyType, ToolType

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Planner Output Models
# ═══════════════════════════════════════════════════════════════════════════════

class SubTaskSpec(BaseModel):
    """Specification for a single sub-task identified by the planner."""
    index: int = Field(..., ge=0)
    description: str = Field(..., min_length=5)
    agent_type: AgentType
    agent_name: str = Field(default="")
    agent_role: str = Field(default="", min_length=0, max_length=4096)
    required_tools: list[ToolType] = Field(default_factory=list)
    depends_on: list[int] = Field(
        default_factory=list,
        description="Indices of sub-tasks that must complete before this one",
    )
    estimated_tokens: int = Field(default=2000, ge=0)


class TaskPlan(BaseModel):
    """The planner's complete output: a structured plan for the pipeline."""
    original_task: str
    reasoning: str = Field(
        default="",
        description="Chain-of-thought explanation of how the plan was derived",
    )
    sub_tasks: list[SubTaskSpec] = Field(..., min_length=1)
    topology: TopologyType
    evaluation_focus: list[str] = Field(
        default_factory=lambda: ["correctness", "completeness"],
        description="Which evaluation dimensions matter most for this task",
    )
    estimated_total_tokens: int = Field(default=0, ge=0)
    complexity_rating: int = Field(
        default=5, ge=1, le=10,
        description="Estimated task complexity (1=trivial, 10=extremely complex)",
    )
    pipeline_mode: str = Field(
        default="standard",
        description="Pipeline execution mode: lean, standard, or full",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Planner Prompt Template
# ═══════════════════════════════════════════════════════════════════════════════

PLANNER_SYSTEM_PROMPT = """\
You are a Principal AI Systems Architect. Your job is to decompose a user's
task into a structured plan for a multi-agent AI pipeline.

You must output a valid JSON object with this exact schema:

{{
  "reasoning": "<your chain-of-thought>",
  "sub_tasks": [
    {{
      "index": 0,
      "description": "<what this sub-task accomplishes>",
      "agent_type": "<one of: research, planning, coding, data_analysis, verification, summarization, critic, custom>",
      "agent_name": "<human-readable name like 'Senior Research Analyst'>",
      "agent_role": "<detailed role description for the agent's system prompt>",
      "required_tools": ["<tool_type>", ...],
      "depends_on": [<indices of prerequisite sub-tasks>],
      "estimated_tokens": <estimated tokens for this step>
    }}
  ],
  "topology": "<one of: sequential, parallel, hierarchical, dag>",
  "pipeline_mode": "<one of: lean, standard, full>",
  "evaluation_focus": ["correctness", "completeness", ...],
  "estimated_total_tokens": <total token budget>,
  "complexity_rating": <1-10>
}}

Rules:
- For simple linear tasks, use "sequential" topology.
- For tasks with independent parallel workstreams, use "parallel".
- For tasks with a manager coordinating specialists, use "hierarchical".
- For complex dependency graphs, use "dag".
- Always include at least one verification or critic step at the end.
- Be specific about agent roles and tool requirements.
- Estimate token usage realistically.
- Set pipeline_mode: "lean" (Research+Code+Verify) for complexity <= 5 and a single data target; "standard" (Research+DataAnalysis+Code+Verify) for complexity 6-7; "full" (all agents allowed) for complexity >= 8 or parallel.

Available tool types: web_search, code_executor, database_query, vector_retrieval, api_caller, file_reader

Available agent types: research, planning, coding, data_analysis, verification, summarization, critic, custom
"""

PLANNER_USER_PROMPT = """\
Decompose the following task into a multi-agent pipeline plan:

TASK:
{task_description}

Respond with ONLY the JSON plan. No markdown, no explanation outside the JSON.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# TaskPlanner Class
# ═══════════════════════════════════════════════════════════════════════════════

class TaskPlanner:
    """
    Decomposes complex tasks into sub-tasks and selects a topology.

    The planner uses an LLM to reason about the task structure, then
    returns a validated TaskPlan that the BlueprintGenerator consumes.
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.meta_agent_model
        self.temperature = temperature if temperature is not None else settings.meta_agent_temperature
        self._llm_client = llm_client  # Injected or lazily created

    @property
    def llm_client(self) -> Any:
        """Lazy-init the LLM client."""
        if self._llm_client is None:
            from langchain_openai import ChatOpenAI
            settings = get_settings()
            kwargs: dict = dict(
                model=self.model,
                temperature=self.temperature,
                max_tokens=settings.meta_agent_max_tokens,
                api_key=settings.openai_api_key,
            )
            # Route to an OpenAI-compatible provider (e.g. Groq) when configured
            if settings.openai_api_base:
                kwargs["base_url"] = settings.openai_api_base
            self._llm_client = ChatOpenAI(**kwargs)
        return self._llm_client

    async def plan(self, task_description: str) -> TaskPlan:
        """
        Decompose a task into a structured plan.

        Args:
            task_description: The raw user task text.

        Returns:
            A validated TaskPlan object.
        """
        logger.info("Planning task: %s", task_description[:120])

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": PLANNER_USER_PROMPT.format(
                task_description=task_description,
            )},
        ]

        response = await self.llm_client.ainvoke(messages)
        raw_content = response.content

        # Parse and validate
        plan_data = self._parse_response(raw_content)
        plan = TaskPlan(
            original_task=task_description,
            **plan_data,
        )

        logger.info(
            "Plan created: %d sub-tasks, topology=%s, complexity=%d",
            len(plan.sub_tasks),
            plan.topology.value,
            plan.complexity_rating,
        )
        return plan

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Extract JSON from the LLM response, handling common issues."""
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (code fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse planner response: %s", e)
            raise ValueError(
                f"Meta-Agent planner produced invalid JSON: {e}\n"
                f"Raw response: {raw[:500]}"
            ) from e

    def _infer_topology(self, sub_tasks: list[SubTaskSpec]) -> TopologyType:
        """
        Heuristic topology inference as a fallback.

        - If no dependencies: PARALLEL
        - If all linear (each depends on previous): SEQUENTIAL
        - If one node depends on multiple: DAG
        - Otherwise: DAG
        """
        if not sub_tasks:
            return TopologyType.SEQUENTIAL

        has_deps = any(t.depends_on for t in sub_tasks)
        if not has_deps:
            return TopologyType.PARALLEL

        # Check if purely linear
        is_linear = all(
            t.depends_on == [t.index - 1] if t.index > 0 else not t.depends_on
            for t in sub_tasks
        )
        if is_linear:
            return TopologyType.SEQUENTIAL

        return TopologyType.DAG
