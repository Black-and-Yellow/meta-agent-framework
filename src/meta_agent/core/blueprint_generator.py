"""
Blueprint Generator
===================

Converts a TaskPlan (from the planner) into a fully validated Blueprint.

The planner says WHAT agents are needed and HOW they relate.
The generator converts that into the precise, validated schema that
the orchestration engine can execute.

It resolves:
  - Agent configurations with proper system prompts
  - Tool assignments with security scopes
  - Edge wiring with conditional routing
  - Evaluation criteria based on task characteristics
"""

from __future__ import annotations

import logging
from typing import Any

from meta_agent.core.planner import SubTaskSpec, TaskPlan
from meta_agent.schemas.blueprint import (
    AgentConfig,
    Blueprint,
    EdgeConditionType,
    EdgeConfig,
    EvaluationConfig,
    EvaluationDimension,
    EvaluationStrategy,
    ExecutionGraph,
    RetryPolicy,
    ToolConfig,
    TopologyType,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Default System Prompt Templates (by agent type)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PROMPTS: dict[str, str] = {
    "research": (
        "You are a senior research analyst. Your job is to find accurate, "
        "up-to-date information on the given topic. Use your search tools "
        "to gather data from multiple sources. Cite your sources. "
        "Be thorough but concise.\n\n"
        "STRICT OUTPUT RULES:\n"
        "- Output ONLY factual findings: location, temperature value, unit, and source URL.\n"
        "- Do NOT write Python code, conversion scripts, functions, or code examples of any kind.\n"
        "- Do NOT suggest how to convert the temperature. That is handled by a separate agent.\n"
        "- Do NOT use placeholder or default values. If you cannot find the temperature, say so explicitly.\n"
        "- Your output must end with: EXTRACTED: {value}°C / {value}°F from {source_title}\n\n"
        "ROLE: {{ role_description }}\n\n"
        "TASK: {{ task_description }}"
    ),
    "planning": (
        "You are a strategic planner. Analyse the information provided and "
        "create a detailed, actionable plan. Identify risks and mitigation "
        "strategies. Structure your plan with clear steps.\n\n"
        "ROLE: {{ role_description }}\n\n"
        "TASK: {{ task_description }}"
    ),
    "coding": (
        "You are an expert software engineer. Write clean, well-documented "
        "code that solves the given problem. Include error handling, type "
        "hints, and tests where appropriate. Use the code execution sandbox "
        "to verify your solution works.\n\n"
        "ROLE: {{ role_description }}\n\n"
        "TASK: {{ task_description }}"
    ),
    "data_analysis": (
        "You are a senior data analyst. Analyse the provided data using "
        "statistical methods and produce clear visualisations and insights. "
        "Use the code executor for computations. Present findings clearly.\n\n"
        "ROLE: {{ role_description }}\n\n"
        "TASK: {{ task_description }}"
    ),
    "verification": (
        "You are a verification specialist. Your job is to check the accuracy "
        "and correctness of the work produced by other agents. Test claims, "
        "validate code, and identify errors or inconsistencies. Be rigorous.\n\n"
        "ROLE: {{ role_description }}\n\n"
        "TASK: {{ task_description }}"
    ),
    "summarization": (
        "You are a senior technical writer. Synthesise the information from "
        "multiple sources into a clear, well-structured document. Use proper "
        "formatting, include key findings, and ensure readability.\n\n"
        "ROLE: {{ role_description }}\n\n"
        "TASK: {{ task_description }}"
    ),
    "critic": (
        "You are a critical reviewer. Evaluate the quality, accuracy, and "
        "completeness of the work. Identify weaknesses, logical flaws, and "
        "areas for improvement. Be constructive but thorough.\n\n"
        "ROLE: {{ role_description }}\n\n"
        "TASK: {{ task_description }}"
    ),
    "custom": (
        "You are an AI assistant with a specific role.\n\n"
        "ROLE: {{ role_description }}\n\n"
        "TASK: {{ task_description }}"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Configuration Defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_TOOL_CONFIGS: dict[str, dict[str, Any]] = {
    "web_search": {
        "name": "Web Search",
        "description": "Search the internet for current information",
        "permission_scope": "read",
        "timeout_seconds": 15,
    },
    "code_executor": {
        "name": "Code Execution Sandbox",
        "description": "Execute Python code in a sandboxed environment",
        "permission_scope": "execute",
        "timeout_seconds": 60,
    },
    "database_query": {
        "name": "Database Query",
        "description": "Execute read-only SQL queries against the database",
        "permission_scope": "read",
        "timeout_seconds": 30,
    },
    "vector_retrieval": {
        "name": "Vector Memory Search",
        "description": "Search long-term vector memory for relevant past knowledge",
        "permission_scope": "read",
        "timeout_seconds": 10,
    },
    "api_caller": {
        "name": "External API Caller",
        "description": "Make HTTP requests to external APIs",
        "permission_scope": "read",
        "timeout_seconds": 30,
    },
    "file_reader": {
        "name": "File Reader",
        "description": "Read files from the allowed file system",
        "permission_scope": "read",
        "timeout_seconds": 10,
    },
}


class BlueprintGenerator:
    """
    Converts a TaskPlan into a validated Blueprint.

    This is a deterministic transformation — given the same plan, it always
    produces the same blueprint. The intelligence is in the planner; the
    generator is purely mechanical.
    """

    def generate(self, plan: TaskPlan) -> Blueprint:
        """
        Generate a complete Blueprint from a TaskPlan.

        Steps:
        1. Create AgentConfig for each sub-task
        2. Create ToolConfig instances for required tools
        3. Wire edges based on topology and dependencies
        4. Build evaluation config from the plan's focus areas
        5. Assemble into an ExecutionGraph and Blueprint
        """
        logger.info("Generating blueprint for %d sub-tasks", len(plan.sub_tasks))

        # Step 0: Enforce pipeline mode constraints
        ALLOWED_AGENTS = {
            "lean":     {"research", "coding", "verification", "custom", "critic", "summarization"},
            "standard": {"research", "data_analysis", "coding", "verification", "custom", "critic", "summarization"},
            "full":     {"research", "planning", "data_analysis", "coding", "verification", "critic", "summarization", "custom"}
        }
        
        mode = getattr(plan, "pipeline_mode", "full")
        allowed = ALLOWED_AGENTS.get(mode, ALLOWED_AGENTS["full"])
        
        valid_sub_tasks = []
        for spec in plan.sub_tasks:
            agent_type_str = spec.agent_type.value if hasattr(spec.agent_type, "value") else str(spec.agent_type)
            if agent_type_str in allowed:
                valid_sub_tasks.append(spec)
            else:
                logger.warning(
                    "Blueprint: dropped agent %s — not permitted in %s mode",
                    agent_type_str,
                    mode,
                )
        # Step 0.5: Wire Extractor and Critic Agents automatically
        if mode in ["lean", "standard"]:
            new_sub_tasks = []
            max_idx = max((s.index for s in valid_sub_tasks), default=0)
            from meta_agent.schemas.blueprint import AgentType as _AT
            
            for spec in valid_sub_tasks:
                new_sub_tasks.append(spec)
                if spec.agent_type.value == "research":
                    max_idx += 1
                    extractor_spec = type(spec)(
                        index=max_idx,
                        description="Extract structured data from research",
                        agent_type=_AT.CUSTOM,
                        agent_name="Extractor Agent",
                        agent_role="Structured Data Extractor",
                        required_tools=[],
                        depends_on=[spec.index],
                        estimated_tokens=500
                    )
                    new_sub_tasks.append(extractor_spec)

                    # Fix explicit dependencies for downstream coding agents
                    for s_down in valid_sub_tasks:
                        if s_down.agent_type.value == "coding" and spec.index in s_down.depends_on:
                            s_down.depends_on.remove(spec.index)
                            s_down.depends_on.append(max_idx)

                elif spec.agent_type.value == "coding":
                    max_idx += 1
                    critic_spec = type(spec)(
                        index=max_idx,
                        description="Critique the code execution output",
                        agent_type=_AT.CRITIC,
                        agent_name="Critic Agent",
                        agent_role="Strict Code Critic",
                        required_tools=[],
                        depends_on=[spec.index],
                        estimated_tokens=1000
                    )
                    new_sub_tasks.append(critic_spec)

                    # Fix explicit dependencies for downstream verification agents
                    for s_down in valid_sub_tasks:
                        if s_down.agent_type.value == "verification" and spec.index in s_down.depends_on:
                            s_down.depends_on.remove(spec.index)
                            s_down.depends_on.append(max_idx)
            valid_sub_tasks = new_sub_tasks

        # Step 0.6: Wire Summarizer Agent at the end in all modes
        has_summarizer = any(s.agent_type.value == "summarization" for s in valid_sub_tasks)
        if not has_summarizer:
            from meta_agent.schemas.blueprint import AgentType as _AT
            max_idx = max((s.index for s in valid_sub_tasks), default=-1) + 1
            last_agent_idx = valid_sub_tasks[-1].index if valid_sub_tasks else -1
            
            summ_spec = SubTaskSpec(
                index=max_idx,
                description="Summarize findings and output final code",
                agent_type=_AT.SUMMARIZATION,
                agent_name="Summarization Agent",
                agent_role="Final Output Formatter",
                required_tools=[],
                depends_on=[last_agent_idx] if last_agent_idx >= 0 else [],
                estimated_tokens=500
            )
            valid_sub_tasks.append(summ_spec)

        plan.sub_tasks = valid_sub_tasks

        # Step 1: Build agents
        agents = [self._build_agent(spec) for spec in plan.sub_tasks]
        agent_id_map = {spec.index: agent.agent_id for spec, agent in zip(plan.sub_tasks, agents)}

        # Step 2: Wire edges
        edges = self._build_edges(plan.sub_tasks, agent_id_map, plan.topology)

        # Step 3: Determine parallel groups
        parallel_groups = self._build_parallel_groups(plan.sub_tasks, agent_id_map)

        # Step 4: Build execution graph
        execution_graph = ExecutionGraph(
            topology=plan.topology,
            entry_point=agents[0].agent_id,
            agents=agents,
            edges=edges,
            parallel_groups=parallel_groups,
        )

        # Step 5: Build evaluation config
        evaluation = self._build_evaluation(plan.evaluation_focus)

        # ── Auto-add data_accuracy dimension for research+coding tasks ──
        agent_types = {s.agent_type for s in plan.sub_tasks}
        from meta_agent.schemas.blueprint import AgentType as _AT
        if _AT.RESEARCH in agent_types and _AT.CODING in agent_types:
            if not any(d.name == "data_accuracy" for d in evaluation.dimensions):
                evaluation.dimensions.append(EvaluationDimension(
                    name="data_accuracy",
                    description=(
                        "Were real-world values sourced from current, relevant "
                        "data rather than historical or forecast articles? Was "
                        "the researched value actually used in the code rather "
                        "than hardcoded?"
                    ),
                    weight=2.0,
                    threshold=0.6,
                ))
            # Raise threshold for complex tasks
            if plan.complexity_rating >= 5:
                evaluation.overall_threshold = 0.75

        # Step 6: Assemble blueprint
        blueprint = Blueprint(
            task_description=plan.original_task,
            task_decomposition=[s.description for s in plan.sub_tasks],
            reasoning_trace=plan.reasoning,
            execution_graph=execution_graph,
            evaluation=evaluation,
            estimated_token_budget=plan.estimated_total_tokens,
            tags=[plan.topology.value, f"complexity_{plan.complexity_rating}"],
            metadata={"pipeline_mode": mode},
        )

        logger.info(
            "Blueprint %s generated: %d agents, %d edges, topology=%s",
            blueprint.blueprint_id,
            len(agents),
            len(edges),
            plan.topology.value,
        )
        return blueprint

    # ── Private Helpers ─────────────────────────────────────────────

    def _build_agent(self, spec: SubTaskSpec) -> AgentConfig:
        """Build an AgentConfig from a SubTaskSpec."""
        tools = [self._build_tool(tool_type) for tool_type in spec.required_tools]

        prompt_template = DEFAULT_PROMPTS.get(
            spec.agent_type.value,
            DEFAULT_PROMPTS["custom"],
        )

        return AgentConfig(
            agent_type=spec.agent_type,
            name=spec.agent_name or f"{spec.agent_type.value.title()} Agent",
            role_description=spec.agent_role or spec.description,
            tools=tools,
            system_prompt_template=prompt_template,
            retry_policy=RetryPolicy(max_retries=2),
            metadata={"sub_task_index": spec.index},
        )

    def _build_tool(self, tool_type_value: Any) -> ToolConfig:
        """Build a ToolConfig from a tool type string."""
        tool_type_str = tool_type_value.value if hasattr(tool_type_value, "value") else str(tool_type_value)
        defaults = DEFAULT_TOOL_CONFIGS.get(tool_type_str, {})

        return ToolConfig(
            tool_type=tool_type_value,
            name=defaults.get("name", tool_type_str.replace("_", " ").title()),
            description=defaults.get("description", ""),
            permission_scope=defaults.get("permission_scope", "read"),
            timeout_seconds=defaults.get("timeout_seconds", 30),
        )

    def _build_edges(
        self,
        sub_tasks: list[SubTaskSpec],
        agent_id_map: dict[int, str],
        topology: TopologyType,
    ) -> list[EdgeConfig]:
        """Wire edges based on declared dependencies and topology type."""
        edges: list[EdgeConfig] = []

        if topology == TopologyType.SEQUENTIAL:
            # Linear chain: 0→1→2→...
            for i in range(len(sub_tasks) - 1):
                edges.append(EdgeConfig(
                    source_agent_id=agent_id_map[sub_tasks[i].index],
                    target_agent_id=agent_id_map[sub_tasks[i + 1].index],
                    condition_type=EdgeConditionType.ON_SUCCESS,
                ))

        elif topology == TopologyType.PARALLEL:
            # All tasks except the last run in parallel, then fan-in to the last
            if len(sub_tasks) > 1:
                final = sub_tasks[-1]
                for spec in sub_tasks[:-1]:
                    edges.append(EdgeConfig(
                        source_agent_id=agent_id_map[spec.index],
                        target_agent_id=agent_id_map[final.index],
                        condition_type=EdgeConditionType.ON_SUCCESS,
                    ))

        elif topology in (TopologyType.DAG, TopologyType.HIERARCHICAL):
            # Use explicit dependencies from the plan
            for spec in sub_tasks:
                for dep_idx in spec.depends_on:
                    if dep_idx in agent_id_map:
                        edges.append(EdgeConfig(
                            source_agent_id=agent_id_map[dep_idx],
                            target_agent_id=agent_id_map[spec.index],
                            condition_type=EdgeConditionType.ON_SUCCESS,
                        ))

        return edges

    def _build_parallel_groups(
        self,
        sub_tasks: list[SubTaskSpec],
        agent_id_map: dict[int, str],
    ) -> list[list[str]]:
        """Identify groups of tasks that can run concurrently."""
        # Group by dependency depth
        depth_map: dict[int, int] = {}
        for spec in sub_tasks:
            if not spec.depends_on:
                depth_map[spec.index] = 0
            else:
                depth_map[spec.index] = max(
                    depth_map.get(d, 0) for d in spec.depends_on
                ) + 1

        # Group by depth level
        depth_groups: dict[int, list[str]] = {}
        for spec in sub_tasks:
            depth = depth_map.get(spec.index, 0)
            depth_groups.setdefault(depth, []).append(agent_id_map[spec.index])

        # Only return groups with 2+ members (actual parallelism)
        return [group for group in depth_groups.values() if len(group) > 1]

    def _build_evaluation(self, focus_areas: list[str]) -> EvaluationConfig:
        """Build evaluation config emphasising the plan's focus areas."""
        dimensions = []
        for focus in focus_areas:
            weight = 3.0 if focus == "correctness" else 2.0
            dimensions.append(EvaluationDimension(
                name=focus,
                description=f"Quality of {focus}",
                weight=weight,
                threshold=0.5,
            ))

        # Always include coherence
        if not any(d.name == "coherence" for d in dimensions):
            dimensions.append(EvaluationDimension(
                name="coherence",
                description="Is the output well-structured and clear?",
                weight=1.0,
                threshold=0.4,
            ))

        eval_config = EvaluationConfig(
            strategy=EvaluationStrategy.LLM_JUDGE,
            dimensions=dimensions,
        )
        eval_config.evaluation_prompt_template = (
            "EVALUATION SCOPE: Score ONLY the output of the final coding agent — the last\n"
            "Python script block in the output. Ignore all code snippets written by research,\n"
            "planning, or data_analysis agents. Those are illustrative intermediate steps, not\n"
            "the deliverable being evaluated.\n\n"
        )
        return eval_config
