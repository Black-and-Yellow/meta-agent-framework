"""
Test Suite — Blueprint Schemas
==============================

Tests for all Pydantic models: validation, serialization, edge cases.
"""

from __future__ import annotations

import pytest

from meta_agent.schemas.blueprint import (
    AgentConfig,
    AgentType,
    Blueprint,
    EdgeConditionType,
    EdgeConfig,
    EvaluationConfig,
    EvaluationDimension,
    ExecutionGraph,
    RetryPolicy,
    ToolConfig,
    ToolType,
    TopologyType,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ToolConfig Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolConfig:
    def test_create_minimal(self):
        tool = ToolConfig(tool_type=ToolType.WEB_SEARCH, name="Search")
        assert tool.tool_type == ToolType.WEB_SEARCH
        assert tool.name == "Search"
        assert tool.timeout_seconds == 30
        assert tool.permission_scope == "read"

    def test_auto_generated_id(self):
        t1 = ToolConfig(tool_type=ToolType.WEB_SEARCH, name="S1")
        t2 = ToolConfig(tool_type=ToolType.WEB_SEARCH, name="S2")
        assert t1.tool_id != t2.tool_id

    def test_serialization_roundtrip(self):
        tool = ToolConfig(
            tool_type=ToolType.CODE_EXECUTOR,
            name="Code Runner",
            description="Runs Python code",
            timeout_seconds=60,
            permission_scope="execute",
        )
        data = tool.model_dump()
        restored = ToolConfig.model_validate(data)
        assert restored.name == "Code Runner"
        assert restored.permission_scope == "execute"


# ═══════════════════════════════════════════════════════════════════════════════
# AgentConfig Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentConfig:
    def test_create_agent(self):
        agent = AgentConfig(
            agent_type=AgentType.RESEARCH,
            name="Researcher",
            role_description="Find information about a topic",
        )
        assert agent.agent_type == AgentType.RESEARCH
        assert agent.temperature == 0.3

    def test_duplicate_tool_ids_rejected(self):
        tools = [
            ToolConfig(tool_id="same", tool_type=ToolType.WEB_SEARCH, name="S1"),
            ToolConfig(tool_id="same", tool_type=ToolType.API_CALLER, name="S2"),
        ]
        with pytest.raises(ValueError, match="Duplicate tool_id"):
            AgentConfig(
                agent_type=AgentType.RESEARCH,
                name="Agent",
                role_description="A test agent for validation",
                tools=tools,
            )

    def test_role_description_min_length(self):
        with pytest.raises(ValueError):
            AgentConfig(
                agent_type=AgentType.CODING,
                name="Coder",
                role_description="short",  # < 10 chars
            )


# ═══════════════════════════════════════════════════════════════════════════════
# EdgeConfig Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeConfig:
    def test_always_edge(self):
        edge = EdgeConfig(
            source_agent_id="a1",
            target_agent_id="a2",
        )
        assert edge.condition_type == EdgeConditionType.ALWAYS

    def test_score_edge_requires_numeric(self):
        with pytest.raises(ValueError, match="numeric condition_value"):
            EdgeConfig(
                source_agent_id="a1",
                target_agent_id="a2",
                condition_type=EdgeConditionType.SCORE_ABOVE,
                condition_value="not_a_number",
            )

    def test_score_edge_valid(self):
        edge = EdgeConfig(
            source_agent_id="a1",
            target_agent_id="a2",
            condition_type=EdgeConditionType.SCORE_ABOVE,
            condition_value=0.7,
        )
        assert edge.condition_value == 0.7

    def test_conditional_edge_requires_string(self):
        with pytest.raises(ValueError, match="non-empty string"):
            EdgeConfig(
                source_agent_id="a1",
                target_agent_id="a2",
                condition_type=EdgeConditionType.CONDITIONAL,
                condition_value=None,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# ExecutionGraph Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecutionGraph:
    def _make_agents(self, n: int) -> list[AgentConfig]:
        return [
            AgentConfig(
                agent_id=f"agent_{i}",
                agent_type=AgentType.RESEARCH,
                name=f"Agent {i}",
                role_description=f"Research agent number {i} in the pipeline",
            )
            for i in range(n)
        ]

    def test_valid_sequential_graph(self):
        agents = self._make_agents(3)
        graph = ExecutionGraph(
            topology=TopologyType.SEQUENTIAL,
            entry_point="agent_0",
            agents=agents,
            edges=[
                EdgeConfig(source_agent_id="agent_0", target_agent_id="agent_1"),
                EdgeConfig(source_agent_id="agent_1", target_agent_id="agent_2"),
            ],
        )
        assert len(graph.agents) == 3
        assert len(graph.edges) == 2

    def test_invalid_entry_point(self):
        agents = self._make_agents(2)
        with pytest.raises(ValueError, match="entry_point"):
            ExecutionGraph(
                topology=TopologyType.SEQUENTIAL,
                entry_point="nonexistent",
                agents=agents,
            )

    def test_invalid_edge_reference(self):
        agents = self._make_agents(2)
        with pytest.raises(ValueError, match="not found"):
            ExecutionGraph(
                topology=TopologyType.SEQUENTIAL,
                entry_point="agent_0",
                agents=agents,
                edges=[
                    EdgeConfig(source_agent_id="agent_0", target_agent_id="ghost"),
                ],
            )

    def test_duplicate_agent_ids_rejected(self):
        agents = [
            AgentConfig(
                agent_id="same_id",
                agent_type=AgentType.RESEARCH,
                name="Agent 1",
                role_description="First agent with duplicate id test",
            ),
            AgentConfig(
                agent_id="same_id",
                agent_type=AgentType.CODING,
                name="Agent 2",
                role_description="Second agent with duplicate id test",
            ),
        ]
        with pytest.raises(ValueError, match="Duplicate agent_id"):
            ExecutionGraph(
                topology=TopologyType.SEQUENTIAL,
                entry_point="same_id",
                agents=agents,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Blueprint Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlueprint:
    def test_full_blueprint(self):
        agents = [
            AgentConfig(
                agent_id="researcher",
                agent_type=AgentType.RESEARCH,
                name="Researcher",
                role_description="Research the topic thoroughly and provide findings",
            ),
            AgentConfig(
                agent_id="writer",
                agent_type=AgentType.SUMMARIZATION,
                name="Writer",
                role_description="Write a summary based on the research findings",
            ),
        ]
        bp = Blueprint(
            task_description="Analyse the competitive landscape for AI tools",
            execution_graph=ExecutionGraph(
                topology=TopologyType.SEQUENTIAL,
                entry_point="researcher",
                agents=agents,
                edges=[
                    EdgeConfig(source_agent_id="researcher", target_agent_id="writer"),
                ],
            ),
        )
        assert bp.blueprint_id.startswith("bp_")
        assert bp.revision == 1

    def test_next_revision(self):
        agents = [
            AgentConfig(
                agent_id="a1",
                agent_type=AgentType.RESEARCH,
                name="Agent",
                role_description="A simple agent for revision testing purposes",
            ),
        ]
        bp = Blueprint(
            task_description="A test task for revision testing in the system",
            execution_graph=ExecutionGraph(
                topology=TopologyType.SEQUENTIAL,
                entry_point="a1",
                agents=agents,
            ),
        )
        bp2 = bp.next_revision()
        assert bp2.revision == 2

    def test_json_roundtrip(self):
        agents = [
            AgentConfig(
                agent_id="a1",
                agent_type=AgentType.CRITIC,
                name="Critic",
                role_description="A critic agent that evaluates output quality",
            ),
        ]
        bp = Blueprint(
            task_description="Test task for JSON serialization roundtrip testing",
            execution_graph=ExecutionGraph(
                topology=TopologyType.DAG,
                entry_point="a1",
                agents=agents,
            ),
        )
        json_str = bp.model_dump_json()
        restored = Blueprint.model_validate_json(json_str)
        assert restored.blueprint_id == bp.blueprint_id
        assert restored.execution_graph.topology == TopologyType.DAG
