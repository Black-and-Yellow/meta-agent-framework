"""
Test Suite — Graph Builder & Router
====================================

Tests for graph construction, edge routing, and cycle detection.
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
    ExecutionGraph,
    TopologyType,
)
from meta_agent.orchestration.graph_builder import GraphBuilder, CompiledGraph
from meta_agent.orchestration.router import (
    detect_cycles,
    compute_execution_tiers,
    TerminationGuard,
)


def _make_blueprint(
    num_agents: int = 3,
    topology: TopologyType = TopologyType.SEQUENTIAL,
) -> Blueprint:
    """Helper to create a test blueprint."""
    agents = [
        AgentConfig(
            agent_id=f"agent_{i}",
            agent_type=AgentType.RESEARCH,
            name=f"Agent {i}",
            role_description=f"Test agent number {i} for graph building tests",
        )
        for i in range(num_agents)
    ]
    edges = [
        EdgeConfig(
            source_agent_id=f"agent_{i}",
            target_agent_id=f"agent_{i + 1}",
            condition_type=EdgeConditionType.ON_SUCCESS,
        )
        for i in range(num_agents - 1)
    ]
    return Blueprint(
        task_description="Test task for graph builder validation testing",
        execution_graph=ExecutionGraph(
            topology=topology,
            entry_point="agent_0",
            agents=agents,
            edges=edges,
        ),
    )


class TestGraphBuilder:
    def test_build_sequential(self):
        bp = _make_blueprint(3, TopologyType.SEQUENTIAL)
        builder = GraphBuilder()
        graph = builder.build(bp)

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph.entry_point == "agent_0"

    def test_execution_order(self):
        bp = _make_blueprint(4, TopologyType.SEQUENTIAL)
        builder = GraphBuilder()
        graph = builder.build(bp)

        order = graph.get_execution_order()
        assert order[0] == "agent_0"
        assert order[-1] == "agent_3"

    def test_no_cycles_in_sequential(self):
        bp = _make_blueprint(3, TopologyType.SEQUENTIAL)
        builder = GraphBuilder()
        graph = builder.build(bp)
        assert not graph.has_cycles()


class TestRouter:
    def test_detect_no_cycles(self):
        edges = [
            EdgeConfig(source_agent_id="a", target_agent_id="b"),
            EdgeConfig(source_agent_id="b", target_agent_id="c"),
        ]
        cycles = detect_cycles(edges)
        assert len(cycles) == 0

    def test_detect_cycle(self):
        edges = [
            EdgeConfig(source_agent_id="a", target_agent_id="b"),
            EdgeConfig(source_agent_id="b", target_agent_id="a"),
        ]
        cycles = detect_cycles(edges)
        assert len(cycles) > 0

    def test_execution_tiers(self):
        edges = [
            EdgeConfig(source_agent_id="a", target_agent_id="c"),
            EdgeConfig(source_agent_id="b", target_agent_id="c"),
        ]
        tiers = compute_execution_tiers(edges, ["a", "b", "c"])
        # a and b should be in the first tier, c in the second
        assert len(tiers) == 2
        assert set(tiers[0]) == {"a", "b"}
        assert tiers[1] == ["c"]


class TestTerminationGuard:
    def test_allows_initial_visits(self):
        guard = TerminationGuard(max_visits_per_node=3, max_total_steps=10)
        assert guard.should_continue("node_a")
        assert guard.should_continue("node_b")

    def test_blocks_excess_visits(self):
        guard = TerminationGuard(max_visits_per_node=2, max_total_steps=10)
        guard.should_continue("node_a")  # visit 1
        guard.should_continue("node_a")  # visit 2
        assert not guard.should_continue("node_a")  # visit 3 → blocked

    def test_blocks_excess_total_steps(self):
        guard = TerminationGuard(max_visits_per_node=100, max_total_steps=3)
        guard.should_continue("a")
        guard.should_continue("b")
        guard.should_continue("c")
        assert not guard.should_continue("d")  # step 4 → blocked

    def test_reset(self):
        guard = TerminationGuard(max_visits_per_node=1, max_total_steps=2)
        guard.should_continue("a")
        guard.should_continue("b")
        guard.reset()
        assert guard.should_continue("a")  # allowed again
