"""
Graph Builder
=============

The Meta-Agent imagines an architecture.
This module makes that imagination executable.
JSON becomes a living graph of cooperating agents.

The GraphBuilder reads a Blueprint and constructs an executable
LangGraph-compatible state graph. It:

1. Creates agent nodes from AgentConfig
2. Wires edges with conditional routing
3. Sets up the entry point and termination conditions
4. Supports all topology types (sequential, parallel, hierarchical, DAG)
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from meta_agent.agents.base_agent import BaseAgent
from meta_agent.agents.factory import AgentFactory
from meta_agent.schemas.blueprint import (
    Blueprint,
    EdgeConditionType,
    ExecutionGraph,
    TopologyType,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Representation
# ═══════════════════════════════════════════════════════════════════════════════

class AgentNode:
    """A node in the execution graph wrapping a live agent instance."""

    def __init__(self, agent: BaseAgent, node_id: str) -> None:
        self.agent = agent
        self.node_id = node_id
        self.outgoing_edges: list[ConditionalEdge] = []

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent and update the state."""
        input_data = {
            "task": state.get("original_task", ""),
            "context": state.get("context", ""),
            "previous_results": state.get("intermediate_results", {}),
        }

        # Forward research_data from previous agents so downstream
        # agents (e.g. coding) can use structured data directly
        intermediate = state.get("intermediate_results", {})
        for _agent_id, agent_result in intermediate.items():
            if isinstance(agent_result, dict) and "research_data" in agent_result:
                input_data["research_data"] = agent_result["research_data"]
                break  # use the first one found

        result = await self.agent.run(input_data)
        state.setdefault("intermediate_results", {})[self.node_id] = result
        state["last_result"] = result
        state["last_node"] = self.node_id
        return state


class ConditionalEdge:
    """
    A directed edge with a runtime condition.

    The condition is evaluated at runtime to decide if this edge
    should be traversed.
    """

    def __init__(
        self,
        source_id: str,
        target_id: str,
        condition_type: EdgeConditionType,
        condition_value: Any = None,
        priority: int = 0,
    ) -> None:
        self.source_id = source_id
        self.target_id = target_id
        self.condition_type = condition_type
        self.condition_value = condition_value
        self.priority = priority

    def evaluate(self, state: dict[str, Any]) -> bool:
        """Evaluate whether this edge should be traversed."""
        if self.condition_type == EdgeConditionType.ALWAYS:
            return True

        last_result = state.get("last_result", {})
        last_status = last_result.get("status", "success")
        last_score = last_result.get("confidence", 0.5)

        if self.condition_type == EdgeConditionType.ON_SUCCESS:
            return last_status != "failed"
        elif self.condition_type == EdgeConditionType.ON_FAILURE:
            return last_status == "failed"
        elif self.condition_type == EdgeConditionType.SCORE_ABOVE:
            return last_score > (self.condition_value or 0.5)
        elif self.condition_type == EdgeConditionType.SCORE_BELOW:
            return last_score < (self.condition_value or 0.5)
        elif self.condition_type == EdgeConditionType.CONDITIONAL:
            # Evaluate a simple expression against the state
            try:
                return bool(eval(str(self.condition_value), {"state": state}))  # noqa: S307
            except Exception:
                logger.warning("Edge condition eval failed: %s", self.condition_value)
                return False

        return True


class CompiledGraph:
    """
    The executable graph — ready to be run by the GraphExecutor.

    Contains:
    - nodes: dict of node_id → AgentNode
    - edges: list of ConditionalEdge
    - entry_point: starting node_id
    - nx_graph: networkx DiGraph for analysis (cycle detection, toposort)
    - parallel_groups: list of node groups that can run concurrently
    """

    def __init__(
        self,
        nodes: dict[str, AgentNode],
        edges: list[ConditionalEdge],
        entry_point: str,
        parallel_groups: list[list[str]],
        topology: TopologyType,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.entry_point = entry_point
        self.parallel_groups = parallel_groups
        self.topology = topology

        # Build networkx graph for analysis
        self.nx_graph = nx.DiGraph()
        for nid in nodes:
            self.nx_graph.add_node(nid)
        for edge in edges:
            self.nx_graph.add_edge(edge.source_id, edge.target_id)

    def get_execution_order(self) -> list[str]:
        """Return a topologically sorted execution order."""
        try:
            return list(nx.topological_sort(self.nx_graph))
        except nx.NetworkXUnfeasible:
            logger.warning("Graph has cycles — falling back to BFS order")
            return list(nx.bfs_tree(self.nx_graph, self.entry_point))

    def has_cycles(self) -> bool:
        """Check for cycles in the graph."""
        return not nx.is_directed_acyclic_graph(self.nx_graph)

    def get_outgoing_edges(self, node_id: str) -> list[ConditionalEdge]:
        """Get all outgoing edges from a node, sorted by priority."""
        edges = [e for e in self.edges if e.source_id == node_id]
        return sorted(edges, key=lambda e: e.priority, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Builder
# ═══════════════════════════════════════════════════════════════════════════════

class GraphBuilder:
    """
    Reads a Blueprint and constructs an executable graph.

    Pseudocode:
        blueprint = Blueprint.parse(...)
        graph = GraphBuilder().build(blueprint)
        result = await GraphExecutor().execute(graph, context)
    """

    def __init__(self, agent_factory: AgentFactory | None = None) -> None:
        self.agent_factory = agent_factory or AgentFactory()

    def build(self, blueprint: Blueprint) -> CompiledGraph:
        """
        Build an executable graph from a blueprint.

        Steps:
        1. Instantiate agent nodes from AgentConfigs
        2. Wire conditional edges from EdgeConfigs
        3. Validate the graph (no dangling refs, cycle handling)
        4. Return a CompiledGraph ready for execution
        """
        exec_graph = blueprint.execution_graph
        logger.info(
            "Building graph: %d agents, %d edges, topology=%s",
            len(exec_graph.agents),
            len(exec_graph.edges),
            exec_graph.topology.value,
        )

        # Step 1: Create agent nodes
        nodes: dict[str, AgentNode] = {}
        for agent_config in exec_graph.agents:
            agent = self.agent_factory.create(agent_config)
            node = AgentNode(agent=agent, node_id=agent_config.agent_id)
            nodes[agent_config.agent_id] = node

        # Step 2: Inject tool registry into every agent
        from meta_agent.tools.registry import ToolRegistry
        registry = ToolRegistry()
        if not registry.list_tools():
            registry.register_defaults()
        for node in nodes.values():
            node.agent.set_tool_registry(registry)

        # Step 3: Wire edges
        edges: list[ConditionalEdge] = []
        for edge_config in exec_graph.edges:
            edge = ConditionalEdge(
                source_id=edge_config.source_agent_id,
                target_id=edge_config.target_agent_id,
                condition_type=edge_config.condition_type,
                condition_value=edge_config.condition_value,
                priority=edge_config.priority,
            )
            edges.append(edge)
            nodes[edge_config.source_agent_id].outgoing_edges.append(edge)

        # Step 3: Build compiled graph
        compiled = CompiledGraph(
            nodes=nodes,
            edges=edges,
            entry_point=exec_graph.entry_point,
            parallel_groups=exec_graph.parallel_groups,
            topology=exec_graph.topology,
        )

        # Step 4: Validate
        self._validate(compiled)

        logger.info(
            "Graph built: %d nodes | execution_order=%s",
            len(nodes),
            compiled.get_execution_order(),
        )
        return compiled

    def _validate(self, graph: CompiledGraph) -> None:
        """Validate the compiled graph for structural issues."""
        # Check entry point exists
        if graph.entry_point not in graph.nodes:
            raise ValueError(
                f"Entry point '{graph.entry_point}' not found in graph nodes"
            )

        # Check for cycles
        if graph.has_cycles():
            logger.warning(
                "Graph contains cycles — the executor must handle termination"
            )

        # Check all edge targets exist
        for edge in graph.edges:
            if edge.target_id not in graph.nodes:
                raise ValueError(
                    f"Edge target '{edge.target_id}' not found in graph nodes"
                )

        # Check for unreachable nodes
        reachable = set(nx.descendants(graph.nx_graph, graph.entry_point))
        reachable.add(graph.entry_point)
        unreachable = set(graph.nodes.keys()) - reachable
        if unreachable:
            logger.warning("Unreachable nodes in graph: %s", unreachable)
