"""
Router — Conditional Edge Evaluation
=====================================

Provides helper functions for edge evaluation, cycle detection,
and termination guard logic. Used by the GraphExecutor when
traversing DAG topologies.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from meta_agent.schemas.blueprint import EdgeConditionType, EdgeConfig

logger = logging.getLogger(__name__)


def evaluate_edge_condition(
    edge: EdgeConfig,
    state: dict[str, Any],
) -> bool:
    """
    Evaluate whether a given edge should be traversed based on state.

    Args:
        edge: The edge configuration with condition type and value.
        state: Current execution state.

    Returns:
        True if the edge should be followed.
    """
    ct = edge.condition_type

    if ct == EdgeConditionType.ALWAYS:
        return True

    last_result = state.get("last_result", {})

    if ct == EdgeConditionType.ON_SUCCESS:
        return last_result.get("status", "success") != "failed"

    if ct == EdgeConditionType.ON_FAILURE:
        return last_result.get("status", "success") == "failed"

    if ct == EdgeConditionType.SCORE_ABOVE:
        score = last_result.get("confidence", 0.0)
        threshold = float(edge.condition_value or 0.5)
        return score > threshold

    if ct == EdgeConditionType.SCORE_BELOW:
        score = last_result.get("confidence", 0.0)
        threshold = float(edge.condition_value or 0.5)
        return score < threshold

    if ct == EdgeConditionType.CONDITIONAL:
        return _evaluate_expression(str(edge.condition_value), state)

    return True


def _evaluate_expression(expr: str, state: dict[str, Any]) -> bool:
    """
    Safely evaluate a simple boolean expression against state.

    Only allows access to the 'state' dict — no builtins, no imports.
    """
    try:
        # Restricted evaluation context
        allowed_names: dict[str, Any] = {"state": state, "len": len, "str": str, "int": int, "float": float}
        return bool(eval(expr, {"__builtins__": {}}, allowed_names))  # noqa: S307
    except Exception as e:
        logger.warning("Expression evaluation failed for '%s': %s", expr, e)
        return False


def detect_cycles(edges: list[EdgeConfig]) -> list[list[str]]:
    """
    Detect cycles in the edge list.

    Returns:
        List of cycles found, where each cycle is a list of agent_ids.
    """
    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge.source_agent_id, edge.target_agent_id)

    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            logger.warning("Detected %d cycle(s) in edge graph: %s", len(cycles), cycles)
        return cycles
    except nx.NetworkXError:
        return []


def compute_execution_tiers(edges: list[EdgeConfig], all_agent_ids: list[str]) -> list[list[str]]:
    """
    Compute execution tiers for parallel scheduling.

    Nodes at the same tier have no dependencies on each other and can
    run concurrently.

    Returns:
        List of tiers, where each tier is a list of agent_ids.
    """
    G = nx.DiGraph()
    for aid in all_agent_ids:
        G.add_node(aid)
    for edge in edges:
        G.add_edge(edge.source_agent_id, edge.target_agent_id)

    if not nx.is_directed_acyclic_graph(G):
        logger.warning("Cannot compute tiers: graph has cycles")
        return [all_agent_ids]

    # Use topological generations for tier computation
    return [list(gen) for gen in nx.topological_generations(G)]


class TerminationGuard:
    """
    Prevents infinite execution in graphs with cycles.

    Tracks how many times each node has been visited and stops
    traversal when limits are exceeded.
    """

    def __init__(self, max_visits_per_node: int = 3, max_total_steps: int = 50) -> None:
        self.max_visits_per_node = max_visits_per_node
        self.max_total_steps = max_total_steps
        self._visit_counts: dict[str, int] = {}
        self._total_steps = 0

    def should_continue(self, node_id: str) -> bool:
        """Check if we should continue executing this node."""
        self._visit_counts[node_id] = self._visit_counts.get(node_id, 0) + 1
        self._total_steps += 1

        if self._total_steps > self.max_total_steps:
            logger.warning("Termination guard: max total steps (%d) exceeded", self.max_total_steps)
            return False

        if self._visit_counts[node_id] > self.max_visits_per_node:
            logger.warning(
                "Termination guard: node '%s' visited %d times (max=%d)",
                node_id,
                self._visit_counts[node_id],
                self.max_visits_per_node,
            )
            return False

        return True

    def reset(self) -> None:
        """Reset all counters."""
        self._visit_counts.clear()
        self._total_steps = 0
