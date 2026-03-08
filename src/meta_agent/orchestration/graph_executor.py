"""
Graph Executor
==============

The graph represents a team of agents.
This executor coordinates their work,
manages state, and prevents chaos.

The GraphExecutor takes a CompiledGraph and drives it through to completion:

- Sequential execution follows topological order
- Parallel fan-out/fan-in for concurrent groups
- Conditional routing at each node's outgoing edges
- Timeout and error handling at every step
- Inner repair loop: verification failure triggers code re-generation
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from typing import Any

from meta_agent.orchestration.graph_builder import CompiledGraph, AgentNode
from meta_agent.schemas.blueprint import TopologyType
from meta_agent.schemas.state import ExecutionContext, TaskStatus

logger = logging.getLogger(__name__)

# Default timeout for a single agent execution (seconds)
DEFAULT_AGENT_TIMEOUT = 120
# Maximum total execution time for the entire graph (seconds)
DEFAULT_GRAPH_TIMEOUT = 600
# Max inner repair attempts when verification fails
DEFAULT_MAX_INNER_REPAIRS = 1

# Simple in-memory cache for research outputs to avoid redundant LLM calls
_RESEARCH_CACHE: dict[str, dict[str, Any]] = {}

# ── Dynamic planning skip ────────────────────────────────────────────
# Patterns that indicate a simple, single-step task
_SIMPLE_TASK_PATTERNS = [
    r"\bwrite\b",
    r"\bconvert\b",
    r"\bcalculate\b",
    r"\bformat\b",
    r"\bsort\b",
    r"\breverse\b",
    r"\bprint\b",
    r"\bgenerate\s+a\b",
    r"\bcreate\s+a\s+function\b",
]
_SIMPLE_TASK_RE = re.compile("|".join(_SIMPLE_TASK_PATTERNS), re.IGNORECASE)

# Action verbs — if a task contains >1, it is considered complex
_ACTION_VERBS = {
    "research", "analyse", "analyze", "summarize", "summarise", "implement",
    "design", "plan", "compare", "evaluate", "investigate", "build",
    "deploy", "integrate", "optimise", "optimize", "refactor",
}

# Sandbox restriction error patterns
_SANDBOX_ERROR_PATTERNS = [
    r"blocked import detected",
    r"network access not allowed",
    r"sandbox restriction",
    r"validation failed.*blocked",
    r"no internet access",
]


class GraphExecutor:
    """
    Executes a CompiledGraph by driving agents through the topology.

    Supports:
    - Sequential pipelines (SEQUENTIAL topology)
    - Parallel fan-out/fan-in (PARALLEL topology)
    - DAG traversal with conditional edges (DAG topology)
    - Hierarchical delegation (HIERARCHICAL topology)
    - Inner repair loop (verification → coding feedback cycle)
    """

    def __init__(
        self,
        agent_timeout: int = DEFAULT_AGENT_TIMEOUT,
        graph_timeout: int = DEFAULT_GRAPH_TIMEOUT,
        max_inner_repairs: int = DEFAULT_MAX_INNER_REPAIRS,
    ) -> None:
        self.agent_timeout = agent_timeout
        self.graph_timeout = graph_timeout
        self.max_inner_repairs = max_inner_repairs

    async def execute(
        self,
        graph: CompiledGraph,
        context: ExecutionContext,
    ) -> Any:
        """
        Execute the compiled graph and return the final result.

        The execution strategy depends on the graph's topology:
        - SEQUENTIAL: nodes run one after another in topological order
        - PARALLEL: independent groups run concurrently, then fan-in
        - DAG: traverse edges conditionally from entry point
        - HIERARCHICAL: first node delegates to workers
        """
        logger.info(
            "Executing graph [%s]: topology=%s, %d nodes",
            context.execution_id,
            graph.topology.value,
            len(graph.nodes),
        )

        context.status = TaskStatus.EXECUTING
        start_time = time.time()

        state: dict[str, Any] = {
            "original_task": context.original_input,
            "intermediate_results": {},
            "context": "",
            "last_result": None,
            "last_node": None,
            "enable_summarizer": getattr(context, "enable_summarizer", True),
        }

        try:
            if graph.topology == TopologyType.SEQUENTIAL:
                state = await self._execute_sequential(graph, state)
            elif graph.topology == TopologyType.PARALLEL:
                state = await self._execute_parallel(graph, state)
            elif graph.topology in (TopologyType.DAG, TopologyType.HIERARCHICAL):
                state = await self._execute_dag(graph, state)
            else:
                state = await self._execute_sequential(graph, state)

        except asyncio.TimeoutError:
            logger.error("Graph execution timed out after %ds", self.graph_timeout)
            context.record_error("Graph execution timed out")
            context.status = TaskStatus.FAILED
            raise
        except Exception as e:
            logger.error("Graph execution failed: %s", e)
            context.record_error(str(e))
            context.status = TaskStatus.FAILED
            raise

        elapsed = time.time() - start_time
        context.wall_clock_seconds = elapsed

        # Collect results from all agents into the context
        intermediate = state.get("intermediate_results", {})
        for agent_id, result in intermediate.items():
            context.add_agent_result(agent_id, result)

        # Build a comprehensive final result that includes ALL agent outputs
        agent_outputs = []
        for agent_id in graph.get_execution_order():
            agent_result = intermediate.get(agent_id, {})
            output_text = agent_result.get("output", "")
            if output_text:
                node = graph.nodes.get(agent_id)
                agent_name = node.agent.name if node else agent_id
                agent_outputs.append(f"### {agent_name}\n{output_text}")

        # Check if summarizer ran and is enabled
        summarizer_output = None
        if getattr(context, "enable_summarizer", True):
            for agent_id in reversed(graph.get_execution_order()):
                node = graph.nodes.get(agent_id)
                if node and self._is_summarization_agent(node):
                    res = intermediate.get(agent_id, {})
                    if "output" in res and not res.get("skipped"):
                        summarizer_output = res["output"]
                    break
        
        if summarizer_output:
            final_output_text = summarizer_output
        else:
            final_output_text = "\n\n---\n\n".join(agent_outputs)

        final: dict[str, Any] = {
            "output": final_output_text,
            "agent_results": intermediate,
            "status": "success" if not any(
                r.get("status") == "failed" for r in intermediate.values()
            ) else "failed",
        }

        context.final_output = final
        logger.info("Graph execution completed in %.1fs", elapsed)

        return final

    # ── Sequential Execution ────────────────────────────────────────

    async def _execute_sequential(
        self,
        graph: CompiledGraph,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute nodes one by one in topological order.

        Includes an inner repair loop: if a verification agent returns
        FAIL, the executor re-runs the preceding coding agent with
        the verifier's feedback, then re-verifies.  This repeats up to
        ``max_inner_repairs`` times.
        """
        order = graph.get_execution_order()
        logger.info("Sequential execution order: %s", order)

        # Extract original task for complexity check
        original_task = state.get("original_task", "")

        for idx, node_id in enumerate(order):
            node = graph.nodes[node_id]

            # ── Dynamic skip for simple tasks ───────────────
            config = getattr(node.agent, 'config', None)
            is_skippable = config and config.agent_type.value in SKIPPABLE_FOR_SIMPLE_TASKS
            if is_skippable and self._is_simple_task(original_task):
                logger.info(
                    "Skipping agent %s — task is simple enough",
                    node_id,
                )
                state.setdefault("intermediate_results", {})[node_id] = {
                    "output": "Agent skipped for simple task",
                    "skipped": True,
                }
                state["last_result"] = state["intermediate_results"][node_id]
                state["last_node"] = node_id
                continue
                
            # ── Dynamic skip for summarization ───────────────
            if self._is_summarization_agent(node) and not state.get("enable_summarizer", True):
                logger.info("Skipping summarization agent %s — disabled in context", node_id)
                state.setdefault("intermediate_results", {})[node_id] = {
                    "output": "Summarization skipped: disabled.",
                    "skipped": True,
                }
                state["last_result"] = state["intermediate_results"][node_id]
                state["last_node"] = node_id
                continue

            # ── Cache-aware research skip ───────────────
            if self._is_research_agent(node):
                task_hash = hashlib.md5(original_task.encode()).hexdigest()
                cached = _RESEARCH_CACHE.get(task_hash)
                if cached and (time.time() - cached["timestamp"]) < 3600:
                    logger.info("Cache hit for research agent %s. Skipping LLM call.", node_id)
                    state.setdefault("intermediate_results", {})[node_id] = cached["result"]
                    state["last_result"] = state["intermediate_results"][node_id]
                    state["last_node"] = node_id
                    continue

            # Conditional verification skip: if the preceding coding agent's
            # sandbox already succeeded, skip the verification LLM call.
            is_verifier = self._is_verification_agent(node)
            if is_verifier and idx > 0:
                prev_id = order[idx - 1]
                prev_result = state.get("intermediate_results", {}).get(prev_id, {})
                code_exec = prev_result.get("code_execution_result", {})
                if isinstance(code_exec, dict) and code_exec.get("success"):
                    stderr = code_exec.get("stderr", "")
                    if not (stderr.strip() and "error" in stderr.lower()):
                        logger.info(
                            "Skipping verification agent %s — coding sandbox already PASSED",
                            node_id,
                        )
                        state.setdefault("intermediate_results", {})[node_id] = {
                            "output": "Verification skipped: sandbox execution passed.",
                            "verification_passed": True,
                            "code_test_result": code_exec,
                            "confidence": 0.9,
                            "repair_instructions": "",
                            "check_results": {},
                        }
                        state["last_result"] = state["intermediate_results"][node_id]
                        state["last_node"] = node_id
                        continue

            # ── Inject structured research data before coding agent ────
            if self._is_coding_agent(node) and idx > 0:
                prev_id = order[idx - 1]
                prev_result = state.get("intermediate_results", {}).get(prev_id, {})
                rd = prev_result.get("research_data", {})
                if isinstance(rd, dict) and (
                    rd.get("temperature_celsius") is not None
                    or rd.get("temperature_fahrenheit") is not None
                ):
                    location = rd.get("location", "unknown")
                    temp_c = rd.get("temperature_celsius")
                    source = rd.get("source", "unknown")
                    
                    data_block = (
                        "RESEARCHED DATA — inject these exact values into your code.\n"
                        "Do NOT hardcode temperature_fahrenheit. Calculate it by calling your conversion function with temperature_celsius.\n"
                        f"- location: {location}\n"
                        f"- temperature_celsius: {temp_c}\n"
                        f"- source: {source}"
                    )
                    if temp_c is None:
                        data_block += "\nTemperature extraction failed. Use a placeholder value of 0.0 and add a comment explaining the data was unavailable."

                    # Prepend to state context so AgentNode.__call__ picks it up
                    existing_ctx = state.get("context", "")
                    state["context"] = (
                        data_block + "\n\n" + existing_ctx
                        if existing_ctx else data_block
                    )

            state = await self._execute_node(node, state)

            last_result = state.get("intermediate_results", {}).get(node_id, {})
            
            # ── Store research results in cache ──
            if self._is_research_agent(node):
                task_hash = hashlib.md5(original_task.encode()).hexdigest()
                _RESEARCH_CACHE[task_hash] = {
                    "timestamp": time.time(),
                    "result": last_result
                }
            
            # ── Critic Agent Inner Repair Loop ──
            is_critic = self._is_critic_agent(node)
            if is_critic and "critic_report" in last_result:
                report = last_result["critic_report"]
                if not report.get("passed", True) and idx > 0:
                    prev_node_id = order[idx - 1]
                    prev_node = graph.nodes[prev_node_id]
                    if self._is_coding_agent(prev_node):
                        issues_str = "\n- ".join(report.get("issues", []))
                        repair_feedback = f"Critic identified issues:\n- {issues_str}\nPlease fix your code."
                        logger.info("═══ CRITIC REPAIR 1/1: re-executing %s with feedback ═══", prev_node_id)
                        state["_repair_feedback"] = repair_feedback
                        # Re-execute the coding agent
                        state = await self._execute_node(prev_node, state)
                        state.pop("_repair_feedback", None)
                        # We don't re-execute the critic; we just proceed to verification
                        last_result = state.get("intermediate_results", {}).get(prev_node_id, {})

            # ── Inner Repair Loop ───────────────────────────────────
            # Check if this node is a verification agent that returned FAIL
            is_verifier = self._is_verification_agent(node)
            verification_failed = (
                is_verifier
                and not last_result.get("verification_passed", True)
            )

            if verification_failed and idx > 0:
                # Find the preceding coding agent to re-execute
                prev_node_id = order[idx - 1]
                prev_node = graph.nodes[prev_node_id]

                if self._is_coding_agent(prev_node):
                    repair_feedback = last_result.get("repair_instructions", "")
                    if not repair_feedback:
                        repair_feedback = last_result.get("output", "Verification failed")

                    # Track previous output for duplicate detection
                    prev_coding_output = (
                        state.get("intermediate_results", {})
                        .get(prev_node_id, {})
                        .get("output", "")
                    )

                    for repair_attempt in range(1, self.max_inner_repairs + 1):
                        logger.info(
                            "═══ INNER REPAIR %d/%d: re-executing %s with feedback ═══",
                            repair_attempt,
                            self.max_inner_repairs,
                            prev_node_id,
                        )

                        # Detect sandbox restriction errors → add stricter constraint
                        prev_exec = (
                            state.get("intermediate_results", {})
                            .get(prev_node_id, {})
                            .get("code_execution_result", {})
                        )
                        if self._is_sandbox_restriction_error(prev_exec):
                            repair_feedback += (
                                "\n\nSANDBOX RESTRICTION: The sandbox does NOT allow "
                                "network access. Do NOT import requests, urllib, httpx, "
                                "aiohttp, socket, or subprocess. Use ONLY the Python "
                                "standard library."
                            )

                        # Inject repair feedback into the coding agent's input
                        state["_repair_feedback"] = repair_feedback

                        # Re-execute coding agent
                        state = await self._execute_node(prev_node, state)

                        # Clear repair feedback
                        state.pop("_repair_feedback", None)

                        # Duplicate detection: abort if output is unchanged
                        new_coding_output = (
                            state.get("intermediate_results", {})
                            .get(prev_node_id, {})
                            .get("output", "")
                        )
                        if self._outputs_are_similar(
                            prev_coding_output, new_coding_output
                        ):
                            logger.warning(
                                "Repair attempt %d produced similar output — aborting repair loop",
                                repair_attempt,
                            )
                            break
                        prev_coding_output = new_coding_output

                        # Re-execute verification agent
                        state = await self._execute_node(node, state)

                        # Check if verification now passes
                        verify_result = state.get("intermediate_results", {}).get(node_id, {})
                        if verify_result.get("verification_passed", False):
                            logger.info(
                                "✓ Repair attempt %d PASSED verification",
                                repair_attempt,
                            )
                            break
                        else:
                            repair_feedback = verify_result.get(
                                "repair_instructions",
                                verify_result.get("output", "Still failing"),
                            )
                            logger.warning(
                                "✗ Repair attempt %d still FAILED",
                                repair_attempt,
                            )
                    else:
                        logger.warning(
                            "Max inner repairs (%d) exhausted — proceeding with best result",
                            self.max_inner_repairs,
                        )

        return state

    # ── Parallel Execution ──────────────────────────────────────────

    async def _execute_parallel(
        self,
        graph: CompiledGraph,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute parallel groups concurrently, then process remaining nodes.
        """
        parallel_node_ids = set()
        for group in graph.parallel_groups:
            parallel_node_ids.update(group)

        # Phase 1: Execute parallel groups
        for group in graph.parallel_groups:
            logger.info("Executing parallel group: %s", group)
            tasks = []
            for node_id in group:
                node = graph.nodes[node_id]
                tasks.append(self._execute_node(node, state.copy()))

            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in group_results:
                if isinstance(result, Exception):
                    logger.error("Parallel node failed: %s", result)
                    continue
                if isinstance(result, dict):
                    state["intermediate_results"].update(
                        result.get("intermediate_results", {})
                    )

        # Phase 2: Execute remaining nodes (post-fan-in)
        order = graph.get_execution_order()
        for node_id in order:
            if node_id not in parallel_node_ids:
                node = graph.nodes[node_id]
                state = await self._execute_node(node, state)

        return state

    # ── DAG Execution ───────────────────────────────────────────────

    async def _execute_dag(
        self,
        graph: CompiledGraph,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Traverse the DAG following conditional edges from the entry point."""
        visited: set[str] = set()
        current_id = graph.entry_point
        max_steps = len(graph.nodes) * 2

        for step in range(max_steps):
            if current_id in visited and not graph.has_cycles():
                logger.warning("Revisiting node %s — stopping", current_id)
                break

            visited.add(current_id)
            node = graph.nodes[current_id]

            logger.info("DAG step %d: executing %s", step, current_id)
            state = await self._execute_node(node, state)

            outgoing = graph.get_outgoing_edges(current_id)
            next_id = None
            for edge in outgoing:
                if edge.evaluate(state):
                    next_id = edge.target_id
                    logger.info(
                        "Edge %s -> %s (condition=%s): MATCHED",
                        edge.source_id, edge.target_id, edge.condition_type.value,
                    )
                    break

            if next_id is None:
                logger.info("No outgoing edges matched — execution complete")
                break

            current_id = next_id

        return state

    # ── Node Execution ──────────────────────────────────────────────

    async def _execute_node(
        self,
        node: AgentNode,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single agent node with timeout handling."""
        # Inject repair feedback if present (from inner repair loop)
        if "_repair_feedback" in state:
            original_call = node.__call__

            async def _patched_call(s: dict[str, Any]) -> dict[str, Any]:
                input_data = {
                    "task": s.get("original_task", ""),
                    "context": s.get("context", ""),
                    "previous_results": s.get("intermediate_results", {}),
                    "repair_feedback": s.get("_repair_feedback", ""),
                }
                result = await node.agent.run(input_data)
                s.setdefault("intermediate_results", {})[node.node_id] = result
                s["last_result"] = result
                s["last_node"] = node.node_id
                return s

            call_fn = _patched_call
        else:
            call_fn = node.__call__

        try:
            state = await asyncio.wait_for(
                call_fn(state),
                timeout=self.agent_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Agent %s timed out after %ds",
                node.node_id,
                self.agent_timeout,
            )
            state.setdefault("intermediate_results", {})[node.node_id] = {
                "output": "Agent execution timed out",
                "status": "failed",
                "error": "TimeoutError",
            }
            state["last_result"] = state["intermediate_results"][node.node_id]
            state["last_node"] = node.node_id
        except Exception as e:
            logger.error("Agent %s failed: %s", node.node_id, e)
            state.setdefault("intermediate_results", {})[node.node_id] = {
                "output": str(e),
                "status": "failed",
                "error": type(e).__name__,
            }
            state["last_result"] = state["intermediate_results"][node.node_id]
            state["last_node"] = node.node_id

        return state

    # ── Agent Type Detection ────────────────────────────────────────

    @staticmethod
    def _is_research_agent(node: AgentNode) -> bool:
        """Check if the node contains a research agent."""
        config = getattr(node.agent, 'config', None)
        if config and hasattr(config, 'agent_type'):
            return config.agent_type.value == "research"
        return type(node.agent).__name__.lower().startswith("research")

    @staticmethod
    def _is_verification_agent(node: AgentNode) -> bool:
        """Check if the node contains a verification agent."""
        config = getattr(node.agent, 'config', None)
        if config and hasattr(config, 'agent_type'):
            return config.agent_type.value == "verification"
        # Fallback: check class name
        return type(node.agent).__name__.lower().startswith("verification")

    @staticmethod
    def _is_critic_agent(node: AgentNode) -> bool:
        """Check if the node contains a critic agent."""
        config = getattr(node.agent, 'config', None)
        if config and hasattr(config, 'agent_type'):
            return config.agent_type.value == "critic"
        return type(node.agent).__name__.lower().startswith("critic")

    @staticmethod
    def _is_summarization_agent(node: AgentNode) -> bool:
        """Check if the node contains a summarization agent."""
        config = getattr(node.agent, 'config', None)
        if config and hasattr(config, 'agent_type'):
            return config.agent_type.value == "summarization"
        return type(node.agent).__name__.lower().startswith("summarization")

    @staticmethod
    def _is_coding_agent(node: AgentNode) -> bool:
        """Check if the node contains a coding agent."""
        config = getattr(node.agent, 'config', None)
        if config and hasattr(config, 'agent_type'):
            return config.agent_type.value == "coding"
        return type(node.agent).__name__.lower().startswith("coding")

    @staticmethod
    def _is_planning_agent(node: AgentNode) -> bool:
        """Check if the node contains a planning agent."""
        config = getattr(node.agent, 'config', None)
        if config and hasattr(config, 'agent_type'):
            return config.agent_type.value == "planning"
        return type(node.agent).__name__.lower().startswith("planning")

    @staticmethod
    def _is_simple_task(task: str) -> bool:
        """Determine if a task is simple enough to skip planning.

        A task is simple when it matches single-step patterns (e.g.
        'write', 'convert', 'calculate', 'format') AND is short
        (< 25 words) AND does **not** contain multiple action verbs
        that would require multi-step reasoning.
        """
        if not task:
            return False
        # Must match at least one simple keyword pattern
        if not _SIMPLE_TASK_RE.search(task):
            return False
        # Short tasks only — long descriptions imply complexity
        if len(task.split()) >= 25:
            return False
        # Count distinct action verbs — more than 1 implies complexity
        # Count distinct action verbs — more than 1 implies complexity
        task_lower = task.lower()
        verb_count = sum(1 for v in _ACTION_VERBS if v in task_lower)
        
        # A task is simple if it matches the verbs AND is short AND has <=1 verb.
        # OR if it has fewer than 2 independent research targets (e.g., no "and", less commas)
        target_indicators = task_lower.count(" and ") + task_lower.count(",")
        
        if verb_count <= 1 and len(task.split()) < 25:
            return True
        if target_indicators < 1:
            return True
            
        return False

    @staticmethod
    def _is_sandbox_restriction_error(code_exec_result: Any) -> bool:
        """Detect if a code execution failure was caused by sandbox restrictions.

        Checks stderr, error message, and structured tool errors for patterns
        like 'Blocked import detected' or 'network access not allowed'.
        """
        if not isinstance(code_exec_result, dict):
            return False
        if code_exec_result.get("success"):
            return False
        # Collect all text fields that may contain the error
        texts = [
            str(code_exec_result.get("stderr", "")),
            str(code_exec_result.get("error", "")),
            str(code_exec_result.get("output", "")),
        ]
        combined = " ".join(texts).lower()
        return any(
            re.search(pat, combined)
            for pat in _SANDBOX_ERROR_PATTERNS
        )

    # ── Duplicate Output Detection ────────────────────────────────────────

    @staticmethod
    def _output_hash(text: str) -> str:
        """Compute a content hash for quick identical-output detection."""
        normalised = text.strip().lower()
        return hashlib.md5(normalised.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _outputs_are_similar(
        output_a: str, output_b: str, threshold: float = 0.92
    ) -> bool:
        """Check if two outputs are near-identical.

        Uses a fast hash check for exact matches, then falls back to
        character-level similarity for near-matches.
        """
        if not output_a or not output_b:
            return False
        a = output_a.strip()
        b = output_b.strip()
        # Fast path: exact match via hash
        if GraphExecutor._output_hash(a) == GraphExecutor._output_hash(b):
            return True
        # Quick length-based reject
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if len(longer) == 0:
            return False
        if len(shorter) / len(longer) < threshold:
            return False
        # Character-level match ratio
        matches = sum(1 for ca, cb in zip(a, b) if ca == cb)
        ratio = matches / max(len(a), len(b))
        return ratio >= threshold
