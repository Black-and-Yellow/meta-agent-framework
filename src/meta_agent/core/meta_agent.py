"""
Meta-Agent — The Brain
======================

This is the brain of the system.
It does not solve tasks directly.
Instead, it designs the machines that will solve them.
Think of it as an AI systems architect.

The Meta-Agent orchestrates the full lifecycle:

  1. Accept a user task
  2. Plan the agent pipeline (via TaskPlanner)
  3. Generate a validated Blueprint (via BlueprintGenerator)
  4. Build and execute the graph (via GraphBuilder + GraphExecutor)
  5. Evaluate the result (via ResultEvaluator)
  6. If quality is insufficient, repair/rebuild (via RepairLoop)
  7. Return the final result

The Meta-Agent is the entry point for all task processing.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from meta_agent.config import get_settings
from meta_agent.core.blueprint_generator import BlueprintGenerator
from meta_agent.core.planner import TaskPlan, TaskPlanner
from meta_agent.schemas.blueprint import Blueprint
from meta_agent.schemas.state import (
    EvaluationResult,
    ExecutionContext,
    RepairAction,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class MetaAgent:
    """
    The central orchestrator — an AI that designs AI systems.

    Usage:
        meta = MetaAgent()
        result = await meta.solve("Analyse competitor pricing and write a report")

    The MetaAgent handles the full reasoning→planning→building→execution→evaluation
    loop, including automatic repair when results don't meet quality thresholds.
    """

    def __init__(
        self,
        planner: TaskPlanner | None = None,
        generator: BlueprintGenerator | None = None,
        llm_client: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.planner = planner or TaskPlanner(llm_client=llm_client)
        self.generator = generator or BlueprintGenerator()
        self.max_repair_iterations = settings.max_repair_iterations
        self.evaluation_threshold = settings.evaluation_threshold

        # These are lazily resolved to avoid circular imports
        self._graph_builder = None
        self._graph_executor = None
        self._evaluator = None

    # ── Lazy Component Resolution ───────────────────────────────────

    @property
    def graph_builder(self) -> Any:
        if self._graph_builder is None:
            from meta_agent.orchestration.graph_builder import GraphBuilder
            self._graph_builder = GraphBuilder()
        return self._graph_builder

    @property
    def graph_executor(self) -> Any:
        if self._graph_executor is None:
            from meta_agent.orchestration.graph_executor import GraphExecutor
            self._graph_executor = GraphExecutor()
        return self._graph_executor

    @property
    def evaluator(self) -> Any:
        if self._evaluator is None:
            from meta_agent.evaluation.evaluator import ResultEvaluator
            self._evaluator = ResultEvaluator()
        return self._evaluator

    # ── Main Entry Point ────────────────────────────────────────────

    async def solve(self, task: str) -> dict[str, Any]:
        """
        End-to-end task solving.

        This is the main entry point:
          task (str) → plan → blueprint → graph → execute → evaluate → result

        Returns:
            Dict with keys: result, blueprint, evaluations, execution_context
        """
        start_time = time.time()
        logger.info("═" * 60)
        logger.info("META-AGENT: Received task")
        logger.info("Task: %s", task[:200])
        logger.info("═" * 60)

        # Phase 1: Plan
        plan = await self._plan(task)

        # Phase 2: Generate Blueprint
        blueprint = self._generate_blueprint(plan)

        # Phase 3–6: Build, Execute, Evaluate (with repair loop)
        result, context, evaluations = await self._execute_with_repair(
            task, blueprint
        )

        elapsed = time.time() - start_time
        logger.info(
            "META-AGENT: Task completed in %.1fs | iterations=%d | final_score=%.2f",
            elapsed,
            len(evaluations),
            evaluations[-1].overall_score if evaluations else 0.0,
        )

        return {
            "result": result,
            "blueprint": blueprint.model_dump(),
            "evaluations": [e.model_dump() for e in evaluations],
            "execution_context": context.model_dump() if context else None,
            "elapsed_seconds": elapsed,
        }

    # ── Phase Methods ───────────────────────────────────────────────

    async def _plan(self, task: str) -> TaskPlan:
        """Phase 1: Decompose the task into a structured plan."""
        logger.info("Phase 1: PLANNING — decomposing task into sub-tasks")
        plan = await self.planner.plan(task)
        logger.info(
            "Plan: %d sub-tasks | topology=%s | complexity=%d",
            len(plan.sub_tasks),
            plan.topology.value,
            plan.complexity_rating,
        )
        return plan

    def _generate_blueprint(self, plan: TaskPlan) -> Blueprint:
        """Phase 2: Convert plan to validated blueprint."""
        logger.info("Phase 2: BLUEPRINT GENERATION — converting plan to executable blueprint")
        blueprint = self.generator.generate(plan)
        logger.info("Blueprint %s generated", blueprint.blueprint_id)
        return blueprint

    async def _execute_with_repair(
        self,
        task: str,
        blueprint: Blueprint,
    ) -> tuple[Any, ExecutionContext | None, list[EvaluationResult]]:
        """
        Phases 3-6: Build graph, execute, evaluate, and repair if needed.

        This is the self-improvement loop. If the result doesn't meet
        quality thresholds, the system can:
          - REFINE: Re-run with adjusted parameters
          - REBUILD: Generate a new blueprint and start over
        """
        evaluations: list[EvaluationResult] = []
        context: ExecutionContext | None = None
        result: Any = None
        current_blueprint = blueprint
        cached_research: dict[str, Any] = {}  # Cache research across iterations

        for iteration in range(1, self.max_repair_iterations + 1):
            logger.info(
                "── Iteration %d/%d ──",
                iteration,
                self.max_repair_iterations,
            )

            # Phase 3: Build the graph
            logger.info("Phase 3: GRAPH CONSTRUCTION")
            graph = self.graph_builder.build(current_blueprint)

            # Phase 4: Execute
            logger.info("Phase 4: EXECUTION")
            context = ExecutionContext(
                task_id=f"task_{id(task)}",
                blueprint_id=current_blueprint.blueprint_id,
                original_input=task,
                status=TaskStatus.EXECUTING,
                current_iteration=iteration,
            )

            # Inject cached research results to skip re-running research agent
            if cached_research and iteration > 1:
                logger.info(
                    "Injecting cached research results from iteration 1 "
                    "(%d keys)", len(cached_research),
                )
                context.intermediate_results.update(cached_research)

            result = await self.graph_executor.execute(graph, context)

            # Cache research results after first iteration
            if iteration == 1 and context.intermediate_results:
                for agent_id, agent_result in context.intermediate_results.items():
                    if isinstance(agent_result, dict) and "research_data" in agent_result:
                        cached_research[agent_id] = agent_result
                        logger.info("Cached research results from agent: %s", agent_id)

            # Phase 5: Evaluate
            logger.info("Phase 5: EVALUATION")
            context.status = TaskStatus.EVALUATING
            evaluation = await self.evaluator.evaluate(
                task=task,
                result=result,
                blueprint=current_blueprint,
                context=context,
            )
            evaluation.iteration = iteration
            evaluations.append(evaluation)

            logger.info(
                "Evaluation: score=%.2f | passed=%s | action=%s",
                evaluation.overall_score,
                evaluation.passed,
                evaluation.recommended_action.value,
            )

            # Phase 6: Decide — accept, refine, or rebuild
            
            # ── 4C: Escalate to rebuild on consecutive failures ──
            action = evaluation.recommended_action
            if iteration > 1 and not evaluation.passed and action != RepairAction.REBUILD:
                prev_eval = evaluations[-2]
                curr_failed_dims = {d.name for d in evaluation.dimension_scores if not d.passed}
                prev_failed_dims = {d.name for d in prev_eval.dimension_scores if not d.passed}
                if curr_failed_dims & prev_failed_dims:
                    logger.warning(
                        "Escalating to REBUILD: same dimension failed twice %s", 
                        curr_failed_dims & prev_failed_dims
                    )
                    action = RepairAction.REBUILD

            # Early termination: if passed, stop regardless of recommended_action
            if evaluation.passed or action == RepairAction.ACCEPT:
                logger.info(
                    "✓ Result accepted (passed=%s, action=%s)",
                    evaluation.passed,
                    action.value,
                )
                context.status = TaskStatus.COMPLETED
                break

            if iteration >= self.max_repair_iterations:
                logger.warning(
                    "Max repair iterations reached (%d). Accepting best result.",
                    self.max_repair_iterations,
                )
                context.status = TaskStatus.COMPLETED
                break

            if action == RepairAction.REBUILD:
                logger.info("Phase 6: REBUILD — generating new blueprint")
                context.status = TaskStatus.REPAIRING
                current_blueprint = await self._rebuild_blueprint(
                    task, current_blueprint, evaluation,
                )
            else:
                logger.info("Phase 6: REFINE — adjusting pipeline parameters")
                context.status = TaskStatus.REPAIRING
                current_blueprint = self._refine_blueprint(
                    current_blueprint, evaluation,
                )

        context.final_output = result
        return result, context, evaluations

    async def _rebuild_blueprint(
        self,
        task: str,
        old_blueprint: Blueprint,
        evaluation: EvaluationResult,
    ) -> Blueprint:
        """
        Instead of retrying blindly, redesign the system.
        Failure becomes a signal for improvement.
        """
        # Create an enhanced task description with lessons learned
        enhanced_task = (
            f"{task}\n\n"
            f"IMPORTANT: A previous attempt scored {evaluation.overall_score:.2f}/1.0.\n"
            f"Issues identified: {evaluation.reasoning}\n"
            f"Suggestions: {'; '.join(evaluation.suggestions)}\n"
            f"Previous topology was: {old_blueprint.execution_graph.topology.value}\n"
            f"Please design a BETTER architecture addressing these issues."
        )

        plan = await self.planner.plan(enhanced_task)
        new_blueprint = self.generator.generate(plan)
        new_blueprint = new_blueprint.model_copy(update={
            "revision": old_blueprint.revision + 1,
        })

        logger.info(
            "Rebuilt blueprint: %s (rev %d → %d)",
            new_blueprint.blueprint_id,
            old_blueprint.revision,
            new_blueprint.revision,
        )
        return new_blueprint

    def _refine_blueprint(
        self,
        blueprint: Blueprint,
        evaluation: EvaluationResult,
    ) -> Blueprint:
        """
        Make targeted adjustments to the existing blueprint.

        - Lower temperature for agents that produced incoherent output
        - Increase token limits for agents that were cut short
        - Add verification steps for accuracy issues
        """
        refined = blueprint.next_revision()
        data = refined.model_dump()

        # Example refinement: lower temperature for all agents
        for agent_data in data["execution_graph"]["agents"]:
            if agent_data.get("temperature", 0.3) > 0.1:
                agent_data["temperature"] = max(0.1, agent_data["temperature"] - 0.1)

        refined = Blueprint.model_validate(data)
        logger.info("Refined blueprint: revision %d", refined.revision)
        return refined
