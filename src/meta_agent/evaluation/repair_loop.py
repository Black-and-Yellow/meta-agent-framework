"""
Repair Loop
============

If the result is bad, the architecture was wrong.
Instead of retrying blindly, we redesign the system.
Failure becomes a signal for improvement.

The RepairLoop decides what to do after evaluation:
  - ACCEPT: Score meets threshold → return the result
  - REFINE: Score is close → tweak parameters and retry
  - REBUILD: Score is far below threshold → redesign the pipeline
"""

from __future__ import annotations

import logging
from typing import Any

from meta_agent.schemas.blueprint import Blueprint, EvaluationConfig
from meta_agent.schemas.state import EvaluationResult, RepairAction

logger = logging.getLogger(__name__)


class RepairLoop:
    """
    Decision logic for the self-improvement loop.

    The repair loop analyses evaluation results and decides the
    appropriate remediation strategy. It considers:
    - Overall score vs threshold
    - Per-dimension scores
    - Number of iterations remaining
    - Specific failure patterns
    """

    def __init__(
        self,
        max_iterations: int = 3,
        accept_threshold: float = 0.7,
        rebuild_threshold: float = 0.4,
    ) -> None:
        self.max_iterations = max_iterations
        self.accept_threshold = accept_threshold
        self.rebuild_threshold = rebuild_threshold

    def decide(
        self,
        evaluation: EvaluationResult,
        iteration: int,
        eval_config: EvaluationConfig | None = None,
    ) -> RepairAction:
        """
        Decide the repair action based on evaluation results.

        Decision matrix:
        ┌──────────────────────────────┬─────────────────────┐
        │ Condition                    │ Action              │
        ├──────────────────────────────┼─────────────────────┤
        │ score >= accept_threshold    │ ACCEPT              │
        │ score < rebuild_threshold    │ REBUILD             │
        │ iteration >= max_iterations  │ ACCEPT (forced)     │
        │ many dimensions failed       │ REBUILD             │
        │ otherwise                    │ REFINE              │
        └──────────────────────────────┴─────────────────────┘
        """
        score = evaluation.overall_score
        threshold = eval_config.overall_threshold if eval_config else self.accept_threshold

        # Rule 1: Good enough → accept
        if score >= threshold:
            logger.info("Score %.2f >= threshold %.2f → ACCEPT", score, threshold)
            return RepairAction.ACCEPT

        # Rule 2: Max iterations reached → forced accept
        if iteration >= self.max_iterations:
            logger.warning(
                "Max iterations (%d) reached with score %.2f → forced ACCEPT",
                self.max_iterations, score,
            )
            return RepairAction.ACCEPT

        # Rule 3: Very low score → rebuild
        if score < self.rebuild_threshold:
            logger.info("Score %.2f < rebuild threshold %.2f → REBUILD", score, self.rebuild_threshold)
            return RepairAction.REBUILD

        # Rule 4: Many dimensions failed → rebuild
        if evaluation.dimension_scores:
            failed_count = sum(1 for d in evaluation.dimension_scores if not d.passed)
            total = len(evaluation.dimension_scores)
            if failed_count > total * 0.6:
                logger.info(
                    "%d/%d dimensions failed → REBUILD",
                    failed_count, total,
                )
                return RepairAction.REBUILD

        # Rule 5: Otherwise → refine
        logger.info("Score %.2f → REFINE", score)
        return RepairAction.REFINE

    def get_repair_guidance(
        self,
        action: RepairAction,
        evaluation: EvaluationResult,
    ) -> dict[str, Any]:
        """
        Generate guidance for the repair action.

        Returns specific instructions for the Meta-Agent on what to fix.
        """
        if action == RepairAction.ACCEPT:
            return {"action": "accept", "message": "Result accepted"}

        guidance: dict[str, Any] = {
            "action": action.value,
            "overall_score": evaluation.overall_score,
            "suggestions": evaluation.suggestions,
        }

        if action == RepairAction.REFINE:
            # Identify worst-performing dimensions
            if evaluation.dimension_scores:
                worst = sorted(
                    evaluation.dimension_scores,
                    key=lambda d: d.score,
                )
                guidance["weak_dimensions"] = [
                    {"name": d.name, "score": d.score, "reasoning": d.reasoning}
                    for d in worst[:3]
                ]
            guidance["strategy"] = (
                "Refine the existing pipeline: lower temperature, increase token limits, "
                "add verification steps, or adjust agent prompts."
            )

        elif action == RepairAction.REBUILD:
            guidance["strategy"] = (
                "The pipeline design is fundamentally flawed. Redesign the architecture "
                "with different agent types, topology, or task decomposition."
            )
            if evaluation.dimension_scores:
                failed = [d for d in evaluation.dimension_scores if not d.passed]
                guidance["failed_dimensions"] = [
                    {"name": d.name, "score": d.score} for d in failed
                ]

        return guidance
