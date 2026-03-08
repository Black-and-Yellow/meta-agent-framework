"""
Result Evaluator
================

Solving a task is not enough.
The system must judge its own work.
This module turns the AI into its own reviewer.

Multi-dimensional evaluation using LLM-as-judge, rule-based scoring,
or composite strategies. Produces an EvaluationResult with per-dimension
scores, an overall score, and a recommended repair action.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from meta_agent.config import get_settings
from meta_agent.schemas.blueprint import Blueprint, EvaluationConfig, EvaluationStrategy
from meta_agent.schemas.state import (
    DimensionScore,
    EvaluationResult,
    ExecutionContext,
    RepairAction,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Judge Prompt
# ═══════════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator. You must judge the quality of an AI system's output
against the original task.

Evaluate on these dimensions:
{dimensions}

For each dimension, provide:
- score: float from 0.0 to 1.0
- reasoning: brief explanation

Then provide:
- overall_score: weighted average
- passed: true if overall_score >= {threshold}
- recommended_action: one of "accept" (good enough), "refine" (minor fixes), "rebuild" (major redesign needed)
- suggestions: list of specific improvement suggestions

Respond with ONLY valid JSON:
{{
  "dimension_scores": [
    {{"name": "...", "score": 0.0-1.0, "reasoning": "..."}}
  ],
  "overall_score": 0.0-1.0,
  "passed": true/false,
  "recommended_action": "accept|refine|rebuild",
  "reasoning": "...",
  "suggestions": ["..."]
}}
"""

JUDGE_USER_PROMPT = """\
ORIGINAL TASK:
{task}

AI SYSTEM OUTPUT:
{result}

Evaluate the quality of this output against the original task.
"""


class ResultEvaluator:
    """
    Evaluates pipeline output against the original task.

    Supports multiple strategies:
    - LLM_JUDGE: Uses a language model to score output quality
    - RULE_BASED: Applies programmatic checks
    - COMPOSITE: Combines multiple evaluation methods
    """

    def __init__(self, llm_client: Any | None = None) -> None:
        self._llm_client = llm_client

    @property
    def llm_client(self) -> Any:
        if self._llm_client is None:
            from langchain_openai import ChatOpenAI
            settings = get_settings()
            kwargs: dict = dict(
                model=settings.meta_agent_model,
                temperature=0.1,  # Low temp for consistent evaluation
                max_tokens=2048,
                api_key=settings.openai_api_key,
            )
            if settings.openai_api_base:
                kwargs["base_url"] = settings.openai_api_base
            self._llm_client = ChatOpenAI(**kwargs)
        return self._llm_client

    async def evaluate(
        self,
        task: str,
        result: Any,
        blueprint: Blueprint,
        context: ExecutionContext | None = None,
    ) -> EvaluationResult:
        """
        Evaluate the pipeline's output.

        Args:
            task: Original user task.
            result: The pipeline's final output.
            blueprint: The blueprint that generated the output.
            context: Full execution context (for resource-aware evaluation).

        Returns:
            EvaluationResult with scores and recommended action.
        """
        eval_config = blueprint.evaluation

        if eval_config.strategy == EvaluationStrategy.LLM_JUDGE:
            return await self._llm_judge(task, result, eval_config)
        elif eval_config.strategy == EvaluationStrategy.RULE_BASED:
            return self._rule_based(task, result, eval_config)
        else:
            return await self._llm_judge(task, result, eval_config)

    async def _llm_judge(
        self,
        task: str,
        result: Any,
        config: EvaluationConfig,
    ) -> EvaluationResult:
        """Use LLM-as-judge for evaluation."""
        dimensions_str = "\n".join(
            f"- {d.name} (weight={d.weight}): {d.description}"
            for d in config.dimensions
        )

        system_prompt = JUDGE_SYSTEM_PROMPT.format(
            dimensions=dimensions_str,
            threshold=config.overall_threshold,
        )

        result_str = json.dumps(result, default=str) if isinstance(result, dict) else str(result)
        user_prompt = JUDGE_USER_PROMPT.format(
            task=task,
            result=result_str[:8000],  # Truncate for token limits
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self.llm_client.ainvoke(messages)
            return self._parse_judge_response(response.content, config)
        except Exception as e:
            logger.error("LLM judge failed: %s", e)
            # Return a neutral evaluation on failure
            return EvaluationResult(
                overall_score=0.5,
                passed=False,
                recommended_action=RepairAction.REFINE,
                reasoning=f"Evaluation failed: {e}",
            )

    def _parse_judge_response(
        self,
        raw: str,
        config: EvaluationConfig,
    ) -> EvaluationResult:
        """Parse the LLM judge's JSON response."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse judge response: %s", e)
            return EvaluationResult(
                overall_score=0.5,
                passed=False,
                recommended_action=RepairAction.REFINE,
                reasoning=f"Could not parse evaluation: {e}",
            )

        # Build dimension scores
        dim_scores = []
        for dim_data in data.get("dimension_scores", []):
            score = float(dim_data.get("score", 0.5))
            dim_config = next(
                (d for d in config.dimensions if d.name == dim_data.get("name")),
                None,
            )
            dim_scores.append(DimensionScore(
                name=dim_data.get("name", "unknown"),
                score=score,
                reasoning=dim_data.get("reasoning", ""),
                passed=score >= (dim_config.threshold if dim_config else 0.5),
            ))

        overall = float(data.get("overall_score", 0.5))
        passed = overall >= config.overall_threshold

        # Force action to ACCEPT when score meets threshold.
        # The LLM judge sometimes returns "refine" even on passing scores.
        if passed:
            action = RepairAction.ACCEPT
        else:
            action_str = data.get("recommended_action", "refine")
            try:
                action = RepairAction(action_str)
            except ValueError:
                action = RepairAction.REFINE

        return EvaluationResult(
            dimension_scores=dim_scores,
            overall_score=overall,
            passed=passed,
            recommended_action=action,
            reasoning=data.get("reasoning", ""),
            suggestions=data.get("suggestions", []),
        )

    def _rule_based(
        self,
        task: str,
        result: Any,
        config: EvaluationConfig,
    ) -> EvaluationResult:
        """Simple rule-based evaluation as a fallback."""
        result_str = str(result)

        # Basic heuristics
        length_score = min(1.0, len(result_str) / 500)
        has_content = 1.0 if len(result_str) > 50 else 0.3
        no_error = 0.0 if "error" in result_str.lower() else 1.0

        overall = (length_score * 0.3 + has_content * 0.3 + no_error * 0.4)

        return EvaluationResult(
            dimension_scores=[
                DimensionScore(name="length", score=length_score, passed=length_score > 0.3),
                DimensionScore(name="has_content", score=has_content, passed=has_content > 0.5),
                DimensionScore(name="no_error", score=no_error, passed=no_error > 0.5),
            ],
            overall_score=overall,
            passed=overall >= config.overall_threshold,
            recommended_action=(
                RepairAction.ACCEPT if overall >= config.overall_threshold
                else RepairAction.REFINE
            ),
            reasoning=f"Rule-based evaluation: score={overall:.2f}",
        )
