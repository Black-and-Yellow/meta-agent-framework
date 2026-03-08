"""
Benchmark Runner
================

Loads benchmark suites, executes tasks through the Meta-Agent,
and collects metrics for evaluation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from meta_agent.core.meta_agent import MetaAgent

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    task_id: str
    description: str
    expected_output: str = ""
    category: str = "general"
    difficulty: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark task execution."""
    task_id: str
    success: bool
    score: float
    elapsed_seconds: float
    total_tokens: int = 0
    repair_iterations: int = 0
    error: str | None = None
    output: Any = None


class BenchmarkRunner:
    """
    Runs benchmark suites against the Meta-Agent.

    Usage:
        runner = BenchmarkRunner()
        results = await runner.run_suite("gaia", tasks)
        runner.print_report(results)
    """

    def __init__(self, meta_agent: MetaAgent | None = None) -> None:
        self.meta_agent = meta_agent or MetaAgent()
        self.results: list[BenchmarkResult] = []

    async def run_suite(
        self,
        suite_name: str,
        tasks: list[BenchmarkTask],
        max_concurrent: int = 1,
    ) -> list[BenchmarkResult]:
        """Run a full benchmark suite."""
        logger.info("Running benchmark suite '%s' (%d tasks)", suite_name, len(tasks))
        results = []

        for i, task in enumerate(tasks):
            logger.info("Benchmark %d/%d: %s", i + 1, len(tasks), task.task_id)
            result = await self._run_single(task)
            results.append(result)

        self.results.extend(results)
        return results

    async def _run_single(self, task: BenchmarkTask) -> BenchmarkResult:
        """Execute a single benchmark task."""
        start = time.time()
        try:
            output = await self.meta_agent.solve(task.description)
            elapsed = time.time() - start

            evaluations = output.get("evaluations", [])
            final_score = evaluations[-1].get("overall_score", 0.0) if evaluations else 0.0

            return BenchmarkResult(
                task_id=task.task_id,
                success=True,
                score=final_score,
                elapsed_seconds=elapsed,
                repair_iterations=len(evaluations),
                output=output.get("result"),
            )
        except Exception as e:
            elapsed = time.time() - start
            logger.error("Benchmark task %s failed: %s", task.task_id, e)
            return BenchmarkResult(
                task_id=task.task_id,
                success=False,
                score=0.0,
                elapsed_seconds=elapsed,
                error=str(e),
            )

    def print_report(self, results: list[BenchmarkResult] | None = None) -> str:
        """Generate a human-readable benchmark report."""
        results = results or self.results
        if not results:
            return "No results to report"

        total = len(results)
        successes = sum(1 for r in results if r.success)
        avg_score = sum(r.score for r in results) / total
        avg_time = sum(r.elapsed_seconds for r in results) / total
        avg_iterations = sum(r.repair_iterations for r in results) / total

        report = [
            "═" * 60,
            "BENCHMARK REPORT",
            "═" * 60,
            f"Total tasks:        {total}",
            f"Successful:         {successes}/{total} ({successes / total * 100:.0f}%)",
            f"Average score:      {avg_score:.3f}",
            f"Average time:       {avg_time:.1f}s",
            f"Average iterations: {avg_iterations:.1f}",
            "─" * 60,
        ]

        for r in results:
            status = "✓" if r.success else "✗"
            report.append(
                f"  {status} {r.task_id}: score={r.score:.2f} "
                f"time={r.elapsed_seconds:.1f}s iters={r.repair_iterations}"
            )

        report.append("═" * 60)
        report_str = "\n".join(report)
        logger.info("\n%s", report_str)
        return report_str
