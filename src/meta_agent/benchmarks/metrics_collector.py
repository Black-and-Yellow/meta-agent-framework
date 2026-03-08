"""
Metrics Collector
=================

Collects and aggregates benchmark metrics across runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from meta_agent.benchmarks.runner import BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a benchmark suite run."""
    suite_name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_tasks: int = 0
    successful_tasks: int = 0
    success_rate: float = 0.0
    average_score: float = 0.0
    median_score: float = 0.0
    average_latency_seconds: float = 0.0
    total_tokens: int = 0
    average_repair_iterations: float = 0.0
    p95_latency_seconds: float = 0.0
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)


class MetricsCollector:
    """Collects, aggregates, and reports benchmark metrics."""

    def __init__(self, output_dir: str = "./benchmark_results") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def aggregate(
        self,
        suite_name: str,
        results: list[BenchmarkResult],
    ) -> BenchmarkMetrics:
        """Aggregate results into metrics."""
        if not results:
            return BenchmarkMetrics(suite_name=suite_name)

        scores = sorted([r.score for r in results])
        latencies = sorted([r.elapsed_seconds for r in results])
        total = len(results)
        successes = sum(1 for r in results if r.success)

        metrics = BenchmarkMetrics(
            suite_name=suite_name,
            total_tasks=total,
            successful_tasks=successes,
            success_rate=successes / total,
            average_score=sum(scores) / total,
            median_score=scores[total // 2],
            average_latency_seconds=sum(latencies) / total,
            total_tokens=sum(r.total_tokens for r in results),
            average_repair_iterations=sum(r.repair_iterations for r in results) / total,
            p95_latency_seconds=latencies[int(total * 0.95)] if total > 1 else latencies[0],
        )

        return metrics

    def save(self, metrics: BenchmarkMetrics) -> Path:
        """Save metrics to a JSON file."""
        filename = f"{metrics.suite_name}_{metrics.timestamp[:10]}.json"
        filepath = self.output_dir / filename
        filepath.write_text(json.dumps(asdict(metrics), indent=2, default=str))
        logger.info("Metrics saved to %s", filepath)
        return filepath
