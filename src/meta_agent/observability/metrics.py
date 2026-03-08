"""
Prometheus Metrics
==================

Agent execution time, token usage, tool call counts,
graph execution duration, and error rates.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge, Info

    # ── Counters ────────────────────────────────────────────────────
    TASKS_SUBMITTED = Counter(
        "meta_agent_tasks_submitted_total",
        "Total number of tasks submitted to the Meta-Agent",
    )
    TASKS_COMPLETED = Counter(
        "meta_agent_tasks_completed_total",
        "Total tasks completed successfully",
        ["status"],  # completed, failed
    )
    AGENT_EXECUTIONS = Counter(
        "meta_agent_agent_executions_total",
        "Total agent executions",
        ["agent_type", "status"],
    )
    TOOL_INVOCATIONS = Counter(
        "meta_agent_tool_invocations_total",
        "Total tool invocations",
        ["tool_name", "status"],
    )
    REPAIR_ITERATIONS = Counter(
        "meta_agent_repair_iterations_total",
        "Total repair iterations performed",
        ["action"],  # accept, refine, rebuild
    )

    # ── Histograms ──────────────────────────────────────────────────
    TASK_DURATION = Histogram(
        "meta_agent_task_duration_seconds",
        "Task execution duration",
        buckets=[1, 5, 10, 30, 60, 120, 300, 600],
    )
    AGENT_DURATION = Histogram(
        "meta_agent_agent_duration_seconds",
        "Individual agent execution duration",
        ["agent_type"],
        buckets=[0.5, 1, 2, 5, 10, 30, 60],
    )
    TOOL_DURATION = Histogram(
        "meta_agent_tool_duration_seconds",
        "Tool invocation duration",
        ["tool_name"],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
    )
    EVALUATION_SCORE = Histogram(
        "meta_agent_evaluation_score",
        "Evaluation scores distribution",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )

    # ── Gauges ──────────────────────────────────────────────────────
    ACTIVE_TASKS = Gauge(
        "meta_agent_active_tasks",
        "Number of currently executing tasks",
    )
    TOKEN_USAGE = Counter(
        "meta_agent_tokens_used_total",
        "Total tokens consumed",
        ["type"],  # prompt, completion
    )

    # ── Info ────────────────────────────────────────────────────────
    SYSTEM_INFO = Info(
        "meta_agent",
        "Meta-Agent system information",
    )

    METRICS_AVAILABLE = True
    logger.info("Prometheus metrics initialized")

except ImportError:
    METRICS_AVAILABLE = False
    logger.info("prometheus_client not available — metrics disabled")

    # Stub classes for when prometheus is not available
    class _NoOp:
        def labels(self, *a, **kw): return self
        def inc(self, *a, **kw): pass
        def dec(self, *a, **kw): pass
        def set(self, *a, **kw): pass
        def observe(self, *a, **kw): pass
        def info(self, *a, **kw): pass

    TASKS_SUBMITTED = _NoOp()
    TASKS_COMPLETED = _NoOp()
    AGENT_EXECUTIONS = _NoOp()
    TOOL_INVOCATIONS = _NoOp()
    REPAIR_ITERATIONS = _NoOp()
    TASK_DURATION = _NoOp()
    AGENT_DURATION = _NoOp()
    TOOL_DURATION = _NoOp()
    EVALUATION_SCORE = _NoOp()
    ACTIVE_TASKS = _NoOp()
    TOKEN_USAGE = _NoOp()
    SYSTEM_INFO = _NoOp()
