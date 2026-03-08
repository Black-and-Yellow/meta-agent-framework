"""
Distributed Tracing
===================

Complex agent systems fail in strange ways.
Tracing every step is the only way to debug them.

OpenTelemetry-based tracing with custom span attributes
for agent reasoning steps, tool invocations, and graph traversal.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def setup_tracing(
    service_name: str = "meta-agent",
    otlp_endpoint: str = "http://localhost:4317",
) -> Any:
    """
    Configure OpenTelemetry tracing with OTLP export.

    Returns the tracer instance for manual span creation.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(service_name)

        logger.info("OpenTelemetry tracing configured (endpoint=%s)", otlp_endpoint)
        return tracer

    except ImportError:
        logger.warning("OpenTelemetry not available — tracing disabled")
        return None
    except Exception as e:
        logger.warning("Tracing setup failed: %s", e)
        return None


def create_agent_span(
    tracer: Any,
    agent_id: str,
    agent_name: str,
    operation: str = "execute",
) -> Any:
    """Create a tracing span for an agent execution."""
    if tracer is None:
        return None

    from opentelemetry import trace

    span = tracer.start_span(
        name=f"agent.{operation}",
        attributes={
            "agent.id": agent_id,
            "agent.name": agent_name,
            "agent.operation": operation,
        },
    )
    return span


def create_tool_span(
    tracer: Any,
    tool_name: str,
    agent_id: str,
) -> Any:
    """Create a tracing span for a tool invocation."""
    if tracer is None:
        return None

    span = tracer.start_span(
        name=f"tool.{tool_name}",
        attributes={
            "tool.name": tool_name,
            "tool.invoker": agent_id,
        },
    )
    return span


class LangSmithCallbackHandler:
    """
    Callback handler for LangSmith integration.

    Sends agent reasoning traces to LangSmith for
    visualization and debugging.
    """

    def __init__(self, api_key: str = "", project: str = "meta-agent") -> None:
        self._enabled = False
        self._client: Any = None

        if api_key:
            try:
                from langsmith import Client
                self._client = Client(api_key=api_key)
                self._enabled = True
                logger.info("LangSmith integration enabled (project=%s)", project)
            except ImportError:
                logger.info("LangSmith SDK not installed — integration disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def on_agent_start(self, agent_id: str, input_data: dict) -> None:
        """Log agent start to LangSmith."""
        if not self._enabled:
            return
        logger.debug("LangSmith: agent %s started", agent_id)

    def on_agent_end(self, agent_id: str, output_data: dict) -> None:
        """Log agent completion to LangSmith."""
        if not self._enabled:
            return
        logger.debug("LangSmith: agent %s completed", agent_id)
