"""
Agent Factory
=============

The factory keeps the system extensible.

It reads an AgentConfig and produces a living agent instance with the right
type, model, prompt, tools, and behaviour. New agent types can be registered
at runtime without modifying existing code.
"""

from __future__ import annotations

import logging
from typing import Any

from meta_agent.agents.base_agent import BaseAgent
from meta_agent.schemas.blueprint import AgentConfig, AgentType

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Registry-pattern factory for creating agent instances from config.

    Usage:
        factory = AgentFactory()
        factory.register("custom_type", MyCustomAgent)
        agent = factory.create(agent_config)
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[BaseAgent]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all built-in agent types."""
        from meta_agent.agents.specialized.research_agent import ResearchAgent
        from meta_agent.agents.specialized.planning_agent import PlanningAgent
        from meta_agent.agents.specialized.coding_agent import CodingAgent
        from meta_agent.agents.specialized.data_analysis_agent import DataAnalysisAgent
        from meta_agent.agents.specialized.verification_agent import VerificationAgent
        from meta_agent.agents.specialized.summarization_agent import SummarizationAgent
        from meta_agent.agents.specialized.critic_agent import CriticAgent

        self._registry = {
            AgentType.RESEARCH.value: ResearchAgent,
            AgentType.PLANNING.value: PlanningAgent,
            AgentType.CODING.value: CodingAgent,
            AgentType.DATA_ANALYSIS.value: DataAnalysisAgent,
            AgentType.VERIFICATION.value: VerificationAgent,
            AgentType.SUMMARIZATION.value: SummarizationAgent,
            AgentType.CRITIC.value: CriticAgent,
        }

    def register(self, agent_type: str, agent_class: type[BaseAgent]) -> None:
        """Register a new agent type or override an existing one."""
        logger.info("Registering agent type: %s → %s", agent_type, agent_class.__name__)
        self._registry[agent_type] = agent_class

    def create(self, config: AgentConfig, **kwargs: Any) -> BaseAgent:
        """
        Create an agent instance from its configuration.

        Args:
            config: The AgentConfig from the blueprint.
            **kwargs: Additional kwargs passed to the agent constructor.

        Returns:
            A fully configured agent instance.

        Raises:
            ValueError: If the agent type is not registered.
        """
        agent_type = config.agent_type.value
        agent_class = self._registry.get(agent_type)

        if agent_class is None:
            if config.agent_type == AgentType.CUSTOM:
                if config.name and "extractor" in config.name.lower():
                    from meta_agent.agents.specialized.extractor_agent import ExtractorAgent
                    agent_class = ExtractorAgent
                    logger.info("Mapped CUSTOM agent type to ExtractorAgent based on agent name.")
                else:
                    from meta_agent.agents.specialized.research_agent import ResearchAgent
                    agent_class = ResearchAgent
                    logger.warning(
                        "Custom agent type not registered; falling back to ResearchAgent"
                    )
            else:
                raise ValueError(
                    f"Unknown agent type: {agent_type}. "
                    f"Registered types: {list(self._registry.keys())}"
                )

        agent = agent_class(config=config, **kwargs)
        logger.info(
            "Created agent [%s] %s (type=%s, model=%s)",
            agent.agent_id,
            agent.name,
            agent_type,
            config.model,
        )
        return agent

    def list_types(self) -> list[str]:
        """Return all registered agent type names."""
        return list(self._registry.keys())

    def get_class(self, agent_type: str) -> type[BaseAgent] | None:
        """Get the class for a registered agent type."""
        return self._registry.get(agent_type)
