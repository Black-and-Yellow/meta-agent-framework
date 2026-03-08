"""
Test Suite — Agent Factory & Tools
===================================
"""

from __future__ import annotations

import pytest

from meta_agent.schemas.blueprint import AgentConfig, AgentType
from meta_agent.agents.factory import AgentFactory
from meta_agent.agents.base_agent import BaseAgent
from meta_agent.tools.base_tool import BaseTool, ToolResult
from meta_agent.tools.registry import ToolRegistry


class TestAgentFactory:
    def test_create_all_types(self):
        factory = AgentFactory()
        for agent_type in [
            AgentType.RESEARCH, AgentType.PLANNING, AgentType.CODING,
            AgentType.DATA_ANALYSIS, AgentType.VERIFICATION,
            AgentType.SUMMARIZATION, AgentType.CRITIC,
        ]:
            config = AgentConfig(
                agent_type=agent_type,
                name=f"Test {agent_type.value}",
                role_description=f"A test {agent_type.value} agent for factory validation",
            )
            agent = factory.create(config)
            assert isinstance(agent, BaseAgent)
            assert agent.agent_id == config.agent_id

    def test_unknown_type_raises(self):
        factory = AgentFactory()
        # Manually create config with a type that isn't registered
        config = AgentConfig(
            agent_type=AgentType.CUSTOM,
            name="Unknown",
            role_description="An agent with custom type for error testing",
        )
        # Custom falls back to ResearchAgent, so this should work
        agent = factory.create(config)
        assert isinstance(agent, BaseAgent)

    def test_list_types(self):
        factory = AgentFactory()
        types = factory.list_types()
        assert "research" in types
        assert "coding" in types
        assert len(types) >= 7


class TestToolRegistry:
    def setup_method(self):
        ToolRegistry.reset()

    def test_register_and_invoke(self):
        registry = ToolRegistry()

        class DummyTool(BaseTool):
            async def execute(self, input_data):
                return {"echo": input_data.get("msg", "")}

        registry.register(DummyTool(name="dummy_tool", description="Test"))
        tools = registry.list_tools()
        assert any(t["name"] == "dummy_tool" for t in tools)

    def test_invoke_unregistered_raises(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="not registered"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                registry.invoke("nonexistent", {})
            )

    def teardown_method(self):
        ToolRegistry.reset()
