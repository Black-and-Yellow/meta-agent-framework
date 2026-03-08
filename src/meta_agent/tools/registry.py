"""
Tool Registry
=============

Central registry for all tools available to agents.

The registry:
- Discovers and registers tools at startup
- Validates tool permissions before invocation
- Provides tool schemas to agents for function-calling
- Enforces security boundaries
"""

from __future__ import annotations

import logging
from typing import Any

from meta_agent.tools.base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Singleton-pattern registry for tool discovery and invocation.

    Usage:
        registry = ToolRegistry()
        registry.register(WebSearchTool())
        result = await registry.invoke("web_search", {"query": "..."})
    """

    _instance: ToolRegistry | None = None

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            self._tools: dict[str, BaseTool] = {}
            self._initialized = True

    def register(self, tool: BaseTool) -> None:
        """Register a tool by its name."""
        logger.info("Registering tool: %s (%s)", tool.name, tool.description[:60])
        self._tools[tool.name.lower().replace(" ", "_")] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        normalized = name.lower().replace(" ", "_")
        if normalized in self._tools:
            del self._tools[normalized]
            logger.info("Unregistered tool: %s", name)

    async def invoke(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        required_scope: str = "read",
    ) -> Any:
        """
        Invoke a tool by name with permission checking.

        Args:
            tool_name: The registered tool name.
            input_data: Input parameters for the tool.
            required_scope: Minimum permission scope required.

        Returns:
            The tool's output data.

        Raises:
            ValueError: If tool not found.
            PermissionError: If scope is insufficient.
        """
        normalized = tool_name.lower().replace(" ", "_")
        tool = self._tools.get(normalized)

        if tool is None:
            raise ValueError(
                f"Tool '{tool_name}' not registered. "
                f"Available: {list(self._tools.keys())}"
            )

        # Permission check
        scope_levels = {"read": 0, "write": 1, "execute": 2}
        tool_level = scope_levels.get(tool.permission_scope, 0)
        required_level = scope_levels.get(required_scope, 0)
        if tool_level < required_level:
            raise PermissionError(
                f"Tool '{tool_name}' has scope '{tool.permission_scope}' "
                f"but '{required_scope}' is required"
            )

        result: ToolResult = await tool(input_data)

        if not result.success:
            logger.error("Tool %s failed: %s", tool_name, result.error)
            raise RuntimeError(f"Tool '{tool_name}' failed: {result.error}")

        return result.data

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool instance by name."""
        return self._tools.get(name.lower().replace(" ", "_"))

    def list_tools(self) -> list[dict[str, Any]]:
        """Return schemas for all registered tools."""
        return [tool.get_schema() for tool in self._tools.values()]

    def get_tools_for_agent(self, tool_names: list[str]) -> list[dict[str, Any]]:
        """Get schemas for specific tools (for agent function-calling setup)."""
        schemas = []
        for name in tool_names:
            tool = self.get_tool(name)
            if tool:
                schemas.append(tool.get_schema())
        return schemas

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (primarily for testing)."""
        cls._instance = None

    def register_defaults(self) -> None:
        """Register all built-in tool implementations."""
        from meta_agent.tools.implementations.web_search import WebSearchTool
        from meta_agent.tools.implementations.code_executor import CodeExecutorTool
        from meta_agent.tools.implementations.database_query import DatabaseQueryTool
        from meta_agent.tools.implementations.vector_retrieval import VectorRetrievalTool
        from meta_agent.tools.implementations.api_caller import APICallerTool
        from meta_agent.tools.implementations.file_reader import FileReaderTool

        self.register(WebSearchTool())
        self.register(CodeExecutorTool())
        self.register(DatabaseQueryTool())
        self.register(VectorRetrievalTool())
        self.register(APICallerTool())
        self.register(FileReaderTool())

        logger.info("Registered %d default tools", len(self._tools))
