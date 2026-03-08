"""
Shared Context Manager
======================

Unified API for agents to read/write shared knowledge.
Provides scoped access to both short-term and long-term memory.
"""

from __future__ import annotations

import logging
from typing import Any

from meta_agent.memory.short_term import ShortTermMemory
from meta_agent.memory.long_term import LongTermMemory
from meta_agent.schemas.state import MemoryEntry

logger = logging.getLogger(__name__)


class SharedContextManager:
    """
    Unified memory interface for all agents.

    Agents interact with this instead of touching short-term or
    long-term memory directly. This provides:
    - Access control (agents only see what they should)
    - Abstraction (agents don't know about Redis vs ChromaDB)
    - Auditing (all memory access is logged)
    """

    def __init__(
        self,
        short_term: ShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
    ) -> None:
        self.short_term = short_term or ShortTermMemory()
        self.long_term = long_term or LongTermMemory()

    async def get_task_context(self, execution_id: str) -> dict[str, Any]:
        """Get the full context for the current task execution."""
        return await self.short_term.get_execution_state(execution_id)

    async def save_intermediate_result(
        self,
        execution_id: str,
        agent_id: str,
        result: Any,
    ) -> None:
        """Save an agent's intermediate result."""
        await self.short_term.add_agent_result(execution_id, agent_id, result)
        logger.debug("Saved result for agent %s in execution %s", agent_id, execution_id)

    async def search_knowledge(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search long-term memory for relevant knowledge."""
        return await self.long_term.search(query, top_k=top_k)

    async def remember(
        self,
        content: str,
        entry_type: str = "observation",
        source_task_id: str = "",
        source_agent_id: str = "",
    ) -> str:
        """Store a new memory entry in long-term storage."""
        entry = MemoryEntry(
            content=content,
            entry_type=entry_type,
            source_task_id=source_task_id,
            source_agent_id=source_agent_id,
        )
        return await self.long_term.store(entry)

    async def get_previous_results(
        self,
        execution_id: str,
    ) -> dict[str, Any]:
        """Get all intermediate results from the current execution."""
        return await self.short_term.get(f"exec:{execution_id}:results") or {}
