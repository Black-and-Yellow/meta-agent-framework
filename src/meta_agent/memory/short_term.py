"""
Short-Term Memory
=================

Short-term memory holds the current conversation.
It stores the transient state of a single pipeline execution.

In production, this is backed by Redis for multi-worker access.
Falls back to in-process dict for development.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    In-process or Redis-backed store for current execution state.

    Stores:
    - Agent intermediate results
    - Inter-agent messages
    - Current pipeline status
    - Temporary data that doesn't need to survive restarts
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis_client: Any | None = None
        self._local_store: dict[str, Any] = {}
        self._redis_url = redis_url

        if redis_url:
            try:
                import redis
                self._redis_client = redis.from_url(redis_url, decode_responses=True)
                logger.info("Short-term memory: Redis at %s", redis_url)
            except Exception as e:
                logger.warning("Redis unavailable, using in-memory store: %s", e)

    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Store a value with optional TTL (seconds)."""
        serialized = json.dumps(value, default=str)

        if self._redis_client:
            try:
                self._redis_client.setex(key, ttl, serialized)
                return
            except Exception as e:
                logger.warning("Redis set failed, using local: %s", e)

        self._local_store[key] = serialized

    async def get(self, key: str) -> Any | None:
        """Retrieve a value by key."""
        if self._redis_client:
            try:
                raw = self._redis_client.get(key)
                return json.loads(raw) if raw else None
            except Exception as e:
                logger.warning("Redis get failed, using local: %s", e)

        raw = self._local_store.get(key)
        return json.loads(raw) if raw else None

    async def delete(self, key: str) -> None:
        """Delete a key."""
        if self._redis_client:
            try:
                self._redis_client.delete(key)
                return
            except Exception as e:
                logger.warning("Redis delete failed: %s", e)

        self._local_store.pop(key, None)

    async def get_execution_state(self, execution_id: str) -> dict[str, Any]:
        """Get the full state for an execution."""
        return await self.get(f"exec:{execution_id}") or {}

    async def save_execution_state(
        self, execution_id: str, state: dict[str, Any]
    ) -> None:
        """Save execution state."""
        await self.set(f"exec:{execution_id}", state)

    async def add_agent_result(
        self, execution_id: str, agent_id: str, result: Any
    ) -> None:
        """Add an intermediate result from an agent."""
        key = f"exec:{execution_id}:results"
        results = await self.get(key) or {}
        results[agent_id] = result
        await self.set(key, results)

    async def clear_execution(self, execution_id: str) -> None:
        """Clear all data for an execution."""
        await self.delete(f"exec:{execution_id}")
        await self.delete(f"exec:{execution_id}:results")
        await self.delete(f"exec:{execution_id}:messages")
