"""Vector Retrieval Tool — semantic search over long-term memory."""

from __future__ import annotations

import logging
from typing import Any

from meta_agent.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class VectorRetrievalTool(BaseTool):
    """
    Search long-term vector memory for relevant past knowledge.

    Integrates with ChromaDB or FAISS to find semantically similar
    entries from past executions, learned patterns, and stored knowledge.
    """

    def __init__(self) -> None:
        super().__init__(
            name="vector_retrieval",
            description="Search long-term vector memory for relevant past knowledge",
            timeout_seconds=10,
            max_retries=2,
            permission_scope="read",
        )

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        if "query" not in input_data:
            return "Missing required field: 'query'"
        return None

    async def execute(self, input_data: dict[str, Any]) -> Any:
        query = input_data["query"]
        top_k = input_data.get("top_k", 5)
        collection = input_data.get("collection", "meta_agent_memory")
        logger.info("Vector search: '%s' (top_k=%d, collection=%s)", query[:60], top_k, collection)

        # Production: integrate with ChromaDB
        try:
            from meta_agent.memory.long_term import LongTermMemory
            memory = LongTermMemory()
            results = await memory.search(query, top_k=top_k)
            return {"query": query, "results": results, "count": len(results)}
        except Exception as e:
            logger.warning("Vector retrieval fallback: %s", e)
            return {"query": query, "results": [], "count": 0}

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Semantic search query"},
                    "top_k": {"type": "integer", "default": 5},
                    "collection": {"type": "string", "default": "meta_agent_memory"},
                },
                "required": ["query"],
            },
        }
