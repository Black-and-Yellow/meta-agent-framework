"""
Long-Term Memory
================

Long-term memory remembers past successes and failures.
Each entry is a semantic chunk that can be retrieved by similarity.

Backed by ChromaDB for persistent vector storage.
Falls back to in-memory FAISS-like search for development.
"""

from __future__ import annotations

import logging
from typing import Any

from meta_agent.schemas.state import MemoryEntry

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    ChromaDB-backed vector store for semantic search over past knowledge.

    Stores:
    - Past execution summaries
    - Successful blueprint patterns
    - Error logs and lessons learned
    - Agent performance history
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "meta_agent_memory",
    ) -> None:
        self._collection: Any | None = None
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._in_memory_entries: list[MemoryEntry] = []

        try:
            import chromadb
            client = chromadb.HttpClient(host=host, port=port)
            self._collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Meta-Agent long-term memory"},
            )
            logger.info("Long-term memory: ChromaDB at %s:%d", host, port)
        except Exception as e:
            logger.warning(
                "ChromaDB unavailable, using in-memory store: %s", e
            )

    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry and return its ID."""
        if self._collection:
            try:
                self._collection.add(
                    ids=[entry.entry_id],
                    documents=[entry.content],
                    metadatas=[{
                        "entry_type": entry.entry_type,
                        "source_task_id": entry.source_task_id,
                        "source_agent_id": entry.source_agent_id,
                        "created_at": entry.created_at.isoformat(),
                    }],
                )
                return entry.entry_id
            except Exception as e:
                logger.warning("ChromaDB store failed: %s", e)

        self._in_memory_entries.append(entry)
        return entry.entry_id

    async def search(
        self,
        query: str,
        top_k: int = 5,
        entry_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for semantically similar entries."""
        if self._collection:
            try:
                where_filter = {"entry_type": entry_type} if entry_type else None
                results = self._collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=where_filter,
                )
                formatted = []
                for i, doc in enumerate(results.get("documents", [[]])[0]):
                    formatted.append({
                        "content": doc,
                        "metadata": (results.get("metadatas", [[]])[0][i]
                                     if results.get("metadatas") else {}),
                        "distance": (results.get("distances", [[]])[0][i]
                                     if results.get("distances") else None),
                    })
                return formatted
            except Exception as e:
                logger.warning("ChromaDB search failed: %s", e)

        # Fallback: simple substring matching on in-memory entries
        matches = []
        query_lower = query.lower()
        for entry in self._in_memory_entries:
            if query_lower in entry.content.lower():
                matches.append({
                    "content": entry.content,
                    "metadata": {"entry_type": entry.entry_type},
                    "distance": 0.5,
                })
            if len(matches) >= top_k:
                break
        return matches

    async def store_blueprint_pattern(
        self,
        task_description: str,
        blueprint_summary: str,
        score: float,
    ) -> str:
        """Store a successful blueprint pattern for future retrieval."""
        entry = MemoryEntry(
            content=(
                f"Task: {task_description}\n"
                f"Blueprint: {blueprint_summary}\n"
                f"Score: {score}"
            ),
            entry_type="blueprint",
            metadata={"score": score},
        )
        return await self.store(entry)

    async def store_lesson(
        self,
        task_description: str,
        lesson: str,
        source_agent_id: str = "",
    ) -> str:
        """Store a lesson learned from a failure or success."""
        entry = MemoryEntry(
            content=f"Lesson from task '{task_description[:100]}': {lesson}",
            entry_type="lesson",
            source_agent_id=source_agent_id,
        )
        return await self.store(entry)
