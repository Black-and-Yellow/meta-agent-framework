"""Database Query Tool — read-only SQL query execution via SQLite."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from meta_agent.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class DatabaseQueryTool(BaseTool):
    """
    Execute read-only SQL queries against a SQLite database.

    Security:
    - Only SELECT statements are allowed
    - Query timeout is enforced
    - Results are truncated to prevent memory exhaustion
    - Dangerous keywords are blocked

    The database path is read from the ``DATABASE_PATH`` setting.
    If no database path is configured, queries run against a transient
    in-memory database.
    """

    BLOCKED_KEYWORDS = {
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
        "CREATE", "TRUNCATE", "EXEC", "ATTACH", "DETACH",
    }
    MAX_ROWS = 500

    def __init__(self) -> None:
        super().__init__(
            name="database_query",
            description="Execute read-only SQL queries against the database",
            timeout_seconds=30,
            max_retries=2,
            permission_scope="read",
        )
        self._db_path: str | None = None
        try:
            from meta_agent.config import get_settings
            settings = get_settings()
            self._db_path = getattr(settings, "database_path", None) or None
        except Exception:
            pass

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        if "query" not in input_data:
            return "Missing required field: 'query'"

        query_upper = input_data["query"].upper().strip()
        for keyword in self.BLOCKED_KEYWORDS:
            if keyword in query_upper:
                return f"Write operation '{keyword}' is not allowed (read-only)"

        if not query_upper.startswith("SELECT"):
            return "Only SELECT queries are allowed"

        return None

    async def execute(self, input_data: dict[str, Any]) -> Any:
        query = input_data["query"]
        db_path = input_data.get("database", self._db_path)
        logger.info("Executing query: %s", query[:100])

        if db_path and not Path(db_path).exists():
            return {
                "query": query,
                "rows": [],
                "columns": [],
                "row_count": 0,
                "error": f"Database file not found: {db_path}",
            }

        connection_target = db_path if db_path else ":memory:"

        try:
            conn = sqlite3.connect(connection_target, timeout=self.timeout_seconds)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            rows_raw = cursor.fetchmany(self.MAX_ROWS)

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = [dict(row) for row in rows_raw]

            conn.close()

            return {
                "query": query,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "truncated": len(rows_raw) >= self.MAX_ROWS,
            }
        except sqlite3.Error as e:
            logger.error("SQLite error: %s", e)
            return {
                "query": query,
                "rows": [],
                "columns": [],
                "row_count": 0,
                "error": str(e),
            }

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Read-only SQL query"},
                    "database": {
                        "type": "string",
                        "description": "Optional path to a SQLite database file",
                    },
                },
                "required": ["query"],
            },
        }
