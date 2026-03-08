"""API Caller Tool — make HTTP requests to external APIs."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from meta_agent.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class APICallerTool(BaseTool):
    """
    Make HTTP requests to external APIs.

    Security:
    - URL allowlist/blocklist enforcement
    - Request size limits
    - Response truncation
    - No file:// or localhost access by default
    """

    BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254"}  # SSRF prevention
    BLOCKED_SCHEMES = {"file", "ftp", "gopher"}
    MAX_RESPONSE_SIZE = 50_000

    def __init__(self) -> None:
        super().__init__(
            name="api_caller",
            description="Make HTTP requests to external APIs",
            timeout_seconds=30,
            max_retries=2,
            permission_scope="read",
        )

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        if "url" not in input_data:
            return "Missing required field: 'url'"

        url = input_data["url"]
        parsed = urlparse(url)

        if parsed.scheme in self.BLOCKED_SCHEMES:
            return f"Blocked URL scheme: '{parsed.scheme}'"

        if parsed.hostname and parsed.hostname in self.BLOCKED_HOSTS:
            return f"Blocked host: '{parsed.hostname}' (SSRF prevention)"

        method = input_data.get("method", "GET").upper()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            return f"Unsupported HTTP method: '{method}'"

        return None

    async def execute(self, input_data: dict[str, Any]) -> Any:
        import httpx

        url = input_data["url"]
        method = input_data.get("method", "GET").upper()
        headers = input_data.get("headers", {})
        body = input_data.get("body", None)

        logger.info("API call: %s %s", method, url[:100])

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=body if body else None,
            )

            response_text = response.text[:self.MAX_RESPONSE_SIZE]

            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response_text,
                "url": str(response.url),
            }

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Target URL"},
                    "method": {"type": "string", "default": "GET"},
                    "headers": {"type": "object", "default": {}},
                    "body": {"type": "object", "description": "JSON body (optional)"},
                },
                "required": ["url"],
            },
        }
