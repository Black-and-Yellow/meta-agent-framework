"""File Reader Tool — sandboxed file reading from the local filesystem."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from meta_agent.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class FileReaderTool(BaseTool):
    """
    Read text files from the local filesystem.

    Security:
    - Only files under allowed directories can be read
    - Maximum file size is enforced
    - Binary files are rejected
    - Symlink traversal is prevented (resolved path must stay in allowed dirs)
    """

    MAX_FILE_SIZE = 1_000_000  # 1 MB
    DEFAULT_ALLOWED_DIRS: list[str] = ["."]  # current working dir by default

    def __init__(self, allowed_dirs: list[str] | None = None) -> None:
        super().__init__(
            name="file_reader",
            description="Read files from the allowed file system",
            timeout_seconds=10,
            max_retries=1,
            permission_scope="read",
        )
        self.allowed_dirs = [
            Path(d).resolve() for d in (allowed_dirs or self.DEFAULT_ALLOWED_DIRS)
        ]

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        if "path" not in input_data:
            return "Missing required field: 'path'"

        path_str = input_data["path"]
        if not isinstance(path_str, str) or not path_str.strip():
            return "'path' must be a non-empty string"

        resolved = Path(path_str).resolve()

        # Security: check the resolved path is under an allowed directory
        if not any(self._is_subpath(resolved, allowed) for allowed in self.allowed_dirs):
            return (
                f"Access denied: '{path_str}' is outside allowed directories. "
                f"Allowed: {[str(d) for d in self.allowed_dirs]}"
            )

        return None

    async def execute(self, input_data: dict[str, Any]) -> Any:
        path = Path(input_data["path"]).resolve()
        encoding = input_data.get("encoding", "utf-8")
        max_lines = input_data.get("max_lines", None)

        logger.info("Reading file: %s", path)

        if not path.exists():
            return {"error": f"File not found: {path}", "content": ""}

        if not path.is_file():
            return {"error": f"Not a file: {path}", "content": ""}

        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            return {
                "error": f"File too large: {file_size:,} bytes (max {self.MAX_FILE_SIZE:,})",
                "content": "",
            }

        try:
            text = path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            return {"error": f"Cannot read binary file: {path}", "content": ""}

        if max_lines is not None:
            lines = text.splitlines(keepends=True)
            text = "".join(lines[:max_lines])
            truncated = len(lines) > max_lines
        else:
            truncated = False

        return {
            "path": str(path),
            "content": text,
            "size_bytes": file_size,
            "line_count": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
            "truncated": truncated,
        }

    @staticmethod
    def _is_subpath(child: Path, parent: Path) -> bool:
        """Check if child is under parent directory."""
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                    "encoding": {
                        "type": "string",
                        "description": "Text encoding",
                        "default": "utf-8",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (optional)",
                    },
                },
                "required": ["path"],
            },
        }
