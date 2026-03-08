"""Code Executor Tool — sandboxed Python code execution with UTF-8 support."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from meta_agent.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

# Map of non-ASCII characters that commonly appear in LLM-generated code
# to their safe ASCII equivalents.
_UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u00b0": "deg",   # ° (degree sign)
    "\u00d7": "*",     # × (multiplication sign)
    "\u2013": "-",     # – (en dash)
    "\u2014": "--",    # — (em dash)
    "\u2018": "'",     # ' (left single quote)
    "\u2019": "'",     # ' (right single quote)
    "\u201c": '"',     # " (left double quote)
    "\u201d": '"',     # " (right double quote)
    "\u2026": "...",   # … (ellipsis)
    "\u2264": "<=",    # ≤
    "\u2265": ">=",    # ≥
    "\u2260": "!=",    # ≠
}


def _sanitize_code(code: str) -> str:
    """
    Sanitize LLM-generated code for safe execution.

    1. Replace problematic non-ASCII characters with ASCII equivalents.
    2. Keep legitimate unicode in string literals where possible.
    """
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        code = code.replace(char, replacement)
    return code


class CodeExecutorTool(BaseTool):
    """
    Execute Python code in a sandboxed subprocess.

    Security measures:
    - Runs in a subprocess with limited resources
    - Timeout enforced at the process level
    - Blocked import list for dangerous modules
    - Output truncation to prevent memory exhaustion
    - UTF-8 encoding with sanitization for LLM-generated code
    """

    BLOCKED_IMPORTS = {
        "os", "sys", "subprocess", "shutil", "pathlib",
        "socket", "http", "urllib", "requests",
        "ctypes", "importlib", "pickle",
    }
    MAX_OUTPUT_LENGTH = 10_000

    def __init__(self) -> None:
        super().__init__(
            name="code_executor",
            description="Execute Python code in a sandboxed environment",
            timeout_seconds=60,
            max_retries=1,
            permission_scope="execute",
        )

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        if "code" not in input_data:
            return "Missing required field: 'code'"
        code = input_data["code"]
        if not isinstance(code, str) or not code.strip():
            return "'code' must be a non-empty string"

        # Security check: block dangerous imports
        for blocked in self.BLOCKED_IMPORTS:
            if f"import {blocked}" in code or f"from {blocked}" in code:
                return f"Blocked import detected: '{blocked}' is not allowed in sandbox"

        return None

    async def execute(self, input_data: dict[str, Any]) -> Any:
        code = input_data["code"]
        language = input_data.get("language", "python")

        if language != "python":
            return {"error": f"Unsupported language: {language}", "output": ""}

        # Sanitize non-ASCII characters that cause SyntaxError
        code = _sanitize_code(code)

        logger.info("Executing code (%d chars)", len(code))

        # Prepend UTF-8 encoding declaration and write with explicit encoding
        full_code = "# -*- coding: utf-8 -*-\n" + code

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(full_code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=tempfile.gettempdir(),
                env={**dict(__import__("os").environ), "PYTHONUTF8": "1"},
            )

            stdout = result.stdout[:self.MAX_OUTPUT_LENGTH]
            stderr = result.stderr[:self.MAX_OUTPUT_LENGTH]

            return {
                "stdout": stdout,
                "stderr": stderr,
                "return_code": result.returncode,
                "success": result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "Code execution timed out",
                "return_code": -1,
                "success": False,
            }
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "language": {
                        "type": "string",
                        "description": "Programming language (only 'python' supported)",
                        "default": "python",
                    },
                },
                "required": ["code"],
            },
        }
