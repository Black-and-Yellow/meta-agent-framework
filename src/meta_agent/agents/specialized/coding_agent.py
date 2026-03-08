"""
Coding Agent
============

Expert software engineer that writes production-safe code.
Uses the code execution sandbox to verify solutions.
Handles repair feedback from verification agents.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

from meta_agent.agents.base_agent import BaseAgent
from meta_agent.schemas.blueprint import AgentConfig

logger = logging.getLogger(__name__)

# Keywords that hint the task needs production-grade validation
_ROBUST_KEYWORDS = {
    "production", "robust", "edge case", "validate", "test suite",
    "enterprise", "comprehensive", "thorough", "all cases",
}

# Modules that the code sandbox blocks (no network access)
_BLOCKED_MODULES = {
    "requests", "urllib", "httpx", "aiohttp", "socket", "subprocess",
}

# Regex that catches `import X`, `import X as Y`, and `from X.sub import ...`
_BLOCKED_IMPORT_RE = re.compile(
    r'^\s*(?:import|from)\s+('
    + "|".join(re.escape(m) for m in sorted(_BLOCKED_MODULES))
    + r')\b',
    re.MULTILINE,
)


class CodingAgent(BaseAgent):
    """
    Agent specialised in software engineering tasks.

    Capabilities:
    - Production-safe code generation with validation and edge cases
    - Code debugging and refactoring from verification feedback
    - Test writing and execution via sandbox
    - ASCII-safe output (no unicode symbols in code)
    """

    def __init__(self, config: AgentConfig, **kwargs: Any) -> None:
        super().__init__(config)

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        _start = time.time()
        _trace_id = str(uuid.uuid4())

        task = input_data.get("task", "")
        context = input_data.get("context", "")
        previous = input_data.get("previous_results", {})
        repair_feedback = input_data.get("repair_feedback", "")
        research_data = input_data.get("research_data", {})

        system_prompt = self.get_system_prompt(task)

        # ── Sandbox restriction rules ────────────────────────────────
        system_prompt += (
            "\n\n=== SANDBOX RESTRICTIONS ==="
            "\nThe code execution sandbox has the following restrictions:"
            "\n- No internet access"
            "\n- No HTTP requests"
            "\n- Do NOT import: requests, urllib, httpx, aiohttp, socket, subprocess"
            "\n- All required data will be provided through agent inputs"
            "\n- Use ONLY the Python standard library (math, json, re, etc.)"
            "\n=== END SANDBOX RESTRICTIONS ==="
        )

        user_content = f"Write code to accomplish the following:\n\n{task}"
        if context:
            user_content += f"\n\nContext and requirements:\n{context}"
        if previous:
            user_content += f"\n\nInput from previous steps:\n{previous}"

        # ── Inject dynamic data from research agent ──────────────────
        # Also check previous_results for research_data if not directly provided
        if not research_data and isinstance(previous, dict):
            for agent_id, agent_result in previous.items():
                if isinstance(agent_result, dict) and "research_data" in agent_result:
                    research_data = agent_result["research_data"]
                    break

        if research_data:
            location = research_data.get("location", "unknown")
            temp_c = research_data.get("temperature_celsius")
            temp_f = research_data.get("temperature_fahrenheit")
            source = research_data.get("source", "unknown")

            user_content += (
                "\n\n=== RESEARCH DATA (MANDATORY INPUT) ==="
                f"\nlocation: {location}"
                f"\ntemperature_celsius: {temp_c}"
                f"\ntemperature_fahrenheit: {temp_f}"
                f"\nsource: {source}"
                "\n"
                "\nYou MUST use these values when generating code."
                "\nDo NOT hardcode temperature values."
                "\nIf temperature_celsius exists, pass it into the conversion function."
                "\nIf a conversion function is defined, the researched temperature MUST be used as its input."
                "\n=== END RESEARCH DATA ==="
            )

        # If we have repair feedback, this is a re-attempt
        if repair_feedback:
            user_content += (
                "\n\n=== REPAIR REQUEST ==="
                "\nYour previous code FAILED verification. Here is the feedback:"
                f"\n{repair_feedback}"
                "\n\nFix ALL issues listed above. Do NOT repeat the same mistakes."
                "\n=== END REPAIR REQUEST ==="
            )

        # Task-proportional requirements to avoid over-engineering
        if _task_needs_robust_code(task):
            user_content += (
                "\n\n=== REQUIREMENTS ==="
                "\n1. ASCII-only code: no unicode symbols (use 'deg' not \u00b0)."
                "\n2. Input validation: check types, handle NaN/infinity/None."
                "\n3. Type hints and docstrings on all functions."
                "\n4. Include `if __name__ == '__main__':` block testing normal and edge cases."
                "\n=== END REQUIREMENTS ==="
            )
        else:
            user_content += (
                "\n\n=== REQUIREMENTS ==="
                "\n1. ASCII-only code: no unicode symbols (use 'deg' not \u00b0)."
                "\n2. Keep the solution minimal and focused. No unnecessary edge-case handling."
                "\n3. Include a brief `if __name__ == '__main__':` demo."
                "\n4. If a function is defined to perform a conversion, the script MUST call that function using the researched temperature value."
                "\n=== END REQUIREMENTS ==="
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = await self.invoke_llm_with_retry(messages)

        # ── Blocked-import gate: scan BEFORE sandbox execution ───────
        code = self._extract_code(response.content)
        if code:
            blocked = self._scan_blocked_imports(code)
            if blocked:
                logger.warning(
                    "Blocked imports detected: %s — regenerating code", blocked,
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "\n\n=== REGENERATION REQUIRED ==="
                        f"\nYour code used blocked imports: {', '.join(blocked)}"
                        "\nThe sandbox has NO network access."
                        "\nDo NOT use HTTP libraries or network access."
                        "\nRewrite the code using ONLY the Python standard library."
                        "\n=== END REGENERATION ==="
                    ),
                })
                response = await self.invoke_llm_with_retry(messages)
                code = self._extract_code(response.content)

        # ── Execute generated code in sandbox ────────────────────────
        code_output = None
        if self._tool_registry and code:
            try:
                code_output = await self.invoke_tool(
                    "code_executor",
                    {"code": code, "language": "python"},
                )
            except Exception as e:
                logger.warning("Code execution failed: %s", e)
                code_output = {"error": str(e), "success": False}

            # ── Dual-print consistency check ────────────────────────
            if isinstance(code_output, dict) and code_output.get("success"):
                stdout = code_output.get("stdout", "") or code_output.get("output", "")
                if self._has_dual_print_inconsistency(stdout):
                    code_output["consistency_warning"] = True
                    logger.warning(
                        "Dual-print inconsistency detected in code output"
                    )

        self.create_message(content=response.content, content_type="code")

        return {
            "output": response.content,
            "code_execution_result": code_output,
            "language": "python",
            "confidence": 0.85 if code_output and code_output.get("success") else 0.6,
            "trace_id": _trace_id,
            "agent_id": self.agent_id,
            "execution_time_ms": round((time.time() - _start) * 1000),
        }

    def _extract_code(self, text: str) -> str:
        """Extract the first code block from markdown-formatted text."""
        lines = text.split("\n")
        in_block = False
        code_lines: list[str] = []
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            elif line.strip().startswith("```") and in_block:
                break
            elif in_block:
                code_lines.append(line)
        return "\n".join(code_lines)

    @staticmethod
    def _scan_blocked_imports(code: str) -> list[str]:
        """Scan code for forbidden imports (requests, urllib, etc.).

        Detects:
        - ``import requests``
        - ``import requests as r``
        - ``from urllib.request import urlopen``
        - ``from httpx import Client``

        Returns a list of blocked module names found, or empty list.
        """
        found: list[str] = []
        for match in _BLOCKED_IMPORT_RE.finditer(code):
            mod = match.group(1)
            if mod not in found:
                found.append(mod)
        return found

    @staticmethod
    def _has_dual_print_inconsistency(stdout: str) -> bool:
        """Detect dual-print inconsistency in code output.

        Returns True if stdout contains two different numeric values on
        consecutive lines — a sign that the code prints both a hardcoded
        value and a calculated value for the same quantity.
        """
        if not stdout:
            return False
        # Extract all lines that contain a standalone numeric value
        number_re = re.compile(r'^.*?(-?\d+(?:\.\d+)?).*$')
        numeric_lines: list[float] = []
        for line in stdout.strip().splitlines():
            m = number_re.match(line.strip())
            if m:
                try:
                    numeric_lines.append(float(m.group(1)))
                except ValueError:
                    continue
        # Check consecutive numeric values for inconsistency
        for i in range(len(numeric_lines) - 1):
            if numeric_lines[i] != numeric_lines[i + 1]:
                # Two different numeric values on consecutive lines
                return True
        return False


def _task_needs_robust_code(task: str) -> bool:
    """Detect if the task implies production / robust code."""
    task_lower = task.lower()
    return any(kw in task_lower for kw in _ROBUST_KEYWORDS)
