"""
Verification Agent
==================

Checks accuracy and correctness of work produced by other agents.
Tests claims, validates code, identifies inconsistencies.
Provides structured feedback for repair loops.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
import uuid
from typing import Any

from meta_agent.agents.base_agent import BaseAgent
from meta_agent.schemas.blueprint import AgentConfig

logger = logging.getLogger(__name__)


class VerificationAgent(BaseAgent):
    """Agent specialised in verifying, testing, and validating results."""

    def __init__(self, config: AgentConfig, **kwargs: Any) -> None:
        super().__init__(config)

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        _start = time.time()
        _trace_id = str(uuid.uuid4())

        task = input_data.get("task", "")
        previous = input_data.get("previous_results", {})

        system_prompt = self.get_system_prompt(task)

        # Always run code through sandbox first, so the LLM has real test output
        code_test_result = None
        sandbox_report = ""
        if self._tool_registry and previous:
            code_block = self._extract_code(str(previous))
            if code_block:
                try:
                    code_test_result = await self.invoke_tool(
                        "code_executor",
                        {"code": code_block, "language": "python"},
                    )
                    success = code_test_result.get("success", False)
                    sandbox_report = (
                        f"\n\nSANDBOX EXECUTION RESULT:"
                        f"\n  Success: {success}"
                        f"\n  stdout: {code_test_result.get('stdout', '(empty)')}"
                        f"\n  stderr: {code_test_result.get('stderr', '(empty)')}"
                        f"\n  return_code: {code_test_result.get('return_code', -1)}"
                    )
                except Exception as e:
                    logger.warning("Code verification sandbox failed: %s", e)
                    sandbox_report = f"\n\nSANDBOX EXECUTION FAILED: {e}"

        user_content = (
            f"Original task: {task}\n\n"
            f"Work to verify:\n{previous}"
            f"{sandbox_report}"
            "\n\n=== VERIFICATION CHECKLIST ==="
            "\n1. Factual accuracy: Are claims correct?"
            "\n2. Logical consistency: Does the logic hold?"
            "\n3. Completeness: Does it address the full task?"
            "\n4. Code correctness: Does the code run without errors?"
            "\n=== END CHECKLIST ==="
            "\n\nIMPORTANT: If the SANDBOX EXECUTION RESULT shows Success: True, "
            "the code correctness check MUST be PASS. Do NOT override sandbox results."
            "\n\nFor EACH check, write: CHECK_NAME: PASS or FAIL"
            "\n\nIf ANY check is FAIL, you MUST provide:"
            "\n- The exact issue description"
            "\n- A concrete fix suggestion"
            "\n\nEnd with: FINAL_VERDICT: PASS or FINAL_VERDICT: FAIL"
            "\n\nIf FAIL, also provide a REPAIR_INSTRUCTIONS section with "
            "specific bullet points describing what needs to change."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = await self.invoke_llm_with_retry(messages)
        self.create_message(content=response.content)

        # Parse structured feedback
        verdict = self._parse_verdict(response.content)
        confidence = self._parse_confidence(response.content)
        repair_instructions = self._extract_repair_instructions(response.content)
        check_results = self._parse_check_results(response.content)

        # Guard: sandbox success overrides hallucinated LLM failures
        code_block_str = self._extract_code(str(previous)) if previous else ""
        if self._sandbox_overrides_llm(code_test_result, verdict, code_block_str):
            logger.info(
                "Verification guard: sandbox PASSED → overriding LLM FAIL verdict"
            )
            verdict = True
            repair_instructions = ""  # No repairs needed

        return {
            "output": response.content,
            "verification_passed": verdict,
            "code_test_result": code_test_result,
            "confidence": confidence,
            "repair_instructions": repair_instructions,
            "check_results": check_results,
            "trace_id": _trace_id,
            "agent_id": self.agent_id,
            "execution_time_ms": round((time.time() - _start) * 1000),
        }

    @staticmethod
    def _parse_verdict(text: str) -> bool:
        """Parse FINAL_VERDICT: PASS/FAIL from the response."""
        # Look for explicit FINAL_VERDICT line first
        match = re.search(r'FINAL_VERDICT\s*:\s*(PASS|FAIL)', text, re.IGNORECASE)
        if match:
            return match.group(1).upper() == "PASS"

        # Fallback: check last few lines
        text_upper = text.upper()
        last_lines = text_upper.strip().split("\n")[-5:]
        last_block = " ".join(last_lines)

        if "FAIL" in last_block:
            return False
        if "PASS" in last_block:
            return True

        # Count-based heuristic
        fail_count = text_upper.count("FAIL")
        pass_count = text_upper.count("PASS")
        return pass_count >= fail_count

    @staticmethod
    def _parse_confidence(text: str) -> float:
        """Extract a confidence value from the response."""
        match = re.search(r'[Cc]onfidence\s*(?:[Ll]evel)?[:\s]+(\d+\.?\d*)', text)
        if match:
            try:
                val = float(match.group(1))
                return val if val <= 1.0 else val / 100.0
            except ValueError:
                pass
        return 0.75

    @staticmethod
    def _extract_repair_instructions(text: str) -> str:
        """Extract the REPAIR_INSTRUCTIONS section if present."""
        # Look for the section between REPAIR_INSTRUCTIONS and the next section/end
        match = re.search(
            r'REPAIR_INSTRUCTIONS\s*:?\s*\n(.*?)(?:\n(?:FINAL_VERDICT|===)|$)',
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # Fallback: if verdict is FAIL, use the suggestions as repair instructions
        if "FAIL" in text.upper():
            # Collect lines that look like fix suggestions
            fix_lines = []
            for line in text.split("\n"):
                line_stripped = line.strip()
                if any(kw in line_stripped.lower() for kw in
                       ["fix:", "should", "must", "change", "replace", "add", "remove",
                        "instead", "use ", "handle", "validate"]):
                    fix_lines.append(line_stripped)
            if fix_lines:
                return "\n".join(fix_lines[:10])

        return ""

    @staticmethod
    def _parse_check_results(text: str) -> dict[str, str]:
        """Parse individual check results like 'Factual accuracy: PASS'."""
        results: dict[str, str] = {}
        for match in re.finditer(r'(\w[\w\s]+?):\s*(PASS|FAIL)', text, re.IGNORECASE):
            name = match.group(1).strip().lower().replace(" ", "_")
            verdict = match.group(2).upper()
            results[name] = verdict
        return results

    @staticmethod
    def _sandbox_overrides_llm(
        code_test_result: dict[str, Any] | None,
        llm_verdict: bool,
        code_block: str,
    ) -> bool:
        """Return True if sandbox success should override an LLM FAIL verdict.

        Uses AST parsing for reliable function/class detection instead of
        fragile string matching.
        """
        if llm_verdict:
            return False  # LLM already says PASS, no override needed
        if not code_test_result:
            return False
        if not code_test_result.get("success"):
            return False
        # Use AST to check the code defines at least one function or class
        if not _code_has_definitions(code_block):
            return False
        # If stderr contains real errors, trust the LLM
        stderr = code_test_result.get("stderr", "")
        if stderr.strip() and "error" in stderr.lower():
            return False
        return True

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract the first Python code block from text."""
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


def _code_has_definitions(code: str) -> bool:
    """Use AST parsing to check if code defines functions or classes.

    More reliable than `'def ' in code` — immune to def appearing
    in strings, comments, or variable names.
    """
    if not code.strip():
        return False
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return True
    except SyntaxError:
        pass
    return False
