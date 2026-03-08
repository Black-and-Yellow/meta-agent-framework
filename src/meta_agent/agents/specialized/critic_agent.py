"""
Critic Agent
============

The system's internal reviewer. Evaluates quality, identifies
weaknesses, and provides constructive improvement suggestions.
"""

from __future__ import annotations

from typing import Any

from meta_agent.agents.base_agent import BaseAgent
from meta_agent.schemas.blueprint import AgentConfig


class CriticAgent(BaseAgent):
    """
    Agent specialised in critical evaluation and quality assessment.

    This agent turns the AI into its own reviewer. It evaluates
    output quality across multiple dimensions and suggests improvements.
    """

    def __init__(self, config: AgentConfig, **kwargs: Any) -> None:
        super().__init__(config)

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        task = input_data.get("task", "")
        context = input_data.get("context", "")

        system_prompt = self.get_system_prompt(task)

        user_content = (
            "You are a strict code critic. Review the following code and output.\n\n"
            f"CONTEXT:\n{context}\n\n"
            "Evaluate EXACTLY three things:\n"
            "1. Is the researched value used in the code, or did the agent hardcode a different value?\n"
            "2. Does the printed output mathematically match the correct conversion from the researched value?\n"
            "3. Are there any duplicate or conflicting print statements (e.g., printing two different temperatures)?\n\n"
            "Respond ONLY with a valid JSON object matching exactly this schema:\n"
            "{\n"
            '  "passed": true / false,\n'
            '  "issues": ["description of issue 1 if any", ...]\n'
            "}\n"
            "No other text, markdown, or explanation."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = await self.llm_client.ainvoke(messages)
        self.create_message(content=response.content)
        
        # Parse JSON
        import json
        passed = False
        issues = []
        try:
            # Strip markdown code blocks if any
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            data = json.loads(content)
            passed = bool(data.get("passed", False))
            issues = data.get("issues", [])
        except json.JSONDecodeError:
            passed = False
            issues = ["Failed to parse Critic JSON response."]

        return {
            "output": "Critic passed." if passed else f"Critic issues: {issues}",
            "critic_report": {
                "passed": passed,
                "issues": issues
            }
        }
