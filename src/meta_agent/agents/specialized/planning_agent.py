"""
Planning Agent
==============

Analyses information and creates detailed, actionable plans.
Identifies risks, dependencies, and mitigation strategies.
"""

from __future__ import annotations

from typing import Any

from meta_agent.agents.base_agent import BaseAgent
from meta_agent.schemas.blueprint import AgentConfig


class PlanningAgent(BaseAgent):
    """
    Agent specialised in strategic planning and task decomposition.

    Creates structured plans with dependencies, timelines, and risk analysis.
    """

    def __init__(self, config: AgentConfig, **kwargs: Any) -> None:
        super().__init__(config)

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        task = input_data.get("task", "")
        context = input_data.get("context", "")
        previous = input_data.get("previous_results", {})

        system_prompt = self.get_system_prompt(task)

        user_content = (
            f"Create a detailed, actionable plan for the following:\n\n{task}"
        )
        if context:
            user_content += f"\n\nContext:\n{context}"
        if previous:
            user_content += f"\n\nInput from previous steps:\n{previous}"

        user_content += (
            "\n\nStructure your plan with:"
            "\n1. Objectives"
            "\n2. Step-by-step actions"
            "\n3. Dependencies between steps"
            "\n4. Risk analysis and mitigations"
            "\n5. Success criteria"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = await self.llm_client.ainvoke(messages)
        self.create_message(content=response.content)

        return {
            "output": response.content,
            "plan_type": "strategic",
            "confidence": 0.85,
        }
