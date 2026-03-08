"""
Data Analysis Agent
===================

Senior data analyst that uses statistical methods, produces
visualisations, and generates actionable insights.
"""

from __future__ import annotations

from typing import Any

from meta_agent.agents.base_agent import BaseAgent
from meta_agent.schemas.blueprint import AgentConfig


class DataAnalysisAgent(BaseAgent):
    """Agent specialised in data analysis, statistics, and visualisation."""

    def __init__(self, config: AgentConfig, **kwargs: Any) -> None:
        super().__init__(config)

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        task = input_data.get("task", "")
        context = input_data.get("context", "")
        previous = input_data.get("previous_results", {})

        system_prompt = self.get_system_prompt(task)

        user_content = f"Analyse the following data/task:\n\n{task}"
        if context:
            user_content += f"\n\nData context:\n{context}"
        if previous:
            user_content += f"\n\nData from previous steps:\n{previous}"

        user_content += (
            "\n\nProvide:"
            "\n1. Key statistical findings"
            "\n2. Patterns and trends"
            "\n3. Actionable insights"
            "\n4. Code for any computations (Python/pandas)"
            "\n5. Visualisation recommendations"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = await self.llm_client.ainvoke(messages)
        self.create_message(content=response.content, content_type="json")

        return {
            "output": response.content,
            "analysis_type": "exploratory",
            "confidence": 0.8,
        }
