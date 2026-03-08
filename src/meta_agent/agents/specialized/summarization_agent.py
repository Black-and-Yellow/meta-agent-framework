"""
Summarization Agent
===================

Synthesises information from multiple sources into clear,
well-structured documents with proper formatting.
"""

from __future__ import annotations

from typing import Any

from meta_agent.agents.base_agent import BaseAgent
from meta_agent.schemas.blueprint import AgentConfig


class SummarizationAgent(BaseAgent):
    """Agent specialised in synthesising and summarising information."""

    def __init__(self, config: AgentConfig, **kwargs: Any) -> None:
        super().__init__(config)

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        task = input_data.get("task", "")
        previous = input_data.get("previous_results", {})

        system_prompt = self.get_system_prompt(task)

        user_content = (
            "You are the final summarization agent. Your job is to condense all prior agent outputs into a single clean final response.\n\n"
            f"Original task: {task}\n\n"
            f"Trace of all previous agents' work:\n{previous}\n\n"
            "REQUIREMENTS:\n"
            "- Extract the final verified working code block from the coding/verification agents.\n"
            "- Strip out all intermediate headers, planning text, data analysis reports, and redundant/broken code snippets.\n"
            "- Output FORMAT MUST BE EXACTLY:\n"
            "  1. One paragraph summarizing the factual findings (e.g., the temperature, source, etc.).\n"
            "  2. Followed by the final code block.\n"
            "- DO NOT add any extra commentary or wrap it in your own headers."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = await self.llm_client.ainvoke(messages)
        self.create_message(content=response.content)

        return {
            "output": response.content,
            "format": "markdown",
            "confidence": 0.9,
        }
