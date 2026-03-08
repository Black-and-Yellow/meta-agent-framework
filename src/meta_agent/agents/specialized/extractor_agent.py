"""
Extractor Agent
===============

Lightweight agent that sits between Research and Coding agents.
Sole job is structured data extraction — no LLM call needed for simple cases.
"""

from __future__ import annotations

import logging
import re
import statistics
import time
import uuid
from typing import Any

from meta_agent.agents.base_agent import BaseAgent
from meta_agent.schemas.blueprint import AgentConfig

logger = logging.getLogger(__name__)

def is_valid_temperature_c(value: float) -> bool:
    return -90 <= value <= 60

def is_valid_temperature_f(value: float) -> bool:
    return -130 <= value <= 140

class ExtractorAgent(BaseAgent):
    """
    Deterministically extracts structured weather data from text using regex,
    falling back to LLM ONLY if 0 candidates are found.
    """

    def __init__(self, config: AgentConfig, **kwargs: Any) -> None:
        super().__init__(config)

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        start_time = time.time()
        context = input_data.get("context", "")
        task = input_data.get("task", "")
        
        # 1. Deterministic Extraction path
        structured_data = self._regex_extract(context)
        
        has_temp = structured_data.get("temperature_celsius") is not None or structured_data.get("temperature_fahrenheit") is not None
        
        # 2. Fall back to LLM if no candidates found
        if not has_temp:
            logger.info("Extractor: No regex candidates found, falling back to LLM.")
            prompt = self.system_prompt_template.replace("{{ role_description }}", self.role_description)
            prompt = prompt.replace("{{ task_description }}", task + "\n\nCONTEXT:\n" + context)
            
            try:
                llm_response = await self.invoke_llm_with_retry([{"role": "user", "content": prompt}])
                structured_data = self._regex_extract(llm_response)
            except Exception as e:
                logger.error("Extractor LLM fallback failed: %s", e)
        
        elapsed = (time.time() - start_time) * 1000
        logger.info("Extractor completed in %.1fms", elapsed)

        return {
            "output": f"Extracted structured data. Celsius: {structured_data.get('temperature_celsius')} F: {structured_data.get('temperature_fahrenheit')}",
            "research_data": structured_data,
        }

    def _regex_extract(self, text: str) -> dict[str, Any]:
        """
        Runs deterministic multi-pass regex extraction.
        Resolves conflicts using majority-vote.
        """
        result = {
            "temperature_celsius": None,
            "temperature_fahrenheit": None,
            "humidity": None,
            "wind_speed": None,
            "pressure": None,
            "uv_index": None,
            "location": "unknown", # Kept generic for this level
            "source": "Extractor",
        }
        
        if not text:
            return result

        # Simple extraction logic for temperatures
        # In actual usage text contains 'raw_extracted' items like: 12C (from: Title)
        # We can parse them out:
        c_candidates = []
        f_candidates = []
        
        matches = re.findall(
            r'(-?\d+\.?\d*)\s*(?:deg(?:rees?)?|°)?\s*([CcFf])(?:.*?\((?:from:\s*)?([^)]+)\))?', 
            text
        )
        
        for val_str, unit, source in matches:
            try:
                val = float(val_str)
                if unit.upper() == 'C' and is_valid_temperature_c(val):
                    c_candidates.append(val)
                elif unit.upper() == 'F' and is_valid_temperature_f(val):
                    f_candidates.append(val)
            except ValueError:
                pass
                
        # Majority vote
        def majority_vote(candidates: list[float]) -> float | None:
            if not candidates: return None
            try:
                # If multimodal, StatisticsError is raised in Python <3.8, or returns first mode in >=3.8
                # Actually, mode() returns first most common.
                return statistics.mode(candidates)
            except statistics.StatisticsError:
                return candidates[0]
                
        result["temperature_celsius"] = majority_vote(c_candidates)
        result["temperature_fahrenheit"] = majority_vote(f_candidates)
        
        # Cross-calculate if one is missing
        if result["temperature_celsius"] is None and result["temperature_fahrenheit"] is not None:
            result["temperature_celsius"] = round((result["temperature_fahrenheit"] - 32) * 5 / 9, 1)
        if result["temperature_fahrenheit"] is None and result["temperature_celsius"] is not None:
            result["temperature_fahrenheit"] = round(result["temperature_celsius"] * 9 / 5 + 32, 1)

        return result
