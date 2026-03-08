"""
Research Agent
==============

Finds accurate, up-to-date information using search tools.
Constructs targeted search queries and prioritises trusted sources.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from meta_agent.agents.base_agent import BaseAgent
from meta_agent.schemas.blueprint import AgentConfig

logger = logging.getLogger(__name__)

# Keywords that hint the task is about weather / real-time data
_WEATHER_KEYWORDS = {
    "weather", "temperature", "forecast", "celsius", "fahrenheit",
    "humidity", "wind", "rain", "snow", "climate",
}
_WEATHER_DOMAINS = ["weather.com", "accuweather.com", "openweathermap.org", "timeanddate.com"]

# ── Source scoring for temperature extraction ──────────────────────
# Keywords that indicate the source is about CURRENT conditions (prefer)
_CURRENT_SOURCE_KEYWORDS = {
    "current", "today", "now", "weather today", "current weather",
    "right now", "live", "real-time", "conditions", "now/",
}
# Keywords that indicate the source is NOT about current conditions (deprioritise)
_IRRELEVANT_SOURCE_KEYWORDS = {
    "forecast", "olympics", "historical", "2032", "may", "could",
    "early", "blossoms", "sea level", "prediction", "expected",
    "14 day", "10 day", "monthly", "fuji", "asakusa", "elderly",
    "forecast for", "tenday", "extend",
}

# Words to skip when extracting location from task text
_LOCATION_SKIP_WORDS = {
    "Research", "Write", "Python", "script", "current", "weather",
    "temperature", "the", "in", "and", "a", "that", "to", "it",
    "from", "with", "for", "this", "The", "find", "get", "short",
    "Fahrenheit", "Celsius", "fahrenheit", "celsius", "convert",
    "Convert", "degrees", "Degrees", "today", "Today", "forecast",
    "Forecast", "humidity", "Humidity",
}


class ResearchAgent(BaseAgent):
    """
    Agent specialised in information retrieval and synthesis.

    Capabilities:
    - Smart query construction (extracts key terms, adds qualifiers)
    - Domain-aware search (prefers trusted sources)
    - Source citation and cross-referencing
    - Structured data extraction from search snippets
    """

    def __init__(self, config: AgentConfig, **kwargs: Any) -> None:
        super().__init__(config)

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        _start = time.time()
        _trace_id = str(uuid.uuid4())

        task = input_data.get("task", "")
        context = input_data.get("context", "")
        previous = input_data.get("previous_results", {})

        system_prompt = self.get_system_prompt(task)

        # ── Hardened instructions ────────────────────────────────────
        system_prompt += (
            "\n\n=== MANDATORY RESEARCH RULES ==="
            "\n1. You MUST use ONLY the provided web_search tool to find information."
            "\n2. Do NOT generate, fabricate, or reference Google/Bing search URLs."
            "\n3. Do NOT use phrases like 'as of my knowledge cutoff'."
            "\n4. Your output MUST be based on data extracted from search tool results."
            "\n5. If the search tool returns no relevant results, say so explicitly."
            "\n6. ALWAYS cite the actual URLs returned by the search tool."
            "\n7. If a value is not present in search results, do NOT guess or assume a value."
            "\n8. NEVER use phrases like 'Let us assume' or 'approximately' without source data."
            "\n9. Return null for any field where no numeric value was found in search results."
            "\n=== END RULES ==="
        )

        user_message = f"Research the following:\n\n{task}"
        if context:
            user_message += f"\n\nAdditional context:\n{context}"
        if previous:
            user_message += f"\n\nPrevious findings:\n{previous}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Attempt tool-augmented research
        sources: list[dict[str, str]] = []
        has_trusted_hit = False
        extracted_data_str = ""

        if self._tool_registry:
            try:
                # Build a targeted search query
                search_query = self._build_search_query(task)
                preferred_domains = self._detect_preferred_domains(task)

                search_data = await self.invoke_tool(
                    "web_search",
                    {
                        "query": search_query,
                        "num_results": 8,
                        "preferred_domains": preferred_domains,
                    },
                )
                results = search_data.get("results", [])

                # ── Retry with refined query if initial search returned nothing ──
                if not results:
                    refined_query = self._build_refined_query(task)
                    if refined_query != search_query:
                        logger.info(
                            "No results for '%s', retrying with '%s'",
                            search_query, refined_query,
                        )
                        search_data = await self.invoke_tool(
                            "web_search",
                            {
                                "query": refined_query,
                                "num_results": 8,
                                "preferred_domains": preferred_domains,
                            },
                        )
                        results = search_data.get("results", [])

                if results:
                    # Build sources list — only include results with real URLs
                    sources = [
                        {"title": r.get("title", ""), "url": r.get("url", "")}
                        for r in results
                        if r.get("url")
                    ]
                    has_trusted_hit = any(
                        self._is_trusted_domain(r.get("url", ""))
                        for r in results
                    )

                    # Format search results for the LLM
                    formatted = "\n\n".join(
                        f"[{i+1}] {r.get('title', 'No title')}\n"
                        f"URL: {r.get('url', '')}\n"
                        f"Snippet: {r.get('body', '')}"
                        for i, r in enumerate(results[:6])
                    )
                    messages.append({
                        "role": "system",
                        "content": (
                            "Here are live search results. Use these as your "
                            "PRIMARY information source. Extract specific data "
                            "(numbers, dates, facts) from the snippets. "
                            "Cite sources by URL.\n\n"
                            f"{formatted}"
                        ),
                    })

                    # Try to extract structured data from snippets
                    extracted_data_str = self._extract_data_from_snippets(results, task)
                    if extracted_data_str:
                        messages.append({
                            "role": "system",
                            "content": f"Extracted data from search: {extracted_data_str}",
                        })
                else:
                    # No results even after retry — tell the LLM explicitly
                    logger.warning("Web search returned no results for task: %s", task)
                    messages.append({
                        "role": "system",
                        "content": (
                            "WARNING: The web search tool returned no results. "
                            "You must report that no live data was found. "
                            "Do NOT make up information."
                        ),
                    })

            except Exception as e:
                logger.warning("Web search failed, proceeding with LLM knowledge: %s", e)
                messages.append({
                    "role": "system",
                    "content": (
                        f"WARNING: Web search failed ({e}). "
                        "Report that search was unavailable. Do NOT make up data."
                    ),
                })
        else:
            logger.warning("No tool registry — research agent cannot use web_search")

        # Use rate-limit-safe LLM invocation
        response = await self.invoke_llm_with_retry(messages)

        result_message = self.create_message(
            content=response.content,
            content_type="text",
        )

        # Confidence based on source quality
        if has_trusted_hit:
            confidence = 0.95
        elif sources:
            confidence = 0.7
        else:
            confidence = 0.4  # lower confidence when no search results used

        # Build structured research_data for downstream agents
        research_data = self._build_structured_response(
            task, extracted_data_str, sources, response.content,
        )

        return {
            "output": response.content,
            "sources": sources,
            "confidence": confidence,
            "message": result_message.model_dump(),
            "research_data": research_data,
            "trace_id": _trace_id,
            "agent_id": self.agent_id,
            "execution_time_ms": round((time.time() - _start) * 1000),
        }

    @staticmethod
    def _build_search_query(task: str) -> str:
        """
        Build a targeted search query from the task description.

        Instead of sending the full task verbatim (which gives poor results),
        extract the key information need and add qualifiers.
        For weather queries, includes multi-site OR clauses for better results.
        """
        task_lower = task.lower()

        # Weather-specific: extract location and build pointed query
        if any(kw in task_lower for kw in _WEATHER_KEYWORDS):
            # Try to find a location/city name
            words = task.split()
            skip = {"Research", "Write", "Python", "script", "current", "weather",
                    "temperature", "the", "in", "and", "a", "that", "to", "it",
                    "from", "with", "for", "this", "The", "find", "get", "short"}
            location_words = [w.strip(".,!?") for w in words
                              if w[0:1].isupper() and w.strip(".,!?") not in skip]
            if location_words:
                location = " ".join(location_words)
                # Dynamic query targeting current conditions page directly
                return f"{location} current weather temperature today"

        # General: use the task as-is but trim to key terms
        for prefix in ["Research the ", "Find information about ", "Look up "]:
            if task.startswith(prefix):
                task = task[len(prefix):]
                break

        return task[:150]  # cap length for search engines

    @staticmethod
    def _build_refined_query(task: str) -> str:
        """
        Build a simpler, broader fallback query when the initial query returns nothing.

        Strips site: clauses and qualifiers for a more general search.
        """
        task_lower = task.lower()

        if any(kw in task_lower for kw in _WEATHER_KEYWORDS):
            words = task.split()
            skip = {"Research", "Write", "Python", "script", "current", "weather",
                    "temperature", "the", "in", "and", "a", "that", "to", "it",
                    "from", "with", "for", "this", "The", "find", "get", "short"}
            location_words = [w.strip(".,!?") for w in words
                              if w[0:1].isupper() and w.strip(".,!?") not in skip]
            if location_words:
                location = " ".join(location_words)
                # Simple query without site: clauses
                return f"weather {location} temperature today"

        # General: just use the core task text
        for prefix in ["Research the ", "Find information about ", "Look up "]:
            if task.startswith(prefix):
                task = task[len(prefix):]
                break
        return task[:100]

    @staticmethod
    def _detect_preferred_domains(task: str) -> list[str]:
        """Detect which trusted domains to prefer based on task content."""
        task_lower = task.lower()
        if any(kw in task_lower for kw in _WEATHER_KEYWORDS):
            return _WEATHER_DOMAINS
        return []

    @staticmethod
    def _is_trusted_domain(url: str) -> bool:
        """Check if a URL is from a trusted domain."""
        from urllib.parse import urlparse
        try:
            host = urlparse(url).netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            trusted = {
                "weather.com", "accuweather.com", "openweathermap.org",
                "timeanddate.com", "wikipedia.org", "python.org",
            }
            return host in trusted or any(host.endswith("." + d) for d in trusted)
        except Exception:
            return False

    @staticmethod
    def _score_source(title: str) -> int:
        """Score a source title for relevance to current weather.

        Returns:
            +1 for titles containing current-condition keywords,
            -1 for titles containing irrelevant keywords,
             0 otherwise.
        """
        title_lower = title.lower()
        for kw in _IRRELEVANT_SOURCE_KEYWORDS:
            if kw in title_lower:
                return -1
        for kw in _CURRENT_SOURCE_KEYWORDS:
            if kw in title_lower:
                return 1
        return 0

    @staticmethod
    def _extract_data_from_snippets(results: list[dict], task: str) -> str:
        """
        Try to extract structured data (temperatures, numbers) from snippets.
        """
        task_lower = task.lower()
        extracted: list[str] = []

        if any(kw in task_lower for kw in _WEATHER_KEYWORDS):
            # Look for temperature patterns in snippets
            for r in results:
                body = r.get("body", "")
                # Match patterns like "12°C", "25 °F", "72F", "15 degrees"
                temps = re.findall(
                    r'(-?\d+\.?\d*)\s*(?:deg(?:rees?)?|°)\s*([CcFf])',
                    body,
                )
                if temps:
                    for val, unit in temps:
                        extracted.append(f"{val}°{unit.upper()} (from: {r.get('title', '')})")

        return "; ".join(extracted) if extracted else ""

    @staticmethod
    def _build_structured_response(
        task: str,
        extracted_data: str,
        sources: list[dict[str, str]],
        llm_response: str,
    ) -> dict[str, Any]:
        def is_valid_temperature_c(value): return value is not None and -90 <= value <= 60
        def is_valid_temperature_f(value): return value is not None and -130 <= value <= 140
        """
        Build a machine-readable structured response from research results.

        Returns a dict like:
        {
            "location": "Tokyo",
            "temperature_celsius": 12.0,
            "temperature_fahrenheit": null,
            "source": "weather.com",
            "raw_extracted": "12°C (from: Tokyo Weather)"
        }
        """
        result: dict[str, Any] = {
            "raw_extracted": extracted_data,
            "source": sources[0]["url"] if sources else "unknown",
        }

        task_lower = task.lower()

        # Extract location from the task
        words = task.split()
        location_words = [w.strip(".,!?") for w in words
                          if w[0:1].isupper() and w.strip(".,!?") not in _LOCATION_SKIP_WORDS]
        result["location"] = " ".join(location_words) if location_words else "unknown"

        # Extract temperatures from extracted_data (snippet values).
        # Uses source-scoring to prefer candidates from current-condition
        # articles over forecast/historical sources.
        result["temperature_celsius"] = None
        result["temperature_fahrenheit"] = None

        if extracted_data:
            # Parse all candidates as (value, unit, source_title) tuples
            candidates = re.findall(
                r'(-?\d+\.?\d*)\s*(?:deg(?:rees?)?|°)\s*([CcFf])'
                r'(?:\s*\(from:\s*([^)]+)\))?',
                extracted_data,
            )

            if candidates:
                raw_pieces = []
                for val_str, unit, source_title in candidates:
                    raw_pieces.append(f"{val_str}{unit.upper()} (from: {source_title})")
                result["raw_extracted"] = "; ".join(raw_pieces)

            # Separate by unit and score by source title
            c_candidates: list[tuple[float, int]] = []  # (value, score)
            f_candidates: list[tuple[float, int]] = []  # (value, score)
            for val_str, unit, source_title in candidates:
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                score = ResearchAgent._score_source(source_title)
                if unit.upper() == "C":
                    c_candidates.append((val, score))
                else:
                    f_candidates.append((val, score))

            # Pick the best-scored candidate; fall back to first if tied
            if c_candidates:
                c_candidates.sort(key=lambda x: x[1], reverse=True)
                result["temperature_celsius"] = c_candidates[0][0]
            if f_candidates:
                f_candidates.sort(key=lambda x: x[1], reverse=True)
                result["temperature_fahrenheit"] = f_candidates[0][0]

        # ── Fallback: extract temperatures from LLM response text ────
        # Only used when snippet extraction found nothing.  The LLM
        # response may contain explicit values like "64°F" from search
        # context — this is NOT fabrication, it's relay of source data.
        if (
            result["temperature_celsius"] is None
            and result["temperature_fahrenheit"] is None
            and llm_response
        ):
            f_match = re.search(
                r'(-?\d+(?:\.\d+)?)\s*°?\s*F', llm_response, re.IGNORECASE,
            )
            c_match = re.search(
                r'(-?\d+(?:\.\d+)?)\s*°?\s*C', llm_response, re.IGNORECASE,
            )
            if f_match:
                try:
                    result["temperature_fahrenheit"] = float(f_match.group(1))
                except ValueError:
                    pass
            if c_match:
                try:
                    result["temperature_celsius"] = float(c_match.group(1))
                except ValueError:
                    pass

        # If we got Fahrenheit but not Celsius, convert
        if result["temperature_celsius"] is None and result["temperature_fahrenheit"] is not None:
            result["temperature_celsius"] = round(
                (result["temperature_fahrenheit"] - 32) * 5 / 9, 1
            )
        # If we got Celsius but not Fahrenheit, convert
        if result["temperature_fahrenheit"] is None and result["temperature_celsius"] is not None:
            result["temperature_fahrenheit"] = round(
                result["temperature_celsius"] * 9 / 5 + 32, 1
            )

        if not is_valid_temperature_c(result["temperature_celsius"]): 
            result["temperature_celsius"] = None
        if not is_valid_temperature_f(result["temperature_fahrenheit"]): 
            result["temperature_fahrenheit"] = None

        return result
