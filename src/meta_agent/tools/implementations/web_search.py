"""Web Search Tool — production-grade internet search via DuckDuckGo.

Uses the ``ddgs`` package (successor to duckduckgo-search, no API key required).
Features:
- Domain relevance scoring (trusted domains ranked first)
- Automatic query refinement with site-specific fallback
- Content validation (keyword-based relevance check)
- Configurable preferred domains per query
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from meta_agent.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

# Domains known to provide high-quality, factual information
TRUSTED_DOMAINS: dict[str, int] = {
    "weather.com": 10,
    "accuweather.com": 10,
    "openweathermap.org": 10,
    "timeanddate.com": 10,
    "wikipedia.org": 8,
    "britannica.com": 8,
    "python.org": 7,
    "docs.python.org": 7,
    "stackoverflow.com": 6,
    "github.com": 5,
    "developer.mozilla.org": 5,
}

# Domains that are almost always irrelevant noise
BLOCKED_DOMAINS: set[str] = {
    "researchgate.net",
    "academia.edu",
    "quora.com",
    "pinterest.com",
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "tiktok.com",
    # Non-English Q&A / content-farm domains that pollute results
    "baidu.com",
    "zhidao.baidu.com",
    "tieba.baidu.com",
    "wenku.baidu.com",
    "zhihu.com",
    "sogou.com",
    "360.cn",
    "hao123.com",
    "douban.com",
    "csdn.net",
}


def _domain_of(url: str) -> str:
    """Extract the root domain from a URL."""
    try:
        host = urlparse(url).netloc.lower()
        # Strip www. prefix
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _score_result(result: dict[str, str]) -> int:
    """Score a search result by domain trustworthiness."""
    domain = _domain_of(result.get("url", ""))
    if domain in BLOCKED_DOMAINS:
        return -1  # will be filtered out
    # Also block subdomains of blocked domains
    for blocked in BLOCKED_DOMAINS:
        if domain.endswith("." + blocked):
            return -1
    for trusted, score in TRUSTED_DOMAINS.items():
        if domain == trusted or domain.endswith("." + trusted):
            return score
    # Bonus for .gov / .edu domains
    if domain.endswith(".gov"):
        return 5
    if domain.endswith(".edu"):
        return 4
    return 1  # unknown domain


class WebSearchTool(BaseTool):
    """
    Search the internet for current information using DuckDuckGo.

    Uses the ``ddgs`` package (the renamed successor to duckduckgo-search).
    Returns results sorted by domain relevance.  Trusted domains like
    weather.com and wikipedia.org are ranked first; noise domains like
    researchgate.net are filtered out entirely.
    """

    def __init__(self) -> None:
        super().__init__(
            name="web_search",
            description="Search the internet for current information on a topic",
            timeout_seconds=15,
            max_retries=2,
            permission_scope="read",
        )

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        if "query" not in input_data:
            return "Missing required field: 'query'"
        if not isinstance(input_data["query"], str) or len(input_data["query"]) < 2:
            return "'query' must be a non-empty string (min 2 chars)"
        return None

    async def execute(self, input_data: dict[str, Any]) -> Any:
        query = input_data["query"]
        num_results = input_data.get("num_results", 8)
        preferred_domains: list[str] = input_data.get("preferred_domains", [])

        logger.info("Web search: '%s' (max %d results)", query, num_results)

        try:
            # ── Use the new `ddgs` package (successor to duckduckgo-search) ──
            from ddgs import DDGS

            results = self._search_and_rank(DDGS, query, num_results)

            # If no trusted-domain results found and we have preferred domains,
            # retry with a site-specific query for better targeting
            has_trusted = any(_score_result(r) >= 5 for r in results)
            if not has_trusted and preferred_domains:
                # Build a multi-site OR query for broader coverage
                site_clauses = " OR ".join(
                    f"site:{d}" for d in preferred_domains[:3]
                )
                site_query = f"{query} {site_clauses}"
                logger.info("Retrying with site-specific query: '%s'", site_query)
                site_results = self._search_and_rank(DDGS, site_query, num_results)
                if site_results:
                    results = site_results

            # If still no results, try a simplified query as last resort
            if not results:
                simplified = _simplify_query(query)
                if simplified != query:
                    logger.info("Retrying with simplified query: '%s'", simplified)
                    results = self._search_and_rank(DDGS, simplified, num_results)

            logger.info(
                "Web search returned %d results for '%s'", len(results), query
            )
            return {
                "query": query,
                "results": results,
                "result_count": len(results),
            }

        except ImportError:
            logger.warning(
                "ddgs not installed — run: pip install ddgs"
            )
            return {
                "query": query,
                "results": [],
                "result_count": 0,
                "error": "ddgs package not installed",
            }
        except Exception as e:
            logger.error("Web search failed: %s", e)
            return {
                "query": query,
                "results": [],
                "result_count": 0,
                "error": str(e),
            }

    @staticmethod
    def _search_and_rank(
        ddgs_cls: type,
        query: str,
        num_results: int,
    ) -> list[dict[str, str]]:
        """Execute search via DDGS API, filter blocked domains, validate content, sort by relevance."""
        try:
            with ddgs_cls() as ddgs:
                raw = list(ddgs.text(query, max_results=num_results))
        except Exception as e:
            logger.warning("DDGS search call failed: %s", e)
            return []

        if not raw:
            return []

        query_keywords = _extract_query_keywords(query)

        scored: list[tuple[int, dict[str, str]]] = []
        for r in raw:
            # Normalise result keys — ddgs uses 'href' for URL and 'body' for snippet
            entry = {
                "title": r.get("title", ""),
                "url": r.get("href", r.get("link", r.get("url", ""))),
                "body": r.get("body", r.get("snippet", "")),
            }
            score = _score_result(entry)
            if score < 0:  # blocked domain
                continue

            # Validate content relevance — down-rank results with no keyword match
            if not _validate_result(entry, query_keywords):
                score = 0

            scored.append((score, entry))

        # Sort descending by score (trusted domains first)
        scored.sort(key=lambda x: x[0], reverse=True)

        # If we have good results (score >= 1), drop the zero-scored ones
        has_good = any(s >= 1 for s, _ in scored)
        if has_good:
            scored = [(s, e) for s, e in scored if s >= 1]

        return [entry for _, entry in scored]

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {
                        "type": "integer",
                        "description": "Max results to return",
                        "default": 8,
                    },
                    "preferred_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Preferred domains to prioritise (e.g. weather.com)",
                    },
                },
                "required": ["query"],
            },
        }


# ── Content validation helpers ────────────────────────────────────────

_STOP_WORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and",
    "is", "it", "by", "with", "from", "or", "as", "be", "was",
    "are", "that", "this", "what", "how", "site",
}


def _extract_query_keywords(query: str) -> set[str]:
    """Extract meaningful keywords from a search query."""
    words = query.lower().split()
    return {w.strip(".,!?\"'") for w in words if len(w) > 2 and w not in _STOP_WORDS}


def _validate_result(result: dict[str, str], query_keywords: set[str]) -> bool:
    """Check if a search result contains at least one query keyword."""
    if not query_keywords:
        return True  # no keywords to validate against

    text = (result.get("title", "") + " " + result.get("body", "")).lower()
    return any(kw in text for kw in query_keywords)


def _simplify_query(query: str) -> str:
    """Strip site: clauses and extra qualifiers to produce a simpler fallback query."""
    import re
    # Remove site:xxx clauses
    simplified = re.sub(r'\bsite:\S+', '', query)
    # Remove OR operators
    simplified = re.sub(r'\bOR\b', '', simplified)
    # Collapse whitespace
    simplified = re.sub(r'\s+', ' ', simplified).strip()
    return simplified
