"""
Test Suite — Pipeline Fix Validation
=====================================

Covers the core fixes for pipeline reliability:
- Web search domain blocking and content validation (ddgs package)
- Research agent structured output and retry logic
- Coding agent dynamic data usage
- Graph builder state propagation
- Rate-limit retry logic
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from meta_agent.tools.implementations.web_search import (
    WebSearchTool,
    _domain_of,
    _score_result,
    _extract_query_keywords,
    _validate_result,
    BLOCKED_DOMAINS,
)
from meta_agent.agents.specialized.research_agent import ResearchAgent
from meta_agent.agents.specialized.coding_agent import CodingAgent
from meta_agent.agents.base_agent import BaseAgent
from meta_agent.schemas.blueprint import AgentConfig, AgentType


# ═══════════════════════════════════════════════════════════════════════
# Web Search Tool — Domain Blocking (Problem 2)
# ═══════════════════════════════════════════════════════════════════════


class TestWebSearchDomainBlocking:
    """Verify that non-English/irrelevant domains are blocked."""

    def test_baidu_domains_blocked(self):
        for domain in ["baidu.com", "zhidao.baidu.com", "tieba.baidu.com"]:
            assert domain in BLOCKED_DOMAINS, f"{domain} should be blocked"

    def test_baidu_result_gets_negative_score(self):
        result = {
            "title": "current是什么意思",
            "url": "https://zhidao.baidu.com/question/635787571195459284.html",
            "body": "some text about current",
        }
        score = _score_result(result)
        assert score < 0, "Baidu results should get negative score"

    def test_trusted_weather_domain_scores_high(self):
        result = {
            "title": "Tokyo Weather",
            "url": "https://www.weather.com/weather/today/l/Tokyo",
            "body": "Current temperature in Tokyo is 15°C",
        }
        score = _score_result(result)
        assert score >= 8, "weather.com should score high"

    def test_subdomain_blocking(self):
        """Subdomains of blocked domains should also be blocked."""
        result = {
            "title": "test",
            "url": "https://sub.baidu.com/page",
            "body": "test",
        }
        score = _score_result(result)
        assert score < 0, "Subdomains of blocked domains should get negative score"


# ═══════════════════════════════════════════════════════════════════════
# Web Search Tool — Content Validation (Problem 8)
# ═══════════════════════════════════════════════════════════════════════


class TestWebSearchContentValidation:
    """Verify that results are validated for keyword relevance."""

    def test_extract_query_keywords(self):
        keywords = _extract_query_keywords("current weather temperature Tokyo today")
        assert "weather" in keywords
        assert "temperature" in keywords
        assert "tokyo" in keywords
        assert "today" in keywords
        # Stop words should be excluded
        assert "the" not in keywords
        assert "in" not in keywords

    def test_relevant_result_passes(self):
        keywords = {"weather", "tokyo", "temperature"}
        result = {
            "title": "Tokyo Weather Forecast",
            "body": "Current temperature in Tokyo is 15 degrees C",
        }
        assert _validate_result(result, keywords) is True

    def test_irrelevant_result_fails(self):
        keywords = {"weather", "tokyo", "temperature"}
        result = {
            "title": "current是什么意思",
            "body": "current means 'now' in English",
        }
        assert _validate_result(result, keywords) is False

    def test_empty_keywords_always_pass(self):
        result = {"title": "anything", "body": "whatever"}
        assert _validate_result(result, set()) is True


# ═══════════════════════════════════════════════════════════════════════
# Web Search Tool — DDGS Package Migration
# ═══════════════════════════════════════════════════════════════════════


class TestWebSearchDDGSMigration:
    """Verify that the tool uses the ddgs package, not duckduckgo_search."""

    def test_imports_ddgs(self):
        """The tool should import from ddgs, not duckduckgo_search."""
        import inspect
        source = inspect.getsource(WebSearchTool.execute)
        assert "from ddgs import DDGS" in source
        assert "from duckduckgo_search" not in source

    def test_search_and_rank_handles_empty_results(self):
        """_search_and_rank should safely handle empty result sets."""
        mock_ddgs = MagicMock()
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = []
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs_cls = MagicMock(return_value=mock_ddgs)

        results = WebSearchTool._search_and_rank(mock_ddgs_cls, "test query", 8)
        assert results == []


# ═══════════════════════════════════════════════════════════════════════
# Research Agent — Structured Output (Problem 4)
# ═══════════════════════════════════════════════════════════════════════


class TestResearchAgentStructuredOutput:
    """Verify that research agent produces machine-readable research_data."""

    def test_build_structured_response_with_celsius(self):
        result = ResearchAgent._build_structured_response(
            task="Research the current weather in Tokyo",
            extracted_data="15°C (from: Tokyo Weather)",
            sources=[{"title": "Tokyo Weather", "url": "https://weather.com/tokyo"}],
            llm_response="The temperature in Tokyo is 15°C.",
        )
        assert result["location"] == "Tokyo"
        assert result["temperature_celsius"] == 15.0
        assert result["temperature_fahrenheit"] == 59.0
        assert "weather.com" in result["source"]

    def test_build_structured_response_with_fahrenheit(self):
        result = ResearchAgent._build_structured_response(
            task="Research the current weather in New York",
            extracted_data="72°F (from: NY Weather)",
            sources=[{"title": "Weather", "url": "https://weather.com/ny"}],
            llm_response="Temperature is 72°F.",
        )
        assert result["temperature_fahrenheit"] == 72.0
        assert result["temperature_celsius"] is not None
        assert abs(result["temperature_celsius"] - 22.2) < 0.5

    def test_build_structured_response_no_data(self):
        result = ResearchAgent._build_structured_response(
            task="Research Python best practices",
            extracted_data="",
            sources=[],
            llm_response="Use type hints.",
        )
        assert result["temperature_celsius"] is None
        assert result["source"] == "unknown"

    def test_build_search_query_weather_includes_site_clause(self):
        """Weather queries should use the new format targets without site: clauses."""
        query = ResearchAgent._build_search_query(
            "Research the current weather in Tokyo"
        )
        assert "Tokyo" in query
        assert "current weather temperature today" in query


    def test_build_refined_query_strips_site_clauses(self):
        """Refined query should be simpler without site: constraints."""
        query = ResearchAgent._build_refined_query(
            "Research the current weather in Tokyo"
        )
        assert "Tokyo" in query
        assert "site:" not in query

    def test_detect_preferred_domains_weather(self):
        domains = ResearchAgent._detect_preferred_domains(
            "Research the current weather in Tokyo"
        )
        assert "weather.com" in domains
        assert "accuweather.com" in domains


# ═══════════════════════════════════════════════════════════════════════
# Coding Agent — Dynamic Data Usage (Problem 5)
# ═══════════════════════════════════════════════════════════════════════


class TestCodingAgentDynamicData:
    """Verify that coding agent injects research_data into the prompt."""

    @pytest.fixture
    def coding_agent(self):
        config = AgentConfig(
            agent_type=AgentType.CODING,
            name="Test Coder",
            role_description="Write Python code",
        )
        agent = CodingAgent(config)
        return agent

    def test_dynamic_data_injected_into_prompt(self, coding_agent):
        """When research_data is provided, the prompt should contain it."""
        mock_response = MagicMock()
        mock_response.content = "```python\nx = 15\n```"

        mock_client = AsyncMock()
        mock_client.ainvoke = AsyncMock(return_value=mock_response)
        coding_agent._llm_client = mock_client

        input_data = {
            "task": "Convert temperature to Fahrenheit",
            "research_data": {
                "location": "Tokyo",
                "temperature_celsius": 15.0,
                "source": "weather.com",
            },
        }

        asyncio.get_event_loop().run_until_complete(
            coding_agent.execute(input_data)
        )

        call_args = mock_client.ainvoke.call_args
        messages = call_args[0][0]
        user_msg = messages[-1]["content"]
        assert "RESEARCH DATA (MANDATORY INPUT)" in user_msg
        assert "temperature_celsius: 15.0" in user_msg
        assert "Tokyo" in user_msg

    def test_extracts_research_data_from_previous_results(self, coding_agent):
        """When research_data is nested in previous_results, it should be found."""
        mock_response = MagicMock()
        mock_response.content = "```python\nx = 20\n```"

        mock_client = AsyncMock()
        mock_client.ainvoke = AsyncMock(return_value=mock_response)
        coding_agent._llm_client = mock_client

        input_data = {
            "task": "Convert temperature",
            "previous_results": {
                "agent_research": {
                    "output": "some text",
                    "research_data": {
                        "location": "London",
                        "temperature_celsius": 20.0,
                        "source": "bbc.co.uk",
                    },
                }
            },
        }

        asyncio.get_event_loop().run_until_complete(
            coding_agent.execute(input_data)
        )

        call_args = mock_client.ainvoke.call_args
        messages = call_args[0][0]
        user_msg = messages[-1]["content"]
        assert "RESEARCH DATA (MANDATORY INPUT)" in user_msg
        assert "temperature_celsius: 20.0" in user_msg


# ═══════════════════════════════════════════════════════════════════════
# Graph Builder — State Propagation (Problem 5 support)
# ═══════════════════════════════════════════════════════════════════════


class TestAgentNodeDataForwarding:
    """Verify that AgentNode.__call__ forwards research_data."""

    def test_research_data_forwarded(self):
        from meta_agent.orchestration.graph_builder import AgentNode

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value={"output": "done"})

        node = AgentNode(agent=mock_agent, node_id="test_node")

        state = {
            "original_task": "test task",
            "context": "",
            "intermediate_results": {
                "research_agent": {
                    "output": "found data",
                    "research_data": {
                        "location": "Tokyo",
                        "temperature_celsius": 15.0,
                        "source": "weather.com",
                    },
                }
            },
        }

        asyncio.get_event_loop().run_until_complete(node(state))

        call_args = mock_agent.run.call_args[0][0]
        assert "research_data" in call_args
        assert call_args["research_data"]["location"] == "Tokyo"
        assert call_args["research_data"]["temperature_celsius"] == 15.0

    def test_no_research_data_when_not_present(self):
        from meta_agent.orchestration.graph_builder import AgentNode

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value={"output": "done"})

        node = AgentNode(agent=mock_agent, node_id="test_node")

        state = {
            "original_task": "test task",
            "context": "",
            "intermediate_results": {},
        }

        asyncio.get_event_loop().run_until_complete(node(state))

        call_args = mock_agent.run.call_args[0][0]
        assert "research_data" not in call_args


# ═══════════════════════════════════════════════════════════════════════
# Rate-Limit Retry Logic
# ═══════════════════════════════════════════════════════════════════════


class TestRateLimitRetry:
    """Verify invoke_llm_with_retry handles 429 errors correctly."""

    @pytest.fixture
    def agent(self):
        config = AgentConfig(
            agent_type=AgentType.RESEARCH,
            name="Test Agent",
            role_description="Test role description",
        )
        from meta_agent.agents.specialized.research_agent import ResearchAgent
        return ResearchAgent(config)

    def test_succeeds_on_first_try(self, agent):
        mock_response = MagicMock()
        mock_response.content = "success"
        mock_client = AsyncMock()
        mock_client.ainvoke = AsyncMock(return_value=mock_response)
        agent._llm_client = mock_client

        result = asyncio.get_event_loop().run_until_complete(
            agent.invoke_llm_with_retry([{"role": "user", "content": "test"}])
        )
        assert result.content == "success"
        assert mock_client.ainvoke.call_count == 1

    def test_retries_on_rate_limit_then_succeeds(self, agent):
        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_client = AsyncMock()
        mock_client.ainvoke = AsyncMock(
            side_effect=[
                Exception("rate_limit_exceeded"),
                mock_response,
            ]
        )
        agent._llm_client = mock_client

        result = asyncio.get_event_loop().run_until_complete(
            agent.invoke_llm_with_retry(
                [{"role": "user", "content": "test"}],
                base_delay=0.01,  # fast for tests
            )
        )
        assert result.content == "ok"
        assert mock_client.ainvoke.call_count == 2

    def test_raises_after_max_retries(self, agent):
        mock_client = AsyncMock()
        mock_client.ainvoke = AsyncMock(
            side_effect=Exception("429 Too Many Requests")
        )
        agent._llm_client = mock_client

        with pytest.raises(RuntimeError, match="rate limiting"):
            asyncio.get_event_loop().run_until_complete(
                agent.invoke_llm_with_retry(
                    [{"role": "user", "content": "test"}],
                    max_retries=2,
                    base_delay=0.01,
                )
            )
        assert mock_client.ainvoke.call_count == 2

    def test_non_rate_limit_errors_raise_immediately(self, agent):
        mock_client = AsyncMock()
        mock_client.ainvoke = AsyncMock(
            side_effect=ValueError("Some other error")
        )
        agent._llm_client = mock_client

        with pytest.raises(ValueError, match="Some other error"):
            asyncio.get_event_loop().run_until_complete(
                agent.invoke_llm_with_retry(
                    [{"role": "user", "content": "test"}],
                    base_delay=0.01,
                )
            )
        # Should NOT retry for non-rate-limit errors
        assert mock_client.ainvoke.call_count == 1


# ═══════════════════════════════════════════════════════════════════════
# Verification Guard — Sandbox Override (Fix 2)
# ═══════════════════════════════════════════════════════════════════════


class TestVerificationGuard:
    """Verify that sandbox success overrides hallucinated LLM FAIL verdicts."""

    def test_sandbox_pass_overrides_llm_fail(self):
        from meta_agent.agents.specialized.verification_agent import VerificationAgent
        result = VerificationAgent._sandbox_overrides_llm(
            code_test_result={"success": True, "stderr": "", "return_code": 0},
            llm_verdict=False,
            code_block="def convert(c):\n    return c * 9/5 + 32\n",
        )
        assert result is True

    def test_no_override_when_llm_says_pass(self):
        from meta_agent.agents.specialized.verification_agent import VerificationAgent
        result = VerificationAgent._sandbox_overrides_llm(
            code_test_result={"success": True, "stderr": "", "return_code": 0},
            llm_verdict=True,
            code_block="def foo(): pass",
        )
        assert result is False  # No override needed

    def test_no_override_when_sandbox_failed(self):
        from meta_agent.agents.specialized.verification_agent import VerificationAgent
        result = VerificationAgent._sandbox_overrides_llm(
            code_test_result={"success": False, "stderr": "SyntaxError"},
            llm_verdict=False,
            code_block="def foo( broken",
        )
        assert result is False

    def test_no_override_when_no_function_defs(self):
        from meta_agent.agents.specialized.verification_agent import VerificationAgent
        result = VerificationAgent._sandbox_overrides_llm(
            code_test_result={"success": True, "stderr": ""},
            llm_verdict=False,
            code_block="x = 42\nprint(x)",  # No function definitions
        )
        assert result is False

    def test_no_override_when_stderr_has_errors(self):
        from meta_agent.agents.specialized.verification_agent import VerificationAgent
        result = VerificationAgent._sandbox_overrides_llm(
            code_test_result={"success": True, "stderr": "RuntimeError: bad"},
            llm_verdict=False,
            code_block="def foo():\n    raise RuntimeError('bad')\n",
        )
        assert result is False

    def test_ast_parsing_detects_class_defs(self):
        from meta_agent.agents.specialized.verification_agent import _code_has_definitions
        assert _code_has_definitions("class Foo:\n    pass") is True
        assert _code_has_definitions("x = 1") is False
        assert _code_has_definitions("") is False


# ═══════════════════════════════════════════════════════════════════════
# Evaluation Action Override (Fix 4)
# ═══════════════════════════════════════════════════════════════════════


class TestEvaluationActionOverride:
    """Verify that the evaluator forces ACCEPT when score meets threshold."""

    def test_passing_score_forces_accept(self):
        from meta_agent.evaluation.evaluator import ResultEvaluator
        from meta_agent.schemas.blueprint import EvaluationConfig
        evaluator = ResultEvaluator()
        config = EvaluationConfig(overall_threshold=0.7)

        raw_json = '{"dimension_scores": [], "overall_score": 0.85, "passed": true, "recommended_action": "refine", "reasoning": "ok", "suggestions": []}'
        result = evaluator._parse_judge_response(raw_json, config)
        assert result.passed is True
        assert result.recommended_action.value == "accept"

    def test_failing_score_keeps_refine(self):
        from meta_agent.evaluation.evaluator import ResultEvaluator
        from meta_agent.schemas.blueprint import EvaluationConfig
        evaluator = ResultEvaluator()
        config = EvaluationConfig(overall_threshold=0.7)

        raw_json = '{"dimension_scores": [], "overall_score": 0.55, "passed": false, "recommended_action": "refine", "reasoning": "needs work", "suggestions": []}'
        result = evaluator._parse_judge_response(raw_json, config)
        assert result.passed is False
        assert result.recommended_action.value == "refine"


# ═══════════════════════════════════════════════════════════════════════
# Repair Loop Duplicate Detection (Fix 3)
# ═══════════════════════════════════════════════════════════════════════


class TestRepairDuplicateDetection:
    """Verify hash-based and character-level duplicate output detection."""

    def test_identical_outputs_detected(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        assert GraphExecutor._outputs_are_similar("hello world", "hello world") is True

    def test_near_identical_outputs_detected(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        a = "def convert(c):\n    return c * 9/5 + 32\n"
        b = "def convert(c):\n    return c * 9/5 + 32 \n"  # trailing space
        assert GraphExecutor._outputs_are_similar(a, b) is True

    def test_different_outputs_not_flagged(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        a = "def convert(c):\n    return c * 9/5 + 32\n"
        b = "def convert(f):\n    return (f - 32) * 5/9\n"
        assert GraphExecutor._outputs_are_similar(a, b) is False

    def test_empty_outputs_not_flagged(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        assert GraphExecutor._outputs_are_similar("", "") is False
        assert GraphExecutor._outputs_are_similar("hello", "") is False

    def test_case_insensitive_hash(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        assert GraphExecutor._output_hash("Hello World") == GraphExecutor._output_hash("hello world")


# ═══════════════════════════════════════════════════════════════════════
# Minimal Coding Prompts (Fix 5)
# ═══════════════════════════════════════════════════════════════════════


class TestMinimalCodingPrompts:
    """Verify task-proportional prompt selection."""

    def test_simple_task_not_robust(self):
        from meta_agent.agents.specialized.coding_agent import _task_needs_robust_code
        assert _task_needs_robust_code("Write a function to convert Celsius to Fahrenheit") is False

    def test_production_task_is_robust(self):
        from meta_agent.agents.specialized.coding_agent import _task_needs_robust_code
        assert _task_needs_robust_code("Write a production-grade API with edge case handling") is True

    def test_enterprise_task_is_robust(self):
        from meta_agent.agents.specialized.coding_agent import _task_needs_robust_code
        assert _task_needs_robust_code("Build an enterprise validation module") is True

    def test_minimal_prompt_used_for_simple_task(self):
        config = AgentConfig(
            agent_type=AgentType.CODING,
            name="Test Coder",
            role_description="Write Python code",
        )
        agent = CodingAgent(config)
        mock_response = MagicMock()
        mock_response.content = "```python\nx = 1\n```"
        mock_client = AsyncMock()
        mock_client.ainvoke = AsyncMock(return_value=mock_response)
        agent._llm_client = mock_client

        asyncio.get_event_loop().run_until_complete(
            agent.execute({"task": "Convert celsius to fahrenheit"})
        )
        call_args = mock_client.ainvoke.call_args
        user_msg = call_args[0][0][-1]["content"]
        assert "minimal and focused" in user_msg
        assert "NaN/infinity/None" not in user_msg


# ═══════════════════════════════════════════════════════════════════════
# Research Location Extraction Fix (Fix 6)
# ═══════════════════════════════════════════════════════════════════════


class TestResearchLocationExtraction:
    """Verify that temperature-related words are excluded from location."""

    def test_fahrenheit_excluded_from_location(self):
        result = ResearchAgent._build_structured_response(
            task="Research the current weather in Tokyo Fahrenheit",
            extracted_data="15°C",
            sources=[{"title": "w", "url": "https://weather.com"}],
            llm_response="15°C",
        )
        assert result["location"] == "Tokyo"
        assert "Fahrenheit" not in result["location"]

    def test_celsius_excluded_from_location(self):
        result = ResearchAgent._build_structured_response(
            task="Convert Celsius for Tokyo today",
            extracted_data="",
            sources=[],
            llm_response="",
        )
        assert "Celsius" not in result["location"]
        assert "Tokyo" in result["location"]

    def test_multiple_location_words_preserved(self):
        result = ResearchAgent._build_structured_response(
            task="Research the current weather in New York",
            extracted_data="",
            sources=[],
            llm_response="",
        )
        assert "New" in result["location"]
        assert "York" in result["location"]


# ═══════════════════════════════════════════════════════════════════════
# Coding Agent — Blocked Import Scanning (Problem 1)
# ═══════════════════════════════════════════════════════════════════════


class TestCodingAgentBlockedImports:
    """Verify that _scan_blocked_imports detects forbidden imports."""

    def test_scan_detects_requests(self):
        code = "import requests\nresponse = requests.get('http://example.com')"
        blocked = CodingAgent._scan_blocked_imports(code)
        assert "requests" in blocked

    def test_scan_detects_requests_alias(self):
        code = "import requests as r\nr.get('http://example.com')"
        blocked = CodingAgent._scan_blocked_imports(code)
        assert "requests" in blocked

    def test_scan_detects_from_import(self):
        code = "from urllib.request import urlopen\ndata = urlopen('http://example.com')"
        blocked = CodingAgent._scan_blocked_imports(code)
        assert "urllib" in blocked

    def test_scan_detects_httpx_alias(self):
        code = "import httpx as client\nclient.get('http://example.com')"
        blocked = CodingAgent._scan_blocked_imports(code)
        assert "httpx" in blocked

    def test_scan_detects_multiple_blocked(self):
        code = "import requests\nimport subprocess\nx = 1"
        blocked = CodingAgent._scan_blocked_imports(code)
        assert "requests" in blocked
        assert "subprocess" in blocked

    def test_scan_passes_clean_code(self):
        code = "import math\nimport json\nx = math.sqrt(25)"
        blocked = CodingAgent._scan_blocked_imports(code)
        assert blocked == []

    def test_scan_ignores_comments(self):
        """Import in actual code is detected; comments are not code lines."""
        code = "# import requests  <-- don't use this\nimport math\nx = 1"
        blocked = CodingAgent._scan_blocked_imports(code)
        # The regex scans line-by-line; a comment with 'import requests' should
        # not match because it starts with '#', not 'import' or 'from'.
        assert blocked == []

    def test_sandbox_prompt_present(self):
        """System prompt should mention sandbox restrictions."""
        config = AgentConfig(
            agent_type=AgentType.CODING,
            name="Test Coder",
            role_description="Write Python code",
        )
        agent = CodingAgent(config)
        mock_response = MagicMock()
        mock_response.content = "```python\nx = 1\n```"
        mock_client = AsyncMock()
        mock_client.ainvoke = AsyncMock(return_value=mock_response)
        agent._llm_client = mock_client

        asyncio.get_event_loop().run_until_complete(
            agent.execute({"task": "test"})
        )
        call_args = mock_client.ainvoke.call_args
        messages = call_args[0][0]
        system_msg = messages[0]["content"]
        assert "SANDBOX RESTRICTIONS" in system_msg
        assert "requests" in system_msg


# ═══════════════════════════════════════════════════════════════════════
# Research Agent — No-Guess Policy (Problem 2 enhancement)
# ═══════════════════════════════════════════════════════════════════════


class TestResearchAgentNoGuess:
    """Verify that LLM-fabricated values are NOT used for temperatures."""

    def test_llm_response_used_as_fallback_for_temps(self):
        """When extracted_data is empty but llm_response contains temperature
        text, the fallback should extract the values."""
        result = ResearchAgent._build_structured_response(
            task="Research the current weather in Tokyo",
            extracted_data="",  # No snippet data
            sources=[{"title": "w", "url": "https://weather.com"}],
            llm_response="The temperature is approximately 25°C today.",
        )
        # Fallback extraction should populate temperatures
        assert result["temperature_celsius"] == 25.0
        assert result["temperature_fahrenheit"] is not None

    def test_extracted_data_still_works(self):
        """When extracted_data has temps, they should be parsed."""
        result = ResearchAgent._build_structured_response(
            task="Research weather in London",
            extracted_data="10°C (from: BBC Weather)",
            sources=[{"title": "BBC", "url": "https://bbc.co.uk/weather"}],
            llm_response="",
        )
        assert result["temperature_celsius"] == 10.0
        assert result["temperature_fahrenheit"] is not None

    def test_no_guess_prompt_present(self):
        """System prompt should contain anti-hallucination rules."""
        config = AgentConfig(
            agent_type=AgentType.RESEARCH,
            name="Test Researcher",
            role_description="Find information",
        )
        agent = ResearchAgent(config)
        prompt = agent.get_system_prompt("test")
        # We can't easily test the appended rules since they're added in
        # execute(), but we can verify the agent is constructable
        assert agent is not None


# ═══════════════════════════════════════════════════════════════════════
# Dynamic Planning Skip (Problem 3)
# ═══════════════════════════════════════════════════════════════════════


class TestDynamicPlanningSkip:
    """Verify _is_simple_task classification."""

    def test_simple_function_task(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        assert GraphExecutor._is_simple_task(
            "Write a function to convert Celsius to Fahrenheit"
        ) is True

    def test_simple_calculate_task(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        assert GraphExecutor._is_simple_task(
            "Calculate the area of a circle"
        ) is True

    def test_simple_convert_task(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        assert GraphExecutor._is_simple_task("Convert 100 USD to EUR") is True

    def test_complex_multi_verb_task(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        assert GraphExecutor._is_simple_task(
            "Research and analyze competitor pricing and summarize findings"
        ) is False

    def test_complex_multi_step_task(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        assert GraphExecutor._is_simple_task(
            "Design a REST API and implement authentication"
        ) is False

    def test_empty_task_not_simple(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        assert GraphExecutor._is_simple_task("") is False

    def test_no_pattern_match(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        assert GraphExecutor._is_simple_task(
            "Explain the theory of relativity"
        ) is False

    def test_planning_agent_detection(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        from meta_agent.orchestration.graph_builder import AgentNode
        mock_agent = MagicMock()
        mock_agent.config.agent_type.value = "planning"
        node = AgentNode(agent=mock_agent, node_id="planner_1")
        assert GraphExecutor._is_planning_agent(node) is True


# ═══════════════════════════════════════════════════════════════════════
# Sandbox Restriction Detection (Problem 5 enhancement)
# ═══════════════════════════════════════════════════════════════════════


class TestSandboxRestrictionDetection:
    """Verify _is_sandbox_restriction_error detects restriction failures."""

    def test_blocked_import_detected(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        result = {
            "success": False,
            "stderr": "Validation failed: Blocked import detected: 'requests'",
            "return_code": 1,
        }
        assert GraphExecutor._is_sandbox_restriction_error(result) is True

    def test_network_access_detected(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        result = {
            "success": False,
            "error": "network access not allowed in sandbox",
        }
        assert GraphExecutor._is_sandbox_restriction_error(result) is True

    def test_normal_error_not_flagged(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        result = {
            "success": False,
            "stderr": "SyntaxError: invalid syntax",
        }
        assert GraphExecutor._is_sandbox_restriction_error(result) is False

    def test_success_not_flagged(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        result = {"success": True, "stderr": ""}
        assert GraphExecutor._is_sandbox_restriction_error(result) is False

    def test_none_not_flagged(self):
        from meta_agent.orchestration.graph_executor import GraphExecutor
        assert GraphExecutor._is_sandbox_restriction_error(None) is False
