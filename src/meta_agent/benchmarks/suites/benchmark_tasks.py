"""
Benchmark Suite Definitions — GAIA, AutoGPT, BabyAGI
=====================================================

Reference benchmark tasks for evaluating the Meta-Agent system.
"""

from meta_agent.benchmarks.runner import BenchmarkTask


# ═══════════════════════════════════════════════════════════════════════════════
# GAIA Benchmark Tasks
# ═══════════════════════════════════════════════════════════════════════════════

GAIA_TASKS = [
    BenchmarkTask(
        task_id="gaia_001",
        description=(
            "What is the total population of the three largest cities "
            "in France as of 2023? Provide the answer as a single number."
        ),
        expected_output="",
        category="factual_qa",
        difficulty="easy",
    ),
    BenchmarkTask(
        task_id="gaia_002",
        description=(
            "Compare the GDP per capita of Japan and South Korea in 2022. "
            "Which country had a higher GDP per capita, and by how much?"
        ),
        expected_output="",
        category="comparative_analysis",
        difficulty="medium",
    ),
    BenchmarkTask(
        task_id="gaia_003",
        description=(
            "Write a Python function that takes a list of integers and returns "
            "the longest increasing subsequence. Include time complexity analysis "
            "and test cases."
        ),
        expected_output="",
        category="coding",
        difficulty="hard",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# AutoGPT-style Tasks
# ═══════════════════════════════════════════════════════════════════════════════

AUTOGPT_TASKS = [
    BenchmarkTask(
        task_id="autogpt_001",
        description=(
            "Research the current state of quantum computing. Identify the "
            "top 5 companies, their approaches, and key milestones achieved "
            "in 2024. Produce a structured report."
        ),
        category="research",
        difficulty="medium",
    ),
    BenchmarkTask(
        task_id="autogpt_002",
        description=(
            "Analyse the competitive landscape of electric vehicle battery "
            "technology. Compare solid-state vs lithium-ion batteries across "
            "cost, energy density, safety, and commercialisation timeline."
        ),
        category="analysis",
        difficulty="hard",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# BabyAGI-style Tasks
# ═══════════════════════════════════════════════════════════════════════════════

BABYAGI_TASKS = [
    BenchmarkTask(
        task_id="babyagi_001",
        description=(
            "Create a comprehensive plan for launching a SaaS product. "
            "Include market research, feature prioritisation, technical "
            "architecture, go-to-market strategy, and success metrics."
        ),
        category="planning",
        difficulty="hard",
    ),
    BenchmarkTask(
        task_id="babyagi_002",
        description=(
            "Design a machine learning pipeline for sentiment analysis "
            "of customer reviews. Include data collection, preprocessing, "
            "model selection, training, evaluation, and deployment steps."
        ),
        category="ml_engineering",
        difficulty="hard",
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# All benchmark tasks
# ═══════════════════════════════════════════════════════════════════════════════

ALL_TASKS = GAIA_TASKS + AUTOGPT_TASKS + BABYAGI_TASKS
