# Meta-Agent

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Framework: FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)

> **An intelligent LLM orchestration framework that dynamically designs, builds, and executes other LLM agents to solve complex tasks.**
>
> Instead of writing pipelines by hand, the Meta-Agent reads a task, invents the architecture, assembles the agents, runs the system, and critiques its own design.
>
> *The code that writes the code to solve the problem.*

---

## 📖 Table of Contents
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Docker](#-docker)
- [Project Structure](#-project-structure)
- [How It Works](#️-how-it-works)
- [Key Design Decisions](#-key-design-decisions)
- [Technology Stack](#-technology-stack)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🏗 System Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE LAYER                         │
│                 FastAPI REST + WebSocket + Swagger                │
├───────────────────────────────────────────────────────────────────┤
│                      API GATEWAY LAYER                            │
│           Auth Middleware │ Rate Limiting │ Request ID             │
├───────────────────────────────────────────────────────────────────┤
│                    ORCHESTRATION LAYER                             │
│    ┌──────────┐  ┌────────────────┐  ┌──────────────┐            │
│    │  Meta-    │→ │   Blueprint    │→ │    Graph     │            │
│    │  Agent    │  │   Generator    │  │   Builder    │            │
│    │ (Planner) │  │                │  │  (networkx)  │            │
│    └──────────┘  └────────────────┘  └──────────────┘            │
│                         ↓                                         │
│              ┌──────────────────────┐                             │
│              │   Graph Executor     │                             │
│              │ Sequential│Parallel  │                             │
│              │   DAG│Hierarchical   │                             │
│              └──────────────────────┘                             │
├───────────────────────────────────────────────────────────────────┤
│                   AGENT RUNTIME LAYER                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Research  │ │ Coding   │ │ Critic   │ │ Summary  │ ...7 types│
│  │ Agent    │ │ Agent    │ │ Agent    │ │ Agent    │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
├───────────────────────────────────────────────────────────────────┤
│                  TOOL EXECUTION LAYER                             │
│  Web Search │ Code Sandbox │ DB Query │ Vector Retrieval │ API   │
├───────────────────────────────────────────────────────────────────┤
│                  MEMORY & STATE LAYER                             │
│      Short-Term (Redis)        │     Long-Term (ChromaDB)        │
├───────────────────────────────────────────────────────────────────┤
│                  OBSERVABILITY LAYER                              │
│   OpenTelemetry │ Prometheus │ Structured Logging │ LangSmith    │
├───────────────────────────────────────────────────────────────────┤
│                  INFRASTRUCTURE LAYER                             │
│     Docker │ Kubernetes │ CI/CD │ HPA Auto-scaling                │
└───────────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

- **Dynamic Pipeline Generation**: No hardcoded graphs. The planner models custom architectures on the fly.
- **Robust Orchestration Engine**: Built-in support for Sequential, Parallel, and DAG-based agent execution.
- **Self-Healing Mechanics**: Employs an LLM-as-a-judge to evaluate outputs and trigger outer or inner repair loops automatically.
- **Extensible Sandboxed Tools**: Safely integrates Web Search, Code Execution, and more through strictly permission-scoped environments.
- **Production Ready**: Fully instrumented with OpenTelemetry and Prometheus, featuring a FastAPI asynchronous backend.

## 🚀 Quick Start

```bash
# Clone and install
cd LLM_Orch
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run the API server
uvicorn meta_agent.api.main:app --reload --port 8080

# Open Swagger UI
# http://localhost:8080/docs

# Run tests
pytest tests/ -v
```

## 🐳 Docker

```bash
# Build and run all services
cd infra
docker-compose up --build

# Services:
# - Meta-Agent API:  http://localhost:8080
# - ChromaDB:        http://localhost:8000
# - Prometheus:      http://localhost:9090
# - Grafana:         http://localhost:3000
```

## 📂 Project Structure

```
LLM_Orch/
├── src/meta_agent/
│   ├── schemas/            # Pydantic models (Blueprint, State)
│   ├── core/               # Meta-Agent engine, Planner, Blueprint Generator
│   ├── agents/             # Base agent + 7 specialized agents + Factory
│   │   └── specialized/    # Research, Planning, Coding, DataAnalysis, etc.
│   ├── orchestration/      # Graph Builder, Executor, Router
│   ├── tools/              # Tool registry + 5 tool implementations
│   │   └── implementations/
│   ├── memory/             # Short-term (Redis) + Long-term (ChromaDB)
│   ├── evaluation/         # LLM-as-Judge evaluator + Repair loop
│   ├── api/                # FastAPI app + routes + middleware
│   ├── observability/      # Tracing, logging, Prometheus metrics
│   └── benchmarks/         # GAIA, AutoGPT, BabyAGI benchmark suites
├── tests/                  # Unit and integration tests
├── infra/                  # Docker, Kubernetes, Prometheus config
│   └── k8s/
├── .github/workflows/      # CI/CD pipeline
├── pyproject.toml
└── .env.example
```

## ⚙️ How It Works

1. **User submits a task** → `POST /api/v1/tasks`
2. **Meta-Agent plans** → decomposes task into sub-tasks, selects topology
3. **Blueprint generated** → validated Pydantic schema defining the agent pipeline
4. **Graph built** → agents instantiated, edges wired, ready to execute
5. **Agents execute** → each agent runs with tools, passes results downstream
6. **Self-evaluation** → LLM-as-judge scores the output
7. **Repair if needed** → refine parameters or rebuild the entire architecture
8. **Result returned** → along with blueprint, evaluations, and metrics

## 🧠 Key Design Decisions

- **Dynamic agent creation** — agents are not hardcoded; they're instantiated from Blueprint configs
- **Graph-based orchestration** — supports sequential, parallel, hierarchical, and DAG topologies
- **Self-improvement loop** — the system critiques its own work and redesigns when quality is low
- **Security-first tools** — all tool access is sandboxed, validated, and permission-scoped
- **Observability by default** — every agent step, tool call, and evaluation is traced and metered

## 🛠 Technology Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI, Uvicorn, Pydantic v2 |
| LLM | LangChain, LangGraph, OpenAI |
| Memory | Redis (short-term), ChromaDB (long-term) |
| Observability | OpenTelemetry, Prometheus, Grafana, LangSmith |
| Infrastructure | Docker, Kubernetes, GitHub Actions |
| Testing | pytest, pytest-asyncio, httpx |

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

MIT
